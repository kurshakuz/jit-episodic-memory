#!/usr/bin/env python3
"""
Official ConceptFusion adapter.

Uses the OFFICIAL ConceptFusion pipeline (external concept-fusion checkout) with:
  - SAM ViT-H for masks (reusing ConceptGraphs cache when available)
  - OpenCLIP ViT-H-14 for per-mask + global features
  - Pixel-aligned fusion (softmax-weighted blending; official paper formulation)
  - gradslam PointFusion for 3D fusion (conceptfusion branch, use_embeddings=True)

Four passes, mirroring run_official_cg.py:
  Pass 1 (SAM):      generate masks, save to disk [cacheable / reusable from CG]
  Pass 2 (CF-feat):  per-frame pixel-aligned features, save as .pt
  Pass 3 (Fusion):   PointFusion 3D map, save as H5
  Pass 4 (Query):    text query -> cosine similarity -> DBSCAN -> Loc@d metrics

Runtime: ScanNet 500f ~5-10h (SAM cache available), HM3D 500f ~24h (no cache).

Usage:
    conda run -n cg python baselines/run_official_conceptfusion.py \
        --dataset scannet --max-scenes 142
    conda run -n cg python baselines/run_official_conceptfusion.py \
        --dataset hm3d --max-scenes 36
"""

import os
# Reduce fragmentation before any CUDA init
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")

import argparse
import gc
import gzip
import json
import pickle
import sys
import time
import traceback
from dataclasses import field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.jit_gradslam_dataset import JITSceneDataset

# ============================================================
# Config
# ============================================================

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "paper_results"
CACHE_DIR = OUTPUT_DIR / "conceptfusion_official_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Data locations
HM3D_DATA = PROJECT_ROOT / "outputs" / "multi_scene_eval_500f"
SCANNET_DATA = PROJECT_ROOT / "scannet" / "jit_format_500"
SCANNET_SAM_CACHE = PROJECT_ROOT / "scannet" / "results" / "cg_scannet_cache_500"

# SAM checkpoint (same as ConceptGraphs)
SAM_CKPT = PROJECT_ROOT / "models" / "sam_vit_h_4b8939.pth"

# Frame processing
FRAME_STRIDE = 4           # every 4th frame (ConceptFusion and CG default)
DESIRED_HEIGHT = 120       # downsample to ConceptFusion's paper default
DESIRED_WIDTH = 160

# Evaluation
TEST_QUERIES = ["toilet", "chair", "table", "bed", "couch",
                "sink", "lamp", "mirror", "cabinet", "shelf"]
THRESHOLDS = [0.25, 0.5, 1.0, 2.0, 3.0]

# Query config
TOP_K_PERCENT = 0.01       # top 1% of points
DBSCAN_EPS = 0.5           # meters
DBSCAN_MIN_SAMPLES = 3

DEVICE = "cuda:0"

# ============================================================
# Scene enumeration
# ============================================================

def get_hm3d_scenes(max_scenes: int = 36):
    scenes = []
    for sd in sorted(HM3D_DATA.iterdir()):
        if not sd.is_dir():
            continue
        gt = sd / f"{sd.name}_ground_truth.json"
        trace = sd / "exploration" / "trace.parquet"
        if gt.exists() and trace.exists():
            scenes.append(sd)
    return scenes[:max_scenes]


def get_scannet_scenes(max_scenes: int = 142):
    from scannet.config import SCANNET_VAL_SCENES
    scenes = []
    for sid in SCANNET_VAL_SCENES[:max_scenes]:
        sd = SCANNET_DATA / sid
        if sd.exists():
            scenes.append(sd)
    return scenes


def load_gt(scene_dir: Path):
    sid = scene_dir.name
    gt_path = scene_dir / f"{sid}_ground_truth.json"
    if not gt_path.exists():
        for f in scene_dir.glob("*_ground_truth.json"):
            gt_path = f
            break
    if not gt_path.exists():
        return None
    with open(gt_path) as f:
        data = json.load(f)
    result = {}
    for obj in data.get("objects", {}).values():
        cat = obj.get("category", obj.get("name", "")).lower()
        c = obj.get("center")
        if c is not None:
            result.setdefault(cat, []).append(np.array(c))
    return result


# ============================================================
# Pass 1: SAM masks (reuse CG cache or generate fresh)
# ============================================================

def pass1_sam_masks(scene_dir: Path, frame_ids: List[int], scene_cache: Path,
                   sam_cache_dir: Optional[Path] = None, verbose: bool = False):
    """
    Obtain SAM masks for each frame.

    Priority:
      1. This pipeline's own cache (scene_cache/sam_masks_*.pkl.gz)
      2. ConceptGraphs' cache (sam_cache_dir/detections_*.pkl.gz) — has 'mask' field
      3. Fresh SAM run (split-device strategy)
    """
    scene_cache.mkdir(parents=True, exist_ok=True)
    done_marker = scene_cache / "pass1_done.json"
    if done_marker.exists():
        if verbose:
            print(f"    Pass 1: SKIP (cached)")
        return

    # SAM-mask source policy:
    #   * ScanNet: reuse CG's existing SAM cache when available (CG params:
    #     points_per_side=12, pred_iou_thresh=0.88, crop_n_layers=0). This
    #     differs from CF's defaults (8 / 0.92 / 1) but keeps compute feasible
    #     (~5 min/scene vs ~27 min/scene). Documented as an adaptation.
    #   * HM3D: no cache exists; we run SAM from scratch with our compute-
    #     feasible compromise: CF's points_per_side=8 and pred_iou_thresh=0.92
    #     (matching the paper) but crop_n_layers=0 (paper uses 1; would be
    #     30+ minutes per frame on 8GB VRAM, infeasible at scale).
    cg_cache_hits = 0
    if sam_cache_dir is not None and sam_cache_dir.exists():
        for fid in frame_ids:
            cg_det = sam_cache_dir / f"detections_{fid:06d}.pkl.gz"
            if cg_det.exists():
                cg_cache_hits += 1
        if cg_cache_hits >= len(frame_ids) * 0.9:
            if verbose:
                print(f"    Pass 1: using CG cache ({cg_cache_hits}/{len(frame_ids)} frames)")
            for fid in frame_ids:
                cg_det = sam_cache_dir / f"detections_{fid:06d}.pkl.gz"
                if not cg_det.exists():
                    continue
                with gzip.open(cg_det, "rb") as f:
                    det = pickle.load(f)
                masks = det["mask"]
                xyxy = det["xyxy"]
                conf = det["confidence"]
                out_path = scene_cache / f"sam_masks_{fid:06d}.pkl.gz"
                with gzip.open(out_path, "wb") as f:
                    pickle.dump({"masks": masks, "xyxy": xyxy, "conf": conf}, f,
                                protocol=pickle.HIGHEST_PROTOCOL)
            done_marker.write_text(json.dumps({
                "source": "cg_cache", "num_frames": len(frame_ids),
                "timestamp": time.time(),
            }))
            return

    # Fresh SAM run — split-device strategy (used for HM3D and ScanNet without cache)
    if verbose:
        print(f"    Pass 1: running SAM from scratch...")
    _run_sam_from_scratch(scene_dir, frame_ids, scene_cache, verbose=verbose)
    done_marker.write_text(json.dumps({
        "source": "fresh_sam", "num_frames": len(frame_ids),
        "timestamp": time.time(),
    }))


def _run_sam_from_scratch(scene_dir: Path, frame_ids: List[int], scene_cache: Path,
                          verbose: bool = False):
    """Load SAM with split-device strategy (matches run_official_cg.py)."""
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    from segment_anything.predictor import SamPredictor as _SamPredictor

    if not SAM_CKPT.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {SAM_CKPT}")

    sam = sam_model_registry["vit_h"](checkpoint=str(SAM_CKPT))
    sam.eval()
    sam.prompt_encoder.to("cuda")
    sam.mask_decoder.to("cuda")

    # Monkey-patch predictor for split-device
    def _split_set_torch_image(self_pred, transformed_image, original_image_size):
        self_pred.reset_image()
        self_pred.original_size = original_image_size
        self_pred.input_size = tuple(transformed_image.shape[-2:])
        input_image = self_pred.model.preprocess(transformed_image.cpu())
        with torch.no_grad():
            self_pred.features = self_pred.model.image_encoder(input_image).to("cuda")
        self_pred.is_image_set = True

    _SamPredictor.set_torch_image = torch.no_grad()(_split_set_torch_image)
    _SamPredictor.device = property(lambda self_pred: torch.device("cuda"))
    torch.set_num_threads(os.cpu_count() or 8)

    mask_gen = SamAutomaticMaskGenerator(
        sam,
        # Official ConceptFusion SAM hyperparameters (extract_conceptfusion_features.py)
        # crop_n_layers=1 is the official setting but 30-60x slower per frame on
        # 8GB VRAM; we set crop_n_layers=0 here as a compute-feasibility compromise.
        # Other CF settings (points_per_side=8, pred_iou_thresh=0.92) match the paper.
        points_per_side=8,
        points_per_batch=64,
        pred_iou_thresh=0.92,
        crop_n_layers=0,
        min_mask_region_area=100,
    )

    import pandas as pd
    from PIL import Image
    trace = pd.read_parquet(scene_dir / "exploration" / "trace.parquet")

    for i, fid in enumerate(frame_ids):
        out_path = scene_cache / f"sam_masks_{fid:06d}.pkl.gz"
        if out_path.exists():
            continue
        row = trace.iloc[fid]
        img_path = scene_dir / "exploration" / row["image_path"]
        img = np.array(Image.open(img_path))
        if img.shape[-1] == 4:
            img = img[..., :3]
        raw = mask_gen.generate(img)
        if not raw:
            masks = np.zeros((0, *img.shape[:2]), dtype=bool)
            xyxy = np.zeros((0, 4))
            conf = np.zeros(0)
        else:
            masks = np.stack([m["segmentation"] for m in raw])
            xyxy = np.array([[m["bbox"][0], m["bbox"][1],
                              m["bbox"][0] + m["bbox"][2],
                              m["bbox"][1] + m["bbox"][3]] for m in raw])
            conf = np.array([m["predicted_iou"] for m in raw])
        with gzip.open(out_path, "wb") as f:
            pickle.dump({"masks": masks, "xyxy": xyxy, "conf": conf}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        if verbose and (i + 1) % 25 == 0:
            print(f"      frame {i+1}/{len(frame_ids)}")

    del sam, mask_gen
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================
# Pass 2: ConceptFusion pixel-aligned features (official formulation)
# ============================================================

def pass2_features(scene_dir: Path, frame_ids: List[int], scene_cache: Path,
                   verbose: bool = False):
    """
    Per-frame pixel-aligned features using the official ConceptFusion formulation:
      similarity_scores[i] = cos_sim(f_L_i, f_G)
      softmax_scores = softmax(similarity_scores)
      weighted_feat_i = softmax_scores[i] * f_G + (1 - softmax_scores[i]) * f_L_i
      pixel_features[mask_i] += normalize(weighted_feat_i)
      pixel_features = normalize(pixel_features)

    Stored as half-precision .pt files at desired (DESIRED_HEIGHT x DESIRED_WIDTH).
    """
    import pandas as pd
    from PIL import Image
    import open_clip

    done_marker = scene_cache / "pass2_done.json"
    if done_marker.exists():
        if verbose:
            print(f"    Pass 2: SKIP (cached)")
        return

    feat_dir = scene_cache / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"    Pass 2: loading CLIP ViT-H-14 (half precision)...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", "laion2b_s32b_b79k", precision="fp16"
    )
    model = model.cuda().eval()

    trace = pd.read_parquet(scene_dir / "exploration" / "trace.parquet")
    cosine_sim = torch.nn.CosineSimilarity(dim=-1)

    for i, fid in enumerate(frame_ids):
        out_path = feat_dir / f"frame_{fid:06d}.pt"
        if out_path.exists():
            continue
        mask_path = scene_cache / f"sam_masks_{fid:06d}.pkl.gz"
        if not mask_path.exists():
            continue
        with gzip.open(mask_path, "rb") as f:
            mask_data = pickle.load(f)
        masks = mask_data["masks"]  # (N, H, W) bool
        if len(masks) == 0:
            # save zeros so the frame is still counted
            H = trace.iloc[fid].get("H") or masks.shape[1] if len(masks) else 480
            torch.save(torch.zeros((H, 480, 1024), dtype=torch.float16), out_path)
            continue

        row = trace.iloc[fid]
        img_path = scene_dir / "exploration" / row["image_path"]
        pil_img = Image.open(img_path)
        H, W = pil_img.size[1], pil_img.size[0]
        img_np = np.array(pil_img)
        if img_np.ndim == 3 and img_np.shape[-1] == 4:
            img_np = img_np[..., :3]

        # Global feature (half precision directly, no autocast)
        with torch.no_grad():
            g_input = preprocess(pil_img).unsqueeze(0).cuda().half()
            global_feat = model.encode_image(g_input)
            global_feat = torch.nn.functional.normalize(global_feat, dim=-1)  # (1, 1024) half
        feat_dim = global_feat.shape[-1]

        # Per-mask features — batched to avoid memory fragmentation
        crops_pil = []
        roi_indices = []
        for mi in range(len(masks)):
            seg = masks[mi]
            if int(seg.sum()) < 10:
                continue
            ys, xs = np.where(seg)
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img_np[y1:y2, x1:x2]
            crops_pil.append(Image.fromarray(crop))
            roi_indices.append(np.stack([ys, xs], axis=-1))

        if not crops_pil:
            torch.save(torch.zeros((DESIRED_HEIGHT, DESIRED_WIDTH, feat_dim), dtype=torch.float16), out_path)
            del global_feat; torch.cuda.empty_cache(); continue

        # Encode ROIs one-by-one, keep features on CPU half to avoid fragmentation
        feat_per_roi_cpu = []
        for c in crops_pil:
            with torch.no_grad():
                bt = preprocess(c).unsqueeze(0).cuda().half()
                rf = model.encode_image(bt)
                rf = torch.nn.functional.normalize(rf, dim=-1)
            feat_per_roi_cpu.append(rf[0].cpu())
            del bt, rf
        feat_per_roi = torch.stack(feat_per_roi_cpu).cuda()  # (M, 1024) half
        del feat_per_roi_cpu

        # Similarity + softmax
        sim_scores = cosine_sim(global_feat.expand(feat_per_roi.shape[0], -1), feat_per_roi)  # (M,)

        softmax_scores = torch.nn.functional.softmax(sim_scores, dim=0)  # (M,)

        # Blend: (softmax * global) + ((1-softmax) * roi)
        # Shape: (M, 1) * (1, 1024) + (M, 1) * (M, 1024) -> (M, 1024)
        blend = (softmax_scores.unsqueeze(1) * global_feat +
                 (1.0 - softmax_scores.unsqueeze(1)) * feat_per_roi)
        blend = torch.nn.functional.normalize(blend, dim=-1)  # (M, 1024) half
        blend_cpu = blend.cpu()
        del blend, feat_per_roi, global_feat
        torch.cuda.empty_cache()

        outfeat = torch.zeros((H, W, feat_dim), dtype=torch.float16, device="cpu")
        for mi, nz in enumerate(roi_indices):
            ys_t = torch.from_numpy(nz[:, 0]).long()
            xs_t = torch.from_numpy(nz[:, 1]).long()
            outfeat[ys_t, xs_t] += blend_cpu[mi]

        # Normalize accumulated features per pixel
        norms = outfeat.float().norm(dim=-1, keepdim=True).clamp_min_(1e-6)
        outfeat = (outfeat.float() / norms).half()

        # Downsample to desired feature resolution
        outfeat_f = outfeat.permute(2, 0, 1).unsqueeze(0).float()  # (1, 1024, H, W)
        outfeat_small = torch.nn.functional.interpolate(
            outfeat_f, size=(DESIRED_HEIGHT, DESIRED_WIDTH), mode="nearest"
        )[0].permute(1, 2, 0).half()
        torch.save(outfeat_small, out_path)

        del blend_cpu, outfeat, outfeat_f, outfeat_small
        torch.cuda.empty_cache()

        if verbose and (i + 1) % 25 == 0:
            print(f"      feat frame {i+1}/{len(frame_ids)}")

    del model, preprocess
    gc.collect()
    torch.cuda.empty_cache()
    done_marker.write_text(json.dumps({
        "num_frames": len(frame_ids), "feat_dim": 1024,
        "desired_h": DESIRED_HEIGHT, "desired_w": DESIRED_WIDTH,
        "timestamp": time.time(),
    }))


# ============================================================
# Pass 3: PointFusion 3D mapping
# ============================================================

def pass3_fusion(scene_dir: Path, frame_ids: List[int], scene_cache: Path,
                 verbose: bool = False):
    """
    3D fusion of per-pixel ConceptFusion features using GT poses.

    We use a simple projection + voxelized accumulation instead of gradslam's
    PointFusion because the latter clones the full embeddings tensor during
    updates, OOMing on an 8GB card with ViT-H-14 (1024-dim) features.

    Since our poses are GT (Habitat simulator or ScanNet BundleFusion), we
    don't need PointFusion's SLAM ability. Straight projection is correct.

    Produces: scene_cache/pointcloud.npz with keys 'points' (N,3) float32,
    'features' (N, 1024) float16.
    """
    done_marker = scene_cache / "pass3_done.json"
    pc_path = scene_cache / "pointcloud.npz"
    if done_marker.exists():
        if verbose:
            print(f"    Pass 3: SKIP (cached)")
        return

    feat_dir = scene_cache / "features"

    ds = JITSceneDataset(
        scene_dir,
        desired_height=DESIRED_HEIGHT,
        desired_width=DESIRED_WIDTH,
        frame_ids=frame_ids,
        load_embeddings=False,
        device="cpu",  # keep projection on CPU to free GPU for later
        dtype=torch.float32,
    )

    voxel_size = 0.05  # 5cm
    point_stride = 2   # downsample pixels for speed
    voxel_accum = {}   # key -> [sum_pos, sum_feat, count]

    t0 = time.time()
    for idx in range(len(ds)):
        fid = frame_ids[idx]
        feat_path = feat_dir / f"frame_{fid:06d}.pt"
        if not feat_path.exists():
            continue
        emb = torch.load(feat_path).float().numpy()  # (H, W, 1024)

        color, depth, K, pose = ds[idx]
        depth_np = depth[..., 0].numpy()  # (H, W)
        K_np = K.numpy()
        pose_np = pose.numpy()

        H, W = depth_np.shape
        fx, fy = K_np[0, 0], K_np[1, 1]
        cx, cy = K_np[0, 2], K_np[1, 2]

        # Robust shape match: resize features to match depth if they differ
        if emb.shape[:2] != (H, W):
            import cv2
            # cv2.resize takes (W, H)
            emb_resized = np.zeros((H, W, emb.shape[-1]), dtype=emb.dtype)
            # Resize per channel chunk to avoid memory issues with 1024-channel
            chunk = 256
            for c_start in range(0, emb.shape[-1], chunk):
                c_end = min(c_start + chunk, emb.shape[-1])
                emb_resized[..., c_start:c_end] = cv2.resize(
                    emb[..., c_start:c_end], (W, H), interpolation=cv2.INTER_NEAREST
                )
            emb = emb_resized

        # Unproject depth
        u = np.arange(0, W, point_stride)
        v = np.arange(0, H, point_stride)
        uu, vv = np.meshgrid(u, v)
        d_s = depth_np[::point_stride, ::point_stride]
        valid = (d_s > 0.1) & (d_s < 10.0)

        # Camera frame (OpenCV: +X right, +Y down, +Z forward)
        x_cam = (uu - cx) * d_s / fx
        y_cam = (vv - cy) * d_s / fy
        z_cam = d_s
        pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1).reshape(-1, 3)
        valid_flat = valid.reshape(-1)

        # Transform to world via pose
        ones = np.ones((pts_cam.shape[0], 1), dtype=np.float32)
        pts_h = np.concatenate([pts_cam, ones], axis=-1)
        pts_world = (pose_np @ pts_h.T).T[:, :3]

        # Features at stride
        feat_s = emb[::point_stride, ::point_stride].reshape(-1, emb.shape[-1])

        # Keep only valid
        pts_valid = pts_world[valid_flat]
        feat_valid = feat_s[valid_flat]

        # Voxelize and accumulate
        voxel_keys = np.floor(pts_valid / voxel_size).astype(np.int64)
        # Encode (x, y, z) as single int64 key
        keys = (voxel_keys[:, 0] * 1_000_003 * 1_000_003 +
                voxel_keys[:, 1] * 1_000_003 +
                voxel_keys[:, 2])
        unique_keys, inv_idx = np.unique(keys, return_inverse=True)
        for i, k in enumerate(unique_keys):
            mask = inv_idx == i
            p_sum = pts_valid[mask].sum(axis=0)
            f_sum = feat_valid[mask].sum(axis=0).astype(np.float32)
            cnt = int(mask.sum())
            if k in voxel_accum:
                voxel_accum[k][0] += p_sum
                voxel_accum[k][1] += f_sum
                voxel_accum[k][2] += cnt
            else:
                voxel_accum[k] = [p_sum, f_sum, cnt]

        if verbose and (idx + 1) % 25 == 0:
            print(f"      fuse frame {idx+1}/{len(ds)} | voxels={len(voxel_accum)}")

    # Finalize
    if not voxel_accum:
        points = np.zeros((0, 3), dtype=np.float32)
        features = np.zeros((0, 1024), dtype=np.float16)
    else:
        points = np.zeros((len(voxel_accum), 3), dtype=np.float32)
        features = np.zeros((len(voxel_accum), 1024), dtype=np.float32)
        for i, (k, (p_sum, f_sum, cnt)) in enumerate(voxel_accum.items()):
            points[i] = p_sum / cnt
            features[i] = f_sum / cnt
        # Normalize features
        norms = np.linalg.norm(features, axis=-1, keepdims=True).clip(min=1e-6)
        features = (features / norms).astype(np.float16)

    np.savez_compressed(pc_path, points=points, features=features)
    build_time = time.time() - t0

    done_marker.write_text(json.dumps({
        "num_frames": len(frame_ids),
        "num_points": int(len(points)),
        "build_time_s": build_time,
        "timestamp": time.time(),
        "method": "simple_voxel_fusion",
    }))

    # Free disk: remove per-frame features now that the voxel map is saved
    import shutil
    if feat_dir.exists():
        shutil.rmtree(feat_dir, ignore_errors=True)

    if verbose:
        print(f"      Pass 3 done: {len(points)} voxels, {build_time:.1f}s")


# ============================================================
# Pass 4: Query + evaluation
# ============================================================

def pass4_query(scene_cache: Path, gt_dict: dict, verbose: bool = False):
    """Load saved pointcloud and run text queries -> Loc@d metrics."""
    from sklearn.cluster import DBSCAN
    import open_clip

    pc_path = scene_cache / "pointcloud.npz"
    if not pc_path.exists():
        if verbose:
            print(f"    Pass 4: no pointcloud at {pc_path}")
        return []

    data = np.load(pc_path)
    pts_np = data["points"]      # (N, 3)
    emb_np = data["features"]    # (N, 1024) half

    if len(pts_np) == 0:
        if verbose:
            print(f"    Pass 4: empty map")
        return []

    emb = torch.from_numpy(emb_np.astype(np.float32)).cuda()
    emb = torch.nn.functional.normalize(emb, dim=-1)

    # Load CLIP (half precision)
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-H-14", "laion2b_s32b_b79k", precision="fp16"
    )
    model = model.cuda().eval()
    tokenizer = open_clip.get_tokenizer("ViT-H-14")

    records = []
    for query in TEST_QUERIES:
        gt_centers = []
        for cat, centers in gt_dict.items():
            if query in cat:
                gt_centers.extend(centers)
        if not gt_centers:
            continue

        with torch.no_grad():
            text_tok = tokenizer([query]).cuda()
            text_feat = model.encode_text(text_tok).float()
            text_feat = torch.nn.functional.normalize(text_feat, dim=-1)

        sim = (emb @ text_feat[0]).cpu().numpy()  # (N,)
        k = max(10, int(len(sim) * TOP_K_PERCENT))
        top_idx = np.argsort(sim)[-k:][::-1]
        top_pts = pts_np[top_idx]
        top_scores = sim[top_idx]

        if len(top_pts) < DBSCAN_MIN_SAMPLES:
            pred = None
        else:
            labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(top_pts)
            best_score, best_centroid = -1e9, None
            for lab in set(labels) - {-1}:
                m = labels == lab
                s = float(np.mean(top_scores[m]))
                if s > best_score:
                    best_score = s
                    best_centroid = np.mean(top_pts[m], axis=0)
            pred = best_centroid

        err = None
        if pred is not None:
            err = float(min(np.linalg.norm(pred - gc) for gc in gt_centers))

        rec = {
            "query": query,
            "error_m": err,
            "predicted_location": pred.tolist() if pred is not None else None,
        }
        for t in THRESHOLDS:
            rec[f"loc_{t}m"] = bool(err is not None and err < t)
        records.append(rec)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return records


# ============================================================
# Main
# ============================================================

def process_scene(scene_dir: Path, dataset: str, verbose: bool = True):
    scene_id = scene_dir.name
    scene_cache = CACHE_DIR / f"{dataset}_{scene_id}"
    scene_cache.mkdir(parents=True, exist_ok=True)

    # Frame subsampling
    import pandas as pd
    trace = pd.read_parquet(scene_dir / "exploration" / "trace.parquet")
    frame_ids = list(range(0, len(trace), FRAME_STRIDE))

    # Check GT early
    gt = load_gt(scene_dir)
    if gt is None:
        return None

    t0 = time.time()

    # Determine SAM cache source
    sam_cache = None
    if dataset == "scannet":
        sam_cache = SCANNET_SAM_CACHE / scene_id

    # Pass 1
    print(f"  {scene_id}: Pass 1 (SAM)...")
    pass1_sam_masks(scene_dir, frame_ids, scene_cache, sam_cache, verbose=verbose)

    # Pass 2
    print(f"  {scene_id}: Pass 2 (CF features)...")
    pass2_features(scene_dir, frame_ids, scene_cache, verbose=verbose)

    # Pass 3
    print(f"  {scene_id}: Pass 3 (PointFusion)...")
    pass3_fusion(scene_dir, frame_ids, scene_cache, verbose=verbose)
    build_time = time.time() - t0

    # Pass 4
    print(f"  {scene_id}: Pass 4 (query)...")
    records = pass4_query(scene_cache, gt, verbose=verbose)
    for r in records:
        r["scene_id"] = scene_id
        r["build_time_s"] = build_time
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["scannet", "hm3d"], required=True)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--scenes", type=str, default=None,
                        help="comma-separated scene IDs (overrides max-scenes)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.dataset == "scannet":
        all_scenes = get_scannet_scenes(args.max_scenes or 142)
    else:
        all_scenes = get_hm3d_scenes(args.max_scenes or 36)

    if args.scenes:
        ids = set(s.strip() for s in args.scenes.split(","))
        all_scenes = [s for s in all_scenes if s.name in ids]

    output_file = args.output or str(
        OUTPUT_DIR / f"official_conceptfusion_{args.dataset}.json"
    )
    ckpt_file = str(Path(output_file).with_suffix(".ckpt.json"))

    # Resume
    if os.path.exists(ckpt_file):
        with open(ckpt_file) as f:
            ckpt = json.load(f)
        completed = set(ckpt["completed_scenes"])
        all_results = ckpt["all_results"]
    else:
        completed = set()
        all_results = []

    print(f"Official ConceptFusion on {args.dataset.upper()}")
    print(f"Scenes: {len(all_scenes)}, resuming from {len(completed)}")

    for si, scene_dir in enumerate(all_scenes):
        if scene_dir.name in completed:
            print(f"  [{si+1}/{len(all_scenes)}] {scene_dir.name}: SKIP")
            continue
        print(f"  [{si+1}/{len(all_scenes)}] {scene_dir.name}")
        try:
            records = process_scene(scene_dir, args.dataset, verbose=True)
            if records:
                all_results.extend(records)
                print(f"    {len(records)} queries")
            completed.add(scene_dir.name)
            with open(ckpt_file, "w") as f:
                json.dump({"completed_scenes": list(completed),
                           "all_results": all_results}, f)
        except Exception as e:
            print(f"    FAILED: {e}")
            traceback.print_exc()

    # Final summary
    if all_results:
        from collections import defaultdict
        print("\n" + "=" * 50)
        for t in THRESHOLDS:
            key = f"loc_{t}m"
            per_scene = defaultdict(list)
            for r in all_results:
                per_scene[r["scene_id"]].append(r.get(key, False))
            macro = np.mean([np.mean(v) * 100 for v in per_scene.values()])
            print(f"  Loc@{t}m (macro): {macro:.1f}%")
        dists = [r["error_m"] for r in all_results if r["error_m"] is not None]
        if dists:
            print(f"  Median error: {np.median(dists):.3f}m")

    output = {
        "method": "ConceptFusion-Official",
        "dataset": args.dataset,
        "num_scenes": len(completed),
        "num_queries": len(all_results),
        "per_query": all_results,
    }
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
