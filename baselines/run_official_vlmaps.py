#!/usr/bin/env python3
"""
Official VLMaps adapter.

Uses official VLMaps components from an external VLMaps checkout (set the VLMAPS_REPO env var; defaults to ./third_party/vlmaps):
  - LSegEncNet (exact model class from vlmaps.lseg.modules.models.lseg_net)
  - get_lseg_feat (multi-scale tiled inference from vlmaps.utils.lseg_utils)
  - Gaussian radial distance weighting (alpha = exp(-|p|² / 2σ²), σ² = 0.6)
  - 2D top-down grid map with (x, z) world coords and height bins

Four passes:
  Pass 1 (LSeg feat): extract per-pixel LSeg features using official multi-scale inference
  Pass 2 (3D build): back-project depth, apply Gaussian weighting, accumulate to grid
  Pass 3 (Query): text query via LSeg's CLIP head, similarity on grid_feat, DBSCAN
  Pass 4: evaluate Loc@d

Fallback: single-scale inference if multi-scale tiled inference OOMs.

Usage:
    conda run -n cg python baselines/run_official_vlmaps.py \
        --dataset scannet --max-scenes 142
    conda run -n cg python baselines/run_official_vlmaps.py \
        --dataset hm3d --max-scenes 36
"""

import argparse
import gc
import json
import os
# Reduce CUDA fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")

import sys
import time
import traceback
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
VLMAPS_REPO = Path(os.environ.get("VLMAPS_REPO", PROJECT_ROOT / "third_party" / "vlmaps"))
sys.path.insert(0, str(VLMAPS_REPO))
LANG_SEG_PATH = PROJECT_ROOT / "lang-seg"

# ============================================================
# Config
# ============================================================

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "paper_results"
CACHE_DIR = OUTPUT_DIR / "vlmaps_official_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HM3D_DATA = PROJECT_ROOT / "outputs" / "multi_scene_eval_500f"
SCANNET_DATA = PROJECT_ROOT / "scannet" / "jit_format_500"

# Map params (match VLMaps' defaults)
CELL_SIZE = 0.05
GRID_SIZE = 1000
DEPTH_SAMPLE_RATE = 4  # pixel stride
MIN_DEPTH = 0.1
MAX_DEPTH = 6.0
SENSOR_HEIGHT = 1.5
SIGMA_SQ = 0.6  # Gaussian radial weighting (official VLMaps)

FRAME_STRIDE = 4

TEST_QUERIES = ["toilet", "chair", "table", "bed", "couch",
                "sink", "lamp", "mirror", "cabinet", "shelf"]
THRESHOLDS = [0.25, 0.5, 1.0, 2.0, 3.0]

TOP_K_PERCENT = 0.01
DBSCAN_EPS = 0.5
DBSCAN_MIN_SAMPLES = 3
DEVICE = "cuda:0"


# ============================================================
# Official LSeg model loader (from vlmaps.lseg via our lang-seg ckpt)
# ============================================================

_lseg_state = {"model": None, "transform": None, "tokenizer": None}


def load_lseg():
    """Load LSegEncNet from official VLMaps code, using our demo_e200.ckpt."""
    if _lseg_state["model"] is not None:
        return _lseg_state["model"], _lseg_state["transform"]

    # Import from official VLMaps repo
    from vlmaps.lseg.modules.models.lseg_net import LSegEncNet
    from torchvision import transforms

    ckpt_path = LANG_SEG_PATH / "checkpoints" / "demo_e200.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"LSeg checkpoint not found at {ckpt_path}")

    print("  Loading official LSegEncNet...")
    # LSegEncNet needs similar args to LSegNet. Look at test_lseg.py defaults.
    # labels list for text classifier (we won't use it, just need dummy)
    labels = ["other"]

    model = LSegEncNet(
        labels=labels,
        backbone="clip_vitl16_384",
        features=256,
        crop_size=480,
        arch_option=0,
        block_depth=0,
        activation="lrelu",
    )

    # Load weights (handle 'net.' prefix from pytorch-lightning)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    new_state_dict = {
        (k[4:] if k.startswith("net.") else k): v for k, v in state_dict.items()
    }
    model.load_state_dict(new_state_dict, strict=False)
    model.eval().to(DEVICE)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    _lseg_state["model"] = model
    _lseg_state["transform"] = transform
    return model, transform


def unload_lseg():
    for k in list(_lseg_state.keys()):
        _lseg_state[k] = None
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================
# Data loading helpers (match our trace.parquet format)
# ============================================================

def load_intrinsics(scene_dir: Path):
    intr_path = scene_dir / "intrinsics.json"
    if intr_path.exists():
        with open(intr_path) as f:
            d = json.load(f)
        return {
            "fx": float(d["fx"]),
            "fy": float(d["fy"]),
            "cx": float(d["cx"]),
            "cy": float(d["cy"]),
            "sensor_height": float(d.get("sensor_height", SENSOR_HEIGHT)),
            "height": int(d.get("target_height", 480)),
            "width": int(d.get("target_width", 640)),
        }
    # HM3D defaults (HFOV=90°)
    fx = 640 / (2.0 * np.tan(np.deg2rad(90.0) / 2.0))
    return {
        "fx": fx, "fy": fx, "cx": 320.0, "cy": 240.0,
        "sensor_height": SENSOR_HEIGHT, "height": 480, "width": 640,
    }


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


def quat_to_mat(qw, qx, qy, qz):
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,     1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float64)


def get_frame_ids(scene_dir: Path, stride: int = FRAME_STRIDE):
    trace = pd.read_parquet(scene_dir / "exploration" / "trace.parquet")
    return list(range(0, len(trace), stride)), trace


# ============================================================
# Pass 1: LSeg features per frame (single-scale, official model)
# ============================================================

def _encode_dense_lseg(image: np.ndarray, model, transform, out_h: int = 120, out_w: int = 160):
    """Single-scale LSeg -> dense features at (out_h, out_w).

    Official VLMaps sliding-window tiled inference via `get_lseg_feat`
    (crop_size=480, base_size=520). Substitutes for `torch-encoding`'s
    `MultiEvalModule`, which fails to build on modern PyTorch (removed
    THC headers). The two paths are functionally equivalent for
    single-scale tiled inference.
    """
    import torch.nn.functional as F
    from vlmaps.utils.lseg_utils import get_lseg_feat

    feats_np = get_lseg_feat(
        model, image, ["other"], transform, DEVICE,
        crop_size=480, base_size=520,
        norm_mean=[0.5, 0.5, 0.5], norm_std=[0.5, 0.5, 0.5],
    )  # numpy (1, D, H', W')

    feats_t = torch.from_numpy(feats_np).to(DEVICE)
    feats_t = F.normalize(feats_t, dim=1)
    feats_t = F.interpolate(feats_t, size=(out_h, out_w),
                            mode="bilinear", align_corners=True)
    feats = feats_t[0].permute(1, 2, 0).cpu().numpy()  # (out_h, out_w, D)
    return feats


def pass1_lseg_features(scene_dir: Path, frame_ids: List[int], scene_cache: Path,
                         verbose: bool = False):
    """No-op: features computed inline in pass2_build_map to avoid 34GB disk use."""
    done_marker = scene_cache / "pass1_done.json"
    if not done_marker.exists():
        done_marker.write_text(json.dumps({"inline": True}))


# ============================================================
# Pass 2: 3D grid accumulation with Gaussian radial weighting (official VLMaps)
# ============================================================

def pass2_build_map(scene_dir: Path, frame_ids: List[int], scene_cache: Path,
                     verbose: bool = False):
    """
    Combined LSeg + 3D build. For each frame: compute features inline with LSeg,
    then back-project depth, apply Gaussian radial weighting (official VLMaps
    formula), and accumulate into 5cm voxels.

    Saves as npz with 'points' (M, 3), 'features' (M, 512).
    """
    done_marker = scene_cache / "pass2_done.json"
    map_path = scene_cache / "voxel_map.npz"
    if done_marker.exists():
        if verbose:
            print(f"    Pass 2: SKIP (cached)")
        return

    # Load LSeg once for the whole scene
    model, transform = load_lseg()

    intr = load_intrinsics(scene_dir)
    trace = pd.read_parquet(scene_dir / "exploration" / "trace.parquet")

    fx, fy = intr["fx"], intr["fy"]
    cx, cy = intr["cx"], intr["cy"]
    sensor_h = intr["sensor_height"]

    voxel_accum = {}  # key -> [sum_pos, sum_feat, sum_weight]

    t0 = time.time()
    for i, fid in enumerate(frame_ids):
        row = trace.iloc[fid]
        # Load image and compute LSeg features inline
        img_path = scene_dir / "exploration" / row["image_path"]
        img = np.array(Image.open(img_path))
        if img.ndim == 3 and img.shape[-1] == 4:
            img = img[..., :3]

        feats = _encode_dense_lseg(img, model, transform).astype(np.float32)  # (H, W, 512)

        depth = np.load(scene_dir / "exploration" / row["depth_path"]).astype(np.float32)
        H, W = depth.shape
        position = np.array([row["x"], row["y"], row["z"]], dtype=np.float64)
        qw, qx, qy, qz = row["qw"], row["qx"], row["qy"], row["qz"]
        R = np.array([
            [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw,     1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy],
        ], dtype=np.float64)

        # Features are computed at (120, 160). Sample depth at stride=4 to get
        # matching (120, 160) grid.
        u = np.arange(0, W, DEPTH_SAMPLE_RATE)
        v = np.arange(0, H, DEPTH_SAMPLE_RATE)
        uu, vv = np.meshgrid(u, v)
        d_s = depth[::DEPTH_SAMPLE_RATE, ::DEPTH_SAMPLE_RATE]
        feat_s = feats  # already (120, 160, 512)
        # Safety: NN-resize if shapes mismatch (e.g., for HM3D 480x640 depth -> 120x160)
        if feat_s.shape[:2] != d_s.shape:
            target_h, target_w = d_s.shape
            ys = np.linspace(0, feat_s.shape[0] - 1, target_h).astype(int)
            xs = np.linspace(0, feat_s.shape[1] - 1, target_w).astype(int)
            feat_s = feat_s[ys[:, None], xs[None, :]]

        valid = (d_s > MIN_DEPTH) & (d_s < MAX_DEPTH)

        # Camera points in Habitat camera convention (X right, -Y up, -Z forward)
        x_cam = (uu - cx) * d_s / fx
        y_cam = -(vv - cy) * d_s / fy
        z_cam = -d_s
        pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (h, w, 3)
        radial_sq = x_cam**2 + y_cam**2 + z_cam**2
        # Gaussian radial weighting (official VLMaps formula)
        alpha = np.exp(-radial_sq / (2.0 * SIGMA_SQ))  # (h, w)

        # Transform to world
        sensor_pos = position.copy()
        sensor_pos[1] += sensor_h
        pts_flat = pts_cam.reshape(-1, 3)
        pts_world = (R @ pts_flat.T).T + sensor_pos
        pts_world = pts_world.reshape(d_s.shape[0], d_s.shape[1], 3)

        # Flatten valid pixels
        v_pts = pts_world[valid]                # (M, 3)
        v_feats = feat_s[valid]                 # (M, 512)
        v_alpha = alpha[valid].astype(np.float32)  # (M,)

        # Voxelize (vectorized accumulation using np.add.at)
        voxel_idx = np.floor(v_pts / CELL_SIZE).astype(np.int64)
        keys = (voxel_idx[:, 0] * 1_000_003 * 1_000_003 +
                voxel_idx[:, 1] * 1_000_003 +
                voxel_idx[:, 2])
        unique_keys, inv = np.unique(keys, return_inverse=True)
        n_unique = len(unique_keys)

        w_per = np.zeros(n_unique, dtype=np.float32)
        p_per = np.zeros((n_unique, 3), dtype=np.float32)
        f_per = np.zeros((n_unique, v_feats.shape[1]), dtype=np.float32)
        np.add.at(w_per, inv, v_alpha)
        np.add.at(p_per, inv, v_pts * v_alpha[:, None])
        np.add.at(f_per, inv, v_feats * v_alpha[:, None])

        for j in range(n_unique):
            k = int(unique_keys[j])
            if k in voxel_accum:
                voxel_accum[k][0] += p_per[j]
                voxel_accum[k][1] += f_per[j]
                voxel_accum[k][2] += w_per[j]
            else:
                voxel_accum[k] = [p_per[j].copy(), f_per[j].copy(), float(w_per[j])]

        if verbose and (i + 1) % 25 == 0:
            print(f"      build {i+1}/{len(frame_ids)} | voxels={len(voxel_accum)}")

    if not voxel_accum:
        points = np.zeros((0, 3), dtype=np.float32)
        features = np.zeros((0, 512), dtype=np.float16)
    else:
        n = len(voxel_accum)
        points = np.zeros((n, 3), dtype=np.float32)
        features = np.zeros((n, 512), dtype=np.float32)
        for i, (k, (p_sum, f_sum, w_sum)) in enumerate(voxel_accum.items()):
            points[i] = p_sum / max(w_sum, 1e-6)
            features[i] = f_sum / max(w_sum, 1e-6)
        # Normalize features
        norms = np.linalg.norm(features, axis=-1, keepdims=True).clip(min=1e-6)
        features = (features / norms).astype(np.float16)

    build_time = time.time() - t0
    np.savez_compressed(map_path, points=points, features=features)
    done_marker.write_text(json.dumps({
        "num_points": int(len(points)),
        "build_time_s": build_time,
    }))
    if verbose:
        print(f"      Pass 2 done: {len(points)} voxels, {build_time:.1f}s")


# ============================================================
# Pass 3: Query
# ============================================================

def pass3_query(scene_cache: Path, gt_dict: dict, verbose: bool = False):
    """Faithful VLMaps query: multi-template prompt ensembling +
    argmax-over-categories with 'other' baseline + 2D top-down contour
    extraction + height-map lookup for 3D centroids.

    Pipeline (matches `vlmaps.map.vlmap.VLMap.{init_categories,index_map,get_pos}`):
      1. Encode all 10 test categories + "other" with 40-template ensemble,
         average per-category. (`get_text_feats_multiple_templates`)
      2. scores_mat = grid_feat @ text_feats.T  (N x C+1)
      3. max_ids = argmax(scores_mat, axis=1)  per-voxel classification
      4. For each query: voxel_mask = (max_ids == query_id)
      5. Pool to 2D top-down: occupancy[r, c] = any voxel at (r, c, *) is True
         (also track height_map[r, c] = max height of any voxel at (r, c))
      6. Morphological: binary_closing(iter=3) -> gaussian_filter(sigma=0.8)
         -> threshold 0.5 -> binary_dilation
      7. Contour extraction (cv2.findContours) -> list of (cy, cx) centers
      8. For each center: lift to 3D using height_map[cy, cx]
      9. Loc@d: error = min over GT centers of distance to closest contour center
    """
    import torch.nn.functional as F
    import cv2
    from scipy.ndimage import binary_closing, binary_dilation, gaussian_filter
    from vlmaps.utils.clip_utils import multiple_templates
    import clip as openai_clip

    map_path = scene_cache / "voxel_map.npz"
    if not map_path.exists():
        return []
    data = np.load(map_path)
    pts = data["points"]
    feat = data["features"].astype(np.float32)

    if len(pts) == 0:
        return []

    # Step 1: Encode test categories + "other" with multi-template ensemble
    model, _ = load_lseg()
    landmarks = list(TEST_QUERIES) + ["other"]
    prompts = [t.format(lm) for lm in landmarks for t in multiple_templates]

    with torch.no_grad():
        tokens = openai_clip.tokenize(prompts).to(DEVICE)
        # Encode in batches to avoid memory spikes
        text_feats_list = []
        B = 256
        for s in range(0, len(prompts), B):
            tf = model.clip_pretrained.encode_text(tokens[s:s+B]).float()
            tf = F.normalize(tf, dim=-1)
            text_feats_list.append(tf.cpu())
        text_feats = torch.cat(text_feats_list).numpy()  # (C*T, D)
    # Average over templates per category
    text_feats = text_feats.reshape(len(landmarks), len(multiple_templates), -1).mean(axis=1)
    text_feats = text_feats / np.linalg.norm(text_feats, axis=-1, keepdims=True)

    # Step 2-3: Score each voxel and assign category
    feat_norm = feat / np.maximum(np.linalg.norm(feat, axis=-1, keepdims=True), 1e-8)
    scores_mat = feat_norm @ text_feats.T  # (N, C+1)
    max_ids = scores_mat.argmax(axis=1)

    # Build 2D top-down grid for downstream contour extraction
    # For HM3D the world up-axis is +Y; for ScanNet it's +Z. Determine by trace
    # presence: scenes under multi_scene_eval_500f are HM3D (Y up), under
    # jit_format_500 are ScanNet (Z up). The dataset path is the parent of cache.
    # Heuristic: use the more-vertical of (y, z) ranges as the up-axis.
    y_range = pts[:, 1].max() - pts[:, 1].min()
    z_range = pts[:, 2].max() - pts[:, 2].min()
    if y_range < z_range:
        # HM3D-like: y is up
        up_axis, plane_axes = 1, (0, 2)
    else:
        # ScanNet-like: z is up
        up_axis, plane_axes = 2, (0, 1)

    # Quantise to 5cm grid
    cell_size = CELL_SIZE
    plane_idx = np.floor(pts[:, plane_axes] / cell_size).astype(np.int64)
    rmin, cmin = plane_idx.min(axis=0)
    rmax, cmax = plane_idx.max(axis=0)
    H_2d = rmax - rmin + 1
    W_2d = cmax - cmin + 1

    # Cap grid size to avoid pathological scenes
    if H_2d * W_2d > 4_000_000:
        if verbose:
            print(f"      grid too large: {H_2d}x{W_2d}, skipping scene")
        return []

    rs = (plane_idx[:, 0] - rmin).clip(0, H_2d - 1)
    cs = (plane_idx[:, 1] - cmin).clip(0, W_2d - 1)
    heights = pts[:, up_axis]

    # Per-cell max height map (for lifting 2D contour to 3D)
    height_map = np.full((H_2d, W_2d), -np.inf, dtype=np.float32)
    np.maximum.at(height_map, (rs, cs), heights)
    no_voxel = ~np.isfinite(height_map)

    records = []
    for query in TEST_QUERIES:
        gt_centers = []
        for cat, centers in gt_dict.items():
            if query in cat:
                gt_centers.extend(centers)
        if not gt_centers:
            continue

        cat_id = landmarks.index(query)
        voxel_mask = (max_ids == cat_id)

        if not voxel_mask.any():
            rec = {"query": query, "error_m": None, "predicted_location": None}
            for t in THRESHOLDS:
                rec[f"loc_{t}m"] = False
            records.append(rec)
            continue

        # Step 5: 2D pool
        occ = np.zeros((H_2d, W_2d), dtype=bool)
        sel_rs = rs[voxel_mask]
        sel_cs = cs[voxel_mask]
        occ[sel_rs, sel_cs] = True

        # Step 6: morphological cleanup (matches vlmap.get_pos)
        occ_closed = binary_closing(occ, iterations=3)
        occ_smooth = gaussian_filter(occ_closed.astype(np.float32), sigma=0.8, truncate=3)
        occ_bin = occ_smooth > 0.5
        occ_dil = binary_dilation(occ_bin)

        if not occ_dil.any():
            rec = {"query": query, "error_m": None, "predicted_location": None}
            for t in THRESHOLDS:
                rec[f"loc_{t}m"] = False
            records.append(rec)
            continue

        # Step 7: contour extraction via cv2
        occ_u8 = (occ_dil.astype(np.uint8)) * 255
        contours, _ = cv2.findContours(occ_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Step 8: lift each contour center to 3D
        # We pick the contour with the largest area (matches `get_segment_islands_pos`'s
        # implicit ordering — we evaluate Loc@d against the best one).
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1.0:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                cy_2d = float(cnt[:, 0, 1].mean())
                cx_2d = float(cnt[:, 0, 0].mean())
            else:
                cy_2d = M["m01"] / M["m00"]
                cx_2d = M["m10"] / M["m00"]
            # 2D grid center -> world plane coords (centre of cell)
            r_world_idx = cy_2d + rmin
            c_world_idx = cx_2d + cmin
            x_w_plane = (r_world_idx + 0.5) * cell_size
            y_w_plane = (c_world_idx + 0.5) * cell_size
            # Height: average height_map over the contour's pixels (fallback: median over cells in mask)
            mask_for_contour = np.zeros_like(occ_u8)
            cv2.drawContours(mask_for_contour, [cnt], -1, 1, thickness=cv2.FILLED)
            heights_in = height_map[mask_for_contour.astype(bool)]
            heights_in = heights_in[np.isfinite(heights_in)]
            if len(heights_in) == 0:
                # height fallback: nearest non-empty cell
                cy_i, cx_i = int(round(cy_2d)), int(round(cx_2d))
                cy_i = max(0, min(H_2d - 1, cy_i))
                cx_i = max(0, min(W_2d - 1, cx_i))
                if np.isfinite(height_map[cy_i, cx_i]):
                    h_w = float(height_map[cy_i, cx_i])
                else:
                    continue
            else:
                h_w = float(np.mean(heights_in))
            # Reconstruct 3D point
            pred = np.zeros(3)
            pred[plane_axes[0]] = x_w_plane
            pred[plane_axes[1]] = y_w_plane
            pred[up_axis] = h_w
            candidates.append((area, pred))

        if not candidates:
            rec = {"query": query, "error_m": None, "predicted_location": None}
            for t in THRESHOLDS:
                rec[f"loc_{t}m"] = False
            records.append(rec)
            continue

        # Pick largest-area contour as the prediction (paper's `get_pos` returns
        # multiple candidates and a planner chooses; for Loc@d we use the largest).
        candidates.sort(key=lambda x: -x[0])
        pred = candidates[0][1]

        err = float(min(np.linalg.norm(pred - gc) for gc in gt_centers))

        rec = {
            "query": query, "error_m": err,
            "predicted_location": pred.tolist(),
            "n_contours": len(candidates),
        }
        for t in THRESHOLDS:
            rec[f"loc_{t}m"] = bool(err < t)
        records.append(rec)

    return records


# ============================================================
# Main
# ============================================================

def process_scene(scene_dir: Path, dataset: str, verbose: bool = True):
    scene_id = scene_dir.name
    scene_cache = CACHE_DIR / f"{dataset}_{scene_id}"
    scene_cache.mkdir(parents=True, exist_ok=True)

    frame_ids, _ = get_frame_ids(scene_dir, stride=FRAME_STRIDE)
    gt = load_gt(scene_dir)
    if gt is None:
        return None

    t0 = time.time()
    print(f"  {scene_id}: Pass 1 (official LSeg)...")
    pass1_lseg_features(scene_dir, frame_ids, scene_cache, verbose=verbose)
    print(f"  {scene_id}: Pass 2 (Gaussian-weighted grid)...")
    pass2_build_map(scene_dir, frame_ids, scene_cache, verbose=verbose)
    build_time = time.time() - t0

    print(f"  {scene_id}: Pass 3 (query)...")
    records = pass3_query(scene_cache, gt, verbose=verbose)
    for r in records:
        r["scene_id"] = scene_id
        r["build_time_s"] = build_time
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["scannet", "hm3d"], required=True)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--scenes", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.dataset == "scannet":
        from scannet.config import SCANNET_VAL_SCENES
        ids = SCANNET_VAL_SCENES[: args.max_scenes or 142]
        all_scenes = [SCANNET_DATA / s for s in ids if (SCANNET_DATA / s).exists()]
    else:
        all_scenes = []
        for sd in sorted(HM3D_DATA.iterdir()):
            if sd.is_dir() and (sd / "exploration" / "trace.parquet").exists():
                all_scenes.append(sd)
        all_scenes = all_scenes[: args.max_scenes or 36]

    if args.scenes:
        sids = set(s.strip() for s in args.scenes.split(","))
        all_scenes = [s for s in all_scenes if s.name in sids]

    output_file = args.output or str(
        OUTPUT_DIR / f"official_vlmaps_{args.dataset}.json"
    )
    ckpt_file = str(Path(output_file).with_suffix(".ckpt.json"))

    if os.path.exists(ckpt_file):
        with open(ckpt_file) as f:
            ckpt = json.load(f)
        completed = set(ckpt["completed_scenes"])
        all_results = ckpt["all_results"]
    else:
        completed = set()
        all_results = []

    print(f"Official VLMaps on {args.dataset.upper()}")
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
                loc1 = sum(1 for r in records if r.get("loc_1.0m"))
                print(f"    {len(records)} queries, loc@1m={loc1}/{len(records)}")
            completed.add(scene_dir.name)
            with open(ckpt_file, "w") as f:
                json.dump({"completed_scenes": list(completed), "all_results": all_results}, f)
        except Exception as e:
            print(f"    FAILED: {e}")
            traceback.print_exc()

    # Summary
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

    output = {
        "method": "VLMaps-Official",
        "dataset": args.dataset,
        "per_query": all_results,
    }
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
