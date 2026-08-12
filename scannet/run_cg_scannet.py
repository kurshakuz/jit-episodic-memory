#!/usr/bin/env python3
"""
Run ConceptGraphs (official pipeline) on ScanNet evaluation scenes.

Adapted from baselines/run_official_cg.py for ScanNet intrinsics:
- Per-scene fx, fy, cx, cy from intrinsics.json (NOT hardcoded HFOV=90°)
- ScanNet trace.parquet format with sensor_height offset
- Same 4-pass pipeline: SAM -> CLIP -> Mapping -> Evaluation

Usage (must use 'cg' conda environment):
    conda run -n cg python scannet/run_cg_scannet.py
    conda run -n cg python scannet/run_cg_scannet.py --scenes scene0568_00
"""

import argparse
import gc
import gzip
import json
import os
import pickle
import sys
import time
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# Mock GradSLAM before importing ConceptGraphs
for _mod in [
    "gradslam",
    "gradslam.datasets",
    "gradslam.datasets.datautils",
    "gradslam.geometry",
    "gradslam.geometry.geometryutils",
    "gradslam.slam",
    "gradslam.slam.pointfusion",
    "gradslam.structures",
    "gradslam.structures.rgbdimages",
]:
    sys.modules[_mod] = ModuleType(_mod)
sys.modules["gradslam.datasets.datautils"].poses_to_transforms = lambda x: x
sys.modules["gradslam.geometry.geometryutils"].relative_transformation = lambda x, y: x
sys.modules["gradslam.slam.pointfusion"].PointFusion = type("PointFusion", (), {})
sys.modules["gradslam.structures.rgbdimages"].RGBDImages = type("RGBDImages", (), {})

# Mock pytorch3d
_mock_p3d = ModuleType("pytorch3d")
_mock_p3d_ops = ModuleType("pytorch3d.ops")
def _box3d_overlap_stub(*a, **kw):
    raise ValueError("pytorch3d unavailable – using axis-aligned IoU fallback")
_mock_p3d_ops.box3d_overlap = _box3d_overlap_stub
_mock_p3d.ops = _mock_p3d_ops
sys.modules["pytorch3d"] = _mock_p3d
sys.modules["pytorch3d.ops"] = _mock_p3d_ops

# Now safe to import ConceptGraphs
from conceptgraph.slam.slam_classes import MapObjectList, DetectionList
from conceptgraph.slam.mapping import (
    compute_spatial_similarities,
    compute_visual_similarities,
    aggregate_similarities,
    merge_detections_to_objects,
)
from conceptgraph.slam.utils import create_object_pcd, process_pcd, get_bounding_box, denoise_objects, filter_objects, merge_objects, filter_gobs, resize_gobs
from conceptgraph.utils.ious import mask_subtract_contained
from conceptgraph.utils.general_utils import to_tensor

torch.set_grad_enabled(False)

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from scannet.config import (
    SCANNET_VAL_SCENES, SCANNET_JIT, SCANNET_RESULTS,
    JIT_QUERIES_10, SCANNET_SYNONYMS, LOCALIZATION_THRESHOLDS,
)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
SAM_H_CKPT = MODELS_DIR / "sam_vit_h_4b8939.pth"
CACHE_DIR = SCANNET_RESULTS / "cg_scannet_cache"

# Default image dimensions (ScanNet extracted to 640x480)
IMG_H, IMG_W = 480, 640

# Official CG hyperparameters
class CGConfig:
    spatial_sim_type = "overlap"
    phys_bias = 0.0
    match_method = "sim_sum"
    sim_threshold = 1.2
    use_contain_number = False
    contain_area_thresh = 0.95
    contain_mismatch_penalty = 0.5
    mask_area_threshold = 25
    mask_conf_threshold = 0.95
    max_bbox_area_ratio = 0.5
    skip_bg = True
    min_points_threshold = 16
    # Memory: at native ScanNet density the 2.5cm default accumulates 22-25GB of object
    # point clouds and OOMs. Coarser voxel cuts RAM ~4x with a <5cm centroid shift (neutral
    # for the 1m localization metric). Env-overridable; default preserves official 2.5cm.
    downsample_voxel_size = float(os.environ.get("CG_VOXEL", "0.025"))
    dbscan_remove_noise = True
    dbscan_eps = 0.1
    dbscan_min_points = 10
    obj_min_points = 0
    obj_min_detections = 1
    merge_overlap_thresh = 0.7
    merge_visual_sim_thresh = 0.8
    merge_text_sim_thresh = 0.8
    denoise_interval = 20
    filter_interval = -1
    merge_interval = 20
    class_agnostic = True
    device = "cpu"

    def __getitem__(self, key):
        return getattr(self, key)
    def __contains__(self, key):
        return hasattr(self, key)

# SAM parameters
SAM_POINTS_PER_SIDE = 12
SAM_POINTS_PER_BATCH = 64
SAM_PRED_IOU_THRESH = 0.88
SAM_STABILITY_SCORE_THRESH = 0.95
SAM_CROP_N_LAYERS = 0
SAM_MIN_MASK_REGION_AREA = 100

BG_CLASSES = ["wall", "floor", "ceiling"]


#  Helpers

def load_scene_intrinsics(scene_dir):
    """Load per-scene camera intrinsics."""
    intr_path = scene_dir / "intrinsics.json"
    with open(intr_path) as f:
        intr = json.load(f)
    fx = intr["fx"]
    fy = intr["fy"]
    cx = intr["cx"]
    cy = intr["cy"]
    sensor_height = intr.get("sensor_height", 1.5)
    cam_k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return cam_k, sensor_height


def load_scene_frames(scene_dir, stride, sensor_height=1.5):
    """Load RGB images, depth maps, and camera poses for a ScanNet scene."""
    import pandas as pd

    # OpenGL->OpenCV coordinate conversion (same as HM3D pipeline)
    P = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1],
    ], dtype=np.float64)

    trace = pd.read_parquet(scene_dir / "exploration" / "trace.parquet")
    frames = []
    for i in range(0, len(trace), stride):
        row = trace.iloc[i]
        img_path = scene_dir / "exploration" / row["image_path"]
        depth_path = scene_dir / "exploration" / row["depth_path"]

        # Build 4x4 camera-to-world pose from quaternion + position
        quat = [row["qw"], row["qx"], row["qy"], row["qz"]]
        pos = [row["x"], row["y"], row["z"]]

        # Add sensor_height offset to Y (positions are agent-base)
        pos[1] += sensor_height

        R_mat = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        c2w = np.eye(4)
        c2w[:3, :3] = R_mat
        c2w[:3, 3] = pos

        # Convert from OpenCV camera frame -> Habitat world
        c2w_cv = c2w @ P

        frames.append({
            "idx": i,
            "image_path": str(img_path),
            "depth_path": str(depth_path),
            "pose": c2w_cv,
        })
    return frames


def load_scene_gt(scene_dir):
    gt_path = scene_dir / f"{scene_dir.name}_ground_truth.json"
    with open(gt_path) as f:
        return json.load(f)


def get_gt_centers_for_query(gt, query):
    centers = []
    synonyms = SCANNET_SYNONYMS.get(query.lower(), [query.lower()])
    synonyms = [query.lower()] + [s.lower() for s in synonyms if s.lower() != query.lower()]
    for obj_id, obj in gt["objects"].items():
        label = obj["category"].lower()
        for syn in synonyms:
            if syn in label:
                centers.append(np.array(obj["center"]))
                break
    return centers


#  Pass 1: SAM

def run_sam_pass(scene_dirs, stride):
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    print("\n" + "="*70)
    print("PASS 1: SAM ViT-H mask generation")
    print("="*70)

    assert SAM_H_CKPT.exists(), f"SAM ViT-H checkpoint not found: {SAM_H_CKPT}"

    print(f"Loading SAM ViT-H from {SAM_H_CKPT}...")
    sam = sam_model_registry["vit_h"](checkpoint=str(SAM_H_CKPT))
    sam.eval()
    sam.prompt_encoder.to("cuda")
    sam.mask_decoder.to("cuda")

    # Monkey-patch for split-device
    from segment_anything.predictor import SamPredictor as _SamPredictor
    def _split_set_torch_image(self, transformed_image, original_image_size):
        self.reset_image()
        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])
        input_image = self.model.preprocess(transformed_image.cpu())
        with torch.no_grad():
            self.features = self.model.image_encoder(input_image).to("cuda")
        self.is_image_set = True
    _SamPredictor.set_torch_image = torch.no_grad()(_split_set_torch_image)
    _SamPredictor.device = property(lambda self: torch.device("cuda"))

    torch.set_num_threads(os.cpu_count() or 8)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=SAM_POINTS_PER_SIDE,
        points_per_batch=SAM_POINTS_PER_BATCH,
        pred_iou_thresh=SAM_PRED_IOU_THRESH,
        stability_score_thresh=SAM_STABILITY_SCORE_THRESH,
        crop_n_layers=SAM_CROP_N_LAYERS,
        min_mask_region_area=SAM_MIN_MASK_REGION_AREA,
    )

    total_masks = 0
    total_frames = 0

    for scene_dir in scene_dirs:
        scene_id = scene_dir.name
        _, sensor_height = load_scene_intrinsics(scene_dir)
        scene_cache = CACHE_DIR / scene_id
        scene_cache.mkdir(parents=True, exist_ok=True)

        done_marker = scene_cache / "sam_pass_done.json"
        if done_marker.exists():
            with open(done_marker) as f:
                info = json.load(f)
            total_masks += info["total_masks"]
            total_frames += info["n_frames"]
            print(f"  [{scene_id}] Already done ({info['n_frames']} frames, {info['total_masks']} masks)")
            continue

        frames = load_scene_frames(scene_dir, stride, sensor_height)
        scene_masks = 0

        for fi, frame in enumerate(tqdm(frames, desc=f"SAM {scene_id}", leave=False)):
            mask_save_path = scene_cache / f"masks_{frame['idx']:06d}.pkl.gz"
            if mask_save_path.exists():
                continue

            image_rgb = np.array(Image.open(frame["image_path"]).convert("RGB"))
            results = mask_generator.generate(image_rgb)
            torch.cuda.synchronize()

            masks_list, xyxy_list, conf_list = [], [], []
            for r in results:
                masks_list.append(r["segmentation"])
                r_xyxy = list(r["bbox"])
                r_xyxy[2] += r_xyxy[0]
                r_xyxy[3] += r_xyxy[1]
                xyxy_list.append(r_xyxy)
                conf_list.append(r["predicted_iou"])

            sam_data = {
                "masks": np.array(masks_list) if masks_list else np.zeros((0, IMG_H, IMG_W), dtype=bool),
                "xyxy": np.array(xyxy_list) if xyxy_list else np.zeros((0, 4)),
                "conf": np.array(conf_list) if conf_list else np.zeros((0,)),
            }

            with gzip.open(mask_save_path, "wb") as f:
                pickle.dump(sam_data, f)
            scene_masks += len(masks_list)

        total_masks += scene_masks
        total_frames += len(frames)

        with open(done_marker, "w") as f:
            json.dump({"n_frames": len(frames), "total_masks": scene_masks}, f)
        print(f"  [{scene_id}] {len(frames)} frames, {scene_masks} masks")

    del mask_generator, sam
    torch.cuda.empty_cache()
    gc.collect()
    print(f"\nSAM pass complete: {total_frames} frames, {total_masks} masks")


#  Pass 2: CLIP ViT-H-14

def run_clip_pass(scene_dirs, stride):
    import open_clip

    print("\n" + "="*70)
    print("PASS 2: CLIP ViT-H-14 feature extraction")
    print("="*70)

    print("Loading CLIP ViT-H-14...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
    clip_model = clip_model.to("cuda").eval()
    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

    item_text = clip_tokenizer(["item"]).to("cuda")
    item_text_feat = clip_model.encode_text(item_text)
    item_text_feat = (item_text_feat / item_text_feat.norm(dim=-1, keepdim=True)).cpu().numpy()

    total_features = 0

    for scene_dir in scene_dirs:
        scene_id = scene_dir.name
        _, sensor_height = load_scene_intrinsics(scene_dir)
        scene_cache = CACHE_DIR / scene_id

        done_marker = scene_cache / "clip_pass_done.json"
        if done_marker.exists():
            with open(done_marker) as f:
                info = json.load(f)
            total_features += info["total_features"]
            print(f"  [{scene_id}] Already done ({info['total_features']} features)")
            continue

        frames = load_scene_frames(scene_dir, stride, sensor_height)
        scene_features = 0

        for fi, frame in enumerate(tqdm(frames, desc=f"CLIP {scene_id}", leave=False)):
            det_save_path = scene_cache / f"detections_{frame['idx']:06d}.pkl.gz"
            if det_save_path.exists():
                scene_features += 1
                continue

            mask_path = scene_cache / f"masks_{frame['idx']:06d}.pkl.gz"
            if not mask_path.exists():
                continue

            with gzip.open(mask_path, "rb") as f:
                sam_data = pickle.load(f)

            masks = sam_data["masks"]
            xyxy = sam_data["xyxy"]
            conf = sam_data["conf"]

            if len(masks) == 0:
                det_data = {
                    "xyxy": np.zeros((0, 4)),
                    "confidence": np.zeros((0,)),
                    "class_id": np.zeros((0,), dtype=int),
                    "mask": np.zeros((0, IMG_H, IMG_W), dtype=bool),
                    "classes": ["item"],
                    "image_crops": [],
                    "image_feats": np.zeros((0, 1024)),
                    "text_feats": np.zeros((0, 1024)),
                }
                with gzip.open(det_save_path, "wb") as f:
                    pickle.dump(det_data, f)
                continue

            image_rgb = np.array(Image.open(frame["image_path"]).convert("RGB"))
            image_pil = Image.fromarray(image_rgb)
            padding = 20
            image_feats_list = []

            for mask_idx in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[mask_idx]
                iw, ih = image_pil.size
                lp = min(padding, x1)
                tp = min(padding, y1)
                rp = min(padding, iw - x2)
                bp = min(padding, ih - y2)
                crop = image_pil.crop((x1 - lp, y1 - tp, x2 + rp, y2 + bp))
                preprocessed = clip_preprocess(crop).unsqueeze(0).to("cuda")
                feat = clip_model.encode_image(preprocessed)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                image_feats_list.append(feat.cpu().numpy())

            image_feats = np.concatenate(image_feats_list, axis=0)
            text_feats = np.tile(item_text_feat, (len(xyxy), 1))

            det_data = {
                "xyxy": xyxy,
                "confidence": conf,
                "class_id": np.zeros(len(conf), dtype=int),
                "mask": masks,
                "classes": ["item"],
                "image_crops": [None] * len(conf),
                "image_feats": image_feats,
                "text_feats": text_feats,
            }

            with gzip.open(det_save_path, "wb") as f:
                pickle.dump(det_data, f)
            scene_features += len(xyxy)

        total_features += scene_features
        with open(done_marker, "w") as f:
            json.dump({"total_features": scene_features}, f)
        print(f"  [{scene_id}] {scene_features} features computed")

    del clip_model, clip_preprocess, clip_tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    print(f"\nCLIP pass complete: {total_features} total features")


#  Pass 3: CG 3D Mapping (per-scene intrinsics)

def run_mapping_pass(scene_dirs, stride):
    print("\n" + "="*70)
    print("PASS 3: Official CG 3D mapping (CPU + Open3D)")
    print("="*70)

    cfg = CGConfig()

    for scene_dir in scene_dirs:
        scene_id = scene_dir.name
        scene_cache = CACHE_DIR / scene_id
        cam_k, sensor_height = load_scene_intrinsics(scene_dir)

        map_save_path = scene_cache / "object_map.pkl.gz"
        if map_save_path.exists():
            print(f"  [{scene_id}] Map already built, skipping")
            continue

        frames = load_scene_frames(scene_dir, stride, sensor_height)
        objects = MapObjectList(device="cpu")
        classes = ["item"]

        n_processed = 0
        for fi, frame in enumerate(tqdm(frames, desc=f"Map {scene_id}", leave=False)):
            det_path = scene_cache / f"detections_{frame['idx']:06d}.pkl.gz"
            if not det_path.exists():
                continue

            with gzip.open(det_path, "rb") as f:
                gobs = pickle.load(f)

            image_rgb = np.array(Image.open(frame["image_path"]).convert("RGB"))
            depth = np.load(frame["depth_path"]).astype(np.float32)
            pose = frame["pose"]

            gobs = resize_gobs(gobs, image_rgb)
            gobs = filter_gobs(cfg, gobs, image_rgb, BG_CLASSES)

            if len(gobs["xyxy"]) == 0:
                n_processed += 1
                continue

            xyxy = gobs["xyxy"]
            mask = gobs["mask"]
            gobs["mask"] = mask_subtract_contained(xyxy, mask)

            fg_detection_list = DetectionList()
            n_masks = len(gobs["xyxy"])

            for mask_idx in range(n_masks):
                local_class_id = gobs["class_id"][mask_idx]
                m = gobs["mask"][mask_idx]
                class_name = gobs["classes"][local_class_id]

                # Use per-scene CAM_K (NOT hardcoded HM3D intrinsics!)
                camera_pcd = create_object_pcd(depth, m, cam_k, image_rgb, obj_color=None)

                if len(camera_pcd.points) < max(cfg.min_points_threshold, 5):
                    continue

                global_pcd = camera_pcd.transform(pose)
                global_pcd = process_pcd(global_pcd, cfg)

                if len(global_pcd.points) < 4:
                    continue

                pcd_bbox = get_bounding_box(cfg, global_pcd)
                pcd_bbox.color = [0, 1, 0]

                if pcd_bbox.volume() < 1e-6:
                    continue

                detected_object = {
                    "image_idx": [frame["idx"]],
                    "mask_idx": [mask_idx],
                    "color_path": [frame["image_path"]],
                    "class_name": [class_name],
                    "class_id": [0],
                    "num_detections": 1,
                    "mask": [m],
                    "xyxy": [gobs["xyxy"][mask_idx]],
                    "conf": [gobs["confidence"][mask_idx]],
                    "n_points": [len(global_pcd.points)],
                    "pixel_area": [m.sum()],
                    "contain_number": [None],
                    "inst_color": np.random.rand(3),
                    "is_background": False,
                    "pcd": global_pcd,
                    "bbox": pcd_bbox,
                    "clip_ft": to_tensor(gobs["image_feats"][mask_idx]),
                    "text_ft": to_tensor(gobs["text_feats"][mask_idx]),
                }
                fg_detection_list.append(detected_object)

            if len(fg_detection_list) == 0:
                n_processed += 1
                continue

            if len(objects) == 0:
                for det in fg_detection_list:
                    objects.append(det)
                n_processed += 1
                continue

            spatial_sim = compute_spatial_similarities(cfg, fg_detection_list, objects)
            visual_sim = compute_visual_similarities(cfg, fg_detection_list, objects)
            agg_sim = aggregate_similarities(cfg, spatial_sim, visual_sim)
            agg_sim[agg_sim < cfg.sim_threshold] = float("-inf")

            objects = merge_detections_to_objects(cfg, fg_detection_list, objects, agg_sim)

            if cfg.denoise_interval > 0 and (n_processed + 1) % cfg.denoise_interval == 0:
                objects = denoise_objects(cfg, objects)
            if cfg.filter_interval > 0 and (n_processed + 1) % cfg.filter_interval == 0:
                objects = filter_objects(cfg, objects)
            if cfg.merge_interval > 0 and (n_processed + 1) % cfg.merge_interval == 0:
                objects = merge_objects(cfg, objects)

            n_processed += 1

        # Final post-processing
        objects = denoise_objects(cfg, objects)
        objects = filter_objects(cfg, objects)
        objects = merge_objects(cfg, objects)

        n_final = len(objects)
        print(f"  [{scene_id}] {n_final} objects after post-processing")

        map_data = {
            "n_objects": n_final,
            "objects": objects.to_serializable(),
        }
        with gzip.open(map_save_path, "wb") as f:
            pickle.dump(map_data, f)


#  Pass 4: Evaluation

def query_map(objects_serialized, query_text, clip_model, clip_tokenizer, device="cuda"):
    if len(objects_serialized) == 0:
        return None

    tokens = clip_tokenizer([query_text]).to(device)
    text_feat = clip_model.encode_text(tokens)
    text_feat = (text_feat / text_feat.norm(dim=-1, keepdim=True)).cpu()

    best_sim = -1.0
    best_centroid = None

    for obj in objects_serialized:
        clip_ft = torch.from_numpy(obj["clip_ft"]) if isinstance(obj["clip_ft"], np.ndarray) else obj["clip_ft"]
        clip_ft = clip_ft.float()
        if clip_ft.dim() == 1:
            clip_ft = clip_ft.unsqueeze(0)

        sim = torch.nn.functional.cosine_similarity(text_feat, clip_ft, dim=-1).item()
        if sim > best_sim:
            best_sim = sim
            pts = obj["pcd_np"]
            best_centroid = pts.mean(axis=0)

    return best_centroid


def run_eval_pass(scene_dirs):
    import open_clip
    from collections import defaultdict

    print("\n" + "="*70)
    print("PASS 4: Evaluation (Loc@d metrics)")
    print("="*70)

    print("Loading CLIP ViT-H-14 for text queries...")
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
    clip_model = clip_model.to("cuda").eval()
    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

    all_results = []
    thresholds = LOCALIZATION_THRESHOLDS

    for scene_dir in scene_dirs:
        scene_id = scene_dir.name
        scene_cache = CACHE_DIR / scene_id
        map_path = scene_cache / "object_map.pkl.gz"

        if not map_path.exists():
            print(f"  [{scene_id}] No map found, skipping")
            continue

        with gzip.open(map_path, "rb") as f:
            map_data = pickle.load(f)

        objects = map_data["objects"]
        gt = load_scene_gt(scene_dir)

        for query in JIT_QUERIES_10:
            gt_centers = get_gt_centers_for_query(gt, query)
            if not gt_centers:
                continue

            pred_pos = query_map(objects, query, clip_model, clip_tokenizer)

            if pred_pos is None:
                error = None
            else:
                dists = [float(np.linalg.norm(pred_pos - gc)) for gc in gt_centers]
                error = min(dists)

            result = {
                "scene_id": scene_id,
                "query": query,
                "method": "conceptgraphs",
                "error_m": error,
                "pred_position": pred_pos.tolist() if pred_pos is not None else None,
                "gt_locations": [gc.tolist() for gc in gt_centers],
            }
            for t in thresholds:
                result[f"loc_{t}m"] = error is not None and error <= t

            all_results.append(result)

            status = f"[OK] {error:.2f}m" if error is not None else "[FAIL] no pred"
            print(f"    [{scene_id}] {query}: {status}")

    del clip_model, clip_tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    return all_results


#  Main

def main():
    parser = argparse.ArgumentParser(description="ConceptGraphs on ScanNet")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--scenes", type=str, default=None)
    parser.add_argument("--skip-sam", action="store_true")
    parser.add_argument("--skip-clip", action="store_true")
    parser.add_argument("--skip-map", action="store_true")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override data directory (default: SCANNET_JIT)")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Override cache directory")
    args = parser.parse_args()

    global CACHE_DIR
    data_dir = Path(args.data_dir) if args.data_dir else SCANNET_JIT
    cache_dir = Path(args.cache_dir) if args.cache_dir else CACHE_DIR
    # Override module-level CACHE_DIR for pass functions
    CACHE_DIR = cache_dir

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    scenes_list = SCANNET_VAL_SCENES
    if args.scenes:
        scenes_list = [s.strip() for s in args.scenes.split(",")]

    scene_dirs = []
    for sid in scenes_list:
        sd = data_dir / sid
        gt = sd / f"{sid}_ground_truth.json"
        trace = sd / "exploration" / "trace.parquet"
        if sd.exists() and gt.exists() and trace.exists():
            scene_dirs.append(sd)

    print("="*70)
    print("ConceptGraphs Pipeline on ScanNet")
    print("="*70)
    print(f"  Scenes: {len(scene_dirs)}")
    print(f"  Stride: {args.stride}")
    print(f"  Cache: {CACHE_DIR}")
    print(f"  Intrinsics: per-scene from intrinsics.json (HFOV≈58°)")

    t_start = time.time()

    if not args.skip_sam:
        run_sam_pass(scene_dirs, args.stride)
    if not args.skip_clip:
        run_clip_pass(scene_dirs, args.stride)
    if not args.skip_map:
        run_mapping_pass(scene_dirs, args.stride)

    all_results = run_eval_pass(scene_dirs)
    elapsed = time.time() - t_start

    # Aggregate
    n = len(all_results)
    print(f"\n{'='*70}")
    print(f"RESULTS: ConceptGraphs on ScanNet ({n} queries, {elapsed/60:.1f}min)")
    print(f"{'='*70}")

    for t in LOCALIZATION_THRESHOLDS:
        hits = sum(1 for r in all_results if r.get(f"loc_{t}m", False))
        print(f"  Loc@{t}m: {100*hits/n:.1f}%" if n > 0 else f"  Loc@{t}m: N/A")

    errors = [r["error_m"] for r in all_results if r["error_m"] is not None]
    if errors:
        print(f"  Median error: {np.median(errors):.2f}m")
    no_pred = sum(1 for r in all_results if r["error_m"] is None)
    print(f"  No prediction: {no_pred}/{n}")

    # Save
    output = {
        "method": "ConceptGraphs (official, ViT-H)",
        "dataset": "scannet",
        "intrinsics": "per-scene (HFOV≈58°)",
        "stride": args.stride,
        "scenes": [sd.name for sd in scene_dirs],
        "aggregate": {
            "n_queries": n,
            "n_scenes": len(scene_dirs),
        },
        "per_query": all_results,
        "elapsed_s": elapsed,
    }

    for t in LOCALIZATION_THRESHOLDS:
        hits = sum(1 for r in all_results if r.get(f"loc_{t}m", False))
        output["aggregate"][f"loc_{t}m"] = round(100 * hits / n, 1) if n > 0 else 0

    results_path = SCANNET_RESULTS / (args.output or "scannet_conceptgraphs_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
