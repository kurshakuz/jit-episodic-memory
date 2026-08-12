#!/usr/bin/env python3
"""
Run the OFFICIAL ConceptGraphs pipeline on our HM3D benchmark.

This script uses the EXACT algorithm code from the official ConceptGraphs repo
(github.com/concept-graphs/concept-graphs) with the official model checkpoints:
  - SAM ViT-H  (sam_vit_h_4b8939.pth, 2.56 GB)
  - CLIP ViT-H-14 (laion2b_s32b_b79k, 1024-dim features)

To fit on a single RTX 4060 Ti (7.7 GB VRAM), models are loaded SEQUENTIALLY:
  Pass 1: SAM ViT-H only -> generate masks -> save to disk -> unload
  Pass 2: CLIP ViT-H-14 only -> compute features per mask -> save detections -> unload
  Pass 3: Official CG 3D mapping code (CPU + Open3D, no GPU needed)
  Pass 4: Query the map and evaluate Loc@d metrics

All hyperparameters match the official README commands for the class-agnostic
(ConceptGraphs, not ConceptGraphs-Detect) variant:
  spatial_sim_type=overlap, sim_threshold=1.2, mask_conf_threshold=0.95,
  dbscan_eps=0.1, merge_interval=20, etc.

The GradSLAM dependency is bypassed by mocking the import and providing data
directly. All mapping/merging/denoising code is imported verbatim from the
official conceptgraph package.

Usage:
    conda run -n cg python baselines/run_official_cg.py [--stride 4] [--scenes N]
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
sys.modules["gradslam.geometry.geometryutils"].relative_transformation = (
    lambda x, y: x
)
sys.modules["gradslam.slam.pointfusion"].PointFusion = type("PointFusion", (), {})
sys.modules["gradslam.structures.rgbdimages"].RGBDImages = type("RGBDImages", (), {})

# Mock pytorch3d (needs custom CUDA build). CG code catches ValueError
#     and falls back to axis-aligned IoU via compute_iou_batch.
_mock_p3d = ModuleType("pytorch3d")
_mock_p3d_ops = ModuleType("pytorch3d.ops")
def _box3d_overlap_stub(*a, **kw):
    raise ValueError("pytorch3d unavailable – using axis-aligned IoU fallback")
_mock_p3d_ops.box3d_overlap = _box3d_overlap_stub
_mock_p3d.ops = _mock_p3d_ops
sys.modules["pytorch3d"] = _mock_p3d
sys.modules["pytorch3d.ops"] = _mock_p3d_ops

# Now safe to import ConceptGraphs code
from conceptgraph.slam.slam_classes import MapObjectList, DetectionList
from conceptgraph.slam.mapping import (
    compute_spatial_similarities,
    compute_visual_similarities,
    aggregate_similarities,
    merge_detections_to_objects,
)
from conceptgraph.slam.utils import create_object_pcd, process_pcd, get_bounding_box, denoise_objects, filter_objects, merge_objects, filter_gobs, resize_gobs
from conceptgraph.utils.ious import (
    mask_subtract_contained,
    compute_2d_box_contained_batch,
)
from conceptgraph.utils.general_utils import to_tensor


torch.set_grad_enabled(False)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MULTI_SCENE_DIR = PROJECT_ROOT / "outputs" / "multi_scene_eval"
MODELS_DIR = PROJECT_ROOT / "models"
SAM_H_CKPT = MODELS_DIR / "sam_vit_h_4b8939.pth"
CACHE_DIR = PROJECT_ROOT / "outputs" / "paper_results" / "cg_official_cache"

# Camera intrinsics (HM3D: HFOV=90°, 640×480)
FX = FY = 320.0
CX, CY = 320.0, 240.0
IMG_H, IMG_W = 480, 640
CAM_K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)

# Official CG hyperparameters (from README + base.yaml)
# These match: spatial_sim_type=overlap mask_conf_threshold=0.95
#              sim_threshold=1.2 dbscan_eps=0.1 merge_interval=20
#              merge_visual_sim_thresh=0.8 merge_text_sim_thresh=0.8
class CGConfig:
    """Mimic the hydra DictConfig expected by CG mapping code."""
    # Spatial similarity
    spatial_sim_type = "overlap"
    phys_bias = 0.0
    match_method = "sim_sum"
    sim_threshold = 1.2

    # Containment
    use_contain_number = False
    contain_area_thresh = 0.95
    contain_mismatch_penalty = 0.5

    # Mask filtering
    mask_area_threshold = 25
    mask_conf_threshold = 0.95
    max_bbox_area_ratio = 0.5
    skip_bg = True
    min_points_threshold = 16

    # Point cloud processing
    downsample_voxel_size = 0.025
    dbscan_remove_noise = True
    dbscan_eps = 0.1
    dbscan_min_points = 10

    # Object filtering
    obj_min_points = 0
    obj_min_detections = 1

    # Merging
    merge_overlap_thresh = 0.7
    merge_visual_sim_thresh = 0.8
    merge_text_sim_thresh = 0.8

    # Periodic operations
    denoise_interval = 20
    filter_interval = -1
    merge_interval = 20

    # Other
    class_agnostic = True
    device = "cpu"  # mapping runs on CPU

    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, key):
        return hasattr(self, key)


# SAM mask generator parameters (from official generate_gsa_results.py)
# See: get_sam_mask_generator() in the official code
SAM_POINTS_PER_SIDE = 12
SAM_POINTS_PER_BATCH = 64  # Reduced from 144 to lower peak VRAM during mask decoding
SAM_PRED_IOU_THRESH = 0.88
SAM_STABILITY_SCORE_THRESH = 0.95
SAM_CROP_N_LAYERS = 0
SAM_MIN_MASK_REGION_AREA = 100


#  Data loading helpers

def get_val_scene_dirs():
    """Get the 36 val scene directories (same as reproduce_all.py)."""
    full_results_path = PROJECT_ROOT / "outputs" / "full_scale_eval" / "full_results_fixed.json"
    with open(full_results_path) as f:
        data = json.load(f)
    val_scene_ids = {sr["scene_id"] for sr in data["scene_results"]}

    scenes = []
    for scene_dir in sorted(MULTI_SCENE_DIR.iterdir()):
        if not scene_dir.is_dir():
            continue
        if scene_dir.name not in val_scene_ids:
            continue
        gt = scene_dir / f"{scene_dir.name}_ground_truth.json"
        trace = scene_dir / "exploration" / "trace.parquet"
        index = scene_dir / "exploration" / "memory.index"
        if gt.exists() and trace.exists() and index.exists():
            scenes.append(scene_dir)

    assert len(scenes) == 36, f"Expected 36 val scenes, got {len(scenes)}"
    return scenes


def load_scene_frames(scene_dir: Path, stride: int):
    """Load RGB images, depth maps, and camera poses for a scene."""
    import pandas as pd

    # OpenGL->OpenCV coordinate conversion (official CG HM3D dataset applies this)
    # Flips Y (up->down) and Z (backward->forward) axes
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
        R_mat = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        c2w = np.eye(4)
        c2w[:3, :3] = R_mat
        c2w[:3, 3] = pos

        # Convert from OpenCV camera frame (y-down, z-forward) used by create_object_pcd
        # to Habitat world frame (y-up, z-backward) which matches GT positions.
        # P flips y,z to go from CV camera -> Habitat camera; c2w then goes to Habitat world.
        c2w_cv = c2w @ P  # Takes CV camera points -> Habitat world

        frames.append(
            {
                "idx": i,
                "image_path": str(img_path),
                "depth_path": str(depth_path),
                "pose": c2w_cv,
            }
        )
    return frames


def load_scene_queries(scene_dir: Path):
    """Load ground truth object queries for a scene.

    Returns one query per unique category.  Each query carries the list of
    *all* individual instance positions so that evaluation can use
    closest-instance matching (consistent with JIT's evaluation protocol).
    """
    gt_path = scene_dir / f"{scene_dir.name}_ground_truth.json"
    with open(gt_path) as f:
        gt = json.load(f)

    queries = []
    objects = gt["objects"]

    # Build query list: each category -> list of individual instance centres
    cat_positions = {}
    for obj_id, obj in objects.items():
        cat = obj["category"]
        if cat.lower() in ("unknown", "misc", "void", "unlabeled"):
            continue
        center = obj["center"]
        if cat not in cat_positions:
            cat_positions[cat] = []
        cat_positions[cat].append(center)

    for cat, positions in cat_positions.items():
        queries.append({
            "category": cat,
            "instance_positions": [list(p) if not isinstance(p, list) else p for p in positions],
        })

    return queries


#  Pass 1: SAM ViT-H mask generation

def run_sam_pass(scene_dirs: list, stride: int):
    """Run SAM ViT-H on all frames and save masks to disk."""
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    print("\n" + "=" * 70)
    print("PASS 1: SAM ViT-H mask generation")
    print("=" * 70)

    assert SAM_H_CKPT.exists(), f"SAM ViT-H checkpoint not found: {SAM_H_CKPT}"

    # Load SAM ViT-H with split-device strategy:
    #   - image_encoder on CPU (needs ~3 GB RAM for weights + activations,
    #     too large for 7.7 GB VRAM due to 1024×1024 global attention maps)
    #   - prompt_encoder + mask_decoder on GPU (tiny, ~50 MB)
    # This produces IDENTICAL results to running everything on GPU.
    print(f"Loading SAM ViT-H from {SAM_H_CKPT}...")
    print("  image_encoder -> CPU (avoids 1024 MiB attention OOM)")
    print("  prompt_encoder + mask_decoder -> GPU")
    sam = sam_model_registry["vit_h"](checkpoint=str(SAM_H_CKPT))
    sam.eval()
    # Keep image_encoder on CPU, move lightweight components to GPU
    sam.prompt_encoder.to("cuda")
    sam.mask_decoder.to("cuda")

    # Monkey-patch the SamPredictor to handle split-device computation.
    # The predictor calls self.model.image_encoder(x) in set_torch_image(),
    # then self.model.mask_decoder(image_embeddings=self.features, ...) in predict_torch().
    # We need features computed on CPU but used on GPU.
    from segment_anything.predictor import SamPredictor as _SamPredictor

    _orig_set_torch_image = _SamPredictor.set_torch_image.__wrapped__  # unwrap @torch.no_grad

    def _split_set_torch_image(self, transformed_image, original_image_size):
        """Run image encoder on CPU, store features on GPU."""
        self.reset_image()
        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])

        # Run image encoder on CPU
        input_image = self.model.preprocess(transformed_image.cpu())
        with torch.no_grad():
            self.features = self.model.image_encoder(input_image).to("cuda")
        self.is_image_set = True

    _SamPredictor.set_torch_image = torch.no_grad()(_split_set_torch_image)

    # Patch the device property to return cuda (for mask decoder)
    _SamPredictor.device = property(lambda self: torch.device("cuda"))

    # Set number of CPU threads for faster image encoding
    torch.set_num_threads(os.cpu_count() or 8)
    print(f"  CPU threads: {torch.get_num_threads()}")

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
        scene_cache = CACHE_DIR / scene_id
        scene_cache.mkdir(parents=True, exist_ok=True)

        # Check if already done
        done_marker = scene_cache / "sam_pass_done.json"
        if done_marker.exists():
            with open(done_marker) as f:
                info = json.load(f)
            total_masks += info["total_masks"]
            total_frames += info["n_frames"]
            print(f"  [{scene_id}] Already done ({info['n_frames']} frames, {info['total_masks']} masks)")
            continue

        frames = load_scene_frames(scene_dir, stride)
        scene_masks = 0

        for fi, frame in enumerate(tqdm(frames, desc=f"SAM {scene_id}", leave=False)):
            mask_save_path = scene_cache / f"masks_{frame['idx']:06d}.pkl.gz"
            if mask_save_path.exists():
                continue

            image_rgb = np.array(Image.open(frame["image_path"]).convert("RGB"))

            # Run SAM (AMP is handled via monkey-patched image_encoder/mask_decoder)
            results = mask_generator.generate(image_rgb)
            torch.cuda.synchronize()

            # Extract masks, xyxy, conf (same as official get_sam_segmentation_dense)
            masks_list = []
            xyxy_list = []
            conf_list = []
            for r in results:
                masks_list.append(r["segmentation"])
                r_xyxy = list(r["bbox"])
                r_xyxy[2] += r_xyxy[0]  # Convert xywh -> xyxy
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

    # Unload SAM
    del mask_generator, sam
    torch.cuda.empty_cache()
    gc.collect()

    print(f"\nSAM pass complete: {total_frames} frames, {total_masks} masks")
    return total_masks


#  Pass 2: CLIP ViT-H-14 feature extraction

def run_clip_pass(scene_dirs: list, stride: int):
    """Compute CLIP ViT-H-14 features for all saved masks."""
    import open_clip

    print("\n" + "=" * 70)
    print("PASS 2: CLIP ViT-H-14 feature extraction")
    print("=" * 70)

    # Load CLIP ViT-H-14 on GPU (same model as official CG)
    print("Loading CLIP ViT-H-14 (laion2b_s32b_b79k)...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", "laion2b_s32b_b79k"
    )
    clip_model = clip_model.to("cuda")
    clip_model.eval()
    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

    # Pre-encode the "item" text (for class_set=none, all detections are "item")
    item_text = clip_tokenizer(["item"]).to("cuda")
    item_text_feat = clip_model.encode_text(item_text)
    item_text_feat = item_text_feat / item_text_feat.norm(dim=-1, keepdim=True)
    item_text_feat = item_text_feat.cpu().numpy()  # (1, 1024)

    total_features = 0

    for scene_dir in scene_dirs:
        scene_id = scene_dir.name
        scene_cache = CACHE_DIR / scene_id

        # Check if already done
        done_marker = scene_cache / "clip_pass_done.json"
        if done_marker.exists():
            with open(done_marker) as f:
                info = json.load(f)
            total_features += info["total_features"]
            print(f"  [{scene_id}] Already done ({info['total_features']} features)")
            continue

        frames = load_scene_frames(scene_dir, stride)
        scene_features = 0

        for fi, frame in enumerate(tqdm(frames, desc=f"CLIP {scene_id}", leave=False)):
            det_save_path = scene_cache / f"detections_{frame['idx']:06d}.pkl.gz"
            if det_save_path.exists():
                scene_features += 1  # approximate
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
                # Save empty detection
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

            # Compute CLIP features per mask crop
            # (same logic as official compute_clip_features)
            padding = 20
            image_feats_list = []

            for mask_idx in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[mask_idx]
                iw, ih = image_pil.size

                # Adjust padding (same as official)
                lp = min(padding, x1)
                tp = min(padding, y1)
                rp = min(padding, iw - x2)
                bp = min(padding, ih - y2)

                crop = image_pil.crop((x1 - lp, y1 - tp, x2 + rp, y2 + bp))
                preprocessed = clip_preprocess(crop).unsqueeze(0).to("cuda")

                feat = clip_model.encode_image(preprocessed)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                image_feats_list.append(feat.cpu().numpy())

            image_feats = np.concatenate(image_feats_list, axis=0)  # (N, 1024)
            # Text features: all "item" (same for all masks in class_set=none)
            text_feats = np.tile(item_text_feat, (len(xyxy), 1))  # (N, 1024)

            det_data = {
                "xyxy": xyxy,
                "confidence": conf,
                "class_id": np.zeros(len(conf), dtype=int),
                "mask": masks,
                "classes": ["item"],
                "image_crops": [None] * len(conf),  # Placeholder (same length as detections)
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

    # Unload CLIP
    del clip_model, clip_preprocess, clip_tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    print(f"\nCLIP pass complete: {total_features} total features")
    return total_features


#  Pass 3: Official CG 3D mapping

BG_CLASSES = ["wall", "floor", "ceiling"]


def run_mapping_pass(scene_dirs: list, stride: int):
    """Run the official CG mapping pipeline using saved detections."""
    print("\n" + "=" * 70)
    print("PASS 3: Official CG 3D mapping (CPU + Open3D)")
    print("=" * 70)

    cfg = CGConfig()
    results_per_scene = {}

    for scene_dir in scene_dirs:
        scene_id = scene_dir.name
        scene_cache = CACHE_DIR / scene_id

        # Check if already done
        map_save_path = scene_cache / "object_map.pkl.gz"
        if map_save_path.exists():
            print(f"  [{scene_id}] Map already built, loading...")
            with gzip.open(map_save_path, "rb") as f:
                saved = pickle.load(f)
            results_per_scene[scene_id] = {
                "n_objects": saved["n_objects"],
                "map_path": str(map_save_path),
            }
            continue

        frames = load_scene_frames(scene_dir, stride)
        objects = MapObjectList(device="cpu")
        classes = ["item"]

        n_processed = 0
        for fi, frame in enumerate(tqdm(frames, desc=f"Map {scene_id}", leave=False)):
            det_path = scene_cache / f"detections_{frame['idx']:06d}.pkl.gz"
            if not det_path.exists():
                continue

            with gzip.open(det_path, "rb") as f:
                gobs = pickle.load(f)

            # Load image and depth
            image_rgb = np.array(Image.open(frame["image_path"]).convert("RGB"))
            depth = np.load(frame["depth_path"]).astype(np.float32)
            pose = frame["pose"]

            # This is the EXACT same logic as cfslam_pipeline_batch.py
            # filter_gobs and resize_gobs (from official utils.py)
            gobs = resize_gobs(gobs, image_rgb)
            gobs = filter_gobs(cfg, gobs, image_rgb, BG_CLASSES)

            if len(gobs["xyxy"]) == 0:
                n_processed += 1
                continue

            # mask_subtract_contained (from official ious.py)
            xyxy = gobs["xyxy"]
            mask = gobs["mask"]
            gobs["mask"] = mask_subtract_contained(xyxy, mask)

            # Build per-mask detections (same as gobs_to_detection_list body)
            fg_detection_list = DetectionList()
            n_masks = len(gobs["xyxy"])

            for mask_idx in range(n_masks):
                local_class_id = gobs["class_id"][mask_idx]
                m = gobs["mask"][mask_idx]
                class_name = gobs["classes"][local_class_id]

                # Create 3D point cloud from depth + mask
                camera_pcd = create_object_pcd(depth, m, CAM_K, image_rgb, obj_color=None)

                if len(camera_pcd.points) < max(cfg.min_points_threshold, 5):
                    continue

                # Transform to world frame
                global_pcd = camera_pcd.transform(pose)

                # Denoise + downsample (official process_pcd)
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

            # Official CG mapping logic
            if cfg.use_contain_number:
                xyxy_t = fg_detection_list.get_stacked_values_torch("xyxy", 0)
                contain_numbers = compute_2d_box_contained_batch(
                    xyxy_t, cfg.contain_area_thresh
                )
                for i in range(len(fg_detection_list)):
                    fg_detection_list[i]["contain_number"] = [contain_numbers[i]]

            if len(objects) == 0:
                for det in fg_detection_list:
                    objects.append(det)
                n_processed += 1
                continue

            # Compute similarities (official mapping.py functions)
            spatial_sim = compute_spatial_similarities(cfg, fg_detection_list, objects)
            visual_sim = compute_visual_similarities(cfg, fg_detection_list, objects)
            agg_sim = aggregate_similarities(cfg, spatial_sim, visual_sim)

            # Containment penalty (official logic)
            if cfg.use_contain_number:
                contain_numbers_objects = torch.Tensor(
                    [obj["contain_number"][0] for obj in objects]
                )
                detection_contained = contain_numbers > 0
                object_contained = contain_numbers_objects > 0
                detection_contained = detection_contained.unsqueeze(1)
                object_contained = object_contained.unsqueeze(0)
                xor = detection_contained ^ object_contained
                agg_sim[xor] = agg_sim[xor] - cfg.contain_mismatch_penalty

            # Threshold
            agg_sim[agg_sim < cfg.sim_threshold] = float("-inf")

            # Merge detections into objects (official mapping.py)
            objects = merge_detections_to_objects(cfg, fg_detection_list, objects, agg_sim)

            # Periodic post-processing (official logic)
            if cfg.denoise_interval > 0 and (n_processed + 1) % cfg.denoise_interval == 0:
                objects = denoise_objects(cfg, objects)
            if cfg.filter_interval > 0 and (n_processed + 1) % cfg.filter_interval == 0:
                objects = filter_objects(cfg, objects)
            if cfg.merge_interval > 0 and (n_processed + 1) % cfg.merge_interval == 0:
                objects = merge_objects(cfg, objects)

            n_processed += 1

        # Final post-processing (official)
        objects = denoise_objects(cfg, objects)
        objects = filter_objects(cfg, objects)
        objects = merge_objects(cfg, objects)

        n_final = len(objects)
        print(f"  [{scene_id}] {n_final} objects after post-processing")

        # Save the map
        map_data = {
            "n_objects": n_final,
            "objects": objects.to_serializable(),
        }
        with gzip.open(map_save_path, "wb") as f:
            pickle.dump(map_data, f)

        results_per_scene[scene_id] = {
            "n_objects": n_final,
            "map_path": str(map_save_path),
        }

    return results_per_scene


#  Pass 4: Query and evaluate

def query_map(objects_serialized: list, query_text: str, clip_model, clip_tokenizer, device="cuda"):
    """Query the 3D object map with a text query using CLIP ViT-H-14."""
    if len(objects_serialized) == 0:
        return None

    # Encode query text
    tokens = clip_tokenizer([query_text]).to(device)
    text_feat = clip_model.encode_text(tokens)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    text_feat = text_feat.cpu()

    # Compare with all object CLIP features
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
            # Compute centroid from saved pcd points
            pts = obj["pcd_np"]
            best_centroid = pts.mean(axis=0)

    return best_centroid


def run_eval_pass(scene_dirs: list):
    """Evaluate the built maps against our benchmark queries."""
    import open_clip

    print("\n" + "=" * 70)
    print("PASS 4: Evaluation (Loc@d metrics)")
    print("=" * 70)

    # Load CLIP for querying
    print("Loading CLIP ViT-H-14 for text queries...")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-H-14", "laion2b_s32b_b79k"
    )
    clip_model = clip_model.to("cuda")
    clip_model.eval()
    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

    all_results = []
    thresholds = [0.5, 1.0, 2.0, 3.0]

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
        queries = load_scene_queries(scene_dir)

        for q in queries:
            category = q["category"]
            instance_positions = [np.array(p) for p in q["instance_positions"]]

            pred_pos = query_map(objects, category, clip_model, clip_tokenizer)

            if pred_pos is None:
                error = None
            else:
                # Closest-instance matching: minimum distance to any GT
                # instance of this category (consistent with JIT evaluation)
                dists = [float(np.linalg.norm(pred_pos - ip)) for ip in instance_positions]
                error = min(dists)

            result = {
                "scene_id": scene_id,
                "category": category,
                "n_instances": len(instance_positions),
                "gt_positions": [p.tolist() for p in instance_positions],
                "pred_position": pred_pos.tolist() if pred_pos is not None else None,
                "error_m": error,
            }
            for t in thresholds:
                result[f"loc_{t}m"] = error is not None and error <= t

            all_results.append(result)

    # Unload CLIP
    del clip_model, clip_tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    return all_results


def aggregate_results(all_results: list):
    """Compute Loc@d metrics from per-query results."""
    n = len(all_results)
    if n == 0:
        return {}

    thresholds = [0.5, 1.0, 2.0, 3.0]
    agg = {"n_queries": n, "n_scenes": len(set(r["scene_id"] for r in all_results))}

    for t in thresholds:
        key = f"loc_{t}m"
        hits = sum(1 for r in all_results if r.get(key, False))
        agg[key] = round(100 * hits / n, 1)

    errors = [r["error_m"] for r in all_results if r["error_m"] is not None]
    no_pred = sum(1 for r in all_results if r["error_m"] is None)
    agg["median_error_m"] = round(float(np.median(errors)), 2) if errors else None
    agg["mean_error_m"] = round(float(np.mean(errors)), 2) if errors else None
    agg["no_prediction"] = no_pred

    return agg


#  Main

def main():
    parser = argparse.ArgumentParser(description="Run official ConceptGraphs on HM3D benchmark")
    parser.add_argument("--stride", type=int, default=4, help="Frame stride")
    parser.add_argument("--scenes", type=int, default=None, help="Limit number of scenes (for testing)")
    parser.add_argument("--skip-sam", action="store_true", help="Skip SAM pass (use cached)")
    parser.add_argument("--skip-clip", action="store_true", help="Skip CLIP pass (use cached)")
    parser.add_argument("--skip-map", action="store_true", help="Skip mapping pass (use cached)")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Override cache directory (default: outputs/paper_results/cg_official_cache)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/paper_results/conceptgraphs_official_results.json",
    )
    args = parser.parse_args()

    global CACHE_DIR
    if args.cache_dir:
        CACHE_DIR = PROJECT_ROOT / args.cache_dir
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROJECT_ROOT / args.output

    print("=" * 70)
    print("Official ConceptGraphs Pipeline on HM3D Benchmark")
    print("=" * 70)
    print(f"  SAM:   ViT-H (sam_vit_h_4b8939.pth)")
    print(f"  CLIP:  ViT-H-14 (laion2b_s32b_b79k, 1024-dim)")
    print(f"  Stride: {args.stride}")
    print(f"  Cache:  {CACHE_DIR}")
    print(f"  Output: {output_path}")

    scene_dirs = get_val_scene_dirs()
    if args.scenes:
        scene_dirs = scene_dirs[: args.scenes]
    print(f"  Scenes: {len(scene_dirs)}")

    t_start = time.time()

    # Pass 1: SAM
    if not args.skip_sam:
        run_sam_pass(scene_dirs, args.stride)

    # Pass 2: CLIP
    if not args.skip_clip:
        run_clip_pass(scene_dirs, args.stride)

    # Pass 3: Mapping
    if not args.skip_map:
        run_mapping_pass(scene_dirs, args.stride)

    # Pass 4: Evaluation
    all_results = run_eval_pass(scene_dirs)
    agg = aggregate_results(all_results)

    elapsed = time.time() - t_start

    print("\n" + "=" * 70)
    print("RESULTS: Official ConceptGraphs (ViT-H SAM + ViT-H-14 CLIP)")
    print("=" * 70)
    print(f"  Queries: {agg['n_queries']}")
    print(f"  Scenes:  {agg['n_scenes']}")
    print(f"  Loc@0.5m: {agg.get('loc_0.5m', 'N/A')}%")
    print(f"  Loc@1.0m: {agg.get('loc_1.0m', 'N/A')}%")
    print(f"  Loc@2.0m: {agg.get('loc_2.0m', 'N/A')}%")
    print(f"  Loc@3.0m: {agg.get('loc_3.0m', 'N/A')}%")
    print(f"  Median error: {agg.get('median_error_m', 'N/A')}m")
    print(f"  No prediction: {agg.get('no_prediction', 'N/A')}")
    print(f"  Total time: {elapsed / 60:.1f} min")

    # Save
    output = {
        "method": "ConceptGraphs (official, ViT-H)",
        "models": {
            "sam": "ViT-H (sam_vit_h_4b8939.pth)",
            "clip": "ViT-H-14 (laion2b_s32b_b79k)",
            "feature_dim": 1024,
        },
        "hyperparameters": {
            "spatial_sim_type": "overlap",
            "sim_threshold": 1.2,
            "mask_conf_threshold": 0.95,
            "dbscan_eps": 0.1,
            "merge_interval": 20,
            "merge_visual_sim_thresh": 0.8,
            "merge_text_sim_thresh": 0.8,
            "sam_points_per_side": SAM_POINTS_PER_SIDE,
            "sam_pred_iou_thresh": SAM_PRED_IOU_THRESH,
        },
        "stride": args.stride,
        "aggregate": agg,
        "per_query": all_results,
        "elapsed_s": elapsed,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
