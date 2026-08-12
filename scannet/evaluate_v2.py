#!/usr/bin/env python3
"""
ScanNet evaluation v2 for JIT Episodic Memory.

Improvements over v1:
- BF + Depth now matches HM3D protocol: single best OWL-ViT detection (no DBSCAN)
- Added JIT + Depth + L3: full 3-level cascade with verification stage
- Added McNemar's test for JIT vs BF comparison
- Added per-category breakdown

Usage:
    python scannet/evaluate_v2.py                              # All methods
    python scannet/evaluate_v2.py --methods jit,jit_l3,bf      # Select methods
    python scannet/evaluate_v2.py --scenes scene0568_00        # Single scene
"""

import os
import sys
import json
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from collections import defaultdict
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from scannet.config import (
    SCANNET_VAL_SCENES, SCANNET_JIT, SCANNET_RESULTS,
    JIT_QUERIES_10, SCANNET_SYNONYMS,
    LOCALIZATION_THRESHOLDS, BOOTSTRAP_ITERATIONS, CONFIDENCE_LEVEL,
    OWL_VIT_THRESHOLD, DBSCAN_EPS, DBSCAN_MIN_SAMPLES,
    DEPTH_PERCENTILE, TOP_K,
)


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class QueryResult:
    scene_id: str
    query: str
    method: str
    predicted_location: Optional[List[float]]
    gt_locations: List[List[float]]
    min_distance: Optional[float]
    correct_at: Dict[str, bool] = field(default_factory=dict)
    latency_ms: float = 0.0
    num_detections: int = 0
    num_clusters: int = 0
    ranked_centroids: Optional[List[Dict]] = None
    recall_at_k: Optional[Dict[str, bool]] = None


# ============================================================================
# Ground truth loading
# ============================================================================

def load_scene_gt(scene_dir: Path) -> dict:
    scene_id = scene_dir.name
    gt_path = scene_dir / f"{scene_id}_ground_truth.json"
    if not gt_path.exists():
        return None
    with open(gt_path) as f:
        return json.load(f)


def get_gt_centers_for_query(gt: dict, query: str) -> List[np.ndarray]:
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


# ============================================================================
# Depth projection (ScanNet-adapted)
# ============================================================================

class ScanNetProjector:
    def __init__(self, intrinsics_path: str):
        with open(intrinsics_path) as f:
            config = json.load(f)
        self.fx = config["fx"]
        self.fy = config["fy"]
        self.cx = config["cx"]
        self.cy = config["cy"]
        self.width = config["target_width"]
        self.height = config["target_height"]
        self.sensor_height = config.get("sensor_height", 1.5)

    def project_detection_to_3d(self, u, v, depth_map, agent_pos, quaternion,
                                 patch_radius=15, depth_percentile=30):
        u_int, v_int = int(round(u)), int(round(v))
        r = patch_radius
        v_min = max(0, v_int - r)
        v_max = min(self.height, v_int + r + 1)
        u_min = max(0, u_int - r)
        u_max = min(self.width, u_int + r + 1)

        patch = depth_map[v_min:v_max, u_min:u_max]
        valid = patch[(patch > 0.1) & (patch < 10.0)]
        if len(valid) == 0:
            return None

        d = np.percentile(valid, depth_percentile)

        x_cam = (u - self.cx) * d / self.fx
        y_cam = -(v - self.cy) * d / self.fy
        z_cam = -d
        p_cam = np.array([x_cam, y_cam, z_cam])

        from scipy.spatial.transform import Rotation
        qw, qx, qy, qz = quaternion
        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        sensor_pos = np.array(agent_pos) + np.array([0.0, self.sensor_height, 0.0])
        p_world = R @ p_cam + sensor_pos
        return p_world


# ============================================================================
# OWL-ViT wrapper
# ============================================================================

_owl_vit_model = None
_owl_vit_processor = None


def load_owlvit():
    global _owl_vit_model, _owl_vit_processor
    if _owl_vit_model is None:
        print("  Loading OWL-ViT model...")
        from transformers import OwlViTProcessor, OwlViTForObjectDetection
        import torch
        _owl_vit_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        _owl_vit_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _owl_vit_model = _owl_vit_model.to(device).eval()
        print(f"  OWL-ViT loaded on {device}")
    return _owl_vit_model, _owl_vit_processor


def run_owlvit_detection(model_tuple, image_np, query, threshold):
    import torch
    if model_tuple is None:
        model, processor = load_owlvit()
    else:
        model, processor = model_tuple

    device = next(model.parameters()).device
    h, w = image_np.shape[:2]

    img_pil = Image.fromarray(image_np)
    inputs = processor(text=[[f"a photo of a {query}"]], images=img_pil,
                       return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([[h, w]], device=device)
    post_fn = getattr(processor, 'post_process_object_detection',
                      getattr(processor, 'post_process_grounded_object_detection', None))
    if post_fn is None:
        raise RuntimeError("No post-processing method found on OwlViTProcessor")
    results = post_fn(outputs=outputs, threshold=threshold, target_sizes=target_sizes)

    detections = []
    if len(results) > 0:
        boxes = results[0]["boxes"].cpu().numpy()
        scores = results[0]["scores"].cpu().numpy()
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            detections.append({
                "center": (float(cx), float(cy)),
                "score": float(score),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
            })
    return detections


def run_owlvit_detection_batch(model_tuple, images_np, query, threshold,
                               batch_size=8):
    """
    Batched OWL-ViT detection: process multiple images in GPU batches.

    Args:
        model_tuple: (model, processor) or None to auto-load
        images_np: list of RGB numpy arrays (H, W, 3)
        query: text query string
        threshold: detection confidence threshold
        batch_size: images per GPU batch (default 8, tune for VRAM)

    Returns:
        list of detection lists, one per input image (same order)
    """
    import torch
    if not images_np:
        return []

    if model_tuple is None:
        model, processor = load_owlvit()
    else:
        model, processor = model_tuple

    device = next(model.parameters()).device
    post_fn = getattr(processor, 'post_process_object_detection',
                      getattr(processor, 'post_process_grounded_object_detection', None))
    if post_fn is None:
        raise RuntimeError("No post-processing method found on OwlViTProcessor")

    all_detections = [[] for _ in range(len(images_np))]
    text_query = f"a photo of a {query}"

    for batch_start in range(0, len(images_np), batch_size):
        batch_imgs = images_np[batch_start:batch_start + batch_size]
        n = len(batch_imgs)

        pil_images = [Image.fromarray(img) for img in batch_imgs]
        target_sizes = torch.tensor(
            [[img.shape[0], img.shape[1]] for img in batch_imgs],
            device=device,
        )

        inputs = processor(
            text=[[text_query]] * n,
            images=pil_images,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        results = post_fn(outputs=outputs, threshold=threshold,
                          target_sizes=target_sizes)

        for i, res in enumerate(results):
            boxes = res["boxes"].cpu().numpy()
            scores = res["scores"].cpu().numpy()
            dets = []
            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                dets.append({
                    "center": (float(cx), float(cy)),
                    "score": float(score),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                })
            all_detections[batch_start + i] = dets

    return all_detections


# ============================================================================
# Method 1: JIT + Depth (L1 + L2, same as v1)
# ============================================================================

def run_jit_depth(query, scene_dir, projector, trace_df,
                  owl_vit_model=None, clip_encoder=None, faiss_indexer=None,
                  owl_batch_size=8):
    """JIT + Depth: CLIP retrieval -> OWL-ViT (batched) + depth + DBSCAN."""
    from sklearn.cluster import DBSCAN
    explore_dir = scene_dir / "exploration"
    t0 = time.time()

    # L1: CLIP retrieval
    text_emb = clip_encoder.encode_text(query)
    candidate_ids, similarities = faiss_indexer.search(text_emb, k=min(TOP_K, len(trace_df)))
    candidate_ids = [int(fid) for fid in candidate_ids]

    # Pre-load all candidate images for batched detection
    valid_rows = []   # (fid, row) pairs with valid image paths
    batch_images = []
    for fid in candidate_ids:
        row = trace_df[trace_df["frame_id"] == fid]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        img_path = explore_dir / row["image_path"]
        if not img_path.exists():
            continue
        img_np = np.array(Image.open(str(img_path)))
        valid_rows.append((fid, row))
        batch_images.append(img_np)

    # L2: Batched OWL-ViT + depth projection
    all_points, all_confs = [], []
    num_detections = 0

    batch_detections = run_owlvit_detection_batch(
        owl_vit_model, batch_images, query, OWL_VIT_THRESHOLD,
        batch_size=owl_batch_size,
    )

    for (fid, row), detections in zip(valid_rows, batch_detections):
        if not detections:
            continue

        depth_path = explore_dir / row["depth_path"]
        if not depth_path.exists():
            continue
        depth = np.load(str(depth_path))

        agent_pos = [row["x"], row["y"], row["z"]]
        quat = [row["qw"], row["qx"], row["qy"], row["qz"]]

        for det in detections:
            cx_det, cy_det = det["center"]
            p3d = projector.project_detection_to_3d(
                cx_det, cy_det, depth, agent_pos, quat,
                depth_percentile=DEPTH_PERCENTILE)
            if p3d is not None:
                all_points.append(p3d)
                all_confs.append(det["score"])
                num_detections += 1

    if len(all_points) == 0:
        return None, 0, 0, (time.time() - t0) * 1000, None

    points = np.array(all_points)
    confs = np.array(all_confs)

    clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(points)
    labels = clustering.labels_
    unique_labels = set(labels) - {-1}
    num_clusters = len(unique_labels)

    if num_clusters == 0:
        best_idx = np.argmax(confs)
        ranked = [{"centroid": points[best_idx].tolist(), "score": float(confs[best_idx]), "n_members": 1}]
        return points[best_idx].tolist(), num_detections, 0, (time.time() - t0) * 1000, ranked

    # Build ranked cluster list (sorted by total confidence descending)
    cluster_list = []
    for label in unique_labels:
        mask = labels == label
        cluster_score = float(confs[mask].sum())
        centroid = points[mask].mean(axis=0)
        cluster_list.append({
            "centroid": centroid.tolist(),
            "score": cluster_score,
            "n_members": int(mask.sum()),
        })
    cluster_list.sort(key=lambda c: c["score"], reverse=True)

    best_centroid = cluster_list[0]["centroid"]
    return best_centroid, num_detections, num_clusters, (time.time() - t0) * 1000, cluster_list


# ============================================================================
# Method 2: JIT + Depth + L3 (full 3-level cascade with verification)
# ============================================================================

def run_jit_depth_l3(query, scene_dir, projector, trace_df,
                     owl_vit_model=None, clip_encoder=None, faiss_indexer=None,
                     l3_max_verify=5, owl_batch_size=8):
    """
    JIT + Depth + L3: Full 3-level cascade.
    L1: CLIP retrieval -> top-k candidates
    L2: OWL-ViT detection (batched) + depth projection + DBSCAN -> clusters
    L3: Re-verify top clusters by running OWL-ViT on each cluster's best frame,
        then re-project using detection bbox for precise localization.
    """
    from sklearn.cluster import DBSCAN
    explore_dir = scene_dir / "exploration"
    t0 = time.time()

    # L1: CLIP retrieval
    text_emb = clip_encoder.encode_text(query)
    candidate_ids, similarities = faiss_indexer.search(text_emb, k=min(TOP_K, len(trace_df)))
    candidate_ids = [int(fid) for fid in candidate_ids]

    # Pre-load all candidate images for batched detection
    valid_rows = []
    batch_images = []
    for fid in candidate_ids:
        row = trace_df[trace_df["frame_id"] == fid]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        img_path = explore_dir / row["image_path"]
        if not img_path.exists():
            continue
        img_np = np.array(Image.open(str(img_path)))
        valid_rows.append((fid, row))
        batch_images.append(img_np)

    # L2: Batched OWL-ViT + depth on all candidates
    all_points, all_confs, all_frame_ids = [], [], []
    num_detections = 0

    batch_detections = run_owlvit_detection_batch(
        owl_vit_model, batch_images, query, OWL_VIT_THRESHOLD,
        batch_size=owl_batch_size,
    )

    for (fid, row), detections in zip(valid_rows, batch_detections):
        if not detections:
            continue

        depth_path = explore_dir / row["depth_path"]
        if not depth_path.exists():
            continue
        depth = np.load(str(depth_path))

        agent_pos = [row["x"], row["y"], row["z"]]
        quat = [row["qw"], row["qx"], row["qy"], row["qz"]]

        for det in detections:
            cx_det, cy_det = det["center"]
            p3d = projector.project_detection_to_3d(
                cx_det, cy_det, depth, agent_pos, quat,
                depth_percentile=DEPTH_PERCENTILE)
            if p3d is not None:
                all_points.append(p3d)
                all_confs.append(det["score"])
                all_frame_ids.append(fid)
                num_detections += 1

    if len(all_points) == 0:
        return None, 0, 0, (time.time() - t0) * 1000, None

    points = np.array(all_points)
    confs = np.array(all_confs)
    frame_ids = np.array(all_frame_ids)

    # DBSCAN clustering
    clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(points)
    labels = clustering.labels_
    unique_labels = set(labels) - {-1}
    num_clusters = len(unique_labels)

    if num_clusters == 0:
        best_idx = np.argmax(confs)
        ranked = [{"centroid": points[best_idx].tolist(), "score": float(confs[best_idx]), "n_members": 1}]
        return points[best_idx].tolist(), num_detections, 0, (time.time() - t0) * 1000, ranked

    # Rank clusters by total confidence
    cluster_info = []
    for label in unique_labels:
        mask = labels == label
        cluster_score = confs[mask].sum()
        # Best frame = frame with highest detection confidence in this cluster
        cluster_confs = confs[mask]
        cluster_frames = frame_ids[mask]
        best_frame_idx = np.argmax(cluster_confs)
        best_frame_id = int(cluster_frames[best_frame_idx])
        centroid = points[mask].mean(axis=0)
        cluster_info.append({
            "label": label,
            "score": cluster_score,
            "best_frame_id": best_frame_id,
            "centroid": centroid,
            "n_points": int(mask.sum()),
        })

    cluster_info.sort(key=lambda c: c["score"], reverse=True)

    # L3: Verify top clusters
    verified_clusters = []
    for ci in cluster_info[:l3_max_verify]:
        fid = ci["best_frame_id"]
        row = trace_df[trace_df["frame_id"] == fid]
        if len(row) == 0:
            continue
        row = row.iloc[0]

        img_path = explore_dir / row["image_path"]
        if not img_path.exists():
            continue
        img_np = np.array(Image.open(str(img_path)))

        # Re-run OWL-ViT for verification
        detections = run_owlvit_detection(owl_vit_model, img_np, query, OWL_VIT_THRESHOLD)
        if not detections:
            continue  # Verification failed — skip this cluster

        # Use the best detection for re-projection
        best_det = max(detections, key=lambda d: d["score"])

        depth_path = explore_dir / row["depth_path"]
        if not depth_path.exists():
            verified_clusters.append({
                "centroid": ci["centroid"],
                "score": best_det["score"],
            })
            continue
        depth = np.load(str(depth_path))

        agent_pos = [row["x"], row["y"], row["z"]]
        quat = [row["qw"], row["qx"], row["qy"], row["qz"]]

        cx_det, cy_det = best_det["center"]
        p3d = projector.project_detection_to_3d(
            cx_det, cy_det, depth, agent_pos, quat,
            depth_percentile=DEPTH_PERCENTILE)

        if p3d is not None:
            verified_clusters.append({
                "centroid": p3d,
                "score": best_det["score"],
            })
        else:
            # Depth projection failed; fallback to L2 cluster centroid
            verified_clusters.append({
                "centroid": ci["centroid"],
                "score": best_det["score"],
            })

    if not verified_clusters:
        # No cluster passed L3; fallback to L2 ranked clusters
        l2_ranked = []
        for ci in cluster_info:
            c = ci["centroid"]
            if isinstance(c, np.ndarray):
                c = c.tolist()
            l2_ranked.append({"centroid": c, "score": float(ci["score"]), "n_members": ci["n_points"]})
        best = cluster_info[0]
        bc = best["centroid"]
        if isinstance(bc, np.ndarray):
            bc = bc.tolist()
        return bc, num_detections, num_clusters, (time.time() - t0) * 1000, l2_ranked

    # Build ranked verified cluster list
    verified_ranked = []
    for vc in verified_clusters:
        c = vc["centroid"]
        if isinstance(c, np.ndarray):
            c = c.tolist()
        verified_ranked.append({"centroid": c, "score": float(vc["score"]), "n_members": 1})
    verified_ranked.sort(key=lambda x: x["score"], reverse=True)

    best_centroid = verified_ranked[0]["centroid"]
    return best_centroid, num_detections, num_clusters, (time.time() - t0) * 1000, verified_ranked


# ============================================================================
# Method 3: Brute Force + Depth (matching HM3D protocol — NO DBSCAN)
# ============================================================================

def run_bruteforce_depth(query, scene_dir, projector, trace_df,
                         owl_vit_model=None, max_frames=100,
                         owl_batch_size=8):
    """
    Brute Force + Depth: OWL-ViT (batched) on random frames, pick single best detection.
    Matches HM3D protocol: no DBSCAN, return single best-scoring projected point.
    """
    explore_dir = scene_dir / "exploration"
    t0 = time.time()

    n = min(max_frames, len(trace_df))
    sample_indices = random.sample(range(len(trace_df)), n)

    # Pre-load all sampled images for batched detection
    valid_rows = []
    batch_images = []
    for idx in sample_indices:
        row = trace_df.iloc[idx]
        img_path = explore_dir / row["image_path"]
        if not img_path.exists():
            continue
        img_np = np.array(Image.open(str(img_path)))
        valid_rows.append(row)
        batch_images.append(img_np)

    # Batched OWL-ViT detection
    batch_detections = run_owlvit_detection_batch(
        owl_vit_model, batch_images, query, OWL_VIT_THRESHOLD,
        batch_size=owl_batch_size,
    )

    best_point = None
    best_score = -1
    num_detections = 0

    for row, detections in zip(valid_rows, batch_detections):
        if not detections:
            continue

        depth_path = explore_dir / row["depth_path"]
        if not depth_path.exists():
            continue
        depth = np.load(str(depth_path))

        agent_pos = [row["x"], row["y"], row["z"]]
        quat = [row["qw"], row["qx"], row["qy"], row["qz"]]

        # Take only the single best detection per frame (matching HM3D)
        best_det = max(detections, key=lambda d: d["score"])
        num_detections += 1

        if best_det["score"] > best_score:
            cx_det, cy_det = best_det["center"]
            p3d = projector.project_detection_to_3d(
                cx_det, cy_det, depth, agent_pos, quat,
                depth_percentile=DEPTH_PERCENTILE)
            if p3d is not None:
                best_point = p3d.tolist()
                best_score = best_det["score"]

    latency = (time.time() - t0) * 1000
    return best_point, num_detections, 0, latency, None


# ============================================================================
# Method 4: L1+OWL+Depth (no DBSCAN) — ablation
# ============================================================================

def run_bf_dbscan(query, scene_dir, projector, trace_df,
                  owl_vit_model=None, max_frames=100,
                  owl_batch_size=8):
    """
    Brute Force + Depth + DBSCAN: OWL-ViT (batched) on random frames, keep ALL
    detections, project to 3D, then cluster with DBSCAN.

    This is the BF+DBSCAN ablation: same random frame selection as BF, but
    instead of picking just the single best detection, we keep all detections
    and cluster them. This tests whether DBSCAN helps BF independently of
    CLIP retrieval.
    """
    from sklearn.cluster import DBSCAN
    explore_dir = scene_dir / "exploration"
    t0 = time.time()

    n = min(max_frames, len(trace_df))
    sample_indices = random.sample(range(len(trace_df)), n)

    # Pre-load all sampled images for batched detection
    valid_rows = []
    batch_images = []
    for idx in sample_indices:
        row = trace_df.iloc[idx]
        img_path = explore_dir / row["image_path"]
        if not img_path.exists():
            continue
        img_np = np.array(Image.open(str(img_path)))
        valid_rows.append(row)
        batch_images.append(img_np)

    # Batched OWL-ViT detection
    batch_detections = run_owlvit_detection_batch(
        owl_vit_model, batch_images, query, OWL_VIT_THRESHOLD,
        batch_size=owl_batch_size,
    )

    all_points, all_confs = [], []
    num_detections = 0

    for row, detections in zip(valid_rows, batch_detections):
        if not detections:
            continue

        depth_path = explore_dir / row["depth_path"]
        if not depth_path.exists():
            continue
        depth = np.load(str(depth_path))

        agent_pos = [row["x"], row["y"], row["z"]]
        quat = [row["qw"], row["qx"], row["qy"], row["qz"]]

        for det in detections:
            cx_det, cy_det = det["center"]
            p3d = projector.project_detection_to_3d(
                cx_det, cy_det, depth, agent_pos, quat,
                depth_percentile=DEPTH_PERCENTILE)
            if p3d is not None:
                all_points.append(p3d)
                all_confs.append(det["score"])
                num_detections += 1

    if len(all_points) == 0:
        return None, 0, 0, (time.time() - t0) * 1000, None

    points = np.array(all_points)
    confs = np.array(all_confs)

    clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(points)
    labels = clustering.labels_
    unique_labels = set(labels) - {-1}
    num_clusters = len(unique_labels)

    if num_clusters == 0:
        best_idx = np.argmax(confs)
        ranked = [{"centroid": points[best_idx].tolist(), "score": float(confs[best_idx]), "n_members": 1}]
        return points[best_idx].tolist(), num_detections, 0, (time.time() - t0) * 1000, ranked

    # Build ranked cluster list (sorted by total confidence descending)
    cluster_list = []
    for label in unique_labels:
        mask = labels == label
        cluster_score = float(confs[mask].sum())
        centroid = points[mask].mean(axis=0)
        cluster_list.append({
            "centroid": centroid.tolist(),
            "score": cluster_score,
            "n_members": int(mask.sum()),
        })
    cluster_list.sort(key=lambda c: c["score"], reverse=True)

    best_centroid = cluster_list[0]["centroid"]
    return best_centroid, num_detections, num_clusters, (time.time() - t0) * 1000, cluster_list


def run_jit_no_dbscan(query, scene_dir, projector, trace_df,
                      owl_vit_model=None, clip_encoder=None, faiss_indexer=None,
                      owl_batch_size=8):
    """L1+OWL+Depth without DBSCAN: CLIP retrieval -> OWL-ViT (batched) -> single
    best detection -> depth.

    This is the no-DBSCAN ablation: same as JIT L1+L2 but picks only the
    single highest-scoring detection across all retrieved frames, without
    clustering. Validates DBSCAN's contribution.
    """
    explore_dir = scene_dir / "exploration"
    t0 = time.time()

    # L1: CLIP retrieval
    text_emb = clip_encoder.encode_text(query)
    candidate_ids, similarities = faiss_indexer.search(text_emb, k=min(TOP_K, len(trace_df)))
    candidate_ids = [int(fid) for fid in candidate_ids]

    # Pre-load all candidate images for batched detection
    valid_rows = []
    batch_images = []
    for fid in candidate_ids:
        row = trace_df[trace_df["frame_id"] == fid]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        img_path = explore_dir / row["image_path"]
        if not img_path.exists():
            continue
        img_np = np.array(Image.open(str(img_path)))
        valid_rows.append(row)
        batch_images.append(img_np)

    # Batched OWL-ViT detection
    batch_detections = run_owlvit_detection_batch(
        owl_vit_model, batch_images, query, OWL_VIT_THRESHOLD,
        batch_size=owl_batch_size,
    )

    best_point = None
    best_score = -1
    num_detections = 0

    for row, detections in zip(valid_rows, batch_detections):
        if not detections:
            continue

        depth_path = explore_dir / row["depth_path"]
        if not depth_path.exists():
            continue
        depth = np.load(str(depth_path))

        agent_pos = [row["x"], row["y"], row["z"]]
        quat = [row["qw"], row["qx"], row["qy"], row["qz"]]

        # Take only the single best detection per frame
        best_det = max(detections, key=lambda d: d["score"])
        num_detections += 1

        if best_det["score"] > best_score:
            cx_det, cy_det = best_det["center"]
            p3d = projector.project_detection_to_3d(
                cx_det, cy_det, depth, agent_pos, quat,
                depth_percentile=DEPTH_PERCENTILE)
            if p3d is not None:
                best_point = p3d.tolist()
                best_score = best_det["score"]

    latency = (time.time() - t0) * 1000
    return best_point, num_detections, 0, latency, None


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(all_results, thresholds=LOCALIZATION_THRESHOLDS):
    by_scene = defaultdict(list)
    for r in all_results:
        by_scene[r.scene_id].append(r)

    scene_accuracies = {t: {} for t in thresholds}
    for scene_id, results in by_scene.items():
        for t in thresholds:
            correct = sum(1 for r in results
                          if r.min_distance is not None and r.min_distance < t)
            total = len(results)
            scene_accuracies[t][scene_id] = correct / total if total > 0 else 0.0

    macro = {}
    for t in thresholds:
        vals = list(scene_accuracies[t].values())
        macro[t] = np.mean(vals) if vals else 0.0

    micro = {}
    for t in thresholds:
        correct = sum(1 for r in all_results
                      if r.min_distance is not None and r.min_distance < t)
        micro[t] = correct / len(all_results) if all_results else 0.0

    return macro, micro, scene_accuracies


def bootstrap_ci(all_results, threshold,
                 n_boot=BOOTSTRAP_ITERATIONS, alpha=1-CONFIDENCE_LEVEL):
    by_scene = defaultdict(list)
    for r in all_results:
        by_scene[r.scene_id].append(r)

    scene_accs = []
    for scene_id, results in by_scene.items():
        correct = sum(1 for r in results
                      if r.min_distance is not None and r.min_distance < threshold)
        total = len(results)
        scene_accs.append(correct / total if total > 0 else 0.0)

    scene_accs = np.array(scene_accs)
    rng = np.random.RandomState(42)
    boot_means = [np.mean(rng.choice(scene_accs, size=len(scene_accs), replace=True))
                  for _ in range(n_boot)]
    boot_means = np.array(boot_means)
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return lo, hi


def mcnemars_test(results_a, results_b, threshold=1.0):
    """
    McNemar's test comparing two methods on same queries.
    Returns (chi2, p_value, n_discordant).
    """
    from scipy.stats import chi2 as chi2_dist

    # Build paired outcomes
    key_fn = lambda r: (r.scene_id, r.query)
    a_dict = {key_fn(r): (r.min_distance is not None and r.min_distance < threshold)
              for r in results_a}
    b_dict = {key_fn(r): (r.min_distance is not None and r.min_distance < threshold)
              for r in results_b}

    common_keys = set(a_dict.keys()) & set(b_dict.keys())
    # b = A correct, B wrong; c = A wrong, B correct
    b_count = sum(1 for k in common_keys if a_dict[k] and not b_dict[k])
    c_count = sum(1 for k in common_keys if not a_dict[k] and b_dict[k])

    n_disc = b_count + c_count
    if n_disc == 0:
        return 0.0, 1.0, 0

    # McNemar's chi-squared (with continuity correction)
    chi2_val = (abs(b_count - c_count) - 1) ** 2 / (b_count + c_count)
    p_val = 1 - chi2_dist.cdf(chi2_val, df=1)
    return chi2_val, p_val, n_disc


# ============================================================================
# Scene evaluation
# ============================================================================

METHOD_NAMES = {
    "jit": "JIT + Depth",
    "jit_l3": "JIT + Depth + L3",
    "jit_no_dbscan": "L1+OWL+Depth (no DBSCAN)",
    "bf": "Brute Force + Depth",
    "bf_dbscan": "BF + Depth + DBSCAN",
    "bf_all": "BF-ALL (all frames)",
}


def evaluate_scene(scene_id, methods, shared_models=None, data_dir=None):
    data_dir = data_dir or SCANNET_JIT
    scene_dir = data_dir / scene_id
    explore_dir = scene_dir / "exploration"

    trace_path = explore_dir / "trace.parquet"
    gt_path = scene_dir / f"{scene_id}_ground_truth.json"
    intrinsics_path = scene_dir / "intrinsics.json"
    index_path = explore_dir / "memory.index"

    for p in [trace_path, gt_path, intrinsics_path]:
        if not p.exists():
            print(f"  Missing: {p}")
            return []

    trace_df = pd.read_parquet(str(trace_path))
    gt = load_scene_gt(scene_dir)
    projector = ScanNetProjector(str(intrinsics_path))

    # Load models
    clip_encoder = None
    faiss_indexer = None
    if any(m in methods for m in ["jit", "jit_l3", "jit_no_dbscan"]):
        if not index_path.exists():
            print(f"  WARNING: FAISS index not found, skipping JIT methods")
            methods = [m for m in methods if m not in ("jit", "jit_l3", "jit_no_dbscan")]
        else:
            if shared_models and "clip" in shared_models:
                clip_encoder = shared_models["clip"]
            else:
                from ingestion.clip_encoder import CLIPEncoder
                clip_encoder = CLIPEncoder()
            from ingestion.faiss_indexer import FAISSIndexer
            faiss_indexer = FAISSIndexer()
            faiss_indexer.load(str(explore_dir / "memory"))

    owl_model = load_owlvit()

    # Get valid queries
    valid_queries = []
    for query in JIT_QUERIES_10:
        centers = get_gt_centers_for_query(gt, query)
        if len(centers) > 0:
            valid_queries.append((query, centers))

    print(f"  {len(valid_queries)} queries with GT, {len(trace_df)} keyframes")

    results = []
    for query, gt_centers in valid_queries:
        for method in methods:
            try:
                ranked = None
                if method == "jit":
                    pred, n_det, n_cls, lat, ranked = run_jit_depth(
                        query, scene_dir, projector, trace_df,
                        owl_model, clip_encoder, faiss_indexer)
                elif method == "jit_l3":
                    pred, n_det, n_cls, lat, ranked = run_jit_depth_l3(
                        query, scene_dir, projector, trace_df,
                        owl_model, clip_encoder, faiss_indexer)
                elif method == "jit_no_dbscan":
                    pred, n_det, n_cls, lat, ranked = run_jit_no_dbscan(
                        query, scene_dir, projector, trace_df,
                        owl_model, clip_encoder, faiss_indexer)
                elif method == "bf":
                    pred, n_det, n_cls, lat, ranked = run_bruteforce_depth(
                        query, scene_dir, projector, trace_df, owl_model)
                elif method == "bf_dbscan":
                    pred, n_det, n_cls, lat, ranked = run_bf_dbscan(
                        query, scene_dir, projector, trace_df, owl_model)
                elif method == "bf_all":
                    pred, n_det, n_cls, lat, ranked = run_bruteforce_depth(
                        query, scene_dir, projector, trace_df, owl_model,
                        max_frames=len(trace_df))
                else:
                    continue

                min_dist = None
                if pred is not None:
                    pred_np = np.array(pred)
                    dists = [np.linalg.norm(pred_np - gc) for gc in gt_centers]
                    min_dist = min(dists)

                correct_at = {}
                for t in LOCALIZATION_THRESHOLDS:
                    correct_at[str(t)] = (min_dist is not None and min_dist < t)

                # Compute Recall@K if ranked centroids available
                recall_at_k = None
                if ranked is not None and len(ranked) > 0:
                    recall_at_k = {}
                    for t in LOCALIZATION_THRESHOLDS:
                        for k_val in [1, 3, 5]:
                            hit = False
                            for rc in ranked[:k_val]:
                                rc_np = np.array(rc["centroid"])
                                for gc in gt_centers:
                                    if np.linalg.norm(rc_np - gc) < t:
                                        hit = True
                                        break
                                if hit:
                                    break
                            recall_at_k[f"recall@{k_val}_{t}m"] = hit

                qr = QueryResult(
                    scene_id=scene_id, query=query, method=method,
                    predicted_location=pred,
                    gt_locations=[c.tolist() for c in gt_centers],
                    min_distance=min_dist, correct_at=correct_at,
                    latency_ms=lat, num_detections=n_det, num_clusters=n_cls,
                    ranked_centroids=ranked, recall_at_k=recall_at_k)
                results.append(qr)

                status = f"[OK] {min_dist:.2f}m" if min_dist is not None else "[FAIL] no pred"
                print(f"    [{method:6s}] {query}: {status} ({lat:.0f}ms)")

            except Exception as e:
                print(f"    [{method:6s}] {query}: ERROR - {e}")
                import traceback
                traceback.print_exc()

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ScanNet JIT evaluation v2")
    parser.add_argument("--scenes", type=str, default=None)
    parser.add_argument("--methods", type=str, default="jit,jit_l3,bf",
                        help="Comma-separated: jit, jit_l3, jit_no_dbscan, bf")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename (default: scannet_eval_v2_results.json)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override data directory (default: SCANNET_JIT)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir) if args.data_dir else SCANNET_JIT

    scenes = SCANNET_VAL_SCENES
    if args.scenes:
        scenes = [s.strip() for s in args.scenes.split(",")]

    methods = [m.strip() for m in args.methods.split(",")]
    os.makedirs(str(SCANNET_RESULTS), exist_ok=True)

    print(f"ScanNet JIT Evaluation v2")
    print(f"Scenes: {len(scenes)}, Methods: {methods}")
    print(f"Queries: {JIT_QUERIES_10}")
    print(f"BF protocol: single best detection (no DBSCAN) — matches HM3D")
    print(f"L3: cluster verification + re-projection")
    print()

    all_results = []
    for i, scene_id in enumerate(scenes):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(scenes)}] {scene_id}")
        print(f"{'='*60}")
        scene_results = evaluate_scene(scene_id, methods, data_dir=data_dir)
        all_results.extend(scene_results)

    # Results summary
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")

    method_results_dict = {}
    for method in methods:
        method_results = [r for r in all_results if r.method == method]
        if not method_results:
            continue
        method_results_dict[method] = method_results

        macro, micro, _ = compute_metrics(method_results)
        print(f"\n--- {METHOD_NAMES.get(method, method)} ({len(method_results)} queries) ---")
        for t in LOCALIZATION_THRESHOLDS:
            lo, hi = bootstrap_ci(method_results, t)
            print(f"  Loc@{t}m: {macro[t]*100:.1f}% [{lo*100:.1f}, {hi*100:.1f}] (macro)")

        lats = [r.latency_ms for r in method_results if r.latency_ms > 0]
        if lats:
            print(f"  Avg latency: {np.mean(lats):.0f}ms")
        no_pred = sum(1 for r in method_results if r.predicted_location is None)
        print(f"  No-answer: {no_pred}/{len(method_results)}")

    # McNemar's tests
    print(f"\n{'='*60}")
    print("McNEMAR'S TESTS (Loc@1m)")
    print(f"{'='*60}")

    test_pairs = [("jit", "bf"), ("jit_l3", "bf"), ("jit_l3", "jit"),
                  ("jit", "jit_no_dbscan"), ("jit_no_dbscan", "bf")]
    for m_a, m_b in test_pairs:
        if m_a in method_results_dict and m_b in method_results_dict:
            chi2, p_val, n_disc = mcnemars_test(
                method_results_dict[m_a], method_results_dict[m_b], threshold=1.0)
            a_name = METHOD_NAMES.get(m_a, m_a)
            b_name = METHOD_NAMES.get(m_b, m_b)
            print(f"  {a_name} vs {b_name}: χ²={chi2:.3f}, p={p_val:.4f}, discordant={n_disc}")

    # Per-category breakdown
    print(f"\n{'='*60}")
    print("PER-CATEGORY BREAKDOWN (Loc@1m)")
    print(f"{'='*60}")

    for method in methods:
        if method not in method_results_dict:
            continue
        mr = method_results_dict[method]
        by_cat = defaultdict(list)
        for r in mr:
            by_cat[r.query].append(r)
        print(f"\n  {METHOD_NAMES.get(method, method)}:")
        for cat in sorted(by_cat.keys()):
            cat_results = by_cat[cat]
            n = len(cat_results)
            correct = sum(1 for r in cat_results
                          if r.min_distance is not None and r.min_distance < 1.0)
            print(f"    {cat:10s}: {correct}/{n} ({100*correct/n:.0f}%)")

    # Recall@K summary
    print(f"\n{'='*60}")
    print("RECALL@K (micro-averaged)")
    print(f"{'='*60}")

    for method in methods:
        if method not in method_results_dict:
            continue
        mr = method_results_dict[method]
        has_recall = any(r.recall_at_k is not None for r in mr)
        if not has_recall:
            continue

        n = len(mr)
        avg_clusters = np.mean([r.num_clusters for r in mr])
        print(f"\n  {METHOD_NAMES.get(method, method)} ({n} queries, avg {avg_clusters:.1f} clusters):")
        for t in LOCALIZATION_THRESHOLDS:
            line = f"    Loc@{t}m:"
            for k in [1, 3, 5]:
                key = f"recall@{k}_{t}m"
                hits = sum(1 for r in mr
                           if r.recall_at_k and r.recall_at_k.get(key, False))
                line += f"  R@{k}={100.0*hits/n:.1f}%"
            print(line)

    # Save results
    results_data = {
        "metadata": {
            "version": "v2",
            "dataset": "scannet",
            "num_scenes": len(scenes),
            "scenes": scenes,
            "methods": methods,
            "queries": JIT_QUERIES_10,
            "thresholds": LOCALIZATION_THRESHOLDS,
            "bf_protocol": "single_best_detection_no_dbscan",
            "jit_l3": "cluster_verification_with_reprojection",
        },
        "per_query": [asdict(r) for r in all_results],
        "summary": {},
        "mcnemars": {},
        "recall_at_k": {},
    }

    for method in methods:
        mr = [r for r in all_results if r.method == method]
        if not mr:
            continue
        macro, micro, _ = compute_metrics(mr)
        cis = {}
        for t in LOCALIZATION_THRESHOLDS:
            lo, hi = bootstrap_ci(mr, t)
            cis[str(t)] = {"lo": lo, "hi": hi}
        lats = [r.latency_ms for r in mr if r.latency_ms > 0]
        results_data["summary"][method] = {
            "name": METHOD_NAMES.get(method, method),
            "macro": {str(t): v for t, v in macro.items()},
            "micro": {str(t): v for t, v in micro.items()},
            "bootstrap_ci": cis,
            "num_queries": len(mr),
            "no_answer": sum(1 for r in mr if r.predicted_location is None),
            "avg_latency_ms": float(np.mean(lats)) if lats else 0,
        }

    for m_a, m_b in test_pairs:
        if m_a in method_results_dict and m_b in method_results_dict:
            chi2, p_val, n_disc = mcnemars_test(
                method_results_dict[m_a], method_results_dict[m_b], threshold=1.0)
            results_data["mcnemars"][f"{m_a}_vs_{m_b}"] = {
                "chi2": chi2, "p_value": p_val, "n_discordant": n_disc}

    # Build Recall@K summary
    for method in methods:
        mr = [r for r in all_results if r.method == method]
        has_recall = any(r.recall_at_k is not None for r in mr)
        if not has_recall or not mr:
            continue
        n = len(mr)
        method_recall = {}
        for t in LOCALIZATION_THRESHOLDS:
            for k in [1, 3, 5]:
                key = f"recall@{k}_{t}m"
                hits = sum(1 for r in mr if r.recall_at_k and r.recall_at_k.get(key, False))
                method_recall[key] = {"hits": hits, "total": n, "pct": round(100.0 * hits / n, 1)}
        method_recall["avg_clusters"] = float(np.mean([r.num_clusters for r in mr]))
        results_data["recall_at_k"][method] = method_recall

    results_path = SCANNET_RESULTS / (args.output or "scannet_eval_v2_results.json")
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
