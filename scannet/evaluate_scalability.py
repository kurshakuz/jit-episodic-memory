#!/usr/bin/env python3
"""
Scalability experiment: evaluate JIT vs BF at different memory bank sizes.

For a 20-scene subset, compare JIT (k=100 CLIP-retrieved) vs BF (100 random)
at memory bank sizes of 160, 500, and ~2500 frames per scene.

At 160 frames, JIT retrieves 62.5% of the bank — nearly all frames.
At 500 frames, JIT retrieves 20% — much more selective.
At 2500 frames, JIT retrieves 4% — highly selective.
BF always picks 100 random frames regardless of bank size.

Usage:
    python scannet/evaluate_scalability.py
"""

import os
import sys
import json
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from scannet.config import PROJECT_ROOT, SCANNET_JIT, SCANNET_RESULTS, JIT_QUERIES_10, LOCALIZATION_THRESHOLDS, OWL_VIT_THRESHOLD, DBSCAN_EPS, DBSCAN_MIN_SAMPLES, DEPTH_PERCENTILE

# 20 scenes used in OWLv2 experiment — evenly spaced from 142 val scenes
SCALE_SCENES = [
    'scene0011_00', 'scene0063_00', 'scene0100_00', 'scene0164_00',
    'scene0217_00', 'scene0256_00', 'scene0314_00', 'scene0351_00',
    'scene0378_00', 'scene0426_00', 'scene0462_00', 'scene0518_00',
    'scene0558_00', 'scene0583_00', 'scene0607_00', 'scene0633_00',
    'scene0653_00', 'scene0670_00', 'scene0693_00', 'scene0704_00',
]

FRAME_COUNTS = [160, 500, 2500]
BF_BUDGET = 100  # BF always inspects 100 random frames
JIT_K = 100      # JIT always retrieves top-100

# Directories for each frame count
def get_data_dir(n_frames):
    if n_frames == 160:
        return SCANNET_JIT
    return PROJECT_ROOT / "scannet" / f"jit_format_{n_frames}"


# ============================================================================
# Reuse evaluate_v2 components
# ============================================================================
from scannet.evaluate_v2 import (
    ScanNetProjector, load_owlvit, run_owlvit_detection,
    load_scene_gt, get_gt_centers_for_query,
)
from sklearn.cluster import DBSCAN


def run_jit_depth_scalable(query, scene_dir, projector, trace_df,
                           owl_vit_model, clip_encoder, faiss_indexer,
                           k=100):
    """JIT + Depth with configurable k."""
    explore_dir = scene_dir / "exploration"
    t0 = time.time()

    text_emb = clip_encoder.encode_text(query)
    candidate_ids, similarities = faiss_indexer.search(
        text_emb, k=min(k, len(trace_df)))
    candidate_ids = [int(fid) for fid in candidate_ids]

    all_points, all_confs = [], []
    num_detections = 0

    for fid in candidate_ids:
        row = trace_df[trace_df["frame_id"] == fid]
        if len(row) == 0:
            continue
        row = row.iloc[0]

        img_path = explore_dir / row["image_path"]
        if not img_path.exists():
            continue
        img_np = np.array(Image.open(str(img_path)))

        detections = run_owlvit_detection(owl_vit_model, img_np, query,
                                          OWL_VIT_THRESHOLD)
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
        return None, 0, 0, (time.time() - t0) * 1000

    points = np.array(all_points)
    confs = np.array(all_confs)

    clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(points)
    labels = clustering.labels_
    unique_labels = set(labels) - {-1}

    if len(unique_labels) == 0:
        best_idx = np.argmax(confs)
        return points[best_idx].tolist(), num_detections, 0, (time.time() - t0) * 1000

    cluster_list = []
    for label in unique_labels:
        mask = labels == label
        cluster_score = float(confs[mask].sum())
        centroid = points[mask].mean(axis=0)
        cluster_list.append({"centroid": centroid.tolist(), "score": cluster_score})
    cluster_list.sort(key=lambda c: c["score"], reverse=True)

    return cluster_list[0]["centroid"], num_detections, len(unique_labels), (time.time() - t0) * 1000


def run_bf_depth_scalable(query, scene_dir, projector, trace_df,
                          owl_vit_model, max_frames=100):
    """BF + Depth: random sample of max_frames, single best detection."""
    explore_dir = scene_dir / "exploration"
    t0 = time.time()

    n = min(max_frames, len(trace_df))
    sample_indices = random.sample(range(len(trace_df)), n)

    best_point = None
    best_score = -1
    num_detections = 0

    for idx in sample_indices:
        row = trace_df.iloc[idx]

        img_path = explore_dir / row["image_path"]
        if not img_path.exists():
            continue
        img_np = np.array(Image.open(str(img_path)))

        detections = run_owlvit_detection(owl_vit_model, img_np, query,
                                          OWL_VIT_THRESHOLD)
        if not detections:
            continue

        depth_path = explore_dir / row["depth_path"]
        if not depth_path.exists():
            continue
        depth = np.load(str(depth_path))

        agent_pos = [row["x"], row["y"], row["z"]]
        quat = [row["qw"], row["qx"], row["qy"], row["qz"]]

        for det in detections:
            if det["score"] > best_score:
                cx_det, cy_det = det["center"]
                p3d = projector.project_detection_to_3d(
                    cx_det, cy_det, depth, agent_pos, quat,
                    depth_percentile=DEPTH_PERCENTILE)
                if p3d is not None:
                    best_point = p3d.tolist()
                    best_score = det["score"]
                    num_detections += 1

    elapsed = (time.time() - t0) * 1000
    return best_point, num_detections, 0, elapsed


def run_bf_all_depth(query, scene_dir, projector, trace_df, owl_vit_model):
    """BF-ALL: run OWL-ViT on ALL frames, single best detection."""
    explore_dir = scene_dir / "exploration"
    t0 = time.time()

    best_point = None
    best_score = -1
    num_detections = 0

    for idx in range(len(trace_df)):
        row = trace_df.iloc[idx]

        img_path = explore_dir / row["image_path"]
        if not img_path.exists():
            continue
        img_np = np.array(Image.open(str(img_path)))

        detections = run_owlvit_detection(owl_vit_model, img_np, query,
                                          OWL_VIT_THRESHOLD)
        if not detections:
            continue

        depth_path = explore_dir / row["depth_path"]
        if not depth_path.exists():
            continue
        depth = np.load(str(depth_path))

        agent_pos = [row["x"], row["y"], row["z"]]
        quat = [row["qw"], row["qx"], row["qy"], row["qz"]]

        for det in detections:
            if det["score"] > best_score:
                cx_det, cy_det = det["center"]
                p3d = projector.project_detection_to_3d(
                    cx_det, cy_det, depth, agent_pos, quat,
                    depth_percentile=DEPTH_PERCENTILE)
                if p3d is not None:
                    best_point = p3d.tolist()
                    best_score = det["score"]
                    num_detections += 1

    elapsed = (time.time() - t0) * 1000
    return best_point, num_detections, 0, elapsed


def evaluate_scene(scene_id, n_frames, owl_vit_model, clip_encoder,
                   skip_bf_all=False):
    """Evaluate all methods on a scene at a given frame count."""
    data_dir = get_data_dir(n_frames)
    scene_dir = data_dir / scene_id
    explore_dir = scene_dir / "exploration"

    # Check if scene exists at this frame count
    trace_path = explore_dir / "trace.parquet"
    if not trace_path.exists():
        print(f"  SKIP {scene_id} at {n_frames} frames (not prepared)")
        return []

    trace_df = pd.read_parquet(str(trace_path))
    actual_frames = len(trace_df)
    print(f"  {scene_id}: {actual_frames} frames (target={n_frames})")

    # Load projector (intrinsics from original or scaled dir)
    intrinsics_path = scene_dir / "intrinsics.json"
    if not intrinsics_path.exists():
        # Fallback to original
        intrinsics_path = SCANNET_JIT / scene_id / "intrinsics.json"
    projector = ScanNetProjector(str(intrinsics_path))

    # Load GT from original jit_format
    gt_dir = SCANNET_JIT / scene_id
    gt = load_scene_gt(gt_dir)
    if gt is None:
        print(f"  SKIP {scene_id}: no GT")
        return []

    # Load FAISS index
    from ingestion.faiss_indexer import FAISSIndexer
    faiss_indexer = FAISSIndexer(index_type="flat")
    index_path = explore_dir / "memory"
    if not Path(f"{index_path}.index").exists():
        print(f"  SKIP {scene_id} at {n_frames}: no FAISS index")
        return []
    faiss_indexer.load(str(index_path))

    results = []
    for query in JIT_QUERIES_10:
        gt_centers = get_gt_centers_for_query(gt, query)
        if not gt_centers:
            continue

        # --- JIT (k=100) ---
        pred_jit, ndet_jit, ncl_jit, lat_jit = run_jit_depth_scalable(
            query, scene_dir, projector, trace_df,
            owl_vit_model, clip_encoder, faiss_indexer, k=JIT_K)

        # --- BF (100 random) ---
        pred_bf, ndet_bf, _, lat_bf = run_bf_depth_scalable(
            query, scene_dir, projector, trace_df,
            owl_vit_model, max_frames=BF_BUDGET)

        # --- BF-ALL (all frames, only for non-160) ---
        if skip_bf_all:
            pred_bfall, ndet_bfall, lat_bfall = None, 0, 0.0
        elif actual_frames > 160:
            pred_bfall, ndet_bfall, _, lat_bfall = run_bf_all_depth(
                query, scene_dir, projector, trace_df, owl_vit_model)
        else:
            pred_bfall, ndet_bfall, lat_bfall = pred_bf, ndet_bf, lat_bf

        # Compute distances
        for method_name, pred, lat in [
            ("jit_k100", pred_jit, lat_jit),
            ("bf_100", pred_bf, lat_bf),
            ("bf_all", pred_bfall, lat_bfall),
        ]:
            min_dist = None
            correct_at = {}
            if pred is not None:
                dists = [np.linalg.norm(np.array(pred) - np.array(gc))
                         for gc in gt_centers]
                min_dist = float(min(dists))
                for t in LOCALIZATION_THRESHOLDS:
                    correct_at[str(t)] = min_dist < t
            else:
                for t in LOCALIZATION_THRESHOLDS:
                    correct_at[str(t)] = False

            results.append({
                "scene_id": scene_id,
                "query": query,
                "method": method_name,
                "n_frames": actual_frames,
                "target_frames": n_frames,
                "min_distance": min_dist,
                "correct_at": correct_at,
                "latency_ms": lat,
            })

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-counts", type=str, default="160,500,2500",
                        help="Comma-separated frame counts to evaluate")
    parser.add_argument("--scenes", type=str, default=None,
                        help="Override scene list (comma-separated)")
    parser.add_argument("--skip-bf-all", action="store_true",
                        help="Skip BF-ALL (very slow at large frame counts)")
    args = parser.parse_args()

    frame_counts = [int(x) for x in args.frame_counts.split(",")]
    scenes = SCALE_SCENES
    if args.scenes:
        scenes = [s.strip() for s in args.scenes.split(",")]

    print(f"Scalability Experiment")
    print(f"  Scenes: {len(scenes)}")
    print(f"  Frame counts: {frame_counts}")
    print(f"  JIT k={JIT_K}, BF budget={BF_BUDGET}")
    print()

    # Load models
    owl_vit_model = load_owlvit()

    from ingestion.clip_encoder import CLIPEncoder
    clip_encoder = CLIPEncoder()

    all_results = []
    for n_frames in frame_counts:
        print(f"\n{'='*60}")
        print(f"Frame count: {n_frames}")
        print(f"{'='*60}")

        data_dir = get_data_dir(n_frames)
        if not data_dir.exists():
            print(f"  Data dir not found: {data_dir}")
            print(f"  Run: python scannet/prepare_scenes.py --target-frames {n_frames} "
                  f"--output-dir {data_dir} --scenes {','.join(scenes)}")
            continue

        for scene_id in scenes:
            try:
                results = evaluate_scene(scene_id, n_frames, owl_vit_model,
                                         clip_encoder,
                                         skip_bf_all=args.skip_bf_all)
                all_results.extend(results)
                # Progress
                if results:
                    jit_ok = sum(1 for r in results if r["method"] == "jit_k100"
                                 and r["correct_at"].get("1.0", False))
                    jit_total = sum(1 for r in results if r["method"] == "jit_k100")
                    bf_ok = sum(1 for r in results if r["method"] == "bf_100"
                                and r["correct_at"].get("1.0", False))
                    bf_total = sum(1 for r in results if r["method"] == "bf_100")
                    print(f"    JIT: {jit_ok}/{jit_total}, BF: {bf_ok}/{bf_total}")
            except Exception as e:
                print(f"  ERROR {scene_id}: {e}")
                import traceback
                traceback.print_exc()

    # Save results
    os.makedirs(str(SCANNET_RESULTS), exist_ok=True)
    out_path = SCANNET_RESULTS / "scalability_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} results to {out_path}")

    # Compute aggregated metrics
    print(f"\n{'='*60}")
    print("SCALABILITY RESULTS (per-scene macro-averaged)")
    print(f"{'='*60}")

    for n_frames in frame_counts:
        frame_results = [r for r in all_results if r["target_frames"] == n_frames]
        if not frame_results:
            continue

        print(f"\n--- {n_frames} frames/scene ---")

        for method in ["jit_k100", "bf_100", "bf_all"]:
            method_results = [r for r in frame_results if r["method"] == method]
            if not method_results:
                continue

            # Per-scene macro average
            scene_metrics = defaultdict(lambda: defaultdict(list))
            for r in method_results:
                for t_str, correct in r["correct_at"].items():
                    scene_metrics[r["scene_id"]][t_str].append(1.0 if correct else 0.0)

            macro_avg = {}
            for t_str in ["0.5", "1.0", "2.0", "3.0"]:
                scene_avgs = []
                for scene_id in scene_metrics:
                    vals = scene_metrics[scene_id].get(t_str, [])
                    if vals:
                        scene_avgs.append(np.mean(vals))
                macro_avg[t_str] = np.mean(scene_avgs) * 100 if scene_avgs else 0.0

            avg_latency = np.mean([r["latency_ms"] for r in method_results])

            selectivity = min(JIT_K if method == "jit_k100" else BF_BUDGET,
                              n_frames) / n_frames * 100
            label = f"{method} ({selectivity:.0f}% of bank)"

            print(f"  {label:35s}: "
                  f"Loc@0.5m={macro_avg['0.5']:.1f}%  "
                  f"Loc@1m={macro_avg['1.0']:.1f}%  "
                  f"Loc@2m={macro_avg['2.0']:.1f}%  "
                  f"Latency={avg_latency:.0f}ms")


if __name__ == "__main__":
    main()
