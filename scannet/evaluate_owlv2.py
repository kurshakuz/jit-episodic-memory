#!/usr/bin/env python3
"""
OWLv2 Ablation: Compare OWLv2 vs OWL-ViT as detection backbone.

Runs JIT cascade with OWLv2 on 20-scene subset of ScanNet,
then compares against existing OWL-ViT results on the same scenes.

OWLv2 (ViT-B/16-ensemble): 219 ms/frame
OWL-ViT (ViT-B/32): 18 ms/frame  (12.1x faster)
"""

import json
import sys
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from collections import defaultdict
from sklearn.cluster import DBSCAN

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from scannet.config import SCANNET_JIT, SCANNET_RESULTS, JIT_QUERIES_10, SCANNET_SYNONYMS, LOCALIZATION_THRESHOLDS, OWL_VIT_THRESHOLD, DBSCAN_EPS, DBSCAN_MIN_SAMPLES, DEPTH_PERCENTILE, TOP_K

RESULTS_DIR = SCANNET_RESULTS
RESULTS_DIR.mkdir(exist_ok=True)
NUM_SUBSET_SCENES = 20

random.seed(42)
np.random.seed(42)

import argparse

# ============================================================================
# Helpers (from evaluate_v2)
# ============================================================================
def load_scene_gt(scene_dir):
    scene_id = scene_dir.name
    gt_path = scene_dir / f"{scene_id}_ground_truth.json"
    if not gt_path.exists():
        return None
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


class ScanNetProjector:
    def __init__(self, intrinsics_path):
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
# OWLv2 detector
# ============================================================================
def load_owlv2():
    import torch
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    print("Loading OWLv2 (google/owlv2-base-patch16-ensemble)...", flush=True)
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"OWLv2 loaded on {device}", flush=True)
    return model, processor


def run_owlv2_detection(model, processor, image_np, query, threshold):
    import torch
    device = next(model.parameters()).device
    h, w = image_np.shape[:2]

    img_pil = Image.fromarray(image_np)
    inputs = processor(text=[[f"a photo of a {query}"]], images=img_pil,
                       return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([[h, w]], device=device)
    # OWLv2 uses post_process_grounded_object_detection
    results = processor.post_process_grounded_object_detection(
        outputs=outputs, threshold=threshold, target_sizes=target_sizes)

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
            })
    return detections


# ============================================================================
# JIT pipeline with OWLv2
# ============================================================================
def run_jit_owlv2(query, scene_dir, projector, trace_df,
                  owlv2_model, owlv2_proc, clip_encoder, faiss_indexer,
                  l3_max_verify=5):
    """Full JIT cascade (L1->L2->L3) using OWLv2 as detector."""
    explore_dir = scene_dir / "exploration"
    t0 = time.time()

    # L1: CLIP retrieval
    text_emb = clip_encoder.encode_text(query)
    candidate_ids, similarities = faiss_indexer.search(text_emb, k=min(TOP_K, len(trace_df)))
    candidate_ids = [int(fid) for fid in candidate_ids]

    # L2: OWLv2 detection + depth projection on all candidates
    all_points, all_confs, all_frame_ids = [], [], []
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

        detections = run_owlv2_detection(owlv2_model, owlv2_proc, img_np, query, OWL_VIT_THRESHOLD)
        if not detections:
            continue

        depth_path = explore_dir / row["depth_path"]
        if not depth_path.exists():
            continue
        depth = np.load(str(depth_path))
        agent_pos = [row["x"], row["y"], row["z"]]
        quat = [row["qw"], row["qx"], row["qy"], row["qz"]]

        for det in detections:
            cx_d, cy_d = det["center"]
            p3d = projector.project_detection_to_3d(
                cx_d, cy_d, depth, agent_pos, quat,
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
        return points[best_idx].tolist(), num_detections, 0, (time.time() - t0) * 1000, [
            {"centroid": points[best_idx].tolist(), "score": float(confs[best_idx])}
        ]

    # Rank clusters by cumulative confidence
    cluster_info = []
    for label in unique_labels:
        mask = labels == label
        c_confs = confs[mask]
        c_frames = frame_ids[mask]
        best_frame_idx = np.argmax(c_confs)
        cluster_info.append({
            "label": label,
            "score": float(c_confs.sum()),
            "best_frame_id": int(c_frames[best_frame_idx]),
            "centroid": points[mask].mean(axis=0),
            "n_points": int(mask.sum()),
        })
    cluster_info.sort(key=lambda c: c["score"], reverse=True)

    # L3: Re-verify top clusters
    verified = []
    for ci in cluster_info[:l3_max_verify]:
        fid = ci["best_frame_id"]
        row = trace_df[trace_df["frame_id"] == fid]
        if len(row) == 0:
            continue
        row = row.iloc[0]

        img_path = explore_dir / row["image_path"]
        if not img_path.exists():
            verified.append({"centroid": ci["centroid"], "score": ci["score"]})
            continue
        img_np = np.array(Image.open(str(img_path)))

        dets = run_owlv2_detection(owlv2_model, owlv2_proc, img_np, query, OWL_VIT_THRESHOLD)
        if not dets:
            continue

        best_det = max(dets, key=lambda d: d["score"])
        depth_path = explore_dir / row["depth_path"]
        if not depth_path.exists():
            verified.append({"centroid": ci["centroid"], "score": best_det["score"]})
            continue
        depth = np.load(str(depth_path))
        agent_pos = [row["x"], row["y"], row["z"]]
        quat = [row["qw"], row["qx"], row["qy"], row["qz"]]
        cx_d, cy_d = best_det["center"]
        p3d = projector.project_detection_to_3d(
            cx_d, cy_d, depth, agent_pos, quat,
            depth_percentile=DEPTH_PERCENTILE)
        if p3d is not None:
            verified.append({"centroid": p3d, "score": best_det["score"]})
        else:
            verified.append({"centroid": ci["centroid"], "score": best_det["score"]})

    if not verified:
        bc = cluster_info[0]["centroid"]
        if isinstance(bc, np.ndarray):
            bc = bc.tolist()
        ranked = [{"centroid": bc, "score": cluster_info[0]["score"]}]
        return bc, num_detections, num_clusters, (time.time() - t0) * 1000, ranked

    verified.sort(key=lambda x: x["score"], reverse=True)
    ranked = []
    for v in verified:
        c = v["centroid"]
        if isinstance(c, np.ndarray):
            c = c.tolist()
        ranked.append({"centroid": c, "score": float(v["score"])})

    return ranked[0]["centroid"], num_detections, num_clusters, (time.time() - t0) * 1000, ranked


# ============================================================================
# BF+Depth with OWLv2
# ============================================================================
def run_bf_owlv2(query, scene_dir, projector, trace_df,
                 owlv2_model, owlv2_proc, max_frames=100):
    """Brute-force + depth using OWLv2 detector."""
    explore_dir = scene_dir / "exploration"
    t0 = time.time()

    n = min(max_frames, len(trace_df))
    sample_indices = random.sample(range(len(trace_df)), n)

    best_point = None
    best_score = -1

    for idx in sample_indices:
        row = trace_df.iloc[idx]
        img_path = explore_dir / row["image_path"]
        if not img_path.exists():
            continue
        img_np = np.array(Image.open(str(img_path)))

        dets = run_owlv2_detection(owlv2_model, owlv2_proc, img_np, query, OWL_VIT_THRESHOLD)
        if not dets:
            continue

        depth_path = explore_dir / row["depth_path"]
        if not depth_path.exists():
            continue
        depth = np.load(str(depth_path))
        agent_pos = [row["x"], row["y"], row["z"]]
        quat = [row["qw"], row["qx"], row["qy"], row["qz"]]

        best_det = max(dets, key=lambda d: d["score"])
        if best_det["score"] > best_score:
            cx_d, cy_d = best_det["center"]
            p3d = projector.project_detection_to_3d(
                cx_d, cy_d, depth, agent_pos, quat,
                depth_percentile=DEPTH_PERCENTILE)
            if p3d is not None:
                best_point = p3d.tolist()
                best_score = best_det["score"]

    latency = (time.time() - t0) * 1000
    return best_point, latency


# ============================================================================
# Main
# ============================================================================
def main():
    import torch

    parser = argparse.ArgumentParser(description="OWLv2 Ablation on ScanNet")
    parser.add_argument("--all-scenes", action="store_true",
                        help="Run on ALL valid scenes instead of 20-scene subset")
    parser.add_argument("--num-scenes", type=int, default=None,
                        help="Run on first N valid scenes")
    args = parser.parse_args()

    mode = "all scenes" if args.all_scenes else f"{NUM_SUBSET_SCENES}-scene subset"
    print("=" * 70, flush=True)
    print(f"OWLv2 Ablation ({mode})", flush=True)
    print("=" * 70, flush=True)

    # Find ALL valid scenes
    all_scene_dirs = sorted([d for d in SCANNET_JIT.iterdir() if d.is_dir()])
    valid_scenes = []
    for sd in all_scene_dirs:
        sid = sd.name
        explore_dir = sd / "exploration"
        if all(p.exists() for p in [
            explore_dir / "trace.parquet",
            sd / f"{sid}_ground_truth.json",
            sd / "intrinsics.json",
            explore_dir / "memory.index",
        ]):
            gt = load_scene_gt(sd)
            if gt:
                has_queries = any(
                    get_gt_centers_for_query(gt, q) for q in JIT_QUERIES_10
                )
                if has_queries:
                    valid_scenes.append(sd)

    print(f"Found {len(valid_scenes)} valid scenes total", flush=True)

    if args.all_scenes:
        subset_scenes = valid_scenes
    elif args.num_scenes:
        subset_scenes = valid_scenes[:args.num_scenes]
    else:
        # Select 20 evenly-spaced scenes (original behavior)
        indices = np.linspace(0, len(valid_scenes) - 1, NUM_SUBSET_SCENES, dtype=int)
        subset_scenes = [valid_scenes[i] for i in indices]

    subset_ids = [s.name for s in subset_scenes]
    print(f"Selected {len(subset_scenes)} scenes: {subset_ids[:5]}...", flush=True)

    # Load models
    owlv2_model, owlv2_proc = load_owlv2()

    print("Loading CLIP encoder...", flush=True)
    from ingestion.clip_encoder import CLIPEncoder
    clip_encoder = CLIPEncoder()
    clip_encoder.load()
    print("CLIP loaded", flush=True)

    # Load existing OWL-ViT results for comparison
    owlvit_results = {}
    agg_path = RESULTS_DIR / "scannet_aggregated_141scenes.json"
    if agg_path.exists():
        with open(agg_path) as f:
            agg = json.load(f)
        for r in agg["per_query"]:
            key = (r["scene_id"], r["query"], r["method"])
            owlvit_results[key] = r
        print(f"Loaded {len(owlvit_results)} existing OWL-ViT results", flush=True)

    # Run evaluation
    results_bf = []
    results_jit = []
    total_queries = 0

    for si, scene_dir in enumerate(subset_scenes):
        scene_id = scene_dir.name
        explore_dir = scene_dir / "exploration"

        trace_df = pd.read_parquet(str(explore_dir / "trace.parquet"))
        gt = load_scene_gt(scene_dir)
        projector = ScanNetProjector(str(scene_dir / "intrinsics.json"))

        from ingestion.faiss_indexer import FAISSIndexer
        faiss_indexer = FAISSIndexer()
        faiss_indexer.load(str(explore_dir / "memory"))

        valid_queries = []
        for query in JIT_QUERIES_10:
            centers = get_gt_centers_for_query(gt, query)
            if len(centers) > 0:
                valid_queries.append((query, [c.tolist() for c in centers]))

        if not valid_queries:
            continue

        scene_t0 = time.time()
        print(f"\n[{si+1}/{len(subset_scenes)}] {scene_id}: {len(valid_queries)} queries, {len(trace_df)} frames",
              flush=True)

        for qi, (query, gt_locs) in enumerate(valid_queries):
            total_queries += 1
            q_t0 = time.time()

            # --- BF+Depth (OWLv2) ---
            random.seed(42 + si)
            pred_bf, lat_bf = run_bf_owlv2(
                query, scene_dir, projector, trace_df,
                owlv2_model, owlv2_proc)

            if pred_bf is not None:
                min_dist_bf = min(np.linalg.norm(np.array(pred_bf) - np.array(g)) for g in gt_locs)
            else:
                min_dist_bf = float("inf")

            correct_bf = {f"loc_{t}m": bool(min_dist_bf <= t) for t in LOCALIZATION_THRESHOLDS}
            results_bf.append({
                "scene_id": scene_id, "query": query, "method": "bf_owlv2",
                "predicted_location": pred_bf, "gt_locations": gt_locs,
                "min_distance": min_dist_bf, "correct_at": correct_bf,
                "latency_ms": lat_bf,
            })

            # --- JIT full cascade (OWLv2) ---
            pred_jit, nd_jit, nc_jit, lat_jit, ranked_jit = run_jit_owlv2(
                query, scene_dir, projector, trace_df,
                owlv2_model, owlv2_proc, clip_encoder, faiss_indexer)

            if pred_jit is not None:
                min_dist_jit = min(np.linalg.norm(np.array(pred_jit) - np.array(g)) for g in gt_locs)
            else:
                min_dist_jit = float("inf")

            correct_jit = {f"loc_{t}m": bool(min_dist_jit <= t) for t in LOCALIZATION_THRESHOLDS}

            # Recall@K
            recall_at_k = {}
            if ranked_jit:
                for k_val in [1, 3, 5]:
                    top_k = ranked_jit[:k_val]
                    hit = any(
                        np.linalg.norm(np.array(rc["centroid"]) - np.array(g)) <= 1.0
                        for rc in top_k for g in gt_locs
                    )
                    recall_at_k[f"recall@{k_val}_1m"] = hit
            else:
                for k_val in [1, 3, 5]:
                    recall_at_k[f"recall@{k_val}_1m"] = correct_jit.get("loc_1.0m", False)

            results_jit.append({
                "scene_id": scene_id, "query": query, "method": "jit_owlv2",
                "predicted_location": pred_jit, "gt_locations": gt_locs,
                "min_distance": min_dist_jit, "correct_at": correct_jit,
                "latency_ms": lat_jit, "num_detections": nd_jit,
                "num_clusters": nc_jit, "recall_at_k": recall_at_k,
            })

            q_time = time.time() - q_t0
            bf_ok = "Y" if correct_bf.get("loc_1.0m", False) else "N"
            jit_ok = "Y" if correct_jit.get("loc_1.0m", False) else "N"
            print(f"  [{qi+1}/{len(valid_queries)}] '{query}': "
                  f"BF={bf_ok} ({min_dist_bf:.2f}m, {lat_bf:.0f}ms) | "
                  f"JIT={jit_ok} ({min_dist_jit:.2f}m, {lat_jit:.0f}ms) | "
                  f"{q_time:.1f}s", flush=True)

        scene_time = time.time() - scene_t0
        print(f"  Scene done in {scene_time:.0f}s | cumulative: {len(results_bf)} BF, {len(results_jit)} JIT queries",
              flush=True)

    # Save results
    suffix = f"_{len(subset_scenes)}scenes" if len(subset_scenes) != NUM_SUBSET_SCENES else ""
    bf_path = RESULTS_DIR / f"scannet_bf_owlv2_results{suffix}.json"
    jit_path = RESULTS_DIR / f"scannet_jit_owlv2_results{suffix}.json"

    with open(bf_path, "w") as f:
        json.dump(results_bf, f, indent=2, default=str)
    with open(jit_path, "w") as f:
        json.dump(results_jit, f, indent=2, default=str)

    print(f"\nSaved: {bf_path} ({len(results_bf)} queries)", flush=True)
    print(f"Saved: {jit_path} ({len(results_jit)} queries)", flush=True)

    # ====================================================================
    # Compute metrics and compare with OWL-ViT
    # ====================================================================
    print("\n" + "=" * 70, flush=True)
    print(f"RESULTS ({len(subset_scenes)} scenes, {total_queries} queries)", flush=True)
    print("=" * 70, flush=True)

    def compute_metrics(results, label):
        scene_metrics = defaultdict(lambda: defaultdict(list))
        for r in results:
            for k, v in r["correct_at"].items():
                scene_metrics[r["scene_id"]][k].append(v)

        macro = {}
        for k in sorted(set(k for m in scene_metrics.values() for k in m)):
            vals = [np.mean(scene_metrics[s][k]) for s in scene_metrics]
            macro[k] = np.mean(vals)

        micro = defaultdict(list)
        for r in results:
            for k, v in r["correct_at"].items():
                micro[k].append(v)
        micro_avg = {k: np.mean(v) for k, v in micro.items()}

        lats = [r["latency_ms"] for r in results]

        print(f"\n{label} ({len(results)} queries, {len(scene_metrics)} scenes):", flush=True)
        print(f"  Macro: " + " | ".join(f"{k}={macro[k]*100:.1f}%" for k in sorted(macro)), flush=True)
        print(f"  Micro: " + " | ".join(f"{k}={micro_avg[k]*100:.1f}%" for k in sorted(micro_avg)), flush=True)
        print(f"  Latency: {np.mean(lats):.0f} ms (median {np.median(lats):.0f} ms)", flush=True)
        return macro, micro_avg

    # OWLv2 results
    bf_macro, bf_micro = compute_metrics(results_bf, "BF+Depth (OWLv2)")
    jit_macro, jit_micro = compute_metrics(results_jit, "JIT full (OWLv2)")

    # JIT Recall@K
    if results_jit:
        scene_recall = defaultdict(lambda: defaultdict(list))
        for r in results_jit:
            if r.get("recall_at_k"):
                for k, v in r["recall_at_k"].items():
                    scene_recall[r["scene_id"]][k].append(v)
        if scene_recall:
            print(f"\n  JIT+OWLv2 Recall@K (macro):", flush=True)
            for k in sorted(list(scene_recall.values())[0].keys()):
                vals = [np.mean(scene_recall[s][k]) for s in scene_recall]
                print(f"    {k}: {np.mean(vals)*100:.1f}%", flush=True)

    # Compare with existing OWL-ViT on same scenes
    if owlvit_results:
        print("\n" + "-" * 70, flush=True)
        print("COMPARISON: OWL-ViT vs OWLv2 (same 20 scenes)", flush=True)
        print("-" * 70, flush=True)

        # Filter OWL-ViT results to our 20 scenes
        owlvit_bf_subset = []
        owlvit_jit_subset = []
        for r in results_bf:
            key_bf = (r["scene_id"], r["query"], "bf")
            if key_bf in owlvit_results:
                owlvit_bf_subset.append(owlvit_results[key_bf])
        for r in results_jit:
            key_jit = (r["scene_id"], r["query"], "jit")
            if key_jit in owlvit_results:
                owlvit_jit_subset.append(owlvit_results[key_jit])

        def compute_owlvit_metrics(results, label):
            if not results:
                print(f"\n  {label}: No matching results found", flush=True)
                return {}, {}
            scene_metrics = defaultdict(lambda: defaultdict(list))
            for r in results:
                ca = r["correct_at"]
                for k, v in ca.items():
                    val = v if isinstance(v, bool) else (v == "True" or v is True)
                    scene_metrics[r["scene_id"]][f"loc_{k}m"].append(val)

            macro = {}
            for k in sorted(set(k for m in scene_metrics.values() for k in m)):
                vals = [np.mean(scene_metrics[s][k]) for s in scene_metrics]
                macro[k] = np.mean(vals)

            micro = defaultdict(list)
            for r in results:
                ca = r["correct_at"]
                for k, v in ca.items():
                    val = v if isinstance(v, bool) else (v == "True" or v is True)
                    micro[f"loc_{k}m"].append(val)
            micro_avg = {k: np.mean(v) for k, v in micro.items()}

            lats = [r.get("latency_ms", 0) for r in results]

            print(f"\n  {label} ({len(results)} queries):", flush=True)
            print(f"    Macro: " + " | ".join(f"{k}={macro[k]*100:.1f}%" for k in sorted(macro)), flush=True)
            print(f"    Micro: " + " | ".join(f"{k}={micro_avg[k]*100:.1f}%" for k in sorted(micro_avg)), flush=True)
            if any(l > 0 for l in lats):
                print(f"    Latency: {np.mean(lats):.0f} ms", flush=True)
            return macro, micro_avg

        compute_owlvit_metrics(owlvit_bf_subset, "BF+Depth (OWL-ViT, same scenes)")
        compute_owlvit_metrics(owlvit_jit_subset, "JIT full (OWL-ViT, same scenes)")

    print("\nDone! OWLv2 ablation complete.", flush=True)


if __name__ == "__main__":
    main()
