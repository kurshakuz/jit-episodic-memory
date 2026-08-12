#!/usr/bin/env python3
"""
FINAL END-TO-END VERIFICATION SCRIPT
=====================================

This script independently re-runs VLMaps, DenseMap, and CG evaluation
on ONE scene (4ok3usBNeis) and compares results against stored per-query files.

It also verifies GT loading and evaluation logic from scratch.
"""

import sys, json, gzip, pickle
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "outputs" / "paper_results"
TEST_SCENE = "4ok3usBNeis"
TEST_QUERIES = ["toilet", "chair", "table", "bed", "couch",
                "sink", "lamp", "mirror", "cabinet", "shelf"]
THRESHOLDS = [0.5, 1.0, 2.0, 3.0]

# 1. GT LOADING — Reproduce EXACTLY

def load_ground_truth_dense(scene_dir):
    """Exact copy of Phase C logic."""
    gt_path = scene_dir / f"{scene_dir.name}_ground_truth.json"
    trace_path = scene_dir / "exploration" / "trace.parquet"

    with open(gt_path) as f:
        gt_data = json.load(f)

    trace = pd.read_parquet(trace_path)
    cam_y_min = trace["y"].min()
    cam_y_max = trace["y"].max()
    floor_y_min = cam_y_min - 1.5
    floor_y_max = cam_y_max + 3.5

    result = {}
    objects = gt_data.get("objects", gt_data)
    for obj_id, obj_info in objects.items():
        if not isinstance(obj_info, dict):
            continue
        category = obj_info.get("category", "").lower()
        center = obj_info.get("center")
        if not category or not center or category == "unknown":
            continue
        obj_y = center[1]
        if not (floor_y_min <= obj_y <= floor_y_max):
            continue
        if category not in result:
            result[category] = []
        result[category].append(np.array(center))
    return result


def load_ground_truth_no_floor_filter(scene_dir):
    """GT loading WITHOUT floor filter (for CG comparison)."""
    gt_path = scene_dir / f"{scene_dir.name}_ground_truth.json"

    with open(gt_path) as f:
        gt_data = json.load(f)

    result = {}
    objects = gt_data.get("objects", gt_data)
    for obj_id, obj_info in objects.items():
        if not isinstance(obj_info, dict):
            continue
        category = obj_info.get("category", "").lower()
        center = obj_info.get("center")
        if not category or not center or category == "unknown":
            continue
        if category not in result:
            result[category] = []
        result[category].append(np.array(center))
    return result


def compute_loc_strict(pred, gt_centers):
    """Strict < threshold."""
    results = {t: False for t in THRESHOLDS}
    if pred is None:
        return results
    for gt_center in gt_centers:
        dist = float(np.linalg.norm(pred - gt_center))
        for t in THRESHOLDS:
            if dist < t:
                results[t] = True
    return results


def evaluate_query(pred_location, gt_centers):
    """Return (error_m, loc_dict)."""
    if pred_location is None:
        return None, {t: False for t in THRESHOLDS}

    min_dist = float("inf")
    for gc in gt_centers:
        d = float(np.linalg.norm(pred_location - gc))
        min_dist = min(min_dist, d)

    loc = compute_loc_strict(pred_location, gt_centers)
    return min_dist, loc


# 2. FIND SCENE DIR

def find_scene_dir():
    base = PROJECT_ROOT / "outputs" / "multi_scene_eval"
    candidate = base / TEST_SCENE
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Cannot find scene dir for {TEST_SCENE} in {base}")


# MAIN VERIFICATION

def main():
    print("=" * 70)
    print("FINAL END-TO-END VERIFICATION")
    print("=" * 70)

    # Find scene
    scene_dir = find_scene_dir()
    print(f"Scene dir: {scene_dir}")
    print(f"Scene ID:  {TEST_SCENE}")

    # Load GT (both with and without floor filter)
    gt_dense = load_ground_truth_dense(scene_dir)
    gt_no_floor = load_ground_truth_no_floor_filter(scene_dir)

    print("\n--- GT comparison (floor-filtered vs unfiltered) ---")
    for query in TEST_QUERIES:
        gt_c_floor = []
        gt_c_no_floor = []
        for cat, centers in gt_dense.items():
            if query in cat:
                gt_c_floor.extend(centers)
        for cat, centers in gt_no_floor.items():
            if query in cat:
                gt_c_no_floor.extend(centers)
        diff = len(gt_c_no_floor) - len(gt_c_floor)
        marker = " ← DIFFERENT" if diff > 0 else ""
        print(f"  {query:10s}: floor_filtered={len(gt_c_floor):3d}, unfiltered={len(gt_c_no_floor):3d}{marker}")

    # Load stored results for comparison
    with open(RESULTS_DIR / "phase_c_vlmaps_perquery.json") as f:
        stored_vlm = {q["query"]: q for q in json.load(f) if q["scene_id"] == TEST_SCENE}
    with open(RESULTS_DIR / "phase_c_densemap_perquery.json") as f:
        stored_dm = {q["query"]: q for q in json.load(f) if q["scene_id"] == TEST_SCENE}
    with open(RESULTS_DIR / "cg_36scenes_consistent_eval.json") as f:
        cg_data = json.load(f)
        stored_cg = {q["category"]: q for q in cg_data["per_query"] if q["scene_id"] == TEST_SCENE}

    # 3. RE-RUN VLMaps from scratch
    print("\n" + "=" * 70)
    print("RE-RUNNING VLMaps on", TEST_SCENE)
    print("=" * 70)

    from baselines.vlmap_baseline import VLMapBaseline
    vlmap = VLMapBaseline(scene_dir, grid_resolution=0.05, grid_size=500, use_lseg=False)
    vlmap.build_map(verbose=True)

    print("\n--- VLMaps per-query comparison ---")
    vlm_ok = True
    for query in TEST_QUERIES:
        gt_centers = []
        for cat, centers in gt_dense.items():
            if query in cat:
                gt_centers.extend(centers)
        if not gt_centers:
            print(f"  {query:10s}: SKIP (no GT)")
            continue

        result = vlmap.query(query)
        error_new, loc_new = evaluate_query(result.predicted_location, gt_centers)

        # Compare with stored
        old = stored_vlm.get(query)
        if old is None:
            print(f"  {query:10s}: NOT IN STORED RESULTS")
            vlm_ok = False
            continue

        error_old = old["error_m"]
        match = True
        if error_new is None and error_old is None:
            pass
        elif error_new is None or error_old is None:
            match = False
        elif abs(error_new - error_old) > 0.001:
            match = False

        status = "[OK]" if match else "[FAIL] MISMATCH"
        print(f"  {query:10s}: new={error_new:.4f}m, stored={error_old:.4f}m  {status}")
        if not match:
            vlm_ok = False
            # Print prediction locations for debugging
            if result.predicted_location is not None:
                print(f"    new_pred: {result.predicted_location}")

    print(f"\nVLMaps verification: {'PASS' if vlm_ok else 'FAIL'}")

    del vlmap
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4. RE-RUN DenseMap from scratch
    print("\n" + "=" * 70)
    print("RE-RUNNING DenseMap on", TEST_SCENE)
    print("=" * 70)

    from baselines.dense_map import DenseMapBaseline
    dmap = DenseMapBaseline(scene_dir, voxel_size=0.05, sample_stride=4)
    dmap.build_map(verbose=True)

    print("\n--- DenseMap per-query comparison ---")
    dm_ok = True
    for query in TEST_QUERIES:
        gt_centers = []
        for cat, centers in gt_dense.items():
            if query in cat:
                gt_centers.extend(centers)
        if not gt_centers:
            print(f"  {query:10s}: SKIP (no GT)")
            continue

        result = dmap.query(query)
        error_new, loc_new = evaluate_query(result.predicted_location, gt_centers)

        old = stored_dm.get(query)
        if old is None:
            print(f"  {query:10s}: NOT IN STORED RESULTS")
            dm_ok = False
            continue

        error_old = old["error_m"]
        match = True
        if error_new is None and error_old is None:
            pass
        elif error_new is None or error_old is None:
            match = False
        elif abs(error_new - error_old) > 0.001:
            match = False

        status = "[OK]" if match else "[FAIL] MISMATCH"
        print(f"  {query:10s}: new={error_new:.4f}m, stored={error_old:.4f}m  {status}")
        if not match:
            dm_ok = False

    print(f"\nDenseMap verification: {'PASS' if dm_ok else 'FAIL'}")

    del dmap
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5. RE-RUN CG evaluation from saved object maps
    print("\n" + "=" * 70)
    print("RE-RUNNING CG evaluation on", TEST_SCENE)
    print("=" * 70)

    import open_clip
    cg_cache_dir = PROJECT_ROOT / "outputs" / "paper_results" / "cg_official_cache"
    scene_cache = None
    for d in cg_cache_dir.iterdir():
        if TEST_SCENE in d.name:
            scene_cache = d
            break

    if scene_cache is None:
        print("ERROR: CG cache not found!")
    else:
        obj_map_path = scene_cache / "object_map.pkl.gz"
        print(f"CG cache: {scene_cache}")
        print(f"Object map: {obj_map_path}")

        with gzip.open(obj_map_path, "rb") as f:
            obj_map = pickle.load(f)

        # How many objects?
        print(f"Number of CG objects: {len(obj_map)}")
        # Inspect first object
        if obj_map:
            first_key = list(obj_map.keys())[0]
            first_obj = obj_map[first_key]
            print(f"Object keys: {list(first_obj.keys()) if isinstance(first_obj, dict) else type(first_obj)}")

        # Load CLIP ViT-H-14
        print("Loading CLIP ViT-H-14...")
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", pretrained="laion2b_s32b_b79k"
        )
        tokenizer = open_clip.get_tokenizer("ViT-H-14")
        model = model.eval()
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        model = model.to(device)

        # GT for CG: NO floor filter (as in recompute_cg_fair.py)
        gt_cg = gt_no_floor

        print("\n--- CG per-query comparison ---")
        cg_ok = True
        for query in TEST_QUERIES:
            gt_centers = []
            for cat, centers in gt_cg.items():
                if query in cat:
                    gt_centers.extend(centers)
            if not gt_centers:
                print(f"  {query:10s}: SKIP (no GT)")
                continue

            # Query CG map with CLIP
            import torch
            text = tokenizer([query]).to(device)
            with torch.no_grad():
                text_feat = model.encode_text(text)
                text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
                text_feat = text_feat.cpu().numpy().flatten()

            best_sim = -1.0
            best_centroid = None
            for obj_id, obj_data in obj_map.items():
                clip_ft = obj_data.get("clip_ft")
                if clip_ft is None:
                    continue
                if isinstance(clip_ft, torch.Tensor):
                    clip_ft = clip_ft.cpu().numpy()
                clip_ft = clip_ft.flatten()
                # Normalize
                norm = np.linalg.norm(clip_ft)
                if norm > 0:
                    clip_ft = clip_ft / norm
                sim = float(np.dot(text_feat, clip_ft))
                if sim > best_sim:
                    best_sim = sim
                    # Centroid of the object
                    if "pcd" in obj_data:
                        pcd = obj_data["pcd"]
                        if hasattr(pcd, 'points'):
                            pts = np.asarray(pcd.points)
                        else:
                            pts = np.array(pcd)
                        best_centroid = pts.mean(axis=0)
                    elif "centroid" in obj_data:
                        best_centroid = np.array(obj_data["centroid"])

            error_new, loc_new = evaluate_query(best_centroid, gt_centers)

            old = stored_cg.get(query)
            if old is None:
                print(f"  {query:10s}: NOT IN STORED RESULTS")
                cg_ok = False
                continue

            error_old = old["error_m"]
            match = True
            if error_new is None and error_old is None:
                pass
            elif error_new is None or error_old is None:
                match = False
            elif abs(error_new - error_old) > 0.01:  # slightly looser for float
                match = False

            status = "[OK]" if match else "[FAIL] MISMATCH"
            e_new_str = f"{error_new:.4f}" if error_new is not None else "None"
            e_old_str = f"{error_old:.4f}" if error_old is not None else "None"
            print(f"  {query:10s}: new={e_new_str}m, stored={e_old_str}m  {status}")
            if not match:
                cg_ok = False
                if best_centroid is not None:
                    print(f"    new_pred: {best_centroid}")
                if old.get("pred_position"):
                    print(f"    stored_pred: {old['pred_position']}")

        print(f"\nCG verification: {'PASS' if cg_ok else 'FAIL'}")

    # 6. FINAL PAPER NUMBERS CHECK
    print("\n" + "=" * 70)
    print("FINAL PAPER NUMBERS VERIFICATION")
    print("=" * 70)

    # Paper Table 3 numbers (from main.tex)
    paper_numbers = {
        "CG":       {"loc_1m": 6.1, "loc_2m": 54.1},
        "DenseMap": {"loc_1m": 16.2, "loc_2m": 41.9},
        "VLMaps":   {"loc_1m": 7.0, "loc_2m": 23.5},
    }

    # Recompute from files
    for method, paper in paper_numbers.items():
        if method == "CG":
            with open(RESULTS_DIR / "cg_36scenes_consistent_eval.json") as f:
                data = json.load(f)["per_query"]
            n = len(data)
            loc1 = 100 * sum(1 for q in data if q["error_m"] is not None and q["error_m"] < 1.0) / n
            loc2 = 100 * sum(1 for q in data if q["error_m"] is not None and q["error_m"] < 2.0) / n
        elif method == "VLMaps":
            with open(RESULTS_DIR / "phase_c_vlmaps_perquery.json") as f:
                data = json.load(f)
            n = len(data)
            loc1 = 100 * sum(1 for q in data if q["error_m"] is not None and q["error_m"] < 1.0) / n
            loc2 = 100 * sum(1 for q in data if q["error_m"] is not None and q["error_m"] < 2.0) / n
        elif method == "DenseMap":
            with open(RESULTS_DIR / "phase_c_densemap_perquery.json") as f:
                data = json.load(f)
            n = len(data)
            loc1 = 100 * sum(1 for q in data if q["error_m"] is not None and q["error_m"] < 1.0) / n
            loc2 = 100 * sum(1 for q in data if q["error_m"] is not None and q["error_m"] < 2.0) / n

        match1 = abs(loc1 - paper["loc_1m"]) < 0.05
        match2 = abs(loc2 - paper["loc_2m"]) < 0.05
        s1 = "[OK]" if match1 else "[FAIL]"
        s2 = "[OK]" if match2 else "[FAIL]"
        print(f"  {method:10s}: Loc@1m={loc1:.1f}% (paper={paper['loc_1m']}%) {s1}  |  Loc@2m={loc2:.1f}% (paper={paper['loc_2m']}%) {s2}")

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
