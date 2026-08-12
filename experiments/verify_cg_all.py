#!/usr/bin/env python3
"""
CG ONLY: End-to-end re-evaluation from saved object maps.
Now with correct object map structure: {"n_objects": N, "objects": [list of dicts]}
Each object has: clip_ft (1024-dim), pcd_np (Nx3 points), etc.
"""
import sys, json, gzip, pickle
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "outputs" / "paper_results"
CG_CACHE = RESULTS_DIR / "cg_official_cache"
MULTI_SCENE = PROJECT_ROOT / "outputs" / "multi_scene_eval"
TEST_QUERIES = ["toilet", "chair", "table", "bed", "couch",
                "sink", "lamp", "mirror", "cabinet", "shelf"]
THRESHOLDS = [0.5, 1.0, 2.0, 3.0]


def load_gt_no_floor(scene_dir):
    """GT without floor filter."""
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


def main():
    import torch
    import open_clip

    print("=" * 70)
    print("CG END-TO-END VERIFICATION (ALL 36 SCENES)")
    print("=" * 70)

    # Load CLIP ViT-H-14
    print("Loading CLIP ViT-H-14...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-H-14")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.eval().to(device)
    print(f"  Loaded on {device}")

    # Pre-encode all query texts
    text_features = {}
    for query in TEST_QUERIES:
        text = tokenizer([query]).to(device)
        with torch.no_grad():
            feat = model.encode_text(text)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            text_features[query] = feat.cpu().numpy().flatten()

    # Load stored CG results
    with open(RESULTS_DIR / "cg_36scenes_consistent_eval.json") as f:
        cg_data = json.load(f)
    stored_results = {}
    for q in cg_data["per_query"]:
        key = (q["scene_id"], q["category"])
        stored_results[key] = q

    # Process each scene
    all_errors_new = []
    all_errors_stored = []
    mismatches = []
    total_queries = 0

    scene_dirs = sorted([d for d in CG_CACHE.iterdir() if d.is_dir()])
    print(f"\nProcessing {len(scene_dirs)} scenes from CG cache...")

    for scene_cache in scene_dirs:
        scene_id = scene_cache.name
        scene_dir = MULTI_SCENE / scene_id

        if not scene_dir.exists():
            print(f"  WARNING: {scene_id} scene dir not found, skipping")
            continue

        # Load object map
        obj_map_path = scene_cache / "object_map.pkl.gz"
        with gzip.open(obj_map_path, "rb") as f:
            obj_map = pickle.load(f)
        objects = obj_map["objects"]

        # Load GT
        gt = load_gt_no_floor(scene_dir)

        # Pre-extract CLIP features from all objects
        obj_features = []
        obj_centroids = []
        for obj in objects:
            clip_ft = obj.get("clip_ft")
            if clip_ft is None:
                continue
            if isinstance(clip_ft, torch.Tensor):
                clip_ft = clip_ft.cpu().numpy()
            clip_ft = clip_ft.flatten().astype(np.float32)
            norm = np.linalg.norm(clip_ft)
            if norm > 0:
                clip_ft = clip_ft / norm

            # Get centroid from pcd_np
            pcd_np = obj.get("pcd_np")
            if pcd_np is None:
                continue
            centroid = pcd_np.mean(axis=0)

            obj_features.append(clip_ft)
            obj_centroids.append(centroid)

        if not obj_features:
            print(f"  {scene_id}: 0 valid CG objects")
            continue

        obj_features = np.array(obj_features)  # (N, 1024)
        obj_centroids = np.array(obj_centroids)  # (N, 3)

        scene_ok = True
        for query in TEST_QUERIES:
            # GT centers with substring matching
            gt_centers = []
            for cat, centers in gt.items():
                if query in cat:
                    gt_centers.extend(centers)
            if not gt_centers:
                continue

            total_queries += 1

            # Cosine similarity
            text_feat = text_features[query]
            sims = obj_features @ text_feat  # (N,)
            best_idx = np.argmax(sims)
            best_centroid = obj_centroids[best_idx]

            # Compute error
            min_dist = min(float(np.linalg.norm(best_centroid - gc)) for gc in gt_centers)

            # Compare with stored
            key = (scene_id, query)
            stored = stored_results.get(key)
            if stored is None:
                print(f"  {scene_id}/{query}: NOT IN STORED RESULTS!")
                mismatches.append((scene_id, query, min_dist, None))
                continue

            error_stored = stored["error_m"]
            all_errors_new.append(min_dist)
            all_errors_stored.append(error_stored)

            if abs(min_dist - error_stored) > 0.01:
                scene_ok = False
                mismatches.append((scene_id, query, min_dist, error_stored))

        if not scene_ok:
            for sid, q, e_new, e_old in mismatches:
                if sid == scene_id:
                    print(f"  [FAIL] {scene_id}/{q}: new={e_new:.4f}, stored={e_old:.4f}, diff={abs(e_new-e_old):.4f}")

    # Final verification
    print(f"\n{'=' * 70}")
    print(f"RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total queries verified: {total_queries}")
    print(f"Mismatches (>0.01m): {len(mismatches)}")

    if mismatches:
        print("\nMISMATCHES:")
        for sid, q, e_new, e_old in mismatches:
            e_old_str = f"{e_old:.4f}" if e_old is not None else "None"
            print(f"  {sid}/{q}: new={e_new:.4f}, stored={e_old_str}")

    # Recompute Loc metrics from scratch
    print("\nLoc metrics from fresh computation:")
    for t in THRESHOLDS:
        hits = sum(1 for e in all_errors_new if e < t)
        pct = 100 * hits / total_queries
        print(f"  Loc@{t}m: {hits}/{total_queries} = {pct:.1f}%")

    print("\nLoc metrics from stored results:")
    for t in THRESHOLDS:
        hits = sum(1 for e in all_errors_stored if e < t)
        pct = 100 * hits / total_queries
        print(f"  Loc@{t}m: {hits}/{total_queries} = {pct:.1f}%")

    # Paper numbers
    print("\n" + "=" * 70)
    print("PAPER NUMBERS CHECK")
    print("=" * 70)
    paper = {"CG": (6.1, 54.1), "DenseMap": (16.2, 41.9), "VLMaps": (7.0, 23.5)}
    files = {
        "CG": ("cg_36scenes_consistent_eval.json", True),
        "DenseMap": ("phase_c_densemap_perquery.json", False),
        "VLMaps": ("phase_c_vlmaps_perquery.json", False),
    }
    for method, (loc1_paper, loc2_paper) in paper.items():
        fname, is_nested = files[method]
        with open(RESULTS_DIR / fname) as f:
            raw = json.load(f)
        data = raw["per_query"] if is_nested else raw
        n = len(data)
        loc1 = 100 * sum(1 for q in data if q["error_m"] is not None and q["error_m"] < 1.0) / n
        loc2 = 100 * sum(1 for q in data if q["error_m"] is not None and q["error_m"] < 2.0) / n
        ok1 = "[OK]" if abs(loc1 - loc1_paper) < 0.05 else "[FAIL]"
        ok2 = "[OK]" if abs(loc2 - loc2_paper) < 0.05 else "[FAIL]"
        print(f"  {method:10s}: Loc@1m={loc1:.1f}% ({ok1})  Loc@2m={loc2:.1f}% ({ok2})")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
