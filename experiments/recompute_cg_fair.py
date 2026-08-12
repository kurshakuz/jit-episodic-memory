#!/usr/bin/env python3
"""Recompute ConceptGraphs metrics with FULLY consistent evaluation.

Fixes three inconsistencies vs JIT / VLMaps / DenseMap evaluation:
  1. Exact -> Substring category matching  (314 -> 327 queries)
  2. <= -> <  threshold comparison          (no practical impact, but correct)
  3. Queries the saved CG object maps with the 10 generic JIT categories
     (not the literal GT category names), then compares to closest GT
     instance via substring matching.

This gives us numbers that are DIRECTLY comparable to VLMaps/DenseMap/JIT.

Usage:
    python experiments/recompute_cg_fair.py
"""

import json
import gzip
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MULTI_SCENE_DIR = PROJECT_ROOT / "outputs" / "multi_scene_eval"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "paper_results"
CACHE_DIR = RESULTS_DIR / "cg_official_cache"

JIT_CATS = ['toilet', 'chair', 'table', 'bed', 'couch',
            'sink', 'lamp', 'mirror', 'cabinet', 'shelf']

THRESHOLDS = [0.5, 1.0, 2.0, 3.0]


def load_gt_substring(scene_id: str):
    """Load GT instances with SUBSTRING matching (same as JIT evaluate.py).

    Returns: dict[query] -> list[np.array]
        For each of the 10 JIT categories, returns all GT instance centres
        whose lowercased category CONTAINS the query as a substring.
    """
    gt_path = MULTI_SCENE_DIR / scene_id / f"{scene_id}_ground_truth.json"
    with open(gt_path) as f:
        gt = json.load(f)

    result = {}
    for query in JIT_CATS:
        positions = []
        for obj_id, obj in gt["objects"].items():
            cat = obj.get("category", "").lower()
            if cat in ("unknown", "misc", "void", "unlabeled"):
                continue
            if query in cat:  # SUBSTRING match, same as JIT
                positions.append(np.array(obj["center"]))
        if positions:
            result[query] = positions

    return result


def load_gt_exact(scene_id: str):
    """Load GT instances with EXACT matching (what CG previously used).

    Returns: dict[category] -> list[np.array]
    """
    gt_path = MULTI_SCENE_DIR / scene_id / f"{scene_id}_ground_truth.json"
    with open(gt_path) as f:
        gt = json.load(f)

    cat_positions = {}
    for obj_id, obj in gt["objects"].items():
        cat = obj["category"]
        if cat.lower() in ("unknown", "misc", "void", "unlabeled"):
            continue
        center = np.array(obj["center"])
        if cat not in cat_positions:
            cat_positions[cat] = []
        cat_positions[cat].append(center)

    return cat_positions


def query_map(objects_serialized, query_text, clip_model, clip_tokenizer,
              device="cuda"):
    """Query the CG object map for the best-matching object.

    Returns the point-cloud centroid of the object with highest CLIP
    similarity to `query_text`.
    """
    if len(objects_serialized) == 0:
        return None

    tokens = clip_tokenizer([query_text]).to(device)
    with torch.no_grad():
        text_feat = clip_model.encode_text(tokens)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    text_feat = text_feat.cpu().float()

    best_sim = -1.0
    best_centroid = None

    for obj in objects_serialized:
        clip_ft = torch.from_numpy(obj["clip_ft"]) if isinstance(
            obj["clip_ft"], np.ndarray) else obj["clip_ft"]
        clip_ft = clip_ft.float()
        if clip_ft.dim() == 1:
            clip_ft = clip_ft.unsqueeze(0)

        sim = F.cosine_similarity(text_feat, clip_ft, dim=-1).item()
        if sim > best_sim:
            best_sim = sim
            best_centroid = obj["pcd_np"].mean(axis=0)

    return best_centroid


def main():
    import open_clip

    print("=" * 70)
    print("CG Fair Recomputation: Substring Matching + Strict Threshold")
    print(f"Date: {datetime.now().isoformat()}")
    print("=" * 70)

    # Discover scenes
    scene_dirs = sorted([
        d for d in MULTI_SCENE_DIR.iterdir()
        if d.is_dir() and (d / f"{d.name}_ground_truth.json").exists()
    ])
    scene_ids = [d.name for d in scene_dirs]
    print(f"Found {len(scene_ids)} scenes")

    # Load CLIP ViT-H-14
    print("Loading CLIP ViT-H-14 for text queries...")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-H-14", "laion2b_s32b_b79k"
    )
    clip_model = clip_model.to("cuda")
    clip_model.eval()
    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

    # Evaluate each scene with 10 JIT categories
    all_results_substring = []  # New: substring matching
    all_results_exact = []      # Old: exact matching (for comparison)

    for scene_id in scene_ids:
        map_path = CACHE_DIR / scene_id / "object_map.pkl.gz"
        if not map_path.exists():
            print(f"  [{scene_id}] No CG object map found, skipping")
            continue

        with gzip.open(map_path, "rb") as f:
            map_data = pickle.load(f)
        objects = map_data["objects"]
        n_objects = len(objects)

        # Load GT both ways
        gt_substring = load_gt_substring(scene_id)
        gt_exact_all = load_gt_exact(scene_id)

        n_sub = len(gt_substring)
        n_exact = sum(1 for c in JIT_CATS if c in [k.lower() for k in gt_exact_all])

        print(f"  [{scene_id}] {n_objects} objects, "
              f"substring queries: {n_sub}, exact queries: {n_exact}")

        # Substring queries (NEW, consistent with JIT/VLMaps)
        for query in JIT_CATS:
            if query not in gt_substring:
                continue

            gt_positions = gt_substring[query]

            # Query CG map with the GENERIC category name
            pred_pos = query_map(objects, query, clip_model, clip_tokenizer)

            if pred_pos is None:
                error = None
            else:
                dists = [float(np.linalg.norm(pred_pos - gp))
                         for gp in gt_positions]
                error = min(dists)

            result = {
                "scene_id": scene_id,
                "category": query,
                "n_gt_instances": len(gt_positions),
                "pred_position": pred_pos.tolist() if pred_pos is not None else None,
                "error_m": error,
            }
            for t in THRESHOLDS:
                result[f"loc_{t}m"] = error is not None and error < t  # STRICT <

            all_results_substring.append(result)

        # Exact queries (OLD, for comparison)
        for query in JIT_CATS:
            # Find exact match only
            exact_cats = [c for c in gt_exact_all if c.lower() == query]
            if not exact_cats:
                continue

            # Gather ALL exact-match instances
            gt_positions = []
            for ec in exact_cats:
                gt_positions.extend(gt_exact_all[ec])

            # Query CG map with the SAME generic category name
            # (reuse the prediction from substring path if already computed)
            pred_pos = query_map(objects, query, clip_model, clip_tokenizer)

            if pred_pos is None:
                error = None
            else:
                dists = [float(np.linalg.norm(pred_pos - gp))
                         for gp in gt_positions]
                error = min(dists)

            result = {
                "scene_id": scene_id,
                "category": query,
                "n_gt_instances": len(gt_positions),
                "pred_position": pred_pos.tolist() if pred_pos is not None else None,
                "error_m": error,
            }
            for t in THRESHOLDS:
                result[f"loc_{t}m"] = error is not None and error < t  # STRICT <

            all_results_exact.append(result)

    # Unload CLIP
    del clip_model, clip_tokenizer
    torch.cuda.empty_cache()

    # Report
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    for label, results in [
        ("Substring matching (NEW, consistent)", all_results_substring),
        ("Exact matching (OLD, for comparison)", all_results_exact),
    ]:
        n = len(results)
        print(f"\n  {label}: {n} queries")
        for t in THRESHOLDS:
            hits = sum(1 for r in results if r[f"loc_{t}m"])
            pct = 100 * hits / n if n > 0 else 0
            print(f"    Loc@{t}m: {hits}/{n} = {pct:.1f}%")

        errors = [r["error_m"] for r in results if r["error_m"] is not None]
        if errors:
            print(f"    Median error: {np.median(errors):.2f}m")
            print(f"    Mean error: {np.mean(errors):.2f}m")

    # Also load the PREVIOUS recomputation for comparison
    prev_path = RESULTS_DIR / "cg_36scenes_fair_eval.json"
    if prev_path.exists():
        with open(prev_path) as f:
            prev = json.load(f)
        print("\n  Previous recomputation:")
        cat10 = prev.get("10_categories", {})
        print(f"    {cat10.get('n_queries', '?')} queries")
        for key in ["loc_0.5m_closest", "loc_1.0m_closest", "loc_2.0m_closest", "loc_3.0m_closest"]:
            if key in cat10:
                print(f"    {key}: {cat10[key]}%")

    # Load old CG results (before any fix) for full comparison
    old_path = RESULTS_DIR / "cg_mindet1_results.json"
    if old_path.exists():
        with open(old_path) as f:
            old_data = json.load(f)
        agg = old_data.get("aggregate", {})
        print(f"\n  Original CG results (centroid GT, exact match, <=):")
        print(f"    {agg.get('n_queries_10cat', '?')} 10-cat queries")
        for key in ["loc_0.5m_10cat", "loc_1.0m_10cat", "loc_2.0m_10cat", "loc_3.0m_10cat"]:
            if key in agg:
                print(f"    {key}: {agg[key]}%")

    # Per-category breakdown
    print("\n" + "=" * 70)
    print("PER-CATEGORY BREAKDOWN (substring matching)")
    print("=" * 70)
    print(f"{'Category':<12s} {'N':>4s} {'Loc@0.5':>8s} {'Loc@1.0':>8s} {'Loc@2.0':>8s} {'Loc@3.0':>8s} {'Med.Err':>8s}")
    print("-" * 60)

    for cat in JIT_CATS:
        cat_results = [r for r in all_results_substring if r["category"] == cat]
        n = len(cat_results)
        if n == 0:
            print(f"  {cat:<12s} {0:>4d}")
            continue
        row = [cat, n]
        for t in THRESHOLDS:
            hits = sum(1 for r in cat_results if r[f"loc_{t}m"])
            row.append(f"{100*hits/n:.1f}%")
        errors = [r["error_m"] for r in cat_results if r["error_m"] is not None]
        med = f"{np.median(errors):.1f}m" if errors else "N/A"
        row.append(med)
        print(f"  {row[0]:<12s} {row[1]:>4d} {row[2]:>8s} {row[3]:>8s} {row[4]:>8s} {row[5]:>8s} {row[6]:>8s}")

    # Save
    n_sub = len(all_results_substring)
    metrics_sub = {}
    for t in THRESHOLDS:
        hits = sum(1 for r in all_results_substring if r[f"loc_{t}m"])
        metrics_sub[f"loc_{t}m"] = round(100 * hits / n_sub, 1) if n_sub else 0

    output = {
        "description": "CG evaluation with CONSISTENT methodology: "
                       "substring matching, strict <, closest-instance, "
                       "10 generic JIT categories queried against CG object maps",
        "date": datetime.now().isoformat(),
        "methodology": {
            "category_matching": "substring (query in category.lower())",
            "threshold_comparison": "strict < (not <=)",
            "gt_instances": "closest-instance (min distance)",
            "query_categories": JIT_CATS,
            "n_queries": n_sub,
            "floor_filtering": False,
            "clip_model": "ViT-H-14 laion2b_s32b_b79k",
        },
        "aggregate": metrics_sub,
        "per_query": all_results_substring,
    }

    out_path = RESULTS_DIR / "cg_36scenes_consistent_eval.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY: CG Evaluation Evolution")
    print("=" * 70)
    print(f"{'Version':<45s} {'Queries':>7s} {'Loc@1m':>8s} {'Loc@2m':>8s}")
    print("-" * 70)
    print(f"  {'Original (centroid GT, exact, <=)':<43s} {'314':>7s} {'0.3%':>8s} {'4.8%':>8s}")
    print(f"  {'Fix 1: closest-instance (exact, <=)':<43s} {'314':>7s} {'2.5%':>8s} {'43.9%':>8s}")

    n = len(all_results_substring)
    loc1 = 100 * sum(1 for r in all_results_substring if r["loc_1.0m"]) / n
    loc2 = 100 * sum(1 for r in all_results_substring if r["loc_2.0m"]) / n
    print(f"  {'Fix 2: + substring + strict < (FINAL)':<43s} {n:>7d} {loc1:>7.1f}% {loc2:>7.1f}%")


if __name__ == "__main__":
    main()
