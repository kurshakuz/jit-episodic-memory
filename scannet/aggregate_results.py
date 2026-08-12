#!/usr/bin/env python3
"""
Aggregate ScanNet evaluation results from all phases.
Merges per-query results from:
  - Phase 1+2 (50 scenes): scannet_eval_v2_results.json
  - Phase 3 (92 new scenes): scannet_eval_v2_new92_results.json
  - Dense baselines: scannet_dense_baselines_results.json + new92 version
  - ConceptGraphs: scannet_conceptgraphs_results.json + new92 version

Produces final aggregated results with McNemar tests and bootstrap CIs.

Usage:
    python scannet/aggregate_results.py
"""

import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from scannet.config import LOCALIZATION_THRESHOLDS

RESULTS_DIR = Path(__file__).parent / "results"


def load_results(filename):
    path = RESULTS_DIR / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def merge_per_query(result_files, method_key=None, nested_keys=None):
    """Merge per_query arrays from multiple result files, deduplicating.
    
    If nested_keys is provided (e.g. ["densemap", "vlmap"]), looks for per_query
    under those nested keys instead of at the top level.
    """
    all_queries = []
    seen = set()
    
    for rf in result_files:
        data = load_results(rf)
        if data is None:
            print(f"  Warning: {rf} not found, skipping")
            continue
        
        # Collect per_query arrays from either nested or top-level
        pq_arrays = []
        if nested_keys:
            for nk in nested_keys:
                if nk in data and "per_query" in data[nk]:
                    pq_arrays.append(data[nk]["per_query"])
        else:
            if "per_query" in data:
                pq_arrays.append(data["per_query"])
        
        for pq in pq_arrays:
            for q in pq:
                key = (q["scene_id"], q["query"], q.get("method", method_key or "unknown"))
                if key not in seen:
                    seen.add(key)
                    all_queries.append(q)
    
    return all_queries


def _get_dist(q):
    """Get distance from either field name."""
    d = q.get("min_distance")
    if d is None:
        d = q.get("error_m")
    return d


def compute_localization(queries, method, threshold):
    """Compute Loc@threshold for a specific method (per-scene macro-averaged).
    
    Groups queries by scene, computes per-scene accuracy, then averages
    across scenes. Returns (macro_avg, total_queries).
    """
    method_queries = [q for q in queries if q.get("method", "") == method]
    if not method_queries:
        return 0.0, 0
    
    # Group by scene
    by_scene = defaultdict(list)
    for q in method_queries:
        by_scene[q["scene_id"]].append(q)
    
    # Per-scene accuracy
    scene_accs = []
    for sid, sq in sorted(by_scene.items()):
        correct = sum(1 for q in sq if (_get_dist(q) is not None and _get_dist(q) < threshold))
        scene_accs.append(correct / len(sq))
    
    macro_avg = float(np.mean(scene_accs))
    return macro_avg, len(method_queries)


def bootstrap_ci(queries, method, threshold, n_iter=10000, alpha=0.05):
    """Bootstrap CI for Loc@threshold (scene-level resampling).
    
    Computes per-scene accuracy, then resamples scenes with replacement
    to produce bootstrap CIs consistent with macro-averaging.
    """
    method_queries = [q for q in queries if q.get("method", "") == method]
    if not method_queries:
        return 0.0, 0.0
    
    # Group by scene, compute per-scene accuracy
    by_scene = defaultdict(list)
    for q in method_queries:
        by_scene[q["scene_id"]].append(q)
    
    scene_accs = []
    for sid, sq in sorted(by_scene.items()):
        correct = sum(1 for q in sq if (_get_dist(q) is not None and _get_dist(q) < threshold))
        scene_accs.append(correct / len(sq))
    
    scene_accs = np.array(scene_accs)
    n_scenes = len(scene_accs)
    
    rng = np.random.RandomState(42)
    boot_means = []
    for _ in range(n_iter):
        idx = rng.choice(n_scenes, size=n_scenes, replace=True)
        boot_means.append(np.mean(scene_accs[idx]))
    
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return lo, hi


def mcnemars_test(queries, method_a, method_b, threshold):
    """McNemar's test between two methods at given threshold."""
    from scipy.stats import chi2 as chi2_dist
    
    # Build per-query correctness for both methods
    a_correct = {}
    b_correct = {}
    
    for q in queries:
        key = (q["scene_id"], q["query"])
        dist = q.get("min_distance")
        if dist is None:
            dist = q.get("error_m")
        is_correct = dist is not None and dist < threshold
        
        method = q.get("method", "")
        if method == method_a:
            a_correct[key] = is_correct
        elif method == method_b:
            b_correct[key] = is_correct
    
    # Align on shared queries
    shared_keys = set(a_correct.keys()) & set(b_correct.keys())
    if not shared_keys:
        return 0, 1.0, 0
    
    # Count discordant pairs
    b_c = 0  # A correct, B wrong
    c_b = 0  # A wrong, B correct
    for key in shared_keys:
        if a_correct[key] and not b_correct[key]:
            b_c += 1
        elif not a_correct[key] and b_correct[key]:
            c_b += 1
    
    n_disc = b_c + c_b
    if n_disc == 0:
        return 0, 1.0, 0
    
    # McNemar's chi-squared (with continuity correction)
    chi2 = (abs(b_c - c_b) - 1) ** 2 / (b_c + c_b)
    p_value = 1 - chi2_dist.cdf(chi2, df=1)
    
    return chi2, p_value, n_disc


def main():
    print("=" * 70)
    print("ScanNet Aggregated Evaluation Results")
    print("=" * 70)
    
    # Collect all JIT/BF results
    # New recall_at_k results (with ranked_centroids) take priority
    jit_files = ["scannet_eval_v2_recall_at_k.json",
                 "scannet_eval_v2_results.json", "scannet_eval_v2_new91_results.json"]
    jit_queries = merge_per_query(jit_files)
    
    # Collect dense baseline results
    dense_files = ["scannet_dense_baselines_results.json", "scannet_dense_new91_results.json"]
    dense_queries = merge_per_query(dense_files, nested_keys=["densemap", "vlmap"])
    
    # Collect ConceptGraphs results
    cg_files = ["scannet_conceptgraphs_results.json", "scannet_cg_new91_results.json"]
    cg_queries = merge_per_query(cg_files)
    
    # All queries combined
    all_queries = jit_queries + dense_queries + cg_queries
    
    # Summary
    methods_present = set(q.get("method", "") for q in all_queries)
    scenes_present = set(q["scene_id"] for q in all_queries)
    print(f"\nMethods: {sorted(methods_present)}")
    print(f"Total scenes: {len(scenes_present)}")
    print(f"Total per-query records: {len(all_queries)}")
    
    for method in sorted(methods_present):
        mq = [q for q in all_queries if q.get("method", "") == method]
        ms = set(q["scene_id"] for q in mq)
        print(f"  {method}: {len(mq)} queries from {len(ms)} scenes")
    
    # Localization results
    METHOD_NAMES = {
        "jit": "JIT (L1+L2)",
        "jit_l3": "JIT + L3",
        "jit_no_dbscan": "L1+OWL+Depth (no DBSCAN)",
        "bf": "BF + Depth",
        "densemap": "DenseMap",
        "vlmap": "VLMaps",
        "conceptgraphs": "ConceptGraphs",
    }
    
    method_order = ["jit_l3", "jit", "jit_no_dbscan", "bf", "conceptgraphs", "densemap", "vlmap"]
    
    print(f"\n{'Method':<20} {'N':>5}", end="")
    for t in LOCALIZATION_THRESHOLDS:
        print(f"  {'Loc@'+str(t)+'m':>12}", end="")
    print()
    print("-" * 80)
    
    for method in method_order:
        if method not in methods_present:
            continue
        mq = [q for q in all_queries if q.get("method", "") == method]
        name = METHOD_NAMES.get(method, method)
        print(f"{name:<20} {len(mq):>5}", end="")
        for t in LOCALIZATION_THRESHOLDS:
            acc, _ = compute_localization(all_queries, method, t)
            lo, hi = bootstrap_ci(all_queries, method, t)
            print(f"  {acc*100:5.1f} [{lo*100:.1f},{hi*100:.1f}]", end="")
        print()
    
    # McNemar tests
    print(f"\n{'='*70}")
    print("McNemar's Tests")
    print(f"{'='*70}")
    
    test_pairs = [
        ("jit_l3", "bf"), ("jit_l3", "conceptgraphs"),
        ("jit", "bf"), ("jit_l3", "jit"),
        ("jit", "conceptgraphs"), ("bf", "conceptgraphs"),
    ]
    
    for t in [0.5, 1.0, 2.0]:
        print(f"\n  Loc@{t}m:")
        for m_a, m_b in test_pairs:
            if m_a in methods_present and m_b in methods_present:
                chi2, p_val, n_disc = mcnemars_test(all_queries, m_a, m_b, t)
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                a_name = METHOD_NAMES.get(m_a, m_a)
                b_name = METHOD_NAMES.get(m_b, m_b)
                a_acc, _ = compute_localization(all_queries, m_a, t)
                b_acc, _ = compute_localization(all_queries, m_b, t)
                print(f"    {a_name:15s} ({a_acc*100:.1f}%) vs {b_name:15s} ({b_acc*100:.1f}%): "
                      f"p={p_val:.4f} {sig} (disc={n_disc})")
    
    # Per-category breakdown
    print(f"\n{'='*70}")
    print("Per-Category Breakdown (Loc@1m)")
    print(f"{'='*70}")
    
    for method in method_order:
        if method not in methods_present:
            continue
        mq = [q for q in all_queries if q.get("method", "") == method]
        by_cat = defaultdict(list)
        for q in mq:
            by_cat[q["query"]].append(q)
        
        print(f"\n  {METHOD_NAMES.get(method, method)}:")
        for cat in sorted(by_cat.keys()):
            cq = by_cat[cat]
            n = len(cq)
            correct = 0
            for q in cq:
                d = q.get("min_distance")
                if d is None:
                    d = q.get("error_m")
                if d is not None and d < 1.0:
                    correct += 1
            print(f"    {cat:12s}: {correct}/{n} ({100*correct/n:.0f}%)")
    
    # Save aggregated results
    output = {
        "metadata": {
            "dataset": "scannet",
            "num_scenes": len(scenes_present),
            "scenes": sorted(scenes_present),
            "total_queries": len(all_queries),
        },
        "per_query": all_queries,
        "summary": {},
        "mcnemars": {},
    }
    
    for method in method_order:
        if method not in methods_present:
            continue
        mq = [q for q in all_queries if q.get("method", "") == method]
        summary = {"name": METHOD_NAMES.get(method, method), "num_queries": len(mq)}
        for t in LOCALIZATION_THRESHOLDS:
            acc, _ = compute_localization(all_queries, method, t)
            lo, hi = bootstrap_ci(all_queries, method, t)
            summary[f"loc_{t}m"] = acc
            summary[f"ci_{t}m"] = [lo, hi]
        output["summary"][method] = summary
    
    for t in [0.5, 1.0, 2.0]:
        for m_a, m_b in test_pairs:
            if m_a in methods_present and m_b in methods_present:
                chi2, p_val, n_disc = mcnemars_test(all_queries, m_a, m_b, t)
                key = f"{m_a}_vs_{m_b}_at_{t}m"
                output["mcnemars"][key] = {
                    "chi2": float(chi2), "p_value": float(p_val), 
                    "n_discordant": int(n_disc)
                }
    
    out_path = RESULTS_DIR / "scannet_aggregated_141scenes.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
