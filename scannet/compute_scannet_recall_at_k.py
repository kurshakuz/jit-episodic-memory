#!/usr/bin/env python3
"""
Compute Recall@K summary from ScanNet evaluation results.

Reads the output of evaluate_v2.py (with ranked_centroids) and computes:
- Recall@K for K={1,3,5} at all thresholds
- Macro-averaged (per-scene) and micro-averaged (per-query)
- Comparison table for paper

Usage:
    python scannet/compute_scannet_recall_at_k.py
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent / "results"


def load_results(filename):
    path = RESULTS_DIR / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def compute_recall_at_k_micro(per_query, method, thresholds=[0.5, 1.0, 2.0, 3.0]):
    """Micro-averaged Recall@K (per-query)."""
    queries = [q for q in per_query if q.get("method") == method]
    if not queries:
        return {}
    
    n = len(queries)
    results = {}
    for t in thresholds:
        for k in [1, 3, 5]:
            key = f"recall@{k}_{t}m"
            hits = sum(1 for q in queries 
                       if q.get("recall_at_k") and q["recall_at_k"].get(key, False))
            results[key] = {"hits": hits, "total": n, "pct": round(100.0 * hits / n, 1)}
    
    # Average clusters
    clusters = [q.get("num_clusters", 0) for q in queries]
    results["avg_clusters"] = round(np.mean(clusters), 1) if clusters else 0
    results["n_queries"] = n
    
    return results


def compute_recall_at_k_macro(per_query, method, thresholds=[0.5, 1.0, 2.0, 3.0]):
    """Macro-averaged Recall@K (per-scene, then averaged across scenes)."""
    queries = [q for q in per_query if q.get("method") == method]
    if not queries:
        return {}
    
    # Group by scene
    by_scene = defaultdict(list)
    for q in queries:
        by_scene[q["scene_id"]].append(q)
    
    results = {}
    for t in thresholds:
        for k in [1, 3, 5]:
            key = f"recall@{k}_{t}m"
            scene_accs = []
            for scene_id, scene_queries in by_scene.items():
                hits = sum(1 for q in scene_queries
                           if q.get("recall_at_k") and q["recall_at_k"].get(key, False))
                scene_accs.append(hits / len(scene_queries) if scene_queries else 0)
            macro_avg = np.mean(scene_accs) if scene_accs else 0
            results[key] = {"pct": round(100.0 * macro_avg, 1), "n_scenes": len(by_scene)}
    
    results["n_queries"] = len(queries)
    results["n_scenes"] = len(by_scene)
    
    # Average clusters per query
    clusters = [q.get("num_clusters", 0) for q in queries]
    results["avg_clusters"] = round(np.mean(clusters), 1)
    
    return results


def main():
    # Try to load the recall_at_k results
    data = load_results("scannet_eval_v2_recall_at_k.json")
    if data is None:
        print("No results found. Run evaluate_v2.py first.")
        return
    
    per_query = data["per_query"]
    methods = set(q.get("method", "") for q in per_query)
    n_scenes = len(set(q["scene_id"] for q in per_query))
    n_queries = len(per_query)
    
    print(f"{'='*70}")
    print(f"ScanNet Recall@K Analysis")
    print(f"{'='*70}")
    print(f"Total queries: {n_queries}, Scenes: {n_scenes}, Methods: {sorted(methods)}")
    
    # Also load BF results from old files for comparison
    bf_queries = []
    for fname in ["scannet_eval_v2_results.json", "scannet_eval_v2_new91_results.json"]:
        old_data = load_results(fname)
        if old_data and "per_query" in old_data:
            for q in old_data["per_query"]:
                if q.get("method") == "bf":
                    bf_queries.append(q)
    
    print(f"\n{'='*70}")
    print("MICRO-AVERAGED Recall@K")
    print(f"{'='*70}")
    
    for method in ["jit", "jit_l3", "jit_no_dbscan"]:
        if method not in methods:
            continue
        micro = compute_recall_at_k_micro(per_query, method)
        if not micro:
            continue
        print(f"\n  {method} ({micro['n_queries']} queries, avg {micro['avg_clusters']} clusters):")
        for t in [0.5, 1.0, 2.0, 3.0]:
            line = f"    Loc@{t}m:"
            for k in [1, 3, 5]:
                key = f"recall@{k}_{t}m"
                if key in micro:
                    line += f"  R@{k}={micro[key]['pct']}%"
            print(line)
    
    # BF comparison (R@K = R@1 for all K, since single prediction)
    if bf_queries:
        n_bf = len(bf_queries)
        print(f"\n  bf ({n_bf} queries, 0 clusters — single prediction):")
        for t in [0.5, 1.0, 2.0, 3.0]:
            d = t
            correct = sum(1 for q in bf_queries 
                         if q.get("min_distance") is not None and q["min_distance"] < d)
            pct = round(100.0 * correct / n_bf, 1)
            print(f"    Loc@{t}m:  R@1={pct}%  R@3={pct}%  R@5={pct}%  (flat)")
    
    print(f"\n{'='*70}")
    print("MACRO-AVERAGED Recall@K (per-scene)")
    print(f"{'='*70}")
    
    for method in ["jit", "jit_l3", "jit_no_dbscan"]:
        if method not in methods:
            continue
        macro = compute_recall_at_k_macro(per_query, method)
        if not macro:
            continue
        print(f"\n  {method} ({macro['n_queries']} queries, {macro['n_scenes']} scenes, avg {macro['avg_clusters']} clusters):")
        for t in [0.5, 1.0, 2.0, 3.0]:
            line = f"    Loc@{t}m:"
            for k in [1, 3, 5]:
                key = f"recall@{k}_{t}m"
                if key in macro:
                    line += f"  R@{k}={macro[key]['pct']}%"
            print(line)
    
    # BF macro comparison
    if bf_queries:
        by_scene = defaultdict(list)
        for q in bf_queries:
            by_scene[q["scene_id"]].append(q)
        n_bf_scenes = len(by_scene)
        print(f"\n  bf ({len(bf_queries)} queries, {n_bf_scenes} scenes — single prediction):")
        for t in [0.5, 1.0, 2.0, 3.0]:
            scene_accs = []
            for sid, sq in by_scene.items():
                correct = sum(1 for q in sq 
                             if q.get("min_distance") is not None and q["min_distance"] < t)
                scene_accs.append(correct / len(sq) if sq else 0)
            pct = round(100.0 * np.mean(scene_accs), 1)
            print(f"    Loc@{t}m:  R@1={pct}%  R@3={pct}%  R@5={pct}%  (flat)")
    
    # LaTeX table for paper
    print(f"\n{'='*70}")
    print("LaTeX TABLE (for supplementary)")
    print(f"{'='*70}")
    
    print(r"""
\begin{table}[h]
  \centering
  \caption{Recall@$K$ on ScanNet v2: fraction of queries where a correct location 
  appears in the top-$K$ DBSCAN clusters. BF returns a single prediction (R@K = R@1 for all K).}
  \begin{tabular}{llccc}
    \toprule
    Method & Threshold & Recall@1 & Recall@3 & Recall@5 \\
    \midrule""")
    
    for method in ["jit", "jit_l3"]:
        if method not in methods:
            continue
        micro = compute_recall_at_k_micro(per_query, method)
        name = "JIT (L1+L2)" if method == "jit" else "JIT + L3"
        for i, t in enumerate([0.5, 1.0, 2.0, 3.0]):
            r1 = micro.get(f"recall@1_{t}m", {}).get("pct", "?")
            r3 = micro.get(f"recall@3_{t}m", {}).get("pct", "?")
            r5 = micro.get(f"recall@5_{t}m", {}).get("pct", "?")
            if i == 0:
                print(f"    \\multirow{{4}}{{*}}{{{name}}} & {t}\\,m & {r1}\\% & {r3}\\% & {r5}\\% \\\\")
            else:
                print(f"     & {t}\\,m & {r1}\\% & {r3}\\% & {r5}\\% \\\\")
        print("    \\midrule")
    
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table}")
    
    # Save summary
    summary = {}
    for method in ["jit", "jit_l3", "jit_no_dbscan"]:
        if method not in methods:
            continue
        summary[method] = {
            "micro": compute_recall_at_k_micro(per_query, method),
            "macro": compute_recall_at_k_macro(per_query, method),
        }
    
    out_path = RESULTS_DIR / "scannet_recall_at_k_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    main()
