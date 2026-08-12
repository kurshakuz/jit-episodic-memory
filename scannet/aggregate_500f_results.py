#!/usr/bin/env python3
"""
Aggregate all 500-frame evaluation results and produce comparison table.

Computes:
- Per-method macro-averaged Loc@{0.5, 1.0, 2.0, 3.0}m
- Bootstrap 95% CIs (10k resamples)
- Side-by-side comparison: 160f vs 500f
- Saves to outputs/paper_results/eval_500frames/

Usage:
    python scannet/aggregate_500f_results.py
"""

import json
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "paper_results" / "eval_500frames"
THRESHOLDS = [0.5, 1.0, 2.0, 3.0]
BOOTSTRAP_N = 10000


def load_jit_bf(fname):
    """Load JIT/BF results (evaluate_v2 format)."""
    d = json.load(open(fname))
    pq = d["per_query"]
    methods = {}
    for r in pq:
        m = r["method"]
        if m not in methods:
            methods[m] = []
        methods[m].append({
            "scene_id": r["scene_id"],
            "query": r["query"],
            "min_distance": r.get("min_distance"),
            "latency_ms": r.get("latency_ms", 0),
            "ranked_centroids": r.get("ranked_centroids"),
        })
    return methods


def load_dense(fname, method_key):
    """Load DenseMap/VLMaps results."""
    d = json.load(open(fname))
    pq = d[method_key]["per_query"]
    results = []
    for r in pq:
        results.append({
            "scene_id": r["scene_id"],
            "query": r["query"],
            "min_distance": r.get("error_m"),
            "latency_ms": r.get("query_time_ms", 0),
        })
    return results


def load_cg(fname):
    """Load ConceptGraphs results."""
    d = json.load(open(fname))
    pq = d.get("per_query", [])
    results = []
    for r in pq:
        # CG stores loc_Xm booleans and error_m
        error = r.get("error_m")
        results.append({
            "scene_id": r["scene_id"],
            "query": r["query"],
            "min_distance": error,
            "latency_ms": 0,
        })
    return results


def compute_macro(results, threshold):
    """Compute per-scene macro-averaged accuracy. None predictions count as failures."""
    scene_acc = defaultdict(list)
    for r in results:
        d = r["min_distance"]
        if d is not None:
            scene_acc[r["scene_id"]].append(d < threshold)
        else:
            scene_acc[r["scene_id"]].append(False)
    per_scene = [np.mean(v) * 100 for v in scene_acc.values()]
    return float(np.mean(per_scene)) if per_scene else 0, per_scene


def bootstrap_ci(per_scene_values, n=BOOTSTRAP_N, ci=0.95):
    """Bootstrap 95% CI from per-scene values."""
    if len(per_scene_values) < 2:
        return 0, 0
    arr = np.array(per_scene_values)
    rng = np.random.RandomState(42)
    means = []
    for _ in range(n):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(np.mean(sample))
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


def compute_median_error(results):
    """Compute median localization error."""
    errors = [r["min_distance"] for r in results
              if r["min_distance"] is not None and r["min_distance"] < 100]
    return float(np.median(errors)) if errors else None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all 500f results
    print("Loading 500-frame results...")
    all_methods = {}

    # JIT/BF
    jit_bf = load_jit_bf(RESULTS_DIR / "scannet_eval_500f_results.json")
    for m in jit_bf:
        all_methods[m] = jit_bf[m]

    # DenseMap
    all_methods["densemap"] = load_dense(
        RESULTS_DIR / "scannet_dense_500f_dm.json", "densemap")

    # VLMaps
    all_methods["vlmap"] = load_dense(
        RESULTS_DIR / "scannet_dense_500f_vl.json", "vlmap")

    # ConceptGraphs
    all_methods["conceptgraphs"] = load_cg(
        RESULTS_DIR / "scannet_cg_500f_all142.json")

    # Compute metrics
    print("\nComputing metrics with bootstrap CIs...")
    results_500f = {}

    for method_name, results in all_methods.items():
        scenes = set(r["scene_id"] for r in results)
        queries = len(results)
        median_err = compute_median_error(results)

        method_data = {
            "method": method_name,
            "n_scenes": len(scenes),
            "n_queries": queries,
            "median_error_m": median_err,
            "accuracy": {},
        }

        for t in THRESHOLDS:
            macro, per_scene = compute_macro(results, t)
            lo, hi = bootstrap_ci(per_scene)
            method_data["accuracy"][str(t)] = {
                "macro": round(macro, 1),
                "ci_lo": round(lo, 1),
                "ci_hi": round(hi, 1),
            }

        results_500f[method_name] = method_data

    # Load 160f results for comparison
    print("Loading 160-frame results for comparison...")
    results_160f = {}

    # JIT/BF 160f
    for fname in ["scannet_eval_v2_results.json", "scannet_eval_v2_new91_results.json"]:
        fpath = RESULTS_DIR / fname
        if fpath.exists():
            d = json.load(open(fpath))
            pq = d.get("per_query", [])
            for r in pq:
                m = r.get("method", "")
                if m not in results_160f:
                    results_160f[m] = []
                dist = r.get("min_distance")
                results_160f[m].append({
                    "scene_id": r["scene_id"],
                    "query": r["query"],
                    "min_distance": dist,
                })

    # Dense 160f
    for fname, mkey in [("scannet_dense_baselines_allscenes_dm.json", "densemap"),
                         ("scannet_dense_baselines_allscenes_vl.json", "vlmap")]:
        fpath = RESULTS_DIR / fname
        if fpath.exists():
            d = json.load(open(fpath))
            pq = d.get(mkey, d).get("per_query", d.get("per_query", []))
            if mkey not in results_160f:
                results_160f[mkey] = []
            for r in pq:
                results_160f[mkey].append({
                    "scene_id": r["scene_id"],
                    "query": r["query"],
                    "min_distance": r.get("error_m"),
                })

    # CG 160f
    for fname in ["scannet_conceptgraphs_results.json", "scannet_cg_new91_results.json"]:
        fpath = RESULTS_DIR / fname
        if fpath.exists():
            d = json.load(open(fpath))
            pq = d.get("per_query", [])
            if "conceptgraphs" not in results_160f:
                results_160f["conceptgraphs"] = []
            for r in pq:
                results_160f["conceptgraphs"].append({
                    "scene_id": r["scene_id"],
                    "query": r["query"],
                    "min_distance": r.get("error_m"),
                })

    # Compute 160f metrics
    metrics_160f = {}
    for method_name, results in results_160f.items():
        scenes = set(r["scene_id"] for r in results)
        method_data = {"n_scenes": len(scenes), "n_queries": len(results), "accuracy": {}}
        for t in THRESHOLDS:
            macro, per_scene = compute_macro(results, t)
            lo, hi = bootstrap_ci(per_scene)
            method_data["accuracy"][str(t)] = {
                "macro": round(macro, 1),
                "ci_lo": round(lo, 1),
                "ci_hi": round(hi, 1),
            }
        metrics_160f[method_name] = method_data

    # Build comparison table
    METHOD_DISPLAY = {
        "jit": "JIT (L1+L2)",
        "jit_l3": "JIT (unified)",
        "bf": "BF",
        "conceptgraphs": "ConceptGraphs",
        "densemap": "DenseMap-CLIP",
        "vlmap": "VLMaps-CLIP",
    }

    comparison = {
        "description": "500-frame vs 160-frame evaluation on ScanNet v2",
        "methods": {},
    }

    for m in ["jit_l3", "jit", "bf", "conceptgraphs", "densemap", "vlmap"]:
        r500 = results_500f.get(m)
        r160 = metrics_160f.get(m, metrics_160f.get(m.replace("_l3", "")))
        if r500:
            entry = {
                "display_name": METHOD_DISPLAY.get(m, m),
                "500f": r500,
            }
            if r160:
                entry["160f"] = r160
                # Compute delta at 1m
                a500 = r500["accuracy"]["1.0"]["macro"]
                a160 = r160["accuracy"]["1.0"]["macro"]
                entry["delta_1m"] = round(a500 - a160, 1)
            comparison["methods"][m] = entry

    # Save
    with open(OUTPUT_DIR / "comparison_table.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # Print table
    print("\n" + "=" * 90)
    print("500-FRAME vs 160-FRAME COMPARISON (ScanNet v2, macro-averaged, 95% bootstrap CIs)")
    print("=" * 90)
    print(f"{'Method':>20} | {'160f Loc@1m':>15} | {'500f Loc@1m':>15} | {'Delta':>7} | {'500f Scenes':>10}")
    print("-" * 90)
    for m in ["jit_l3", "jit", "bf", "conceptgraphs", "densemap", "vlmap"]:
        entry = comparison["methods"].get(m)
        if not entry:
            continue
        name = entry["display_name"]
        r500 = entry["500f"]["accuracy"]["1.0"]
        s500 = f"{r500['macro']:.1f}% [{r500['ci_lo']:.1f},{r500['ci_hi']:.1f}]"
        n500 = entry["500f"]["n_scenes"]
        if "160f" in entry:
            r160 = entry["160f"]["accuracy"]["1.0"]
            s160 = f"{r160['macro']:.1f}% [{r160['ci_lo']:.1f},{r160['ci_hi']:.1f}]"
            delta = f"{entry['delta_1m']:+.1f}pp"
        else:
            s160 = "N/A"
            delta = "N/A"
        print(f"{name:>20} | {s160:>15} | {s500:>15} | {delta:>7} | {n500:>10}")

    # Full 500f table
    print("\n" + "=" * 90)
    print("FULL 500-FRAME TABLE (all thresholds)")
    print("=" * 90)
    print(f"{'Method':>20} | {'Loc@0.5m':>10} | {'Loc@1m':>10} | {'Loc@2m':>10} | {'Loc@3m':>10} | {'Med.Err':>7}")
    print("-" * 90)
    for m in ["jit_l3", "jit", "bf", "conceptgraphs", "densemap", "vlmap"]:
        r = results_500f.get(m)
        if not r:
            continue
        name = METHOD_DISPLAY.get(m, m)
        a = r["accuracy"]
        me = f"{r['median_error_m']:.2f}m" if r['median_error_m'] else "N/A"
        print(f"{name:>20} | {a['0.5']['macro']:>8.1f}% | {a['1.0']['macro']:>8.1f}% | {a['2.0']['macro']:>8.1f}% | {a['3.0']['macro']:>8.1f}% | {me:>7}")

    # Save markdown
    with open(OUTPUT_DIR / "comparison_table.md", "w") as f:
        f.write("# 500-Frame Evaluation Results\n\n")
        f.write("## 500f vs 160f Comparison (Loc@1m, macro-averaged)\n\n")
        f.write("| Method | 160f Loc@1m | 500f Loc@1m | Delta |\n")
        f.write("|--------|:--:|:--:|:--:|\n")
        for m in ["jit_l3", "jit", "bf", "conceptgraphs", "densemap", "vlmap"]:
            entry = comparison["methods"].get(m)
            if not entry:
                continue
            name = entry["display_name"]
            r500 = entry["500f"]["accuracy"]["1.0"]
            s500 = f'{r500["macro"]:.1f}% [{r500["ci_lo"]:.1f},{r500["ci_hi"]:.1f}]'
            if "160f" in entry:
                r160 = entry["160f"]["accuracy"]["1.0"]
                s160 = f'{r160["macro"]:.1f}%'
                delta = f'{entry["delta_1m"]:+.1f}pp'
            else:
                s160 = "N/A"
                delta = "N/A"
            f.write(f"| **{name}** | {s160} | {s500} | {delta} |\n")

        f.write("\n## Full 500f Table\n\n")
        f.write("| Method | Loc@0.5m | Loc@1m | Loc@2m | Loc@3m | Median Error |\n")
        f.write("|--------|:--:|:--:|:--:|:--:|:--:|\n")
        for m in ["jit_l3", "jit", "bf", "conceptgraphs", "densemap", "vlmap"]:
            r = results_500f.get(m)
            if not r:
                continue
            name = METHOD_DISPLAY.get(m, m)
            a = r["accuracy"]
            me = f'{r["median_error_m"]:.2f}m' if r["median_error_m"] else "N/A"
            f.write(f"| **{name}** | {a['0.5']['macro']:.1f}% | {a['1.0']['macro']:.1f}% | {a['2.0']['macro']:.1f}% | {a['3.0']['macro']:.1f}% | {me} |\n")

    print(f"\nSaved to {OUTPUT_DIR}/")
    print(f"  comparison_table.json")
    print(f"  comparison_table.md")


if __name__ == "__main__":
    main()
