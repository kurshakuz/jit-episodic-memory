#!/usr/bin/env python3
"""Compare CG stride=1 vs stride=4 results from the scaling experiment.

Reads the JSON outputs from both runs and produces a matched per-scene
comparison table, plus aggregate statistics for the paper.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def per_scene_metrics(results: dict) -> dict:
    """Compute per-scene Loc@d from per_query results."""
    thresholds = [0.5, 1.0, 2.0, 3.0]
    scenes = {}
    for q in results["per_query"]:
        sid = q["scene_id"]
        if sid not in scenes:
            scenes[sid] = []
        scenes[sid].append(q)

    out = {}
    for sid, queries in scenes.items():
        n = len(queries)
        metrics = {"n_queries": n}
        for t in thresholds:
            key = f"loc_{t}m"
            hits = sum(1 for q in queries if q.get(key, False))
            metrics[key] = round(100 * hits / n, 1)
        errors = [q["error_m"] for q in queries if q["error_m"] is not None]
        metrics["median_error_m"] = round(sorted(errors)[len(errors)//2], 2) if errors else None
        out[sid] = metrics

    return out


def main():
    s1_path = PROJECT_ROOT / "outputs" / "paper_results" / "cg_stride1_results.json"
    s4_timed_path = PROJECT_ROOT / "outputs" / "paper_results" / "cg_stride4_timed_results.json"
    s4_full_path = PROJECT_ROOT / "outputs" / "paper_results" / "cg_mindet1_results.json"

    # Also load JIT results for context
    jit_path = PROJECT_ROOT / "outputs" / "full_scale_eval" / "full_results_fixed.json"

    files_needed = [s1_path, s4_full_path, jit_path]
    for p in files_needed:
        if not p.exists():
            print(f"ERROR: Missing {p}")
            sys.exit(1)

    s1 = load_results(s1_path)
    s4_full = load_results(s4_full_path)

    # Timed stride=4 results (if available)
    s4_timed = load_results(s4_timed_path) if s4_timed_path.exists() else None

    jit = load_results(jit_path)

    # Per-scene analysis
    s1_scenes = per_scene_metrics(s1)
    s4_scenes = per_scene_metrics(s4_full)
    
    # Common scenes
    common = sorted(set(s1_scenes.keys()) & set(s4_scenes.keys()))

    print("=" * 80)
    print("CG SCALING EXPERIMENT: stride=1 vs stride=4")
    print("=" * 80)
    print()

    # Timing comparison
    s1_time = s1.get("elapsed_s", 0)
    s4_time = s4_timed["elapsed_s"] if s4_timed else None
    
    print(f"  Stride=1: {len(s1_scenes)} scenes, {s1['aggregate']['n_queries']} queries")
    print(f"  Stride=4: {len(common)} matched scenes from full 36-scene run")
    print(f"  Stride=1 runtime: {s1_time/60:.1f} min ({s1_time:.0f}s)")
    if s4_time:
        print(f"  Stride=4 runtime (fresh, matched scenes): {s4_time/60:.1f} min ({s4_time:.0f}s)")
        print(f"  Compute ratio: {s1_time/s4_time:.1f}×")
    print()

    # Per-scene table
    header = f"{'Scene':<16} {'Queries':>7}  {'s4 L@1m':>7} {'s1 L@1m':>7}  {'s4 L@2m':>7} {'s1 L@2m':>7}  {'s4 Med':>6} {'s1 Med':>6}"
    print(header)
    print("-" * len(header))

    s1_total_q = 0
    s1_total_l1 = 0
    s1_total_l2 = 0
    s4_total_l1 = 0
    s4_total_l2 = 0

    for sid in common:
        s4s = s4_scenes[sid]
        s1s = s1_scenes[sid]
        n = s4s["n_queries"]  # should match
        s1_total_q += n
        
        s4_l1 = s4s["loc_1.0m"]
        s1_l1 = s1s["loc_1.0m"]
        s4_l2 = s4s["loc_2.0m"]
        s1_l2 = s1s["loc_2.0m"]
        s4_med = s4s["median_error_m"]
        s1_med = s1s["median_error_m"]

        s4_total_l1 += s4_l1 * n / 100
        s1_total_l1 += s1_l1 * n / 100
        s4_total_l2 += s4_l2 * n / 100
        s1_total_l2 += s1_l2 * n / 100

        print(f"{sid:<16} {n:>7}  {s4_l1:>6.1f}% {s1_l1:>6.1f}%  {s4_l2:>6.1f}% {s1_l2:>6.1f}%  {s4_med:>6.2f} {s1_med:>6.2f}")

    print("-" * len(header))
    if s1_total_q > 0:
        print(f"{'AGGREGATE':<16} {s1_total_q:>7}  "
              f"{100*s4_total_l1/s1_total_q:>6.1f}% {100*s1_total_l1/s1_total_q:>6.1f}%  "
              f"{100*s4_total_l2/s1_total_q:>6.1f}% {100*s1_total_l2/s1_total_q:>6.1f}%")
    print()

    # Stride=1 aggregate
    print("Stride=1 aggregate (all queries):")
    for k, v in s1["aggregate"].items():
        print(f"  {k}: {v}")
    print()

    # JIT comparison context
    print("JIT cascade (from full_results_fixed.json):")
    if "aggregate" in jit:
        for k, v in jit["aggregate"].items():
            print(f"  {k}: {v}")
    print()

    # Summary for paper
    print("=" * 80)
    print("SUMMARY FOR PAPER:")
    print("=" * 80)
    print(f"CG stride=4: Loc@1m = {s4_full['aggregate']['loc_1.0m']}% ({s4_full['aggregate']['n_scenes']} scenes, {s4_full['aggregate']['n_queries']} queries)")
    print(f"CG stride=1: Loc@1m = {s1['aggregate']['loc_1.0m']}% ({s1['aggregate']['n_scenes']} scenes, {s1['aggregate']['n_queries']} queries)")
    if s4_time:
        print(f"Compute cost: stride=1 takes {s1_time/s4_time:.1f}× longer than stride=4")
    print(f"Frames per scene: stride=4=40, stride=1=160 (4× more)")

    # Save comparison
    comparison = {
        "experiment": "CG scaling: stride=1 vs stride=4",
        "n_common_scenes": len(common),
        "common_scene_ids": common,
        "stride1": {
            "aggregate": s1["aggregate"],
            "elapsed_s": s1_time,
            "frames_per_scene": 160,
        },
        "stride4_full": {
            "aggregate": s4_full["aggregate"],
            "frames_per_scene": 40,
        },
        "per_scene_comparison": {},
    }
    if s4_timed:
        comparison["stride4_timed"] = {
            "aggregate": s4_timed["aggregate"],
            "elapsed_s": s4_timed["elapsed_s"],
        }

    for sid in common:
        comparison["per_scene_comparison"][sid] = {
            "stride4": s4_scenes[sid],
            "stride1": s1_scenes[sid],
        }

    out_path = PROJECT_ROOT / "outputs" / "paper_results" / "cg_scaling_comparison.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nComparison saved to {out_path}")


if __name__ == "__main__":
    main()
