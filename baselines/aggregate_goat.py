#!/usr/bin/env python3
"""Aggregate per-dataset goat_<ds>.json into:
  (1) a `dense_baselines.goat` block in outputs/paper_results/all_aggregated.json
      (ScanNet micro schema, mirroring densemap/vlmaps/conceptgraphs), and
  (2) a printed per-dataset macro Loc@Xm + efficiency summary for Table III / audit /
      the head-to-head prose.
Runs in any env with numpy (no home_robot needed)."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "paper_results"
DSETS = ["hm3d", "scannet", "arkit", "replica"]


def micro_loc(per_query, t):
    if not per_query:
        return None
    hits = sum(1 for r in per_query
               if r.get("min_distance") is not None and r["min_distance"] < t)
    return 100.0 * hits / len(per_query)


def per_scene_mean(per_query, key):
    by = {}
    for r in per_query:
        if r.get(key) is not None:
            by[r["scene_id"]] = r[key]           # constant within a scene
    return float(np.mean(list(by.values()))) if by else None


def macro_loc(per_query, t):
    by = defaultdict(list)
    for r in per_query:
        d = r.get("min_distance")
        by[r["scene_id"]].append(d is not None and d < t)
    if not by:
        return None
    return 100.0 * sum(sum(v) / len(v) for v in by.values()) / len(by)


def summarize(path):
    d = json.load(open(path))
    pq = d["per_query"]
    return dict(
        n_queries=len(pq),
        n_scenes=d.get("n_scenes"),
        macro={t: macro_loc(pq, t) for t in [0.5, 1.0, 2.0, 3.0]},
        micro={t: micro_loc(pq, t) for t in [0.5, 1.0, 2.0, 3.0]},
        avg_build_s=per_scene_mean(pq, "build_time_s"),
        avg_storage_mb=per_scene_mean(pq, "storage_mb"),
        avg_query_ms=float(np.mean([r["query_ms"] for r in pq
                                    if r.get("query_ms") is not None])) if pq else None,
    )


def main():
    print("=== GOAT per-dataset summary (macro Loc@Xm) ===")
    summaries = {}
    for ds in DSETS:
        p = OUT / f"goat_{ds}.json"
        if not p.exists():
            print(f"  {ds:8s}: (no file yet)")
            continue
        s = summaries[ds] = summarize(p)
        m = s["macro"]
        print(f"  {ds:8s}: n_sc={s['n_scenes']} n_q={s['n_queries']} | "
              f"macro Loc@0.5/1/2/3 = "
              f"{m[0.5]:.1f}/{m[1.0]:.1f}/{m[2.0]:.1f}/{m[3.0]:.1f} | "
              f"build={s['avg_build_s']:.1f}s storage={s['avg_storage_mb']:.1f}MB "
              f"query={s['avg_query_ms']:.1f}ms")

    # dense_baselines.goat block from ScanNet (327-query micro comparison)
    if "scannet" in summaries:
        s = summaries["scannet"]
        agg_path = OUT / "all_aggregated.json"
        agg = json.load(open(agg_path))
        agg.setdefault("dense_baselines", {})["goat"] = {
            "n_queries": s["n_queries"],
            "micro": {
                "loc_0.5m": s["micro"][0.5], "loc_1m": s["micro"][1.0],
                "loc_2m": s["micro"][2.0], "loc_3m": s["micro"][3.0],
            },
            "avg_build_s": s["avg_build_s"],
            "avg_storage_mb": s["avg_storage_mb"],
            "avg_query_ms": s["avg_query_ms"],
            "evaluation_note": (
                "Official GOAT Object Instance Memory (facebookresearch/home-robot): "
                "Detic detect-at-collection + Categorical2DSemanticMap cross-frame "
                "instance association. Localization-only; category-goal query, "
                "closest-instance, substring matching, strict <, misses counted. "
                "Navigation policy out of scope."),
        }
        json.dump(agg, open(agg_path, "w"), indent=1)
        print(f"\nwrote dense_baselines.goat -> {agg_path}")
        print("  Table III row (ScanNet): GOAT & "
              f"{s['avg_build_s']:.0f} & {s['avg_storage_mb']:.1f} & "
              f"{s['avg_query_ms']:.0f} & {s['macro'][1.0]:.1f} \\\\")


if __name__ == "__main__":
    main()
