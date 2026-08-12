#!/usr/bin/env python3
"""Re-score GOAT per-dataset outputs from the cached per-scene instance memories
(goat_official_cache/<ds>_<scene>/instances.json), using the current score_scene in
run_official_goat.py. Decouples scoring from the expensive Detic build so scoring
fixes don't require a rebuild. Writes outputs/paper_results/goat_<ds>.json.

Runs in any env with numpy (run_official_goat's top-level import is home_robot-free)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import baselines.run_official_goat as rog  # noqa: E402


def rescore_dataset(ds: str):
    queries = rog.DATASETS[ds]["queries"]
    root = rog.DATASETS[ds]["root"]
    records = []
    scenes = 0
    for scdir in sorted(rog.CACHE_DIR.glob(f"{ds}_*")):
        inst_file = scdir / "instances.json"
        if not inst_file.exists():
            continue
        scene_id = scdir.name[len(ds) + 1:]
        gt = rog.load_scene_gt(root / scene_id)
        if gt is None:
            continue
        blob = json.load(open(inst_file))
        recs = rog.score_scene(scene_id, blob["instances"], gt, queries,
                               blob["build_time_s"], blob["storage_mb"],
                               blob.get("query_ms", 0.0))
        records.extend(recs)
        scenes += 1
    if not records:
        print(f"  {ds:8s}: no cached scenes")
        return None

    macro = rog.macro_loc_from_records(records)
    builds = {r["scene_id"]: r["build_time_s"] for r in records}
    stores = {r["scene_id"]: r["storage_mb"] for r in records}
    out = dict(
        method="goat", dataset=ds, n_scenes=scenes, n_queries=len(records),
        macro_loc=macro,
        avg_build_s=float(np.mean(list(builds.values()))),
        avg_storage_mb=float(np.mean(list(stores.values()))),
        avg_query_ms=float(np.mean([r["query_ms"] for r in records])),
        elapsed_s=None, per_query=records,
    )
    outp = rog.OUTPUT_DIR / f"goat_{ds}.json"
    json.dump(out, open(outp, "w"), indent=2, default=str)
    m = macro
    print(f"  {ds:8s}: n_sc={scenes} n_q={len(records)} | "
          f"Loc@0.25/0.5/1/2/3 = {m.get('0.25',float('nan')):.1f}/"
          f"{m.get('0.5',float('nan')):.1f}/{m.get('1.0',float('nan')):.1f}/"
          f"{m.get('2.0',float('nan')):.1f}/{m.get('3.0',float('nan')):.1f} | "
          f"build={out['avg_build_s']:.1f}s store={out['avg_storage_mb']:.1f}MB -> {outp.name}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="replica,arkit,hm3d,scannet")
    args = ap.parse_args()
    print("=== re-score GOAT from cached instance memories ===")
    for ds in args.datasets.split(","):
        rescore_dataset(ds.strip())


if __name__ == "__main__":
    main()
