#!/usr/bin/env python3
"""
E4: ZED vs LiDAR depth agreement + how localization degrades with range and improves with
views. The accuracy_*.json rows already carry, per detection, the ZED range and the
independent LiDAR range, and the loc_err_xy is exactly the gap between a ZED-depth and a
LiDAR-depth localization of the same object. Here we (a) pool ZED-vs-LiDAR depth error
against range, and (b) subsample the views of the view-rich objects to measure how the
multi-view cluster tightens as views accumulate.

    python realbot/exp_analyze.py     (reads _out/experiments/accuracy_*.json + result JSONs)
Writes exp_analysis.json + exp_analysis.png.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent
OUT = R / "_out" / "experiments"
ACC = sorted(OUT.glob("accuracy_*.json"))

# ---- (a) ZED vs LiDAR depth error against range ----
ranges, derrs, labels = [], [], []
for f in ACC:
    for q, o in json.load(open(f)).items():
        for v in o["views"]:
            ranges.append(v["lidar_range"]); derrs.append(v["depth_err"]); labels.append(q)
ranges, derrs = np.array(ranges), np.array(derrs)
corr = float(np.corrcoef(ranges, derrs)[0, 1])
# binned mean depth error
bins = [0, 3, 6, 9, 12, 20]
binned = []
for lo, hi in zip(bins, bins[1:]):
    msk = (ranges >= lo) & (ranges < hi)
    if msk.sum():
        binned.append({"range": f"{lo}-{hi}m", "n": int(msk.sum()),
                       "mean_depth_err_m": round(float(derrs[msk].mean()), 3)})

# ---- (b) views vs centroid accuracy: does fusing more views converge to the LiDAR GT? ----
TRACES = {"trace": ["car"], "trace_c3": ["white car", "gray suv"]}
views_curves = {}
rng = np.random.RandomState(0)
for tname, qs in TRACES.items():
    acc = json.load(open(OUT / f"accuracy_{tname}.json"))
    for q in qs:
        p = R / tname / f"result_{q}_map.json"
        if not p.exists() or q not in acc:
            continue
        res = json.load(open(p))
        inst = max(res["instances"], key=lambda i: i["n_views"])
        P = np.array([m["xyz"][:2] for m in inst["members"]])
        gt = np.array(acc[q]["gt_centroid"][:2]); N = len(P)
        curve = []
        for n in [1, 2, 3, 5, 8, 12, 16, 20, 30, 40, 60]:
            if n > N:
                break
            errs = [np.linalg.norm(P[rng.choice(N, n, replace=False)].mean(0) - gt) for _ in range(60)]
            curve.append({"n_views": n, "centroid_err_m": round(float(np.mean(errs)), 3),
                          "centroid_err_std": round(float(np.std(errs)), 3)})
        views_curves[f"{tname}:{q}"] = {"N": N, "curve": curve}

out = {"depth_vs_range": {"pearson_r": round(corr, 3), "binned": binned,
                          "overall_mean_depth_err_m": round(float(derrs.mean()), 3), "n": len(derrs)},
       "views_vs_spread": views_curves}
json.dump(out, open(OUT / "exp_analysis.json", "w"), indent=1)
print(f"depth-vs-range: r={corr:.2f}, overall mean depth err {derrs.mean():.2f}m over {len(derrs)} detections")
for b in binned:
    print(f"  {b['range']:7s} n={b['n']:3d}  mean depth err {b['mean_depth_err_m']}m")
for k, v in views_curves.items():
    print(f"{k}: centroid err {v['curve'][0]['centroid_err_m']}m @1 view -> {v['curve'][-1]['centroid_err_m']}m @{v['curve'][-1]['n_views']} views")

# ---- figure ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].scatter(ranges, derrs, c=derrs, cmap="viridis", s=18, alpha=.7)
ax[0].set_xlabel("object range (LiDAR, m)"); ax[0].set_ylabel("ZED–LiDAR depth error (m)")
ax[0].set_title(f"Depth agreement vs range (r={corr:.2f})"); ax[0].grid(alpha=.3)
for k, v in views_curves.items():
    c = v["curve"]; ax[1].plot([p["n_views"] for p in c], [p["centroid_err_m"] for p in c], "o-", label=k.split(":")[1])
ax[1].set_xlabel("views fused"); ax[1].set_ylabel("centroid error vs LiDAR GT (m)")
ax[1].set_title("Fusing more views converges to ground truth"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(OUT / "exp_analysis.png", dpi=110)
print(f"-> {OUT/'exp_analysis.json'} + exp_analysis.png")
