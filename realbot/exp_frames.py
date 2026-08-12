#!/usr/bin/env python3
"""
E3: does the drift-free map frame matter? For each object we take the exact same detector
hits (from result_<q>_map.json) and back-project them two ways — through the drift-free
map pose (T_map_optical) and through the raw odometry pose (T_odom_optical) — then cluster
both. A tighter, less fragmented cluster in the map frame is the drift-free memory paying
off; identical spreads would mean odometry drift is negligible over that pass.

    REALBOT_TRACE=realbot/trace_c3 python realbot/exp_frames.py bench "white car" "gray suv"
"""
import os, sys, json
from pathlib import Path
import numpy as np
from sklearn.cluster import DBSCAN

T = Path(os.environ.get("REALBOT_TRACE", str(Path(__file__).resolve().parent / "trace")))
OUT = Path(os.environ.get("REALBOT_OUT", str(Path(__file__).resolve().parent / "_out"))) / "experiments"
OUT.mkdir(parents=True, exist_ok=True)
QUERIES = sys.argv[1:] or ["car"]
meta = json.load(open(T / "meta.json")); kfs = meta["keyframes"]
fx, fy, cx, cy = (meta["intrinsics"][k] for k in ("fx", "fy", "cx", "cy"))
P_map = np.load(T / "poses_map.npy")
P_odom = np.load(T / "poses.npy")


def backproj(kf, bbox, pose):
    depth = np.load(T / kfs[kf]["depth"]).astype(np.float32); H, W = depth.shape
    u = int((bbox[0] + bbox[2]) / 2 * W); v = int((bbox[1] + bbox[3]) / 2 * H)
    reg = depth[max(0, v - 7):v + 7, max(0, u - 7):u + 7]
    val = reg[np.isfinite(reg) & (reg > 0.3) & (reg < 40)]
    if val.size < 4:
        return None
    Z = float(np.median(val)); X = (u - cx) * Z / fx; Y = (v - cy) * Z / fy
    return (pose[kf] @ np.array([X, Y, Z, 1.0]))[:3]


def cluster_stats(pts):
    P = np.array(pts)
    lab = DBSCAN(eps=2.0, min_samples=2).fit(P[:, :2]).labels_
    n_clusters = len({l for l in lab if l >= 0})
    if (lab >= 0).sum() == 0:
        return {"n_clusters": 0, "dominant_spread_m": None, "dominant_n": 0}
    best = max((l for l in set(lab) if l >= 0), key=lambda l: (lab == l).sum())
    Q = P[lab == best]
    c = Q[:, :2].mean(0); spread = float(np.linalg.norm(Q[:, :2] - c, axis=1).mean())
    return {"n_clusters": n_clusters, "dominant_spread_m": round(spread, 3), "dominant_n": int(len(Q))}


results = {}
for q in QUERIES:
    p = T / f"result_{q}_map.json"
    if not p.exists():
        print(f"skip {q}"); continue
    res = json.load(open(p))
    members = [m for ins in res["instances"] for m in ins["members"]]
    mp, od = [], []
    for m in members:
        a = backproj(m["kf"], m["bbox"], P_map); b = backproj(m["kf"], m["bbox"], P_odom)
        if a is not None and b is not None:
            mp.append(a); od.append(b)
    if len(mp) < 2:
        print(f"{q}: too few"); continue
    sm, so = cluster_stats(mp), cluster_stats(od)
    results[q] = {"n_det": len(mp), "map": sm, "odom": so}
    print(f"{q:12s} n={len(mp):3d}  MAP: {sm['n_clusters']}cl spread={sm['dominant_spread_m']}m   "
          f"ODOM: {so['n_clusters']}cl spread={so['dominant_spread_m']}m")

json.dump(results, open(OUT / f"frames_{T.name}.json", "w"), indent=1)
print(f"-> {OUT / f'frames_{T.name}.json'}")
