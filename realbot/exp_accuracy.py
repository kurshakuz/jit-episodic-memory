#!/usr/bin/env python3
"""
E1 + E4 data: real-robot localization accuracy vs LiDAR ground truth.

For each object (dominant instance of a query) we project the independent RoboSense LiDAR
into the ZED camera at every detection, read the true object range, and compare JIT's
ZED-depth map-frame localization to the LiDAR-derived one. Reads each chunk's LiDAR once.

    REALBOT_TRACE=realbot/trace_c3 python realbot/exp_accuracy.py "<chunk.mcap>" bench "white car" "gray suv"

Writes realbot/_out/experiments/accuracy_<trace>.json with per-object loc error, mean
depth error, and per-view (range, zed, lidar) rows.
"""
import os, sys, json
from pathlib import Path
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from static_tf import load_static

MCAP = sys.argv[1]
QUERIES = sys.argv[2:]
T = Path(os.environ.get("REALBOT_TRACE", str(Path(__file__).resolve().parent / "trace")))
OUT = Path(os.environ.get("REALBOT_OUT", str(Path(__file__).resolve().parent / "_out"))) / "experiments"
OUT.mkdir(parents=True, exist_ok=True)
meta = json.load(open(T / "meta.json")); kfs = meta["keyframes"]
fx, fy, cx, cy = (meta["intrinsics"][k] for k in ("fx", "fy", "cx", "cy"))
poses_map = np.load(T / "poses_map.npy")

CHAIN = [("base_link", "zed_camera_link"), ("zed_camera_link", "zed_camera_center"),
         ("zed_camera_center", "zed_left_camera_frame"), ("zed_left_camera_frame", "zed_left_camera_frame_optical")]
static = load_static(CHAIN + [("base_link", "rslidar")])
T_base_opt = static[CHAIN[0]] @ static[CHAIN[1]] @ static[CHAIN[2]] @ static[CHAIN[3]]
T_opt_rslidar = np.linalg.inv(T_base_opt) @ static[("base_link", "rslidar")]

# dominant (most-viewed) instance per query -> its member detections
objects = {}
for q in QUERIES:
    p = T / f"result_{q}_map.json"
    if not p.exists():
        print(f"skip {q}: no result json"); continue
    res = json.load(open(p))
    if not res["instances"]:
        continue
    inst = max(res["instances"], key=lambda i: i["n_views"])
    objects[q] = {"inst": inst, "members": {m["kf"]: m for m in inst["members"]}}

want_kfs = {kf for o in objects.values() for kf in o["members"]}
want_ts = {kf: kfs[kf]["t"] for kf in want_kfs}
scans = {kf: (1e9, None) for kf in want_kfs}


def pc2_xyz(m):
    off = {f.name: f.offset for f in m.fields}
    n = m.width * m.height
    buf = np.frombuffer(m.data, np.uint8).reshape(n, m.point_step)
    xyz = np.stack([buf[:, off[c]:off[c] + 4].copy().view(np.float32).ravel() for c in ("x", "y", "z")], 1)
    return xyz[np.isfinite(xyz).all(1)]


reader = make_reader(open(MCAP, "rb"), decoder_factories=[DecoderFactory()])
for schema, ch, message, ros in reader.iter_decoded_messages(topics=["/rslidar_points"]):
    t = ros.header.stamp.sec + ros.header.stamp.nanosec * 1e-9
    pc = None
    for kf, tt in want_ts.items():
        if abs(t - tt) < scans[kf][0]:
            if pc is None:
                pc = pc2_xyz(ros)
            scans[kf] = (abs(t - tt), pc)

results = {}
for q, o in objects.items():
    inst = o["inst"]
    rows, gt_pts = [], []
    for kf, m in o["members"].items():
        dt, pc = scans[kf]
        if pc is None or dt > 0.15:
            continue
        P = (T_opt_rslidar @ np.c_[pc, np.ones(len(pc))].T).T[:, :3]
        P = P[P[:, 2] > 0.3]
        u = fx * P[:, 0] / P[:, 2] + cx
        v = fy * P[:, 1] / P[:, 2] + cy
        b = m["bbox"]; W, H = 640, 400
        u0, v0 = (b[0] + b[2]) / 2 * W, (b[1] + b[3]) / 2 * H
        near = (np.abs(u - u0) < 18) & (np.abs(v - v0) < 18)
        if near.sum() < 3:
            continue
        z_lidar = float(np.median(P[near, 2]))
        z_zed = float((np.linalg.inv(poses_map[kf]) @ np.array([*m["xyz"], 1.0]))[2])
        X = (u0 - cx) * z_lidar / fx; Y = (v0 - cy) * z_lidar / fy
        gt_map = (poses_map[kf] @ np.array([X, Y, z_lidar, 1.0]))[:3]
        gt_pts.append(gt_map)
        rows.append({"kf": int(kf), "zed_range": round(z_zed, 3), "lidar_range": round(z_lidar, 3),
                     "depth_err": round(abs(z_zed - z_lidar), 3)})
    if not rows:
        print(f"{q}: no LiDAR matches"); continue
    jit_c = np.array(inst["centroid"]); gt_c = np.mean(gt_pts, axis=0)
    loc_err = float(np.linalg.norm((jit_c - gt_c)[:2]))
    depth_err = float(np.mean([r["depth_err"] for r in rows]))
    results[q] = {"n_views": inst["n_views"], "n_lidar_matched": len(rows),
                  "spread_m": inst["spread_m"], "jit_centroid": [round(x, 2) for x in jit_c.tolist()],
                  "gt_centroid": [round(float(x), 2) for x in gt_c.tolist()],
                  "loc_err_xy_m": round(loc_err, 3), "mean_depth_err_m": round(depth_err, 3),
                  "views": rows}
    print(f"{q:12s} n={inst['n_views']:2d} matched={len(rows):2d}  loc_err={loc_err:.2f}m  depth_err={depth_err:.2f}m")

trace_name = T.name
json.dump(results, open(OUT / f"accuracy_{trace_name}.json", "w"), indent=1)
print(f"-> {OUT / f'accuracy_{trace_name}.json'}")
