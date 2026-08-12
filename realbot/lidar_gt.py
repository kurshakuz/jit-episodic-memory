#!/usr/bin/env python3
"""
LiDAR ground truth for a JIT-localized object. The RoboSense LiDAR is an independent,
accurate range sensor; we project its points into the ZED camera at each detection and
read the true object depth, giving a sensor-independent 3D ground-truth position to
score JIT's (ZED-depth) localization against.

    python realbot/lidar_gt.py "<query>" "<file.mcap>"
"""
import sys, json, os
from pathlib import Path
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
def _quat_R(x, y, z, w):
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array([[1 - (yy + zz), xy - wz, xz + wy], [xy + wz, 1 - (xx + zz), yz - wx],
                     [xz - wy, yz + wx, 1 - (xx + yy)]])


def T_from(p, q):
    T = np.eye(4); T[:3, :3] = _quat_R(q.x, q.y, q.z, q.w); T[:3, 3] = [p.x, p.y, p.z]
    return T

QUERY = sys.argv[1] if len(sys.argv) > 1 else "car"
MCAP = sys.argv[2]
T = Path(os.environ.get("REALBOT_TRACE", str(Path(__file__).resolve().parent / "trace")))
meta = json.load(open(T / "meta.json"))
kfs = meta["keyframes"]
fx, fy, cx, cy = (meta["intrinsics"][k] for k in ("fx", "fy", "cx", "cy"))
poses_map = np.load(T / "poses_map.npy")
res = json.load(open(T / f"result_{QUERY}_map.json"))
# best-localized instance: most views among the tight (spread<0.6 m) ones
cands = [i for i in res["instances"] if i["n_views"] >= 4 and i["spread_m"] < 0.6] or res["instances"]
inst = sorted(cands, key=lambda i: (i["spread_m"], -i["n_views"]))[0]
targets = {m["kf"]: m for m in inst["members"]}            # kf idx -> member (bbox, zed xyz)
target_ts = {kf: kfs[kf]["t"] for kf in targets}
print(f"GT target: '{QUERY}' instance {inst['n_views']} views, JIT centroid {inst['centroid']} m, spread {inst['spread_m']} m")

CHAIN = [("base_link", "zed_camera_link"), ("zed_camera_link", "zed_camera_center"),
         ("zed_camera_center", "zed_left_camera_frame"), ("zed_left_camera_frame", "zed_left_camera_frame_optical")]
from static_tf import load_static
static = load_static(CHAIN + [("base_link", "rslidar")])   # fallback: /tf_static only in chunk 0
scans = {kf: (1e9, None) for kf in targets}                # kf -> (dt, xyz points)


def pc2_xyz(m):
    off = {f.name: f.offset for f in m.fields}
    n = m.width * m.height
    buf = np.frombuffer(m.data, np.uint8).reshape(n, m.point_step)
    xyz = np.stack([buf[:, off[c]:off[c] + 4].copy().view(np.float32).ravel() for c in ("x", "y", "z")], 1)
    return xyz[np.isfinite(xyz).all(1)]


reader = make_reader(open(MCAP, "rb"), decoder_factories=[DecoderFactory()])
for schema, ch, message, ros in reader.iter_decoded_messages(topics=["/tf", "/tf_static", "/rslidar_points"]):
    if ch.topic in ("/tf", "/tf_static"):
        for tr in ros.transforms:
            k = (tr.header.frame_id, tr.child_frame_id)
            if (k in CHAIN or k == ("base_link", "rslidar")) and k not in static:
                static[k] = T_from(tr.transform.translation, tr.transform.rotation)
    elif ch.topic == "/rslidar_points":
        t = ros.header.stamp.sec + ros.header.stamp.nanosec * 1e-9
        for kf, tt in target_ts.items():
            if abs(t - tt) < scans[kf][0]:
                scans[kf] = (abs(t - tt), pc2_xyz(ros))

T_base_opt = static[CHAIN[0]] @ static[CHAIN[1]] @ static[CHAIN[2]] @ static[CHAIN[3]]
T_opt_rslidar = np.linalg.inv(T_base_opt) @ static[("base_link", "rslidar")]

rows, lidar_pts_map = [], []
for kf, m in targets.items():
    dt, pc = scans[kf]
    if pc is None or dt > 0.15:
        continue
    # LiDAR points -> camera optical frame, project to pixels
    P = (T_opt_rslidar @ np.c_[pc, np.ones(len(pc))].T).T[:, :3]
    front = P[:, 2] > 0.3
    P = P[front]
    u = fx * P[:, 0] / P[:, 2] + cx
    v = fy * P[:, 1] / P[:, 2] + cy
    b = m["bbox"]; H, W = 400, 640
    u0, v0 = (b[0] + b[2]) / 2 * W, (b[1] + b[3]) / 2 * H
    near = (np.abs(u - u0) < 18) & (np.abs(v - v0) < 18)
    if near.sum() < 3:
        continue
    z_lidar = float(np.median(P[near, 2]))
    z_zed = float((np.linalg.inv(poses_map[kf]) @ np.array([*m["xyz"], 1.0]))[2])  # ZED optical-frame depth
    # LiDAR-GT map position at this detection
    X = (u0 - cx) * z_lidar / fx; Y = (v0 - cy) * z_lidar / fy
    gt_map = (poses_map[kf] @ np.array([X, Y, z_lidar, 1.0]))[:3]
    lidar_pts_map.append(gt_map)
    rows.append((kf, z_zed, z_lidar, np.array(m["xyz"]), gt_map))

if rows:
    zed_err = np.mean([abs(r[1] - r[2]) for r in rows])
    jit_c = np.array(inst["centroid"])
    gt_c = np.mean(lidar_pts_map, axis=0)
    loc_err = float(np.linalg.norm((jit_c - gt_c)[:2]))
    print(f"\nmatched {len(rows)} views with LiDAR (max dt 150 ms)")
    for kf, zz, zl, jx, gx in rows[:8]:
        print(f"  kf{kf:3d}: ZED range {zz:5.2f} m | LiDAR {zl:5.2f} m | depth err {abs(zz-zl):.2f} m")
    print(f"\nmean ZED-vs-LiDAR depth error : {zed_err:.2f} m")
    print(f"JIT map centroid  : ({jit_c[0]:.2f}, {jit_c[1]:.2f})")
    print(f"LiDAR-GT centroid : ({gt_c[0]:.2f}, {gt_c[1]:.2f})")
    print(f">>> JIT localization error vs LiDAR GT : {loc_err:.2f} m  (xy)")
else:
    print("no LiDAR matches found")
