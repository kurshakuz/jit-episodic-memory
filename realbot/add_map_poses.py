#!/usr/bin/env python3
"""
Add global map-frame camera poses to the trace (drift-free, localized against the
robot's prior LiDAR map):
    T_map_optical = T_map_base (/localization/pose)  x  T_base_optical (static TF chain).
Reads only the small localization + TF topics (no images), matches to existing keyframe
timestamps, writes trace/poses_map.npy.

    python realbot/add_map_poses.py "<file.mcap>" [trace_dir]
"""
import sys, json
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

MCAP = sys.argv[1]
T = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent / "trace"
meta = json.load(open(T / "meta.json"))
kts = np.array([kf["t"] for kf in meta["keyframes"]])

CHAIN = [("base_link", "zed_camera_link"), ("zed_camera_link", "zed_camera_center"),
         ("zed_camera_center", "zed_left_camera_frame"), ("zed_left_camera_frame", "zed_left_camera_frame_optical")]
from static_tf import load_static
static, locs = load_static(CHAIN), []   # fallback: /tf_static only exists in chunk 0
reader = make_reader(open(MCAP, "rb"), decoder_factories=[DecoderFactory()])
for schema, ch, message, ros in reader.iter_decoded_messages(topics=["/tf", "/tf_static", "/localization/pose"]):
    if ch.topic in ("/tf", "/tf_static"):
        for tr in ros.transforms:
            k = (tr.header.frame_id, tr.child_frame_id)
            if k in CHAIN and k not in static:
                static[k] = T_from(tr.transform.translation, tr.transform.rotation)
    elif ch.topic == "/localization/pose":
        t = ros.header.stamp.sec + ros.header.stamp.nanosec * 1e-9
        locs.append((t, T_from(ros.pose.pose.position, ros.pose.pose.orientation)))

assert len(static) == 4, f"missing static chain edges: have {list(static)}"
T_base_opt = static[CHAIN[0]] @ static[CHAIN[1]] @ static[CHAIN[2]] @ static[CHAIN[3]]
lt = np.array([l[0] for l in locs]); lT = np.array([l[1] for l in locs])
print(f"localization poses: {len(locs)}  ({lt.min():.1f}..{lt.max():.1f}s) | keyframes: {len(kts)}")

poses_map = np.zeros((len(kts), 4, 4))
dmax = 0.0
for i, t in enumerate(kts):
    j = int(np.argmin(np.abs(lt - t))); dmax = max(dmax, abs(lt[j] - t))
    poses_map[i] = lT[j] @ T_base_opt
np.save(T / "poses_map.npy", poses_map)
pos = poses_map[:, :3, 3]
print(f"wrote poses_map.npy | max ts match gap {dmax*1000:.0f} ms")
print(f"map trajectory: x[{pos[:,0].min():.1f},{pos[:,0].max():.1f}] "
      f"y[{pos[:,1].min():.1f},{pos[:,1].max():.1f}] z[{pos[:,2].min():.2f},{pos[:,2].max():.2f}] "
      f"len~{np.sum(np.linalg.norm(np.diff(pos,axis=0),axis=1)):.1f}m")
