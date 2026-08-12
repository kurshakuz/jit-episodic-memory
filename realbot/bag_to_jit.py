#!/usr/bin/env python3
"""
Extract a JIT episodic-memory trace from a ZED .mcap (real quadruped robot).

For each keyframe: synchronized rectified RGB + registered depth (metres) + the
camera-optical pose in the odom world frame, composed as
    T_odom_optical = T_odom_zed_camera_link (odometry)  x  T_cameralink_optical (static chain).

    python realbot/bag_to_jit.py "<file.mcap>" [out_dir] [stride]

Writes: <out>/images/kf_XXXX.jpg, <out>/depth/kf_XXXX.npy (float16 m),
        <out>/poses.npy (N x 4x4), <out>/meta.json.
"""
import sys, json
from pathlib import Path
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from PIL import Image
from static_tf import load_static

MCAP = sys.argv[1]
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent / "trace"
STRIDE = int(sys.argv[3]) if len(sys.argv) > 3 else 8
(OUT / "images").mkdir(parents=True, exist_ok=True)
(OUT / "depth").mkdir(parents=True, exist_ok=True)

RGB, DEPTH = "/zed/zed_node/rgb/color/rect/image", "/zed/zed_node/depth/depth_registered"
INFO, ODOM, TF, TFS = "/zed/zed_node/rgb/color/rect/camera_info", "/zed/zed_node/odom", "/tf", "/tf_static"
CHAIN = [("zed_camera_link", "zed_camera_center"),
         ("zed_camera_center", "zed_left_camera_frame"),
         ("zed_left_camera_frame", "zed_left_camera_frame_optical")]


def quat_to_R(x, y, z, w):
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array([[1 - (yy + zz), xy - wz, xz + wy],
                     [xy + wz, 1 - (xx + zz), yz - wx],
                     [xz - wy, yz + wx, 1 - (xx + yy)]])


def T_from(p, q):
    T = np.eye(4)
    T[:3, :3] = quat_to_R(q.x, q.y, q.z, q.w)
    T[:3, 3] = [p.x, p.y, p.z]
    return T


def stamp(hdr):
    return hdr.stamp.sec + hdr.stamp.nanosec * 1e-9


reader = make_reader(open(MCAP, "rb"), decoder_factories=[DecoderFactory()])
static = load_static(CHAIN)   # fallback: /tf_static only exists in chunk 0
K = None
latest_depth = None      # (t, HxW float32)
latest_odom = None       # (t, 4x4 T_odom_cameralink)
n_rgb = 0
poses, meta = [], []
saved = 0

for schema, ch, message, ros in reader.iter_decoded_messages(
        topics=[RGB, DEPTH, INFO, ODOM, TF, TFS], log_time_order=True):
    t = ch.topic
    if t in (TF, TFS):
        for tr in ros.transforms:
            key = (tr.header.frame_id, tr.child_frame_id)
            if key in CHAIN and key not in static:
                static[key] = T_from(tr.transform.translation, tr.transform.rotation)
    elif t == INFO and K is None:
        k = ros.k
        K = (float(k[0]), float(k[4]), float(k[2]), float(k[5]))  # fx,fy,cx,cy
    elif t == ODOM:
        latest_odom = (stamp(ros.header), T_from(ros.pose.pose.position, ros.pose.pose.orientation))
    elif t == DEPTH:
        d = np.frombuffer(ros.data, dtype=np.float32).reshape(ros.height, ros.width)
        latest_depth = (stamp(ros.header), d)
    elif t == RGB:
        n_rgb += 1
        if n_rgb % STRIDE != 1:
            continue
        if K is None or latest_depth is None or latest_odom is None or len(static) < 3:
            continue
        T_cl_opt = static[CHAIN[0]] @ static[CHAIN[1]] @ static[CHAIN[2]]
        T_odom_opt = latest_odom[1] @ T_cl_opt
        rgb = np.frombuffer(ros.data, dtype=np.uint8).reshape(ros.height, ros.width, 4)[:, :, [2, 1, 0]]
        Image.fromarray(rgb).save(OUT / "images" / f"kf_{saved:04d}.jpg", quality=90)
        np.save(OUT / "depth" / f"kf_{saved:04d}.npy", latest_depth[1].astype(np.float16))
        poses.append(T_odom_opt)
        meta.append({"i": saved, "t": stamp(ros.header),
                     "image": f"images/kf_{saved:04d}.jpg", "depth": f"depth/kf_{saved:04d}.npy"})
        saved += 1

np.save(OUT / "poses.npy", np.array(poses))
json.dump({"world_frame": "odom", "intrinsics": {"fx": K[0], "fy": K[1], "cx": K[2], "cy": K[3]},
           "n_keyframes": saved, "rgb_seen": n_rgb, "keyframes": meta},
          open(OUT / "meta.json", "w"), indent=1)
print(f"extracted {saved} keyframes from {n_rgb} RGB frames  (stride {STRIDE})")
print(f"intrinsics fx={K[0]:.1f} fy={K[1]:.1f} cx={K[2]:.1f} cy={K[3]:.1f} | static chain edges: {len(static)}")
print(f"-> {OUT}")
