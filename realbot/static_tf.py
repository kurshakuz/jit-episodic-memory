#!/usr/bin/env python3
"""
Static TF fallback. /tf_static is latched and only recorded in the first .mcap chunk;
later chunks contain no /tf_static, so the rigid camera-mount chain must be sourced from
static_tf.json (dumped once from chunk 0). Side-effect free:
safe to import. `load_static([(parent, child), ...])` returns {edge: 4x4}.
"""
import json
from pathlib import Path
import numpy as np


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


def _T(vals):
    x, y, z, qx, qy, qz, qw = vals
    T = np.eye(4)
    T[:3, :3] = _quat_R(qx, qy, qz, qw)
    T[:3, 3] = [x, y, z]
    return T


def load_static(keys=None):
    p = Path(__file__).resolve().parent / "static_tf.json"
    if not p.exists():
        return {}
    raw = json.load(open(p))
    out = {}
    for k, v in raw.items():
        parent, child = k.split("|")
        if keys is None or (parent, child) in keys:
            out[(parent, child)] = _T(v)
    return out
