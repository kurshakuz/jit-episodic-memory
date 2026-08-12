#!/usr/bin/env python3
"""Dump the latched /tf_static tree from a recording to static_tf.json.

/tf_static is transient-local: in a multi-chunk recording it is stored only in the
first chunk. bag_to_jit.py / add_map_poses.py / lidar_gt.py fall back to this file so
that later chunks (which carry no /tf_static) can still resolve the rigid camera/LiDAR
mount chain. Run once on the first chunk:

    python realbot/dump_static_tf.py "<chunk0.mcap>"    -> writes realbot/static_tf.json
"""
import sys, json
from pathlib import Path
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

MCAP = sys.argv[1]
OUT = Path(__file__).resolve().parent / "static_tf.json"
st = {}
reader = make_reader(open(MCAP, "rb"), decoder_factories=[DecoderFactory()])
for schema, ch, message, ros in reader.iter_decoded_messages(topics=["/tf_static"]):
    for tr in ros.transforms:
        t, q = tr.transform.translation, tr.transform.rotation
        st[f"{tr.header.frame_id}|{tr.child_frame_id}"] = [t.x, t.y, t.z, q.x, q.y, q.z, q.w]
if not st:
    sys.exit("no /tf_static messages found — is this the first chunk of the recording?")
json.dump(st, open(OUT, "w"), indent=1)
print(f"wrote {OUT} with {len(st)} static transforms:")
for k in sorted(st):
    print(" ", k)
