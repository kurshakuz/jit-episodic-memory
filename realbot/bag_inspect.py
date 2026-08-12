#!/usr/bin/env python3
"""
Discovery pass over ONE .mcap file (standalone, no ROS install) — dumps the
schema/encoding/intrinsics/TF details needed to build the MCAP->JIT extractor.

    python realbot/bag_inspect.py "<chunk0.mcap>"
"""
import sys
import collections
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

path = sys.argv[1]
reader = make_reader(open(path, "rb"), decoder_factories=[DecoderFactory()])
summ = reader.get_summary()

print("=== channels (topic [type] msgs) ===")
schemas = {s.id: s for s in summ.schemas.values()}
counts = summ.statistics.channel_message_counts if summ.statistics else {}
for cid, ch in summ.channels.items():
    tname = schemas[ch.schema_id].name if ch.schema_id in schemas else "?"
    print(f"  {ch.topic:45s} [{tname:40s}] {counts.get(cid, '?')}")

WANT = ["/zed/zed_node/rgb/color/rect/image",
        "/zed/zed_node/depth/depth_registered",
        "/zed/zed_node/rgb/color/rect/camera_info",
        "/zed/zed_node/depth/camera_info",
        "/localization/pose", "/zed/zed_node/odom",
        "/tf_static", "/tf"]
seen = collections.Counter()
tf_edges = set()
print("\n=== samples ===")
for schema, channel, message, ros in reader.iter_decoded_messages(topics=WANT):
    t = channel.topic
    if t in ("/tf", "/tf_static"):
        for tr in ros.transforms:
            tf_edges.add((tr.header.frame_id, tr.child_frame_id))
    if seen[t] == 0:
        try:
            if t.endswith("/image") or "depth_registered" in t:
                print(f"[{t}] encoding={ros.encoding} {ros.height}x{ros.width} step={ros.step} frame_id={ros.header.frame_id}")
            elif "camera_info" in t:
                k = list(ros.k)
                print(f"[{t}] {ros.height}x{ros.width} fx={k[0]:.1f} fy={k[4]:.1f} cx={k[2]:.1f} cy={k[5]:.1f} frame_id={ros.header.frame_id}")
            elif t == "/localization/pose":
                p = ros.pose.pose.position
                print(f"[{t}] frame_id={ros.header.frame_id} pos=({p.x:.2f},{p.y:.2f},{p.z:.2f})")
            elif t == "/zed/zed_node/odom":
                print(f"[{t}] frame_id={ros.header.frame_id} child_frame_id={ros.child_frame_id}")
        except Exception as e:
            print(f"[{t}] sample error: {e}")
    seen[t] += 1
    if all(seen[x] for x in WANT if x != "/tf") and seen["/tf"] > 400:
        break

print("\n=== TF tree (parent -> child) ===")
for a, b in sorted(tf_edges):
    print(f"  {a} -> {b}")
print(f"\nsampled: {dict(seen)}")
