#!/usr/bin/env python3
"""Infer the robot platform + onboard stack from an .mcap: writer library, all message
schema names (vendor packages are the giveaway), every topic, the full TF frame tree,
any URDF on /robot_description, and the LiDAR point format.
    python realbot/robot_forensics.py <chunk.mcap>"""
import sys, re
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

MCAP = sys.argv[1]
r = make_reader(open(MCAP, "rb"), decoder_factories=[DecoderFactory()])
h = r.get_header()
print(f"== mcap writer ==\n  profile={h.profile!r}  library={h.library!r}\n")

summ = r.get_summary()
schemas = {s.id: s for s in summ.schemas.values()}
chans = summ.channels.values()
print("== all topics + message type ==")
for c in sorted(chans, key=lambda c: c.topic):
    sc = schemas.get(c.schema_id)
    print(f"  {c.topic:44s} {sc.name if sc else '?':38s} enc={c.message_encoding}")
    if c.metadata:
        print(f"        metadata: {dict(c.metadata)}")

print("\n== distinct schema names (vendor packages reveal the robot) ==")
names = sorted({s.name for s in schemas.values()})
for n in names:
    print("  ", n)
vendor = [n for n in names if not n.split("/")[0] in {
    "std_msgs", "sensor_msgs", "geometry_msgs", "nav_msgs", "tf2_msgs", "builtin_interfaces",
    "diagnostic_msgs", "visualization_msgs", "rosgraph_msgs", "rcl_interfaces", "action_msgs",
    "std_srvs", "lifecycle_msgs", "statistics_msgs", "unique_identifier_msgs", "shape_msgs",
    "stereo_msgs", "trajectory_msgs", "actionlib_msgs"}]
print("\n  >> NON-STANDARD (vendor) packages:", sorted({n.split('/')[0] for n in vendor}) or "none")

# TF frame tree + any URDF
frames, urdf = set(), None
for schema, ch, msg, ros in r.iter_decoded_messages(topics=["/tf", "/tf_static", "/robot_description"]):
    if ch.topic == "/robot_description":
        urdf = ros.data if urdf is None else urdf
    else:
        for t in ros.transforms:
            frames.add((t.header.frame_id, t.child_frame_id))
print("\n== TF frames (child links name the robot's body plan) ==")
for a, b in sorted(frames):
    print(f"  {a} -> {b}")

if urdf:
    print("\n== /robot_description URDF ==")
    m = re.search(r'<robot[^>]*\bname="([^"]+)"', urdf)
    print("  robot name attr:", m.group(1) if m else "?")
    for kw in ("unitree", "deeprobotics", "jueying", "go1", "go2", "b1", "b2", "aliengo",
               "lite3", "x20", "x30", "anymal", "spot", "cyberdog", "aselsan", "magic"):
        if kw in urdf.lower():
            print(f"  URDF mentions: {kw}")
else:
    print("\n== no /robot_description topic recorded ==")

# LiDAR point format
for schema, ch, msg, ros in r.iter_decoded_messages(topics=["/rslidar_points"]):
    fields = [(f.name, f.offset, f.datatype) for f in ros.fields]
    print(f"\n== RoboSense /rslidar_points ==\n  frame={ros.header.frame_id} {ros.width}x{ros.height} "
          f"point_step={ros.point_step} fields={fields}")
    break
