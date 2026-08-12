#!/usr/bin/env python3
"""
E6: the whole run as one drift-free episodic memory. Draws the full six-chunk ~1 km
campus loop in the map frame (each chunk a coloured segment) and overlays every
query-localized object at its map centroid. This is what the map frame buys that odometry
cannot: a single globally consistent memory the robot can query anywhere along the walk.

    python realbot/exp_global_map.py "<dir-with-mcaps>"
"""
import sys, json
from pathlib import Path
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent
OUT = R / "_out" / "experiments"; OUT.mkdir(parents=True, exist_ok=True)
D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
files = sorted(D.glob("*.mcap"))
if not files:
    sys.exit(f"no .mcap files in {D} — pass the recording directory as the first argument")

# full map-frame trajectory per chunk from /localization/pose
trajs = []
for f in files:
    P = []
    for s, ch, m, ros in make_reader(open(f, "rb"), decoder_factories=[DecoderFactory()]).iter_decoded_messages(topics=["/localization/pose"]):
        P.append((ros.pose.pose.position.x, ros.pose.pose.position.y))
    trajs.append(np.array(P))
total = sum(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)) for P in trajs if len(P) > 1)

# verified objects: dominant map centroid per query (privacy: no people)
OBJS = [("trace", "car"), ("trace", "tree"), ("trace", "building"), ("trace", "bush"),
        ("trace", "column"), ("trace", "lamp post"),
        ("trace_c1", "manhole cover"), ("trace_c1", "stop sign"), ("trace_c1", "hedge"),
        ("trace_c2", "stairs"),
        ("trace_c3", "bench"), ("trace_c3", "white car"), ("trace_c3", "gray suv"),
        ("trace_c4", "bicycle")]
objs = []
for trace, q in OBJS:
    p = R / trace / f"result_{q}_map.json"
    if not p.exists():
        continue
    res = json.load(open(p))
    if not res["instances"]:
        continue
    inst = max(res["instances"], key=lambda i: i["n_views"])
    objs.append((q, inst["centroid"][0], inst["centroid"][1]))

# committed decimated artifact for the paper figure (decouples the paper build from the
# private .mcap recordings): per-chunk polylines (every 10th pose) + the object centroids.
paper_dir = R.parents[0] / "paper"
if paper_dir.exists():   # only when the (private) paper tree is present locally
    out = paper_dir / "real_robot"
    out.mkdir(parents=True, exist_ok=True)
    paper_data = out / "global_map_data.json"
    json.dump({"total_m": round(float(total), 1),
               "chunks": [np.round(P[::10], 2).tolist() for P in trajs if len(P)],
               "objects": [[q, round(float(x), 2), round(float(y), 2)] for (q, x, y) in objs]},
              open(paper_data, "w"))
    print("->", paper_data)

fig, ax = plt.subplots(figsize=(9, 8.4))
cols = plt.cm.viridis(np.linspace(0, .92, len(trajs)))
for i, (P, c) in enumerate(zip(trajs, cols)):
    if len(P):
        ax.plot(P[:, 0], P[:, 1], "-", color=c, lw=2.4, label=f"chunk {i}", alpha=.9)
offs = [(9, 5), (9, -13), (-10, 6), (7, 10), (-12, -12)]
for i, (q, x, y) in enumerate(objs):
    ax.scatter([x], [y], s=130, marker="*", color="#e0483d", edgecolor="white", zorder=5, linewidth=1)
    dx, dy = offs[i % len(offs)]
    ha = "right" if dx < 0 else "left"
    ax.annotate(q, (x, y), textcoords="offset points", xytext=(dx, dy), fontsize=9.5,
                fontweight="bold", color="#111", ha=ha)
ax.set_aspect("equal"); ax.grid(alpha=.3)
ax.set_xlabel("map x (m)"); ax.set_ylabel("map y (m)")
ax.legend(loc="upper right", fontsize=9, ncol=2)
# web version (no title — the page supplies its own heading)
plt.tight_layout(); plt.savefig(OUT / "global_map.png", dpi=120)
web = R.parents[0] / "docs" / "static" / "images" / "realbot" / "global_map.png"
web.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(web, dpi=110)
# titled version for the record
ax.set_title(f"JIT episodic memory over a {total:.0f} m campus loop  ({len(objs)} query-localized objects)", fontsize=12)
plt.tight_layout(); plt.savefig(OUT / "global_map_titled.png", dpi=120)
print(f"total path length: {total:.0f} m across {len(files)} chunks; {len(objs)} objects marked")
print("->", web)
