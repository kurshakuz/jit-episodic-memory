#!/usr/bin/env python3
"""One glanceable figure of the 'more interesting than car' real-robot JIT queries:
best instance per query, its top-scoring keyframe with the OWL-ViT box, labeled with
query, #views, xy-spread, and map-frame centroid. Sources the committed result_*.json."""
import os, json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

R = Path(__file__).resolve().parent
OUT = Path(os.environ.get("REALBOT_OUT", str(R / "_out")))
OUT.mkdir(parents=True, exist_ok=True)
# (trace, query result file, display label, scene)
ITEMS = [
    ("trace_c4", "result_bicycle_map.json", "bicycle", "indoor corridor"),
    ("trace_c3", "result_white car_map.json", "white car", "parking row"),
    ("trace_c3", "result_gray suv_map.json", "gray suv", "parking row"),
    ("trace_c3", "result_bench_map.json", "bench", "building wall"),
]
TW, TH = 380, 238


def font(sz, b=False):
    p = f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if b else ''}.ttf"
    return ImageFont.truetype(p, sz)


tiles = []
for trace, rf, label, scene in ITEMS:
    T = R / trace
    kfs = json.load(open(T / "meta.json"))["keyframes"]
    res = json.load(open(T / rf))
    inst = max(res["instances"], key=lambda i: i["n_views"])       # dominant instance
    m = sorted(inst["members"], key=lambda m: -m["score"])[0]      # crispest view
    im = Image.open(T / kfs[m["kf"]]["image"]).convert("RGB").resize((TW, TH))
    d = ImageDraw.Draw(im)
    b = m["bbox"]
    d.rectangle([b[0] * TW, b[1] * TH, b[2] * TW, b[3] * TH], outline=(20, 230, 120), width=4)
    c = inst["centroid"]
    d.rectangle([0, 0, TW, 44], fill=(9, 11, 24))
    d.text((8, 4), f'"{label}"', font=font(19, True), fill=(255, 255, 255))
    d.text((TW - 120, 8), scene, font=font(12), fill=(180, 180, 190))
    d.text((8, 26), f'{inst["n_views"]} views · spread {inst["spread_m"]} m · map ({c[0]:.0f}, {c[1]:.0f}) m',
           font=font(12), fill=(150, 200, 255))
    tiles.append(im)

cols = 2
rows = (len(tiles) + cols - 1) // cols
G = 6
sheet = Image.new("RGB", (TW * cols + G * (cols + 1), TH * rows + G * (rows + 1)), (24, 26, 36))
for t, im in enumerate(tiles):
    x = G + (t % cols) * (TW + G)
    y = G + (t // cols) * (TH + G)
    sheet.paste(im, (x, y))
out = OUT / "interesting_summary.png"
sheet.save(out)
print("saved", out, sheet.size)
