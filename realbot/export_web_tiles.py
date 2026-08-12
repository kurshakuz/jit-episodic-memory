#!/usr/bin/env python3
"""Export web-sized boxed query tiles for the project page's real-robot section.
For each query: the best 3D instance's crispest keyframe with the OWL-ViT box drawn,
downscaled to 640 px wide -> docs/static/images/realbot/<slug>.jpg. Sources the
committed result_*.json in the (uncommitted) traces, so the images are the release
artifact; the traces are not needed to serve the page."""
import json
from pathlib import Path
from PIL import Image, ImageDraw

R = Path(__file__).resolve().parent
OUT = R.parents[0] / "docs" / "static" / "images" / "realbot"
OUT.mkdir(parents=True, exist_ok=True)
W = 640
# slug, trace, result file  (labels chosen to match the object actually present)
ITEMS = [
    ("bicycle", "trace_c4", "result_bicycle_map.json"),
    ("bench", "trace_c3", "result_bench_map.json"),
    ("white_car", "trace_c3", "result_white car_map.json"),
    ("gray_suv", "trace_c3", "result_gray suv_map.json"),
]
for slug, trace, rf in ITEMS:
    T = R / trace
    kfs = json.load(open(T / "meta.json"))["keyframes"]
    res = json.load(open(T / rf))
    # the dominant (most-viewed) instance — the one the verification montage shows
    inst = max(res["instances"], key=lambda i: i["n_views"])
    m = sorted(inst["members"], key=lambda m: -m["score"])[0]
    im = Image.open(T / kfs[m["kf"]]["image"]).convert("RGB")
    W0, H0 = im.size
    b = m["bbox"]
    ImageDraw.Draw(im).rectangle([b[0] * W0, b[1] * H0, b[2] * W0, b[3] * H0], outline=(20, 230, 120), width=5)
    im = im.resize((W, int(W * H0 / W0)))
    im.save(OUT / f"{slug}.jpg", quality=86)
    c = inst["centroid"]
    print(f"{slug:10s} {inst['n_views']:3d} views  spread {inst['spread_m']}m  map({c[0]:.0f},{c[1]:.0f}) -> {slug}.jpg")
print("->", OUT)
