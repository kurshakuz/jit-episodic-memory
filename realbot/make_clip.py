#!/usr/bin/env python3
"""
Real-robot JIT rollout clip: the quadruped's ZED walk with a top-down map-frame minimap
where the trajectory draws itself and JIT's localized objects (car/tree/building/bush)
accumulate, car detections boxed on the egocentric view, and a HUD. For the page.

    python realbot/make_clip.py            (encodes realbot_rollout.mp4)
"""
import json, subprocess, tempfile, shutil, os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

R = Path(__file__).resolve().parent
T = R / "trace"
OUT = R.parents[0] / "docs" / "static" / "videos" / "realbot_rollout.mp4"
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
meta = json.load(open(T / "meta.json")); kfs = meta["keyframes"]
poses = np.load(T / "poses_map.npy")
traj = poses[:, :2, 3]
CW, CH = 960, 600           # canvas (RGB upscaled 640x400 -> 960x600)

CATS = [("car", (28, 210, 170)), ("tree", (60, 200, 90)), ("building", (150, 160, 190)),
        ("bush", (120, 170, 60))]
objs = {}            # cat -> list of centroid xy
car_boxes = {}       # kf idx -> list of bbox
for cat, _ in CATS:
    p = T / f"result_{cat}_map.json"
    if not p.exists():
        continue
    r = json.load(open(p))
    objs[cat] = [(i["centroid"][0], i["centroid"][1]) for i in r["instances"] if i["n_views"] >= 2]
    if cat == "car":
        for ins in r["instances"]:
            for m in ins["members"]:
                car_boxes.setdefault(m["kf"], []).append(m["bbox"])

# minimap geometry
allxy = np.vstack([traj] + [np.array(v) for v in objs.values() if v])
xmin, ymin = allxy.min(0) - 4; xmax, ymax = allxy.max(0) + 4
MW, MH, MARGIN = 300, 250, 16
mx0, my0 = CW - MW - MARGIN, CH - MH - MARGIN
sc = min((MW - 24) / (xmax - xmin), (MH - 24) / (ymax - ymin))
cx0, cy0 = mx0 + MW / 2, my0 + MH / 2
mcx, mcy = (xmin + xmax) / 2, (ymin + ymax) / 2


def to_map(x, y):
    return (cx0 + (x - mcx) * sc, cy0 - (y - mcy) * sc)   # flip y for image coords


def font(sz, b=False):
    p = f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if b else ''}.ttf"
    return ImageFont.truetype(p, sz) if os.path.exists(p) else ImageFont.load_default()


def panel(d, box, fill=(9, 11, 24, 205)):
    d.rounded_rectangle(box, radius=12, fill=fill)


fT, fS, fXS = font(20, True), font(15), font(12)
tmp = Path(tempfile.mkdtemp())
N = len(kfs)
for i in range(N):
    base = Image.open(T / kfs[i]["image"]).convert("RGB").resize((CW, CH))
    ov = Image.new("RGBA", (CW, CH), (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    # car detections boxed on the egocentric view
    for b in car_boxes.get(i, []):
        d.rectangle([b[0] * CW, b[1] * CH, b[2] * CW, b[3] * CH], outline=(28, 230, 180), width=4)
        d.text((b[0] * CW + 3, b[1] * CH - 16), "car", font=fXS, fill=(28, 230, 180))
    # HUD
    panel(d, (16, 14, 566, 96))
    d.text((30, 22), "JIT EPISODIC MEMORY · REAL ROBOT", font=fT, fill=(240, 242, 255))
    d.text((30, 48), "quadruped + ZED RGB-D · query-time 3D object localization", font=fXS, fill=(150, 156, 190))
    d.text((30, 68), "query “car” -> localized within 0.20 m (LiDAR-verified)", font=fS, fill=(28, 230, 180))
    # minimap
    panel(d, (mx0, my0, mx0 + MW, my0 + MH))
    d.text((mx0 + 12, my0 + 8), "episodic memory (map frame)", font=fXS, fill=(150, 156, 190))
    pts = [to_map(*traj[j]) for j in range(0, i + 1, 2)]
    if len(pts) > 1:
        d.line(pts, fill=(150, 156, 220, 200), width=2)
    for cat, col in CATS:
        for (ox, oy) in objs.get(cat, []):
            mxp, myp = to_map(ox, oy)
            d.ellipse((mxp - 4, myp - 4, mxp + 4, myp + 4), fill=col + (255,))
    ax, ay = to_map(*traj[i])
    d.ellipse((ax - 5, ay - 5, ax + 5, ay + 5), fill=(255, 255, 255, 255), outline=(28, 230, 180), width=2)
    # legend
    lx, ly = mx0 + 12, my0 + MH - 20
    for cat, col in CATS:
        d.ellipse((lx, ly, lx + 8, ly + 8), fill=col + (255,)); d.text((lx + 11, ly - 3), cat, font=fXS, fill=(200, 205, 230))
        lx += 12 + 8 + len(cat) * 7
    Image.alpha_composite(base.convert("RGBA"), ov).convert("RGB").save(tmp / f"f_{i:04d}.png")

OUT.parent.mkdir(parents=True, exist_ok=True)
subprocess.run([FFMPEG, "-y", "-framerate", "30", "-i", str(tmp / "f_%04d.png"),
                "-vf", "scale=960:-2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "25", "-movflags", "+faststart", str(OUT)], check=False)
subprocess.run([FFMPEG, "-y", "-i", str(tmp / f"f_{N//2:04d}.png"), str(OUT.with_suffix(".jpg"))], check=False)
shutil.rmtree(tmp, ignore_errors=True)
print(f"clip: {OUT} ({OUT.stat().st_size/1e6:.1f} MB, {N} frames)")
