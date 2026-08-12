#!/usr/bin/env python3
"""
JIT episodic-memory query on the real robot trace.
CLIP retrieve (cached embeddings) -> OWL-ViT detect -> back-project registered ZED
depth -> transform to world frame (map|odom) -> x-y DBSCAN (ground objects) ->
ranked 3D instances. Writes result_<q>_<frame>.json + a boxed verification montage.

    python realbot/run_query.py "<query>" [map|odom] [k]
"""
import sys, json, os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch, open_clip
from sklearn.cluster import DBSCAN
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval.level3_verification import OWLViTDetector

T = Path(os.environ.get("REALBOT_TRACE", str(Path(__file__).resolve().parent / "trace")))
OUT = Path(os.environ.get("REALBOT_OUT", str(Path(__file__).resolve().parent / "_out")))
OUT.mkdir(parents=True, exist_ok=True)
QUERY = sys.argv[1] if len(sys.argv) > 1 else "car"
FRAME = sys.argv[2] if len(sys.argv) > 2 else "map"
K = int(sys.argv[3]) if len(sys.argv) > 3 else 60

meta = json.load(open(T / "meta.json")); kfs = meta["keyframes"]
fx, fy, cx, cy = (meta["intrinsics"][k] for k in ("fx", "fy", "cx", "cy"))
poses = np.load(T / ("poses_map.npy" if FRAME == "map" else "poses.npy"))

model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="laion400m_e32")
tok = open_clip.get_tokenizer("ViT-B-32-quickgelu"); model = model.eval().cuda()
emb_f = T / "embeddings.npy"
if emb_f.exists():
    embs = np.load(emb_f)
else:
    embs = np.zeros((len(kfs), 512), np.float32)
    with torch.no_grad():
        for i, kf in enumerate(kfs):
            im = preprocess(Image.open(T / kf["image"]).convert("RGB")).unsqueeze(0).cuda()
            e = model.encode_image(im)
            embs[i] = (e / e.norm(dim=-1, keepdim=True)).cpu().numpy()[0]
    np.save(emb_f, embs)
with torch.no_grad():
    te = model.encode_text(tok([QUERY]).cuda()); te = (te / te.norm(dim=-1, keepdim=True)).cpu().numpy()[0]
order = np.argsort(-(embs @ te))[:K]

det = OWLViTDetector(device="cuda", score_threshold=0.1)
pts, members = [], []
for idx in order:
    kf = kfs[idx]; rgb = np.array(Image.open(T / kf["image"]).convert("RGB")); H, W = rgb.shape[:2]
    depth = None
    for d in det.detect(rgb, [QUERY])[:2]:
        if d.score < 0.12:
            break
        x1, y1, x2, y2 = d.bbox; u, v = int((x1 + x2) / 2 * W), int((y1 + y2) / 2 * H)
        if depth is None:
            depth = np.load(T / kf["depth"]).astype(np.float32)
        reg = depth[max(0, v - 7):v + 7, max(0, u - 7):u + 7]
        valid = reg[np.isfinite(reg) & (reg > 0.3) & (reg < 40)]
        if valid.size < 4:
            continue
        Z = float(np.median(valid)); X = (u - cx) * Z / fx; Y = (v - cy) * Z / fy
        pw = (poses[idx] @ np.array([X, Y, Z, 1.0]))[:3]
        pts.append(pw)
        members.append({"kf": int(idx), "bbox": [float(b) for b in d.bbox], "score": float(d.score),
                        "xyz": [float(x) for x in pw]})
pts = np.array(pts)
print(f"[{QUERY}|{FRAME}] top-{K} frames -> {len(pts)} detections back-projected")

instances = []
if len(pts) >= 2:
    labels = DBSCAN(eps=2.0, min_samples=2).fit(pts[:, :2]).labels_
    clus = {}
    for l, m in zip(labels, members):
        if l >= 0:
            clus.setdefault(l, []).append(m)
    for grp in sorted(clus.values(), key=lambda v: -len(v)):
        P = np.array([m["xyz"] for m in grp])
        c = [float(P[:, 0].mean()), float(P[:, 1].mean()), float(np.median(P[:, 2]))]
        spread = float(np.linalg.norm(P[:, :2] - np.array(c[:2]), axis=1).mean())
        instances.append({"centroid": [round(x, 2) for x in c], "n_views": len(grp),
                          "spread_m": round(spread, 2), "members": grp})
    print(f"{len(instances)} '{QUERY}' instance(s) [{FRAME}]:")
    for j, ins in enumerate(instances[:8]):
        c = ins["centroid"]
        print(f"  #{j+1}: {ins['n_views']:2d} views  ({c[0]:7.1f},{c[1]:7.1f},{c[2]:5.1f})m  xy-spread {ins['spread_m']}m")
json.dump({"query": QUERY, "frame": FRAME, "k": K, "n_detections": len(pts), "instances": instances},
          open(T / f"result_{QUERY}_{FRAME}.json", "w"), indent=1)

if instances:
    tiles = []
    for m in instances[0]["members"][:6]:
        im = Image.open(T / kfs[m["kf"]]["image"]).convert("RGB"); W0, H0 = im.size
        b = m["bbox"]
        ImageDraw.Draw(im).rectangle([b[0] * W0, b[1] * H0, b[2] * W0, b[3] * H0], outline=(20, 230, 120), width=4)
        tiles.append(im.resize((320, 200)))
    cols = 3; rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (320 * cols, 200 * rows), "black")
    for t, im in enumerate(tiles):
        sheet.paste(im, ((t % cols) * 320, (t // cols) * 200))
    sheet.save(OUT / f"{QUERY}_verify.png")
    print(f"montage -> {OUT / (QUERY + '_verify.png')}")
