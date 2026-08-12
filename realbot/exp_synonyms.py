#!/usr/bin/env python3
"""
E5: open-vocabulary synonym robustness. If JIT is genuinely grounding meaning (not
memorising a string), synonyms for the same object should localize to the same 3D point.
Runs each synonym through the full pipeline (CLIP retrieve -> OWL-ViT -> back-project ->
cluster) and reports, per synonym group, how far apart the dominant-instance centroids land.

    python realbot/exp_synonyms.py     (models load once; queries the c0/c3/c4 traces)
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch, open_clip
from sklearn.cluster import DBSCAN
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval.level3_verification import OWLViTDetector

R = Path(__file__).resolve().parent
OUT = R / "_out" / "experiments"; OUT.mkdir(parents=True, exist_ok=True)
# groups of synonyms, each tied to a trace
GROUPS = [
    ("trace",    "car",     ["car", "vehicle", "automobile", "sedan"]),
    ("trace_c3", "bench",    ["bench", "seat", "wooden bench"]),
    ("trace_c3", "gray suv", ["gray suv", "grey suv", "silver suv", "gray car"]),
    ("trace_c4", "bicycle",  ["bicycle", "bike", "a bicycle"]),
]
model, _, pre = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="laion400m_e32")
tok = open_clip.get_tokenizer("ViT-B-32-quickgelu"); model = model.eval().cuda()
det = OWLViTDetector(device="cuda", score_threshold=0.1)


def localize(trace, query, K=60):
    T = R / trace
    meta = json.load(open(T / "meta.json")); kfs = meta["keyframes"]
    fx, fy, cx, cy = (meta["intrinsics"][k] for k in ("fx", "fy", "cx", "cy"))
    poses = np.load(T / "poses_map.npy"); embs = np.load(T / "embeddings.npy")
    with torch.no_grad():
        te = model.encode_text(tok([query]).cuda()); te = (te / te.norm(dim=-1, keepdim=True)).cpu().numpy()[0]
    order = np.argsort(-(embs @ te))[:K]
    pts = []
    for idx in order:
        kf = kfs[idx]; rgb = np.array(Image.open(T / kf["image"]).convert("RGB")); H, W = rgb.shape[:2]
        ds = det.detect(rgb, [query])[:1]
        if not ds or ds[0].score < 0.12:
            continue
        x1, y1, x2, y2 = ds[0].bbox; u, v = int((x1 + x2) / 2 * W), int((y1 + y2) / 2 * H)
        depth = np.load(T / kf["depth"]).astype(np.float32)
        reg = depth[max(0, v - 7):v + 7, max(0, u - 7):u + 7]
        val = reg[np.isfinite(reg) & (reg > 0.3) & (reg < 40)]
        if val.size < 4:
            continue
        Z = float(np.median(val)); X = (u - cx) * Z / fx; Y = (v - cy) * Z / fy
        pts.append((poses[idx] @ np.array([X, Y, Z, 1.0]))[:3])
    P = np.array(pts)
    if len(P) < 2:
        return None, 0
    lab = DBSCAN(eps=2.0, min_samples=2).fit(P[:, :2]).labels_
    if (lab >= 0).sum() == 0:
        return None, len(P)
    best = max((l for l in set(lab) if l >= 0), key=lambda l: (lab == l).sum())
    return P[lab == best][:, :2].mean(0), int((lab == best).sum())


results = {}
for trace, name, syns in GROUPS:
    cents = {}
    for q in syns:
        c, n = localize(trace, q)
        cents[q] = (c, n)
        print(f"  [{name}] '{q}': {'('+', '.join(f'{x:.1f}' for x in c)+')m, '+str(n)+' views' if c is not None else 'no localization'}")
    valid = {q: c for q, (c, n) in cents.items() if c is not None}
    if len(valid) >= 2:
        ref = valid[syns[0]] if syns[0] in valid else list(valid.values())[0]
        dists = {q: round(float(np.linalg.norm(c - ref)), 2) for q, c in valid.items()}
        maxd = max(dists.values())
        results[name] = {"centroids": {q: [round(float(x), 2) for x in c] for q, c in valid.items()},
                         "dist_to_ref_m": dists, "max_pairwise_m": maxd}
        print(f"{name}: synonyms localize within {maxd} m of each other  {dists}\n")
json.dump(results, open(OUT / "synonyms.json", "w"), indent=1)
print("->", OUT / "synonyms.json")
