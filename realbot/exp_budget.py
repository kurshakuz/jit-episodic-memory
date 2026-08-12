#!/usr/bin/env python3
"""
E2: frame-efficiency budget curve on real data. JIT runs the detector only on the top-k
CLIP-retrieved keyframes. We test the paper's central claim on real footage: does CLIP
ranking surface the object-bearing frames early, so localization converges at a far
smaller detector budget than random frame selection?

Runs OWL-ViT once on every keyframe (caching each frame's best detection + back-projected
map point), then reconstructs, for each budget k, the localization JIT would produce from
(a) the top-k CLIP-ranked frames vs (b) random k-frame subsets. Scores against the LiDAR
ground-truth centroid from accuracy_<trace>.json.

    REALBOT_TRACE=realbot/trace python realbot/exp_budget.py "car"
"""
import os, sys, json
from pathlib import Path
import numpy as np
from PIL import Image
import torch, open_clip
from sklearn.cluster import DBSCAN
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval.level3_verification import OWLViTDetector

T = Path(os.environ.get("REALBOT_TRACE", str(Path(__file__).resolve().parent / "trace")))
OUT = Path(os.environ.get("REALBOT_OUT", str(Path(__file__).resolve().parent / "_out"))) / "experiments"
OUT.mkdir(parents=True, exist_ok=True)
QUERY = sys.argv[1] if len(sys.argv) > 1 else "car"
meta = json.load(open(T / "meta.json")); kfs = meta["keyframes"]
fx, fy, cx, cy = (meta["intrinsics"][k] for k in ("fx", "fy", "cx", "cy"))
poses = np.load(T / "poses_map.npy")
gt = json.load(open(OUT / f"accuracy_{T.name}.json"))[QUERY]["gt_centroid"]
gt = np.array(gt[:2])

# CLIP ranking over all keyframes
model, _, pre = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="laion400m_e32")
tok = open_clip.get_tokenizer("ViT-B-32-quickgelu"); model = model.eval().cuda()
embs = np.load(T / "embeddings.npy")
with torch.no_grad():
    te = model.encode_text(tok([QUERY]).cuda()); te = (te / te.norm(dim=-1, keepdim=True)).cpu().numpy()[0]
clip_score = embs @ te
clip_order = list(np.argsort(-clip_score))          # best-first

# one detector pass over every keyframe: cache best detection -> back-projected map point
det = OWLViTDetector(device="cuda", score_threshold=0.1)
pt = {}                                              # kf -> map xy of the detection (if valid)
for i, kf in enumerate(kfs):
    rgb = np.array(Image.open(T / kf["image"]).convert("RGB")); H, W = rgb.shape[:2]
    ds = det.detect(rgb, [QUERY])[:1]
    if not ds or ds[0].score < 0.12:
        continue
    x1, y1, x2, y2 = ds[0].bbox; u, v = int((x1 + x2) / 2 * W), int((y1 + y2) / 2 * H)
    depth = np.load(T / kf["depth"]).astype(np.float32)
    reg = depth[max(0, v - 7):v + 7, max(0, u - 7):u + 7]
    val = reg[np.isfinite(reg) & (reg > 0.3) & (reg < 40)]
    if val.size < 4:
        continue
    Z = float(np.median(val)); X = (u - cx) * Z / fx; Y = (v - cy) * Z / fy
    pw = (poses[i] @ np.array([X, Y, Z, 1.0]))[:3]
    pt[i] = pw[:2]
    if (i + 1) % 100 == 0:
        print(f"  detector pass {i+1}/{len(kfs)}  ({len(pt)} hits so far)")
print(f"detector hits: {len(pt)}/{len(kfs)} keyframes contain '{QUERY}'")


def localize(frames):
    """map-frame localization JIT would output from this frame subset (dominant cluster)."""
    P = np.array([pt[i] for i in frames if i in pt])
    if len(P) < 2:
        return None, len(P)
    lab = DBSCAN(eps=2.0, min_samples=2).fit(P).labels_
    if (lab >= 0).sum() == 0:
        return None, len(P)
    best = max((l for l in set(lab) if l >= 0), key=lambda l: (lab == l).sum())
    return P[lab == best].mean(0), len(P)


KS = [1, 2, 3, 5, 8, 12, 16, 20, 30, 40, 50, 60]
rng = np.random.RandomState(0)
curve = []
for k in KS:
    c_pt, c_n = localize(clip_order[:k])
    c_err = float(np.linalg.norm(c_pt - gt)) if c_pt is not None else None
    r_errs, r_ns, r_found = [], [], 0
    for _ in range(30):
        sub = rng.choice(len(kfs), k, replace=False)
        r_pt, r_n = localize(sub)
        r_ns.append(r_n)
        if r_pt is not None:
            e = float(np.linalg.norm(r_pt - gt)); r_errs.append(e); r_found += (e < 0.5)
    curve.append({"k": k, "clip_err": c_err, "clip_hits": c_n,
                  "rand_err_median": (round(float(np.median(r_errs)), 3) if r_errs else None),
                  "rand_hits_mean": round(float(np.mean(r_ns)), 2),
                  "rand_localized_frac": round(len(r_errs) / 30, 2),
                  "rand_within0.5_frac": round(r_found / 30, 2)})
    ce = f"{c_err:.2f}" if c_err is not None else "  -"
    rm = f"{np.median(r_errs):.2f}" if r_errs else "  -"
    print(f"k={k:2d}  CLIP: err={ce}m hits={c_n:2d}   RANDOM: err(med)={rm}m hits={np.mean(r_ns):.1f} localized={len(r_errs)}/30")

json.dump({"query": QUERY, "trace": T.name, "gt_centroid": gt.tolist(),
           "n_object_frames": len(pt), "n_keyframes": len(kfs), "curve": curve},
          open(OUT / f"budget_{T.name}_{QUERY}.json", "w"), indent=1)
print(f"-> {OUT / f'budget_{T.name}_{QUERY}.json'}")
