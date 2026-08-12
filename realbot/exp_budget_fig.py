#!/usr/bin/env python3
"""Figure for E2: localization error vs detector budget k, CLIP-ranked vs random frame
selection, from the saved budget_*.json curves. Log-y because random localizes the wrong
instance (tens of metres) while CLIP locks onto the queried one within ~1 m at small k."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "_out" / "experiments"
PANELS = [("budget_trace_car.json", "“car”  (chunk 0, 100 m car-lined underpass)"),
          ("budget_trace_c3_white car.json", "“white car”  (chunk 3, parked row)")]
fig, axes = plt.subplots(1, len(PANELS), figsize=(11.5, 4.3))
for ax, (fn, title) in zip(axes, PANELS):
    d = json.load(open(OUT / fn)); c = d["curve"]
    ks = [p["k"] for p in c]
    clip = [p["clip_err"] for p in c]
    rand = [p["rand_err_median"] for p in c]
    kk = [k for k, e in zip(ks, clip) if e is not None]; cc = [e for e in clip if e is not None]
    kr = [k for k, e in zip(ks, rand) if e is not None]; rr = [e for e in rand if e is not None]
    ax.plot(kk, cc, "o-", color="#12b3a6", lw=2.2, label="JIT: CLIP-ranked budget")
    ax.plot(kr, rr, "s--", color="#8b90a3", lw=1.8, label="random frame budget")
    ax.axhline(1.0, color="#f0873a", lw=1, ls=":", alpha=.8); ax.text(ks[-1], 1.06, "1 m", color="#f0873a", ha="right", fontsize=9)
    ax.set_yscale("log"); ax.set_xlabel("detector budget  k  (frames scored)")
    ax.set_ylabel("localization error vs LiDAR GT (m)")
    ax.set_title(title, fontsize=11); ax.grid(alpha=.3, which="both"); ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(OUT / "budget_curve.png", dpi=120)
print("->", OUT / "budget_curve.png")

# headline: detector calls to first reach <1 m
for fn, _ in PANELS:
    d = json.load(open(OUT / fn)); c = d["curve"]
    kc = next((p["k"] for p in c if p["clip_err"] is not None and p["clip_err"] < 1.0), None)
    kr = next((p["k"] for p in c if p["rand_err_median"] is not None and p["rand_err_median"] < 1.0), None)
    print(f"{d['query']:10s}: CLIP reaches <1 m at k={kc}; random median <1 m at k={kr}  "
          f"(object in {d['n_object_frames']}/{d['n_keyframes']} frames)")
