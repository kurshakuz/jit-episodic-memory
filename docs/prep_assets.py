#!/usr/bin/env python3
"""
Prepare static assets for the JIT project page (docs/).

Emits small chart-ready JSON into docs/static/data/ and copies + downscales a
curated figure set into docs/static/images/. Runs in the `jit` conda env (needs
Pillow). No model inference here; the interactive query-demo and footage are baked
separately by docs/make_footage.py.

Two data sources:
  * budget curve + scalability are DISTILLED from the raw result artifacts.
  * the small headline tables (ScanNet Table II, cross-dataset Table V, build-cost
    crossover, ranked recall) are the VERIFIED paper values, embedded as constants
    with their source noted. They were cross-checked against the artifacts /
    audit_headline_numbers.py; the raw all_aggregated.json mixes datasets and is
    NOT used for these.
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # repo root
PR = ROOT / "outputs" / "paper_results"
DOCS = ROOT / "docs"
DATA = DOCS / "static" / "data"
IMG = DOCS / "static" / "images"
DATA.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


def dump(name, obj):
    p = DATA / name
    p.write_text(json.dumps(obj, indent=1))
    print(f"  wrote {p.relative_to(ROOT)}  ({p.stat().st_size} B)")


# --------------------------------------------------------------------------
# 1. Budget curve (HM3D): Loc@Xm vs detector budget k, JIT vs equal-budget random
#    Source: outputs/paper_results/budget_curve_canonical.json (per-query booleans)
# --------------------------------------------------------------------------
def build_budget():
    d = json.load(open(PR / "budget_curve_canonical.json"))
    ks = d["k_values"]
    out = {"k": ks, "note": d.get("dataset", "HM3D"), "arms": {}}
    for arm in ("jit", "random"):
        loc1, loc05 = [], []
        for k in ks:
            rows = [r for r in d["records"] if r["arm"] == arm and r["k"] == k]
            n = len(rows) or 1
            loc1.append(round(100 * sum(bool(r["loc_1m"]) for r in rows) / n, 1))
            loc05.append(round(100 * sum(bool(r["loc_0.5m"]) for r in rows) / n, 1))
        out["arms"][arm] = {"loc_1m": loc1, "loc_0.5m": loc05}
    dump("budget.json", out)
    print(f"    check: jit@k100 Loc@1m={out['arms']['jit']['loc_1m'][ks.index(100)]} "
          f"(expect ~78.4), random@k100={out['arms']['random']['loc_1m'][ks.index(100)]} (expect ~67.2)")


# --------------------------------------------------------------------------
# 2. Scalability: accuracy + latency vs frame count, JIT (flat) vs brute force
#    Source: outputs/paper_results/scalability_full.json
# --------------------------------------------------------------------------
def build_scalability():
    d = json.load(open(PR / "scalability_full.json"))
    frames = [160, 500, 2500]
    methods = {"jit": "jit", "bf100": "bf100", "bf_all": "bf_all"}
    out = {"frames": frames, "series": {}}
    for key, pref in methods.items():
        acc, lat = [], []
        for f in frames:
            e = d[f"{pref}_{f}"]
            acc.append(round(e["accuracy"]["1.0"], 1))
            lat.append(round(e["latency_mean_ms"] / 1000.0, 2))   # -> seconds
        out["series"][key] = {"acc_1m": acc, "latency_s": lat}
    dump("scalability.json", out)
    print(f"    check: jit acc {out['series']['jit']['acc_1m']}, "
          f"bf_all latency_s {out['series']['bf_all']['latency_s']} (expect ~38.5 at 2500)")


# --------------------------------------------------------------------------
# 3. ScanNet headline table (Table II) — VERIFIED paper values.
#    Loc@1m / build(s) / storage(MB own artifact) / query(ms).
# --------------------------------------------------------------------------
RESULTS_SCANNET = {
    "note": "ScanNet v2, 141 scenes, 457 queries. Loc@1m macro. Storage = each method's own "
            "built artifact (shared RGB-D keyframes excluded from all rows).",
    "methods": [
        {"name": "JIT (L1+L2)", "loc_1m": 79.1, "build_s": 3.1, "storage_mb": 1.0,
         "query_ms": 2500, "lazy": True, "retained_mb": 375},
        {"name": "JIT (+L3)", "loc_1m": 81.3, "build_s": 3.1, "storage_mb": 1.0,
         "query_ms": 2500, "lazy": True, "retained_mb": 375},
        {"name": "ConceptFusion", "loc_1m": 74.1, "build_s": 286, "storage_mb": 250, "query_ms": 25},
        {"name": "VLMaps", "loc_1m": 71.8, "build_s": 47, "storage_mb": 490, "query_ms": 48},
        {"name": "GOAT", "loc_1m": 71.7, "build_s": 27, "storage_mb": 58, "query_ms": 1},
        {"name": "ConceptGraphs", "loc_1m": 70.3, "build_s": 2160, "storage_mb": 7, "query_ms": 10},
    ],
}

# --------------------------------------------------------------------------
# 4. Cross-dataset generalization (Table V) — VERIFIED paper values, Loc@1m.
# --------------------------------------------------------------------------
CROSS_DATASET = {
    "columns": ["JIT (L1+L2)", "JIT (+L3)", "Random-100+DBSCAN", "ConceptGraphs", "GOAT"],
    "rows": [
        {"dataset": "ScanNet",     "scenes": 141, "vals": [79.1, 81.3, 72.8, 70.3, 71.7]},
        {"dataset": "HM3D",        "scenes": 36,  "vals": [78.4, None, 67.2, 6.1, 55.4]},
        {"dataset": "ARKitScenes", "scenes": 12,  "vals": [71.5, 73.6, 71.7, 66.9, 55.8]},
        {"dataset": "Replica",     "scenes": 8,   "vals": [38.3, 25.7, 35.4, 26.0, 31.5]},
    ],
}

# --------------------------------------------------------------------------
# 5. Build-cost crossover (wall-clock) — VERIFIED. Cumulative time = build + n*query.
#    JIT: build 3.1s, query 2.5s. Eager query times from Table II.
# --------------------------------------------------------------------------
CROSSOVER = {
    "jit": {"build_s": 3.1, "query_s": 2.5},
    "eager": [
        {"name": "ConceptGraphs", "build_s": 2160, "query_s": 0.010, "crossover": 860},
        {"name": "ConceptFusion", "build_s": 286,  "query_s": 0.025, "crossover": 114},
        {"name": "VLMaps",        "build_s": 47,   "query_s": 0.048, "crossover": 18},
    ],
    "note": "Queries per scene at which JIT's cumulative wall-clock cost overtakes an eager "
            "map. Cumulative energy crossover is far later (CG ~18k-28k queries).",
}

# --------------------------------------------------------------------------
# Figures: copy + downscale a curated set.
# --------------------------------------------------------------------------
FIGURES = {
    "architecture.png": "paper/images/fig_architecture_v2.png",
    "spatial_denoiser.png": "paper/images/fig4_spatial_denoiser.png",
    "qualitative.png": "paper/images/fig_qualitative.png",
}
MAX_W = 1200


def copy_figures():
    from PIL import Image
    for dst, src in FIGURES.items():
        s = ROOT / src
        if not s.exists():
            print(f"  MISSING {src}"); continue
        im = Image.open(s)
        if im.width > MAX_W:
            h = round(im.height * MAX_W / im.width)
            im = im.resize((MAX_W, h), Image.LANCZOS)
        if im.mode in ("RGBA", "P"):
            im = im.convert("RGB")
        out = IMG / dst
        im.save(out, optimize=True)
        print(f"  fig {dst}  {out.stat().st_size} B  ({im.width}x{im.height})")
    # paper PDF
    pdf = ROOT / "paper" / "main.pdf"
    if pdf.exists():
        shutil.copy(pdf, DOCS / "static" / "jit_paper.pdf")
        print(f"  copied jit_paper.pdf ({pdf.stat().st_size} B)")


if __name__ == "__main__":
    print("distilling data ->", DATA.relative_to(ROOT))
    build_budget()
    build_scalability()
    dump("results_scannet.json", RESULTS_SCANNET)
    dump("cross_dataset.json", CROSS_DATASET)
    dump("crossover.json", CROSSOVER)
    print("copying figures ->", IMG.relative_to(ROOT))
    copy_figures()
    print("done.")
