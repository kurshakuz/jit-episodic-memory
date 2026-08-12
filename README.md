# Spatially-Grounded Just-in-Time Episodic Memory for Mobile Robots

Reference implementation of **JIT**, a three-level retrieval cascade that answers
open-vocabulary "where is the *object*?" queries by deferring 3D geometry to query
time instead of committing to a pre-built semantic map. An exploring robot stores
posed RGB-D keyframes; localization runs only when a query arrives, over a small,
fixed budget of retrieved frames.

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Status:** the accompanying paper is under review. This repository provides the
> implementation and the scripts to reproduce the reported results; the final
> citation will be added on acceptance.

---

## Why defer the map?

Eager systems (ConceptGraphs, VLMaps, ConceptFusion, GOAT) fuse every frame into a
3D map at capture time. That map costs seconds to tens of minutes to build and tens
to hundreds of megabytes to store, and it fixes the detector vocabulary before any
query is known. JIT instead keeps the raw keyframes and a lightweight index, then
localizes on demand:

- **Build:** 3.1 s to index a scene, versus 27–2,160 s for eager maps (≈9×–700× less
  pre-compute), recovered after roughly 18–860 queries per scene.
- **Bounded query cost:** detection is capped at `k = 100` frames regardless of how
  large the memory grows, so query latency stays near-constant while a brute-force
  scan grows with the trace.
- **One configuration** of geometric hyperparameters generalizes across 197 scenes in
  four datasets (real video, consumer-LiDAR, and simulation).

The trade-off is per-query compute: a query runs object detection at answer time
(≈2.5 s, detector-bound), whereas an eager map answers in milliseconds. JIT wins on
total cost when queries are sparse relative to build cost.

---

## Architecture

```
Query: "Where is the couch?"
        │
        ▼
┌────────────────────────────────────────────────┐
│  Level 1 — Semantic retrieval                    │
│  CLIP text embedding → FAISS inner-product search│   sub-millisecond
│  N stored keyframes → k = 100 candidate frames   │
└────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────┐
│  Level 2 — Detect · project · cluster            │
│  OWL-ViT (τ = 0.1) on the k frames               │   ≈2.5 s total
│  → 30th-percentile depth back-projection          │   (detector-bound)
│  → DBSCAN (ε = 1 m) → ranked 3D clusters          │
└────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────┐
│  Level 3 — Verification (optional)               │
│  re-detect in top-5 clusters, refine centroids    │   dense captures only
│  auto-disabled on wide-baseline traces            │   (e.g. ScanNet: on; HM3D: off)
└────────────────────────────────────────────────┘
        │
        ▼
Ranked 3D locations (x, y, z) + confidence
```

The base system is **L1+L2**. L3 verification helps dense captures (ScanNet,
ARKitScenes) but hurts wide-baseline ones (Replica), so results default to L1+L2 and
report L3 as an ablation.

---

## Results

Evaluated on **197 scenes across four datasets** with one shared set of geometric
hyperparameters. Numbers below are macro Loc@1m (fraction of queries localized within
1 m of ground truth) unless noted.

### Real-scene video — ScanNet v2 (141 scenes)

| Method | Loc@1m | Build | Storage | Query |
|--------|:------:|:-----:|:-------:|:-----:|
| **JIT (L1+L2)** | **79.1** | **3.1 s** | **1.0 MB** | 2.5 s |
| **JIT (+L3)** | **81.3** | **3.1 s** | **1.0 MB** | 2.5 s |
| ConceptFusion | 74.1 | 286 s | ~250 MB | 25 ms |
| VLMaps | 71.8 | 47 s | 490 MB | 48 ms |
| GOAT | 71.7 | 27 s | 58 MB | <1 ms |
| ConceptGraphs | 70.3 | 2,160 s | 7 MB | <10 ms |

The *Storage* column counts each method's **own built artifact**; the raw RGB-D
keyframes that every method ingests at build time are excluded from all rows. JIT's
1.0 MB index is the smallest artifact here — 7×–490× under the eager maps. JIT alone
*retains* those shared keyframes (~375 MB) so it can detect at query time, whereas the
eager maps fuse the frames into their map and discard them; JIT's full on-robot
footprint is therefore ~376 MB. That retention is a deliberate storage-for-compute
trade — it is what lets JIT build in 3.1 s rather than up to 2,160 s and keep the
detector vocabulary open until the query is known.

JIT exceeds every eager dense-map baseline at the 1 m threshold while building its
index in seconds. Against an equal-budget random-sampling baseline with the same
clustering (Random-100 + DBSCAN, 72.8), JIT gains **+6.3 pp**. On dense video JIT is
statistically level with equal-budget random *single-best* sampling (80.7 ± 1.0); the
guaranteed benefit there is the bounded, memory-independent detection budget rather
than raw accuracy — the accuracy separation is larger on sparser captures.

### Generalization across datasets (Loc@1m, one hyperparameter set)

| Dataset (scenes) | JIT (L1+L2) | Random-100 + DBSCAN | ConceptGraphs | GOAT |
|------------------|:-----------:|:-------------------:|:-------------:|:----:|
| ScanNet (141)    | 79.1        | 72.8                | 70.3          | 71.7 |
| HM3D (36)        | **78.4**    | 67.2                | 6.1           | 55.4 |
| ARKitScenes (12) | 71.5        | 71.7                | 66.9          | 55.8 |
| Replica (8)      | **38.3**    | 35.4                | 26.0          | 31.5 |

On the wide-baseline HM3D captures every eager map collapses (ConceptGraphs 6.1),
while JIT holds 78.4 — a **+11.2 pp** margin over the equal-budget random+DBSCAN
baseline. The frame budget also buys efficiency: JIT reaches random sampling's best
accuracy with roughly **4× fewer** detector calls (`k=50` vs `k=200`).

### Ranked localization (correct location within top-*k* clusters)

| Method | R@1 | R@3 | R@5 |
|--------|:---:|:---:|:---:|
| **JIT (+L3)** | **81.3** | **86.7** | **87.7** |
| JIT (L1+L2) | 79.1 | 83.9 | 84.2 |
| Random-100 (single-best) | 80.0 | 80.0 | 80.0 |
| Random-100 + DBSCAN | 74.2 | 80.5 | 80.8 |

The ranked cluster list places the correct location in the top-3 for **86.7 %** of
queries (+6.7 pp over the best single-prediction baseline).

### Scalability

Because detection is capped at `k = 100` frames, query latency stays ≈2.5 s as the
memory grows, while a brute-force all-frames scan rises to 38.5 s at 2,500 frames
(≈15×). JIT's Loc@1m holds near 79 % across that range.

> **Caveats (measured honestly):** latency is measured in simulation, not on robot
> hardware. L3 verification is dataset-dependent (helps dense captures, hurts
> wide-baseline). The build-cost advantage is a *wall-clock / build-energy* result;
> because JIT runs detection per query and does not amortize it, the crossover on
> *cumulative* energy is much later than on wall-clock.

---

## Installation

```bash
# 0. Clone the repository
git clone https://github.com/kurshakuz/jit-episodic-memory
cd jit-memory

# 1. Conda environment with Habitat-Sim (for the HM3D / simulation path)
conda create -n habitat python=3.9 habitat-sim -c conda-forge -c aihabitat -y
conda activate habitat

# 2. Python dependencies
pip install -r requirements.txt      # or: conda env create -f environment.yml
```

Verify:

```bash
python -c "from retrieval import JITRetrievalCascade; print('JIT cascade: OK')"
```

### Datasets

| Dataset | Role | Source |
|---------|------|--------|
| HM3D | simulation (Habitat-Sim) | [habitat-sim DATASETS.md](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md) — set `HM3D_DATA` |
| ScanNet v2 | real RGB-D video | [ScanNet](https://github.com/ScanNet/ScanNet) — set `SCANNET_REPO`; download with `scannet/download_*.py` |
| ARKitScenes | real consumer-LiDAR | [ARKitScenes](https://github.com/apple/ARKitScenes) |
| Replica | synthetic RGB-D | [Replica](https://github.com/facebookresearch/Replica-Dataset) |

Environment variables:

| Variable | Description |
|----------|-------------|
| `HM3D_DATA` | Path to the HM3D scene datasets (simulation path). |
| `SCANNET_REPO` | Path to a ScanNet repo checkout; used only for the official val split file. |

The dense-map baselines (ConceptGraphs, VLMaps, ConceptFusion, GOAT) wrap their
official codebases and need their own environments — see comments in the relevant
`baselines/run_official_*.py` and `scannet/run_cg_scannet.py` headers.

---

## Repository structure

```
jit_memory/
├── ingestion/            # Memory building: CLIP encode, FAISS index, keyframe selection
├── retrieval/            # 3-level cascade: level1_semantic, level2_geometric,
│                         #   level3_verification, cascade (orchestration)
├── evaluation/           # Pose/depth evaluation, metrics, energy benchmark, figure gen
├── oracle/               # Ground-truth generation
├── habitat_benchmark/    # Downstream Habitat ObjectNav benchmark
├── baselines/            # Eager dense-map baselines (ConceptGraphs, VLMaps,
│                         #   ConceptFusion, GOAT) + comparison scripts
├── scannet/              # Real-scene pipeline: download, prepare_scenes, build_gt,
│                         #   evaluate_v2 / evaluate_scalability / evaluate_owlv2, config
├── experiments/          # Scaling study + ConceptGraphs / paper verification
├── scripts/              # run_exploration.py (HM3D memory-bank generation)
├── configs/default.yaml  # HM3D pipeline configuration
├── environment.yml / requirements.txt
└── LICENSE
```

ScanNet-specific configuration (scene lists, category maps, hyperparameters) lives in
`scannet/config.py`.

---

## Quick start

### Query an existing memory bank (Python API)

```python
from retrieval import JITRetrievalCascade

cascade = JITRetrievalCascade("/path/to/scene/exploration")   # posed RGB-D trace
result = cascade.query("couch")                                # or query_fast() to skip L3

if result.success:
    loc = result.best_location
    x, y, z = loc.centroid_3d
    print(f"Found at ({x:.2f}, {y:.2f}, {z:.2f}), "
          f"score {loc.best_detection_score:.2f}, {result.total_time_ms:.0f} ms")
```

### Simulation path (HM3D via Habitat-Sim)

```bash
export HM3D_DATA=/path/to/scene_datasets/hm3d

# Generate memory banks by exploring scenes
python scripts/run_exploration.py --scene /path/to/scene.glb --steps 500
python evaluation/multi_scene_eval.py            # or all HM3D scenes

# Evaluate the cascade and baselines
python evaluation/full_eval_v2.py
python evaluation/extended_baselines.py          # depth-based baselines
```

### Real-scene path (ScanNet)

```bash
export SCANNET_REPO=/path/to/ScanNet

# Download annotations + .sens, then convert to posed RGB-D streams
python scannet/download_batch.py --all
python scannet/prepare_scenes.py --scenes scene0568_00

# Evaluate JIT (L1+L2 and +L3) and the brute-force reference
python scannet/evaluate_v2.py --methods jit,jit_l3,bf --scenes scene0568_00

# Eager dense-map baselines
python scannet/run_dense_baselines.py --methods densemap,vlmap --scenes scene0568_00
```

Run any script with `--scenes` omitted to process the full evaluation set.

---

## Configuration

The HM3D pipeline reads `configs/default.yaml`; the ScanNet pipeline reads
`scannet/config.py`. The headline hyperparameters (one set across all datasets):

| Stage | Parameter | Value |
|-------|-----------|-------|
| L1 semantic | retrieval budget `k` | 100 frames (the fixed detection budget) |
| L2 geometric | DBSCAN `ε` / `min_samples` | 1.0 m / 1 |
| L2 geometric | depth sampling | 30th percentile in a 30×30 patch |
| L3 verification | max clusters verified | 5 (dense captures only) |
| Detector | OWL-ViT confidence `τ` | 0.1 |

`k = 100` is the only materially sensitive knob; over-retrieving past it degrades
accuracy, and `k = 50` roughly halves query latency for ≈3.5 pp of Loc@1m on HM3D.

---

## Models

| Component | Model | Notes |
|-----------|-------|-------|
| Text/image embedding | OpenCLIP **ViT-B/32** | L1 retrieval, inner-product similarity |
| Object detection | **OWL-ViT** `google/owlvit-base-patch32` | L2/L3 default detector, τ = 0.1 |
| Vector search | **FAISS** flat, inner product | O(1) insert, sub-ms search |
| Clustering | **DBSCAN** (ε = 1 m) | min_samples = 1; clusters ranked by cumulative confidence |

OWLv2 and Grounding DINO appear only in the detector-swap **ablation** (OWLv2:
+8.4 pp Loc@1m at ≈7.8× latency); OWL-ViT is the detector used throughout the
headline system.

---

## Citation

```bibtex
@unpublished{jit_episodic_memory,
  title  = {Spatially-Grounded Just-in-Time Episodic Memory for Mobile Robots},
  author = {Shyngyskhan Abilkassov and Almas Shintemirov},
  year   = {2026},
  note   = {Under review}
}
```

## License

MIT License — see [LICENSE](LICENSE).

