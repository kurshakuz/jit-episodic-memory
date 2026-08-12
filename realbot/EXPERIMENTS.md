# Real-robot experiments

Hypotheses tested on the real quadruped data, with honest outcomes. Scripts:
`exp_accuracy.py`, `exp_budget.py` / `exp_budget_fig.py`, `exp_frames.py`,
`exp_analyze.py`, `exp_synonyms.py`, `exp_global_map.py`. Outputs land in
`_out/experiments/` (git-ignored — regenerate from the traces).

## E1 — Does sim-level accuracy transfer to real hardware? **Yes.**

ZED-depth map-frame localization vs independent RoboSense LiDAR ground truth, solid
objects with LiDAR returns:

| object | chunk | views | loc. error (xy) | mean ZED–LiDAR depth err |
|---|---|---|---|---|
| white car | 3 | 46 | **0.06 m** | 0.17 m |
| gray suv | 3 | 26 | **0.09 m** | 0.15 m |
| manhole cover | 1 | 5 | **0.11 m** | 0.10 m |
| bench | 3 | 12 | **0.12 m** | 0.07 m |
| car | 0 | 61 | **0.22 m** | 0.17 m |
| tree | 0 | 9 | **0.29 m** | 0.45 m |
| building | 0 | 9 | **0.29 m** | 0.31 m |
| stop sign | 1 | 45 | **0.35 m** | 0.77 m |
| column | 0 | 16 | **0.40 m** | 0.40 m |

All 9/9 within 0.5 m → real-robot **Loc@0.5 m = 100 %**, mean ≈ **0.21 m**. (Thin objects —
bicycle, lamp post — excluded: LiDAR is unfair to them. Extended surfaces — hedge, stairs —
localize but a single "error" is ill-defined.)

### Deep footage survey (all 6 chunks)

A dense vision survey of every chunk (extracting the two unexplored ones, 1 & 2 —
`deep_survey.py`) turned up new object classes beyond the original set. Localized and
LiDAR-checked: **manhole cover** (novel, recurring across chunks 1–3, 11 instances in
chunk 2 alone, 0.04–0.23 m spread), **column ≡ pillar** (overpass supports; the two
synonyms hit the same structures — another E5 point), **stop sign**, **lamp post**,
**hedge**. The whole run's episodic memory now holds **14 objects** on one 919 m map.

**Honest negative — large structures don't ground.** The glass pedestrian skywalk does not
localize: OWL-ViT is an object detector, not a scene parser. `pedestrian bridge` → 1
detection; `bridge` → mislocalized onto the stairs. Open-vocabulary detection is
object-scale, not building-scale.

## E2 — Is CLIP retrieval frame-efficient on real footage? **Yes, for the queried instance.**

Localization error vs detector budget *k*, CLIP-ranked vs random frame selection:
CLIP reaches <1 m at **k = 2** ("car"); **random never converges** (it localizes a
*different* instance — 10–85 m off — through k = 60). Honest nuance: the advantage is
about *which* instance you localize. For an object filling most frames the base rate is
high, so random still detects *a* car, just not the queried one; and for a rare/false-
positive-heavy object ("bench") CLIP retrieval itself is weaker (locks on only by k ≈ 20).

## E3 — Does the drift-free map frame beat odometry *within a pass*? **No — and that's fine.**

Same detections, clustered in map vs odom frame. Odometry is locally *as tight or tighter*
(gray suv: odom 0.46 m vs map 1.72 m; tree: 0.87 vs 1.84), because `/localization/pose`
snaps to the prior map with jitter while odometry is locally smooth. The map frame's value
is **not** within-pass tightness — it is global consistency (E6): a single memory across
the whole traverse, which per-chunk odometry cannot provide.

## E4 — Depth vs range, and does multi-view fusion help? **6 cm up close; yes.**

- ZED–LiDAR depth agreement: **0.06 m at 0–3 m**, degrading to 0.30/0.37/0.64 m at
  3–6 / 6–9 / 9–12 m (Pearson r = 0.41 over 107 detections). Textbook stereo behaviour.
- Multi-view fusion: a **single** detection sits ~1–2 m from GT; fusing views converges to
  **<0.25 m** (car 1.87→0.23 m, white car 0.95→0.07 m, gray suv 1.64→0.19 m). This is the
  clustering stage earning its place.

## E5 — Is grounding real (synonyms → same point)? **Yes for singletons.**

- bench ≡ "wooden bench" within **0.09 m**; bicycle ≡ bike ≡ "a bicycle" within **0.19 m**;
  gray suv ≡ grey ≡ silver suv ≡ "gray car" within **0.78 m**.
- Honest edges: "seat" fails to fire on the bench (out of detector vocab); "vehicle" lands
  on a *different* car 78 m down the multi-car underpass (car/automobile/sedan agree ~3 m).

## E6 — One global memory over the walk. **919 m loop.**

The six chunks are one continuous **919 m** campus loop; every query-localized object drops
onto a single drift-free map, and the trajectory closes (chunk 4 returns to chunk 0's start)
with no visible drift jump — the global consistency E3 pointed to. See `global_map.png`.

## E7 — On-robot compute & latency (measured on the actual robot)

The recordings identify the platform: **Unitree B2-W** (vendor package `b2w_robot_interfaces`,
Unitree "SportMode" API, and 16 joints = per-leg hip/thigh/calf **+ an actuated wheel
`foot_joint`**), whose onboard computer is an **NVIDIA Jetson Orin NX 16 GB** (L4T R36.4.4 /
JetPack 6.2). We ran the pipeline **on that Orin** (MAXN power mode) in a disposable
`l4t-pytorch` container over real ZED frames (float32; nothing installed on the robot host).

Per-stage (Orin NX, MAXN): **OWL-ViT detect 184 ms/frame** (the bottleneck, 768² input),
CLIP image encode 36 ms/frame (build only), CLIP text encode 17 ms/query, retrieval 0.05 ms,
back-project + DBSCAN 1.5 ms. Index build (500 keyframes) **17.9 s**.

End-to-end query latency is detector-bound (≈ *k* × 184 ms):

| detector budget k | on-robot latency |
|---|---|
| 2 | 0.39 s |
| 8 | 1.49 s |
| 16 | 2.96 s |
| 60 | 11.06 s |

**Key finding:** latency scales with the frame budget, so the frame-efficiency result (E2) is
also what makes on-robot latency tractable — at a large budget (k=60) latency is an impractical
11 s, but the k that actually suffices on real data (k=2–16) lands at **0.4–3 s**. The build-cost
advantage also holds on hardware (17.9 s one-time vs the eager maps' minutes-to-hours). Caveats:
these are **float32** numbers and the iGPU is already saturated by one forward pass (batching 8
gives only 184→168 ms/frame); the robot's own perception runs FP16, so a TensorRT/FP16 or
distilled detector is the obvious speedup (future work). Reproduce: `robot_forensics.py` (platform
ID) and the container recipe in the project notes.
