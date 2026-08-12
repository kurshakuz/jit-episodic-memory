# Real-robot JIT episodic memory

The same Just-in-Time pipeline from the paper — CLIP–FAISS retrieval → OWL-ViT
detection → RGB-D back-projection → DBSCAN clustering — run on a walking quadruped
instead of the HM3D simulator, with no fine-tuning or calibration to the robot or the
scene. It turns raw ROS 2 `.mcap` recordings into query-time 3D object localizations
in a drift-free map frame, and scores them against onboard LiDAR.

These scripts produce the results in the **"On a real robot"** section of the project page.

## Data

ROS 2 (Jazzy) `.mcap` recordings from a ZED stereo camera + RoboSense LiDAR on a
quadruped. Topics used:

| Topic | Use |
|---|---|
| `/zed/zed_node/rgb/color/rect/image` (`bgra8`) | keyframe RGB |
| `/zed/zed_node/depth/depth_registered` (`32FC1`, metres) | back-projection depth |
| `/zed/zed_node/rgb/color/rect/camera_info` | intrinsics |
| `/zed/zed_node/odom` | odometry pose (drifts) |
| `/localization/pose` | map-frame pose (drift-free, from the robot's prior LiDAR map) |
| `/tf`, `/tf_static` | rigid camera / LiDAR mount chain |
| `/rslidar_points` (`PointCloud2`) | LiDAR ground truth |

The recordings themselves are private; everything below regenerates from them, and the
released page assets under `docs/static/images/realbot/` are the committed artifacts.

> **Multi-chunk recordings:** `/tf_static` is transient-local, so in a chunked recording
> it is stored **only in the first chunk**. Run `dump_static_tf.py` once on chunk 0; the
> extraction scripts fall back to the resulting `static_tf.json` for every later chunk.

## Install

```bash
pip install mcap mcap-ros2-support        # standalone .mcap reading (no ROS install)
# CLIP / OWL-ViT / FAISS deps come from the main repo (open_clip, torch, transformers, sklearn)
```

## Pipeline

```bash
# 0. (multi-chunk only) dump the static TF tree from the first chunk
python realbot/dump_static_tf.py run_0.mcap

# 1. extract a JIT trace: synced RGB + registered depth + odom pose, every 8th frame
python realbot/bag_to_jit.py run_3.mcap realbot/trace_c3 8

# 2. add drift-free map-frame poses (T_map_optical = /localization/pose x static chain)
python realbot/add_map_poses.py run_3.mcap realbot/trace_c3

# 3. query the memory (CLIP retrieve -> OWL-ViT detect -> back-project -> x-y DBSCAN)
REALBOT_TRACE=realbot/trace_c3 python realbot/run_query.py "gray suv"  map 60
REALBOT_TRACE=realbot/trace_c4 python realbot/run_query.py "bicycle"   map 60

# 4. LiDAR ground truth for a localized object (projects /rslidar_points into the camera)
REALBOT_TRACE=realbot/trace_c4 python realbot/lidar_gt.py "bicycle" run_4.mcap

# figures / video for the page
python realbot/summary_montage.py                 # multi-query montage
python realbot/make_clip.py                        # docs/static/videos/realbot_rollout.mp4
python realbot/export_web_tiles.py                 # docs/static/images/realbot/*.jpg
```

`bag_inspect.py` and `survey_chunks.py` are discovery helpers (dump channels/encodings,
or grid RGB frames across chunks). Verification montages and diagnostics go to
`realbot/_out/` (override with `REALBOT_OUT`).

## Findings (honest)

- **`car` localizes to 0.20 m** of RoboSense LiDAR ground truth.
- Open-vocabulary queries localize diverse objects never seen in training: a thin
  **bicycle** (14 views, 0.33 m spread), a **white car** vs a **gray SUV** (~15 m apart in
  the same parked row, resolved by colour), and a **bench**.
- **Grounding depends on the queried attribute actually being present.** `white car` and
  `gray suv` resolve the two vehicles correctly, but a query for an attribute that is not
  there — a `blue car`, when none is blue — returns the nearest match. Detection is also
  prompt-sensitive: a bare noun can score far below a short descriptive phrase.
- **Sparse LiDAR is not a fair ground truth for thin objects:** a beam shoots *through* a
  bicycle frame to the wall behind it. On the view where the bike fills the box, ZED and
  LiDAR agree to 0.05 m; the aggregate error is inflated by the see-through beams. For
  solid objects (the car) LiDAR confirms the localization.

## Experiments

Six hypotheses tested on this data — accuracy vs LiDAR (100% Loc@0.5 m), frame efficiency
(CLIP vs random), map-vs-odom frame, depth-vs-range + multi-view convergence, synonym
robustness, and the 919 m global object map — with honest outcomes (including a negative
result) written up in `EXPERIMENTS.md`. Scripts: `exp_accuracy.py`, `exp_budget.py`
(+`exp_budget_fig.py`), `exp_frames.py`, `exp_analyze.py`, `exp_synonyms.py`,
`exp_global_map.py`.
