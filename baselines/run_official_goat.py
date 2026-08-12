#!/usr/bin/env python3
"""Official GOAT baseline (localization) for the JIT paper.

Runs the OFFICIAL GOAT Object Instance Memory from facebookresearch/home-robot
(the `theo/offline_goat` offline driver — byte-identical instance-memory build to
`goat-sim`) over our jit_format posed RGB-D streams, and scores it with the same
Loc@Xm + build-time + storage protocol as the other official dense baselines
(ConceptFusion / VLMaps / ConceptGraphs).

Scope = LOCALIZATION ONLY. GOAT is a navigation system, but JIT is a localization
method (Loc@Xm). We use GOAT's eager, detect-at-collection Object Instance Memory
(official Detic detector + the official Categorical2DSemanticMap instance-association
module) and query it by category (GOAT's object-goal mode). We do NOT run GOAT's
navigation policy. This is the apples-to-apples comparison the reviewers asked for:
GOAT (eager instance memory) vs JIT (lazy CLIP-FAISS index), on identical inputs.

What is official vs adapted (documented for the reviewer):
  * OFFICIAL, unmodified: Detic detection (Swin-B, LCOCOI21k), the InstanceMemory
    class, and the Categorical2DSemanticMapModule cross-frame instance association
    (`_get_local_to_global_instance_mapping`). This is GOAT's real "commit at
    collection" object memory.
  * ADAPTED (scoring only): each committed instance's 3D world location is computed
    by unprojecting its stored detection mask with the SAME OpenCV c2w convention
    used by every other baseline in this repo (Habitat quat -> c2w @ diag(1,-1,-1,1),
    GT-aligned world), so GOAT's instance centroids live in the same metric frame as
    the ground-truth object centers. GOAT's own map frame is an arbitrary map-cell
    origin; expressing predictions in the GT world frame is required to score at all
    and does not change the physical points.

Fairness:
  * Detic runs OPEN-VOCABULARY on the exact eval categories (so "lamp/mirror/
    cabinet/shelf", absent from COCO-80, are detectable) — mirroring JIT's OWL-ViT
    open-vocab on the same queries.
  * A query category that is present in the GT but never committed by GOAT during its
    single eager pass counts as a MISS (correct_at = all-False), exactly as JIT/BF are
    scored on the full GT-present query set. This makes GOAT's "eager pass can miss"
    behavior visible rather than hidden — the paper's "defer beats commit" thesis.

Usage:
    GOAT_REPO=/abs/path/to/home-robot \
    PYTHONPATH=/abs/path/to/home-robot/src/home_robot \
    conda run -n goat python baselines/run_official_goat.py \
        --dataset {hm3d,scannet,arkit,replica} [--max-scenes N] [--scenes a,b] [--output f.json]
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")

import argparse
import gc
import json
import pickle
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ============================================================
# Paths / official GOAT on PYTHONPATH
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOAT_REPO = Path(os.environ.get("GOAT_REPO", str(PROJECT_ROOT.parent / "home-robot")))
_HR = GOAT_REPO / "src" / "home_robot"
if str(_HR) not in sys.path:
    sys.path.insert(0, str(_HR))

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "paper_results"
CACHE_DIR = OUTPUT_DIR / "goat_official_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Canonical eval constants (mirrored from scannet/config.py so this
# runner is self-contained and importable from the isolated `goat` env)
# ============================================================
JIT_QUERIES_10 = ["toilet", "chair", "table", "bed", "couch",
                  "sink", "lamp", "mirror", "cabinet", "shelf"]
ARKIT_QUERIES = ["chair", "table", "sofa", "bed", "sink", "cabinet", "toilet",
                 "refrigerator", "oven", "stove", "tv", "shelf", "bathtub"]
THREERSCAN_QUERIES = ["chair", "table", "sofa", "couch", "bed", "cabinet", "desk",
                      "stool", "tv", "lamp", "sink", "toilet", "shelf", "armchair", "bench"]
SCANNET_SYNONYMS = {"couch": ["couch", "sofa"], "shelf": ["shelf", "bookshelf"]}

# Superset of thresholds; the audit + Table III read 1.0m. Kept in this order.
THRESHOLDS = [0.25, 0.5, 1.0, 2.0, 3.0]

FRAME_STRIDE = 4          # every 4th frame (CG / CF default)
MAX_SIDE = 960            # cap long image side for Detic memory (K scaled to match)
MODULE_MIN_DEPTH = 0.3    # meters, for GOAT's internal map/association
MODULE_MAX_DEPTH = 5.0    # meters
CENTROID_MIN_DEPTH = 0.1  # meters, for the world-centroid unprojection
CENTROID_MAX_DEPTH = 10.0
DEVICE = "cuda:0"

DATASETS = {
    "hm3d":    dict(root=PROJECT_ROOT / "outputs" / "multi_scene_eval_500f",
                    queries=JIT_QUERIES_10, image_subdir="images"),
    "scannet": dict(root=PROJECT_ROOT / "scannet" / "jit_format_500",
                    queries=JIT_QUERIES_10, image_subdir="rgb"),
    "arkit":   dict(root=PROJECT_ROOT / "data" / "arkitscenes_jit",
                    queries=ARKIT_QUERIES, image_subdir="rgb"),
    "replica": dict(root=PROJECT_ROOT / "data" / "replica_jit",
                    queries=JIT_QUERIES_10, image_subdir="rgb"),
    "threerscan": dict(root=PROJECT_ROOT / "data" / "threerscan_jit",
                       queries=THREERSCAN_QUERIES, image_subdir="rgb"),
}

# ============================================================
# Pose / intrinsics (self-contained; identical math to
# baselines/jit_gradslam_dataset.py -> proven GT-aligned convention)
# ============================================================

def quat_wxyz_to_R(qw, qx, qy, qz) -> np.ndarray:
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,     1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float64)


# Habitat/OpenGL camera -> OpenCV camera flip (same as jit_gradslam_dataset / run_official_cg)
_P_GL2CV = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float64)
# Habitat world (Y up) -> home-robot world (Z up, +X fwd, +Y left)
_R_H2R = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
# home-robot camera coords -> OpenCV camera coords (X_fwd/Y_left/Z_up -> X_right/Y_down/Z_fwd)
_C_OCV_FROM_HR = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float64)


def load_intrinsics(scene_dir: Path) -> dict:
    """ScanNet/ARKit/Replica: read intrinsics.json. HM3D: synthesize (hfov 90, 640x480)."""
    p = scene_dir / "intrinsics.json"
    if p.exists():
        d = json.load(open(p))
        H = int(d.get("target_height", 480))
        W = int(d.get("target_width", 640))
        return dict(fx=float(d["fx"]), fy=float(d["fy"]), cx=float(d["cx"]), cy=float(d["cy"]),
                    H=H, W=W, sensor_height=float(d.get("sensor_height", 1.5)))
    H, W, hfov = 480, 640, 90.0
    fx = W / (2.0 * np.tan(np.deg2rad(hfov) / 2.0))
    return dict(fx=fx, fy=fx, cx=W / 2.0, cy=H / 2.0, H=H, W=W, sensor_height=1.5)


def c2w_opencv(row, sensor_height: float) -> np.ndarray:
    """Camera-to-world in OpenCV convention, GT-aligned world (Habitat @ P flip)."""
    R = quat_wxyz_to_R(row["qw"], row["qx"], row["qy"], row["qz"])
    t = np.array([row["x"], row["y"] + sensor_height, row["z"]], dtype=np.float64)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R
    c2w[:3, 3] = t
    return c2w @ _P_GL2CV


def homerobot_pose(c2w_ocv: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Derive home-robot (camera_pose 4x4 Z-up, gps[x,y], compass yaw) from OpenCV c2w.

    All three are mutually consistent by construction: camera_pose_hr = R_H2R @ c2w_ocv @ C.
    Used ONLY for GOAT's internal instance association; the world centroid is computed
    independently from c2w_ocv, so the arbitrary association frame never affects Loc@Xm.
    """
    cam = _R_H2R @ c2w_ocv @ _C_OCV_FROM_HR
    gps = np.array([cam[0, 3], cam[1, 3]], dtype=np.float32)     # +X fwd, +Y left
    compass = np.array([np.arctan2(cam[1, 0], cam[0, 0])], dtype=np.float32)
    return cam.astype(np.float32), gps, compass


# ============================================================
# GT loading + query matching (identical to scannet/evaluate_v2.py)
# ============================================================

def load_scene_gt(scene_dir: Path) -> Optional[dict]:
    # is_file() (not exists()) so a broken symlink resolves to None, not a crash
    p = scene_dir / f"{scene_dir.name}_ground_truth.json"
    if not p.is_file():
        g = [x for x in scene_dir.glob("*_ground_truth.json") if x.is_file()]
        if not g:
            return None
        p = g[0]
    return json.load(open(p))


def get_gt_centers_for_query(gt: dict, query: str) -> List[np.ndarray]:
    centers = []
    synonyms = SCANNET_SYNONYMS.get(query.lower(), [query.lower()])
    synonyms = [query.lower()] + [s.lower() for s in synonyms if s.lower() != query.lower()]
    for _, obj in gt["objects"].items():
        label = obj["category"].lower()
        if any(syn in label for syn in synonyms):
            centers.append(np.array(obj["center"], dtype=np.float64))
    return centers


# ============================================================
# Scene enumeration
# ============================================================

def get_scenes(dataset: str, max_scenes: Optional[int]) -> List[Path]:
    root = DATASETS[dataset]["root"]
    if dataset == "scannet":
        try:
            sys.path.insert(0, str(PROJECT_ROOT))
            from scannet.config import SCANNET_VAL_SCENES
            order = SCANNET_VAL_SCENES
        except Exception:
            order = sorted(d.name for d in root.iterdir() if d.is_dir())
        scenes = [root / s for s in order if (root / s).exists()]
    else:
        scenes = []
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            if (d / f"{d.name}_ground_truth.json").exists() and \
               (d / "exploration" / "trace.parquet").exists():
                scenes.append(d)
    return scenes[:max_scenes] if max_scenes else scenes


def load_trace(scene_dir: Path):
    import pandas as pd
    return pd.read_parquet(scene_dir / "exploration" / "trace.parquet")


# ============================================================
# Frame loading (standalone; no gradslam dependency)
# ============================================================

def _resize_dims(H: int, W: int) -> Tuple[int, int, float]:
    ratio = min(1.0, MAX_SIDE / float(max(H, W)))
    return int(round(H * ratio)), int(round(W * ratio)), ratio


def load_frame(scene_dir: Path, row, intr: dict):
    """Return (rgb_uint8 HxWx3, depth_m HxW float32, K 3x3, c2w_ocv 4x4) at capped res."""
    import cv2
    from PIL import Image
    img_path = scene_dir / "exploration" / row["image_path"]
    dep_path = scene_dir / "exploration" / row["depth_path"]

    rgb = np.array(Image.open(img_path))
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]
    depth = np.load(dep_path).astype(np.float32)

    H0, W0 = depth.shape[:2]
    Ht, Wt, ratio = _resize_dims(H0, W0)
    if (Ht, Wt) != (H0, W0):
        rgb = cv2.resize(rgb, (Wt, Ht), interpolation=cv2.INTER_AREA)
        depth = cv2.resize(depth, (Wt, Ht), interpolation=cv2.INTER_NEAREST)
    # intrinsics were given at native (target) resolution -> scale to depth's native then cap
    sx = (Wt / float(intr["W"]))
    sy = (Ht / float(intr["H"]))
    K = np.array([[intr["fx"] * sx, 0, intr["cx"] * sx],
                  [0, intr["fy"] * sy, intr["cy"] * sy],
                  [0, 0, 1]], dtype=np.float64)
    c2w = c2w_opencv(row, intr["sensor_height"])
    return rgb.astype(np.uint8), depth, K, c2w


# ============================================================
# Build GOAT Object Instance Memory over a scene (OFFICIAL modules)
# ============================================================

def build_instance_memory(scene_dir: Path, frame_ids: List[int], queries: List[str],
                          detic, verbose: bool = False):
    """Step the official Categorical2DSemanticMapModule + InstanceMemory over the
    posed RGB-D stream. Returns (instance_memory, per_timestep_cache, build_time_s)."""
    import home_robot.utils.pose as pu
    from home_robot.core.interfaces import Observations
    from home_robot.mapping.semantic.categorical_2d_semantic_map_module import (
        Categorical2DSemanticMapModule,
    )
    from home_robot.mapping.semantic.categorical_2d_semantic_map_state import (
        Categorical2DSemanticMapState,
    )
    from home_robot.mapping.semantic.instance_tracking_modules import InstanceMemory

    device = torch.device(DEVICE)
    intr = load_intrinsics(scene_dir)
    trace = load_trace(scene_dir)

    # categories mirror build_map.py exactly: ["other", *queries, "other"]
    categories = ["other", *queries, "other"]
    num_sem_categories = len(categories) - 1          # = len(queries) + 1
    one_hot = torch.eye(num_sem_categories, device=device)

    # hfov for GOAT's internal camera matrix, from the true fx (cap-invariant)
    hfov = float(np.degrees(2.0 * np.arctan((intr["W"] / 2.0) / intr["fx"])))

    instance_memory = InstanceMemory(1, 4, debug_visualize=False)
    smap = Categorical2DSemanticMapState(
        device=device, num_environments=1, num_sem_categories=num_sem_categories,
        map_resolution=5, map_size_cm=4800, global_downscaling=2,
        record_instance_ids=True, instance_memory=instance_memory,
    )
    smap.init_map_and_pose()

    # size the module to the (capped) frame resolution + set the camera height for
    # voxel setup, from frame 0 (the constructor builds camera_matrix + voxel dims eagerly)
    r0 = trace.iloc[frame_ids[0]]
    _, depth0, _, c2w0 = load_frame(scene_dir, r0, intr)
    Hf, Wf = depth0.shape
    cam0, _, _ = homerobot_pose(c2w0)
    module = Categorical2DSemanticMapModule(
        frame_height=Hf, frame_width=Wf,
        camera_height=float(cam0[2, 3]), hfov=hfov,
        num_sem_categories=num_sem_categories,
        map_size_cm=4800, map_resolution=5, vision_range=100,
        explored_radius=150, been_close_to_radius=200, global_downscaling=2,
        du_scale=4, cat_pred_threshold=5.0, exp_pred_threshold=1.0, map_pred_threshold=1.0,
        min_depth=MODULE_MIN_DEPTH, max_depth=MODULE_MAX_DEPTH, must_explore_close=False,
        min_obs_height_cm=10, record_instance_ids=True, instance_memory=instance_memory,
    ).to(device)

    cache: Dict[int, dict] = {}           # timestep -> {depth, K, c2w}
    last_pose = np.zeros(3)
    t0 = time.time()

    for step, fid in enumerate(frame_ids):
        row = trace.iloc[fid]
        rgb, depth, K, c2w = load_frame(scene_dir, row, intr)
        H, W = depth.shape

        # --- OFFICIAL Detic detection ---
        obs = Observations(gps=np.zeros(2), compass=np.zeros(1), rgb=rgb, depth=depth,
                           semantic=None, task_observations={})
        obs = detic.predict(obs, depth_threshold=0.5)

        # --- build_map.py post-processing (verbatim) ---
        obs.semantic[obs.semantic == 0] = len(categories) - 1
        obs.semantic = obs.semantic - 1
        obs.task_observations["instance_map"] += 1
        obs.task_observations["instance_map"] = obs.task_observations["instance_map"].astype(int)

        # --- preprocess to module tensor layout (verbatim build_map.preprocess_obs) ---
        rgb_t = torch.from_numpy(obs.rgb).to(device)
        depth_t = torch.from_numpy(obs.depth).unsqueeze(-1).to(device) * 100.0   # m -> cm
        sem_t = one_hot[torch.from_numpy(obs.semantic).long().to(device)]
        obs_pre = torch.cat([rgb_t, depth_t, sem_t], dim=-1)

        instances = obs.task_observations["instance_map"]
        inst_ids = np.unique(instances)
        id_to_idx = {iid: i for i, iid in enumerate(inst_ids)}
        inst_idx = torch.from_numpy(np.vectorize(id_to_idx.get)(instances)).to(device)
        inst_oh = torch.eye(len(inst_ids), device=device)[inst_idx]
        obs_pre = torch.cat([obs_pre, inst_oh], dim=-1)
        obs_pre = obs_pre.unsqueeze(0).permute(0, 3, 1, 2)     # (1, C, H, W)

        # pose in home-robot convention (for association only)
        cam_hr, gps, compass = homerobot_pose(c2w)
        curr_pose = np.array([gps[0], gps[1], compass[0]])
        pose_delta = torch.tensor(pu.get_rel_pose_change(curr_pose, last_pose)).unsqueeze(0).to(device)
        camera_pose = torch.tensor(cam_hr).unsqueeze(0).to(device)

        dones = torch.tensor([False]).to(device)
        update_global = torch.tensor([True]).to(device)
        (_, smap.local_map, smap.global_map, seq_lp, seq_gp, seq_lmb, seq_or) = module(
            obs_pre.unsqueeze(1), pose_delta.unsqueeze(1), dones.unsqueeze(1),
            update_global.unsqueeze(1), camera_pose,
            smap.local_map, smap.global_map, smap.local_pose, smap.global_pose,
            smap.lmb, smap.origins,
        )
        smap.local_pose = seq_lp[:, -1]
        smap.global_pose = seq_gp[:, -1]
        smap.lmb = seq_lmb[:, -1]
        smap.origins = seq_or[:, -1]
        last_pose = curr_pose

        # cache what the centroid pass needs, indexed by InstanceMemory timestep.
        # process_instances increments timesteps[0] AFTER storing views at value `step`.
        cache[step] = dict(depth=depth, K=K, c2w=c2w)

        if verbose and (step + 1) % 25 == 0:
            n_inst = len(instance_memory.instance_views[0])
            print(f"      frame {step+1}/{len(frame_ids)} | global instances={n_inst}")

    build_time = time.time() - t0
    return instance_memory, cache, build_time


# ============================================================
# Extract per-instance world centroids (adapted scoring path)
# ============================================================

def _view_world_points(mask: np.ndarray, depth: np.ndarray, K: np.ndarray,
                       c2w: np.ndarray) -> Optional[np.ndarray]:
    if mask.shape != depth.shape:
        import cv2
        mask = cv2.resize(mask.astype(np.uint8), (depth.shape[1], depth.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    good = mask & (depth > CENTROID_MIN_DEPTH) & (depth < CENTROID_MAX_DEPTH)
    ys, xs = np.where(good)
    if len(xs) == 0:
        return None
    z = depth[ys, xs].astype(np.float64)
    x = (xs - K[0, 2]) * z / K[0, 0]
    y = (ys - K[1, 2]) * z / K[1, 1]
    pts_cam = np.stack([x, y, z], axis=1)                 # OpenCV camera frame
    pts_world = (c2w[:3, :3] @ pts_cam.T + c2w[:3, 3:4]).T
    return pts_world


def extract_instances(instance_memory, cache: Dict[int, dict], queries: List[str],
                      verbose: bool = False) -> List[dict]:
    """One record per committed GOAT instance whose category is an eval query."""
    out = []
    inst_dict = instance_memory.instance_views[0]           # {global_id: Instance}
    N = len(queries)
    for gid, inst in inst_dict.items():
        cid = int(inst.category_id) if inst.category_id is not None else N
        if cid < 0 or cid >= N:
            continue                                        # "other"/background
        query = queries[cid]
        per_view_centroids = []
        total_pts = 0
        for view in inst.instance_views:
            ts = int(view.timestep)
            fr = cache.get(ts)
            if fr is None or view.mask is None:
                continue
            pw = _view_world_points(np.asarray(view.mask, dtype=bool),
                                    fr["depth"], fr["K"], fr["c2w"])
            if pw is None or len(pw) == 0:
                continue
            per_view_centroids.append(np.median(pw, axis=0))
            total_pts += len(pw)
        if not per_view_centroids:
            continue
        centroid = np.median(np.stack(per_view_centroids, axis=0), axis=0)
        out.append(dict(query=query, centroid=centroid.tolist(),
                        n_views=len(per_view_centroids), n_points=int(total_pts)))
    if verbose:
        by_q = defaultdict(int)
        for r in out:
            by_q[r["query"]] += 1
        print(f"      committed instances: {dict(by_q)}")
    return out


# ============================================================
# Score one scene
# ============================================================

def score_scene(scene_id: str, instances: List[dict], gt: dict, queries: List[str],
                build_time_s: float, storage_mb: float, query_ms: float) -> List[dict]:
    records = []
    for query in queries:
        gcs = get_gt_centers_for_query(gt, query)
        if not gcs:
            continue                                         # category absent from GT -> skip (as all baselines)
        cands = [r for r in instances if r["query"] == query]
        if cands:
            # SINGLE prediction = GOAT's highest-confidence instance of the category
            # (most-observed: n_views then n_points), symmetric with JIT's top-scored
            # cluster and CG's single best-CLIP object. Then min distance to nearest GT.
            # (NOT min over predicted instances -- that would be an unfair advantage.)
            best = max(cands, key=lambda r: (r.get("n_views", 0), r.get("n_points", 0)))
            pred = np.array(best["centroid"])
            min_distance = min(float(np.linalg.norm(pred - g)) for g in gcs)
            pred = pred.tolist()
        else:
            min_distance = None                              # committed nothing for this category -> MISS
            pred = None
        rec = dict(scene_id=scene_id, query=query, method="goat",
                   min_distance=min_distance,
                   predicted_location=pred,
                   gt_locations=[g.tolist() for g in gcs],
                   n_instances_pred=len(cands),
                   build_time_s=build_time_s, storage_mb=storage_mb, query_ms=query_ms,
                   correct_at={f"{t}": (min_distance is not None and min_distance < t)
                               for t in THRESHOLDS})
        records.append(rec)
    return records


def macro_loc_from_records(records: List[dict]) -> Dict[str, float]:
    """Per-scene macro Loc@thr (strict <), miss counted as False. Matches audit macro_loc."""
    out = {}
    for t in THRESHOLDS:
        by_scene = defaultdict(list)
        for r in records:
            d = r.get("min_distance")
            ok = (d is not None and d < t)
            by_scene[r["scene_id"]].append(ok)
        if by_scene:
            out[f"{t}"] = 100.0 * sum(sum(v) / len(v) for v in by_scene.values()) / len(by_scene)
    return out


# ============================================================
# Per-scene driver (with instance-level cache + resume)
# ============================================================

def process_scene(scene_dir: Path, dataset: str, detic, verbose: bool = True) -> Optional[List[dict]]:
    scene_id = scene_dir.name
    queries = DATASETS[dataset]["queries"]
    gt = load_scene_gt(scene_dir)
    if gt is None:
        return None

    scene_cache = CACHE_DIR / f"{dataset}_{scene_id}"
    scene_cache.mkdir(parents=True, exist_ok=True)
    inst_file = scene_cache / "instances.json"

    if inst_file.exists():
        blob = json.load(open(inst_file))
        instances = blob["instances"]
        build_time_s = blob["build_time_s"]
        storage_mb = blob["storage_mb"]
        query_ms = blob.get("query_ms", 0.0)
        if verbose:
            print(f"    {scene_id}: instances cached ({len(instances)}), re-scoring")
    else:
        trace = load_trace(scene_dir)
        frame_ids = list(range(0, len(trace), FRAME_STRIDE))
        instance_memory, cache, build_time_s = build_instance_memory(
            scene_dir, frame_ids, queries, detic, verbose=verbose)
        # storage = serialized eager instance DB (GOAT's committed memory)
        storage_mb = len(pickle.dumps(instance_memory.instance_views[0],
                                      protocol=pickle.HIGHEST_PROTOCOL)) / (1024 * 1024)
        instances = extract_instances(instance_memory, cache, queries, verbose=verbose)
        # query latency = category->instance lookup over the committed memory
        qt0 = time.time()
        _ = score_scene(scene_id, instances, gt, queries, 0, 0, 0)
        query_ms = 1000.0 * (time.time() - qt0) / max(1, len(queries))
        json.dump(dict(instances=instances, build_time_s=build_time_s,
                       storage_mb=storage_mb, query_ms=query_ms,
                       n_frames=len(frame_ids)), open(inst_file, "w"))
        del instance_memory, cache
        gc.collect()
        torch.cuda.empty_cache()

    return score_scene(scene_id, instances, gt, queries, build_time_s, storage_mb, query_ms)


# ============================================================
# Main
# ============================================================

def build_detic(queries: List[str]):
    from home_robot.perception.detection.detic.detic_perception import DeticPerception
    categories = ["other", *queries, "other"]
    return DeticPerception(vocabulary="custom",
                           custom_vocabulary=",".join(categories),
                           sem_gpu_id=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS.keys()), required=True)
    ap.add_argument("--max-scenes", type=int, default=None)
    ap.add_argument("--scenes", type=str, default=None, help="comma-separated scene ids")
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    scenes = get_scenes(args.dataset, args.max_scenes)
    if args.scenes:
        want = set(s.strip() for s in args.scenes.split(","))
        scenes = [s for s in scenes if s.name in want]

    output_file = args.output or str(OUTPUT_DIR / f"goat_{args.dataset}.json")
    ckpt_file = str(Path(output_file).with_suffix(".ckpt.json"))
    if os.path.exists(ckpt_file):
        ck = json.load(open(ckpt_file))
        completed = set(ck["completed"])
        all_records = ck["records"]
    else:
        completed, all_records = set(), []

    print(f"Official GOAT on {args.dataset.upper()} | {len(scenes)} scenes | "
          f"resuming from {len(completed)}")
    detic = build_detic(DATASETS[args.dataset]["queries"])

    t_start = time.time()
    for i, sd in enumerate(scenes):
        if sd.name in completed:
            print(f"  [{i+1}/{len(scenes)}] {sd.name}: SKIP")
            continue
        print(f"  [{i+1}/{len(scenes)}] {sd.name}")
        try:
            recs = process_scene(sd, args.dataset, detic, verbose=True)
            if recs:
                all_records.extend(recs)
                print(f"    {len(recs)} scored queries")
            completed.add(sd.name)
            json.dump(dict(completed=list(completed), records=all_records), open(ckpt_file, "w"))
        except Exception as e:
            print(f"    FAILED: {e}")
            traceback.print_exc()

    macro = macro_loc_from_records(all_records)
    build_times = [r["build_time_s"] for r in all_records]
    storages = [r["storage_mb"] for r in all_records]
    qms = [r["query_ms"] for r in all_records]
    output = dict(
        method="goat", dataset=args.dataset,
        n_scenes=len(completed), n_queries=len(all_records),
        macro_loc=macro,
        avg_build_s=float(np.mean(build_times)) if build_times else None,
        avg_storage_mb=float(np.mean(storages)) if storages else None,
        avg_query_ms=float(np.mean(qms)) if qms else None,
        elapsed_s=time.time() - t_start,
        per_query=all_records,
    )
    json.dump(output, open(output_file, "w"), indent=2, default=str)
    print("\n==== GOAT", args.dataset.upper(), "====")
    for t in THRESHOLDS:
        if f"{t}" in macro:
            print(f"  Loc@{t}m (macro): {macro[f'{t}']:.1f}%")
    print(f"  avg build {output['avg_build_s']}, storage {output['avg_storage_mb']} MB")
    print(f"  saved: {output_file}")


if __name__ == "__main__":
    main()
