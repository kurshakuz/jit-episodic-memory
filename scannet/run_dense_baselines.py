#!/usr/bin/env python3
"""
Run dense baselines (DenseMap, VLMaps) on ScanNet evaluation.

Adapts the HM3D baselines for ScanNet camera intrinsics:
- ScanNet HFOV ≈ 58° (NOT 90° like HM3D)
- Per-scene intrinsics from intrinsics.json
- sensor_height = 1.5 (agent-base positions stored in trace)

Usage:
    python scannet/run_dense_baselines.py --methods densemap,vlmap
    python scannet/run_dense_baselines.py --methods densemap --scenes scene0568_00
"""

import os
import sys
import json
import time
import gc
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from scannet.config import SCANNET_VAL_SCENES, SCANNET_JIT, SCANNET_RESULTS, JIT_QUERIES_10, SCANNET_SYNONYMS, LOCALIZATION_THRESHOLDS, BOOTSTRAP_ITERATIONS
from baselines.dense_map import DenseMapBaseline
from baselines.vlmap_baseline import VLMapBaseline


# ============================================================================
# ScanNet-adapted DenseMap (override intrinsics)
# ============================================================================

class ScanNetDenseMap(DenseMapBaseline):
    """DenseMap baseline adapted for ScanNet camera intrinsics."""

    def __init__(self, scene_dir, **kwargs):
        super().__init__(scene_dir, **kwargs)
        # Load per-scene intrinsics
        intrinsics_path = Path(scene_dir) / "intrinsics.json"
        with open(intrinsics_path) as f:
            intr = json.load(f)
        self._fx = intr["fx"]
        self._fy = intr["fy"]
        self._cx = intr["cx"]
        self._cy = intr["cy"]
        self._sensor_height = intr.get("sensor_height", 1.5)
        self._img_w = intr["target_width"]
        self._img_h = intr["target_height"]

    def _project_depth_to_3d(self, depth, position, rotation, stride=4):
        """Override with ScanNet intrinsics instead of hardcoded HFOV=90°."""
        h, w = depth.shape

        # Use actual ScanNet intrinsics (NOT hardcoded HFOV=90°)
        fx = self._fx
        fy = self._fy
        cx = self._cx
        cy = self._cy

        # Create pixel grids
        u = np.arange(0, w, stride)
        v = np.arange(0, h, stride)
        uu, vv = np.meshgrid(u, v)

        # Sample depth
        depth_sampled = depth[::stride, ::stride]

        # Valid depth mask
        valid_mask = (depth_sampled > 0.1) & (depth_sampled < self.max_depth)

        # Deproject to camera frame
        x_cam = (uu - cx) * depth_sampled / fx
        y_cam = -(vv - cy) * depth_sampled / fy
        z_cam = -depth_sampled

        points_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)

        # Transform to world frame
        w_q, x_q, y_q, z_q = rotation
        R = np.array([
            [1 - 2*y_q*y_q - 2*z_q*z_q,     2*x_q*y_q - 2*z_q*w_q,     2*x_q*z_q + 2*y_q*w_q],
            [    2*x_q*y_q + 2*z_q*w_q, 1 - 2*x_q*x_q - 2*z_q*z_q,     2*y_q*z_q - 2*x_q*w_q],
            [    2*x_q*z_q - 2*y_q*w_q,     2*y_q*z_q + 2*x_q*w_q, 1 - 2*x_q*x_q - 2*y_q*y_q],
        ])

        # Sensor position (agent base + sensor height)
        sensor_pos = position.copy()
        sensor_pos[1] += self._sensor_height

        # Flatten for batch transform
        points_flat = points_cam.reshape(-1, 3)
        points_world = (R @ points_flat.T).T + sensor_pos
        points_3d = points_world.reshape(points_cam.shape)

        return points_3d, valid_mask


# ============================================================================
# ScanNet-adapted VLMap (override intrinsics)
# ============================================================================

class ScanNetVLMap(VLMapBaseline):
    """VLMap baseline adapted for ScanNet camera intrinsics."""

    def __init__(self, scene_dir, **kwargs):
        super().__init__(scene_dir, **kwargs)
        # Load per-scene intrinsics
        intrinsics_path = Path(scene_dir) / "intrinsics.json"
        with open(intrinsics_path) as f:
            intr = json.load(f)
        self._fx = intr["fx"]
        self._fy = intr["fy"]
        self._cx = intr["cx"]
        self._cy = intr["cy"]
        self._sensor_height = intr.get("sensor_height", 1.5)

    def _get_camera_intrinsics(self, height, width):
        """Override with ScanNet intrinsics instead of hardcoded HFOV=90°."""
        K = np.array([
            [self._fx, 0, self._cx],
            [0, self._fy, self._cy],
            [0, 0, 1]
        ], dtype=np.float32)
        return K

    def _project_to_world(self, depth, position, rotation):
        """Override with ScanNet intrinsics and sensor_height."""
        H, W = depth.shape
        K = self._get_camera_intrinsics(H, W)
        K_inv = np.linalg.inv(K)

        # Create pixel grid
        u = np.arange(W)
        v = np.arange(H)
        uu, vv = np.meshgrid(u, v)

        # Homogeneous pixel coordinates
        ones = np.ones_like(uu)
        pixels = np.stack([uu, vv, ones], axis=-1)

        # Backproject to camera frame
        pixels_flat = pixels.reshape(-1, 3)
        depth_flat = depth.reshape(-1)

        points_cam = (K_inv @ pixels_flat.T).T * depth_flat[:, np.newaxis]

        # Habitat camera convention: -Y is up, -Z is forward
        points_cam_std = np.stack([
            points_cam[:, 0],
            -points_cam[:, 1],
            -points_cam[:, 2],
        ], axis=-1)

        # Rotation matrix
        R = self._quaternion_to_rotation_matrix(rotation)

        # Transform to world
        sensor_pos = position.copy()
        sensor_pos[1] += self._sensor_height

        points_world = (R @ points_cam_std.T).T + sensor_pos
        points_world = points_world.reshape(H, W, 3)

        # Valid mask
        valid_mask = (depth > 0.1) & (depth < 10.0)

        return points_world, valid_mask


# ============================================================================
# Ground truth and evaluation helpers
# ============================================================================

def load_scene_gt(scene_dir):
    scene_id = scene_dir.name
    gt_path = scene_dir / f"{scene_id}_ground_truth.json"
    if not gt_path.exists():
        return None
    with open(gt_path) as f:
        return json.load(f)


def get_gt_centers_for_query(gt, query):
    centers = []
    synonyms = SCANNET_SYNONYMS.get(query.lower(), [query.lower()])
    synonyms = [query.lower()] + [s.lower() for s in synonyms if s.lower() != query.lower()]
    for obj_id, obj in gt["objects"].items():
        label = obj["category"].lower()
        for syn in synonyms:
            if syn in label:
                centers.append(np.array(obj["center"]))
                break
    return centers


def compute_metrics(results_list, thresholds=LOCALIZATION_THRESHOLDS):
    """Compute macro/micro Loc@d metrics."""
    by_scene = defaultdict(list)
    for r in results_list:
        by_scene[r["scene_id"]].append(r)

    macro = {}
    for t in thresholds:
        scene_accs = []
        for scene_id, results in by_scene.items():
            correct = sum(1 for r in results
                          if r["error_m"] is not None and r["error_m"] < t)
            scene_accs.append(correct / len(results) if results else 0.0)
        macro[t] = np.mean(scene_accs) if scene_accs else 0.0

    micro = {}
    for t in thresholds:
        correct = sum(1 for r in results_list
                      if r["error_m"] is not None and r["error_m"] < t)
        micro[t] = correct / len(results_list) if results_list else 0.0

    return macro, micro


def bootstrap_ci(results_list, threshold, n_boot=BOOTSTRAP_ITERATIONS, alpha=0.05):
    by_scene = defaultdict(list)
    for r in results_list:
        by_scene[r["scene_id"]].append(r)

    scene_accs = []
    for scene_id, results in by_scene.items():
        correct = sum(1 for r in results
                      if r["error_m"] is not None and r["error_m"] < threshold)
        scene_accs.append(correct / len(results) if results else 0.0)

    scene_accs = np.array(scene_accs)
    rng = np.random.RandomState(42)
    boot_means = [np.mean(rng.choice(scene_accs, size=len(scene_accs), replace=True))
                  for _ in range(n_boot)]
    boot_means = np.array(boot_means)
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return lo, hi


# ============================================================================
# Run a single baseline on all scenes
# ============================================================================

def run_baseline(method_name, scenes, verbose=True, data_dir=None):
    """Run DenseMap or VLMap baseline on all ScanNet scenes."""
    data_dir = data_dir or SCANNET_JIT
    all_results = []
    all_build_stats = []

    for i, scene_id in enumerate(scenes):
        scene_dir = data_dir / scene_id
        if not scene_dir.exists():
            print(f"  [{scene_id}] Scene dir not found, skipping")
            continue

        gt = load_scene_gt(scene_dir)
        if gt is None:
            print(f"  [{scene_id}] No GT found, skipping")
            continue

        # Get valid queries
        valid_queries = []
        for query in JIT_QUERIES_10:
            centers = get_gt_centers_for_query(gt, query)
            if centers:
                valid_queries.append((query, centers))

        if not valid_queries:
            print(f"  [{scene_id}] No valid queries, skipping")
            continue

        print(f"\n[{i+1}/{len(scenes)}] {scene_id} ({len(valid_queries)} queries)")

        # Build map
        try:
            if method_name == "densemap":
                baseline = ScanNetDenseMap(
                    scene_dir,
                    voxel_size=0.05,
                    sample_stride=4,
                    max_depth=10.0,
                    clip_model="ViT-B-32-quickgelu",
                    clip_pretrained="laion400m_e32",
                )
            elif method_name == "vlmap":
                baseline = ScanNetVLMap(
                    scene_dir,
                    grid_resolution=0.05,
                    grid_size=500,
                    use_lseg=True,   # Use proper LSeg encoder (faithful to VLMaps paper)
                )
            else:
                raise ValueError(f"Unknown method: {method_name}")

            build_stats = baseline.build_map(verbose=verbose)
            print(f"  Built map: {build_stats.build_time_seconds:.1f}s, "
                  f"{build_stats.memory_size_mb:.1f}MB")
            all_build_stats.append({
                "scene_id": scene_id,
                "build_time_s": build_stats.build_time_seconds,
                "storage_mb": build_stats.memory_size_mb,
            })

        except Exception as e:
            print(f"  BUILD ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Query
        for query, gt_centers in valid_queries:
            try:
                result = baseline.query(query)

                if result.success and result.predicted_location is not None:
                    pred = np.array(result.predicted_location)
                    dists = [np.linalg.norm(pred - gc) for gc in gt_centers]
                    error = min(dists)
                else:
                    error = None

                correct_at = {}
                for t in LOCALIZATION_THRESHOLDS:
                    correct_at[str(t)] = (error is not None and error < t)

                r = {
                    "scene_id": scene_id,
                    "query": query,
                    "method": method_name,
                    "predicted_location": result.predicted_location.tolist()
                        if result.predicted_location is not None else None,
                    "gt_locations": [c.tolist() for c in gt_centers],
                    "error_m": error,
                    "correct_at": correct_at,
                    "query_time_ms": result.query_time_ms,
                    "confidence": result.confidence,
                }
                all_results.append(r)

                status = f"[OK] {error:.2f}m" if error is not None else "[FAIL] no pred"
                print(f"    {query}: {status} ({result.query_time_ms:.0f}ms)")

            except Exception as e:
                print(f"    {query}: ERROR - {e}")
                import traceback
                traceback.print_exc()

        # Free memory after each scene
        del baseline
        gc.collect()

    return all_results, all_build_stats


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ScanNet dense baselines evaluation")
    parser.add_argument("--methods", type=str, default="densemap,vlmap",
                        help="Comma-separated: densemap, vlmap")
    parser.add_argument("--scenes", type=str, default=None,
                        help="Comma-separated scene IDs (default: all)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override data directory (default: SCANNET_JIT)")
    args = parser.parse_args()

    from pathlib import Path
    data_dir = Path(args.data_dir) if args.data_dir else SCANNET_JIT

    methods = [m.strip() for m in args.methods.split(",")]
    scenes = SCANNET_VAL_SCENES
    if args.scenes:
        scenes = [s.strip() for s in args.scenes.split(",")]

    os.makedirs(str(SCANNET_RESULTS), exist_ok=True)

    print("=" * 60)
    print("ScanNet Dense Baselines Evaluation")
    print("=" * 60)
    print(f"  Methods: {methods}")
    print(f"  Scenes: {len(scenes)}")
    print(f"  Queries: {JIT_QUERIES_10}")
    print(f"  Using ScanNet intrinsics (HFOV≈58°, per-scene fx/fy/cx/cy)")
    print()

    all_method_results = {}
    all_method_build_stats = {}

    for method in methods:
        print(f"\n{'='*60}")
        print(f"Running {method.upper()}")
        print(f"{'='*60}")

        t0 = time.time()
        results, build_stats = run_baseline(method, scenes, verbose=True, data_dir=data_dir)
        elapsed = time.time() - t0

        all_method_results[method] = results
        all_method_build_stats[method] = build_stats

        # Print summary
        if results:
            macro, micro = compute_metrics(results)
            print(f"\n--- {method.upper()} RESULTS ({len(results)} queries, {elapsed/60:.1f}min) ---")
            for t in LOCALIZATION_THRESHOLDS:
                lo, hi = bootstrap_ci(results, t)
                print(f"  Loc@{t}m: {macro[t]*100:.1f}% [{lo*100:.1f}, {hi*100:.1f}] (macro)")

            errors = [r["error_m"] for r in results if r["error_m"] is not None]
            no_pred = sum(1 for r in results if r["error_m"] is None)
            if errors:
                print(f"  Median error: {np.median(errors):.2f}m")
                print(f"  Mean error: {np.mean(errors):.2f}m")
            print(f"  No-answer: {no_pred}/{len(results)}")

            # Per-category
            by_cat = defaultdict(list)
            for r in results:
                by_cat[r["query"]].append(r)
            print(f"\n  Per-category (Loc@1m):")
            for cat in sorted(by_cat.keys()):
                cat_r = by_cat[cat]
                correct = sum(1 for r in cat_r
                              if r["error_m"] is not None and r["error_m"] < 1.0)
                print(f"    {cat:10s}: {correct}/{len(cat_r)} ({100*correct/len(cat_r):.0f}%)")

    # Save all results
    output = {
        "metadata": {
            "dataset": "scannet",
            "num_scenes": len(scenes),
            "scenes": scenes,
            "methods": methods,
            "queries": JIT_QUERIES_10,
            "thresholds": LOCALIZATION_THRESHOLDS,
            "intrinsics": "per-scene ScanNet (HFOV≈58°)",
        },
    }

    for method, results in all_method_results.items():
        if not results:
            continue
        macro, micro = compute_metrics(results)
        cis = {}
        for t in LOCALIZATION_THRESHOLDS:
            lo, hi = bootstrap_ci(results, t)
            cis[str(t)] = {"lo": lo, "hi": hi}

        errors = [r["error_m"] for r in results if r["error_m"] is not None]

        # Build stats summary
        bstats = all_method_build_stats.get(method, [])
        if bstats:
            avg_build_s = np.mean([b["build_time_s"] for b in bstats])
            avg_storage_mb = np.mean([b["storage_mb"] for b in bstats])
        else:
            avg_build_s = None
            avg_storage_mb = None

        output[method] = {
            "per_query": results,
            "summary": {
                "macro": {str(t): v for t, v in macro.items()},
                "micro": {str(t): v for t, v in micro.items()},
                "bootstrap_ci": cis,
                "num_queries": len(results),
                "no_answer": sum(1 for r in results if r["error_m"] is None),
                "median_error": float(np.median(errors)) if errors else None,
                "mean_error": float(np.mean(errors)) if errors else None,
                "avg_build_time_s": avg_build_s,
                "avg_storage_mb": avg_storage_mb,
                "num_scenes_built": len(bstats),
            },
            "per_scene_build_stats": bstats,
        }

    results_path = SCANNET_RESULTS / (args.output or "scannet_dense_baselines_results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
