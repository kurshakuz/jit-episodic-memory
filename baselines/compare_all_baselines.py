#!/usr/bin/env python3
"""
Comprehensive Baseline Comparison
=================================

This script compares three approaches for vision-language object localization:

1. **VLMaps** (Huang et al., 2023): Dense per-pixel LSeg features projected to 2D grid
2. **DenseMap-CLIP**: Global CLIP embeddings projected to 3D voxels (simplified VLMaps)
3. **JIT Cascade** (Ours): Lazy evaluation with 3-level cascade

This provides a fair comparison by:
- Using same geometric processing (DBSCAN clustering)
- Isolating the feature extraction method (LSeg vs CLIP)
- Isolating the map representation (2D grid vs 3D voxels vs lazy)

Usage:
    python compare_all_baselines.py --output-dir outputs/baseline_comparison
    python compare_all_baselines.py --scene-limit 5  # Quick test
"""

import sys
import json
import time
import argparse
import gc
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Evaluation constants
TEST_QUERIES = [
    "toilet", "chair", "table", "bed", "couch",
    "sink", "lamp", "mirror", "cabinet", "shelf"
]

LOCALIZATION_THRESHOLDS = [0.5, 1.0, 2.0, 3.0]


@dataclass
class SceneResult:
    """Results for a single scene."""
    scene_id: str
    method: str
    build_time_seconds: float
    storage_mb: float
    num_frames: int
    query_results: List[Dict]
    avg_query_time_ms: float
    total_queries: int
    loc_at_05m: float
    loc_at_1m: float
    loc_at_2m: float
    loc_at_3m: float


@dataclass
class MethodSummary:
    """Summary for a method across all scenes."""
    method: str
    num_scenes: int
    avg_build_time: float
    avg_storage_mb: float
    avg_query_time_ms: float
    total_queries: int
    overall_loc_at_05m: float
    overall_loc_at_1m: float
    overall_loc_at_2m: float
    overall_loc_at_3m: float


def get_validation_scenes(base_dir: Path, max_scenes: Optional[int] = None) -> List[Path]:
    """Get validation scenes with ground truth."""
    multi_scene_dir = base_dir / "outputs" / "multi_scene_eval"
    
    scenes = []
    for scene_dir in sorted(multi_scene_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        
        trace_path = scene_dir / "exploration" / "trace.parquet"
        gt_path = scene_dir / f"{scene_dir.name}_ground_truth.json"
        
        if trace_path.exists() and gt_path.exists():
            scenes.append(scene_dir)
    
    scenes = scenes[:36]  # Validation split
    if max_scenes:
        scenes = scenes[:max_scenes]
    
    return scenes


def load_ground_truth(scene_dir: Path) -> Dict[str, np.ndarray]:
    """Load ground truth object locations.
    
    IMPORTANT: Filter to objects on the exploration floor to avoid
    multi-floor coordinate mismatches.
    """
    gt_path = scene_dir / f"{scene_dir.name}_ground_truth.json"
    trace_path = scene_dir / "exploration" / "trace.parquet"
    
    with open(gt_path) as f:
        gt_data = json.load(f)
    
    # Get camera Y range to identify exploration floor
    trace = pd.read_parquet(trace_path)
    cam_y_min = trace['y'].min()
    cam_y_max = trace['y'].max()
    
    # Objects on same floor: Y within range of camera Y + some margin
    # Camera is at agent Y, sensor adds 1.5m, objects can be up to ~3m tall
    floor_y_min = cam_y_min - 1.5  # Allow below camera
    floor_y_max = cam_y_max + 3.5  # Allow for tall objects
    
    if "objects" in gt_data:
        result = {}
        for obj_id, obj_info in gt_data["objects"].items():
            category = obj_info.get("category", "").lower()
            center = obj_info.get("center")
            
            if not category or not center or category == "unknown":
                continue
                
            # Check if object is on the exploration floor
            obj_y = center[1]
            if not (floor_y_min <= obj_y <= floor_y_max):
                continue
                
            # Take first instance of each category on this floor
            if category not in result:
                result[category] = np.array(center)
        return result
    
    return {k: np.array(v) for k, v in gt_data.items() if isinstance(v, list) and len(v) == 3}


def compute_error(predicted: Optional[np.ndarray], gt: np.ndarray) -> Optional[float]:
    """Compute localization error."""
    if predicted is None:
        return None
    return float(np.linalg.norm(predicted - gt))


def evaluate_vlmap(scene_dir: Path, queries: List[str], gt: Dict[str, np.ndarray], 
                   verbose: bool = False, use_lseg: bool = True) -> SceneResult:
    """Evaluate VLMaps baseline.
    
    Args:
        scene_dir: Path to scene data
        queries: List of object queries
        gt: Ground truth object locations
        verbose: Print progress
        use_lseg: If True, use real LSeg encoder. If False, use CLIP patch tokens.
    """
    from baselines.vlmap_baseline import VLMapBaseline
    
    method_name = "VLMaps-LSeg" if use_lseg else "VLMaps-CLIP"
    vlmap = VLMapBaseline(scene_dir, grid_resolution=0.05, grid_size=500, use_lseg=use_lseg)
    build_stats = vlmap.build_map(verbose=verbose)
    
    query_results = []
    for query in queries:
        if query not in gt:
            continue
        
        result = vlmap.query(query)
        error = compute_error(result.predicted_location, gt[query])
        
        query_results.append({
            "query": query,
            "success": result.success,
            "error_m": error,
            "query_time_ms": result.query_time_ms,
            "confidence": result.confidence,
        })
    
    return _compute_scene_result(scene_dir.name, method_name, build_stats, query_results)


def evaluate_dense_map(scene_dir: Path, queries: List[str], gt: Dict[str, np.ndarray], verbose: bool = False) -> SceneResult:
    """Evaluate DenseMap-CLIP baseline."""
    from baselines.dense_map import DenseMapBaseline
    
    dense_map = DenseMapBaseline(scene_dir, voxel_size=0.05, sample_stride=4)
    build_stats = dense_map.build_map(verbose=verbose)
    
    query_results = []
    for query in queries:
        if query not in gt:
            continue
        
        result = dense_map.query(query)
        error = compute_error(result.predicted_location, gt[query])
        
        query_results.append({
            "query": query,
            "success": result.success,
            "error_m": error,
            "query_time_ms": result.query_time_ms,
            "confidence": result.confidence,
        })
    
    return _compute_scene_result(scene_dir.name, "DenseMap-CLIP", build_stats, query_results)


def evaluate_jit(scene_dir: Path, queries: List[str], gt: Dict[str, np.ndarray], verbose: bool = False) -> SceneResult:
    """Evaluate JIT Cascade."""
    from baselines.dense_map import JITBaselineWrapper
    
    jit = JITBaselineWrapper(scene_dir)
    build_stats = jit.build_map()
    
    query_results = []
    for query in queries:
        if query not in gt:
            continue
        
        result = jit.query(query)
        error = compute_error(result.predicted_location, gt[query])
        
        query_results.append({
            "query": query,
            "success": result.success,
            "error_m": error,
            "query_time_ms": result.query_time_ms,
            "confidence": result.confidence,
        })
    
    return _compute_scene_result(scene_dir.name, "JIT-Cascade", build_stats, query_results)


def _compute_scene_result(scene_id: str, method: str, build_stats, query_results: List[Dict]) -> SceneResult:
    """Compute scene-level metrics."""
    total = len(query_results)
    if total == 0:
        return SceneResult(
            scene_id=scene_id, method=method,
            build_time_seconds=build_stats.build_time_seconds,
            storage_mb=build_stats.memory_size_mb,
            num_frames=build_stats.num_frames_processed,
            query_results=[], avg_query_time_ms=0, total_queries=0,
            loc_at_05m=0, loc_at_1m=0, loc_at_2m=0, loc_at_3m=0,
        )
    
    errors = [r["error_m"] for r in query_results if r["error_m"] is not None]
    n = len(errors) if errors else 1
    
    return SceneResult(
        scene_id=scene_id,
        method=method,
        build_time_seconds=build_stats.build_time_seconds,
        storage_mb=build_stats.memory_size_mb,
        num_frames=build_stats.num_frames_processed,
        query_results=query_results,
        avg_query_time_ms=np.mean([r["query_time_ms"] for r in query_results]),
        total_queries=total,
        loc_at_05m=sum(1 for e in errors if e <= 0.5) / total if errors else 0,
        loc_at_1m=sum(1 for e in errors if e <= 1.0) / total if errors else 0,
        loc_at_2m=sum(1 for e in errors if e <= 2.0) / total if errors else 0,
        loc_at_3m=sum(1 for e in errors if e <= 3.0) / total if errors else 0,
    )


def aggregate_results(results: List[SceneResult], method: str) -> MethodSummary:
    """Aggregate results across scenes."""
    if not results:
        return MethodSummary(method=method, num_scenes=0, avg_build_time=0,
                            avg_storage_mb=0, avg_query_time_ms=0, total_queries=0,
                            overall_loc_at_05m=0, overall_loc_at_1m=0,
                            overall_loc_at_2m=0, overall_loc_at_3m=0)
    
    # Collect all errors
    all_errors = []
    for r in results:
        for qr in r.query_results:
            if qr["error_m"] is not None:
                all_errors.append(qr["error_m"])
    
    total_queries = sum(r.total_queries for r in results)
    n = len(all_errors) if all_errors else 1
    
    return MethodSummary(
        method=method,
        num_scenes=len(results),
        avg_build_time=np.mean([r.build_time_seconds for r in results]),
        avg_storage_mb=np.mean([r.storage_mb for r in results]),
        avg_query_time_ms=sum(r.avg_query_time_ms * r.total_queries for r in results) / total_queries if total_queries > 0 else 0,
        total_queries=total_queries,
        overall_loc_at_05m=sum(1 for e in all_errors if e <= 0.5) / n if all_errors else 0,
        overall_loc_at_1m=sum(1 for e in all_errors if e <= 1.0) / n if all_errors else 0,
        overall_loc_at_2m=sum(1 for e in all_errors if e <= 2.0) / n if all_errors else 0,
        overall_loc_at_3m=sum(1 for e in all_errors if e <= 3.0) / n if all_errors else 0,
    )


def generate_plots(summaries: Dict[str, MethodSummary], per_scene: Dict[str, List[SceneResult]], output_dir: Path):
    """Generate comparison plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
    })
    
    # Colors
    colors = {
        "VLMaps-LSeg": "#3498DB",      # Blue
        "DenseMap-CLIP": "#E74C3C",  # Red
        "JIT-Cascade": "#27AE60",    # Green
    }
    
    methods = list(summaries.keys())
    
    # 1. Summary bar chart
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    
    # Storage
    ax = axes[0]
    vals = [summaries[m].avg_storage_mb for m in methods]
    bars = ax.bar(methods, vals, color=[colors[m] for m in methods])
    ax.set_ylabel('Storage (MB)')
    ax.set_title('Avg Storage per Scene')
    ax.set_yscale('log')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f'{val:.1f}', ha='center', fontsize=9)
    
    # Build time
    ax = axes[1]
    vals = [summaries[m].avg_build_time for m in methods]
    bars = ax.bar(methods, vals, color=[colors[m] for m in methods])
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Avg Pre-compute Time')
    ax.set_yscale('log')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f'{val:.2f}s', ha='center', fontsize=9)
    
    # Query time
    ax = axes[2]
    vals = [summaries[m].avg_query_time_ms for m in methods]
    bars = ax.bar(methods, vals, color=[colors[m] for m in methods])
    ax.set_ylabel('Time (ms)')
    ax.set_title('Avg Query Time')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{val:.0f}ms', ha='center', fontsize=9)
    
    # Accuracy
    ax = axes[3]
    vals = [summaries[m].overall_loc_at_1m * 100 for m in methods]
    bars = ax.bar(methods, vals, color=[colors[m] for m in methods])
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Loc@1m Accuracy')
    ax.set_ylim(0, max(vals) * 1.3 if vals else 100)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'summary_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    # 2. Accuracy at thresholds
    fig, ax = plt.subplots(figsize=(10, 6))
    
    thresholds = [0.5, 1.0, 2.0, 3.0]
    x = np.arange(len(thresholds))
    width = 0.25
    
    for i, method in enumerate(methods):
        s = summaries[method]
        vals = [s.overall_loc_at_05m * 100, s.overall_loc_at_1m * 100,
                s.overall_loc_at_2m * 100, s.overall_loc_at_3m * 100]
        offset = (i - len(methods)/2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=method, color=colors[method])
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}',
                            xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Localization Threshold (m)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Localization Accuracy at Different Thresholds')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{t}m' for t in thresholds])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_at_thresholds.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'accuracy_at_thresholds.pdf', bbox_inches='tight')
    plt.close()
    
    # 3. Storage vs Accuracy scatter
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method in methods:
        results = per_scene[method]
        storage = [r.storage_mb for r in results]
        acc = [r.loc_at_1m * 100 for r in results]
        ax.scatter(storage, acc, alpha=0.5, label=f'{method} (per-scene)', color=colors[method])
        
        # Summary point
        s = summaries[method]
        ax.scatter([s.avg_storage_mb], [s.overall_loc_at_1m * 100],
                   s=200, marker='*', edgecolors='black', linewidths=1.5,
                   color=colors[method], zorder=10)
    
    ax.set_xlabel('Storage per Scene (MB)')
    ax.set_ylabel('Loc@1m Accuracy (%)')
    ax.set_title('Accuracy vs Storage Tradeoff')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_vs_storage.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'accuracy_vs_storage.pdf', bbox_inches='tight')
    plt.close()
    
    # 4. Build time vs Accuracy scatter
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method in methods:
        results = per_scene[method]
        build_time = [r.build_time_seconds for r in results]
        acc = [r.loc_at_1m * 100 for r in results]
        ax.scatter(build_time, acc, alpha=0.5, label=f'{method}', color=colors[method])
        
        s = summaries[method]
        ax.scatter([s.avg_build_time], [s.overall_loc_at_1m * 100],
                   s=200, marker='*', edgecolors='black', linewidths=1.5,
                   color=colors[method], zorder=10)
    
    ax.set_xlabel('Pre-compute Time per Scene (seconds)')
    ax.set_ylabel('Loc@1m Accuracy (%)')
    ax.set_title('Accuracy vs Pre-compute Time')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_vs_precompute.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'accuracy_vs_precompute.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Generated plots in {output_dir}/")


def print_summary(summaries: Dict[str, MethodSummary]):
    """Print summary table."""
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    
    methods = list(summaries.keys())
    
    print(f"\n{'Metric':<25}", end="")
    for m in methods:
        print(f"{m:<20}", end="")
    print()
    print("-" * 80)
    
    print(f"{'Pre-compute (s)':<25}", end="")
    for m in methods:
        print(f"{summaries[m].avg_build_time:<20.3f}", end="")
    print()
    
    print(f"{'Storage (MB)':<25}", end="")
    for m in methods:
        print(f"{summaries[m].avg_storage_mb:<20.1f}", end="")
    print()
    
    print(f"{'Query time (ms)':<25}", end="")
    for m in methods:
        print(f"{summaries[m].avg_query_time_ms:<20.1f}", end="")
    print()
    
    print("-" * 80)
    
    print(f"{'Loc@0.5m (%)':<25}", end="")
    for m in methods:
        print(f"{summaries[m].overall_loc_at_05m * 100:<20.1f}", end="")
    print()
    
    print(f"{'Loc@1m (%)':<25}", end="")
    for m in methods:
        print(f"{summaries[m].overall_loc_at_1m * 100:<20.1f}", end="")
    print()
    
    print(f"{'Loc@2m (%)':<25}", end="")
    for m in methods:
        print(f"{summaries[m].overall_loc_at_2m * 100:<20.1f}", end="")
    print()
    
    print(f"{'Loc@3m (%)':<25}", end="")
    for m in methods:
        print(f"{summaries[m].overall_loc_at_3m * 100:<20.1f}", end="")
    print()
    
    # LaTeX table (only if all methods present)
    if len(summaries) == 3 and all(m in summaries for m in ['VLMaps-LSeg', 'DenseMap-CLIP', 'JIT-Cascade']):
        print("\n" + "-" * 80)
        print("LaTeX Table:")
        print(r"""
\begin{table}[h]
\centering
\caption{Comparison of Object Localization Methods}
\label{tab:baseline_comparison}
\begin{tabular}{lccc}
\toprule
\textbf{Metric} & \textbf{VLMaps-LSeg} & \textbf{DenseMap-CLIP} & \textbf{JIT Cascade} \\
\midrule""")
    
        print(f"Pre-compute (s) & {summaries['VLMaps-LSeg'].avg_build_time:.2f} & {summaries['DenseMap-CLIP'].avg_build_time:.2f} & {summaries['JIT-Cascade'].avg_build_time:.3f} \\\\")
        print(f"Storage (MB) & {summaries['VLMaps-LSeg'].avg_storage_mb:.1f} & {summaries['DenseMap-CLIP'].avg_storage_mb:.1f} & {summaries['JIT-Cascade'].avg_storage_mb:.1f} \\\\")
        print(f"Query time (ms) & {summaries['VLMaps-LSeg'].avg_query_time_ms:.0f} & {summaries['DenseMap-CLIP'].avg_query_time_ms:.0f} & {summaries['JIT-Cascade'].avg_query_time_ms:.0f} \\\\")
        print(r"\midrule")
        print(f"Loc@1m (\\%) & {summaries['VLMaps-LSeg'].overall_loc_at_1m * 100:.1f} & {summaries['DenseMap-CLIP'].overall_loc_at_1m * 100:.1f} & {summaries['JIT-Cascade'].overall_loc_at_1m * 100:.1f} \\\\")
        print(f"Loc@2m (\\%) & {summaries['VLMaps-LSeg'].overall_loc_at_2m * 100:.1f} & {summaries['DenseMap-CLIP'].overall_loc_at_2m * 100:.1f} & {summaries['JIT-Cascade'].overall_loc_at_2m * 100:.1f} \\\\")
        print(f"Loc@3m (\\%) & {summaries['VLMaps-LSeg'].overall_loc_at_3m * 100:.1f} & {summaries['DenseMap-CLIP'].overall_loc_at_3m * 100:.1f} & {summaries['JIT-Cascade'].overall_loc_at_3m * 100:.1f} \\\\")
        print(r"""\bottomrule
\end{tabular}
\end{table}
""")


def main():
    parser = argparse.ArgumentParser(description="Compare all baseline methods")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).parent.parent)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--scene-limit", type=int, default=None)
    parser.add_argument("--skip-vlmap", action="store_true", help="Skip VLMaps evaluation")
    parser.add_argument("--skip-dense", action="store_true", help="Skip DenseMap evaluation")
    parser.add_argument("--skip-jit", action="store_true", help="Skip JIT evaluation")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.base_dir / "outputs" / "baseline_comparison"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Comprehensive Baseline Comparison")
    print("=" * 80)
    
    # Get scenes
    scenes = get_validation_scenes(args.base_dir, args.scene_limit)
    print(f"\nEvaluating on {len(scenes)} scenes")
    
    # Results storage
    per_scene_results = {
        "VLMaps-LSeg": [],
        "DenseMap-CLIP": [],
        "JIT-Cascade": [],
    }
    
    for i, scene_dir in enumerate(scenes):
        print(f"\n[Scene {i+1}/{len(scenes)}] {scene_dir.name}")
        
        gt = load_ground_truth(scene_dir)
        available_queries = [q for q in TEST_QUERIES if q in gt]
        print(f"  Queries available: {len(available_queries)}")
        
        if not available_queries:
            continue
        
        # VLMaps with LSeg (real VLMaps)
        if not args.skip_vlmap:
            try:
                print(f"  [VLMaps-LSeg] Building...")
                result = evaluate_vlmap(scene_dir, TEST_QUERIES, gt, verbose=args.verbose, use_lseg=True)
                per_scene_results["VLMaps-LSeg"].append(result)
                print(f"  [VLMaps-LSeg] Build: {result.build_time_seconds:.1f}s, "
                      f"Storage: {result.storage_mb:.1f}MB, Loc@1m: {result.loc_at_1m*100:.1f}%")
            except Exception as e:
                print(f"  [VLMaps-LSeg] ERROR: {e}")
        
        # DenseMap-CLIP
        if not args.skip_dense:
            try:
                print(f"  [DenseMap-CLIP] Building...")
                result = evaluate_dense_map(scene_dir, TEST_QUERIES, gt, verbose=args.verbose)
                per_scene_results["DenseMap-CLIP"].append(result)
                print(f"  [DenseMap-CLIP] Build: {result.build_time_seconds:.1f}s, "
                      f"Storage: {result.storage_mb:.1f}MB, Loc@1m: {result.loc_at_1m*100:.1f}%")
            except Exception as e:
                print(f"  [DenseMap-CLIP] ERROR: {e}")
        
        # JIT Cascade
        if not args.skip_jit:
            try:
                print(f"  [JIT-Cascade] Loading...")
                result = evaluate_jit(scene_dir, TEST_QUERIES, gt, verbose=args.verbose)
                per_scene_results["JIT-Cascade"].append(result)
                print(f"  [JIT-Cascade] Build: {result.build_time_seconds:.3f}s, "
                      f"Storage: {result.storage_mb:.1f}MB, Loc@1m: {result.loc_at_1m*100:.1f}%")
            except Exception as e:
                print(f"  [JIT-Cascade] ERROR: {e}")
        
        # Clear GPU memory
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
    
    # Aggregate
    print("\n[Aggregating results...]")
    summaries = {}
    for method in per_scene_results:
        if per_scene_results[method]:
            summaries[method] = aggregate_results(per_scene_results[method], method)
    
    # Print summary
    if summaries:
        print_summary(summaries)
    
    # Save results
    results = {
        "summaries": {k: asdict(v) for k, v in summaries.items()},
        "per_scene": {
            method: [
                {
                    "scene_id": r.scene_id,
                    "build_time": r.build_time_seconds,
                    "storage_mb": r.storage_mb,
                    "loc_at_1m": r.loc_at_1m,
                    "loc_at_2m": r.loc_at_2m,
                    "avg_query_time_ms": r.avg_query_time_ms,
                }
                for r in results_list
            ]
            for method, results_list in per_scene_results.items()
        },
        "config": {
            "num_scenes": len(scenes),
            "queries": TEST_QUERIES,
        }
    }
    
    with open(args.output_dir / "evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate plots
    if len(summaries) >= 2:
        generate_plots(summaries, per_scene_results, args.output_dir)
    
    print(f"\n{'=' * 80}")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
