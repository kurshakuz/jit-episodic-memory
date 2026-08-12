#!/usr/bin/env python3
"""
Recompute VLMaps results using only TEST_QUERIES for fair comparison.

This filters the existing VLMaps results to only include the same 10 queries
used by DenseMap-CLIP and JIT-Cascade, ensuring an apples-to-apples comparison.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict

# Same TEST_QUERIES used by DenseMap and JIT
TEST_QUERIES = [
    'toilet', 'chair', 'table', 'bed', 'couch',
    'sink', 'lamp', 'mirror', 'cabinet', 'shelf'
]

LOCALIZATION_THRESHOLDS = [0.5, 1.0, 2.0, 3.0]


def filter_vlmap_results(input_path: Path, output_path: Path) -> Dict:
    """Filter VLMaps results to only include TEST_QUERIES."""
    
    print(f"Loading VLMaps results from: {input_path}")
    with open(input_path) as f:
        data = json.load(f)
    
    print(f"Original: {data['total_queries']} queries across {data['num_scenes']} scenes")
    print(f"Filtering to TEST_QUERIES: {TEST_QUERIES}")
    
    # Filter per-scene results
    filtered_scenes = []
    total_filtered_queries = 0
    all_errors = []
    all_query_times = []
    
    for scene in data['per_scene']:
        scene_id = scene['scene_id']
        
        # Filter query results to only TEST_QUERIES
        filtered_queries = []
        for qr in scene['query_results']:
            query = qr['query'].lower().strip()
            
            # Check if query matches any TEST_QUERY
            if query in TEST_QUERIES:
                filtered_queries.append(qr)
                if qr['error_m'] is not None:
                    all_errors.append(qr['error_m'])
                all_query_times.append(qr['query_time_ms'])
        
        if filtered_queries:
            # Recompute scene-level metrics
            scene_errors = [q['error_m'] for q in filtered_queries if q['error_m'] is not None]
            n = len(filtered_queries)
            
            loc_05m = sum(1 for e in scene_errors if e <= 0.5) / n if scene_errors else 0
            loc_1m = sum(1 for e in scene_errors if e <= 1.0) / n if scene_errors else 0
            loc_2m = sum(1 for e in scene_errors if e <= 2.0) / n if scene_errors else 0
            loc_3m = sum(1 for e in scene_errors if e <= 3.0) / n if scene_errors else 0
            
            filtered_scene = {
                'scene_id': scene_id,
                'method': 'VLMaps-LSeg',
                'build_time_seconds': scene['build_time_seconds'],
                'storage_mb': scene['storage_mb'],
                'num_frames': scene['num_frames'],
                'query_results': filtered_queries,
                'avg_query_time_ms': np.mean([q['query_time_ms'] for q in filtered_queries]),
                'total_queries': n,
                'loc_at_05m': loc_05m,
                'loc_at_1m': loc_1m,
                'loc_at_2m': loc_2m,
                'loc_at_3m': loc_3m,
            }
            filtered_scenes.append(filtered_scene)
            total_filtered_queries += n
    
    # Compute overall metrics
    n_total = len(all_errors)
    overall_loc_05m = sum(1 for e in all_errors if e <= 0.5) / n_total if n_total > 0 else 0
    overall_loc_1m = sum(1 for e in all_errors if e <= 1.0) / n_total if n_total > 0 else 0
    overall_loc_2m = sum(1 for e in all_errors if e <= 2.0) / n_total if n_total > 0 else 0
    overall_loc_3m = sum(1 for e in all_errors if e <= 3.0) / n_total if n_total > 0 else 0
    
    avg_query_time = np.mean(all_query_times) if all_query_times else 0
    avg_build_time = np.mean([s['build_time_seconds'] for s in filtered_scenes])
    avg_storage = np.mean([s['storage_mb'] for s in filtered_scenes])
    
    # Build filtered summary
    filtered_data = {
        'method': 'VLMaps-LSeg',
        'query_set': 'TEST_QUERIES (10 categories)',
        'test_queries': TEST_QUERIES,
        'num_scenes': len(filtered_scenes),
        'total_queries': total_filtered_queries,
        'avg_build_time': avg_build_time,
        'avg_storage_mb': avg_storage,
        'avg_query_time_ms': avg_query_time,
        'loc_at_05m': overall_loc_05m,
        'loc_at_1m': overall_loc_1m,
        'loc_at_2m': overall_loc_2m,
        'loc_at_3m': overall_loc_3m,
        'per_scene': filtered_scenes,
    }
    
    # Save filtered results
    with open(output_path, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    print("\nFiltered results:")
    print(f"  Scenes: {len(filtered_scenes)}")
    print(f"  Queries: {total_filtered_queries} (was {data['total_queries']})")
    print(f"  Avg build time: {avg_build_time:.1f}s")
    print(f"  Avg storage: {avg_storage:.1f}MB")
    print(f"  Avg query time: {avg_query_time:.1f}ms")
    print(f"  Loc@0.5m: {overall_loc_05m*100:.1f}%")
    print(f"  Loc@1m: {overall_loc_1m*100:.1f}%")
    print(f"  Loc@2m: {overall_loc_2m*100:.1f}%")
    print(f"  Loc@3m: {overall_loc_3m*100:.1f}%")
    
    print(f"\nSaved to: {output_path}")
    
    return filtered_data


def load_other_results(results_path: Path) -> Dict:
    """Load DenseMap and JIT results."""
    with open(results_path) as f:
        return json.load(f)


def print_comparison_table(vlmap: Dict, densemap: Dict, jit: Dict):
    """Print comparison table with all three methods."""
    
    print("COMPREHENSIVE COMPARISON (All methods on TEST_QUERIES)")
    
    print(f"\n{'Metric':<25} {'VLMaps-LSeg':<18} {'DenseMap-CLIP':<18} {'JIT-Cascade':<18}")
    print("-" * 80)
    
    # Build/storage metrics
    print(f"{'Pre-compute (s)':<25} {vlmap['avg_build_time']:<18.1f} {densemap['avg_build_time']:<18.1f} {jit['avg_build_time']:<18.3f}")
    print(f"{'Storage (MB)':<25} {vlmap['avg_storage_mb']:<18.1f} {densemap['avg_storage_mb']:<18.1f} {jit['avg_storage_mb']:<18.1f}")
    print(f"{'Query time (ms)':<25} {vlmap['avg_query_time_ms']:<18.1f} {densemap['avg_query_time_ms']:<18.1f} {jit['avg_query_time_ms']:<18.1f}")
    
    print("-" * 80)
    
    # Localization metrics
    print(f"{'Loc@0.5m (%)':<25} {vlmap['loc_at_05m']*100:<18.1f} {densemap['overall_loc_at_05m']*100:<18.1f} {jit['overall_loc_at_05m']*100:<18.1f}")
    print(f"{'Loc@1m (%)':<25} {vlmap['loc_at_1m']*100:<18.1f} {densemap['overall_loc_at_1m']*100:<18.1f} {jit['overall_loc_at_1m']*100:<18.1f}")
    print(f"{'Loc@2m (%)':<25} {vlmap['loc_at_2m']*100:<18.1f} {densemap['overall_loc_at_2m']*100:<18.1f} {jit['overall_loc_at_2m']*100:<18.1f}")
    print(f"{'Loc@3m (%)':<25} {vlmap['loc_at_3m']*100:<18.1f} {densemap['overall_loc_at_3m']*100:<18.1f} {jit['overall_loc_at_3m']*100:<18.1f}")
    
    print("-" * 80)
    print(f"{'Total queries':<25} {vlmap['total_queries']:<18} {densemap['total_queries']:<18} {jit['total_queries']:<18}")
    print(f"{'Scenes':<25} {vlmap['num_scenes']:<18} {densemap['num_scenes']:<18} {jit['num_scenes']:<18}")
    
    # Efficiency ratios
    print("EFFICIENCY ANALYSIS (vs JIT-Cascade)")
    
    vlmap_storage_ratio = vlmap['avg_storage_mb'] / jit['avg_storage_mb']
    densemap_storage_ratio = densemap['avg_storage_mb'] / jit['avg_storage_mb']
    
    vlmap_build_ratio = vlmap['avg_build_time'] / max(jit['avg_build_time'], 0.001)
    densemap_build_ratio = densemap['avg_build_time'] / max(jit['avg_build_time'], 0.001)
    
    jit_query_slower_vlmap = jit['avg_query_time_ms'] / vlmap['avg_query_time_ms']
    jit_query_slower_dense = jit['avg_query_time_ms'] / densemap['avg_query_time_ms']
    
    print("\nStorage overhead:")
    print(f"  VLMaps uses {vlmap_storage_ratio:.0f}× more storage than JIT")
    print(f"  DenseMap uses {densemap_storage_ratio:.0f}× more storage than JIT")
    
    print("\nPre-compute time:")
    print(f"  VLMaps takes {vlmap_build_ratio:.0f}× longer to build than JIT")
    print(f"  DenseMap takes {densemap_build_ratio:.0f}× longer to build than JIT")
    
    print("\nQuery time trade-off:")
    print(f"  JIT is {jit_query_slower_vlmap:.1f}× slower than VLMaps at query time")
    print(f"  JIT is {jit_query_slower_dense:.1f}× slower than DenseMap at query time")
    
    print("\nAccuracy improvement (Loc@1m):")
    jit_vs_vlmap = jit['overall_loc_at_1m'] / max(vlmap['loc_at_1m'], 0.001)
    jit_vs_dense = jit['overall_loc_at_1m'] / max(densemap['overall_loc_at_1m'], 0.001)
    print(f"  JIT is {jit_vs_vlmap:.1f}× more accurate than VLMaps")
    print(f"  JIT is {jit_vs_dense:.1f}× more accurate than DenseMap")


def main():
    base_dir = Path(__file__).parent.parent
    input_path = base_dir / "outputs" / "full_validation" / "vlmap_results.json"
    output_path = base_dir / "outputs" / "full_validation" / "vlmap_results_filtered.json"
    other_results_path = base_dir / "outputs" / "full_validation" / "evaluation_results.json"
    
    # Filter VLMaps results
    vlmap_filtered = filter_vlmap_results(input_path, output_path)
    
    # Load other results for comparison
    if other_results_path.exists():
        other_data = load_other_results(other_results_path)
        densemap = other_data['summaries']['DenseMap-CLIP']
        jit = other_data['summaries']['JIT-Cascade']
        
        print_comparison_table(vlmap_filtered, densemap, jit)
    else:
        print(f"\nNote: {other_results_path} not found, skipping comparison")


if __name__ == "__main__":
    main()
