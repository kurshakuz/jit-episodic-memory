#!/usr/bin/env python3
"""
Build ground truth from ScanNet mesh annotations.

For each scene:
1. Load _vh_clean_2.ply -> vertex positions
2. Load _vh_clean_2.0.010000.segs.json -> vertex-to-segment mapping
3. Load .aggregation.json -> segment-to-instance with labels
4. Compute per-instance AABB centroids
5. Filter by JIT query vocabulary
6. Save as JSON in JIT-compatible format

Usage:
    python scannet/build_gt.py                     # All 20 scenes
    python scannet/build_gt.py --scenes scene0568_00  # Single scene
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from scannet.config import SCANNET_VAL_SCENES, SCANNET_SCANS, SCANNET_JIT, JIT_QUERIES_10, SCANNET_SYNONYMS


def load_ply_vertices(ply_path: str) -> np.ndarray:
    """
    Load vertex positions from a binary PLY file.
    Returns: (N, 3) float32 array of [x, y, z] positions.
    """
    with open(ply_path, 'rb') as f:
        # Parse header
        header_lines = []
        num_vertices = 0
        header_end = False
        while not header_end:
            line = f.readline().decode('ascii', errors='replace').strip()
            header_lines.append(line)
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            if line == 'end_header':
                header_end = True

        # Determine vertex format from header
        # ScanNet _vh_clean_2.ply typically has:
        # property float x, y, z, (possibly nx, ny, nz, red, green, blue, alpha)
        props = []
        in_vertex = False
        for line in header_lines:
            if line.startswith('element vertex'):
                in_vertex = True
                continue
            if in_vertex and line.startswith('property'):
                parts = line.split()
                dtype = parts[1]
                name = parts[2]
                props.append((name, dtype))
            elif in_vertex and line.startswith('element'):
                in_vertex = False

        # Build numpy dtype for binary reading
        dtype_map = {
            'float': np.float32, 'float32': np.float32,
            'double': np.float64, 'float64': np.float64,
            'uchar': np.uint8, 'uint8': np.uint8,
            'char': np.int8, 'int8': np.int8,
            'ushort': np.uint16, 'uint16': np.uint16,
            'short': np.int16, 'int16': np.int16,
            'uint': np.uint32, 'uint32': np.uint32,
            'int': np.int32, 'int32': np.int32,
        }

        vertex_dtype = np.dtype([(name, dtype_map[dt]) for name, dt in props])
        
        # Read vertex data
        vertex_data = np.frombuffer(f.read(num_vertices * vertex_dtype.itemsize),
                                     dtype=vertex_dtype)

        # We might also need to skip face data, but we only need vertices
        vertices = np.column_stack([
            vertex_data['x'], vertex_data['y'], vertex_data['z']
        ]).astype(np.float32)

    return vertices


def load_segments(segs_path: str) -> np.ndarray:
    """
    Load mesh over-segmentation: maps each vertex to a segment ID.
    Returns: (N,) int array where segIndices[i] = segment ID of vertex i.
    """
    with open(segs_path, 'r') as f:
        data = json.load(f)
    return np.array(data['segIndices'], dtype=np.int64)


def load_aggregation(agg_path: str):
    """
    Load instance annotations: maps segments to semantic objects.
    Returns: list of dicts with keys 'id', 'objectId', 'label', 'segments'
    """
    with open(agg_path, 'r') as f:
        data = json.load(f)
    return data['segGroups']


def matches_query(label: str, query: str) -> bool:
    """
    Check if a ScanNet label matches a JIT query using substring matching.
    Same protocol as HM3D evaluation.
    """
    label_lower = label.lower().strip()
    query_lower = query.lower().strip()

    # Direct substring match
    if query_lower in label_lower:
        return True

    # Check synonyms
    synonyms = SCANNET_SYNONYMS.get(query_lower, [])
    for syn in synonyms:
        if syn.lower() in label_lower:
            return True

    return False


def build_scene_gt(scene_id: str, scans_dir: Path) -> dict:
    """
    Build ground truth for a single ScanNet scene.
    
    Returns dict in JIT-compatible format:
    {
        "scene_id": str,
        "dataset": "scannet",
        "objects": {
            "obj_id": {
                "id": str,
                "category": str,       # raw ScanNet label
                "center": [x, y, z],   # AABB centroid
                "num_vertices": int,
                "matched_queries": [str],  # which JIT queries match
            },
            ...
        },
        "query_stats": {
            "query": {"num_instances": int, "instance_ids": [str]},
            ...
        }
    }
    """
    scene_dir = scans_dir / scene_id
    
    # File paths
    ply_path = scene_dir / f"{scene_id}_vh_clean_2.ply"
    segs_path = scene_dir / f"{scene_id}_vh_clean_2.0.010000.segs.json"
    agg_path = scene_dir / f"{scene_id}.aggregation.json"

    for p in [ply_path, segs_path, agg_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    # Load data
    print(f"  Loading mesh vertices from {ply_path.name}...")
    vertices = load_ply_vertices(str(ply_path))
    print(f"    {len(vertices)} vertices loaded")

    print(f"  Loading segmentation from {segs_path.name}...")
    seg_indices = load_segments(str(segs_path))
    assert len(seg_indices) == len(vertices), \
        f"Vertex count mismatch: {len(vertices)} vertices vs {len(seg_indices)} segments"

    print(f"  Loading aggregation from {agg_path.name}...")
    seg_groups = load_aggregation(str(agg_path))
    print(f"    {len(seg_groups)} instances found")

    # Build segment-to-vertex mapping
    seg_to_verts = defaultdict(list)
    for vid, sid in enumerate(seg_indices):
        seg_to_verts[sid].append(vid)

    # Build per-instance data
    objects = {}
    for group in seg_groups:
        obj_id = str(group.get('objectId', group.get('id', 0)))
        label = group.get('label', 'unknown')
        segments = group.get('segments', [])

        # Collect all vertices for this instance
        vert_ids = []
        for seg_id in segments:
            vert_ids.extend(seg_to_verts.get(seg_id, []))

        if len(vert_ids) == 0:
            continue

        instance_verts = vertices[vert_ids]

        # AABB centroid
        bbox_min = instance_verts.min(axis=0)
        bbox_max = instance_verts.max(axis=0)
        center = ((bbox_min + bbox_max) / 2.0).tolist()

        # Check which queries match
        matched = [q for q in JIT_QUERIES_10 if matches_query(label, q)]

        objects[obj_id] = {
            "id": obj_id,
            "category": label,
            "center": center,
            "num_vertices": len(vert_ids),
            "matched_queries": matched,
            "bbox_min": bbox_min.tolist(),
            "bbox_max": bbox_max.tolist(),
        }

    # Build query stats
    query_stats = {}
    for query in JIT_QUERIES_10:
        matching_ids = [oid for oid, obj in objects.items()
                        if query in obj["matched_queries"]]
        if matching_ids:
            query_stats[query] = {
                "num_instances": len(matching_ids),
                "instance_ids": matching_ids,
            }

    return {
        "scene_id": scene_id,
        "dataset": "scannet",
        "objects": objects,
        "query_stats": query_stats,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build ScanNet ground truth")
    parser.add_argument("--scenes", type=str, default=None,
                        help="Comma-separated scene IDs (default: all 20)")
    args = parser.parse_args()

    scenes = SCANNET_VAL_SCENES
    if args.scenes:
        scenes = [s.strip() for s in args.scenes.split(",")]

    os.makedirs(str(SCANNET_JIT), exist_ok=True)

    total_queries = 0
    total_instances = 0
    scene_summary = []

    for i, scene_id in enumerate(scenes):
        print(f"\n[{i+1}/{len(scenes)}] Building GT for {scene_id}")
        try:
            gt = build_scene_gt(scene_id, SCANNET_SCANS)

            # Save GT
            scene_jit_dir = SCANNET_JIT / scene_id
            os.makedirs(str(scene_jit_dir), exist_ok=True)
            gt_path = scene_jit_dir / f"{scene_id}_ground_truth.json"
            with open(gt_path, 'w') as f:
                json.dump(gt, f, indent=2)
            print(f"  Saved to {gt_path}")

            # Summary
            n_objects = len(gt["objects"])
            n_queries = len(gt["query_stats"])
            n_instances = sum(v["num_instances"] for v in gt["query_stats"].values())
            total_queries += n_queries
            total_instances += n_instances

            matched_cats = sorted(gt["query_stats"].keys())
            print(f"  Objects: {n_objects}, Matched categories: {n_queries}")
            print(f"  Categories: {matched_cats}")
            for cat in matched_cats:
                info = gt["query_stats"][cat]
                print(f"    {cat}: {info['num_instances']} instances")

            scene_summary.append({
                "scene_id": scene_id,
                "total_objects": n_objects,
                "matched_categories": n_queries,
                "matched_instances": n_instances,
                "categories": matched_cats,
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Print overall summary
    print(f"\n{'='*60}")
    print(f"GT Construction Summary")
    print(f"{'='*60}")
    print(f"Scenes processed: {len(scene_summary)}/{len(scenes)}")
    print(f"Total matchable categories across scenes: {total_queries}")
    print(f"Total matchable instances: {total_instances}")

    # Category coverage
    cat_coverage = defaultdict(int)
    for s in scene_summary:
        for c in s["categories"]:
            cat_coverage[c] += 1
    print(f"\nCategory coverage (scenes with ≥1 instance):")
    for cat, count in sorted(cat_coverage.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}/{len(scene_summary)} scenes")

    # Save summary
    summary_path = SCANNET_JIT / "gt_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "scenes": scene_summary,
            "category_coverage": dict(cat_coverage),
            "total_instances": total_instances,
        }, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
