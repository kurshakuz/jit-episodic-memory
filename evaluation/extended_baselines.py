#!/usr/bin/env python3
"""
Extended Baselines with Depth Projection
==========================================

Implements additional baselines for fair comparison:
1. L1 + OWL-ViT + Depth (no DBSCAN) - Project best detection to 3D
2. Brute Force + Depth (no DBSCAN) - Project best detection from all frames
3. Mean/Median of Projected Points - Aggregate without clustering
"""

import os
import sys
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
from PIL import Image

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import shared components
from evaluation.full_eval_v2 import VISIBILITY_THRESHOLD, LOCALIZATION_THRESHOLDS, MethodResult, load_ground_truth, compute_localization_accuracy

# Import DepthProjector from retrieval module (avoid duplication)
from retrieval.level2_geometric import DepthProjector


def run_l1_owlvit_depth(trace_dir: Path, clip_encoder, owl_detector,
                         test_queries: List[str], gt_objects: Dict,
                         trace_df: pd.DataFrame, k: int = 20) -> MethodResult:
    """
    L1 + OWL-ViT + Depth (no DBSCAN).
    
    Use CLIP to get top-k frames, run OWL-ViT, project best detection to 3D.
    No clustering - just take the highest confidence projected point.
    """
    import faiss
    
    index_path = trace_dir / "memory.index"
    if not index_path.exists():
        return MethodResult()
    
    index = faiss.read_index(str(index_path))
    gt_objects_list = list(gt_objects.values())
    projector = DepthProjector()
    
    detection_tp, detection_fp = 0, 0
    loc_correct = {t: 0 for t in LOCALIZATION_THRESHOLDS}
    total_queries = 0
    latencies = []
    
    for query in test_queries:
        matching_gt = [obj for obj in gt_objects_list 
                      if query in obj.get('category', obj.get('name', '')).lower()]
        if not matching_gt:
            continue
        
        total_queries += 1
        start_time = time.time()
        
        # L1: CLIP retrieval
        query_embedding = clip_encoder.encode_text(query)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        D, I = index.search(query_embedding.reshape(1, -1).astype(np.float32), k)
        
        detection_success = False
        best_pred_3d = None
        best_confidence = -1
        
        for idx in I[0]:
            if idx < 0 or idx >= len(trace_df):
                continue
            
            row = trace_df.iloc[idx]
            image_path = trace_dir / row['image_path']
            depth_path = trace_dir / row['depth_path']
            
            if not image_path.exists() or not depth_path.exists():
                continue
            
            try:
                image = np.array(Image.open(image_path).convert('RGB'))
                depth = np.load(depth_path)
                
                position = np.array([row['x'], row['y'], row['z']])
                rotation = np.array([row['qw'], row['qx'], row['qy'], row['qz']])
                
                detections = owl_detector.detect(image, [query])
                
                if detections:
                    # Mark detection success
                    for gt_obj in matching_gt:
                        gt_loc = np.array(gt_obj.get('center', gt_obj.get('position', [0,0,0])))
                        if np.linalg.norm(position - gt_loc) < VISIBILITY_THRESHOLD:
                            detection_success = True
                            break
                    
                    # Project best detection to 3D
                    best_det = detections[0]  # Highest confidence
                    if best_det.score > best_confidence:
                        point_3d = projector.project_bbox_to_3d(
                            best_det.bbox, depth, position, rotation
                        )
                        if point_3d is not None:
                            best_pred_3d = point_3d
                            best_confidence = best_det.score
                            
            except Exception as e:
                continue
        
        latencies.append((time.time() - start_time) * 1000)
        
        if detection_success:
            detection_tp += 1
        else:
            detection_fp += 1
        
        if best_pred_3d is not None:
            loc_results = compute_localization_accuracy(best_pred_3d, matching_gt)
            for t, correct in loc_results.items():
                if correct:
                    loc_correct[t] += 1
    
    if total_queries == 0:
        return MethodResult()
    
    return MethodResult(
        detection_recall=100.0 * detection_tp / total_queries,
        detection_precision=100.0 * detection_tp / max(1, detection_tp + detection_fp),
        detection_tp=detection_tp,
        detection_fp=detection_fp,
        localization_recall_0_5m=100.0 * loc_correct[0.5] / total_queries,
        localization_recall_1m=100.0 * loc_correct[1.0] / total_queries,
        localization_recall_2m=100.0 * loc_correct[2.0] / total_queries,
        localization_recall_3m=100.0 * loc_correct[3.0] / total_queries,
        avg_latency_ms=np.mean(latencies) if latencies else 0,
        total_queries=total_queries
    )


def run_brute_force_depth(trace_dir: Path, owl_detector,
                           test_queries: List[str], gt_objects: Dict,
                           trace_df: pd.DataFrame, max_frames: int = 100) -> MethodResult:
    """
    Brute Force + Depth (no DBSCAN).
    
    Run OWL-ViT on all frames, project best detection to 3D.
    No clustering - return the single best projected point.
    """
    gt_objects_list = list(gt_objects.values())
    projector = DepthProjector()
    
    detection_tp, detection_fp = 0, 0
    loc_correct = {t: 0 for t in LOCALIZATION_THRESHOLDS}
    total_queries = 0
    latencies = []
    
    if len(trace_df) > max_frames:
        sample_indices = sorted(random.sample(range(len(trace_df)), max_frames))
    else:
        sample_indices = list(range(len(trace_df)))
    
    for query in test_queries:
        matching_gt = [obj for obj in gt_objects_list 
                      if query in obj.get('category', obj.get('name', '')).lower()]
        if not matching_gt:
            continue
        
        total_queries += 1
        start_time = time.time()
        
        detection_success = False
        best_pred_3d = None
        best_confidence = -1
        
        for idx in sample_indices:
            row = trace_df.iloc[idx]
            image_path = trace_dir / row['image_path']
            depth_path = trace_dir / row['depth_path']
            
            if not image_path.exists() or not depth_path.exists():
                continue
            
            try:
                image = np.array(Image.open(image_path).convert('RGB'))
                depth = np.load(depth_path)
                
                position = np.array([row['x'], row['y'], row['z']])
                rotation = np.array([row['qw'], row['qx'], row['qy'], row['qz']])
                
                detections = owl_detector.detect(image, [query])
                
                if detections:
                    # Mark detection success
                    for gt_obj in matching_gt:
                        gt_loc = np.array(gt_obj.get('center', gt_obj.get('position', [0,0,0])))
                        if np.linalg.norm(position - gt_loc) < VISIBILITY_THRESHOLD:
                            detection_success = True
                            break
                    
                    # Project best detection to 3D
                    best_det = detections[0]
                    if best_det.score > best_confidence:
                        point_3d = projector.project_bbox_to_3d(
                            best_det.bbox, depth, position, rotation
                        )
                        if point_3d is not None:
                            best_pred_3d = point_3d
                            best_confidence = best_det.score
                            
            except Exception as e:
                continue
        
        latencies.append((time.time() - start_time) * 1000)
        
        if detection_success:
            detection_tp += 1
        else:
            detection_fp += 1
        
        if best_pred_3d is not None:
            loc_results = compute_localization_accuracy(best_pred_3d, matching_gt)
            for t, correct in loc_results.items():
                if correct:
                    loc_correct[t] += 1
    
    if total_queries == 0:
        return MethodResult()
    
    return MethodResult(
        detection_recall=100.0 * detection_tp / total_queries,
        detection_precision=100.0 * detection_tp / max(1, detection_tp + detection_fp),
        detection_tp=detection_tp,
        detection_fp=detection_fp,
        localization_recall_0_5m=100.0 * loc_correct[0.5] / total_queries,
        localization_recall_1m=100.0 * loc_correct[1.0] / total_queries,
        localization_recall_2m=100.0 * loc_correct[2.0] / total_queries,
        localization_recall_3m=100.0 * loc_correct[3.0] / total_queries,
        avg_latency_ms=np.mean(latencies) if latencies else 0,
        total_queries=total_queries
    )


def run_mean_projection(trace_dir: Path, clip_encoder, owl_detector,
                         test_queries: List[str], gt_objects: Dict,
                         trace_df: pd.DataFrame, k: int = 20) -> MethodResult:
    """
    Mean of Projected Points (no DBSCAN).
    
    Get all detections from top-k frames, project all to 3D, return mean.
    """
    import faiss
    
    index_path = trace_dir / "memory.index"
    if not index_path.exists():
        return MethodResult()
    
    index = faiss.read_index(str(index_path))
    gt_objects_list = list(gt_objects.values())
    projector = DepthProjector()
    
    detection_tp, detection_fp = 0, 0
    loc_correct = {t: 0 for t in LOCALIZATION_THRESHOLDS}
    total_queries = 0
    latencies = []
    
    for query in test_queries:
        matching_gt = [obj for obj in gt_objects_list 
                      if query in obj.get('category', obj.get('name', '')).lower()]
        if not matching_gt:
            continue
        
        total_queries += 1
        start_time = time.time()
        
        # L1: CLIP retrieval
        query_embedding = clip_encoder.encode_text(query)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        D, I = index.search(query_embedding.reshape(1, -1).astype(np.float32), k)
        
        detection_success = False
        projected_points = []
        
        for idx in I[0]:
            if idx < 0 or idx >= len(trace_df):
                continue
            
            row = trace_df.iloc[idx]
            image_path = trace_dir / row['image_path']
            depth_path = trace_dir / row['depth_path']
            
            if not image_path.exists() or not depth_path.exists():
                continue
            
            try:
                image = np.array(Image.open(image_path).convert('RGB'))
                depth = np.load(depth_path)
                
                position = np.array([row['x'], row['y'], row['z']])
                rotation = np.array([row['qw'], row['qx'], row['qy'], row['qz']])
                
                detections = owl_detector.detect(image, [query])
                
                if detections:
                    for gt_obj in matching_gt:
                        gt_loc = np.array(gt_obj.get('center', gt_obj.get('position', [0,0,0])))
                        if np.linalg.norm(position - gt_loc) < VISIBILITY_THRESHOLD:
                            detection_success = True
                            break
                    
                    # Project ALL detections to 3D
                    for det in detections:
                        point_3d = projector.project_bbox_to_3d(
                            det.bbox, depth, position, rotation
                        )
                        if point_3d is not None:
                            projected_points.append(point_3d)
                            
            except Exception as e:
                continue
        
        latencies.append((time.time() - start_time) * 1000)
        
        if detection_success:
            detection_tp += 1
        else:
            detection_fp += 1
        
        # Return mean of all projected points
        if projected_points:
            mean_point = np.mean(projected_points, axis=0)
            loc_results = compute_localization_accuracy(mean_point, matching_gt)
            for t, correct in loc_results.items():
                if correct:
                    loc_correct[t] += 1
    
    if total_queries == 0:
        return MethodResult()
    
    return MethodResult(
        detection_recall=100.0 * detection_tp / total_queries,
        detection_precision=100.0 * detection_tp / max(1, detection_tp + detection_fp),
        detection_tp=detection_tp,
        detection_fp=detection_fp,
        localization_recall_0_5m=100.0 * loc_correct[0.5] / total_queries,
        localization_recall_1m=100.0 * loc_correct[1.0] / total_queries,
        localization_recall_2m=100.0 * loc_correct[2.0] / total_queries,
        localization_recall_3m=100.0 * loc_correct[3.0] / total_queries,
        avg_latency_ms=np.mean(latencies) if latencies else 0,
        total_queries=total_queries
    )


def test_single_scene():
    """Test baselines on a single scene."""
    from retrieval.level3_verification import OWLViTDetector
    from ingestion import CLIPEncoder
    
    print("Testing extended baselines on single scene...")
    
    # Get first val scene
    scenes_dir = Path('outputs/multi_scene_eval')
    hm3d_base = Path(os.environ.get('HM3D_DATA', '/path/to/scene_datasets/hm3d'))
    
    # Map scenes to splits
    split_mapping = {}
    for split in ['minival', 'val', 'train']:
        split_path = hm3d_base / split
        if split_path.exists():
            for d in split_path.iterdir():
                if d.is_dir() and '-' in d.name:
                    scene_id = d.name.split('-')[1]
                    split_mapping[scene_id] = split
    
    # Find first val scene
    val_scene = None
    for scene_dir in scenes_dir.iterdir():
        if scene_dir.is_dir():
            scene_id = scene_dir.name
            if split_mapping.get(scene_id) == 'val':
                val_scene = scene_dir
                break
    
    if val_scene is None:
        val_scene = list(scenes_dir.iterdir())[0]
    
    scene_id = val_scene.name
    trace_dir = val_scene / "exploration"
    gt_file = val_scene / f"{scene_id}_ground_truth.json"
    
    print(f"Testing on scene: {scene_id}")
    
    # Load data
    trace_df = pd.read_parquet(trace_dir / "trace.parquet")
    gt_objects = load_ground_truth(gt_file)
    
    print(f"  Frames: {len(trace_df)}, GT objects: {len(gt_objects)}")
    
    # Initialize models
    clip_encoder = CLIPEncoder()
    owl_detector = OWLViTDetector()
    
    # Test queries
    queries = ['chair', 'table', 'bed']
    
    print("\n--- L1 + OWL-ViT + Depth ---")
    result = run_l1_owlvit_depth(trace_dir, clip_encoder, owl_detector,
                                  queries, gt_objects, trace_df, k=20)
    print(f"  Det: {result.detection_recall:.1f}%, Loc@1m: {result.localization_recall_1m:.1f}%")
    
    print("\n--- Brute Force + Depth ---")
    result = run_brute_force_depth(trace_dir, owl_detector,
                                    queries, gt_objects, trace_df, max_frames=50)
    print(f"  Det: {result.detection_recall:.1f}%, Loc@1m: {result.localization_recall_1m:.1f}%")
    
    print("\n--- Mean Projection ---")
    result = run_mean_projection(trace_dir, clip_encoder, owl_detector,
                                  queries, gt_objects, trace_df, k=20)
    print(f"  Det: {result.detection_recall:.1f}%, Loc@1m: {result.localization_recall_1m:.1f}%")
    
    print("\n[OK] Single scene test complete!")


if __name__ == "__main__":
    test_single_scene()
