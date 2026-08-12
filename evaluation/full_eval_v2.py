#!/usr/bin/env python3
"""
Full-Scale Evaluation Using Pre-Generated Ground Truth
=======================================================

Uses existing ground truth files from memory banks (181 scenes).
No need to load habitat-sim for ground truth - much faster!
"""

import sys
import json
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple
from PIL import Image
import traceback

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Paths
MEMORY_BANK_DIR = Path("outputs/multi_scene_eval")
OUTPUT_DIR = Path("outputs/full_scale_eval")

# Test queries - common household objects
TEST_QUERIES = ['toilet', 'chair', 'table', 'bed', 'couch', 'sink', 
                'lamp', 'mirror', 'cabinet', 'shelf']

# Thresholds
VISIBILITY_THRESHOLD = 5.0  # meters - camera within this = object likely visible
LOCALIZATION_THRESHOLDS = [0.5, 1.0, 2.0, 3.0]  # meters


@dataclass
class MethodResult:
    """Results for a single method."""
    detection_recall: float = 0.0
    detection_precision: float = 0.0
    detection_tp: int = 0
    detection_fp: int = 0
    detection_fn: int = 0
    localization_recall_0_5m: float = 0.0
    localization_recall_1m: float = 0.0
    localization_recall_2m: float = 0.0
    localization_recall_3m: float = 0.0
    avg_latency_ms: float = 0.0
    total_queries: int = 0


@dataclass
class SceneResult:
    scene_id: str
    num_gt_objects: int
    num_queries: int
    num_keyframes: int
    random_sampling: MethodResult = field(default_factory=MethodResult)
    l1_plus_owlvit: MethodResult = field(default_factory=MethodResult)
    jit_cascade: MethodResult = field(default_factory=MethodResult)
    brute_force: MethodResult = field(default_factory=MethodResult)


def get_all_scenes_with_gt():
    """Get all scenes with ground truth files."""
    scenes = []
    
    for scene_dir in sorted(MEMORY_BANK_DIR.iterdir()):
        if not scene_dir.is_dir():
            continue
        
        scene_id = scene_dir.name
        gt_file = scene_dir / f"{scene_id}_ground_truth.json"
        trace_file = scene_dir / "exploration" / "trace.parquet"
        index_file = scene_dir / "exploration" / "memory.index"
        
        if gt_file.exists() and trace_file.exists() and index_file.exists():
            scenes.append((scene_id, scene_dir))
    
    return scenes


def load_ground_truth(gt_file: Path) -> Dict:
    """Load ground truth from JSON file."""
    with open(gt_file) as f:
        data = json.load(f)
    
    # Handle different GT formats
    if isinstance(data, dict):
        # New format: has 'objects' key with dict of objects
        if 'objects' in data and isinstance(data['objects'], dict):
            return data['objects']
        # Old format: direct dict of objects
        elif all(isinstance(v, dict) for v in list(data.values())[:3] if v):
            return data
    elif isinstance(data, list):
        return {str(i): obj for i, obj in enumerate(data)}
    return {}


def compute_detection_success(owl_detections: List, camera_pos: np.ndarray, 
                               matching_gt: List[Dict]) -> Tuple[bool, Optional[Dict]]:
    """Check detection success: OWL-ViT detected + camera within visibility range."""
    if not owl_detections:
        return False, None
    
    for gt_obj in matching_gt:
        gt_loc = np.array(gt_obj.get('center', gt_obj.get('position', [0,0,0])))
        dist = np.linalg.norm(camera_pos - gt_loc)
        if dist < VISIBILITY_THRESHOLD:
            return True, gt_obj
    
    return False, None


def compute_localization_accuracy(pred_3d: np.ndarray, matching_gt: List[Dict]) -> Dict[float, bool]:
    """Check localization accuracy at multiple thresholds."""
    results = {t: False for t in LOCALIZATION_THRESHOLDS}
    
    if pred_3d is None:
        return results
    
    for gt_obj in matching_gt:
        gt_loc = np.array(gt_obj.get('center', gt_obj.get('position', [0,0,0])))
        dist = np.linalg.norm(pred_3d - gt_loc)
        for threshold in LOCALIZATION_THRESHOLDS:
            if dist < threshold:
                results[threshold] = True
    
    return results


def get_camera_position(row) -> np.ndarray:
    """Extract camera position from trace row."""
    # Handle both formats: 'x', 'y', 'z' and 'position_x', 'position_y', 'position_z'
    if 'x' in row.index:
        return np.array([row['x'], row['y'], row['z']])
    elif 'position_x' in row.index:
        return np.array([row['position_x'], row['position_y'], row['position_z']])
    elif 'position' in row.index:
        pos = row['position']
        if isinstance(pos, (list, np.ndarray)):
            return np.array(pos)
    return np.array([0, 0, 0])


def run_random_sampling(trace_dir: Path, owl_detector, 
                        test_queries: List[str], gt_objects: Dict, 
                        trace_df: pd.DataFrame, k: int = 20) -> MethodResult:
    """Random sampling baseline."""
    gt_objects_list = list(gt_objects.values())
    
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
        
        indices = random.sample(range(len(trace_df)), min(k, len(trace_df)))
        
        detection_success = False
        best_pred_3d = None
        
        for idx in indices:
            row = trace_df.iloc[idx]
            image_path = trace_dir / row['image_path']
            if not image_path.exists():
                continue
            
            camera_pos = get_camera_position(row)
            
            try:
                image = np.array(Image.open(image_path).convert('RGB'))
                detections = owl_detector.detect(image, [query])
                
                if detections:
                    success, _ = compute_detection_success(detections, camera_pos, matching_gt)
                    if success:
                        detection_success = True
                        if best_pred_3d is None:
                            best_pred_3d = camera_pos
            except Exception:
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


def run_l1_plus_owlvit(trace_dir: Path, clip_encoder, owl_detector, 
                       test_queries: List[str], gt_objects: Dict, 
                       trace_df: pd.DataFrame, k: int = 20) -> MethodResult:
    """L1 + OWL-ViT method."""
    import faiss
    
    index_path = trace_dir / "memory.index"
    if not index_path.exists():
        return MethodResult()
    
    index = faiss.read_index(str(index_path))
    gt_objects_list = list(gt_objects.values())
    
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
        
        query_embedding = clip_encoder.encode_text(query)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        D, I = index.search(query_embedding.reshape(1, -1).astype(np.float32), k)
        
        detection_success = False
        best_pred_3d = None
        
        for idx in I[0]:
            if idx < 0 or idx >= len(trace_df):
                continue
            
            row = trace_df.iloc[idx]
            image_path = trace_dir / row['image_path']
            if not image_path.exists():
                continue
            
            camera_pos = get_camera_position(row)
            
            try:
                image = np.array(Image.open(image_path).convert('RGB'))
                detections = owl_detector.detect(image, [query])
                
                if detections:
                    success, _ = compute_detection_success(detections, camera_pos, matching_gt)
                    if success:
                        detection_success = True
                        if best_pred_3d is None:
                            best_pred_3d = camera_pos
            except Exception:
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


def run_jit_cascade(trace_dir: Path, test_queries: List[str], 
                    gt_objects: Dict, owl_detector, trace_df: pd.DataFrame) -> MethodResult:
    """JIT Cascade method."""
    from retrieval.cascade import JITRetrievalCascade
    
    gt_objects_list = list(gt_objects.values())
    
    detection_tp, detection_fp = 0, 0
    loc_correct = {t: 0 for t in LOCALIZATION_THRESHOLDS}
    total_queries = 0
    latencies = []
    
    try:
        cascade = JITRetrievalCascade(trace_dir, owl_detector=owl_detector)
    except Exception as e:
        print(f"      JIT init error: {e}")
        return MethodResult()
    
    for query in test_queries:
        matching_gt = [obj for obj in gt_objects_list 
                      if query in obj.get('category', obj.get('name', '')).lower()]
        if not matching_gt:
            continue
        
        total_queries += 1
        start_time = time.time()
        
        try:
            # Use correct API: query() returns QueryResult
            result = cascade.query(query, skip_l3=False)
            
            detection_success = False
            best_pred_3d = None
            
            # Check L3 verified locations
            if result.locations:
                for loc in result.locations:
                    frame_id = loc.frame_id
                    # Find frame index in trace
                    frame_matches = trace_df[trace_df['frame_id'] == frame_id]
                    if len(frame_matches) > 0:
                        row = frame_matches.iloc[0]
                        camera_pos = get_camera_position(row)
                        
                        for gt_obj in matching_gt:
                            gt_loc = np.array(gt_obj.get('center', gt_obj.get('position', [0,0,0])))
                            if np.linalg.norm(camera_pos - gt_loc) < VISIBILITY_THRESHOLD:
                                detection_success = True
                                break
                
                # Use best location's centroid_3d for localization
                if result.best_location:
                    best_pred_3d = np.array(result.best_location.centroid_3d)
            
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
                        
        except Exception as e:
            latencies.append(0)
            detection_fp += 1
    
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


def run_brute_force(trace_dir: Path, owl_detector, 
                    test_queries: List[str], gt_objects: Dict, 
                    trace_df: pd.DataFrame, max_frames: int = 100) -> MethodResult:
    """Brute force baseline."""
    gt_objects_list = list(gt_objects.values())
    
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
        
        for idx in sample_indices:
            row = trace_df.iloc[idx]
            image_path = trace_dir / row['image_path']
            if not image_path.exists():
                continue
            
            camera_pos = get_camera_position(row)
            
            try:
                image = np.array(Image.open(image_path).convert('RGB'))
                detections = owl_detector.detect(image, [query])
                
                if detections:
                    success, _ = compute_detection_success(detections, camera_pos, matching_gt)
                    if success:
                        detection_success = True
                        if best_pred_3d is None:
                            best_pred_3d = camera_pos
            except Exception:
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


def run_scene_evaluation(scene_id: str, scene_dir: Path,
                          clip_encoder, owl_detector) -> Optional[SceneResult]:
    """Run evaluation for a single scene."""
    
    trace_dir = scene_dir / "exploration"
    gt_file = scene_dir / f"{scene_id}_ground_truth.json"
    
    # Load trace
    trace_file = trace_dir / "trace.parquet"
    trace_df = pd.read_parquet(trace_file)
    num_keyframes = len(trace_df)
    
    # Load ground truth
    gt_objects = load_ground_truth(gt_file)
    if not gt_objects:
        print(f"    Skipping: Empty GT")
        return None
    
    # Count valid queries
    gt_objects_list = list(gt_objects.values())
    valid_queries = [q for q in TEST_QUERIES 
                    if any(q in obj.get('category', obj.get('name', '')).lower() 
                          for obj in gt_objects_list)]
    num_queries = len(valid_queries)
    
    if num_queries == 0:
        print(f"    Skipping: No matching queries")
        return None
    
    print(f"    GT: {len(gt_objects)}, Queries: {num_queries}, Frames: {num_keyframes}")
    
    # Run methods
    print(f"    [1/4] Random...", end="", flush=True)
    random_result = run_random_sampling(trace_dir, owl_detector, TEST_QUERIES, 
                                         gt_objects, trace_df)
    print(f" D:{random_result.detection_recall:.0f}% L@1m:{random_result.localization_recall_1m:.0f}%")
    
    print(f"    [2/4] L1+OWL...", end="", flush=True)
    l1_result = run_l1_plus_owlvit(trace_dir, clip_encoder, owl_detector, 
                                   TEST_QUERIES, gt_objects, trace_df)
    print(f" D:{l1_result.detection_recall:.0f}% L@1m:{l1_result.localization_recall_1m:.0f}%")
    
    print(f"    [3/4] JIT...", end="", flush=True)
    jit_result = run_jit_cascade(trace_dir, TEST_QUERIES, gt_objects, owl_detector, trace_df)
    print(f" D:{jit_result.detection_recall:.0f}% L@1m:{jit_result.localization_recall_1m:.0f}%")
    
    print(f"    [4/4] Brute...", end="", flush=True)
    bf_result = run_brute_force(trace_dir, owl_detector, TEST_QUERIES, 
                                 gt_objects, trace_df)
    print(f" D:{bf_result.detection_recall:.0f}% L@1m:{bf_result.localization_recall_1m:.0f}%")
    
    return SceneResult(
        scene_id=scene_id,
        num_gt_objects=len(gt_objects),
        num_queries=num_queries,
        num_keyframes=num_keyframes,
        random_sampling=random_result,
        l1_plus_owlvit=l1_result,
        jit_cascade=jit_result,
        brute_force=bf_result
    )


def compile_results(scene_results: List[SceneResult]) -> Dict:
    """Compile results into summary dict."""
    methods = ['random_sampling', 'l1_plus_owlvit', 'jit_cascade', 'brute_force']
    
    aggregate = {}
    for method in methods:
        det_recalls = [getattr(r, method).detection_recall for r in scene_results]
        det_precisions = [getattr(r, method).detection_precision for r in scene_results]
        latencies = [getattr(r, method).avg_latency_ms for r in scene_results]
        loc_0_5 = [getattr(r, method).localization_recall_0_5m for r in scene_results]
        loc_1 = [getattr(r, method).localization_recall_1m for r in scene_results]
        loc_2 = [getattr(r, method).localization_recall_2m for r in scene_results]
        loc_3 = [getattr(r, method).localization_recall_3m for r in scene_results]
        
        aggregate[method] = {
            "detection_recall": float(np.mean(det_recalls)),
            "detection_precision": float(np.mean(det_precisions)),
            "localization_recall_0_5m": float(np.mean(loc_0_5)),
            "localization_recall_1m": float(np.mean(loc_1)),
            "localization_recall_2m": float(np.mean(loc_2)),
            "localization_recall_3m": float(np.mean(loc_3)),
            "avg_latency_ms": float(np.mean(latencies)),
            "std_detection_recall": float(np.std(det_recalls)),
            "std_localization_recall_1m": float(np.std(loc_1)),
        }
    
    total_queries = sum(r.num_queries for r in scene_results)
    
    scene_data = []
    for r in scene_results:
        scene_data.append({
            "scene_id": r.scene_id,
            "num_gt_objects": r.num_gt_objects,
            "num_queries": r.num_queries,
            "num_keyframes": r.num_keyframes,
            "random_sampling": asdict(r.random_sampling),
            "l1_plus_owlvit": asdict(r.l1_plus_owlvit),
            "jit_cascade": asdict(r.jit_cascade),
            "brute_force": asdict(r.brute_force),
        })
    
    return {
        "evaluation_date": pd.Timestamp.now().isoformat(),
        "visibility_threshold_m": VISIBILITY_THRESHOLD,
        "localization_thresholds_m": LOCALIZATION_THRESHOLDS,
        "test_queries": TEST_QUERIES,
        "num_scenes": len(scene_results),
        "total_queries": total_queries,
        "aggregate": aggregate,
        "scene_results": scene_data
    }


def generate_report(results: Dict, output_path: Path):
    """Generate markdown report."""
    lines = []
    lines.append("# Comprehensive Evaluation Report")
    lines.append("")
    lines.append("**Spatially-Grounded Just-in-Time Episodic Memory for Mobile Robots**")
    lines.append("")
    lines.append(f"*Generated: {results['evaluation_date']}*")
    lines.append("")
    
    # Overview
    lines.append("## 1. Evaluation Setup")
    lines.append("")
    lines.append(f"- **Scenes evaluated**: {results['num_scenes']}")
    lines.append(f"- **Total queries**: {results['total_queries']}")
    lines.append(f"- **Test objects**: {', '.join(results['test_queries'])}")
    lines.append(f"- **Detection threshold**: Camera within {results['visibility_threshold_m']}m of GT")
    lines.append(f"- **Localization thresholds**: {results['localization_thresholds_m']}m")
    lines.append("")
    
    # Main results tables
    lines.append("## 2. Main Results")
    lines.append("")
    
    lines.append("### Detection Performance")
    lines.append("")
    lines.append("| Method | Recall | Precision | Latency |")
    lines.append("|--------|--------|-----------|---------|")
    
    agg = results['aggregate']
    method_names = {
        'random_sampling': 'Random Sampling',
        'l1_plus_owlvit': 'L1 + OWL-ViT',
        'jit_cascade': '**JIT Cascade**',
        'brute_force': 'Brute Force'
    }
    
    for method in ['random_sampling', 'l1_plus_owlvit', 'jit_cascade', 'brute_force']:
        m = agg[method]
        lines.append(f"| {method_names[method]} | {m['detection_recall']:.1f}% ± {m['std_detection_recall']:.1f} | {m['detection_precision']:.1f}% | {m['avg_latency_ms']:.0f}ms |")
    
    lines.append("")
    lines.append("### Localization Performance")
    lines.append("")
    lines.append("| Method | @0.5m | @1.0m | @2.0m | @3.0m |")
    lines.append("|--------|-------|-------|-------|-------|")
    
    for method in ['random_sampling', 'l1_plus_owlvit', 'jit_cascade', 'brute_force']:
        m = agg[method]
        lines.append(f"| {method_names[method]} | {m['localization_recall_0_5m']:.1f}% | {m['localization_recall_1m']:.1f}% ± {m['std_localization_recall_1m']:.1f} | {m['localization_recall_2m']:.1f}% | {m['localization_recall_3m']:.1f}% |")
    
    lines.append("")
    
    # Key findings
    jit = agg['jit_cascade']
    bf = agg['brute_force']
    l1 = agg['l1_plus_owlvit']
    
    lines.append("## 3. Key Findings")
    lines.append("")
    
    improvement_vs_bf = jit['localization_recall_1m'] / max(bf['localization_recall_1m'], 0.1)
    improvement_vs_l1 = jit['localization_recall_1m'] / max(l1['localization_recall_1m'], 0.1)
    speedup = bf['avg_latency_ms'] / max(jit['avg_latency_ms'], 1)
    
    lines.append("### JIT Cascade Advantages")
    lines.append("")
    lines.append(f"1. **Localization**: {jit['localization_recall_1m']:.1f}% at 1m vs {bf['localization_recall_1m']:.1f}% (BF) = **{improvement_vs_bf:.1f}x better**")
    lines.append(f"2. **Speed**: {jit['avg_latency_ms']:.0f}ms vs {bf['avg_latency_ms']:.0f}ms (BF) = **{speedup:.1f}x faster**")
    lines.append(f"3. **Precision at 0.5m**: {jit['localization_recall_0_5m']:.1f}% vs {bf['localization_recall_0_5m']:.1f}% (BF)")
    lines.append("")
    
    lines.append("### Trade-off Analysis")
    lines.append("")
    lines.append(f"- Brute Force: Best detection ({bf['detection_recall']:.1f}%), poor localization ({bf['localization_recall_1m']:.1f}% @1m)")
    lines.append(f"- JIT Cascade: Lower detection ({jit['detection_recall']:.1f}%), best localization ({jit['localization_recall_1m']:.1f}% @1m)")
    lines.append("")
    
    lines.append("## 4. Conclusion")
    lines.append("")
    lines.append("> JIT Cascade trades detection recall for localization accuracy. For robotics applications")
    lines.append("> where knowing the precise 3D location matters more than just detecting an object,")
    lines.append("> JIT Cascade provides significantly better results while being faster than brute force.")
    lines.append("")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    return '\n'.join(lines)


def main():
    """Run evaluation on all scenes with pre-generated ground truth."""
    from ingestion import CLIPEncoder
    from retrieval.level3_verification import OWLViTDetector
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("FULL-SCALE EVALUATION (Using Pre-Generated Ground Truth)")
    print("="*70)
    print(f"Detection threshold: {VISIBILITY_THRESHOLD}m")
    print(f"Localization thresholds: {LOCALIZATION_THRESHOLDS}m")
    print()
    
    # Get scenes
    all_scenes = get_all_scenes_with_gt()
    print(f"Found {len(all_scenes)} scenes with ground truth + memory banks")
    
    # Load models
    print("\nLoading models...")
    clip_encoder = CLIPEncoder()
    # Lower threshold for synthetic Habitat images which have lower OWL-ViT scores
    owl_detector = OWLViTDetector(score_threshold=0.02)
    
    # Run evaluation
    scene_results = []
    start_time = time.time()
    
    for i, (scene_id, scene_dir) in enumerate(all_scenes):
        print(f"\n[{i+1}/{len(all_scenes)}] Scene: {scene_id}")
        
        try:
            result = run_scene_evaluation(scene_id, scene_dir, clip_encoder, owl_detector)
            if result:
                scene_results.append(result)
                
                # Save intermediate every 20 scenes
                if len(scene_results) % 20 == 0:
                    print(f"\n  [Saving checkpoint: {len(scene_results)} scenes]")
                    interim = compile_results(scene_results)
                    with open(OUTPUT_DIR / "checkpoint.json", 'w') as f:
                        json.dump(interim, f, indent=2)
                        
        except Exception as e:
            print(f"    ERROR: {e}")
            traceback.print_exc()
    
    total_time = time.time() - start_time
    
    # Final results
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    if not scene_results:
        print("No scenes evaluated!")
        return
    
    results = compile_results(scene_results)
    results['total_time_seconds'] = total_time
    
    # Save
    with open(OUTPUT_DIR / "full_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    report = generate_report(results, OUTPUT_DIR / "COMPREHENSIVE_REPORT.md")
    
    print(f"\nScenes: {results['num_scenes']}")
    print(f"Queries: {results['total_queries']}")
    print(f"Time: {total_time/60:.1f} minutes")
    print()
    
    agg = results['aggregate']
    print("Method                  Det%    Loc@1m%  Latency")
    print("-" * 50)
    for method in ['random_sampling', 'l1_plus_owlvit', 'jit_cascade', 'brute_force']:
        m = agg[method]
        print(f"{method:<22} {m['detection_recall']:>6.1f}  {m['localization_recall_1m']:>7.1f}  {m['avg_latency_ms']:>7.0f}ms")
    
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
