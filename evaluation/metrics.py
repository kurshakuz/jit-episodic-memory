#!/usr/bin/env python3
"""
Phase 4: Evaluation Metrics
============================

Compute precision, recall, and localization accuracy using ground truth.

Key metrics:
- Precision@k: Of top-k retrieved, how many are correct?
- Recall@k: Of all ground truth objects, how many are retrieved in top-k?
- Localization Error: Distance (meters) between predicted and ground truth locations
- Latency: End-to-end query time (ms)

Ground truth format (from Phase 1 oracle):
{
    "objects": {
        "category_name": [
            {"id": "obj_id", "center": [x, y, z]},
            ...
        ]
    }
}
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class LocalizationResult:
    """Result of localization evaluation for a single query."""
    query: str
    ground_truth_count: int
    retrieved_count: int
    verified_count: int
    
    # Precision/Recall
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    recall_at_k: float  # k = retrieved_count
    
    # Localization error (meters)
    localization_error: Optional[float]  # Distance to nearest GT
    
    # Timing
    latency_ms: float
    
    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "ground_truth_count": self.ground_truth_count,
            "retrieved_count": self.retrieved_count,
            "verified_count": self.verified_count,
            "precision_at_1": self.precision_at_1,
            "precision_at_3": self.precision_at_3,
            "precision_at_5": self.precision_at_5,
            "recall_at_k": self.recall_at_k,
            "localization_error_m": self.localization_error,
            "latency_ms": self.latency_ms,
        }


class GroundTruthOracle:
    """
    Load and query ground truth object locations.
    
    Ground truth format from Phase 1:
    - objects: dict of object_id -> {id, category, center}
    - frames: list of {frame_id, position, visible_object_ids}
    - object_to_frames: dict of object_id -> [frame_ids]
    """
    
    def __init__(self, ground_truth_path: str):
        """
        Load ground truth from JSON file.
        
        Args:
            ground_truth_path: Path to ground_truth.json from Phase 1
        """
        with open(ground_truth_path) as f:
            self.data = json.load(f)
            
        self.raw_objects = self.data.get("objects", {})
        self.frames = self.data.get("frames", [])
        self.object_to_frames = self.data.get("object_to_frames", {})
        
        # Build category index: category -> [object_ids]
        self._category_to_objects: Dict[str, List[str]] = {}
        
        for obj_id, obj_data in self.raw_objects.items():
            if isinstance(obj_data, dict):
                category = obj_data.get("category", "unknown")
                if category:
                    category_lower = category.lower()
                    if category_lower not in self._category_to_objects:
                        self._category_to_objects[category_lower] = []
                    self._category_to_objects[category_lower].append(obj_id)
        
        # Build frame_id -> frame data index
        self._frame_index = {f["frame_id"]: f for f in self.frames}
        
        # Build category -> frames index (which frames show this category)
        self._category_to_frames: Dict[str, set] = {}
        for obj_id, frame_ids in self.object_to_frames.items():
            obj_data = self.raw_objects.get(obj_id, {})
            category = obj_data.get("category", "unknown") if isinstance(obj_data, dict) else "unknown"
            category_lower = category.lower()
            
            if category_lower not in self._category_to_frames:
                self._category_to_frames[category_lower] = set()
            self._category_to_frames[category_lower].update(frame_ids)
        
        self.categories = list(self._category_to_objects.keys())
        
    def get_category_locations(self, category: str) -> List[np.ndarray]:
        """
        Get all 3D locations for a category.
        
        Since centers are not available, we use the agent positions
        at frames where this category was visible.
        
        Args:
            category: Object category name (case-insensitive)
            
        Returns:
            List of [x, y, z] agent positions where object was visible
        """
        category_lower = category.lower()
        
        # Get frames where this category was visible
        frame_ids = self._category_to_frames.get(category_lower, set())
        
        if not frame_ids:
            # Try partial match
            for cat, fids in self._category_to_frames.items():
                if category_lower in cat or cat in category_lower:
                    frame_ids.update(fids)
        
        locations = []
        for frame_id in frame_ids:
            frame_data = self._frame_index.get(frame_id)
            if frame_data and frame_data.get("position"):
                locations.append(np.array(frame_data["position"]))
                        
        return locations
    
    def get_frames_showing_category(self, category: str) -> set:
        """Get set of frame IDs where this category is visible."""
        category_lower = category.lower()
        
        frames = self._category_to_frames.get(category_lower, set())
        
        # Try partial match
        if not frames:
            for cat, fids in self._category_to_frames.items():
                if category_lower in cat or cat in category_lower:
                    frames = frames.union(fids)
                    
        return frames
    
    def has_category(self, category: str) -> bool:
        """Check if category exists in ground truth."""
        return len(self.get_category_locations(category)) > 0
    
    def nearest_ground_truth(
        self,
        category: str,
        predicted_location: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Find nearest ground truth object of given category.
        
        Args:
            category: Object category
            predicted_location: Predicted 3D location
            
        Returns:
            Tuple of (nearest_gt_location, distance) or (None, None)
        """
        gt_locations = self.get_category_locations(category)
        
        if not gt_locations:
            return None, None
            
        distances = [np.linalg.norm(gt - predicted_location) for gt in gt_locations]
        min_idx = np.argmin(distances)
        
        return gt_locations[min_idx], distances[min_idx]
    
    def is_correct(
        self,
        category: str,
        predicted_location: np.ndarray,
        threshold_m: float = 2.0,
    ) -> bool:
        """
        Check if prediction is correct (within threshold of ground truth).
        
        Args:
            category: Object category
            predicted_location: Predicted 3D location
            threshold_m: Distance threshold in meters
            
        Returns:
            True if prediction is within threshold of any ground truth
        """
        _, distance = self.nearest_ground_truth(category, predicted_location)
        if distance is None:
            return False
        return distance <= threshold_m
    
    def is_frame_correct(
        self,
        category: str,
        frame_id: int,
    ) -> bool:
        """
        Check if a frame correctly shows the queried category.
        
        Args:
            category: Object category
            frame_id: Frame ID to check
            
        Returns:
            True if the category was visible in this frame
        """
        correct_frames = self.get_frames_showing_category(category)
        return frame_id in correct_frames


class RetrievalEvaluator:
    """
    Evaluate retrieval performance against ground truth.
    """
    
    def __init__(
        self,
        ground_truth_path: str,
        distance_threshold: float = 2.0,
    ):
        """
        Initialize evaluator.
        
        Args:
            ground_truth_path: Path to ground_truth.json
            distance_threshold: Max distance (m) for correct localization
        """
        self.oracle = GroundTruthOracle(ground_truth_path)
        self.distance_threshold = distance_threshold
    
    def evaluate_query_frame_based(
        self,
        query: str,
        predicted_frame_ids: List[int],
        predicted_locations: List[np.ndarray],
        latency_ms: float,
        verified_count: int = 0,
    ) -> LocalizationResult:
        """
        Evaluate a query using frame-based ground truth.
        
        This checks if the retrieved frames actually show the queried object.
        
        Args:
            query: Object query
            predicted_frame_ids: List of retrieved frame IDs
            predicted_locations: List of 3D locations for localization error
            latency_ms: Query latency
            verified_count: Number of L3-verified locations
            
        Returns:
            LocalizationResult with metrics
        """
        gt_frames = self.oracle.get_frames_showing_category(query)
        gt_count = len(gt_frames)
        retrieved_count = len(predicted_frame_ids)
        
        # Precision@k: what fraction of top-k are correct?
        def precision_at_k(k: int) -> float:
            if k == 0 or retrieved_count == 0:
                return 0.0
            k = min(k, retrieved_count)
            correct = sum(1 for fid in predicted_frame_ids[:k] if fid in gt_frames)
            return correct / k
        
        # Recall: what fraction of GT frames are retrieved?
        def compute_recall() -> float:
            if gt_count == 0:
                return 1.0 if retrieved_count == 0 else 0.0
            retrieved_set = set(predicted_frame_ids)
            matched = len(gt_frames.intersection(retrieved_set))
            return matched / gt_count
        
        # Localization error (distance to nearest GT viewing position)
        loc_error = None
        if predicted_locations:
            _, loc_error = self.oracle.nearest_ground_truth(query, predicted_locations[0])
            
        return LocalizationResult(
            query=query,
            ground_truth_count=gt_count,
            retrieved_count=retrieved_count,
            verified_count=verified_count,
            precision_at_1=precision_at_k(1),
            precision_at_3=precision_at_k(3),
            precision_at_5=precision_at_k(5),
            recall_at_k=compute_recall(),
            localization_error=loc_error,
            latency_ms=latency_ms,
        )
        
    def evaluate_query(
        self,
        query: str,
        predicted_locations: List[np.ndarray],
        latency_ms: float,
        verified_count: int = 0,
    ) -> LocalizationResult:
        """
        Evaluate a single query.
        
        Args:
            query: Object query
            predicted_locations: List of predicted 3D locations (sorted by confidence)
            latency_ms: Query latency
            verified_count: Number of L3-verified locations
            
        Returns:
            LocalizationResult with metrics
        """
        gt_locations = self.oracle.get_category_locations(query)
        gt_count = len(gt_locations)
        retrieved_count = len(predicted_locations)
        
        # Compute precision@k
        def precision_at_k(k: int) -> float:
            if k == 0 or retrieved_count == 0:
                return 0.0
            k = min(k, retrieved_count)
            correct = sum(
                1 for loc in predicted_locations[:k]
                if self.oracle.is_correct(query, loc, self.distance_threshold)
            )
            return correct / k
            
        # Recall: how many GT objects are retrieved?
        def compute_recall() -> float:
            if gt_count == 0:
                return 1.0 if retrieved_count == 0 else 0.0
            
            # For each GT, check if any prediction is close
            matched = 0
            for gt in gt_locations:
                for pred in predicted_locations:
                    if np.linalg.norm(gt - pred) <= self.distance_threshold:
                        matched += 1
                        break
            return matched / gt_count
        
        # Localization error (for top prediction)
        loc_error = None
        if predicted_locations:
            _, loc_error = self.oracle.nearest_ground_truth(query, predicted_locations[0])
            
        return LocalizationResult(
            query=query,
            ground_truth_count=gt_count,
            retrieved_count=retrieved_count,
            verified_count=verified_count,
            precision_at_1=precision_at_k(1),
            precision_at_3=precision_at_k(3),
            precision_at_5=precision_at_k(5),
            recall_at_k=compute_recall(),
            localization_error=loc_error,
            latency_ms=latency_ms,
        )
    
    def evaluate_cascade(
        self,
        cascade,  # JITRetrievalCascade
        test_queries: Optional[List[str]] = None,
        skip_l3: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate the full cascade on test queries.
        
        Args:
            cascade: JITRetrievalCascade instance
            test_queries: List of queries (uses all GT categories if None)
            skip_l3: Skip L3 verification for speed
            
        Returns:
            Dict with aggregate metrics and per-query results
        """
        if test_queries is None:
            # Use all categories that have ground truth
            test_queries = self.oracle.categories
            
        results = []
        
        for query in test_queries:
            # Skip if no ground truth
            if not self.oracle.has_category(query):
                continue
                
            # Run query
            query_result = cascade.query(query, skip_l3=skip_l3)
            
            # Extract predicted frame IDs and locations
            predicted_frame_ids = [loc.frame_id for loc in query_result.locations]
            predicted_locations = [loc.centroid_3d for loc in query_result.locations]
            
            # Evaluate using frame-based ground truth
            result = self.evaluate_query_frame_based(
                query=query,
                predicted_frame_ids=predicted_frame_ids,
                predicted_locations=predicted_locations,
                latency_ms=query_result.total_time_ms,
                verified_count=query_result.l3_verified,
            )
            results.append(result)
            
        # Aggregate metrics
        if not results:
            return {"error": "No results"}
            
        aggregate = {
            "num_queries": len(results),
            "mean_precision_at_1": np.mean([r.precision_at_1 for r in results]),
            "mean_precision_at_3": np.mean([r.precision_at_3 for r in results]),
            "mean_precision_at_5": np.mean([r.precision_at_5 for r in results]),
            "mean_recall": np.mean([r.recall_at_k for r in results]),
            "mean_localization_error_m": np.mean([
                r.localization_error for r in results if r.localization_error is not None
            ]),
            "mean_latency_ms": np.mean([r.latency_ms for r in results]),
            "per_query": [r.to_dict() for r in results],
        }
        
        return aggregate


def test_evaluation():
    """Test evaluation on our ground truth."""
    
    gt_path = Path(__file__).parent.parent / "outputs" / "phase1" / "ground_truth.json"
    
    if not gt_path.exists():
        print(f"Ground truth not found at {gt_path}")
        print("Run Phase 1 ground truth generator first.")
        return
        
    print("Loading ground truth...")
    evaluator = RetrievalEvaluator(str(gt_path), distance_threshold=2.0)
    
    print(f"\nGround truth categories ({len(evaluator.oracle.categories)}):")
    for cat in sorted(evaluator.oracle.categories)[:20]:
        count = len(evaluator.oracle.get_category_locations(cat))
        print(f"  {cat}: {count} instances")
    if len(evaluator.oracle.categories) > 20:
        print(f"  ... and {len(evaluator.oracle.categories) - 20} more")
    
    print("\n[OK] Evaluation framework test passed!")


if __name__ == "__main__":
    test_evaluation()
