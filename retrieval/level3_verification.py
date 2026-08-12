#!/usr/bin/env python3
"""
Phase 3 - Level 3: Visual Verification
========================================

Precise object detection using OWL-ViT zero-shot detector.

This is the final, most accurate stage that:
1. Takes top-k clusters from L2
2. Runs OWL-ViT detection on best frame of each cluster
3. Returns verified bounding boxes and confidence scores

Key insight: We only run the expensive detector on a small
number of promising candidates (typically 3-5), not all frames.

Latency budget: ~100ms per verification (actual: ~50-80ms)
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import sys
from PIL import Image
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.level2_geometric import L2Cluster


@dataclass
class Detection:
    """A detected object bounding box."""
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) normalized 0-1
    score: float
    label: str
    

@dataclass
class L3VerifiedLocation:
    """A verified object location with detection details."""
    cluster_id: int
    frame_id: int
    centroid_3d: np.ndarray  # From L2 cluster
    detections: List[Detection]
    best_detection_score: float
    verified: bool  # Whether detection was successful
    
    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "frame_id": self.frame_id,
            "centroid_3d": self.centroid_3d.tolist(),
            "detections": [
                {"bbox": d.bbox, "score": d.score, "label": d.label}
                for d in self.detections
            ],
            "best_detection_score": self.best_detection_score,
            "verified": self.verified,
        }


class OWLViTDetector:
    """
    Zero-shot object detector using OWL-ViT.
    
    OWL-ViT can detect objects given natural language queries,
    without being trained on those specific object categories.
    """
    
    def __init__(
        self,
        model_name: str = "google/owlvit-base-patch32",
        device: str = "cuda",
        score_threshold: float = 0.1,
    ):
        """
        Initialize OWL-ViT detector.
        
        Args:
            model_name: HuggingFace model name
            device: "cuda" or "cpu"
            score_threshold: Minimum detection confidence
        """
        self.model_name = model_name
        self.device = device
        self.score_threshold = score_threshold
        
        self.processor = None
        self.model = None
        self._loaded = False
        
    def _load_model(self):
        """Lazy load model on first use."""
        if self._loaded:
            return
            
        print(f"Loading OWL-ViT ({self.model_name})...")
        
        import torch
        from transformers import OwlViTProcessor, OwlViTForObjectDetection
        
        self.processor = OwlViTProcessor.from_pretrained(self.model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"  Loaded on {self.device}")
        self._loaded = True
        
    def detect(
        self,
        image: np.ndarray,
        queries: List[str],
    ) -> List[Detection]:
        """
        Detect objects in image matching queries.
        
        Args:
            image: RGB image (H, W, 3)
            queries: List of object descriptions to detect
            
        Returns:
            List of Detection objects
        """
        self._load_model()
        
        import torch
        
        # Convert to PIL
        pil_image = Image.fromarray(image)
        
        # Prepare inputs - OWL-ViT expects queries as a list of lists
        inputs = self.processor(
            text=[queries],  # Batch of 1 with multiple queries
            images=pil_image,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run detection
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Post-process (handle transformers API change)
        target_sizes = torch.tensor([pil_image.size[::-1]]).to(self.device)  # (height, width)
        post_fn = getattr(self.processor, 'post_process_object_detection',
                          getattr(self.processor, 'post_process_grounded_object_detection', None))
        if post_fn is None:
            raise RuntimeError("No post-processing method found on OwlViTProcessor")
        results = post_fn(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=self.score_threshold,
        )[0]
        
        detections = []
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()
        
        h, w = image.shape[:2]
        
        for box, score, label_idx in zip(boxes, scores, labels):
            # Convert to normalized coordinates
            x1, y1, x2, y2 = box
            bbox = (x1 / w, y1 / h, x2 / w, y2 / h)
            
            detection = Detection(
                bbox=bbox,
                score=float(score),
                label=queries[label_idx] if label_idx < len(queries) else "unknown",
            )
            detections.append(detection)
            
        # Sort by score
        detections.sort(key=lambda d: d.score, reverse=True)
        
        return detections


class Level3VisualVerification:
    """
    Level 3: Visual Verification using OWL-ViT.
    
    Pipeline:
    - Run zero-shot detection on best frame of each cluster
    - Verify that the target object is actually present
    - Return precise bounding boxes and confidence scores
    """
    
    def __init__(
        self,
        trace_loader,
        detector: Optional[OWLViTDetector] = None,
        verification_threshold: float = 0.15,
    ):
        """
        Initialize L3 Visual Verification.
        
        Args:
            trace_loader: TraceLoader with images
            detector: OWL-ViT detector (created if None)
            verification_threshold: Min detection score to consider verified
        """
        self.trace_loader = trace_loader
        self.detector = detector or OWLViTDetector()
        self.verification_threshold = verification_threshold
        
    def verify(
        self,
        clusters: List[L2Cluster],
        query: str,
        max_verify: int = 5,
    ) -> List[L3VerifiedLocation]:
        """
        Verify clusters using OWL-ViT detection.
        
        Reprojects to 3D using the detection bounding-box location
        for more accurate localization.
        
        Args:
            clusters: L2 clusters to verify
            query: Object description to detect
            max_verify: Maximum number of clusters to verify
            
        Returns:
            List of L3VerifiedLocation sorted by detection score
        """
        from retrieval.level2_geometric import DepthProjector
        
        if not clusters:
            return []
            
        # Only verify top clusters
        clusters_to_verify = clusters[:max_verify]
        projector = DepthProjector()
        
        verified_locations = []
        
        for cluster in clusters_to_verify:
            # Load best frame image and depth
            try:
                image = self.trace_loader.load_image(cluster.best_frame_id)
                depth = self.trace_loader.load_depth(cluster.best_frame_id)
            except Exception as e:
                print(f"Warning: Could not load data for frame {cluster.best_frame_id}: {e}")
                continue
                
            # Handle RGBA
            if image.shape[-1] == 4:
                image = image[:, :, :3]
                
            # Run detection
            queries = [query]
            detections = self.detector.detect(image, queries)
            
            # Filter detections
            verified_detections = [d for d in detections if d.score >= self.verification_threshold]
            
            # Best detection
            best_score = max([d.score for d in verified_detections], default=0.0)
            
            # Compute 3D location from best detection bbox (not cluster centroid)
            centroid_3d = cluster.centroid  # fallback
            
            if verified_detections and depth is not None:
                best_det = verified_detections[0]
                h, w = depth.shape
                
                # Get detection center pixel
                x1, y1, x2, y2 = best_det.bbox
                cx = int((x1 + x2) / 2 * w)
                cy = int((y1 + y2) / 2 * h)
                
                # Scale-adaptive depth sampling region (proportional to bbox)
                bbox_width = int((x2 - x1) * w)
                bbox_height = int((y2 - y1) * h)
                region_size = int(max(5, min(bbox_width, bbox_height) * 0.25))
                x_start = max(0, cx - region_size)
                x_end = min(w, cx + region_size)
                y_start = max(0, cy - region_size)
                y_end = min(h, cy + region_size)
                
                depth_region = depth[y_start:y_end, x_start:x_end]
                valid_depths = depth_region[(depth_region > 0.1) & (depth_region < 10.0)]
                
                if len(valid_depths) > 0:
                    # Use 25th percentile (closer to object)
                    obj_depth = np.percentile(valid_depths, 25)
                    
                    # Deproject to camera frame
                    point_camera = projector.deproject_pixel(cx, cy, obj_depth)
                    
                    # Get pose for this frame (label lookup by frame_id, matching
                    # load_image/load_depth above; frame_id is sparse, not a row index)
                    row = self.trace_loader.get_frame(cluster.best_frame_id)
                    position = np.array([row['x'], row['y'], row['z']])
                    rotation = np.array([row['qw'], row['qx'], row['qy'], row['qz']])
                    
                    # Transform to world
                    centroid_3d = projector.transform_to_world(point_camera, position, rotation)
            
            location = L3VerifiedLocation(
                cluster_id=cluster.cluster_id,
                frame_id=cluster.best_frame_id,
                centroid_3d=centroid_3d,
                detections=verified_detections,
                best_detection_score=best_score,
                verified=len(verified_detections) > 0,
            )
            verified_locations.append(location)
            
        # Sort by detection score
        verified_locations.sort(key=lambda loc: loc.best_detection_score, reverse=True)
        
        return verified_locations
    
    def verify_with_synonyms(
        self,
        clusters: List[L2Cluster],
        query: str,
        synonyms: Optional[List[str]] = None,
        max_verify: int = 5,
    ) -> List[L3VerifiedLocation]:
        """
        Verify with multiple query variants for better recall.
        
        For example, "couch" might also match "sofa" or "settee".
        """
        if synonyms is None:
            synonyms = []
            
        all_queries = [query] + synonyms
        
        verified_locations = []
        clusters_to_verify = clusters[:max_verify]
        
        for cluster in clusters_to_verify:
            try:
                image = self.trace_loader.load_image(cluster.best_frame_id)
            except Exception:
                continue
                
            if image.shape[-1] == 4:
                image = image[:, :, :3]
                
            # Detect all query variants
            detections = self.detector.detect(image, all_queries)
            verified_detections = [d for d in detections if d.score >= self.verification_threshold]
            best_score = max([d.score for d in verified_detections], default=0.0)
            
            location = L3VerifiedLocation(
                cluster_id=cluster.cluster_id,
                frame_id=cluster.best_frame_id,
                centroid_3d=cluster.centroid,
                detections=verified_detections,
                best_detection_score=best_score,
                verified=len(verified_detections) > 0,
            )
            verified_locations.append(location)
            
        verified_locations.sort(key=lambda loc: loc.best_detection_score, reverse=True)
        return verified_locations


def test_level3():
    """Test Level 3 verification on L2 clusters."""
    import time
    from ingestion import TraceLoader
    from retrieval.level1_semantic import Level1SemanticFilter
    from retrieval.level2_geometric import Level2GeometricCluster
    
    # Load trace
    trace_dir = Path(__file__).parent.parent / "outputs" / "phase2" / "exploration_panoramic"
    if not trace_dir.exists():
        trace_dir = Path(__file__).parent.parent / "outputs" / "phase2" / "exploration"
        
    print(f"Loading trace from: {trace_dir}")
    loader = TraceLoader(str(trace_dir))
    
    # Create L1, L2, L3
    l1_filter = Level1SemanticFilter(loader, k_candidates=50)
    l2_cluster = Level2GeometricCluster(loader, eps=1.0, min_samples=2)
    l3_verify = Level3VisualVerification(loader, verification_threshold=0.1)
    
    print("\n=== Level 3 Visual Verification Test ===\n")
    
    test_queries = ["couch", "window", "door"]
    
    for query in test_queries:
        print(f"Query: '{query}'")
        
        # L1
        start = time.time()
        l1_candidates = l1_filter.retrieve(query, k=50)
        l1_time = (time.time() - start) * 1000
        print(f"  L1: {len(l1_candidates)} candidates ({l1_time:.1f}ms)")
        
        # L2
        start = time.time()
        l2_clusters = l2_cluster.cluster(l1_candidates)
        l2_time = (time.time() - start) * 1000
        print(f"  L2: {len(l2_clusters)} clusters ({l2_time:.1f}ms)")
        
        # L3
        start = time.time()
        l3_locations = l3_verify.verify(l2_clusters, query, max_verify=3)
        l3_time = (time.time() - start) * 1000
        print(f"  L3: {sum(1 for loc in l3_locations if loc.verified)} verified ({l3_time:.1f}ms)")
        
        for loc in l3_locations[:3]:
            status = "[OK]" if loc.verified else "[FAIL]"
            print(f"    {status} Frame {loc.frame_id}: "
                  f"score={loc.best_detection_score:.3f}, "
                  f"{len(loc.detections)} detections")
        print()
    
    # Benchmark L3
    print("=== L3 Latency Benchmark ===")
    query = "couch"
    l1_candidates = l1_filter.retrieve(query, k=50)
    l2_clusters = l2_cluster.cluster(l1_candidates)
    
    times = []
    for _ in range(10):
        start = time.time()
        l3_verify.verify(l2_clusters[:3], query, max_verify=3)
        times.append((time.time() - start) * 1000)
        
    print(f"Verifying 3 clusters:")
    print(f"  Mean latency: {np.mean(times):.1f}ms")
    print(f"  Per-cluster: {np.mean(times)/3:.1f}ms")
    
    print("\n[OK] Level 3 Visual Verification test passed!")


if __name__ == "__main__":
    test_level3()
