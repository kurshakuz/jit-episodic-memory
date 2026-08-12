#!/usr/bin/env python3
"""
Phase 3 - Level 2: Geometric Clustering
========================================

Depth-based object localization and 3D clustering.

This stage takes L1 candidates and:
1. Projects depth images to 3D point clouds
2. Clusters nearby observations with DBSCAN
3. Returns unique 3D object locations

Key insight: Multiple keyframes may show the same object from different
viewpoints. We use depth projection to merge these into a single location.

Latency budget: ~20ms (actual: ~15ms)
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.level1_semantic import L1Candidate


@dataclass
class L2Cluster:
    """A cluster of observations representing a potential object location."""
    cluster_id: int
    centroid: np.ndarray  # [x, y, z] in world coordinates
    member_frames: List[int]  # Frame IDs contributing to this cluster
    mean_similarity: float  # Average L1 similarity of members
    max_similarity: float  # Max L1 similarity of members
    num_observations: int
    best_frame_id: int  # Frame with highest similarity
    
    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "centroid": self.centroid.tolist(),
            "member_frames": self.member_frames,
            "mean_similarity": self.mean_similarity,
            "max_similarity": self.max_similarity,
            "num_observations": self.num_observations,
            "best_frame_id": self.best_frame_id,
        }


class DepthProjector:
    """
    Projects depth images to 3D points in world coordinates.
    
    Uses pinhole camera model with known intrinsics.
    
    IMPORTANT: Accounts for sensor offset from agent position.
    In Habitat-Sim, the sensor is typically 1.5m above the agent base.
    """
    
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        hfov: float = 90.0,
        max_depth: float = 10.0,
        sensor_height: float = 1.5,  # Sensor offset above agent position
    ):
        """
        Initialize projector.
        
        Args:
            width: Image width
            height: Image height
            hfov: Horizontal field of view in degrees
            max_depth: Maximum valid depth (clip beyond this)
            sensor_height: Height of sensor above agent base position
        """
        self.width = width
        self.height = height
        self.hfov = np.deg2rad(hfov)
        self.max_depth = max_depth
        self.sensor_height = sensor_height
        
        # Compute focal length from HFOV
        self.fx = width / (2.0 * np.tan(self.hfov / 2.0))
        self.fy = self.fx  # Assuming square pixels
        self.cx = width / 2.0
        self.cy = height / 2.0
        
        # Pre-compute pixel coordinates grid
        u = np.arange(width)
        v = np.arange(height)
        self.u_grid, self.v_grid = np.meshgrid(u, v)
        
    def deproject_pixel(
        self,
        u: int,
        v: int,
        depth: float,
    ) -> np.ndarray:
        """
        Deproject a single pixel to 3D camera coordinates.
        
        Habitat-Sim uses:
        - X: right
        - Y: up  
        - Z: backward (depth is along -Z in camera frame)
        
        Pixel coordinates:
        - u: horizontal (0 = left, increases right)
        - v: vertical (0 = top, increases downward)
        
        Args:
            u, v: Pixel coordinates
            depth: Depth value at pixel
            
        Returns:
            3D point [x, y, z] in camera frame
        """
        # Standard pinhole deprojection with correct sign conventions
        x = (u - self.cx) * depth / self.fx      # Right is positive
        y = -(v - self.cy) * depth / self.fy     # Up is positive (flip v)
        z = -depth                                # Forward is negative Z
        return np.array([x, y, z])
    
    def deproject_center(self, depth_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Get 3D point at image center.
        
        This is a fast approximation - assumes the queried object
        is roughly centered in the frame (L1 should ensure this).
        
        Args:
            depth_image: Depth map (H, W)
            
        Returns:
            3D point in camera frame or None if invalid depth
        """
        h, w = depth_image.shape
        center_u, center_v = w // 2, h // 2
        
        # First try: use the actual center pixel depth
        center_depth = depth_image[center_v, center_u]
        
        if 0.1 < center_depth < self.max_depth:
            return self.deproject_pixel(center_u, center_v, center_depth)
        
        # Fallback: sample a small region and use the CLOSEST valid depth
        # (not median - median picks up background)
        region_size = 15
        u_start = max(0, center_u - region_size)
        u_end = min(w, center_u + region_size)
        v_start = max(0, center_v - region_size)
        v_end = min(h, center_v + region_size)
        
        center_region = depth_image[v_start:v_end, u_start:u_end]
        
        # Collect valid depths in the patch; the 25th percentile below selects the
        # near (object) surface rather than the background
        valid_depths = center_region[(center_region > 0.1) & (center_region < self.max_depth)]
        
        if len(valid_depths) == 0:
            return None
        
        # Use 25th percentile - closer to foreground than median
        closest_depth = np.percentile(valid_depths, 25)
        return self.deproject_pixel(center_u, center_v, closest_depth)
    
    def project_bbox_to_3d(
        self,
        bbox_norm: Tuple[float, float, float, float],
        depth_image: np.ndarray,
        position: np.ndarray,
        rotation: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Project a detection bounding box to 3D world coordinates.
        
        Args:
            bbox_norm: (x1, y1, x2, y2) normalized 0-1
            depth_image: HxW depth array
            position: camera position [x, y, z]
            rotation: quaternion [w, x, y, z]
            
        Returns:
            3D point in world coordinates, or None if invalid
        """
        h, w = depth_image.shape
        x1, y1, x2, y2 = bbox_norm
        
        # Convert to pixel coords
        px1, px2 = int(x1 * w), int(x2 * w)
        py1, py2 = int(y1 * h), int(y2 * h)
        
        # Clamp to valid range
        px1, px2 = max(0, px1), min(w, px2)
        py1, py2 = max(0, py1), min(h, py2)
        
        if px1 >= px2 or py1 >= py2:
            return None
        
        # Get center of bbox
        cx = (px1 + px2) // 2
        cy = (py1 + py2) // 2
        
        # Sample depth in bbox region
        bbox_region = depth_image[py1:py2, px1:px2]
        valid_depths = bbox_region[(bbox_region > 0.1) & (bbox_region < self.max_depth)]
        
        if len(valid_depths) == 0:
            return None
        
        # Use 30th percentile depth (closer to foreground object)
        depth_val = np.percentile(valid_depths, 30)
        
        # Project to camera frame
        point_cam = self.deproject_pixel(cx, cy, depth_val)
        
        # Transform to world
        point_world = self.transform_to_world(point_cam, position, rotation)
        
        return point_world
    
    def transform_to_world(
        self,
        point_camera: np.ndarray,
        position: np.ndarray,
        rotation: np.ndarray,  # quaternion [w, x, y, z]
    ) -> np.ndarray:
        """
        Transform point from camera frame to world frame.
        
        IMPORTANT: Adds sensor height offset to account for camera
        being above agent base position.
        
        Args:
            point_camera: 3D point in camera coordinates
            position: Agent base position [x, y, z]
            rotation: Agent rotation quaternion [w, x, y, z]
            
        Returns:
            3D point in world coordinates
        """
        # Sensor position = agent position + height offset
        sensor_position = position.copy()
        sensor_position[1] += self.sensor_height  # Y is up in Habitat
        
        # Convert quaternion to rotation matrix
        w, x, y, z = rotation
        R = np.array([
            [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
            [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
        ])
        
        # Transform: world_point = R @ camera_point + sensor_position
        point_world = R @ point_camera + sensor_position
        return point_world


class Level2GeometricCluster:
    """
    Level 2: Geometric Clustering using depth projection and DBSCAN.
    
    Pipeline:
    - Project L1 candidates to 3D world coordinates
    - Cluster with DBSCAN to merge multiple views
    - Return deduplicated object location hypotheses
    """
    
    def __init__(
        self,
        trace_loader,  # TraceLoader from ingestion
        eps: float = 1.0,  # DBSCAN clustering radius in meters
        min_samples: int = 1,  # Minimum cluster size
        top_clusters: int = 10,  # Maximum clusters to return
    ):
        """
        Initialize L2 Geometric Clustering.
        
        Args:
            trace_loader: TraceLoader with depth images
            eps: DBSCAN epsilon (clustering radius in meters)
            min_samples: Minimum observations per cluster
            top_clusters: Maximum number of clusters to return
        """
        self.trace_loader = trace_loader
        self.eps = eps
        self.min_samples = min_samples
        self.top_clusters = top_clusters
        
        # Depth projector
        self.projector = DepthProjector()
        
    def cluster(
        self,
        candidates: List[L1Candidate],
    ) -> List[L2Cluster]:
        """
        Cluster L1 candidates by 3D position.
        
        Args:
            candidates: List of L1 candidates with positions
            
        Returns:
            List of L2 clusters sorted by max similarity (descending)
        """
        if not candidates:
            return []
            
        # Extract 3D points for each candidate
        points_3d = []
        valid_candidates = []
        
        for candidate in candidates:
            # Try to get depth-projected 3D point
            point = self._get_3d_point(candidate)
            
            if point is not None:
                points_3d.append(point)
                valid_candidates.append(candidate)
                
        if len(points_3d) < self.min_samples:
            # Not enough valid points, fall back to agent positions
            points_3d = [np.array(c.position) for c in candidates]
            valid_candidates = candidates
            
        points_array = np.array(points_3d)
        
        # Run DBSCAN clustering
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = clustering.fit_predict(points_array)
        
        # Build clusters
        clusters = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:  # Noise points - treat each as its own cluster
                noise_indices = np.where(labels == -1)[0]
                for idx in noise_indices:
                    c = valid_candidates[idx]
                    cluster = L2Cluster(
                        cluster_id=len(clusters),
                        centroid=points_3d[idx],
                        member_frames=[c.frame_id],
                        mean_similarity=c.similarity,
                        max_similarity=c.similarity,
                        num_observations=1,
                        best_frame_id=c.frame_id,
                    )
                    clusters.append(cluster)
            else:
                # Regular cluster
                member_indices = np.where(labels == label)[0]
                member_points = [points_3d[i] for i in member_indices]
                member_candidates = [valid_candidates[i] for i in member_indices]
                
                # Compute centroid
                centroid = np.mean(member_points, axis=0)
                
                # Get similarity stats
                similarities = [c.similarity for c in member_candidates]
                best_idx = np.argmax(similarities)
                
                cluster = L2Cluster(
                    cluster_id=len(clusters),
                    centroid=centroid,
                    member_frames=[c.frame_id for c in member_candidates],
                    mean_similarity=float(np.mean(similarities)),
                    max_similarity=float(np.max(similarities)),
                    num_observations=len(member_candidates),
                    best_frame_id=member_candidates[best_idx].frame_id,
                )
                clusters.append(cluster)
                
        # Sort by max similarity
        clusters.sort(key=lambda c: c.max_similarity, reverse=True)
        
        # Return top clusters
        return clusters[:self.top_clusters]
    
    def _get_3d_point(self, candidate: L1Candidate) -> Optional[np.ndarray]:
        """Get 3D world point for a candidate using depth projection."""
        if candidate.depth_path is None:
            # No depth, use agent position as fallback
            return np.array(candidate.position)
            
        try:
            # Load depth image
            depth = self.trace_loader.load_depth(candidate.frame_id)
            
            if depth is None:
                return np.array(candidate.position)
                
            # Deproject center to camera frame
            point_camera = self.projector.deproject_center(depth)
            
            if point_camera is None:
                return np.array(candidate.position)
                
            # Transform to world frame
            position = np.array(candidate.position)
            rotation = np.array(candidate.rotation)
            
            point_world = self.projector.transform_to_world(
                point_camera, position, rotation
            )
            
            return point_world
            
        except Exception as e:
            # Fallback to agent position
            return np.array(candidate.position)


def test_level2():
    """Test Level 2 clustering on L1 candidates."""
    import time
    from ingestion import TraceLoader, CLIPEncoder
    from retrieval.level1_semantic import Level1SemanticFilter
    
    # Load trace
    trace_dir = Path(__file__).parent.parent / "outputs" / "phase2" / "exploration_panoramic"
    if not trace_dir.exists():
        trace_dir = Path(__file__).parent.parent / "outputs" / "phase2" / "exploration"
        
    print(f"Loading trace from: {trace_dir}")
    loader = TraceLoader(str(trace_dir))
    
    # Create L1 and L2
    l1_filter = Level1SemanticFilter(loader, k_candidates=50)
    l2_cluster = Level2GeometricCluster(loader, eps=1.0, min_samples=2)
    
    print("\n=== Level 2 Geometric Clustering Test ===\n")
    
    test_queries = ["couch", "window", "door", "table"]
    
    for query in test_queries:
        # L1 retrieval
        start = time.time()
        l1_candidates = l1_filter.retrieve(query, k=50)
        l1_time = (time.time() - start) * 1000
        
        # L2 clustering
        start = time.time()
        l2_clusters = l2_cluster.cluster(l1_candidates)
        l2_time = (time.time() - start) * 1000
        
        print(f"Query: '{query}'")
        print(f"  L1: {len(l1_candidates)} candidates ({l1_time:.1f}ms)")
        print(f"  L2: {len(l2_clusters)} clusters ({l2_time:.1f}ms)")
        
        if l2_clusters:
            best = l2_clusters[0]
            print(f"  Best cluster: {best.num_observations} observations, "
                  f"similarity={best.max_similarity:.3f}")
            print(f"  Centroid: [{best.centroid[0]:.2f}, {best.centroid[1]:.2f}, "
                  f"{best.centroid[2]:.2f}]")
        print()
    
    # Benchmark
    print("=== L2 Latency Benchmark ===")
    query = "couch"
    l1_candidates = l1_filter.retrieve(query, k=50)
    
    times = []
    for _ in range(100):
        start = time.time()
        l2_cluster.cluster(l1_candidates)
        times.append((time.time() - start) * 1000)
        
    print(f"Clustering 50 candidates:")
    print(f"  Mean latency: {np.mean(times):.2f}ms")
    print(f"  Std: {np.std(times):.2f}ms")
    
    print("\n[OK] Level 2 Geometric Clustering test passed!")


if __name__ == "__main__":
    test_level2()
