#!/usr/bin/env python3
"""
Dense Semantic Map Baseline
===========================

This implements a "Dense Mapping" baseline that represents the VLMaps/ConceptFusion
approach: pre-compute everything during exploration, fast queries at runtime.

This serves as a fair architectural comparison with the JIT Cascade:
- Same backbone (CLIP ViT-B-32, OWL-ViT)
- Same geometric processing (DBSCAN clustering)
- Different timing: Dense Map does work upfront, JIT does work at query time

Key insight: This isolates the ARCHITECTURAL choice (Lazy vs Eager) from the
backbone choice, providing a scientifically valid comparison.

Usage:
    # Build map for a scene
    dense_map = DenseMapBaseline(scene_dir)
    build_stats = dense_map.build_map()  # Heavy upfront work
    
    # Query (fast)
    result = dense_map.query("couch")  # Just dot product + clustering
"""

import sys
import time
import json
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple
from PIL import Image
from sklearn.cluster import DBSCAN
import gc

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class MapBuildStats:
    """Statistics from building the dense map."""
    scene_id: str
    num_frames_processed: int
    num_points_total: int
    num_points_after_voxel: int
    build_time_seconds: float
    memory_size_mb: float
    frames_per_second: float
    

@dataclass 
class MapQueryResult:
    """Result of querying the dense map."""
    query: str
    success: bool
    predicted_location: Optional[np.ndarray]
    confidence: float
    query_time_ms: float
    num_points_matched: int
    cluster_size: int


class DenseMapBaseline:
    """
    Dense Semantic Map baseline - VLMaps/ConceptFusion style.
    
    Architecture:
    1. BUILD PHASE (offline, per scene):
       - Process ALL frames
       - Project depth to 3D for each frame
       - Assign CLIP embedding to all valid 3D points
       - Optionally voxelize to reduce size
       
    2. QUERY PHASE (online, per query):
       - Encode text query with CLIP
       - Compute similarity with all map points
       - Cluster top-k matches with DBSCAN
       - Return centroid of best cluster
    
    This is the "eager" approach: do all work upfront.
    """
    
    def __init__(
        self,
        scene_dir: Path,
        voxel_size: float = 0.05,  # 5cm voxels
        sample_stride: int = 4,    # Sample every Nth pixel for speed
        max_depth: float = 10.0,
        clip_model: str = "ViT-B-32-quickgelu",
        clip_pretrained: str = "laion400m_e32",
        device: Optional[str] = None,
    ):
        """
        Initialize Dense Map baseline.
        
        Args:
            scene_dir: Path to scene memory bank
            voxel_size: Voxel grid size for downsampling (meters)
            sample_stride: Pixel stride for depth projection (1=all, 4=every 4th)
            max_depth: Maximum valid depth
            clip_model: CLIP model name
            clip_pretrained: CLIP pretrained weights
            device: Device for inference
        """
        self.scene_dir = Path(scene_dir)
        self.voxel_size = voxel_size
        self.sample_stride = sample_stride
        self.max_depth = max_depth
        
        # Model config (same as JIT for fair comparison)
        self.clip_model = clip_model
        self.clip_pretrained = clip_pretrained
        self.device = device or ("cuda" if self._cuda_available() else "cpu")
        
        # Map data (populated during build)
        self.points_3d: Optional[np.ndarray] = None  # (N, 3) world coords
        self.point_features: Optional[np.ndarray] = None  # (N, 512) CLIP embeddings
        self.point_frame_ids: Optional[np.ndarray] = None  # (N,) source frame
        
        # Stats
        self.build_stats: Optional[MapBuildStats] = None
        self._map_built = False
        
        # Lazy-loaded models
        self._clip_encoder = None
        
    def _cuda_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
            
    def _get_clip_encoder(self):
        """Lazy load CLIP encoder."""
        if self._clip_encoder is None:
            from ingestion import CLIPEncoder
            self._clip_encoder = CLIPEncoder(
                model_name=self.clip_model,
                pretrained=self.clip_pretrained,
                device=self.device,
            )
        return self._clip_encoder
        
    def _load_trace(self) -> pd.DataFrame:
        """Load the exploration trace."""
        trace_path = self.scene_dir / "exploration" / "trace.parquet"
        return pd.read_parquet(trace_path)
    
    def _load_depth(self, frame_id: int, trace: pd.DataFrame) -> Optional[np.ndarray]:
        """Load depth image for a frame."""
        row = trace.iloc[frame_id]
        depth_path = row.get('depth_path')
        
        if depth_path is None:
            return None
            
        full_path = self.scene_dir / "exploration" / depth_path
        if not full_path.exists():
            return None
            
        return np.load(full_path).astype(np.float32)
    
    def _load_image(self, frame_id: int, trace: pd.DataFrame) -> Optional[np.ndarray]:
        """Load RGB image for a frame."""
        row = trace.iloc[frame_id]
        image_path = row.get('image_path')
        
        if image_path is None:
            return None
            
        full_path = self.scene_dir / "exploration" / image_path
        if not full_path.exists():
            return None
            
        img = Image.open(full_path)
        return np.array(img)
    
    def _project_depth_to_3d(
        self,
        depth: np.ndarray,
        position: np.ndarray,
        rotation: np.ndarray,
        stride: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project depth image to 3D world coordinates.
        
        Returns:
            Tuple of (points_3d, valid_mask) where:
            - points_3d: (H//stride, W//stride, 3) world coordinates
            - valid_mask: (H//stride, W//stride) boolean mask of valid points
        """
        h, w = depth.shape
        
        # Camera intrinsics (Habitat defaults)
        hfov = np.deg2rad(90.0)
        fx = w / (2.0 * np.tan(hfov / 2.0))
        fy = fx
        cx, cy = w / 2.0, h / 2.0
        sensor_height = 1.5  # Camera above agent base
        
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
        
        points_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (H', W', 3)
        
        # Transform to world frame
        w_q, x_q, y_q, z_q = rotation
        R = np.array([
            [1 - 2*y_q*y_q - 2*z_q*z_q,     2*x_q*y_q - 2*z_q*w_q,     2*x_q*z_q + 2*y_q*w_q],
            [    2*x_q*y_q + 2*z_q*w_q, 1 - 2*x_q*x_q - 2*z_q*z_q,     2*y_q*z_q - 2*x_q*w_q],
            [    2*x_q*z_q - 2*y_q*w_q,     2*y_q*z_q + 2*x_q*w_q, 1 - 2*x_q*x_q - 2*y_q*y_q],
        ])
        
        # Sensor position
        sensor_pos = position.copy()
        sensor_pos[1] += sensor_height
        
        # Flatten for batch transform
        points_flat = points_cam.reshape(-1, 3)
        points_world = (R @ points_flat.T).T + sensor_pos
        points_3d = points_world.reshape(points_cam.shape)
        
        return points_3d, valid_mask
    
    def build_map(self, max_frames: Optional[int] = None, verbose: bool = True) -> MapBuildStats:
        """
        Build the dense semantic map from all frames.
        
        This is the EXPENSIVE upfront work that VLMaps/ConceptFusion does.
        Memory-efficient: voxelize every N frames to avoid OOM.
        
        Args:
            max_frames: Limit number of frames (for testing)
            verbose: Print progress
            
        Returns:
            Build statistics
        """
        start_time = time.time()
        
        # Load trace
        trace = self._load_trace()
        num_frames = len(trace)
        if max_frames:
            num_frames = min(num_frames, max_frames)
            
        if verbose:
            print(f"Building dense map for {self.scene_dir.name}")
            print(f"  Processing {num_frames} frames...")
        
        # Get CLIP encoder
        clip = self._get_clip_encoder()
        
        # Use a voxel dictionary for memory efficiency
        # Key: voxel index tuple, Value: (point_sum, feature_sum, count, frame_id)
        voxel_dict = {}
        num_points_raw = 0
        BATCH_SIZE = 20  # Voxelize every 20 frames
        
        temp_points = []
        temp_features = []
        temp_frame_ids = []
        
        for frame_id in range(num_frames):
            # Load depth
            depth = self._load_depth(frame_id, trace)
            if depth is None:
                continue
                
            # Load and encode image
            image = self._load_image(frame_id, trace)
            if image is None:
                continue
                
            # Get CLIP embedding for this frame
            embedding = clip.encode_image(image)  # (512,)
            
            # Get camera pose
            row = trace.iloc[frame_id]
            position = np.array([row['x'], row['y'], row['z']])
            rotation = np.array([row['qw'], row['qx'], row['qy'], row['qz']])
            
            # Project depth to 3D
            points_3d, valid_mask = self._project_depth_to_3d(
                depth, position, rotation, stride=self.sample_stride
            )
            
            # Get valid points
            valid_points = points_3d[valid_mask]  # (N_valid, 3)
            
            if len(valid_points) == 0:
                continue
            
            num_points_raw += len(valid_points)
            
            # Assign CLIP embedding to ALL valid points
            valid_features = np.tile(embedding, (len(valid_points), 1))  # (N_valid, 512)
            valid_frame_ids = np.full(len(valid_points), frame_id, dtype=np.int32)
            
            temp_points.append(valid_points)
            temp_features.append(valid_features)
            temp_frame_ids.append(valid_frame_ids)
            
            # Periodic voxelization to save memory
            if len(temp_points) >= BATCH_SIZE:
                self._merge_to_voxel_dict(
                    voxel_dict,
                    np.vstack(temp_points),
                    np.vstack(temp_features),
                    np.concatenate(temp_frame_ids)
                )
                temp_points = []
                temp_features = []
                temp_frame_ids = []
                gc.collect()
            
            if verbose and (frame_id + 1) % 20 == 0:
                print(f"    Frame {frame_id + 1}/{num_frames}: {len(voxel_dict):,} voxels, {num_points_raw:,} raw points")
        
        # Final merge
        if temp_points:
            self._merge_to_voxel_dict(
                voxel_dict,
                np.vstack(temp_points),
                np.vstack(temp_features),
                np.concatenate(temp_frame_ids)
            )
            temp_points = []
            temp_features = []
            temp_frame_ids = []
            gc.collect()
        
        if verbose:
            print(f"  Raw points: {num_points_raw:,}")
            print(f"  After voxelization ({self.voxel_size}m): {len(voxel_dict):,} voxels")
        
        # Convert voxel dict to arrays
        num_voxels = len(voxel_dict)
        if num_voxels == 0:
            raise ValueError("No valid points extracted!")
            
        points = np.zeros((num_voxels, 3), dtype=np.float32)
        features = np.zeros((num_voxels, 512), dtype=np.float32)
        frame_ids = np.zeros(num_voxels, dtype=np.int32)
        
        for i, (key, (pt_sum, feat_sum, count, fid)) in enumerate(voxel_dict.items()):
            points[i] = pt_sum / count
            features[i] = feat_sum / count
            frame_ids[i] = fid
        
        # Re-normalize features
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        features = features / norms
        
        # Store map
        self.points_3d = points
        self.point_features = features
        self.point_frame_ids = frame_ids
        self._map_built = True
        
        # Calculate memory size
        memory_bytes = (
            points.nbytes + 
            features.nbytes + 
            frame_ids.nbytes
        )
        memory_mb = memory_bytes / (1024 * 1024)
        
        build_time = time.time() - start_time
        
        self.build_stats = MapBuildStats(
            scene_id=self.scene_dir.name,
            num_frames_processed=num_frames,
            num_points_total=num_points_raw,
            num_points_after_voxel=num_voxels,
            build_time_seconds=build_time,
            memory_size_mb=memory_mb,
            frames_per_second=num_frames / build_time if build_time > 0 else 0,
        )
        
        if verbose:
            print(f"  Build time: {build_time:.1f}s ({self.build_stats.frames_per_second:.1f} fps)")
            print(f"  Memory: {memory_mb:.1f} MB")
        
        return self.build_stats
    
    def _merge_to_voxel_dict(
        self,
        voxel_dict: Dict,
        points: np.ndarray,
        features: np.ndarray,
        frame_ids: np.ndarray,
    ):
        """Merge points into voxel dictionary (incremental voxelization)."""
        # Quantize to voxel indices
        voxel_indices = np.floor(points / self.voxel_size).astype(np.int32)
        
        for i in range(len(points)):
            key = (voxel_indices[i, 0], voxel_indices[i, 1], voxel_indices[i, 2])
            if key in voxel_dict:
                pt_sum, feat_sum, count, fid = voxel_dict[key]
                voxel_dict[key] = (
                    pt_sum + points[i],
                    feat_sum + features[i],
                    count + 1,
                    fid  # Keep first frame_id
                )
            else:
                voxel_dict[key] = (
                    points[i].copy(),
                    features[i].copy(),
                    1,
                    frame_ids[i]
                )
    
    def query(
        self,
        text_query: str,
        top_k_percent: float = 0.01,  # Top 1% of points
        cluster_eps: float = 0.5,
        cluster_min_samples: int = 3,
    ) -> MapQueryResult:
        """
        Query the dense map for an object.
        
        This is the FAST part - just similarity search + clustering.
        
        Args:
            text_query: Natural language query
            top_k_percent: Percentage of top-scoring points to cluster
            cluster_eps: DBSCAN epsilon
            cluster_min_samples: DBSCAN min samples
            
        Returns:
            Query result with predicted location
        """
        if not self._map_built:
            raise ValueError("Map not built! Call build_map() first.")
            
        start_time = time.time()
        
        # Encode text query
        clip = self._get_clip_encoder()
        text_embedding = clip.encode_text(text_query)  # (512,)
        
        # Compute similarity with all map points
        # Dot product (embeddings are normalized)
        similarities = self.point_features @ text_embedding  # (N,)
        
        # Get top-k points
        k = max(10, int(len(similarities) * top_k_percent))
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        top_points = self.points_3d[top_indices]
        top_scores = similarities[top_indices]
        
        # Cluster with DBSCAN
        if len(top_points) < cluster_min_samples:
            # Not enough points
            query_time = (time.time() - start_time) * 1000
            return MapQueryResult(
                query=text_query,
                success=False,
                predicted_location=None,
                confidence=0.0,
                query_time_ms=query_time,
                num_points_matched=len(top_points),
                cluster_size=0,
            )
        
        clustering = DBSCAN(eps=cluster_eps, min_samples=cluster_min_samples)
        labels = clustering.fit_predict(top_points)
        
        # Find best cluster (highest average score)
        best_cluster_score = -1
        best_centroid = None
        best_cluster_size = 0
        
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:
                continue
                
            mask = labels == label
            cluster_points = top_points[mask]
            cluster_scores = top_scores[mask]
            
            avg_score = np.mean(cluster_scores)
            if avg_score > best_cluster_score:
                best_cluster_score = avg_score
                best_centroid = np.mean(cluster_points, axis=0)
                best_cluster_size = len(cluster_points)
        
        query_time = (time.time() - start_time) * 1000
        
        success = best_centroid is not None
        
        return MapQueryResult(
            query=text_query,
            success=success,
            predicted_location=best_centroid,
            confidence=float(best_cluster_score) if success else 0.0,
            query_time_ms=query_time,
            num_points_matched=k,
            cluster_size=best_cluster_size,
        )
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage of the map in MB."""
        if not self._map_built:
            return {"total_mb": 0}
            
        return {
            "points_mb": self.points_3d.nbytes / (1024 * 1024),
            "features_mb": self.point_features.nbytes / (1024 * 1024),
            "frame_ids_mb": self.point_frame_ids.nbytes / (1024 * 1024),
            "total_mb": (
                self.points_3d.nbytes + 
                self.point_features.nbytes + 
                self.point_frame_ids.nbytes
            ) / (1024 * 1024),
        }
    
    def save_map(self, output_path: Path):
        """Save the built map to disk."""
        if not self._map_built:
            raise ValueError("Map not built!")
            
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        np.save(output_path / "points_3d.npy", self.points_3d)
        np.save(output_path / "point_features.npy", self.point_features)
        np.save(output_path / "point_frame_ids.npy", self.point_frame_ids)
        
        with open(output_path / "build_stats.json", 'w') as f:
            json.dump(asdict(self.build_stats), f, indent=2)
    
    def load_map(self, map_path: Path):
        """Load a pre-built map from disk."""
        map_path = Path(map_path)
        
        self.points_3d = np.load(map_path / "points_3d.npy")
        self.point_features = np.load(map_path / "point_features.npy")
        self.point_frame_ids = np.load(map_path / "point_frame_ids.npy")
        
        with open(map_path / "build_stats.json") as f:
            stats_dict = json.load(f)
            self.build_stats = MapBuildStats(**stats_dict)
        
        self._map_built = True


class JITBaselineWrapper:
    """
    Wrapper around JIT Cascade for fair comparison.
    
    Provides same interface as DenseMapBaseline for benchmarking.
    """
    
    def __init__(self, scene_dir: Path, **jit_kwargs):
        """Initialize JIT baseline."""
        self.scene_dir = Path(scene_dir)
        self.jit_kwargs = jit_kwargs
        self._cascade = None
        self._load_time = 0.0
        
    def build_map(self, **kwargs) -> MapBuildStats:
        """
        "Build" for JIT = just load the index (very fast).
        """
        start_time = time.time()
        
        from retrieval.cascade import JITRetrievalCascade
        
        trace_dir = str(self.scene_dir / "exploration")
        self._cascade = JITRetrievalCascade(trace_dir, **self.jit_kwargs)
        
        # Force initialization to measure load time
        self._cascade._initialize_pipeline()
        
        self._load_time = time.time() - start_time
        
        # Calculate "storage" - just the FAISS index + trace metadata
        index_path = self.scene_dir / "exploration" / "memory.index"
        trace_path = self.scene_dir / "exploration" / "trace.parquet"
        
        storage_mb = 0
        if index_path.exists():
            storage_mb += index_path.stat().st_size / (1024 * 1024)
        if trace_path.exists():
            storage_mb += trace_path.stat().st_size / (1024 * 1024)
        
        return MapBuildStats(
            scene_id=self.scene_dir.name,
            num_frames_processed=0,  # No frames processed upfront
            num_points_total=0,
            num_points_after_voxel=0,
            build_time_seconds=self._load_time,
            memory_size_mb=storage_mb,
            frames_per_second=float('inf'),  # Instant
        )
    
    def query(self, text_query: str, **kwargs) -> MapQueryResult:
        """Query using JIT cascade."""
        if self._cascade is None:
            raise ValueError("Not initialized! Call build_map() first.")
            
        start_time = time.time()
        
        result = self._cascade.query(text_query)
        
        query_time = (time.time() - start_time) * 1000
        
        success = result.success and result.best_location is not None
        location = result.best_location.centroid_3d if success else None
        confidence = result.best_location.best_detection_score if success else 0.0
        
        return MapQueryResult(
            query=text_query,
            success=success,
            predicted_location=location,
            confidence=float(confidence),
            query_time_ms=query_time,
            num_points_matched=result.l1_candidates,
            cluster_size=result.l2_clusters,
        )
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage (index size)."""
        index_path = self.scene_dir / "exploration" / "memory.index"
        trace_path = self.scene_dir / "exploration" / "trace.parquet"
        
        total_mb = 0
        if index_path.exists():
            total_mb += index_path.stat().st_size / (1024 * 1024)
        if trace_path.exists():
            total_mb += trace_path.stat().st_size / (1024 * 1024)
            
        return {"total_mb": total_mb}


if __name__ == "__main__":
    # Quick test
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--query", type=str, default="couch")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Dense Map Baseline Test")
    print("=" * 60)
    
    # Test Dense Map
    print("\n[1] Testing Dense Map...")
    dense_map = DenseMapBaseline(args.scene_dir)
    build_stats = dense_map.build_map(verbose=True)
    
    print(f"\n[2] Querying for '{args.query}'...")
    result = dense_map.query(args.query)
    print(f"  Success: {result.success}")
    print(f"  Location: {result.predicted_location}")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Query time: {result.query_time_ms:.1f} ms")
    
    # Compare with JIT
    print("\n[3] Comparing with JIT Cascade...")
    jit = JITBaselineWrapper(args.scene_dir)
    jit_build = jit.build_map()
    jit_result = jit.query(args.query)
    
    print(f"\n{'Metric':<25} {'Dense Map':<20} {'JIT Cascade':<20}")
    print("-" * 65)
    print(f"{'Build time (s)':<25} {build_stats.build_time_seconds:<20.2f} {jit_build.build_time_seconds:<20.2f}")
    print(f"{'Memory (MB)':<25} {build_stats.memory_size_mb:<20.1f} {jit_build.memory_size_mb:<20.1f}")
    print(f"{'Query time (ms)':<25} {result.query_time_ms:<20.1f} {jit_result.query_time_ms:<20.1f}")
    print(f"{'Success':<25} {str(result.success):<20} {str(jit_result.success):<20}")
