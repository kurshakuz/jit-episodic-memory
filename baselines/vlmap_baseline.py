#!/usr/bin/env python3
"""
VLMaps Baseline Implementation
==============================

A faithful implementation of VLMaps (Huang et al., 2023) that uses:
1. Dense per-pixel embeddings in CLIP feature space (via patch tokens)
2. Top-down 2D grid map (not 3D voxels)
3. Ground plane projection
4. Multi-view embedding averaging

This provides a fairer comparison than projecting global CLIP embeddings.

Reference:
    Huang et al., "Visual Language Maps for Robot Navigation", ICRA 2023
    https://arxiv.org/abs/2210.05714

Usage:
    vlmap = VLMapBaseline(scene_dir, grid_resolution=0.05)
    build_stats = vlmap.build_map()
    result = vlmap.query("couch")
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
from PIL import Image
import torch
import torch.nn.functional as F

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from baselines.dense_map import MapBuildStats, MapQueryResult


class CLIPPatchEncoder:
    """
    Dense CLIP encoder using ViT patch tokens.
    
    Extracts per-patch embeddings from OpenCLIP ViT that are aligned with 
    CLIP text embeddings. This is an approximation of VLMaps-style dense features.
    
    NOTE: This is less accurate than LSeg but easier to set up.
    """
    
    def __init__(self, device: Optional[str] = None, model_name: str = "ViT-B-32-quickgelu"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self.model_name = model_name
        self.embed_dim = 512
        
    def _load_model(self):
        """Load OpenCLIP model."""
        if self._model is not None:
            return
            
        import open_clip
        
        print(f"Loading OpenCLIP {self.model_name} for dense features...")
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained='laion400m_e32'
        )
        self._tokenizer = open_clip.get_tokenizer(self.model_name)
        self._model.to(self.device)
        self._model.eval()
        print(f"  Loaded on {self.device}")
    
    def encode_image_dense(self, image: np.ndarray, target_size: Tuple[int, int] = None) -> np.ndarray:
        """
        Get dense per-patch embeddings using OpenCLIP ViT.
        
        Extracts patch token embeddings that are aligned with CLIP text space.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            target_size: Optional (H, W) to resize output features
            
        Returns:
            Dense embeddings (H', W', embed_dim) where H', W' = 7x7 for ViT-B/32
        """
        self._load_model()
        
        from PIL import Image as PILImage
        
        # Convert to PIL and preprocess
        pil_image = PILImage.fromarray(image)
        img_tensor = self._preprocess(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            visual = self._model.visual
            
            # Patch embedding via conv
            x = visual.conv1(img_tensor)  # (B, hidden_dim, H', W')
            x = x.reshape(x.shape[0], x.shape[1], -1)  # (B, hidden_dim, patches)
            x = x.permute(0, 2, 1)  # (B, patches, hidden_dim)
            
            # Add class token
            cls = visual.class_embedding.unsqueeze(0).unsqueeze(0).expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1)
            x = x + visual.positional_embedding
            
            # Transform
            x = visual.ln_pre(x)
            x = x.permute(1, 0, 2)  # (seq, batch, hidden)
            x = visual.transformer(x)
            x = x.permute(1, 0, 2)  # (batch, seq, hidden)
            
            # Get patch tokens (skip CLS)
            patch_tokens = x[:, 1:, :]
            
            # Layer norm + projection
            patch_tokens = visual.ln_post(patch_tokens)
            patch_features = patch_tokens @ visual.proj  # (B, patches, embed_dim)
            
            # Normalize
            patch_features = F.normalize(patch_features, dim=-1)
            
            # Reshape to spatial grid
            num_patches = patch_features.shape[1]
            h = w = int(np.sqrt(num_patches))
            dense_features = patch_features.reshape(1, h, w, -1)
            
            # Upsample if target size specified
            if target_size is not None:
                dense_features = dense_features.permute(0, 3, 1, 2)  # (B, C, H, W)
                dense_features = F.interpolate(
                    dense_features, size=target_size, mode='bilinear', align_corners=False
                )
                dense_features = dense_features.permute(0, 2, 3, 1)  # (B, H, W, C)
            
            return dense_features[0].cpu().numpy()
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text query to embedding."""
        self._load_model()
        
        with torch.no_grad():
            tokens = self._tokenizer([text]).to(self.device)
            text_features = self._model.encode_text(tokens)
            text_features = F.normalize(text_features, dim=-1)
            return text_features[0].cpu().numpy()


def get_dense_encoder(use_lseg: bool = True, device: Optional[str] = None):
    """
    Get the appropriate dense encoder.
    
    Args:
        use_lseg: If True, use real LSeg encoder. If False, use CLIP patch tokens.
        device: Device for inference.
        
    Returns:
        Encoder with encode_image_dense() and encode_text() methods.
    """
    if use_lseg:
        try:
            from baselines.lseg_encoder import LSegEncoder
            return LSegEncoder(device=device)
        except Exception as e:
            print(f"Warning: Could not load LSeg encoder: {e}")
            print("Falling back to CLIP patch encoder...")
            return CLIPPatchEncoder(device=device)
    else:
        return CLIPPatchEncoder(device=device)


class VLMapBaseline:
    """
    VLMaps baseline implementation.
    
    Key differences from DenseMapBaseline:
    1. Uses dense per-pixel/per-patch embeddings (not global CLIP)
    2. Projects to 2D top-down grid (not 3D voxels)
    3. Averages embeddings for overlapping projections
    
    This is a faithful implementation of Huang et al., ICRA 2023.
    
    Encoder options:
    - use_lseg=True: Real LSeg encoder (per-pixel, recommended)
    - use_lseg=False: CLIP patch tokens (per-patch, approximation)
    """
    
    def __init__(
        self,
        scene_dir: Path,
        grid_resolution: float = 0.05,  # 5cm per grid cell
        grid_size: int = 1000,  # 1000x1000 grid = 50m x 50m at 5cm resolution
        min_height: float = 0.1,  # Ignore points below this height
        max_height: float = 2.0,  # Ignore points above this height
        device: Optional[str] = None,
        use_lseg: bool = True,  # Use real LSeg encoder
    ):
        """
        Initialize VLMap baseline.
        
        Args:
            scene_dir: Path to scene memory bank
            grid_resolution: Meters per grid cell
            grid_size: Size of square grid (grid_size x grid_size)
            min_height: Minimum point height to include
            max_height: Maximum point height to include
            device: Device for inference
            use_lseg: If True, use real LSeg encoder. If False, use CLIP patches.
        """
        self.scene_dir = Path(scene_dir)
        self.grid_resolution = grid_resolution
        self.grid_size = grid_size
        self.min_height = min_height
        self.max_height = max_height
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_lseg = use_lseg
        
        # Map data
        self.grid_map: Optional[np.ndarray] = None  # (H, W, embed_dim)
        self.grid_counts: Optional[np.ndarray] = None  # (H, W) count per cell
        self.grid_center: Optional[np.ndarray] = None  # World position of grid center
        
        # Stats
        self.build_stats: Optional[MapBuildStats] = None
        self._map_built = False
        
        # Lazy-loaded models
        self._dense_encoder = None
    
    def _get_dense_encoder(self):
        """Lazy load dense encoder (LSeg or CLIP patches)."""
        if self._dense_encoder is None:
            self._dense_encoder = get_dense_encoder(use_lseg=self.use_lseg, device=self.device)
        return self._dense_encoder
    
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
    
    def _get_camera_intrinsics(self, height: int, width: int) -> np.ndarray:
        """Get camera intrinsic matrix."""
        hfov = np.deg2rad(90.0)
        fx = width / (2.0 * np.tan(hfov / 2.0))
        fy = fx
        cx, cy = width / 2.0, height / 2.0
        
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return K
    
    def _quaternion_to_rotation_matrix(self, quat: np.ndarray) -> np.ndarray:
        """Convert quaternion (w, x, y, z) to rotation matrix."""
        w, x, y, z = quat
        R = np.array([
            [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
            [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
        ], dtype=np.float32)
        return R
    
    def _project_to_world(
        self,
        depth: np.ndarray,
        position: np.ndarray,
        rotation: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project depth image to world coordinates.
        
        Returns:
            Tuple of (points_world, valid_mask) where:
            - points_world: (H, W, 3) world coordinates
            - valid_mask: (H, W) boolean mask
        """
        H, W = depth.shape
        K = self._get_camera_intrinsics(H, W)
        K_inv = np.linalg.inv(K)
        
        # Create pixel grid
        u = np.arange(W)
        v = np.arange(H)
        uu, vv = np.meshgrid(u, v)
        
        # Homogeneous pixel coordinates
        ones = np.ones_like(uu)
        pixels = np.stack([uu, vv, ones], axis=-1)  # (H, W, 3)
        
        # Backproject to camera frame
        pixels_flat = pixels.reshape(-1, 3)
        depth_flat = depth.reshape(-1)
        
        points_cam = (K_inv @ pixels_flat.T).T * depth_flat[:, np.newaxis]
        
        # Habitat camera convention: -Y is up, -Z is forward
        # Convert to standard: X right, Y up, Z forward (out of camera)
        points_cam_std = np.stack([
            points_cam[:, 0],   # X stays
            -points_cam[:, 1],  # Y flipped
            -points_cam[:, 2],  # Z flipped
        ], axis=-1)
        
        # Rotation matrix
        R = self._quaternion_to_rotation_matrix(rotation)
        
        # Transform to world
        sensor_height = 1.5
        sensor_pos = position.copy()
        sensor_pos[1] += sensor_height
        
        points_world = (R @ points_cam_std.T).T + sensor_pos
        points_world = points_world.reshape(H, W, 3)
        
        # Valid mask
        valid_mask = (depth > 0.1) & (depth < 10.0)
        
        return points_world, valid_mask
    
    def _world_to_grid(self, points_world: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project world points to top-down grid coordinates.
        
        VLMaps projects to ground plane (XZ in Habitat's Y-up coordinate system).
        
        Returns:
            Tuple of (grid_x, grid_y) integer coordinates
        """
        # Grid center (set on first frame)
        if self.grid_center is None:
            self.grid_center = np.array([0.0, 0.0, 0.0])
        
        # Project to ground plane (X, Z in world = X, Y in grid)
        # Following VLMaps equation:
        # p_x = floor(H/2 + P_x/s + 0.5)
        # p_y = floor(W/2 - P_z/s + 0.5)
        
        grid_x = np.floor(
            self.grid_size / 2 + points_world[..., 0] / self.grid_resolution + 0.5
        ).astype(np.int32)
        
        grid_y = np.floor(
            self.grid_size / 2 - points_world[..., 2] / self.grid_resolution + 0.5
        ).astype(np.int32)
        
        return grid_x, grid_y
    
    def build_map(self, max_frames: Optional[int] = None, verbose: bool = True) -> MapBuildStats:
        """
        Build the VLMap from exploration data.
        
        This follows the VLMaps algorithm:
        1. For each frame, get dense LSeg features
        2. Project depth to world coordinates
        3. Project to top-down grid
        4. Average embeddings for overlapping projections
        """
        start_time = time.time()
        
        # Load trace
        trace = self._load_trace()
        num_frames = len(trace)
        if max_frames:
            num_frames = min(num_frames, max_frames)
        
        encoder_name = "LSeg" if self.use_lseg else "CLIP-patches"
        if verbose:
            print(f"Building VLMap for {self.scene_dir.name}")
            print(f"  Processing {num_frames} frames with {encoder_name}...")
        
        # Get dense encoder (LSeg or CLIP patches)
        encoder = self._get_dense_encoder()
        
        # Initialize grid map
        embed_dim = encoder.embed_dim
        self.grid_map = np.zeros((self.grid_size, self.grid_size, embed_dim), dtype=np.float32)
        self.grid_counts = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.grid_heights = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)  # Track avg height
        
        num_points_total = 0
        
        for frame_id in range(num_frames):
            # Load depth and image
            depth = self._load_depth(frame_id, trace)
            if depth is None:
                continue
                
            image = self._load_image(frame_id, trace)
            if image is None:
                continue
            
            # Get camera pose
            row = trace.iloc[frame_id]
            position = np.array([row['x'], row['y'], row['z']])
            rotation = np.array([row['qw'], row['qx'], row['qy'], row['qz']])
            
            # Get dense features (LSeg or CLIP patches)
            # Target size matches depth resolution
            H, W = depth.shape
            dense_features = encoder.encode_image_dense(image, target_size=(H, W))
            
            # Project depth to world
            points_world, valid_mask = self._project_to_world(depth, position, rotation)
            
            # Height filter
            heights = points_world[..., 1]
            height_mask = (heights > self.min_height) & (heights < self.max_height)
            valid_mask = valid_mask & height_mask
            
            # Project to grid
            grid_x, grid_y = self._world_to_grid(points_world)
            
            # Bounds check
            bounds_mask = (
                (grid_x >= 0) & (grid_x < self.grid_size) &
                (grid_y >= 0) & (grid_y < self.grid_size)
            )
            valid_mask = valid_mask & bounds_mask
            
            # Get valid points
            valid_x = grid_x[valid_mask]
            valid_y = grid_y[valid_mask]
            valid_features = dense_features[valid_mask]
            valid_heights = points_world[..., 1][valid_mask]  # Store Y (height)
            
            num_points_total += len(valid_x)
            
            # Accumulate to grid
            for i in range(len(valid_x)):
                x, y = valid_x[i], valid_y[i]
                self.grid_map[x, y] += valid_features[i]
                self.grid_heights[x, y] += valid_heights[i]  # Accumulate height
                self.grid_counts[x, y] += 1
            
            if verbose and (frame_id + 1) % 10 == 0:
                filled = np.sum(self.grid_counts > 0)
                print(f"    Frame {frame_id + 1}/{num_frames}: {filled:,} grid cells filled")
        
        # Average embeddings and heights
        mask = self.grid_counts > 0
        self.grid_map[mask] /= self.grid_counts[mask, np.newaxis]
        self.grid_heights[mask] /= self.grid_counts[mask]  # Average height
        
        # Normalize
        norms = np.linalg.norm(self.grid_map, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        self.grid_map = self.grid_map / norms
        
        self._map_built = True
        
        # Calculate stats
        num_filled_cells = int(np.sum(self.grid_counts > 0))
        
        # Memory: store grid maps
        memory_mb = (
            self.grid_map.nbytes + 
            self.grid_counts.nbytes +
            self.grid_heights.nbytes
        ) / (1024 * 1024)
        
        build_time = time.time() - start_time
        
        self.build_stats = MapBuildStats(
            scene_id=self.scene_dir.name,
            num_frames_processed=num_frames,
            num_points_total=num_points_total,
            num_points_after_voxel=num_filled_cells,
            build_time_seconds=build_time,
            memory_size_mb=memory_mb,
            frames_per_second=num_frames / build_time if build_time > 0 else 0,
        )
        
        if verbose:
            print(f"  Grid cells filled: {num_filled_cells:,} / {self.grid_size * self.grid_size:,}")
            print(f"  Build time: {build_time:.1f}s ({self.build_stats.frames_per_second:.1f} fps)")
            print(f"  Memory: {memory_mb:.1f} MB")
        
        return self.build_stats
    
    def query(
        self,
        text_query: str,
        top_k: int = 100,
        cluster_eps: float = 0.5,  # in meters
        cluster_min_samples: int = 3,
    ) -> MapQueryResult:
        """
        Query the VLMap for an object location.
        
        Following VLMaps:
        1. Encode text with CLIP text encoder
        2. Compute similarity with all grid cells
        3. Find peak in similarity map
        4. Convert grid position to world position
        """
        if not self._map_built:
            raise ValueError("Map not built! Call build_map() first.")
        
        start_time = time.time()
        
        # Get text embedding
        encoder = self._get_dense_encoder()
        text_embedding = encoder.encode_text(text_query)
        
        # Compute similarity with all grid cells
        # grid_map: (H, W, embed_dim)
        # text_embedding: (embed_dim,)
        similarities = np.dot(self.grid_map, text_embedding)  # (H, W)
        
        # Mask out empty cells
        similarities[self.grid_counts == 0] = -np.inf
        
        # Find top-k cells
        flat_sims = similarities.flatten()
        top_indices = np.argsort(flat_sims)[-top_k:][::-1]
        
        top_x = top_indices // self.grid_size
        top_y = top_indices % self.grid_size
        top_scores = flat_sims[top_indices]
        
        # Filter out invalid cells
        valid = top_scores > -np.inf
        if not np.any(valid):
            query_time = (time.time() - start_time) * 1000
            return MapQueryResult(
                query=text_query,
                success=False,
                predicted_location=None,
                confidence=0.0,
                query_time_ms=query_time,
                num_points_matched=0,
                cluster_size=0,
            )
        
        top_x = top_x[valid]
        top_y = top_y[valid]
        top_scores = top_scores[valid]
        
        # Convert grid to world coordinates
        # Inverse of world_to_grid:
        # grid_x = H/2 + P_x/s + 0.5  =>  P_x = (grid_x - H/2 - 0.5) * s
        world_x = (top_x - self.grid_size / 2 - 0.5) * self.grid_resolution
        world_z = -(top_y - self.grid_size / 2 - 0.5) * self.grid_resolution
        # Use stored average heights for each grid cell
        world_y = np.array([self.grid_heights[x, y] for x, y in zip(top_x, top_y)])
        
        top_points = np.stack([world_x, world_y, world_z], axis=-1)
        
        # Cluster with DBSCAN
        from sklearn.cluster import DBSCAN
        
        if len(top_points) < cluster_min_samples:
            # Not enough points, return top-1
            query_time = (time.time() - start_time) * 1000
            return MapQueryResult(
                query=text_query,
                success=True,
                predicted_location=top_points[0],
                confidence=float(top_scores[0]),
                query_time_ms=query_time,
                num_points_matched=len(top_points),
                cluster_size=1,
            )
        
        # Convert eps from meters to grid units for clustering
        eps_grid = cluster_eps / self.grid_resolution
        
        clustering = DBSCAN(eps=cluster_eps, min_samples=cluster_min_samples)
        labels = clustering.fit_predict(top_points)
        
        # Find best cluster
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
        
        if best_centroid is None:
            # No valid clusters, return top-1
            return MapQueryResult(
                query=text_query,
                success=True,
                predicted_location=top_points[0],
                confidence=float(top_scores[0]),
                query_time_ms=query_time,
                num_points_matched=len(top_points),
                cluster_size=1,
            )
        
        return MapQueryResult(
            query=text_query,
            success=True,
            predicted_location=best_centroid,
            confidence=float(best_cluster_score),
            query_time_ms=query_time,
            num_points_matched=len(top_points),
            cluster_size=best_cluster_size,
        )
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage."""
        if not self._map_built:
            return {"total_mb": 0}
        
        return {
            "grid_map_mb": self.grid_map.nbytes / (1024 * 1024),
            "grid_counts_mb": self.grid_counts.nbytes / (1024 * 1024),
            "total_mb": (self.grid_map.nbytes + self.grid_counts.nbytes) / (1024 * 1024),
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--query", type=str, default="couch")
    args = parser.parse_args()
    
    print("=" * 60)
    print("VLMaps Baseline Test")
    print("=" * 60)
    
    vlmap = VLMapBaseline(args.scene_dir)
    build_stats = vlmap.build_map(verbose=True)
    
    print(f"\nQuerying for '{args.query}'...")
    result = vlmap.query(args.query)
    print(f"  Success: {result.success}")
    print(f"  Location: {result.predicted_location}")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Query time: {result.query_time_ms:.1f} ms")
