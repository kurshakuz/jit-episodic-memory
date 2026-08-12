#!/usr/bin/env python3
"""
Phase 2: Memory Logger
======================

The "Lazy Ingestion" pipeline that records robot experience with minimal compute.

Key features:
- Adaptive keyframe selection (not every frame)
- Lightweight CLIP embeddings
- Efficient Parquet storage for metadata
- FAISS index for fast retrieval
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import json
from PIL import Image
import time

from .clip_encoder import CLIPEncoder
from .keyframe_selector import KeyframeSelector
from .faiss_indexer import FAISSIndexer


@dataclass
class KeyframeRecord:
    """Record of a single keyframe."""
    frame_id: int
    timestamp: float
    position: List[float]  # [x, y, z]
    rotation: List[float]  # quaternion [w, x, y, z]
    image_path: str
    depth_path: Optional[str]
    embedding: Optional[np.ndarray]  # CLIP embedding


class MemoryLogger:
    """
    Logs robot experience with "Lazy Ingestion" - minimal processing at capture time.
    
    Heavy perception is deferred to query time.
    """
    
    def __init__(
        self,
        output_dir: str,
        clip_encoder: Optional[CLIPEncoder] = None,
        keyframe_selector: Optional[KeyframeSelector] = None,
        save_depth: bool = True,
        image_quality: int = 85,
    ):
        """
        Initialize Memory Logger.
        
        Args:
            output_dir: Directory to save keyframes and index
            clip_encoder: CLIP encoder instance (created if None)
            keyframe_selector: Keyframe selector (created if None)
            save_depth: Whether to save depth images
            image_quality: JPEG quality (0-100)
        """
        self.output_dir = Path(output_dir)
        self.save_depth = save_depth
        self.image_quality = image_quality
        
        # Create directories
        self.images_dir = self.output_dir / "images"
        self.depth_dir = self.output_dir / "depth"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        if save_depth:
            self.depth_dir.mkdir(parents=True, exist_ok=True)
            
        # Components
        self.clip_encoder = clip_encoder or CLIPEncoder()
        self.selector = keyframe_selector or KeyframeSelector()
        self.indexer = FAISSIndexer(embedding_dim=self.clip_encoder.embedding_dim)
        
        # State
        self.keyframes: List[KeyframeRecord] = []
        self.embeddings: List[np.ndarray] = []
        self.session_start = time.time()
        self.total_frames_seen = 0
        
    def log_frame(
        self,
        rgb: np.ndarray,
        depth: Optional[np.ndarray],
        position: List[float],
        rotation: List[float],
        frame_id: Optional[int] = None,
        force_save: bool = False,
    ) -> Tuple[bool, Optional[KeyframeRecord]]:
        """
        Process a frame and optionally save as keyframe.
        
        This is the "lazy" part - we only do heavy work if needed.
        
        Args:
            rgb: RGB image (H, W, 3) or (H, W, 4)
            depth: Depth image (H, W) in meters
            position: Agent position [x, y, z]
            rotation: Agent rotation quaternion [w, x, y, z]
            frame_id: Optional frame ID (auto-increment if None)
            force_save: Force save this frame regardless of selection
            
        Returns:
            Tuple of (was_saved, keyframe_record or None)
        """
        self.total_frames_seen += 1
        
        if frame_id is None:
            frame_id = self.total_frames_seen
            
        # Compute CLIP embedding (this is the "light" work we do per frame)
        embedding = self.clip_encoder.encode_image(rgb)
        
        # Decide if this should be a keyframe
        decision = self.selector.should_save_keyframe(
            embedding=embedding,
            position=np.array(position),
            rotation=np.array(rotation),
        )
        
        if not decision.should_save and not force_save:
            return False, None
            
        # Save keyframe
        timestamp = time.time() - self.session_start
        
        # Save RGB image
        image_filename = f"frame_{frame_id:06d}.jpg"
        image_path = self.images_dir / image_filename
        
        # Handle RGBA
        if rgb.shape[-1] == 4:
            rgb = rgb[:, :, :3]
        
        Image.fromarray(rgb).save(
            image_path, 
            quality=self.image_quality,
            optimize=True,
        )
        
        # Save depth if enabled
        depth_path = None
        if self.save_depth and depth is not None:
            depth_filename = f"depth_{frame_id:06d}.npy"
            depth_path = self.depth_dir / depth_filename
            np.save(depth_path, depth.astype(np.float16))  # Half precision to save space
            
        # Create record
        record = KeyframeRecord(
            frame_id=frame_id,
            timestamp=timestamp,
            position=list(position),
            rotation=list(rotation),
            image_path=str(image_path.relative_to(self.output_dir)),
            depth_path=str(depth_path.relative_to(self.output_dir)) if depth_path else None,
            embedding=embedding,
        )
        
        # Store
        self.keyframes.append(record)
        self.embeddings.append(embedding)
        
        # Add to FAISS index for incremental building
        self.indexer.add(embedding, frame_id)
        
        return True, record
    
    def finalize(self) -> Dict[str, Any]:
        """
        Finalize the trace and save to disk.
        
        Returns:
            Statistics about the recorded session
        """
        if not self.keyframes:
            print("Warning: No keyframes recorded!")
            return {}
            
        # Build final FAISS index from all embeddings
        if self.embeddings:
            embeddings_array = np.stack(self.embeddings)
            frame_ids = [kf.frame_id for kf in self.keyframes]
            self.indexer.build_index(embeddings_array, frame_ids)
            
        # Save FAISS index
        self.indexer.save(str(self.output_dir / "memory"))
        
        # Save trace as Parquet (efficient columnar storage)
        trace_data = []
        for kf in self.keyframes:
            row = {
                "frame_id": kf.frame_id,
                "timestamp": kf.timestamp,
                "x": kf.position[0],
                "y": kf.position[1],
                "z": kf.position[2],
                "qw": kf.rotation[0],
                "qx": kf.rotation[1],
                "qy": kf.rotation[2],
                "qz": kf.rotation[3],
                "image_path": kf.image_path,
                "depth_path": kf.depth_path,
            }
            trace_data.append(row)
            
        df = pd.DataFrame(trace_data)
        trace_path = self.output_dir / "trace.parquet"
        df.to_parquet(trace_path, index=False)
        print(f"Saved trace to {trace_path}")
        
        # Save embeddings as numpy array
        if self.embeddings:
            embeddings_path = self.output_dir / "embeddings.npy"
            np.save(embeddings_path, embeddings_array)
            print(f"Saved embeddings to {embeddings_path}")
        
        # Compute statistics
        stats = {
            "total_frames_seen": self.total_frames_seen,
            "keyframes_saved": len(self.keyframes),
            "compression_ratio": self.total_frames_seen / max(1, len(self.keyframes)),
            "keyframe_percentage": 100.0 * len(self.keyframes) / max(1, self.total_frames_seen),
            "session_duration": time.time() - self.session_start,
            "output_dir": str(self.output_dir),
        }
        
        # Save stats
        with open(self.output_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)
            
        print(f"\n--- Memory Logger Statistics ---")
        print(f"Total frames seen: {stats['total_frames_seen']}")
        print(f"Keyframes saved: {stats['keyframes_saved']}")
        print(f"Compression ratio: {stats['compression_ratio']:.1f}x")
        print(f"Keyframe percentage: {stats['keyframe_percentage']:.1f}%")
        
        return stats
    

class TraceLoader:
    """Utility class to load and query a saved trace."""
    
    def __init__(self, trace_dir: str):
        """
        Load a saved trace.
        
        Args:
            trace_dir: Directory containing saved trace files
        """
        self.trace_dir = Path(trace_dir)
        
        # Load trace metadata
        self.trace = pd.read_parquet(self.trace_dir / "trace.parquet")
        
        # Load embeddings
        embeddings_path = self.trace_dir / "embeddings.npy"
        if embeddings_path.exists():
            self.embeddings = np.load(embeddings_path)
        else:
            self.embeddings = None
            
        # Load FAISS index
        self.indexer = FAISSIndexer()
        self.indexer.load(str(self.trace_dir / "memory"))
        
        print(f"Loaded trace with {len(self.trace)} keyframes")
        
    def get_frame(self, frame_id: int) -> dict:
        """Get frame data by ID."""
        row = self.trace[self.trace["frame_id"] == frame_id].iloc[0]
        return row.to_dict()
    
    def load_image(self, frame_id: int) -> np.ndarray:
        """Load RGB image for a frame."""
        row = self.get_frame(frame_id)
        image_path = self.trace_dir / row["image_path"]
        return np.array(Image.open(image_path))
    
    def load_depth(self, frame_id: int) -> Optional[np.ndarray]:
        """Load depth map for a frame."""
        row = self.get_frame(frame_id)
        if row["depth_path"] is None:
            return None
        depth_path = self.trace_dir / row["depth_path"]
        return np.load(depth_path).astype(np.float32)
    
    def search(self, query_embedding: np.ndarray, k: int = 100) -> Tuple[List[int], List[float]]:
        """Search for similar frames."""
        return self.indexer.search(query_embedding, k)


# Simple test
if __name__ == "__main__":
    print("Testing Memory Logger...")
    
    # Create test output directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = MemoryLogger(output_dir=tmpdir)
        
        # Simulate logging frames
        np.random.seed(42)
        
        for i in range(100):
            # Random image
            rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            depth = np.random.rand(480, 640).astype(np.float32) * 10
            
            # Random pose
            position = [float(i * 0.1), 0.0, 0.0]
            rotation = [1.0, 0.0, 0.0, 0.0]
            
            saved, record = logger.log_frame(rgb, depth, position, rotation)
            if saved:
                print(f"Frame {i}: saved as keyframe")
                
        # Finalize
        stats = logger.finalize()
        
        # Test loading
        loader = TraceLoader(tmpdir)
        
        # Test search
        query = np.random.randn(512).astype(np.float32)
        frame_ids, similarities = loader.search(query, k=5)
        print(f"\nSearch results: {frame_ids}")
        
    print("[OK] Memory Logger test passed!")
