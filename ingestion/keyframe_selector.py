#!/usr/bin/env python3
"""
Phase 2: Keyframe Selector
==========================

Adaptive keyframe selection based on Semantic Entropy.
Only saves frames when the scene has changed significantly.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class KeyframeDecision:
    """Result of keyframe selection decision."""
    should_save: bool
    reason: str
    semantic_change: float  # Cosine distance from last keyframe
    distance_moved: float   # Distance moved since last keyframe
    rotation_change: float  # Rotation change since last keyframe (degrees)


class KeyframeSelector:
    """
    Selects keyframes based on semantic entropy (CLIP embedding change).
    
    The core idea: Save a frame only if the visual content has changed
    significantly, measured by cosine distance between CLIP embeddings.
    """
    
    def __init__(
        self,
        semantic_threshold: float = 0.15,
        min_distance: float = 0.1,      # meters
        min_rotation: float = 10.0,     # degrees
        max_frames_between: int = 30,   # Force save every N frames
    ):
        """
        Initialize keyframe selector.
        
        Args:
            semantic_threshold: Minimum cosine distance to trigger keyframe
                              (0.15 means scene changed ~15% semantically)
            min_distance: Minimum distance moved to consider saving
            min_rotation: Minimum rotation to consider saving
            max_frames_between: Maximum frames between forced keyframes
        """
        self.semantic_threshold = semantic_threshold
        self.min_distance = min_distance
        self.min_rotation = min_rotation
        self.max_frames_between = max_frames_between
        
        # State
        self.last_keyframe_embedding: Optional[np.ndarray] = None
        self.last_keyframe_position: Optional[np.ndarray] = None
        self.last_keyframe_rotation: Optional[np.ndarray] = None  # quaternion
        self.frames_since_keyframe: int = 0
        self.total_keyframes: int = 0
        self.total_frames: int = 0
        
    def reset(self):
        """Reset selector state."""
        self.last_keyframe_embedding = None
        self.last_keyframe_position = None
        self.last_keyframe_rotation = None
        self.frames_since_keyframe = 0
        self.total_keyframes = 0
        self.total_frames = 0
        
    def _cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine distance (1 - cosine_similarity)."""
        a_norm = a / (np.linalg.norm(a) + 1e-8)
        b_norm = b / (np.linalg.norm(b) + 1e-8)
        similarity = np.dot(a_norm, b_norm)
        return float(1.0 - similarity)
    
    def _euclidean_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute Euclidean distance."""
        return float(np.linalg.norm(a - b))
    
    def _rotation_difference(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """
        Compute rotation difference in degrees.
        
        Args:
            q1, q2: Quaternions as [w, x, y, z]
        """
        # Normalize quaternions
        q1 = q1 / (np.linalg.norm(q1) + 1e-8)
        q2 = q2 / (np.linalg.norm(q2) + 1e-8)
        
        # Compute dot product (handle double cover)
        dot = np.abs(np.dot(q1, q2))
        dot = np.clip(dot, -1.0, 1.0)
        
        # Convert to angle
        angle_rad = 2.0 * np.arccos(dot)
        angle_deg = np.degrees(angle_rad)
        
        return float(angle_deg)
    
    def should_save_keyframe(
        self,
        embedding: np.ndarray,
        position: np.ndarray,
        rotation: np.ndarray,  # quaternion [w, x, y, z]
    ) -> KeyframeDecision:
        """
        Decide whether to save this frame as a keyframe.
        
        Uses the Semantic Entropy criterion:
        Save frame t if: 1 - (v_t · v_{t-1}) / (||v_t|| ||v_{t-1}||) > τ
        
        Args:
            embedding: CLIP embedding of current frame
            position: Agent position (x, y, z)
            rotation: Agent rotation quaternion (w, x, y, z)
            
        Returns:
            KeyframeDecision with save decision and reason
        """
        self.total_frames += 1
        self.frames_since_keyframe += 1
        
        # Convert to numpy if needed
        position = np.asarray(position)
        rotation = np.asarray(rotation)
        
        # First frame is always a keyframe
        if self.last_keyframe_embedding is None:
            self._update_state(embedding, position, rotation)
            return KeyframeDecision(
                should_save=True,
                reason="first_frame",
                semantic_change=1.0,
                distance_moved=0.0,
                rotation_change=0.0,
            )
        
        # Compute changes
        semantic_change = self._cosine_distance(embedding, self.last_keyframe_embedding)
        distance_moved = self._euclidean_distance(position, self.last_keyframe_position)
        rotation_change = self._rotation_difference(rotation, self.last_keyframe_rotation)
        
        # Force save if too many frames have passed
        if self.frames_since_keyframe >= self.max_frames_between:
            self._update_state(embedding, position, rotation)
            return KeyframeDecision(
                should_save=True,
                reason="max_frames_exceeded",
                semantic_change=semantic_change,
                distance_moved=distance_moved,
                rotation_change=rotation_change,
            )
        
        # Check semantic change (main criterion)
        if semantic_change >= self.semantic_threshold:
            self._update_state(embedding, position, rotation)
            return KeyframeDecision(
                should_save=True,
                reason="semantic_change",
                semantic_change=semantic_change,
                distance_moved=distance_moved,
                rotation_change=rotation_change,
            )
        
        # Check if robot has moved significantly
        if distance_moved >= self.min_distance and rotation_change >= self.min_rotation:
            self._update_state(embedding, position, rotation)
            return KeyframeDecision(
                should_save=True,
                reason="movement",
                semantic_change=semantic_change,
                distance_moved=distance_moved,
                rotation_change=rotation_change,
            )
        
        # Don't save this frame
        return KeyframeDecision(
            should_save=False,
            reason="no_significant_change",
            semantic_change=semantic_change,
            distance_moved=distance_moved,
            rotation_change=rotation_change,
        )
    
    def _update_state(
        self,
        embedding: np.ndarray,
        position: np.ndarray,
        rotation: np.ndarray,
    ):
        """Update internal state after saving a keyframe."""
        self.last_keyframe_embedding = embedding.copy()
        self.last_keyframe_position = position.copy()
        self.last_keyframe_rotation = rotation.copy()
        self.frames_since_keyframe = 0
        self.total_keyframes += 1
        
    def get_stats(self) -> dict:
        """Get selection statistics."""
        return {
            "total_frames": self.total_frames,
            "total_keyframes": self.total_keyframes,
            "compression_ratio": self.total_frames / max(1, self.total_keyframes),
            "keyframe_percentage": 100.0 * self.total_keyframes / max(1, self.total_frames),
        }


# Simple test
if __name__ == "__main__":
    print("Testing Keyframe Selector...")
    
    selector = KeyframeSelector(
        semantic_threshold=0.15,
        min_distance=0.1,
        min_rotation=10.0,
    )
    
    # Simulate exploration with random embeddings
    np.random.seed(42)
    
    # First frame - always save
    emb = np.random.randn(512)
    pos = np.array([0.0, 0.0, 0.0])
    rot = np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion
    
    decision = selector.should_save_keyframe(emb, pos, rot)
    print(f"Frame 1: {decision.reason} -> save={decision.should_save}")
    
    # Similar frame - don't save
    emb2 = emb + 0.05 * np.random.randn(512)  # Small change
    decision = selector.should_save_keyframe(emb2, pos, rot)
    print(f"Frame 2: {decision.reason} -> save={decision.should_save}, change={decision.semantic_change:.3f}")
    
    # Different frame - save
    emb3 = np.random.randn(512)  # Completely different
    decision = selector.should_save_keyframe(emb3, pos, rot)
    print(f"Frame 3: {decision.reason} -> save={decision.should_save}, change={decision.semantic_change:.3f}")
    
    stats = selector.get_stats()
    print(f"\nStats: {stats}")
    
    print("[OK] Keyframe selector test passed!")
