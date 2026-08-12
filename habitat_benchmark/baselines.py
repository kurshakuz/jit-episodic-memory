"""
Baselines for ObjectNav Evaluation
==================================

Implementations of different goal prediction methods:

1. PoseOnlyBaseline: Use camera pose from best CLIP frame (no depth)
2. L1OwlDepthBaseline: L1 CLIP + OWL-ViT + Depth projection (no DBSCAN)
3. BruteForceBaseline: Run OWL-ViT on all frames, pick best
4. JITCascadeBaseline: Full L1->L2->L3 pipeline with depth
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def project_bbox_to_world(
    box: np.ndarray,
    depth: np.ndarray,
    trace_row: pd.Series,
    image_size: Tuple[int, int] = (640, 480),
    hfov: float = 90.0,
    sensor_height: float = 1.5,
) -> Optional[np.ndarray]:
    """
    Project detection box to 3D world coordinates.

    Uses proper camera-to-world transformation with:
    - Pinhole camera model
    - Quaternion rotation
    - Sensor height offset

    Args:
        box: [x1, y1, x2, y2] bounding box in pixel coords
        depth: HxW depth image
        trace_row: Row from trace.parquet with pose info
        image_size: (width, height) of original image
        hfov: Horizontal field of view in degrees
        sensor_height: Height of sensor above agent base (default 1.5m)

    Returns:
        3D position in world coordinates or None if invalid
    """
    H_img, W_img = depth.shape

    # Get bbox bounds
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    x1, x2 = max(0, x1), min(W_img, x2)
    y1, y2 = max(0, y1), min(H_img, y2)

    if x1 >= x2 or y1 >= y2:
        return None

    # Get center of bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # Sample depth in bbox region (use 30th percentile for foreground)
    bbox_region = depth[y1:y2, x1:x2]
    valid_depths = bbox_region[(bbox_region > 0.1) & (bbox_region < 10.0)]

    if len(valid_depths) == 0:
        return None

    z = float(np.percentile(valid_depths, 30))

    # Camera intrinsics
    W, H = image_size
    fx = W / (2 * np.tan(np.radians(hfov) / 2))
    fy = fx  # Square pixels
    cx_cam, cy_cam = W / 2, H / 2

    # Deproject to camera frame (Habitat convention)
    # X: right, Y: up, Z: backward
    x_cam = (cx - cx_cam) * z / fx      # Right is positive
    y_cam = -(cy - cy_cam) * z / fy     # Up is positive (flip v)
    z_cam = -z                           # Forward is negative Z
    point_cam = np.array([x_cam, y_cam, z_cam])

    # Get agent pose
    agent_pos = np.array([trace_row['x'], trace_row['y'], trace_row['z']])

    # Add sensor height offset (camera is above agent base)
    sensor_pos = agent_pos.copy()
    sensor_pos[1] += sensor_height  # Y is up in Habitat

    # Transform to world frame using quaternion rotation
    if 'qw' in trace_row.index:
        qw, qx, qy, qz = trace_row['qw'], trace_row['qx'], trace_row['qy'], trace_row['qz']

        # Quaternion to rotation matrix
        R = np.array([
            [1 - 2*qy*qy - 2*qz*qz,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
            [    2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz,     2*qy*qz - 2*qx*qw],
            [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
        ])

        # Transform: world_point = R @ camera_point + sensor_position
        world_pos = R @ point_cam + sensor_pos
    else:
        # Fallback: no rotation info
        world_pos = sensor_pos + point_cam

    return world_pos


@dataclass
class GoalPrediction:
    """Prediction result from a baseline."""
    goal: Optional[np.ndarray]  # 3D position or None if not found
    success: bool  # Whether object was detected
    score: float  # Detection confidence
    latency_ms: float  # Time taken
    frame_idx: Optional[int] = None  # Which frame was used
    method_info: Optional[Dict] = None  # Additional info


class BaselineMethod(ABC):
    """Abstract base class for goal prediction methods."""

    def __init__(self, name: str):
        self.name = name
        self._models_loaded = False

    @abstractmethod
    def load_models(self):
        """Load required models."""
        pass

    @abstractmethod
    def predict(
        self,
        query: str,
        exploration_dir: Path,
    ) -> GoalPrediction:
        """
        Predict goal location for a query.

        Args:
            query: Object category to find (e.g., "chair")
            exploration_dir: Path to exploration data

        Returns:
            GoalPrediction with goal position and metadata
        """
        pass

    def ensure_models_loaded(self):
        """Ensure models are loaded before prediction."""
        if not self._models_loaded:
            self.load_models()
            self._models_loaded = True


class PoseOnlyBaseline(BaselineMethod):
    """
    Pose-Only Baseline: Use camera pose from best CLIP frame.

    - L1: CLIP retrieval to find best matching frame
    - Goal: Camera position of that frame (no depth projection)
    """

    def __init__(self, k: int = 1):
        super().__init__("pose_only")
        self.k = k
        self.clip_model = None
        self.tokenizer = None

    def load_models(self):
        """Load CLIP model."""
        import torch
        import open_clip

        print(f"Loading CLIP model for {self.name}...")
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            'ViT-B-32-quickgelu', pretrained='openai'
        )
        self.clip_model.eval()
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32-quickgelu')
        self.torch = torch

    def predict(
        self,
        query: str,
        exploration_dir: Path,
    ) -> GoalPrediction:
        self.ensure_models_loaded()

        start_time = time.time()

        # Load exploration data
        embeddings = np.load(exploration_dir / "embeddings.npy").astype('float32')
        trace = pd.read_parquet(exploration_dir / "trace.parquet")

        # Encode query
        import faiss
        with self.torch.no_grad():
            text_tokens = self.tokenizer([query])
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            query_embedding = text_features.numpy().astype('float32')

        # Search
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        scores, indices = index.search(query_embedding, self.k)

        # Get pose of best frame
        best_idx = indices[0][0]
        best_score = float(scores[0][0])

        row = trace.iloc[best_idx]
        goal = np.array([row['x'], row['y'], row['z']])

        latency_ms = (time.time() - start_time) * 1000

        return GoalPrediction(
            goal=goal,
            success=True,  # Always "succeeds" - just returns pose
            score=best_score,
            latency_ms=latency_ms,
            frame_idx=int(best_idx),
            method_info={'clip_score': best_score},
        )


class L1OwlDepthBaseline(BaselineMethod):
    """
    L1 + OWL-ViT + Depth Baseline (no geometric clustering).

    - L1: CLIP retrieval
    - L3: OWL-ViT detection on best frame
    - Depth: Project detection to 3D using depth map
    """

    def __init__(self, k: int = 10, detection_threshold: float = 0.1):
        super().__init__("l1_owl_depth")
        self.k = k
        self.detection_threshold = detection_threshold

    def load_models(self):
        """Load CLIP and OWL-ViT models."""
        import torch
        import open_clip
        from transformers import OwlViTProcessor, OwlViTForObjectDetection
        from PIL import Image

        print(f"Loading models for {self.name}...")

        # CLIP
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            'ViT-B-32-quickgelu', pretrained='openai'
        )
        self.clip_model.eval()
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32-quickgelu')

        # OWL-ViT
        self.owl_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.owl_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        self.owl_model.eval()

        self.torch = torch
        self.Image = Image

    def _run_owl_detection(
        self,
        image_path: Path,
        query: str
    ) -> Tuple[Optional[np.ndarray], float]:
        """Run OWL-ViT detection, return best box and score."""
        img = self.Image.open(image_path).convert('RGB')

        inputs = self.owl_processor(text=[[query]], images=img, return_tensors="pt")
        with self.torch.no_grad():
            outputs = self.owl_model(**inputs)

        target_sizes = self.torch.tensor([img.size[::-1]])
        # Handle API change in newer transformers versions
        if hasattr(self.owl_processor, 'post_process_object_detection'):
            results = self.owl_processor.post_process_object_detection(
                outputs, threshold=self.detection_threshold, target_sizes=target_sizes
            )[0]
        elif hasattr(self.owl_processor, 'post_process_grounded_object_detection'):
            results = self.owl_processor.post_process_grounded_object_detection(
                outputs, threshold=self.detection_threshold, target_sizes=target_sizes
            )[0]
        else:
            # Fallback: use image_processor
            results = self.owl_processor.image_processor.post_process_object_detection(
                outputs, threshold=self.detection_threshold, target_sizes=target_sizes
            )[0]

        if len(results["boxes"]) > 0:
            best_idx = results["scores"].argmax()
            box = results["boxes"][best_idx].numpy()
            score = float(results["scores"][best_idx])
            return box, score

        return None, 0.0

    def _project_to_3d(
        self,
        box: np.ndarray,
        depth_path: Path,
        trace_row: pd.Series,
        image_size: Tuple[int, int] = (640, 480),
        hfov: float = 90.0,
        sensor_height: float = 1.5,
    ) -> Optional[np.ndarray]:
        """
        Project detection box center to 3D using depth.

        Uses proper camera-to-world transformation with:
        - Pinhole camera model
        - Quaternion rotation
        - Sensor height offset
        """
        if not depth_path.exists():
            return None

        depth = np.load(depth_path)
        H_img, W_img = depth.shape

        # Box center in pixel coords
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2

        # Sample depth in bbox region (use 30th percentile for foreground)
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        x1, x2 = max(0, x1), min(W_img, x2)
        y1, y2 = max(0, y1), min(H_img, y2)

        if x1 >= x2 or y1 >= y2:
            return None

        bbox_region = depth[y1:y2, x1:x2]
        valid_depths = bbox_region[(bbox_region > 0.1) & (bbox_region < 10.0)]

        if len(valid_depths) == 0:
            return None

        z = float(np.percentile(valid_depths, 30))

        # Camera intrinsics
        W, H = image_size
        fx = W / (2 * np.tan(np.radians(hfov) / 2))
        fy = fx  # Square pixels
        cx_cam, cy_cam = W / 2, H / 2

        # Deproject to camera frame (Habitat convention)
        # X: right, Y: up, Z: backward
        x_cam = (cx - cx_cam) * z / fx      # Right is positive
        y_cam = -(cy - cy_cam) * z / fy     # Up is positive (flip v)
        z_cam = -z                           # Forward is negative Z
        point_cam = np.array([x_cam, y_cam, z_cam])

        # Get agent pose
        agent_pos = np.array([trace_row['x'], trace_row['y'], trace_row['z']])

        # Add sensor height offset (camera is above agent base)
        sensor_pos = agent_pos.copy()
        sensor_pos[1] += sensor_height  # Y is up in Habitat

        # Transform to world frame using quaternion rotation
        if 'qw' in trace_row.index:
            qw, qx, qy, qz = trace_row['qw'], trace_row['qx'], trace_row['qy'], trace_row['qz']

            # Quaternion to rotation matrix
            R = np.array([
                [1 - 2*qy*qy - 2*qz*qz,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
                [    2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz,     2*qy*qz - 2*qx*qw],
                [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
            ])

            # Transform: world_point = R @ camera_point + sensor_position
            world_pos = R @ point_cam + sensor_pos
        else:
            # Fallback: no rotation info
            world_pos = sensor_pos + point_cam

        return world_pos

    def predict(
        self,
        query: str,
        exploration_dir: Path,
    ) -> GoalPrediction:
        self.ensure_models_loaded()

        start_time = time.time()

        # Load exploration data
        import faiss
        embeddings = np.load(exploration_dir / "embeddings.npy").astype('float32')
        trace = pd.read_parquet(exploration_dir / "trace.parquet")

        # L1: CLIP retrieval
        with self.torch.no_grad():
            text_tokens = self.tokenizer([query])
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            query_embedding = text_features.numpy().astype('float32')

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        scores, indices = index.search(query_embedding, self.k)

        # Try OWL-ViT on top candidates
        best_goal = None
        best_score = 0.0
        best_frame = None

        for idx in indices[0]:
            image_path = exploration_dir / "images" / f"frame_{idx:06d}.jpg"
            if not image_path.exists():
                continue

            box, score = self._run_owl_detection(image_path, query)

            if box is not None and score > best_score:
                # Project to 3D
                depth_path = exploration_dir / "depth" / f"depth_{idx:06d}.npy"
                goal = self._project_to_3d(box, depth_path, trace.iloc[idx])

                if goal is not None:
                    best_goal = goal
                    best_score = score
                    best_frame = int(idx)

        latency_ms = (time.time() - start_time) * 1000

        if best_goal is None:
            # Fallback to pose-only
            best_idx = indices[0][0]
            row = trace.iloc[best_idx]
            best_goal = np.array([row['x'], row['y'], row['z']])
            success = False
        else:
            success = True

        return GoalPrediction(
            goal=best_goal,
            success=success,
            score=best_score,
            latency_ms=latency_ms,
            frame_idx=best_frame,
        )


class BruteForceBaseline(BaselineMethod):
    """
    Brute Force Baseline: Run OWL-ViT on all frames.

    - Run OWL-ViT detection on every frame
    - Pick the frame with highest detection score
    - Project to 3D using depth
    """

    def __init__(self, detection_threshold: float = 0.1, max_frames: int = 200):
        super().__init__("brute_force")
        self.detection_threshold = detection_threshold
        self.max_frames = max_frames

    def load_models(self):
        """Load OWL-ViT model."""
        import torch
        from transformers import OwlViTProcessor, OwlViTForObjectDetection
        from PIL import Image

        print(f"Loading OWL-ViT for {self.name}...")

        self.owl_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.owl_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        self.owl_model.eval()

        self.torch = torch
        self.Image = Image

    def predict(
        self,
        query: str,
        exploration_dir: Path,
    ) -> GoalPrediction:
        self.ensure_models_loaded()

        start_time = time.time()

        trace = pd.read_parquet(exploration_dir / "trace.parquet")
        images_dir = exploration_dir / "images"
        depth_dir = exploration_dir / "depth"

        images = sorted(images_dir.glob("*.jpg"))[:self.max_frames]

        best_goal = None
        best_score = 0.0
        best_frame = None

        for img_path in images:
            frame_idx = int(img_path.stem.replace("frame_", ""))

            img = self.Image.open(img_path).convert('RGB')
            inputs = self.owl_processor(text=[[query]], images=img, return_tensors="pt")

            with self.torch.no_grad():
                outputs = self.owl_model(**inputs)

            target_sizes = self.torch.tensor([img.size[::-1]])
            if hasattr(self.owl_processor, 'post_process_object_detection'):
                results = self.owl_processor.post_process_object_detection(
                    outputs, threshold=self.detection_threshold, target_sizes=target_sizes
                )[0]
            elif hasattr(self.owl_processor, 'post_process_grounded_object_detection'):
                results = self.owl_processor.post_process_grounded_object_detection(
                    outputs, threshold=self.detection_threshold, target_sizes=target_sizes
                )[0]
            else:
                results = self.owl_processor.image_processor.post_process_object_detection(
                    outputs, threshold=self.detection_threshold, target_sizes=target_sizes
                )[0]

            if len(results["boxes"]) > 0:
                score = float(results["scores"].max())

                if score > best_score:
                    box = results["boxes"][results["scores"].argmax()].numpy()

                    # Project to 3D using proper transform
                    depth_path = depth_dir / f"depth_{frame_idx:06d}.npy"
                    if depth_path.exists():
                        depth = np.load(depth_path)
                        row = trace.iloc[frame_idx] if frame_idx < len(trace) else trace.iloc[-1]
                        goal = project_bbox_to_world(box, depth, row)

                        if goal is not None:
                            best_goal = goal
                            best_score = score
                            best_frame = frame_idx

        latency_ms = (time.time() - start_time) * 1000

        if best_goal is None:
            # Fallback to first frame position
            row = trace.iloc[0]
            best_goal = np.array([row['x'], row['y'], row['z']])
            success = False
        else:
            success = True

        return GoalPrediction(
            goal=best_goal,
            success=success,
            score=best_score,
            latency_ms=latency_ms,
            frame_idx=best_frame,
        )


class JITCascadeBaseline(BaselineMethod):
    """
    JIT Cascade: Full L1->L2->L3 pipeline using the actual JITRetrievalCascade.

    Uses the real implementation from retrieval.cascade.
    """

    def __init__(self):
        super().__init__("jit_cascade")
        self._cascade = None
        self._current_trace_dir = None

    def load_models(self):
        """Models are loaded by JITRetrievalCascade."""
        print(f"Loading models for {self.name} (via JITRetrievalCascade)...")
        # Import here to avoid circular imports
        from retrieval.cascade import JITRetrievalCascade
        self.JITRetrievalCascade = JITRetrievalCascade

    def _get_cascade(self, exploration_dir: Path) -> 'JITRetrievalCascade':
        """Get cascade for exploration dir. Reuses same cascade instance and just reloads trace."""
        key = str(exploration_dir)

        if self._cascade is None:
            # First time - create cascade
            self._cascade = self.JITRetrievalCascade(key, lazy_load_models=False)
            self._current_trace_dir = key
        elif self._current_trace_dir != key:
            # Different scene - reload trace
            from ingestion import TraceLoader
            self._cascade.trace_loader = TraceLoader(key)
            self._cascade.trace_dir = key
            # Reinitialize L1, L2, L3 with new trace loader
            self._cascade.l1_filter.trace_loader = self._cascade.trace_loader
            self._cascade.l2_cluster.trace_loader = self._cascade.trace_loader
            self._cascade.l3_verify.trace_loader = self._cascade.trace_loader
            self._current_trace_dir = key

        return self._cascade

    def predict(
        self,
        query: str,
        exploration_dir: Path,
    ) -> GoalPrediction:
        self.ensure_models_loaded()

        start_time = time.time()

        # Use actual JIT cascade
        cascade = self._get_cascade(exploration_dir)
        result = cascade.query(query)

        latency_ms = (time.time() - start_time) * 1000

        if result.success and result.best_location:
            return GoalPrediction(
                goal=np.array(result.best_location.centroid_3d),
                success=True,
                score=result.best_location.best_detection_score,
                latency_ms=latency_ms,
                frame_idx=result.best_location.frame_id,
                method_info={
                    'l1_candidates': result.l1_candidates,
                    'l2_clusters': result.l2_clusters,
                    'l3_verified': result.l3_verified,
                    'timing': {
                        'l1_ms': result.l1_time_ms,
                        'l2_ms': result.l2_time_ms,
                        'l3_ms': result.l3_time_ms,
                    }
                },
            )
        else:
            # Fallback to L2 cluster centroid or agent position
            trace = pd.read_parquet(exploration_dir / "trace.parquet")
            row = trace.iloc[0]
            fallback_goal = np.array([row['x'], row['y'], row['z']])

            return GoalPrediction(
                goal=fallback_goal,
                success=False,
                score=0.0,
                latency_ms=latency_ms,
                frame_idx=None,
                method_info={
                    'l1_candidates': result.l1_candidates,
                    'l2_clusters': result.l2_clusters,
                    'l3_verified': result.l3_verified,
                },
            )


def get_baseline(name: str, **kwargs) -> BaselineMethod:
    """Factory function to get baseline by name."""
    baselines = {
        'pose_only': PoseOnlyBaseline,
        'l1_owl_depth': L1OwlDepthBaseline,
        'brute_force': BruteForceBaseline,
        'jit_cascade': JITCascadeBaseline,
    }

    if name not in baselines:
        raise ValueError(f"Unknown baseline: {name}. Available: {list(baselines.keys())}")

    return baselines[name](**kwargs)


def get_all_baselines() -> List[BaselineMethod]:
    """Get all baseline methods."""
    return [
        PoseOnlyBaseline(),
        L1OwlDepthBaseline(),
        BruteForceBaseline(),
        JITCascadeBaseline(),
    ]
