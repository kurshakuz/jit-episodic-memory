#!/usr/bin/env python3
"""
Custom GradSLAMDataset for our JIT scene format (trace.parquet + jpg + npy).

Used by the official-ConceptFusion adapter to feed our data into
gradslam's PointFusion pipeline.

Data format expected in scene_dir:
  exploration/
    trace.parquet           (frame_id, x, y, z, qw, qx, qy, qz, image_path, depth_path)
    images/                 or  rgb/    (.jpg files)
    depth/                  (.npy files, float meters)
  intrinsics.json           (optional, for ScanNet per-scene intrinsics)

For HM3D we default to HFOV=90°, 640x480; for ScanNet we read intrinsics.json.
"""

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from gradslam.datasets.basedataset import GradSLAMDataset
from PIL import Image


def quaternion_wxyz_to_matrix(qw, qx, qy, qz):
    """Convert Habitat quaternion [w, x, y, z] to 3x3 rotation matrix."""
    R = np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,     1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float64)
    return R


def load_scene_intrinsics(scene_dir: Path, default_hfov_deg=90.0,
                           default_h=480, default_w=640, default_sensor_height=1.5):
    """Load ScanNet-style intrinsics.json, else return Habitat defaults."""
    intr_path = scene_dir / "intrinsics.json"
    if intr_path.exists():
        with open(intr_path) as f:
            d = json.load(f)
        return {
            "fx": float(d["fx"]),
            "fy": float(d["fy"]),
            "cx": float(d["cx"]),
            "cy": float(d["cy"]),
            "height": int(d.get("target_height", default_h)),
            "width": int(d.get("target_width", default_w)),
            "sensor_height": float(d.get("sensor_height", default_sensor_height)),
            "png_depth_scale": 1.0,  # our npy depths are already float meters
        }
    # Habitat defaults (HM3D)
    fx = default_w / (2.0 * np.tan(np.deg2rad(default_hfov_deg) / 2.0))
    return {
        "fx": fx, "fy": fx,
        "cx": default_w / 2.0, "cy": default_h / 2.0,
        "height": default_h, "width": default_w,
        "sensor_height": default_sensor_height,
        "png_depth_scale": 1.0,
    }


class JITSceneDataset(GradSLAMDataset):
    """
    Loads a single scene's frames from trace.parquet and yields the tuple
    expected by gradslam: (color, depth, intrinsics, pose [, embedding]).

    Key conversions:
      - Pose: Habitat world (+Y up) + sensor_height offset -> OpenCV/gradslam convention
      - Depth: .npy float meters -> torch tensor in meters
      - Color: .jpg RGB -> tensor (H, W, 3)
    """

    def __init__(
        self,
        scene_dir: Path,
        desired_height: int = 480,
        desired_width: int = 640,
        stride: int = 1,
        start: int = 0,
        end: int = -1,
        frame_ids: Optional[List[int]] = None,
        load_embeddings: bool = False,
        embedding_dir: str = "conceptfusion_feat",
        embedding_dim: int = 1024,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float,
        channels_first: bool = False,
        normalize_color: bool = False,
    ):
        self.scene_dir = Path(scene_dir)
        self.trace = pd.read_parquet(self.scene_dir / "exploration" / "trace.parquet")

        if frame_ids is None:
            frame_ids = list(range(0, len(self.trace), stride))
        else:
            frame_ids = list(frame_ids)
        self.frame_ids = frame_ids

        self._intr = load_scene_intrinsics(self.scene_dir)

        # Fake config_dict so parent __init__ works
        config_dict = {
            "dataset_name": "jit",
            "camera_params": {
                "image_height": self._intr["height"],
                "image_width": self._intr["width"],
                "fx": self._intr["fx"],
                "fy": self._intr["fy"],
                "cx": self._intr["cx"],
                "cy": self._intr["cy"],
                "png_depth_scale": self._intr["png_depth_scale"],
            },
        }

        # Parent __init__ will call self.get_filepaths() and self.load_poses()
        super().__init__(
            config_dict,
            stride=1,  # we already subsampled via frame_ids
            start=0,
            end=-1,
            desired_height=desired_height,
            desired_width=desired_width,
            channels_first=channels_first,
            normalize_color=normalize_color,
            device=device,
            dtype=dtype,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
        )

    # ---- required abstract methods ----
    def get_filepaths(self):
        color_paths, depth_paths, embedding_paths = [], [], []
        for fid in self.frame_ids:
            row = self.trace.iloc[fid]
            cp = self.scene_dir / "exploration" / row["image_path"]
            dp = self.scene_dir / "exploration" / row["depth_path"]
            color_paths.append(str(cp))
            depth_paths.append(str(dp))
            if self.load_embeddings if False else False:
                # populated below if needed
                pass
        if self.load_embeddings:
            for fid in self.frame_ids:
                ep = self.scene_dir / "exploration" / self.embedding_dir / f"frame_{fid:06d}.pt"
                embedding_paths.append(str(ep))
        return color_paths, depth_paths, embedding_paths if self.load_embeddings else None

    def _preprocess_poses(self, poses: torch.Tensor):
        """Keep poses in the WORLD frame (don't make them relative to frame 0).

        gradslam's default _preprocess_poses sets pose[0] = I and makes all
        others relative. We need world-frame poses so 3D points land in the
        same frame as ground-truth object centers.
        """
        return poses

    def load_poses(self):
        poses = []
        sensor_h = self._intr["sensor_height"]
        for fid in self.frame_ids:
            row = self.trace.iloc[fid]
            R = quaternion_wxyz_to_matrix(row["qw"], row["qx"], row["qy"], row["qz"])
            t = np.array([row["x"], row["y"] + sensor_h, row["z"]], dtype=np.float64)

            # Habitat camera convention: -Y is up, -Z is forward in camera frame.
            # gradslam uses OpenCV camera convention (Y down, Z forward).
            # Conversion: flip Y and Z axes of the camera frame.
            P = np.array([[1, 0, 0, 0],
                          [0, -1, 0, 0],
                          [0, 0, -1, 0],
                          [0, 0, 0, 1]], dtype=np.float64)

            c2w_habitat = np.eye(4, dtype=np.float64)
            c2w_habitat[:3, :3] = R
            c2w_habitat[:3, 3] = t
            c2w_opencv = c2w_habitat @ P
            poses.append(torch.from_numpy(c2w_opencv))
        return poses

    # ---- override depth loading (.npy, float meters) ----
    def _load_depth(self, depth_path: str) -> np.ndarray:
        d = np.load(depth_path).astype(np.float32)
        # basedataset expects (H, W) or (H, W, 1); we get (H, W)
        return d

    def __getitem__(self, index):
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]

        color = np.array(Image.open(color_path))
        if color.ndim == 3 and color.shape[-1] == 4:
            color = color[:, :, :3]
        color = self._preprocess_color(color)  # resizes to desired_h/w
        color = torch.from_numpy(color)

        depth = self._load_depth(depth_path)
        # resize to desired dims manually (bypass _preprocess_depth that assumes png scaling)
        import cv2
        depth = cv2.resize(depth, (self.desired_width, self.desired_height),
                           interpolation=cv2.INTER_NEAREST)
        depth = depth[..., None]  # (H, W, 1)
        depth = torch.from_numpy(depth)

        # Build 4x4 intrinsics (scaled to desired size)
        K = np.eye(4, dtype=np.float32)
        K[0, 0] = self._intr["fx"] * self.width_downsample_ratio
        K[1, 1] = self._intr["fy"] * self.height_downsample_ratio
        K[0, 2] = self._intr["cx"] * self.width_downsample_ratio
        K[1, 2] = self._intr["cy"] * self.height_downsample_ratio
        intrinsics = torch.from_numpy(K)

        pose = self.transformed_poses[index]

        if self.load_embeddings:
            emb = self.read_embedding_from_file(self.embedding_paths[index])
            return (
                color.to(self.device).type(self.dtype),
                depth.to(self.device).type(self.dtype),
                intrinsics.to(self.device).type(self.dtype),
                pose.to(self.device).type(self.dtype),
                emb.to(self.device),
            )

        return (
            color.to(self.device).type(self.dtype),
            depth.to(self.device).type(self.dtype),
            intrinsics.to(self.device).type(self.dtype),
            pose.to(self.device).type(self.dtype),
        )

    def read_embedding_from_file(self, path: str) -> torch.Tensor:
        e = torch.load(path)
        if e.dtype == torch.float16:
            e = e.float()
        return e
