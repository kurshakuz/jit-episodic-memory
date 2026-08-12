#!/usr/bin/env python3
"""
Python 3 ScanNet .sens file reader.
Based on the official SensReader but rewritten for Python 3 compatibility
and streaming frame extraction.

The .sens binary format:
  Header:
    - version (uint32)
    - sensor_name (string with length prefix)
    - intrinsic_color (4x4 float32)
    - extrinsic_color (4x4 float32)
    - intrinsic_depth (4x4 float32)
    - extrinsic_depth (4x4 float32)
    - color_compression_type (int32)
    - depth_compression_type (int32)
    - color_width, color_height (uint32)
    - depth_width, depth_height (uint32)
    - depth_shift (float32)
    - num_frames (uint64)
  Per frame:
    - camera_to_world (4x4 float32)
    - timestamp_color (uint64)
    - timestamp_depth (uint64)
    - color_size_bytes (uint64)
    - color_data (bytes)
    - depth_size_bytes (uint64)
    - depth_data (bytes)
"""

import struct
import numpy as np
import zlib
import io
from dataclasses import dataclass
from typing import Optional, Tuple
from PIL import Image


COMPRESSION_TYPE_COLOR = {-1: 'unknown', 0: 'raw', 1: 'png', 2: 'jpeg'}
COMPRESSION_TYPE_DEPTH = {-1: 'unknown', 0: 'raw_ushort', 1: 'zlib_ushort', 2: 'occi_ushort'}


@dataclass
class SensHeader:
    """Header information from a .sens file."""
    version: int
    sensor_name: str
    intrinsic_color: np.ndarray    # 4x4
    extrinsic_color: np.ndarray    # 4x4
    intrinsic_depth: np.ndarray    # 4x4
    extrinsic_depth: np.ndarray    # 4x4
    color_compression: str
    depth_compression: str
    color_width: int
    color_height: int
    depth_width: int
    depth_height: int
    depth_shift: float             # Divide raw uint16 by this to get meters
    num_frames: int


@dataclass
class SensFrame:
    """A single RGB-D frame from a .sens file."""
    index: int
    camera_to_world: np.ndarray    # 4x4 float32
    timestamp_color: int
    timestamp_depth: int
    color: Optional[np.ndarray] = None     # H x W x 3 uint8
    depth: Optional[np.ndarray] = None     # H x W uint16 (raw, divide by depth_shift for meters)


class SensReader:
    """Streaming reader for ScanNet .sens files (Python 3)."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.header: Optional[SensHeader] = None

    def read_header(self) -> SensHeader:
        """Read only the header (fast, no frame data)."""
        with open(self.filepath, 'rb') as f:
            self.header = self._parse_header(f)
        return self.header

    def _parse_header(self, f) -> SensHeader:
        version = struct.unpack('I', f.read(4))[0]
        assert version == 4, f"Expected .sens version 4, got {version}"

        strlen = struct.unpack('Q', f.read(8))[0]
        sensor_name = f.read(strlen).decode('utf-8', errors='replace')

        intrinsic_color = np.frombuffer(f.read(16 * 4), dtype=np.float32).reshape(4, 4).copy()
        extrinsic_color = np.frombuffer(f.read(16 * 4), dtype=np.float32).reshape(4, 4).copy()
        intrinsic_depth = np.frombuffer(f.read(16 * 4), dtype=np.float32).reshape(4, 4).copy()
        extrinsic_depth = np.frombuffer(f.read(16 * 4), dtype=np.float32).reshape(4, 4).copy()

        color_compression = COMPRESSION_TYPE_COLOR[struct.unpack('i', f.read(4))[0]]
        depth_compression = COMPRESSION_TYPE_DEPTH[struct.unpack('i', f.read(4))[0]]

        color_width = struct.unpack('I', f.read(4))[0]
        color_height = struct.unpack('I', f.read(4))[0]
        depth_width = struct.unpack('I', f.read(4))[0]
        depth_height = struct.unpack('I', f.read(4))[0]
        depth_shift = struct.unpack('f', f.read(4))[0]
        num_frames = struct.unpack('Q', f.read(8))[0]

        return SensHeader(
            version=version,
            sensor_name=sensor_name,
            intrinsic_color=intrinsic_color,
            extrinsic_color=extrinsic_color,
            intrinsic_depth=intrinsic_depth,
            extrinsic_depth=extrinsic_depth,
            color_compression=color_compression,
            depth_compression=depth_compression,
            color_width=color_width,
            color_height=color_height,
            depth_width=depth_width,
            depth_height=depth_height,
            depth_shift=depth_shift,
            num_frames=num_frames,
        )

    def _read_frame(self, f, index: int, decode: bool = True) -> SensFrame:
        """Read a single frame from the current file position."""
        cam_to_world = np.frombuffer(f.read(16 * 4), dtype=np.float32).reshape(4, 4).copy()
        timestamp_color = struct.unpack('Q', f.read(8))[0]
        timestamp_depth = struct.unpack('Q', f.read(8))[0]

        # Both sizes are stored consecutively BEFORE the data
        color_size = struct.unpack('Q', f.read(8))[0]
        depth_size = struct.unpack('Q', f.read(8))[0]
        color_data = f.read(color_size)
        depth_data = f.read(depth_size)

        frame = SensFrame(
            index=index,
            camera_to_world=cam_to_world,
            timestamp_color=timestamp_color,
            timestamp_depth=timestamp_depth,
        )

        if decode:
            # Decode color
            if self.header.color_compression == 'jpeg':
                frame.color = np.array(Image.open(io.BytesIO(color_data)))
            elif self.header.color_compression == 'png':
                frame.color = np.array(Image.open(io.BytesIO(color_data)))
            else:
                frame.color = np.frombuffer(color_data, dtype=np.uint8).reshape(
                    self.header.color_height, self.header.color_width, 3)

            # Decode depth
            if self.header.depth_compression == 'zlib_ushort':
                raw = zlib.decompress(depth_data)
                frame.depth = np.frombuffer(raw, dtype=np.uint16).reshape(
                    self.header.depth_height, self.header.depth_width).copy()
            elif self.header.depth_compression == 'raw_ushort':
                frame.depth = np.frombuffer(depth_data, dtype=np.uint16).reshape(
                    self.header.depth_height, self.header.depth_width).copy()
            else:
                raise ValueError(f"Unsupported depth compression: {self.header.depth_compression}")

        return frame

    def _skip_frame(self, f):
        """Skip a frame without decoding (fast seek)."""
        f.read(16 * 4)  # cam_to_world
        f.read(8)        # timestamp_color
        f.read(8)        # timestamp_depth
        color_size = struct.unpack('Q', f.read(8))[0]
        depth_size = struct.unpack('Q', f.read(8))[0]
        f.read(color_size)
        f.read(depth_size)

    def _read_frame_pose_only(self, f, index: int) -> SensFrame:
        """Read only the pose from a frame (skip image data)."""
        cam_to_world = np.frombuffer(f.read(16 * 4), dtype=np.float32).reshape(4, 4).copy()
        timestamp_color = struct.unpack('Q', f.read(8))[0]
        timestamp_depth = struct.unpack('Q', f.read(8))[0]
        color_size = struct.unpack('Q', f.read(8))[0]
        depth_size = struct.unpack('Q', f.read(8))[0]
        f.read(color_size)
        f.read(depth_size)
        return SensFrame(
            index=index,
            camera_to_world=cam_to_world,
            timestamp_color=timestamp_color,
            timestamp_depth=timestamp_depth,
        )

    def extract_frames(self, target_indices=None, stride=None, max_frames=None):
        """
        Generator that yields selected frames.
        
        Args:
            target_indices: Specific frame indices to extract (set or list)
            stride: Extract every Nth frame
            max_frames: Maximum number of frames to extract
        
        Yields:
            SensFrame objects with decoded color and depth
        """
        with open(self.filepath, 'rb') as f:
            self.header = self._parse_header(f)

            if stride is None and target_indices is None:
                # Default: compute stride to get ~max_frames
                if max_frames and self.header.num_frames > max_frames:
                    stride = self.header.num_frames // max_frames
                else:
                    stride = 1

            if target_indices is not None:
                target_set = set(target_indices)
            else:
                target_set = None

            count = 0
            for i in range(self.header.num_frames):
                should_extract = False
                if target_set is not None:
                    should_extract = i in target_set
                elif stride is not None:
                    should_extract = (i % stride == 0)

                if should_extract:
                    frame = self._read_frame(f, i, decode=True)
                    # Skip frames with invalid poses
                    if self._is_valid_pose(frame.camera_to_world):
                        yield frame
                        count += 1
                        if max_frames and count >= max_frames:
                            return
                else:
                    self._skip_frame(f)

    def read_all_poses(self):
        """Read all frame poses without decoding images (fast)."""
        poses = []
        with open(self.filepath, 'rb') as f:
            self.header = self._parse_header(f)
            for i in range(self.header.num_frames):
                frame = self._read_frame_pose_only(f, i)
                poses.append((i, frame.camera_to_world, frame.timestamp_color))
        return poses

    @staticmethod
    def _is_valid_pose(cam_to_world: np.ndarray) -> bool:
        """Check if a camera-to-world matrix is valid."""
        if np.any(np.isinf(cam_to_world)) or np.any(np.isnan(cam_to_world)):
            return False
        # Check it's not identity or zero (common for invalid frames)
        if np.allclose(cam_to_world, np.eye(4), atol=1e-6):
            return False
        if np.allclose(cam_to_world, np.zeros((4, 4)), atol=1e-6):
            return False
        return True

    def get_depth_intrinsics(self) -> Tuple[float, float, float, float]:
        """Return (fx, fy, cx, cy) from depth intrinsic matrix."""
        if self.header is None:
            self.read_header()
        K = self.header.intrinsic_depth
        return K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    def get_color_intrinsics(self) -> Tuple[float, float, float, float]:
        """Return (fx, fy, cx, cy) from color intrinsic matrix."""
        if self.header is None:
            self.read_header()
        K = self.header.intrinsic_color
        return K[0, 0], K[1, 1], K[0, 2], K[1, 2]


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python sens_reader.py <path_to.sens>")
        sys.exit(1)

    reader = SensReader(sys.argv[1])
    header = reader.read_header()
    print(f"Sensor: {header.sensor_name}")
    print(f"Color: {header.color_width}x{header.color_height} ({header.color_compression})")
    print(f"Depth: {header.depth_width}x{header.depth_height} ({header.depth_compression})")
    print(f"Depth shift: {header.depth_shift}")
    print(f"Frames: {header.num_frames}")
    print(f"\nColor intrinsics:\n{header.intrinsic_color}")
    print(f"\nDepth intrinsics:\n{header.intrinsic_depth}")
