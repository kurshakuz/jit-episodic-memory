#!/usr/bin/env python3
"""
Prepare ScanNet scenes for JIT evaluation.

For each scene:
1. Extract RGB-D frames from .sens file (subsampled to ~160 keyframes)
2. Convert to JIT-compatible format:
   - RGB -> 640x480 JPEG
   - Depth -> 640x480 float32 .npy (meters)
   - Poses -> Habitat convention quaternion + position
   - trace.parquet with all metadata
3. Build CLIP-FAISS index
4. Save per-scene intrinsics config

Usage:
    python scannet/prepare_scenes.py                        # All 20 scenes
    python scannet/prepare_scenes.py --scenes scene0568_00  # Single scene
    python scannet/prepare_scenes.py --skip-index           # Skip FAISS indexing
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent.parent))
from scannet.config import (
    SCANNET_VAL_SCENES, SCANNET_SCANS, SCANNET_JIT,
    TARGET_NUM_KEYFRAMES, TARGET_WIDTH, TARGET_HEIGHT,
    MIN_VALID_DEPTH_FRAC, FAISS_INDEX_TYPE,
)
from scannet.sens_reader import SensReader


def cam_to_world_to_habitat(cam_to_world: np.ndarray, sensor_height: float = 1.5):
    """
    Convert a ScanNet 4x4 camera-to-world matrix to Habitat convention.
    
    ScanNet: OpenCV camera (X-right, Y-down, Z-forward)
    Habitat: (X-right, Y-up, -Z-forward)
    
    The conversion: R_habitat = R_scannet @ diag(1, -1, -1)
    This ensures depth back-projection in Habitat convention produces
    the correct world coordinates.
    
    Returns:
        position: [x, y, z] agent position (camera pos minus sensor offset)
        quaternion: [w, x, y, z] Habitat convention rotation
    """
    R_scannet = cam_to_world[:3, :3]
    t_world = cam_to_world[:3, 3]

    # Convert OpenCV camera -> Habitat camera convention
    # Flip Y and Z axes of the camera frame
    R_cv_to_hab = np.diag([1.0, -1.0, -1.0])
    R_habitat = R_scannet @ R_cv_to_hab

    # Convert rotation matrix to quaternion [x, y, z, w] (scipy convention)
    rot = Rotation.from_matrix(R_habitat)
    quat_xyzw = rot.as_quat()  # [x, y, z, w]
    # Convert to Habitat [w, x, y, z]
    qw, qx, qy, qz = quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]

    # Agent position = camera position minus sensor height offset
    # Habitat adds sensor_height to agent Y to get camera position,
    # so we subtract it here to store the "agent" position.
    agent_pos = t_world.copy()
    agent_pos[1] -= sensor_height

    return agent_pos, np.array([qw, qx, qy, qz])


def select_keyframes_by_stride(reader: SensReader, target_count: int):
    """
    Select frame indices using uniform stride, filtering invalid poses.
    First pass: read all poses (fast, no image decoding).
    """
    print("  Reading all poses (fast pass)...")
    poses = reader.read_all_poses()

    # Filter valid poses
    valid = [(idx, pose, ts) for idx, pose, ts in poses
             if SensReader._is_valid_pose(pose)]
    print(f"  Valid poses: {len(valid)}/{len(poses)}")

    if len(valid) <= target_count:
        return [v[0] for v in valid]

    # Select uniformly with stride
    stride = max(1, len(valid) // target_count)
    selected = [valid[i][0] for i in range(0, len(valid), stride)]

    # Trim to target
    if len(selected) > target_count:
        selected = selected[:target_count]

    return selected


def compute_scaled_intrinsics(header, target_w: int, target_h: int):
    """
    Compute depth intrinsics scaled to target resolution.
    Returns (fx, fy, cx, cy) and equivalent HFOV.
    """
    K = header.intrinsic_depth
    fx_orig, fy_orig = K[0, 0], K[1, 1]
    cx_orig, cy_orig = K[0, 2], K[1, 2]

    # Scale factors
    sx = target_w / header.depth_width
    sy = target_h / header.depth_height

    fx = fx_orig * sx
    fy = fy_orig * sy
    cx = cx_orig * sx
    cy = cy_orig * sy

    # Equivalent HFOV
    hfov = 2 * np.degrees(np.arctan(target_w / (2 * fx)))

    return fx, fy, cx, cy, hfov


def prepare_scene(scene_id: str, skip_index: bool = False,
                   target_frames: int = None, output_base: Path = None):
    """Process a single ScanNet scene into JIT format."""
    sens_path = SCANNET_SCANS / scene_id / f"{scene_id}.sens"
    if not sens_path.exists():
        print(f"  ERROR: .sens file not found: {sens_path}")
        return None

    # Use override target frame count if provided
    num_keyframes = target_frames if target_frames else TARGET_NUM_KEYFRAMES
    # Use override output base if provided
    base_dir = output_base if output_base else SCANNET_JIT

    # Output directories
    scene_out = base_dir / scene_id
    explore_dir = scene_out / "exploration"
    rgb_dir = explore_dir / "rgb"
    depth_dir = explore_dir / "depth"
    for d in [scene_out, explore_dir, rgb_dir, depth_dir]:
        os.makedirs(str(d), exist_ok=True)

    # Check if already processed
    trace_path = explore_dir / "trace.parquet"
    if trace_path.exists() and not skip_index:
        index_path = explore_dir / "memory.index"
        if index_path.exists():
            print(f"  Already processed (trace + index exist), skipping")
            return scene_out
        print(f"  Trace exists but index missing, will rebuild index")

    # Initialize reader
    reader = SensReader(str(sens_path))
    header = reader.read_header()
    print(f"  .sens: {header.num_frames} frames, "
          f"color {header.color_width}x{header.color_height}, "
          f"depth {header.depth_width}x{header.depth_height}, "
          f"depth_shift={header.depth_shift}")

    # Compute scaled intrinsics
    fx, fy, cx, cy, hfov = compute_scaled_intrinsics(header, TARGET_WIDTH, TARGET_HEIGHT)
    print(f"  Intrinsics (scaled to {TARGET_WIDTH}x{TARGET_HEIGHT}): "
          f"fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}, HFOV={hfov:.1f}°")

    # Save intrinsics config
    intrinsics_config = {
        "original_depth_width": int(header.depth_width),
        "original_depth_height": int(header.depth_height),
        "original_color_width": int(header.color_width),
        "original_color_height": int(header.color_height),
        "target_width": TARGET_WIDTH,
        "target_height": TARGET_HEIGHT,
        "depth_shift": float(header.depth_shift),
        "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
        "hfov": float(hfov),
        "sensor_height": 1.5,  # Convention matching Habitat
        "intrinsic_depth_original": header.intrinsic_depth.astype(float).tolist(),
        "intrinsic_color_original": header.intrinsic_color.astype(float).tolist(),
    }
    with open(scene_out / "intrinsics.json", 'w') as f:
        json.dump(intrinsics_config, f, indent=2)

    # Select keyframes
    if not trace_path.exists():
        target_indices = select_keyframes_by_stride(reader, num_keyframes)
        print(f"  Selected {len(target_indices)} keyframes (target={num_keyframes})")

        # Extract frames
        print(f"  Extracting frames...")
        trace_rows = []
        frame_count = 0
        t0 = time.time()

        for frame in reader.extract_frames(target_indices=target_indices):
            # Resize color to target resolution
            color_img = Image.fromarray(frame.color)
            color_resized = color_img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS)
            color_np = np.array(color_resized)

            # Resize depth to target resolution (nearest-neighbor to preserve values)
            depth_raw = frame.depth  # uint16
            if depth_raw.shape != (TARGET_HEIGHT, TARGET_WIDTH):
                depth_img = Image.fromarray(depth_raw)
                depth_resized = depth_img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.NEAREST)
                depth_raw = np.array(depth_resized)

            # Convert depth to meters
            depth_meters = depth_raw.astype(np.float32) / header.depth_shift
            # Set zero/invalid depth to 0
            depth_meters[depth_raw == 0] = 0.0

            # Check valid depth fraction
            valid_frac = np.count_nonzero(depth_meters > 0.1) / depth_meters.size
            if valid_frac < MIN_VALID_DEPTH_FRAC:
                continue

            # Save RGB
            rgb_filename = f"{frame_count:04d}.jpg"
            rgb_path = rgb_dir / rgb_filename
            Image.fromarray(color_np).save(str(rgb_path), quality=90)

            # Save depth
            depth_filename = f"{frame_count:04d}.npy"
            depth_path = depth_dir / depth_filename
            np.save(str(depth_path), depth_meters)

            # Convert pose to Habitat convention
            agent_pos, quat = cam_to_world_to_habitat(frame.camera_to_world)

            trace_rows.append({
                "frame_id": frame_count,
                "timestamp": float(frame.timestamp_color) / 1e9,  # ns -> s
                "x": float(agent_pos[0]),
                "y": float(agent_pos[1]),
                "z": float(agent_pos[2]),
                "qw": float(quat[0]),
                "qx": float(quat[1]),
                "qy": float(quat[2]),
                "qz": float(quat[3]),
                "image_path": f"rgb/{rgb_filename}",
                "depth_path": f"depth/{depth_filename}",
                "scannet_frame_idx": frame.index,
            })
            frame_count += 1

        elapsed = time.time() - t0
        print(f"  Extracted {frame_count} keyframes in {elapsed:.1f}s")

        # Save trace
        df = pd.DataFrame(trace_rows)
        df.to_parquet(str(trace_path), index=False)
        print(f"  Saved trace to {trace_path}")
    else:
        df = pd.read_parquet(str(trace_path))
        frame_count = len(df)
        print(f"  Loaded existing trace: {frame_count} keyframes")

    # Build CLIP-FAISS index
    if not skip_index:
        index_path = explore_dir / "memory.index"
        meta_path = explore_dir / "memory_meta.json"
        if not index_path.exists():
            print(f"  Building CLIP-FAISS index...")
            try:
                from ingestion.clip_encoder import CLIPEncoder
                from ingestion.faiss_indexer import FAISSIndexer

                encoder = CLIPEncoder()
                indexer = FAISSIndexer(index_type=FAISS_INDEX_TYPE)

                embeddings = []
                frame_ids = []
                for _, row in df.iterrows():
                    img_path = explore_dir / row["image_path"]
                    img = np.array(Image.open(str(img_path)))
                    emb = encoder.encode_image(img)
                    embeddings.append(emb)
                    frame_ids.append(int(row["frame_id"]))

                emb_matrix = np.vstack(embeddings).astype(np.float32)
                indexer.build_index(emb_matrix, frame_ids)
                indexer.save(str(explore_dir / "memory"))
                print(f"  FAISS index saved ({len(frame_ids)} vectors)")
            except Exception as e:
                print(f"  WARNING: FAISS indexing failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  FAISS index already exists")

    return scene_out


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prepare ScanNet scenes for JIT")
    parser.add_argument("--scenes", type=str, default=None,
                        help="Comma-separated scene IDs")
    parser.add_argument("--skip-index", action="store_true",
                        help="Skip FAISS index building")
    parser.add_argument("--target-frames", type=int, default=None,
                        help="Override TARGET_NUM_KEYFRAMES (e.g. 500, 2500)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output base directory")
    args = parser.parse_args()

    scenes = SCANNET_VAL_SCENES
    if args.scenes:
        scenes = [s.strip() for s in args.scenes.split(",")]

    target_frames = args.target_frames
    output_base = Path(args.output_dir) if args.output_dir else None
    if output_base:
        os.makedirs(str(output_base), exist_ok=True)

    actual_target = target_frames if target_frames else TARGET_NUM_KEYFRAMES
    print(f"Preparing {len(scenes)} ScanNet scenes for JIT evaluation")
    print(f"Target: {actual_target} keyframes per scene at {TARGET_WIDTH}x{TARGET_HEIGHT}")
    if output_base:
        print(f"Output dir: {output_base}")
    print()

    results = []
    for i, scene_id in enumerate(scenes):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(scenes)}] {scene_id}")
        print(f"{'='*60}")

        try:
            out_dir = prepare_scene(scene_id, skip_index=args.skip_index,
                                    target_frames=target_frames,
                                    output_base=output_base)
            if out_dir:
                results.append(scene_id)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Prepared {len(results)}/{len(scenes)} scenes")
    print(f"Output: {SCANNET_JIT}")


if __name__ == "__main__":
    main()
