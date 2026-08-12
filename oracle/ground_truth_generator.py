#!/usr/bin/env python3
"""
Ground Truth Generator
======================

Generates ground-truth object centers (3D) from the Habitat semantic scene graph.
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import habitat_sim
    from habitat_sim.utils.common import quat_from_angle_axis
except ImportError:
    print("habitat_sim not found. Run in habitat conda environment.")
    sys.exit(1)


def find_hm3d_scenes(data_dir: Path) -> List[Tuple[str, str, str]]:
    """Find all HM3D scenes with semantic annotations.
    
    Returns:
        List of (scene_id, scene_path, dataset_config) tuples
    """
    scenes = []
    
    # Check for HM3D scenes in various locations
    search_paths = [
        data_dir / "hm3d",
        data_dir / "hm3d" / "example",
        data_dir / "hm3d" / "train",
        data_dir / "hm3d" / "val",
        data_dir / "hm3d" / "minival",
        data_dir / "scene_datasets" / "hm3d",
        data_dir,
    ]
    
    # Find dataset config
    config_options = [
        data_dir / "hm3d" / "example" / "hm3d_annotated_basis.scene_dataset_config.json",
        data_dir / "hm3d" / "hm3d_annotated_basis.scene_dataset_config.json",
        data_dir / "hm3d_annotated_basis.scene_dataset_config.json",
    ]
    
    dataset_config = None
    for cfg in config_options:
        if cfg.exists():
            dataset_config = str(cfg)
            print(f"Found dataset config: {dataset_config}")
            break
    
    if dataset_config is None:
        print("Warning: No dataset config found, trying direct scene loading")
    
    # Search for .basis.glb files (HM3D format)
    for search_path in search_paths:
        if not search_path.exists():
            continue
        
        for glb_file in search_path.rglob("*.basis.glb"):
            scene_id = glb_file.stem.replace(".basis", "")
            
            # Check for semantic annotations
            semantic_file = glb_file.parent / f"{scene_id}.semantic.glb"
            if semantic_file.exists():
                if (scene_id, str(glb_file), dataset_config) not in scenes:
                    scenes.append((scene_id, str(glb_file), dataset_config))
                    print(f"  Found scene: {scene_id}")
    
    return scenes


def generate_ground_truth_for_scene(
    scene_id: str,
    scene_path: str,
    dataset_config: Optional[str],
    output_dir: Path,
    num_positions: int = 100,
    views_per_position: int = 8,
    width: int = 640,
    height: int = 480,
    hfov: int = 90,
    sensor_height: float = 1.5,
) -> Optional[Dict]:
    """Generate ground truth for a single scene."""
    
    print(f"\n=== Generating Ground Truth for {scene_id} ===")
    print(f"Scene path: {scene_path}")
    
    # Configure simulator
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_path
    if dataset_config:
        backend_cfg.scene_dataset_config_file = dataset_config
    backend_cfg.enable_physics = False
    
    # Sensors
    sensor_specs = []
    
    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "color_sensor"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [height, width]
    rgb_spec.position = [0.0, sensor_height, 0.0]
    rgb_spec.hfov = hfov
    sensor_specs.append(rgb_spec)
    
    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid = "depth_sensor"
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_spec.resolution = [height, width]
    depth_spec.position = [0.0, sensor_height, 0.0]
    depth_spec.hfov = hfov
    sensor_specs.append(depth_spec)
    
    semantic_spec = habitat_sim.CameraSensorSpec()
    semantic_spec.uuid = "semantic_sensor"
    semantic_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_spec.resolution = [height, width]
    semantic_spec.position = [0.0, sensor_height, 0.0]
    semantic_spec.hfov = hfov
    sensor_specs.append(semantic_spec)
    
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    
    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    
    try:
        sim = habitat_sim.Simulator(cfg)
    except Exception as e:
        print(f"Error initializing simulator: {e}")
        return None
    
    # Load semantic objects with proper centers
    scene = sim.semantic_scene
    if scene is None:
        print("No semantic scene available")
        sim.close()
        return None
    
    objects = {}
    categories = defaultdict(list)
    semantic_idx_to_id = {}
    
    print("Loading semantic objects...")
    for idx, obj in enumerate(scene.objects):
        if obj is None:
            continue
        
        category = obj.category.name() if obj.category else "unknown"
        obj_id = str(obj.id) if obj.id else f"{category}_{idx}"
        
        # Get object center from AABB
        center = None
        try:
            if hasattr(obj, 'aabb') and obj.aabb is not None:
                aabb = obj.aabb
                if hasattr(aabb, 'center'):
                    c = aabb.center
                    center = [float(c[0]), float(c[1]), float(c[2])]
                elif hasattr(aabb, 'min') and hasattr(aabb, 'max'):
                    min_pt = np.array(aabb.min)
                    max_pt = np.array(aabb.max)
                    c = (min_pt + max_pt) / 2
                    center = [float(c[0]), float(c[1]), float(c[2])]
        except Exception:
            pass
        
        # Alternative: try obb (oriented bounding box)
        if center is None:
            try:
                if hasattr(obj, 'obb') and obj.obb is not None:
                    c = obj.obb.center
                    center = [float(c[0]), float(c[1]), float(c[2])]
            except Exception:
                pass
        
        objects[obj_id] = {
            "id": obj_id,
            "category": category,
            "center": center,
            "semantic_idx": idx,
        }
        categories[category].append(obj_id)
        semantic_idx_to_id[idx] = obj_id
    
    # Count objects with centers
    with_centers = sum(1 for o in objects.values() if o.get("center") is not None)
    print(f"  Loaded {len(objects)} objects ({with_centers} with centers)")
    
    # Explore scene
    frames = []
    object_to_frames = defaultdict(list)
    
    frame_id = 0
    agent = sim.get_agent(0)
    
    print(f"Exploring scene ({num_positions} positions, {views_per_position} views each)...")
    
    for pos_idx in range(num_positions):
        if not sim.pathfinder.is_loaded:
            break
        
        try:
            start_pos = sim.pathfinder.get_random_navigable_point()
        except Exception:
            continue
        
        for view_idx in range(views_per_position):
            state = agent.get_state()
            state.position = start_pos
            
            angle = (view_idx / views_per_position) * 360
            state.rotation = quat_from_angle_axis(
                np.radians(angle), np.array([0, 1, 0])
            )
            agent.set_state(state)
            
            # Get observations
            obs = sim.get_sensor_observations()
            semantic_obs = obs.get("semantic_sensor")
            
            if semantic_obs is None:
                continue
            
            # Get visible objects
            unique_indices = np.unique(semantic_obs)
            visible_obj_ids = []
            
            for idx in unique_indices:
                idx = int(idx)
                if idx in semantic_idx_to_id:
                    pixel_count = np.sum(semantic_obs == idx)
                    if pixel_count >= 100:
                        visible_obj_ids.append(semantic_idx_to_id[idx])
            
            # Record frame
            position = [float(x) for x in state.position]
            rotation = [
                float(state.rotation.w),
                float(state.rotation.x),
                float(state.rotation.y),
                float(state.rotation.z),
            ]
            
            frames.append({
                "frame_id": frame_id,
                "position": position,
                "rotation": rotation,
                "visible_object_ids": visible_obj_ids,
            })
            
            for obj_id in visible_obj_ids:
                object_to_frames[obj_id].append(frame_id)
            
            frame_id += 1
    
    sim.close()
    
    # For objects without centers, compute from viewing positions
    print("Computing centers for objects without AABB...")
    for obj_id, obj_data in objects.items():
        if obj_data["center"] is None:
            visible_frames = object_to_frames.get(obj_id, [])
            if visible_frames:
                positions = [frames[fid]["position"] for fid in visible_frames if fid < len(frames)]
                if positions:
                    avg_pos = np.mean(positions, axis=0)
                    obj_data["center"] = [float(x) for x in avg_pos]
    
    final_with_centers = sum(1 for o in objects.values() if o.get("center") is not None)
    print(f"  Final: {final_with_centers} objects with centers")
    
    ground_truth = {
        "scene_id": scene_id,
        "scene_path": scene_path,
        "total_frames": len(frames),
        "objects": objects,
        "categories": dict(categories),
        "frames": frames,
        "object_to_frames": dict(object_to_frames),
    }
    
    # Save
    output_file = output_dir / f"{scene_id}_ground_truth.json"
    with open(output_file, 'w') as f:
        json.dump(ground_truth, f, indent=2)
    print(f"Saved to: {output_file}")
    
    return ground_truth


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate ground truth for HM3D scenes")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to data directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--scene-id", type=str, default=None,
                        help="Specific scene ID (optional)")
    parser.add_argument("--positions", type=int, default=100,
                        help="Number of random positions per scene")
    
    args = parser.parse_args()
    
    # Defaults
    base_dir = Path(__file__).parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else Path.home() / ".habitat-sim" / "data"
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "outputs" / "ground_truth"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Ground Truth Generator with Object Centers")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Find scenes
    scenes = find_hm3d_scenes(data_dir)
    print(f"\nFound {len(scenes)} HM3D scenes with semantic annotations")
    
    if args.scene_id:
        scenes = [(s, p, c) for s, p, c in scenes if s == args.scene_id]
    
    if not scenes:
        print("No scenes found!")
        return
    
    # Generate ground truth for each scene
    results = {}
    for scene_id, scene_path, dataset_config in scenes:
        gt = generate_ground_truth_for_scene(
            scene_id=scene_id,
            scene_path=scene_path,
            dataset_config=dataset_config,
            output_dir=output_dir,
            num_positions=args.positions,
        )
        if gt:
            results[scene_id] = {
                "total_frames": gt["total_frames"],
                "total_objects": len(gt["objects"]),
                "objects_with_centers": sum(1 for o in gt["objects"].values() if o.get("center")),
                "categories": len(gt["categories"]),
            }
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Scene':<15} {'Frames':>8} {'Objects':>8} {'W/Centers':>10} {'Categories':>10}")
    print("-" * 55)
    for scene_id, stats in results.items():
        print(f"{scene_id:<15} {stats['total_frames']:>8} {stats['total_objects']:>8} "
              f"{stats['objects_with_centers']:>10} {stats['categories']:>10}")
    
    # Save summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
