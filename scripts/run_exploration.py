#!/usr/bin/env python3
"""
Phase 2: Run Exploration
========================

Run an agent through a Habitat scene and record the exploration trace.

This script demonstrates the "Lazy Ingestion" pipeline:
1. Agent explores the scene
2. Each frame gets a lightweight CLIP embedding
3. Keyframe selector decides what to save
4. Trace is stored efficiently for later retrieval

Usage:
    python run_exploration.py  # Uses default config
    python run_exploration.py --scene /path/to/scene.glb --steps 500
"""

import os
import sys
import argparse
import yaml
import numpy as np
from pathlib import Path
import time

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import habitat_sim

from ingestion import MemoryLogger, CLIPEncoder, KeyframeSelector


def load_config(config_path: str = None) -> dict:
    """Load configuration."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def make_sim_config(scene_path: str, dataset_config: str, config: dict) -> habitat_sim.Configuration:
    """Create Habitat simulator configuration."""
    
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.scene_dataset_config_file = dataset_config
    sim_cfg.enable_physics = False  # We don't need physics for exploration
    
    # Agent configuration
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    
    # Get sensor settings
    sensor_cfg = config.get("simulator", {}).get("sensors", {})
    rgb_cfg = sensor_cfg.get("rgb", {"width": 640, "height": 480, "hfov": 90})
    depth_cfg = sensor_cfg.get("depth", {"width": 640, "height": 480, "hfov": 90})
    
    # RGB sensor
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "rgb"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [rgb_cfg["height"], rgb_cfg["width"]]
    rgb_sensor_spec.position = [0.0, 1.5, 0.0]  # 1.5m height (eye level)
    rgb_sensor_spec.hfov = rgb_cfg.get("hfov", 90)
    
    # Depth sensor
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [depth_cfg["height"], depth_cfg["width"]]
    depth_sensor_spec.position = [0.0, 1.5, 0.0]
    depth_sensor_spec.hfov = depth_cfg.get("hfov", 90)
    
    agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec]
    
    # Action space
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=15)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=15)
        ),
        "look_up": habitat_sim.agent.ActionSpec(
            "look_up", habitat_sim.agent.ActuationSpec(amount=10)
        ),
        "look_down": habitat_sim.agent.ActionSpec(
            "look_down", habitat_sim.agent.ActuationSpec(amount=10)
        ),
    }
    
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


class RandomExplorer:
    """Simple random exploration agent."""
    
    def __init__(self, forward_prob: float = 0.7, seed: int = 42):
        """
        Initialize explorer.
        
        Args:
            forward_prob: Probability of moving forward vs turning
            seed: Random seed
        """
        self.forward_prob = forward_prob
        self.rng = np.random.RandomState(seed)
        self.collision_count = 0
        self.last_action = None
        
    def get_action(self, collision: bool = False) -> str:
        """
        Get next action.
        
        Simple strategy:
        - Usually go forward
        - If collision, turn
        - Occasionally turn to explore
        """
        if collision:
            self.collision_count += 1
            # Turn away from obstacle
            return "turn_left" if self.rng.rand() > 0.5 else "turn_right"
            
        if self.rng.rand() < self.forward_prob:
            return "move_forward"
        else:
            # Random turn
            return self.rng.choice(["turn_left", "turn_right", "look_up", "look_down"])


class PanoramicExplorer:
    """Explorer that does panoramic sweeps at each location."""
    
    def __init__(
        self,
        steps_per_position: int = 24,  # 24 * 15° = 360°
        forward_prob: float = 0.8,
        seed: int = 42,
    ):
        """
        Initialize panoramic explorer.
        
        Args:
            steps_per_position: Number of turns to do at each position (360°/15° = 24)
            forward_prob: Probability of moving forward after panorama
            seed: Random seed
        """
        self.steps_per_position = steps_per_position
        self.forward_prob = forward_prob
        self.rng = np.random.RandomState(seed)
        
        self.current_turn_count = 0
        self.in_panorama = False
        
    def get_action(self, collision: bool = False) -> str:
        """Get next action with panoramic behavior."""
        
        if collision:
            # Hit obstacle, turn
            self.in_panorama = False
            self.current_turn_count = 0
            return "turn_left" if self.rng.rand() > 0.5 else "turn_right"
            
        if self.in_panorama:
            self.current_turn_count += 1
            if self.current_turn_count >= self.steps_per_position:
                self.in_panorama = False
                self.current_turn_count = 0
                return "move_forward"
            return "turn_right"
            
        # Decide whether to start panorama or just move
        if self.rng.rand() > self.forward_prob:
            self.in_panorama = True
            self.current_turn_count = 0
            return "turn_right"
            
        return "move_forward"


def run_exploration(
    scene_path: str,
    dataset_config: str,
    output_dir: str,
    num_steps: int = 500,
    explorer_type: str = "random",
    config_path: str = None,
    verbose: bool = True,
):
    """
    Run exploration and record trace.
    
    Args:
        scene_path: Path to scene file
        dataset_config: Path to dataset config file
        output_dir: Where to save the trace
        num_steps: Number of exploration steps
        explorer_type: "random" or "panoramic"
        config_path: Path to config file
        verbose: Print progress
    """
    # Load config
    config = load_config(config_path)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create simulator
    if verbose:
        print(f"Loading scene: {scene_path}")
    
    sim_config = make_sim_config(scene_path, dataset_config, config)
    sim = habitat_sim.Simulator(sim_config)
    
    # Initialize agent at a random navigable point
    agent = sim.get_agent(0)
    agent_state = agent.get_state()
    
    # Try to find navigable position
    navmesh = sim.pathfinder
    if navmesh.is_loaded:
        start_pos = navmesh.get_random_navigable_point()
        agent_state.position = start_pos
        agent.set_state(agent_state)
        if verbose:
            print(f"Starting at navigable position: {start_pos}")
    else:
        if verbose:
            print("No navmesh, using default position")
    
    # Create ingestion components
    ingestion_cfg = config.get("ingestion", {})
    
    clip_encoder = CLIPEncoder(
        model_name=ingestion_cfg.get("clip_model", "ViT-B-32-quickgelu"),
    )

    keyframe_selector = KeyframeSelector(
        semantic_threshold=ingestion_cfg.get("semantic_entropy_threshold", 0.15),
        min_distance=ingestion_cfg.get("min_distance_threshold", 0.1),
        min_rotation=ingestion_cfg.get("min_rotation_threshold", 10),
    )
    
    memory_logger = MemoryLogger(
        output_dir=output_dir,
        clip_encoder=clip_encoder,
        keyframe_selector=keyframe_selector,
    )
    
    # Create explorer
    if explorer_type == "panoramic":
        explorer = PanoramicExplorer()
    else:
        explorer = RandomExplorer()
    
    if verbose:
        print(f"Running {num_steps} exploration steps...")
        print(f"Explorer type: {explorer_type}")
        print()
    
    # Exploration loop
    start_time = time.time()
    keyframes_saved = 0
    collision_count = 0
    
    for step in range(num_steps):
        # Get observations
        observations = sim.get_sensor_observations()
        rgb = observations["rgb"]
        depth = observations["depth"]
        
        # Get agent pose
        agent_state = agent.get_state()
        position = agent_state.position.tolist()
        rotation = [
            agent_state.rotation.w,
            agent_state.rotation.x,
            agent_state.rotation.y,
            agent_state.rotation.z,
        ]
        
        # Log frame
        saved, record = memory_logger.log_frame(
            rgb=rgb,
            depth=depth,
            position=position,
            rotation=rotation,
            frame_id=step,
        )
        
        if saved:
            keyframes_saved += 1
            
        # Get action from explorer
        collision = False
        if step > 0:
            # Check if last action resulted in collision (position unchanged)
            collision = (
                hasattr(explorer, "last_position") 
                and np.allclose(position, explorer.last_position, atol=0.01)
                and explorer.last_action == "move_forward"
            )
            if collision:
                collision_count += 1
                
        explorer.last_position = position
        action = explorer.get_action(collision)
        explorer.last_action = action
        
        # Execute action
        sim.step(action)
        
        # Progress
        if verbose and (step + 1) % 50 == 0:
            elapsed = time.time() - start_time
            fps = (step + 1) / elapsed
            print(
                f"Step {step+1}/{num_steps} | "
                f"Keyframes: {keyframes_saved} ({100*keyframes_saved/(step+1):.1f}%) | "
                f"Collisions: {collision_count} | "
                f"FPS: {fps:.1f}"
            )
    
    # Finalize trace
    if verbose:
        print("\nFinalizing trace...")
        
    stats = memory_logger.finalize()
    
    # Add exploration stats
    stats["explorer_type"] = explorer_type
    stats["collision_count"] = collision_count
    stats["scene_path"] = scene_path
    
    # Cleanup
    sim.close()
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"\nExploration complete in {elapsed:.1f}s")
        print(f"Trace saved to: {output_dir}")
        
    return stats


def main():
    parser = argparse.ArgumentParser(description="Run exploration and record trace")
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Path to scene file (uses config default if not specified)",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Path to scene dataset config file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (uses config default if not specified)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of exploration steps",
    )
    parser.add_argument(
        "--explorer",
        type=str,
        choices=["random", "panoramic"],
        default="random",
        help="Type of explorer",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file",
    )
    
    args = parser.parse_args()
    
    # Load config for defaults
    config = load_config(args.config)
    
    # Get HM3D data path from environment or relative path
    hm3d_base = os.environ.get('HM3D_DATA')
    if hm3d_base is None:
        # Try to find relative to project
        jit_memory_path = Path(__file__).parent.parent.resolve()
        habitat_base = jit_memory_path.parent.parent
        habitat_sim_path = habitat_base / "habitat-sim"
        hm3d_base = habitat_sim_path / "data" / "scene_datasets" / "hm3d" / "example"
    else:
        hm3d_base = Path(hm3d_base) / "example"
    
    # Get scene path
    scene_path = args.scene
    if scene_path is None:
        scene_path = config.get("paths", {}).get("scene_path")
        if scene_path is None:
            # Use the HM3D example scene
            scene_path = str(hm3d_base / "00861-GLAQ4DNUx5U" / "GLAQ4DNUx5U.basis.glb")
    
    # Get dataset config
    dataset_config = args.dataset_config
    if dataset_config is None:
        dataset_config = config.get("paths", {}).get("dataset_config")
        if dataset_config is None:
            dataset_config = str(hm3d_base / "hm3d_annotated_example_basis.scene_dataset_config.json")
            
    # Get output directory
    output_dir = args.output
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "outputs" / "phase2" / "exploration"
        
    # Run exploration
    stats = run_exploration(
        scene_path=scene_path,
        dataset_config=dataset_config,
        output_dir=str(output_dir),
        num_steps=args.steps,
        explorer_type=args.explorer,
        config_path=args.config,
    )
    
    print("\n=== Exploration Summary ===")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
