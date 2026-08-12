"""
Episode Generator for ObjectNav-style Evaluation
=================================================

Generates evaluation episodes from HM3D scenes:
1. Sample a scene
2. Sample a start position (navigable)
3. Run exploration to build episodic memory
4. Sample target objects from scene's semantic annotations
5. Create episode with (scene, start, memory, target_object, ground_truth_location)
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np


@dataclass
class NavEpisode:
    """A single navigation episode."""
    episode_id: str
    scene_id: str
    scene_path: str
    start_position: np.ndarray
    start_rotation: np.ndarray
    target_object: str
    target_position: np.ndarray
    target_object_id: str
    exploration_data_path: Path
    geodesic_distance: Optional[float] = None
    euclidean_distance: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            'episode_id': self.episode_id,
            'scene_id': self.scene_id,
            'scene_path': self.scene_path,
            'start_position': self.start_position.tolist(),
            'start_rotation': self.start_rotation.tolist(),
            'target_object': self.target_object,
            'target_position': self.target_position.tolist(),
            'target_object_id': self.target_object_id,
            'exploration_data_path': str(self.exploration_data_path),
            'geodesic_distance': self.geodesic_distance,
            'euclidean_distance': self.euclidean_distance,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'NavEpisode':
        return cls(
            episode_id=data['episode_id'],
            scene_id=data['scene_id'],
            scene_path=data['scene_path'],
            start_position=np.array(data['start_position']),
            start_rotation=np.array(data['start_rotation']),
            target_object=data['target_object'],
            target_position=np.array(data['target_position']),
            target_object_id=data['target_object_id'],
            exploration_data_path=Path(data['exploration_data_path']),
            geodesic_distance=data.get('geodesic_distance'),
            euclidean_distance=data.get('euclidean_distance'),
        )


class EpisodeGenerator:
    """Generate ObjectNav episodes from existing exploration data."""

    # Common object categories for ObjectNav
    TARGET_CATEGORIES = [
        'chair', 'table', 'sofa', 'bed', 'toilet', 'tv', 'sink',
        'bathtub', 'refrigerator', 'oven', 'microwave', 'plant',
        'lamp', 'clock', 'mirror', 'desk', 'couch', 'cabinet'
    ]

    def __init__(
        self,
        data_root: Path,
        hm3d_root: Optional[Path] = None,
        min_geodesic_distance: float = 1.0,
        max_geodesic_distance: float = 30.0,
        episodes_per_scene: int = 5,
        seed: int = 42,
    ):
        """
        Args:
            data_root: Path to multi_scene_eval outputs
            hm3d_root: Path to HM3D dataset (for geodesic computation)
            min_geodesic_distance: Minimum distance to target
            max_geodesic_distance: Maximum distance to target
            episodes_per_scene: Number of episodes to generate per scene
            seed: Random seed
        """
        self.data_root = Path(data_root)
        self.hm3d_root = Path(hm3d_root) if hm3d_root else None
        self.min_geodesic_distance = min_geodesic_distance
        self.max_geodesic_distance = max_geodesic_distance
        self.episodes_per_scene = episodes_per_scene
        self.rng = random.Random(seed)
        np.random.seed(seed)

        # Find all scenes with exploration data
        self.scenes = self._discover_scenes()
        print(f"Found {len(self.scenes)} scenes with exploration data")

    def _discover_scenes(self) -> List[Dict]:
        """Find all scenes with exploration data and ground truth."""
        scenes = []

        for scene_dir in sorted(self.data_root.iterdir()):
            if not scene_dir.is_dir():
                continue

            exploration_dir = scene_dir / "exploration"
            gt_file = scene_dir / f"{scene_dir.name}_ground_truth.json"

            # Check required files exist
            required_files = [
                exploration_dir / "embeddings.npy",
                exploration_dir / "trace.parquet",
                exploration_dir / "images",
            ]

            if all(f.exists() for f in required_files) and gt_file.exists():
                with open(gt_file) as f:
                    gt_data = json.load(f)

                scenes.append({
                    'scene_id': scene_dir.name,
                    'scene_path': gt_data.get('scene_path', ''),
                    'exploration_dir': exploration_dir,
                    'ground_truth': gt_data,
                })

        return scenes

    def _get_target_objects(self, ground_truth: Dict) -> List[Dict]:
        """Extract valid target objects from ground truth."""
        valid_objects = []

        objects = ground_truth.get('objects', {})
        if isinstance(objects, dict):
            objects = list(objects.values())

        for obj in objects:
            category = obj.get('category', '').lower()

            # Check if category is in our target list
            if any(target.lower() in category for target in self.TARGET_CATEGORIES):
                center = obj.get('center')
                if center is not None:
                    valid_objects.append({
                        'id': obj.get('id', 'unknown'),
                        'category': category,
                        'position': np.array(center),
                    })

        return valid_objects

    def _get_start_position(self, trace_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Get a valid start position from exploration trace."""
        import pandas as pd

        trace = pd.read_parquet(trace_path)

        # Use the first position from exploration as start
        first_row = trace.iloc[0]
        position = np.array([first_row['x'], first_row['y'], first_row['z']])

        # Extract rotation (quaternion) if available, else default
        if 'qw' in trace.columns:
            rotation = np.array([
                first_row['qw'], first_row['qx'],
                first_row['qy'], first_row['qz']
            ])
        else:
            rotation = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion

        return position, rotation

    def _compute_euclidean_distance(
        self,
        start: np.ndarray,
        target: np.ndarray
    ) -> float:
        """Compute Euclidean distance between positions."""
        return float(np.linalg.norm(start - target))

    def generate_episodes(
        self,
        max_scenes: Optional[int] = None,
        target_categories: Optional[List[str]] = None,
    ) -> List[NavEpisode]:
        """
        Generate navigation episodes.

        Args:
            max_scenes: Maximum number of scenes to use
            target_categories: Specific categories to target (default: all)

        Returns:
            List of NavEpisode objects
        """
        episodes = []
        episode_counter = 0

        scenes_to_use = self.scenes[:max_scenes] if max_scenes else self.scenes

        for scene_info in scenes_to_use:
            scene_id = scene_info['scene_id']
            exploration_dir = scene_info['exploration_dir']
            ground_truth = scene_info['ground_truth']

            # Get valid target objects
            target_objects = self._get_target_objects(ground_truth)

            if target_categories:
                target_objects = [
                    obj for obj in target_objects
                    if any(cat.lower() in obj['category']
                           for cat in target_categories)
                ]

            if not target_objects:
                continue

            # Get start position
            trace_path = exploration_dir / "trace.parquet"
            start_position, start_rotation = self._get_start_position(trace_path)

            # Sample episodes for this scene
            sampled_objects = self.rng.sample(
                target_objects,
                min(self.episodes_per_scene, len(target_objects))
            )

            for obj in sampled_objects:
                euclidean_dist = self._compute_euclidean_distance(
                    start_position, obj['position']
                )

                # Filter by distance constraints
                if euclidean_dist < self.min_geodesic_distance:
                    continue
                if euclidean_dist > self.max_geodesic_distance:
                    continue

                episode = NavEpisode(
                    episode_id=f"ep_{episode_counter:05d}",
                    scene_id=scene_id,
                    scene_path=scene_info['scene_path'],
                    start_position=start_position,
                    start_rotation=start_rotation,
                    target_object=obj['category'],
                    target_position=obj['position'],
                    target_object_id=obj['id'],
                    exploration_data_path=exploration_dir,
                    euclidean_distance=euclidean_dist,
                )

                episodes.append(episode)
                episode_counter += 1

        print(f"Generated {len(episodes)} episodes from {len(scenes_to_use)} scenes")
        return episodes

    def save_episodes(self, episodes: List[NavEpisode], output_path: Path):
        """Save episodes to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'num_episodes': len(episodes),
            'episodes': [ep.to_dict() for ep in episodes],
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Saved {len(episodes)} episodes to {output_path}")

    def load_episodes(self, input_path: Path) -> List[NavEpisode]:
        """Load episodes from JSON file."""
        with open(input_path) as f:
            data = json.load(f)

        episodes = [NavEpisode.from_dict(ep) for ep in data['episodes']]
        print(f"Loaded {len(episodes)} episodes from {input_path}")
        return episodes


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path,
                        default=Path('outputs/multi_scene_eval'))
    parser.add_argument('--output', type=Path,
                        default=Path('outputs/habitat_benchmark/episodes.json'))
    parser.add_argument('--episodes-per-scene', type=int, default=5)
    parser.add_argument('--max-scenes', type=int, default=None)
    args = parser.parse_args()

    generator = EpisodeGenerator(
        data_root=args.data_root,
        episodes_per_scene=args.episodes_per_scene,
    )

    episodes = generator.generate_episodes(max_scenes=args.max_scenes)
    generator.save_episodes(episodes, args.output)
