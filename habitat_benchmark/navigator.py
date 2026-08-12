"""
Navigation Evaluator for ObjectNav Benchmark
=============================================

Evaluates navigation performance:
1. Given a goal prediction from a baseline/method
2. Compute shortest path to goal using Habitat's pathfinder
3. Execute navigation (or simulate it)
4. Report Success/SPL metrics
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import json
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from .episode_generator import NavEpisode
from .metrics import compute_spl


@dataclass
class NavResult:
    """Result of a single navigation episode."""
    episode_id: str
    scene_id: str
    target_object: str

    # Predictions
    predicted_goal: Optional[np.ndarray]
    detection_success: bool
    detection_score: Optional[float]

    # Ground truth
    gt_position: np.ndarray

    # Navigation metrics
    path_length: Optional[float]
    geodesic_distance: Optional[float]
    final_distance: float
    success: bool
    spl: float

    # Additional info
    steps_taken: int
    method_latency_ms: float

    def to_dict(self) -> Dict:
        return {
            'episode_id': self.episode_id,
            'scene_id': self.scene_id,
            'target_object': self.target_object,
            'predicted_goal': self.predicted_goal.tolist() if self.predicted_goal is not None else None,
            'detection_success': self.detection_success,
            'detection_score': self.detection_score,
            'gt_position': self.gt_position.tolist(),
            'path_length': self.path_length,
            'geodesic_distance': self.geodesic_distance,
            'final_distance': self.final_distance,
            'success': self.success,
            'spl': self.spl,
            'steps_taken': self.steps_taken,
            'method_latency_ms': self.method_latency_ms,
        }


class NavigationEvaluator:
    """
    Evaluate navigation performance for ObjectNav.

    This evaluator computes metrics without running full Habitat simulation:
    - Uses Euclidean distance as proxy for geodesic (or Habitat if available)
    - Computes success based on distance threshold
    - Computes SPL using path length estimates
    """

    SUCCESS_THRESHOLDS = [0.5, 1.0, 2.0, 3.0]  # meters

    def __init__(
        self,
        use_habitat_sim: bool = False,
        hm3d_root: Optional[Path] = None,
        success_threshold: float = 1.0,
    ):
        """
        Args:
            use_habitat_sim: Whether to use Habitat simulator for path planning
            hm3d_root: Path to HM3D dataset
            success_threshold: Default success threshold in meters
        """
        self.use_habitat_sim = use_habitat_sim
        self.hm3d_root = hm3d_root
        self.success_threshold = success_threshold

        self.simulator = None
        if use_habitat_sim:
            self._init_habitat_sim()

    def _init_habitat_sim(self):
        """Initialize Habitat simulator for geodesic computation."""
        try:
            import habitat_sim
            print("Habitat-sim available, will use for geodesic distances")
            self.habitat_sim = habitat_sim
        except ImportError:
            print("Warning: habitat-sim not available, using Euclidean distances")
            self.use_habitat_sim = False

    def _load_scene(self, scene_path: str) -> Optional[Any]:
        """Load a scene in Habitat simulator."""
        if not self.use_habitat_sim:
            return None

        try:
            cfg = self.habitat_sim.SimulatorConfiguration()
            cfg.scene_id = scene_path
            cfg.enable_physics = False

            agent_cfg = self.habitat_sim.AgentConfiguration()

            sim_cfg = self.habitat_sim.Configuration()
            sim_cfg.sim_cfg = cfg
            sim_cfg.agents = [agent_cfg]

            if self.simulator is not None:
                self.simulator.close()

            self.simulator = self.habitat_sim.Simulator(sim_cfg)
            return self.simulator
        except Exception as e:
            print(f"Failed to load scene: {e}")
            return None

    def _compute_geodesic_distance(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        scene_path: Optional[str] = None,
    ) -> Tuple[float, float]:
        """
        Compute geodesic distance and path length.

        Returns:
            (geodesic_distance, path_length)
        """
        if self.use_habitat_sim and scene_path:
            try:
                if self.simulator is None or self.simulator.config.sim_cfg.scene_id != scene_path:
                    self._load_scene(scene_path)

                if self.simulator is not None:
                    path = self.habitat_sim.ShortestPath()
                    path.requested_start = start.tolist()
                    path.requested_end = goal.tolist()

                    if self.simulator.pathfinder.find_path(path):
                        return path.geodesic_distance, path.geodesic_distance
            except Exception:
                pass

        # Fallback to Euclidean distance
        euclidean = float(np.linalg.norm(start - goal))
        # Estimate path length as 1.2x Euclidean (typical indoor ratio)
        path_length = euclidean * 1.2
        return euclidean, path_length

    def _compute_final_distance(
        self,
        predicted_goal: Optional[np.ndarray],
        gt_position: np.ndarray,
    ) -> float:
        """Compute distance from predicted goal to ground truth."""
        if predicted_goal is None:
            return float('inf')
        return float(np.linalg.norm(predicted_goal - gt_position))

    def evaluate_episode(
        self,
        episode: NavEpisode,
        predicted_goal: Optional[np.ndarray],
        detection_success: bool,
        detection_score: Optional[float] = None,
        method_latency_ms: float = 0.0,
    ) -> NavResult:
        """
        Evaluate a single navigation episode.

        Args:
            episode: Navigation episode
            predicted_goal: 3D goal position predicted by method
            detection_success: Whether method detected the object
            detection_score: Detection confidence score
            method_latency_ms: Time taken by method

        Returns:
            NavResult with all metrics
        """
        # Compute distance to ground truth
        final_distance = self._compute_final_distance(
            predicted_goal, episode.target_position
        )

        # Success at default threshold
        success = final_distance <= self.success_threshold

        # Compute path lengths
        if predicted_goal is not None:
            geodesic_to_goal, path_length = self._compute_geodesic_distance(
                episode.start_position,
                predicted_goal,
                episode.scene_path,
            )
        else:
            geodesic_to_goal = float('inf')
            path_length = float('inf')

        # Optimal geodesic (to ground truth)
        optimal_geodesic, _ = self._compute_geodesic_distance(
            episode.start_position,
            episode.target_position,
            episode.scene_path,
        )

        # Compute SPL
        spl = compute_spl(
            success=success,
            path_length=path_length,
            optimal_distance=optimal_geodesic,
        )

        # Estimate steps (assuming 0.25m per step)
        steps_taken = int(path_length / 0.25) if path_length != float('inf') else 0

        return NavResult(
            episode_id=episode.episode_id,
            scene_id=episode.scene_id,
            target_object=episode.target_object,
            predicted_goal=predicted_goal,
            detection_success=detection_success,
            detection_score=detection_score,
            gt_position=episode.target_position,
            path_length=path_length,
            geodesic_distance=optimal_geodesic,
            final_distance=final_distance,
            success=success,
            spl=spl,
            steps_taken=steps_taken,
            method_latency_ms=method_latency_ms,
        )

    def evaluate_batch(
        self,
        episodes: List[NavEpisode],
        predictions: List[Dict],
    ) -> List[NavResult]:
        """
        Evaluate a batch of episodes.

        Args:
            episodes: List of episodes
            predictions: List of dicts with 'goal', 'success', 'score', 'latency_ms'

        Returns:
            List of NavResult
        """
        results = []

        for episode, pred in tqdm(zip(episodes, predictions),
                                   total=len(episodes),
                                   desc="Evaluating navigation"):
            result = self.evaluate_episode(
                episode=episode,
                predicted_goal=pred.get('goal'),
                detection_success=pred.get('success', False),
                detection_score=pred.get('score'),
                method_latency_ms=pred.get('latency_ms', 0.0),
            )
            results.append(result)

        return results

    def compute_aggregate_metrics(
        self,
        results: List[NavResult],
    ) -> Dict[str, float]:
        """Compute aggregate metrics over all results."""
        if not results:
            return {}

        # Basic metrics
        detection_successes = sum(1 for r in results if r.detection_success)
        nav_successes = sum(1 for r in results if r.success)

        metrics = {
            'num_episodes': len(results),
            'detection_recall': detection_successes / len(results) * 100,
            'navigation_success': nav_successes / len(results) * 100,
            'avg_spl': np.mean([r.spl for r in results]) * 100,
            'avg_final_distance': np.mean([r.final_distance for r in results
                                           if r.final_distance != float('inf')]),
            'avg_path_length': np.mean([r.path_length for r in results
                                        if r.path_length != float('inf')]),
            'avg_latency_ms': np.mean([r.method_latency_ms for r in results]),
        }

        # Success at different thresholds
        for threshold in self.SUCCESS_THRESHOLDS:
            successes = sum(1 for r in results if r.final_distance <= threshold)
            metrics[f'success_at_{threshold}m'] = successes / len(results) * 100

            # SPL at this threshold
            spls = []
            for r in results:
                success = r.final_distance <= threshold
                if r.path_length != float('inf') and r.geodesic_distance:
                    spl = compute_spl(success, r.path_length, r.geodesic_distance)
                else:
                    spl = 0.0
                spls.append(spl)
            metrics[f'spl_at_{threshold}m'] = np.mean(spls) * 100

        # Per-category metrics
        categories = set(r.target_object for r in results)
        for cat in categories:
            cat_results = [r for r in results if r.target_object == cat]
            cat_successes = sum(1 for r in cat_results if r.success)
            metrics[f'success_{cat}'] = cat_successes / len(cat_results) * 100

        return metrics

    def save_results(
        self,
        results: List[NavResult],
        metrics: Dict[str, float],
        output_path: Path,
    ):
        """Save results and metrics to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'metrics': metrics,
            'results': [r.to_dict() for r in results],
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Saved {len(results)} results to {output_path}")
