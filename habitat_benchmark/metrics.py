"""
Metrics for ObjectNav Evaluation
================================

Standard metrics for embodied navigation:
- Success Rate
- SPL (Success weighted by Path Length)
- Distance to Goal
- Soft SPL
"""

from typing import List, Dict
import numpy as np


def compute_success(
    final_distance: float,
    threshold: float = 1.0,
) -> bool:
    """
    Compute navigation success.

    Args:
        final_distance: Distance from agent to goal at end of episode
        threshold: Success threshold in meters

    Returns:
        True if successful
    """
    return final_distance <= threshold


def compute_spl(
    success: bool,
    path_length: float,
    optimal_distance: float,
    eps: float = 1e-6,
) -> float:
    """
    Compute Success weighted by Path Length (SPL).

    SPL = Success * (optimal_distance / max(path_length, optimal_distance))

    Args:
        success: Whether navigation was successful
        path_length: Actual path length taken
        optimal_distance: Geodesic distance (optimal path)
        eps: Small epsilon for numerical stability

    Returns:
        SPL value in [0, 1]
    """
    if not success:
        return 0.0

    if path_length == float('inf') or optimal_distance == float('inf'):
        return 0.0

    return optimal_distance / max(path_length, optimal_distance + eps)


def compute_metrics(
    results: List[Dict],
    thresholds: List[float] = [0.5, 1.0, 2.0, 3.0],
) -> Dict[str, float]:
    """
    Compute aggregate metrics from results.

    Args:
        results: List of result dicts with 'success', 'spl', 'final_distance', etc.
        thresholds: Distance thresholds for success computation

    Returns:
        Dictionary of aggregate metrics
    """
    if not results:
        return {}

    n = len(results)

    metrics = {
        'num_episodes': n,
        'avg_spl': np.mean([r['spl'] for r in results]) * 100,
        'avg_final_distance': np.mean([r['final_distance'] for r in results]),
    }

    # Success/SPL at different thresholds
    for t in thresholds:
        successes = sum(1 for r in results if r['final_distance'] <= t)
        metrics[f'success@{t}m'] = successes / n * 100

        # SPL at threshold
        spls = []
        for r in results:
            s = r['final_distance'] <= t
            spl = compute_spl(s, r['path_length'], r['geodesic_distance'])
            spls.append(spl)
        metrics[f'spl@{t}m'] = np.mean(spls) * 100

    return metrics
