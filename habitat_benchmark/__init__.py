"""
Habitat Embodied Benchmark for JIT Episodic Memory
===================================================

This module implements ObjectNav-style evaluation on HM3D using the JIT Memory system.

Pipeline:
1. Exploration Phase: Robot explores environment, builds episodic memory
2. Query Phase: Given object category, JIT Memory predicts 3D goal location
3. Navigation Phase: Robot navigates to predicted goal using shortest-path follower
4. Evaluation: Success/SPL, time-to-goal, success vs distance threshold

Baselines:
- Pose-only: Use camera pose from L1 CLIP retrieval (no depth)
- L1+OWL+Depth: Single-stage retrieval with depth projection (no DBSCAN)
- Brute-Force: Run OWL-ViT on all frames, pick best
- JIT Cascade: Full L1->L2->L3 pipeline with depth

Metrics:
- Success Rate: Did robot reach within 1m of ground truth?
- SPL (Success weighted by Path Length): Efficiency metric
- Distance to Goal: Final distance to ground truth object
- Steps to Goal: Number of navigation steps taken
"""

from .episode_generator import EpisodeGenerator
from .navigator import NavigationEvaluator
from .metrics import compute_metrics, compute_spl
from .baselines import PoseOnlyBaseline, L1OwlDepthBaseline, BruteForceBaseline, JITCascadeBaseline

__all__ = [
    'EpisodeGenerator',
    'NavigationEvaluator',
    'compute_metrics',
    'compute_spl',
    'PoseOnlyBaseline',
    'L1OwlDepthBaseline',
    'BruteForceBaseline',
    'JITCascadeBaseline',
]
