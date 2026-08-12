#!/usr/bin/env python3
"""
Run ObjectNav Benchmark Evaluation
==================================

Main script to run the ObjectNav-style evaluation on HM3D.

Usage:
    python run_benchmark.py --data-root outputs/multi_scene_eval --output outputs/habitat_benchmark

    # Run specific baseline
    python run_benchmark.py --baseline jit_cascade --max-episodes 100

    # Run all baselines
    python run_benchmark.py --all-baselines
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict
import numpy as np
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from habitat_benchmark.episode_generator import EpisodeGenerator, NavEpisode
from habitat_benchmark.navigator import NavigationEvaluator, NavResult
from habitat_benchmark.baselines import (
    get_baseline,
    get_all_baselines,
    BaselineMethod,
)


def run_baseline_evaluation(
    baseline: BaselineMethod,
    episodes: List[NavEpisode],
    evaluator: NavigationEvaluator,
    verbose: bool = True,
) -> tuple[List[NavResult], Dict[str, float]]:
    """
    Run evaluation for a single baseline.

    Args:
        baseline: Baseline method to evaluate
        episodes: List of episodes
        evaluator: Navigation evaluator
        verbose: Print progress

    Returns:
        (results, metrics)
    """
    baseline.ensure_models_loaded()

    results = []

    iterator = tqdm(episodes, desc=f"Evaluating {baseline.name}") if verbose else episodes

    for episode in iterator:
        # Get goal prediction from baseline
        prediction = baseline.predict(
            query=episode.target_object,
            exploration_dir=episode.exploration_data_path,
        )

        # Evaluate navigation
        result = evaluator.evaluate_episode(
            episode=episode,
            predicted_goal=prediction.goal,
            detection_success=prediction.success,
            detection_score=prediction.score,
            method_latency_ms=prediction.latency_ms,
        )

        results.append(result)

    # Compute aggregate metrics
    metrics = evaluator.compute_aggregate_metrics(results)

    return results, metrics


def run_all_baselines(
    episodes: List[NavEpisode],
    evaluator: NavigationEvaluator,
    output_dir: Path,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Run evaluation for all baselines.

    Returns:
        Dictionary mapping baseline name to metrics
    """
    all_metrics = {}

    baselines = get_all_baselines()

    for baseline in baselines:
        print(f"\n{'='*60}")
        print(f"Evaluating: {baseline.name}")
        print('='*60)

        results, metrics = run_baseline_evaluation(
            baseline=baseline,
            episodes=episodes,
            evaluator=evaluator,
            verbose=verbose,
        )

        all_metrics[baseline.name] = metrics

        # Save results
        result_path = output_dir / f"results_{baseline.name}.json"
        evaluator.save_results(results, metrics, result_path)

        # Print summary
        print(f"\n{baseline.name} Summary:")
        print(f"  Detection Recall: {metrics.get('detection_recall', 0):.1f}%")
        print(f"  Nav Success @1m:  {metrics.get('success_at_1.0m', 0):.1f}%")
        print(f"  SPL @1m:          {metrics.get('spl_at_1.0m', 0):.1f}%")
        print(f"  Avg Latency:      {metrics.get('avg_latency_ms', 0):.0f}ms")

    return all_metrics


def print_comparison_table(all_metrics: Dict[str, Dict[str, float]]):
    """Print comparison table of all baselines."""
    print("\n" + "="*80)
    print("OBJECTNAV BENCHMARK RESULTS")
    print("="*80)

    # Header
    header = f"{'Method':<20} {'Det.%':>8} {'Succ@0.5m':>10} {'Succ@1m':>10} {'Succ@2m':>10} {'SPL@1m':>10} {'Latency':>10}"
    print(header)
    print("-"*80)

    # Rows
    for name, metrics in all_metrics.items():
        row = (
            f"{name:<20} "
            f"{metrics.get('detection_recall', 0):>7.1f}% "
            f"{metrics.get('success_at_0.5m', 0):>9.1f}% "
            f"{metrics.get('success_at_1.0m', 0):>9.1f}% "
            f"{metrics.get('success_at_2.0m', 0):>9.1f}% "
            f"{metrics.get('spl_at_1.0m', 0):>9.1f}% "
            f"{metrics.get('avg_latency_ms', 0):>8.0f}ms"
        )
        print(row)

    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="ObjectNav Benchmark Evaluation")

    parser.add_argument('--data-root', type=Path,
                        default=Path('outputs/multi_scene_eval'),
                        help='Path to exploration data')
    parser.add_argument('--output', type=Path,
                        default=Path('outputs/habitat_benchmark'),
                        help='Output directory for results')
    parser.add_argument('--episodes-file', type=Path, default=None,
                        help='Pre-generated episodes file (optional)')

    # Episode generation
    parser.add_argument('--episodes-per-scene', type=int, default=5,
                        help='Episodes per scene')
    parser.add_argument('--max-scenes', type=int, default=None,
                        help='Maximum scenes to use')
    parser.add_argument('--max-episodes', type=int, default=None,
                        help='Maximum episodes to evaluate')

    # Baseline selection
    parser.add_argument('--baseline', type=str, default=None,
                        choices=['pose_only', 'l1_owl_depth', 'brute_force', 'jit_cascade'],
                        help='Single baseline to run')
    parser.add_argument('--all-baselines', action='store_true',
                        help='Run all baselines')

    # Evaluation settings
    parser.add_argument('--success-threshold', type=float, default=1.0,
                        help='Success distance threshold (meters)')
    parser.add_argument('--use-habitat-sim', action='store_true',
                        help='Use Habitat simulator for geodesic distances')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quiet', action='store_true')

    args = parser.parse_args()

    # Setup
    args.output.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    print("="*60)
    print("ObjectNav Benchmark Evaluation")
    print("="*60)
    print(f"Data root: {args.data_root}")
    print(f"Output: {args.output}")

    # Generate or load episodes
    if args.episodes_file and args.episodes_file.exists():
        print(f"\nLoading episodes from {args.episodes_file}")
        generator = EpisodeGenerator(args.data_root)
        episodes = generator.load_episodes(args.episodes_file)
    else:
        print("\nGenerating episodes...")
        generator = EpisodeGenerator(
            data_root=args.data_root,
            episodes_per_scene=args.episodes_per_scene,
            seed=args.seed,
        )
        episodes = generator.generate_episodes(max_scenes=args.max_scenes)

        # Save episodes
        episodes_path = args.output / "episodes.json"
        generator.save_episodes(episodes, episodes_path)

    # Limit episodes if specified
    if args.max_episodes and len(episodes) > args.max_episodes:
        episodes = episodes[:args.max_episodes]
        print(f"Limited to {len(episodes)} episodes")

    print(f"\nTotal episodes: {len(episodes)}")

    # Create evaluator
    evaluator = NavigationEvaluator(
        use_habitat_sim=args.use_habitat_sim,
        success_threshold=args.success_threshold,
    )

    # Run evaluation
    if args.all_baselines:
        all_metrics = run_all_baselines(
            episodes=episodes,
            evaluator=evaluator,
            output_dir=args.output,
            verbose=not args.quiet,
        )

        # Print comparison
        print_comparison_table(all_metrics)

        # Save combined metrics
        combined_path = args.output / "all_results.json"
        with open(combined_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nSaved combined results to {combined_path}")

    elif args.baseline:
        baseline = get_baseline(args.baseline)
        results, metrics = run_baseline_evaluation(
            baseline=baseline,
            episodes=episodes,
            evaluator=evaluator,
            verbose=not args.quiet,
        )

        # Save results
        result_path = args.output / f"results_{args.baseline}.json"
        evaluator.save_results(results, metrics, result_path)

        # Print summary
        print(f"\n{args.baseline} Results:")
        for k, v in sorted(metrics.items()):
            if isinstance(v, float):
                print(f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v}")
    else:
        print("\nNo baseline specified. Use --baseline or --all-baselines")
        print("Available baselines: pose_only, l1_owl_depth, brute_force, jit_cascade")


if __name__ == "__main__":
    main()
