# Habitat ObjectNav Benchmark

Embodied navigation evaluation for JIT Episodic Memory on HM3D.

## Overview

This benchmark evaluates the **practical utility** of the JIT Memory system by testing:
- Can the robot successfully navigate to predicted object locations?
- How does efficiency (SPL) compare across methods?
- What is the trade-off between accuracy and latency?

## Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     ObjectNav Evaluation                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. EXPLORATION PHASE                                           │
│     ┌──────────┐    ┌──────────────┐    ┌────────────────┐     │
│     │  Robot   │───▶│  Explore     │───▶│  Build Memory  │     │
│     │  Start   │    │  Environment │    │  (CLIP + Depth)│     │
│     └──────────┘    └──────────────┘    └────────────────┘     │
│                                                                 │
│  2. QUERY PHASE                                                 │
│     ┌──────────┐    ┌──────────────┐    ┌────────────────┐     │
│     │  "Find   │───▶│  JIT Memory  │───▶│  3D Goal       │     │
│     │  chair"  │    │  Retrieval   │    │  Prediction    │     │
│     └──────────┘    └──────────────┘    └────────────────┘     │
│                                                                 │
│  3. NAVIGATION PHASE                                            │
│     ┌──────────┐    ┌──────────────┐    ┌────────────────┐     │
│     │  Shortest│───▶│  Execute     │───▶│  Evaluate      │     │
│     │  Path    │    │  Navigation  │    │  Success/SPL   │     │
│     └──────────┘    └──────────────┘    └────────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Baselines

| Method | Description |
|--------|-------------|
| **Pose-Only** | CLIP retrieval → use camera pose as goal (no depth) |
| **L1+OWL+Depth** | CLIP + OWL-ViT on top-k + depth projection (no DBSCAN) |
| **Brute Force** | OWL-ViT on ALL frames + depth projection |
| **JIT Cascade** | Full L1→L2→L3 pipeline with DBSCAN + depth |

## Metrics

- **Success Rate** (`Success@Xm`): Percentage reaching within X meters of target
- **SPL** (Success weighted by Path Length): Efficiency metric (0-100%)
- **Distance to Goal**: Average final distance from target
- **Latency**: Time for goal prediction

## Quick Start

```bash
cd habitat_benchmark

# Generate episodes and run all baselines
python run_benchmark.py --all-baselines --data-root ../outputs/multi_scene_eval

# Run specific baseline
python run_benchmark.py --baseline jit_cascade --max-episodes 100

# Use pre-generated episodes
python run_benchmark.py --episodes-file episodes.json --baseline pose_only
```

## Usage

### 1. Generate Episodes

```python
from habitat_benchmark import EpisodeGenerator

generator = EpisodeGenerator(
    data_root="outputs/multi_scene_eval",
    episodes_per_scene=5,
)

episodes = generator.generate_episodes(max_scenes=50)
generator.save_episodes(episodes, "episodes.json")
```

### 2. Run Evaluation

```python
from habitat_benchmark import NavigationEvaluator
from habitat_benchmark.baselines import JITCascadeBaseline

evaluator = NavigationEvaluator(success_threshold=1.0)
baseline = JITCascadeBaseline()
baseline.load_models()

for episode in episodes:
    prediction = baseline.predict(
        query=episode.target_object,
        exploration_dir=episode.exploration_data_path,
    )
    
    result = evaluator.evaluate_episode(
        episode=episode,
        predicted_goal=prediction.goal,
        detection_success=prediction.success,
    )
    print(f"{episode.episode_id}: Success={result.success}, SPL={result.spl:.2f}")
```

### 3. Compute Metrics

```python
metrics = evaluator.compute_aggregate_metrics(results)
print(f"Success@1m: {metrics['success_at_1.0m']:.1f}%")
print(f"SPL@1m: {metrics['spl_at_1.0m']:.1f}%")
```

## Expected Results

Evaluation on 538 episodes across 181 HM3D scenes:

| Method | Det.% | Loc@1m | Loc@2m | Loc@3m | Med. Err | Latency |
|--------|-------|--------|--------|--------|----------|---------|
| Pose-Only | 100% | 0.0% | 2.4% | 6.1% | 8.00m | 34ms |
| **JIT Cascade** | 37.5% | **1.5%** | **4.1%** | **10.6%** | **7.17m** | 408ms |

**Key Findings:**
- JIT Cascade achieves **+73% relative improvement** on Loc@3m
- **0.83m reduction** in median localization error
- Latency of 408ms is **within 500ms robotics budget**

### Best-Performing Object Categories

| Category | Detection Rate |
|----------|---------------|
| bed | 100% (19/19) |
| kitchen cabinet | 73% (16/22) |
| tv | 71% (5/7) |
| sink | 58% (11/19) |
| ceiling lamp | 55% (24/44) |

### Notes
- Ground truth: HM3D semantic object centroids
- Detection Rate: OWL-ViT successfully detects the queried object
- Loc@Xm: Prediction within X meters of ground truth

## File Structure

```
habitat_benchmark/
├── __init__.py           # Module exports
├── episode_generator.py  # Generate ObjectNav episodes
├── navigator.py          # Navigation evaluation
├── baselines.py          # Baseline implementations
├── metrics.py            # Success/SPL computation
├── run_benchmark.py      # Main evaluation script
├── config.yaml           # Configuration
└── README.md             # This file
```

## Configuration

Edit `config.yaml` to customize:

```yaml
episodes:
  per_scene: 5
  min_geodesic_distance: 1.0

evaluation:
  success_thresholds: [0.5, 1.0, 2.0, 3.0]
  default_threshold: 1.0

baselines:
  jit_cascade:
    k: 20
    dbscan_eps: 0.5
```

## Citation

```bibtex
@unpublished{jit_episodic_memory,
  title={Spatially-Grounded Just-in-Time Episodic Memory for Mobile Robots},
  author={Shyngyskhan Abilkassov and Almas Shintemirov},
  year={2026},
  note={Under review}
}
```
