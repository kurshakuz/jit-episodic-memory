#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# CG Scaling Experiment: stride=1 vs stride=4 on 5 matched scenes
#
# This script runs a controlled comparison of ConceptGraphs at two frame
# densities using the EXACT same exploration traces.
#
# Phase 1: Pre-populate stride=1 cache with stride=4 frame data (symlinks)
# Phase 2: Run CG at stride=1 (SAM→CLIP→Mapping→Eval) on 5 scenes
# Phase 3: Run CG at stride=4 on same 5 scenes (fresh cache, for timing)
# Phase 4: Compare results
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

CONDA_CG="${CONDA_CG:-python}"
STRIDE4_CACHE="$PROJECT_ROOT/outputs/paper_results/cg_official_cache"
STRIDE1_CACHE="$PROJECT_ROOT/outputs/paper_results/cg_stride1_cache"
STRIDE4_TIMED_CACHE="$PROJECT_ROOT/outputs/paper_results/cg_stride4_timed_cache"
N_SCENES=5

echo "═══════════════════════════════════════════════════════════════════"
echo "CG Scaling Experiment: stride=1 vs stride=4"
echo "═══════════════════════════════════════════════════════════════════"
echo "  Project root: $PROJECT_ROOT"
echo "  Scenes: $N_SCENES"
echo "  Stride=4 cache: $STRIDE4_CACHE"
echo "  Stride=1 cache: $STRIDE1_CACHE"
echo ""

# ─── Phase 1: Pre-populate stride=1 cache with stride=4 frame data ──────────
echo "Phase 1: Pre-populating stride=1 cache..."
$CONDA_CG "$PROJECT_ROOT/experiments/prepopulate_cache.py" \
    --source "$STRIDE4_CACHE" \
    --dest "$STRIDE1_CACHE" \
    --scenes $N_SCENES
echo "Phase 1 complete."
echo ""

# ─── Phase 2: Run CG at stride=1 ────────────────────────────────────────────
echo "Phase 2: Running CG pipeline at stride=1 on $N_SCENES scenes..."
echo "  This will process 160 frames/scene (120 new + 40 pre-cached)"
echo "  Estimated time: ~3 hours"
echo ""

$CONDA_CG "$PROJECT_ROOT/baselines/run_official_cg.py" \
    --stride 1 \
    --scenes $N_SCENES \
    --cache-dir "outputs/paper_results/cg_stride1_cache" \
    --output "outputs/paper_results/cg_stride1_results.json" \
    2>&1 | tee "$PROJECT_ROOT/outputs/paper_results/cg_stride1_run.log"

echo ""
echo "Phase 2 complete."
echo ""

# ─── Phase 3: Run CG at stride=4 on same scenes (fresh, for matched timing) ─
echo "Phase 3: Running CG at stride=4 on same $N_SCENES scenes (fresh cache)..."
echo "  This will process 40 frames/scene"
echo ""

$CONDA_CG "$PROJECT_ROOT/baselines/run_official_cg.py" \
    --stride 4 \
    --scenes $N_SCENES \
    --cache-dir "outputs/paper_results/cg_stride4_timed_cache" \
    --output "outputs/paper_results/cg_stride4_timed_results.json" \
    2>&1 | tee "$PROJECT_ROOT/outputs/paper_results/cg_stride4_timed_run.log"

echo ""
echo "Phase 3 complete."
echo ""

# ─── Phase 4: Compare results ───────────────────────────────────────────────
echo "Phase 4: Comparing results..."
$CONDA_CG "$PROJECT_ROOT/experiments/compare_scaling_results.py"

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "Experiment complete!"
echo "═══════════════════════════════════════════════════════════════════"
