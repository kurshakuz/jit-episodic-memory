#!/usr/bin/env python3
"""
Master pipeline: Run all methods on the 92 new ScanNet scenes.
Designed to be run incrementally as .sens files finish downloading.

Steps per scene:
1. Check .sens downloaded -> prepare_scenes (keyframes + CLIP-FAISS)
2. Run JIT (L1+L2), JIT + L3, BF + Depth 
3. Run DenseMap, VLMaps
4. (ConceptGraphs handled separately via cg env)

Usage:
    python scannet/run_all_new92.py                  # Process all available
    python scannet/run_all_new92.py --step prepare   # Only prepare
    python scannet/run_all_new92.py --step jit       # Only JIT eval
    python scannet/run_all_new92.py --step dense     # Only DenseMap/VLMaps
"""

import sys
import json
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
SCENES_FILE = Path(__file__).parent / "data" / "new_92_scenes.json"
SCANS_DIR = Path(__file__).parent / "data" / "scans"
JIT_DIR = Path(__file__).parent / "jit_format"
RESULTS_DIR = Path(__file__).parent / "results"
PROGRESS_FILE = Path(__file__).parent / "data" / "run_all_progress.json"

MIN_SENS_SIZE = 10 * 1024 * 1024


def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"prepared": [], "jit_done": [], "dense_done": [], "failed": {}}


def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def has_sens(scene_id):
    sf = SCANS_DIR / scene_id / f"{scene_id}.sens"
    return sf.exists() and sf.stat().st_size > MIN_SENS_SIZE


def is_prepared(scene_id):
    d = JIT_DIR / scene_id
    return (d / "exploration" / "trace.parquet").exists() and (d / "exploration" / "memory.index").exists()


def has_gt(scene_id):
    return (JIT_DIR / scene_id / f"{scene_id}_ground_truth.json").exists()


def run_cmd(cmd, label=""):
    print(f"  Running: {label or ' '.join(cmd[:3])}")
    result = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[-500:]}")
        return False
    return True


def prepare_scene(scene_id):
    """Prepare a single scene (keyframes + CLIP-FAISS)."""
    if is_prepared(scene_id):
        return True
    return run_cmd(
        [sys.executable, "scannet/prepare_scenes.py", "--scenes", scene_id],
        f"prepare {scene_id}"
    )


def run_jit_scene(scene_id):
    """Run JIT, JIT+L3, BF on a single scene."""
    return run_cmd(
        [sys.executable, "scannet/evaluate_v2.py", "--scenes", scene_id, "--methods", "jit,jit_l3,bf"],
        f"JIT eval {scene_id}"
    )


def run_dense_scene(scene_id):
    """Run DenseMap + VLMaps on a single scene."""
    return run_cmd(
        [sys.executable, "scannet/run_dense_baselines.py", "--methods", "densemap,vlmap", "--scenes", scene_id],
        f"dense baselines {scene_id}"
    )


def main():
    step = "all"
    if "--step" in sys.argv:
        idx = sys.argv.index("--step")
        step = sys.argv[idx + 1]
    
    with open(SCENES_FILE) as f:
        all_scenes = json.load(f)["new_scenes"]
    
    progress = load_progress()
    
    # Status check
    downloaded = [s for s in all_scenes if has_sens(s)]
    prepared = [s for s in all_scenes if is_prepared(s)]
    with_gt = [s for s in all_scenes if has_gt(s)]
    
    print(f"=== Pipeline Status ===")
    print(f"Total scenes: {len(all_scenes)}")
    print(f"Downloaded (.sens): {len(downloaded)}")
    print(f"With GT: {len(with_gt)}")
    print(f"Prepared: {len(prepared)}")
    print(f"JIT evaluated: {len(progress['jit_done'])}")
    print(f"Dense evaluated: {len(progress['dense_done'])}")
    print(f"Step: {step}")
    print()
    
    # Step 1: Prepare scenes
    if step in ("all", "prepare"):
        to_prepare = [s for s in downloaded if not is_prepared(s) and s not in progress.get("prepare_failed", [])]
        print(f"--- Preparing {len(to_prepare)} scenes ---")
        for i, scene_id in enumerate(to_prepare):
            print(f"\n[{i+1}/{len(to_prepare)}] Preparing {scene_id}")
            if prepare_scene(scene_id):
                if scene_id not in progress["prepared"]:
                    progress["prepared"].append(scene_id)
            else:
                progress.setdefault("prepare_failed", []).append(scene_id)
            save_progress(progress)
    
    # Step 2: Run JIT eval
    if step in ("all", "jit"):
        ready = [s for s in all_scenes if is_prepared(s) and has_gt(s) and s not in progress["jit_done"]]
        print(f"\n--- Running JIT on {len(ready)} scenes ---")
        for i, scene_id in enumerate(ready):
            print(f"\n[{i+1}/{len(ready)}] JIT eval {scene_id}")
            if run_jit_scene(scene_id):
                progress["jit_done"].append(scene_id)
            save_progress(progress)
    
    # Step 3: Run dense baselines
    if step in ("all", "dense"):
        ready = [s for s in all_scenes if is_prepared(s) and has_gt(s) and s not in progress["dense_done"]]
        print(f"\n--- Running Dense baselines on {len(ready)} scenes ---")
        for i, scene_id in enumerate(ready):
            print(f"\n[{i+1}/{len(ready)}] Dense eval {scene_id}")
            if run_dense_scene(scene_id):
                progress["dense_done"].append(scene_id)
            save_progress(progress)
    
    print(f"\n=== Final Status ===")
    print(f"Prepared: {len([s for s in all_scenes if is_prepared(s)])}/{len(all_scenes)}")
    print(f"JIT done: {len(progress['jit_done'])}/{len(all_scenes)}")
    print(f"Dense done: {len(progress['dense_done'])}/{len(all_scenes)}")


if __name__ == "__main__":
    main()
