"""Run VLMaps-LSeg evaluation one scene at a time to avoid OOM.
Saves intermediate results after each scene to allow resuming on crash."""
import sys
import gc
import torch
import json
import argparse
from pathlib import Path
from dataclasses import asdict
sys.path.insert(0, str(Path(__file__).parent.parent))

from baselines.compare_all_baselines import (
    get_validation_scenes, 
    load_ground_truth, 
    evaluate_vlmap,
)


def load_checkpoint(checkpoint_path: Path) -> dict:
    """Load existing results if checkpoint exists."""
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            return json.load(f)
    return {"completed_scenes": {}, "all_results": []}


def save_checkpoint(checkpoint_path: Path, completed: dict, results: list):
    """Save checkpoint after each scene."""
    data = {"completed_scenes": completed, "all_results": results}
    with open(checkpoint_path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/full_validation")
    parser.add_argument("--scene-limit", type=int, default=None)
    parser.add_argument("--restart", action="store_true", help="Ignore checkpoint and restart")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Checkpoint file for resuming
    checkpoint_path = output_dir / "vlmap_checkpoint.json"
    
    base_dir = Path(__file__).parent.parent
    scenes = get_validation_scenes(base_dir, args.scene_limit)
    
    # Load checkpoint or start fresh
    if args.restart and checkpoint_path.exists():
        checkpoint_path.unlink()
        print("[RESTART] Cleared checkpoint, starting fresh")
    
    checkpoint = load_checkpoint(checkpoint_path)
    completed_scenes = checkpoint["completed_scenes"]
    all_results = checkpoint["all_results"]
    
    if completed_scenes:
        print(f"[RESUME] Found checkpoint with {len(completed_scenes)} completed scenes")
    
    for i, scene_dir in enumerate(scenes):
        scene_id = scene_dir.name
        
        # Skip already completed scenes
        if scene_id in completed_scenes:
            print(f"\n[Scene {i+1}/{len(scenes)}] {scene_id} - SKIPPED (already done)")
            continue
        
        print(f"\n[Scene {i+1}/{len(scenes)}] {scene_id}")
        
        # Force cleanup before each scene
        gc.collect()
        torch.cuda.empty_cache()
        
        # Load ground truth queries for this scene
        gt = load_ground_truth(scene_dir)
        queries = list(gt.keys())
        
        if not queries:
            print("  No queries, skipping")
            completed_scenes[scene_id] = {"status": "no_queries"}
            save_checkpoint(checkpoint_path, completed_scenes, all_results)
            continue
        
        print(f"  Queries available: {len(queries)}")
        
        try:
            # Build VLMap with LSeg
            print("  [VLMaps-LSeg] Building...")
            result = evaluate_vlmap(scene_dir, queries, gt, verbose=False, use_lseg=True)
            result_dict = asdict(result)
            result_dict["scene_id"] = scene_id
            all_results.append(result_dict)
            
            print(f"  [VLMaps-LSeg] Build: {result.build_time_seconds:.1f}s, "
                  f"Storage: {result.storage_mb:.1f}MB, "
                  f"Loc@1m: {result.loc_at_1m*100:.1f}%")
            
            # Mark as completed and save checkpoint immediately
            completed_scenes[scene_id] = {"status": "success", "loc_at_1m": result.loc_at_1m}
            save_checkpoint(checkpoint_path, completed_scenes, all_results)
            print(f"  [CHECKPOINT] Saved ({len(completed_scenes)}/{len(scenes)} scenes)")
            
            # Force cleanup after each scene
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            import traceback
            print(f"  [VLMaps-LSeg] ERROR: {e}")
            traceback.print_exc()
            completed_scenes[scene_id] = {"status": "error", "error": str(e)}
            save_checkpoint(checkpoint_path, completed_scenes, all_results)
            continue
    
    # Compute aggregates
    if all_results:
        n = len(all_results)
        avg_build = sum(r["build_time_seconds"] for r in all_results) / n
        avg_storage = sum(r["storage_mb"] for r in all_results) / n
        avg_query = sum(r["avg_query_time_ms"] for r in all_results) / n
        
        # Weighted average by query count
        total_queries = sum(r["total_queries"] for r in all_results)
        avg_loc_05 = sum(r["loc_at_05m"] * r["total_queries"] for r in all_results) / total_queries
        avg_loc_1 = sum(r["loc_at_1m"] * r["total_queries"] for r in all_results) / total_queries
        avg_loc_2 = sum(r["loc_at_2m"] * r["total_queries"] for r in all_results) / total_queries
        avg_loc_3 = sum(r["loc_at_3m"] * r["total_queries"] for r in all_results) / total_queries
        
        print("VLMaps-LSeg SUMMARY (36 scenes)")
        print(f"Pre-compute:  {avg_build:.1f}s")
        print(f"Storage:      {avg_storage:.1f}MB")
        print(f"Query time:   {avg_query:.1f}ms")
        print(f"Loc@0.5m:     {avg_loc_05*100:.1f}%")
        print(f"Loc@1m:       {avg_loc_1*100:.1f}%")
        print(f"Loc@2m:       {avg_loc_2*100:.1f}%")
        print(f"Loc@3m:       {avg_loc_3*100:.1f}%")
        
        # Save results
        summary = {
            "method": "VLMaps-LSeg",
            "num_scenes": n,
            "total_queries": total_queries,
            "avg_build_time": avg_build,
            "avg_storage_mb": avg_storage,
            "avg_query_time_ms": avg_query,
            "loc_at_05m": avg_loc_05,
            "loc_at_1m": avg_loc_1,
            "loc_at_2m": avg_loc_2,
            "loc_at_3m": avg_loc_3,
            "per_scene": all_results
        }
        
        with open(output_dir / "vlmap_results.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nResults saved to: {output_dir / 'vlmap_results.json'}")

if __name__ == "__main__":
    main()
