#!/usr/bin/env python3
"""Pre-populate stride=1 cache with existing stride=4 frame-level data.

Symlinks mask and detection files from the stride=4 cache into the stride=1
cache directory. This avoids re-computing SAM/CLIP for frames already processed
at stride=4 (every 4th frame).

Does NOT copy done markers or object maps, so the pipeline will:
  - SAM pass: process only the 120 new frames (1,2,3,5,6,7,...)
  - CLIP pass: process only the 120 new frames  
  - Mapping: re-run from scratch with all 160 frames
"""

import argparse
import json
import os
from pathlib import Path


def get_val_scene_ids(project_root: Path):
    """Get the 36 val scene IDs from full_results_fixed.json."""
    path = project_root / "outputs" / "full_scale_eval" / "full_results_fixed.json"
    with open(path) as f:
        data = json.load(f)
    return sorted(set(sr["scene_id"] for sr in data["scene_results"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Stride=4 cache dir")
    parser.add_argument("--dest", required=True, help="Stride=1 cache dir")
    parser.add_argument("--scenes", type=int, default=5, help="Number of scenes")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    source = Path(args.source)
    dest = Path(args.dest)

    scene_ids = get_val_scene_ids(project_root)[:args.scenes]
    print(f"Pre-populating stride=1 cache for {len(scene_ids)} scenes")

    total_linked = 0
    for scene_id in scene_ids:
        src_dir = source / scene_id
        dst_dir = dest / scene_id
        dst_dir.mkdir(parents=True, exist_ok=True)

        # Symlink mask and detection files (NOT done markers or object maps)
        linked = 0
        for pattern in ["masks_*.pkl.gz", "detections_*.pkl.gz"]:
            for src_file in sorted(src_dir.glob(pattern)):
                dst_file = dst_dir / src_file.name
                if dst_file.exists():
                    continue
                # Use relative symlink for portability
                rel = os.path.relpath(src_file, dst_dir)
                os.symlink(rel, dst_file)
                linked += 1

        total_linked += linked
        print(f"  {scene_id}: {linked} files symlinked")

    print(f"\nTotal: {total_linked} files symlinked across {len(scene_ids)} scenes")
    print("Done markers and object maps NOT copied (will be regenerated)")


if __name__ == "__main__":
    main()
