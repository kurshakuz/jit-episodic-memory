#!/usr/bin/env python3
"""
Non-interactive ScanNet batch downloader.
Downloads specific file types for selected scenes, bypassing TOS prompts.
Reuses download logic from the official download-scannet.py.

Usage:
    python scannet/download_batch.py --annotations   # Small files first
    python scannet/download_batch.py --sens           # Large .sens files
    python scannet/download_batch.py --label-map      # Label mapping file
    python scannet/download_batch.py --all             # Everything
"""

import os
import sys
import argparse
import tempfile
import ssl
import time

ssl._create_default_https_context = ssl._create_unverified_context

# ScanNet download constants (from official script)
BASE_URL = "http://kaldir.vc.cit.tum.de/scannet/"
RELEASE = "v2/scans"
RELEASE_V1 = "v1/scans"
RELEASE_TASKS = "v2/tasks"
LABEL_MAP_FILE = "scannetv2-labels.combined.tsv"

# Import our config
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scannet.config import SCANNET_VAL_SCENES, ANNOTATION_FILETYPES, SENS_FILETYPE, SCANNET_SCANS, LABEL_MAP_FILE as LABEL_MAP_PATH


def download_file(url, out_file, retries=3):
    """Download a file with retry logic using wget for progress display."""
    import subprocess
    out_dir = os.path.dirname(out_file)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    if os.path.isfile(out_file):
        print(f"  [skip] {os.path.basename(out_file)} (exists)")
        return True

    for attempt in range(retries):
        try:
            # Use HTTPS URL (server redirects HTTP -> HTTPS)
            https_url = url.replace("http://", "https://")
            print(f"  [download] {https_url}")
            print(f"         -> {out_file}")
            fh, tmp = tempfile.mkstemp(dir=out_dir)
            os.close(fh)
            # Use wget with progress bar for large files
            result = subprocess.run(
                ["wget", "-q", "--show-progress", "--no-check-certificate",
                 "-O", tmp, https_url],
                timeout=7200,  # 2 hour timeout per file
            )
            if result.returncode != 0:
                raise RuntimeError(f"wget failed with code {result.returncode}")
            os.rename(tmp, out_file)
            size_mb = os.path.getsize(out_file) / (1024 * 1024)
            print(f"  [done] {size_mb:.1f} MB")
            return True
        except Exception as e:
            print(f"  [ERROR] Attempt {attempt+1}/{retries}: {e}")
            if os.path.exists(tmp):
                os.remove(tmp)
            if attempt < retries - 1:
                time.sleep(5)
    return False


def download_scan_files(scan_id, out_dir, file_types):
    """Download specific file types for a single scan."""
    print(f"\n{'='*60}")
    print(f"Scene: {scan_id}")
    print(f"{'='*60}")
    
    scan_out = os.path.join(out_dir, scan_id)
    if not os.path.isdir(scan_out):
        os.makedirs(scan_out)

    success = True
    for ft in file_types:
        # ScanNet v2 uses v1 .sens files
        if ft == ".sens":
            url = f"{BASE_URL}{RELEASE_V1}/{scan_id}/{scan_id}{ft}"
        else:
            url = f"{BASE_URL}{RELEASE}/{scan_id}/{scan_id}{ft}"
        
        out_file = os.path.join(scan_out, f"{scan_id}{ft}")
        if not download_file(url, out_file):
            success = False
    return success


def download_label_map():
    """Download the label mapping TSV file."""
    print("\nDownloading label map...")
    url = f"{BASE_URL}{RELEASE_TASKS}/{LABEL_MAP_FILE}"
    out_file = str(LABEL_MAP_PATH)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    return download_file(url, out_file)


def main():
    parser = argparse.ArgumentParser(description="Batch ScanNet downloader")
    parser.add_argument("--annotations", action="store_true",
                        help="Download annotation files (small)")
    parser.add_argument("--sens", action="store_true",
                        help="Download .sens files (large)")
    parser.add_argument("--label-map", action="store_true",
                        help="Download label mapping file")
    parser.add_argument("--all", action="store_true",
                        help="Download everything")
    parser.add_argument("--scenes", type=str, default=None,
                        help="Comma-separated scene IDs (default: all 20 val)")
    args = parser.parse_args()

    if not any([args.annotations, args.sens, args.label_map, args.all]):
        parser.print_help()
        sys.exit(1)

    scenes = SCANNET_VAL_SCENES
    if args.scenes:
        scenes = [s.strip() for s in args.scenes.split(",")]

    out_dir = str(SCANNET_SCANS)
    os.makedirs(out_dir, exist_ok=True)

    # Label map
    if args.label_map or args.all:
        download_label_map()

    # Annotations (small, fast)
    if args.annotations or args.all:
        print(f"\n{'#'*60}")
        print(f"Downloading annotations for {len(scenes)} scenes")
        print(f"{'#'*60}")
        for i, scene_id in enumerate(scenes):
            print(f"\n[{i+1}/{len(scenes)}]", end="")
            download_scan_files(scene_id, out_dir, ANNOTATION_FILETYPES)

    # .sens files (large, slow)
    if args.sens or args.all:
        print(f"\n{'#'*60}")
        print(f"Downloading .sens files for {len(scenes)} scenes")
        print(f"WARNING: Each .sens file is ~2-3 GB. Total: ~40-60 GB")
        print(f"{'#'*60}")
        for i, scene_id in enumerate(scenes):
            print(f"\n[{i+1}/{len(scenes)}]", end="")
            download_scan_files(scene_id, out_dir, SENS_FILETYPE)

    print("\n" + "=" * 60)
    print("Download complete!")
    
    # Summary
    total_files = 0
    total_size = 0
    for scene_id in scenes:
        scene_dir = os.path.join(out_dir, scene_id)
        if os.path.isdir(scene_dir):
            for f in os.listdir(scene_dir):
                fp = os.path.join(scene_dir, f)
                if os.path.isfile(fp):
                    total_files += 1
                    total_size += os.path.getsize(fp)
    print(f"Total: {total_files} files, {total_size / (1024**3):.2f} GB")


if __name__ == "__main__":
    main()
