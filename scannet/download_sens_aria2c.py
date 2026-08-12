#!/usr/bin/env python3
"""
Download .sens files using aria2c for faster multi-segment downloads.
aria2c supports multiple connections per file, dramatically improving speed.

Usage:
    python scannet/download_sens_aria2c.py
"""

import json
import subprocess
from pathlib import Path

SCENES_FILE = Path(__file__).parent / "data" / "new_92_scenes.json"
SCANS_DIR = Path(__file__).parent / "data" / "scans"
BASE_URL = "https://kaldir.vc.cit.tum.de/scannet/v1/scans"
MIN_SENS_SIZE = 10 * 1024 * 1024


def main():
    with open(SCENES_FILE) as f:
        scenes = json.load(f)["new_scenes"]
    
    # Filter to ones that still need downloading
    to_download = []
    already = 0
    for s in scenes:
        sf = SCANS_DIR / s / f"{s}.sens"
        if sf.exists() and sf.stat().st_size > MIN_SENS_SIZE:
            already += 1
        else:
            # Remove partial downloads
            if sf.exists():
                sf.unlink()
            to_download.append(s)
    
    print(f"Already have: {already}/92")
    print(f"To download: {len(to_download)}")
    
    if not to_download:
        print("Nothing to download!")
        return
    
    # Create aria2c input file with all URLs
    input_file = SCANS_DIR.parent / "aria2c_input.txt"
    with open(input_file, "w") as f:
        for s in to_download:
            url = f"{BASE_URL}/{s}/{s}.sens"
            out_dir = SCANS_DIR / s
            out_dir.mkdir(parents=True, exist_ok=True)
            f.write(f"{url}\n")
            f.write(f"  dir={out_dir}\n")
            f.write(f"  out={s}.sens\n")
    
    print(f"Created input file: {input_file}")
    print(f"Starting aria2c with 8 connections per file, 4 concurrent downloads...")
    
    # Run aria2c with optimized settings
    cmd = [
        "aria2c",
        "--input-file", str(input_file),
        "--max-concurrent-downloads=4",
        "--split=8",               # 8 segments per file
        "--max-connection-per-server=8",
        "--min-split-size=10M",
        "--check-certificate=false",
        "--continue=true",         # Resume support
        "--auto-file-renaming=false",
        "--console-log-level=notice",
        "--summary-interval=30",
        "--retry-wait=5",
        "--max-tries=5",
    ]
    
    subprocess.run(cmd)
    
    # Verify
    final_count = sum(1 for s in scenes 
                     if (SCANS_DIR / s / f"{s}.sens").exists() 
                     and (SCANS_DIR / s / f"{s}.sens").stat().st_size > MIN_SENS_SIZE)
    print(f"\nFinal: {final_count}/92 scenes downloaded")


if __name__ == "__main__":
    main()
