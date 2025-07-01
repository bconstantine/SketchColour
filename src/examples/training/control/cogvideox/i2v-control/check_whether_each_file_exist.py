#!/usr/bin/env python3
"""
check_dataset_files.py

Usage
-----
# A) If you copy-paste the JSON directly into this file (see `sample_json` below):
python check_dataset_files.py

# B) If your JSON is saved in a separate file, e.g. dataset.json:
python check_dataset_files.py --json_file dataset.json
"""

import json
import os
import argparse
from pathlib import Path
from typing import List, Dict

# ----------------------------------------------------------------------
# Optional: paste your JSON here so you can run without an external file.
#           Comment out or delete `sample_json` if you prefer --json_file.
# ----------------------------------------------------------------------
sample_json = r'''
{
  "data": [
    {
      "caption": "In the midst of a tunnel, ...",
      "image_path": "/gpfs/junlab/wangwanding24/KeyMotion/dataset/finetrainers_val/images/16760-Scene-4-Key_Start.png",
      "control_video_path": "/gpfs/junlab/wangwanding24/KeyMotion/dataset/finetrainers_val/sketches/16760-Scene-4-Crop_sketch.mp4"
    },
    {
      "caption": "The scene opens with a woman running ...",
      "image_path": "/gpfs/junlab/wangwanding24/KeyMotion/dataset/finetrainers_val/images/16782-Scene-1-Key_Start.png",
      "control_video_path": "/gpfs/junlab/wangwanding24/KeyMotion/dataset/finetrainers_val/sketches/16782-Scene-1-Crop_sketch.mp4"
    }
    /* … rest of your items … */
  ]
}
'''

# ----------------------------------------------------------------------
def load_json(path: str = None) -> Dict:
    """Load JSON from a file if `path` is given, otherwise use `sample_json`."""
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(sample_json)

def check_paths(entries: List[Dict]) -> None:
    """Check existence of image and video paths and print a small report."""
    missing: List[str] = []
    print(f"Checking {len(entries)} dataset entries …")

    for idx, item in enumerate(entries, start=1):
        img = Path(item["image_path"])
        vid = Path(item["control_video_path"])

        img_exists = img.exists()
        vid_exists = vid.exists()

        status = "OK" if (img_exists and vid_exists) else "MISSING"
        print(f"[{idx:02d}] {status}: {img.name} / {vid.name}")

        if not img_exists:
            missing.append(str(img))
        if not vid_exists:
            missing.append(str(vid))

    # Summary
    if missing:
        print("\nSome paths are missing:")
        for p in missing:
            print("  -", p)
    else:
        print("\nAll files are present ✔")

def main() -> None:
    parser = argparse.ArgumentParser(description="Check dataset file paths")
    parser.add_argument(
        "--json_file",
        type=str,
        default=None,
        help="Path to JSON file (defaults to built-in `sample_json`)",
    )
    args = parser.parse_args()

    data_dict = load_json(args.json_file)
    if "data" not in data_dict or not isinstance(data_dict["data"], list):
        raise ValueError("JSON must contain a top-level `data` list")

    check_paths(data_dict["data"])

if __name__ == "__main__":
    main()