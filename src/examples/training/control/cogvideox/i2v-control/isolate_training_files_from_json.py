#!/usr/bin/env python3
"""
json_dataset_parser.py   (clean-first version)

Parse the KeyMotion JSON, generate prompts.txt / images.txt / sketches.txt /
videos.txt, and copy the assets into images/, sketches/, videos/ sub-folders.
If the destination folder already exists, it is deleted first.
"""

import argparse
import json
import os
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", help="Path to the JSON file you showed above", \
        default="/gpfs/junlab/wangwanding24/KeyMotion/finetrainers/examples/training/control/cogvideox/i2v-control/validation_ft_cheat.json")
    parser.add_argument("--dest_dir", help="Destination folder that will be CREATED", \
        default="/gpfs/junlab/wangwanding24/KeyMotion/dataset/finetrainers_validating_from_train")
    args = parser.parse_args()

    dest_root = Path(args.dest_dir).resolve()

    # ------------------------------------------------------------------ #
    #                NEW: clean destination before proceeding            #
    # ------------------------------------------------------------------ #
    if dest_root.exists():
        print(f"[INFO] Removing previous contents of {dest_root} …")
        shutil.rmtree(dest_root)

    # Now (re-)create the folder tree
    (dest_root / "images").mkdir(parents=True, exist_ok=True)
    (dest_root / "sketches").mkdir(exist_ok=True)
    (dest_root / "videos").mkdir(exist_ok=True)

    # ------------------- load JSON and prepare I/O files --------------- #
    with open(args.json_file, "r", encoding="utf-8") as f:
        data = json.load(f)["data"]

    prompts_f  = (dest_root / "prompts.txt").open("w", encoding="utf-8")
    images_f   = (dest_root / "images.txt").open("w", encoding="utf-8")
    sketches_f = (dest_root / "sketches.txt").open("w", encoding="utf-8")
    videos_f   = (dest_root / "videos.txt").open("w", encoding="utf-8")

    # -------------------------- main loop ------------------------------ #
    for item in data:
        # 1. captions
        print(item["caption"].strip(), file=prompts_f)

        # 2. images
        src_img_path = Path(item["image_path"]).resolve()
        img_name_start = src_img_path.name                      # …Key_Start.png
        img_name_end   = img_name_start.replace("Key_Start", "Key_End")

        print(img_name_start, file=images_f)
        print(img_name_end,   file=images_f)

        shutil.copy2(src_img_path, dest_root / "images" / img_name_start)
        end_src = src_img_path.with_name(img_name_end)
        if end_src.exists():
            shutil.copy2(end_src, dest_root / "images" / img_name_end)

        # 3. sketches
        src_sketch_path = Path(item["control_video_path"]).resolve()
        sketch_name = src_sketch_path.name                      # *_Crop_sketch.mp4
        print(sketch_name, file=sketches_f)
        shutil.copy2(src_sketch_path, dest_root / "sketches" / sketch_name)

        # 4. ground-truth videos
        video_name = sketch_name.replace("_sketch", "")         # *_Crop.mp4
        print(video_name, file=videos_f)

        src_video_path = src_sketch_path.with_name(video_name).with_suffix(".mp4")
        src_video_path = Path(str(src_video_path).replace("/sketches/", "/videos/"))
        if src_video_path.exists():
            shutil.copy2(src_video_path, dest_root / "videos" / video_name)
        else:
            print(f"[WARN] Ground-truth video missing: {src_video_path}")

    # --------------------------- tidy up ------------------------------- #
    for fh in (prompts_f, images_f, sketches_f, videos_f):
        fh.close()

    print(f"✓ Done. Outputs written under {dest_root}")


if __name__ == "__main__":
    main()
