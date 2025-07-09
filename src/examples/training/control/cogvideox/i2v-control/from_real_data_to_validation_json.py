#!/usr/bin/env python3
import argparse, json, os, random, sys

def load_every_other_line(path):
    """Read all lines, strip them, and return only the odd-indexed ones (0,2,4, …)."""
    with open(path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    return lines[::2]

def load_lines(path):
    """Read all non-empty, stripped lines."""
    with open(path, 'r', encoding='utf-8') as f:
        return [l.strip() for l in f if l.strip()]

def main():
    p = argparse.ArgumentParser(
        description="Sample n entries from a paired dataset and emit JSON.")
    p.add_argument('--images-dir',      required=True, help="Directory of starting images")
    p.add_argument('--sketches-dir',    required=True, help="Directory of sketch videos")
    p.add_argument('--images-txt',      required=True, help="Path to images.txt")
    p.add_argument('--prompts-txt',     required=True, help="Path to prompts.txt")
    p.add_argument('--sketches-txt',    required=True, help="Path to sketches.txt")
    p.add_argument('--num-samples',     type=int, required=True, help="How many entries to sample")
    p.add_argument('--num-steps',       type=int, default=50, help="num_inference_steps")
    p.add_argument('--height',          type=int, default=480, help="Fixed frame height")
    p.add_argument('--width',           type=int, default=720, help="Fixed frame width")
    p.add_argument('--num-frames',      type=int, default=17, help="Fixed number of frames")
    p.add_argument('--frame-rate',      type=int, default=25, help="Fixed frame rate")
    p.add_argument('--output',          help="If set, write JSON here instead of stdout")
    args = p.parse_args()

    images = load_lines(args.images_txt)
    prompts = load_lines(args.prompts_txt)
    sketches = load_lines(args.sketches_txt)

    if not (len(images) == len(prompts) == len(sketches)):
        sys.exit(f"Error: mismatched counts: images={len(images)}, prompts={len(prompts)}, sketches={len(sketches)}")

    if args.num_samples > len(prompts):
        sys.exit(f"Error: num-samples ({args.num_samples}) > available entries ({len(prompts)})")

    chosen = random.sample(range(len(prompts)), args.num_samples)
    data = []
    for idx in chosen:
        img_fn   = images[idx]
        skt_fn   = sketches[idx]
        prompt   = prompts[idx]

        # build full paths
        img_path = os.path.join(args.images_dir, img_fn)
        vid_path = os.path.join(args.sketches_dir, skt_fn)

        data.append({
            "caption": prompt,
            "image_path": img_path,
            "control_video_path": vid_path,
            "num_inference_steps": args.num_steps,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "frame_rate": args.frame_rate
        })

    out = {"data": data}
    js = json.dumps(out, indent=2, ensure_ascii=False)

    if args.output:
        print("outputing done!")
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(js)
    else:
        print(js)

if __name__ == '__main__':
    main()