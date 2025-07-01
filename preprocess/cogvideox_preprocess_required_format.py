import os
import shutil
import argparse
import pandas as pd
import subprocess
from tqdm import tqdm


def clean_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    if os.path.exists(path):
        print(f"Directory {path} cleaned")
    else:
        print(f"Failed to clean directory {path}")


def process_split(input_dir, split_name, output_dir, parquet_filename, clean=False):
    # Paths
    keyframe_dir = os.path.join(input_dir, 'keyframe')
    parquet_path = os.path.join(input_dir, 'parquet', parquet_filename)

    # Output structure
    out_base = output_dir
    out_videos = os.path.join(out_base, 'videos')
    out_images = os.path.join(out_base, 'images')

    # Clean output if requested
    if clean:
        clean_dir(out_base)
        os.makedirs(out_videos, exist_ok=True)
        os.makedirs(out_images, exist_ok=True)
    else:
        os.makedirs(out_videos, exist_ok=True)
        os.makedirs(out_images, exist_ok=True)

    # Load prompts
    df = pd.read_parquet(parquet_path)
    # Expect columns: 'identifier' ("{video_name}:{scene_number}"), and 'prompt'
    prompt_map = dict(zip(df['identifier'], df['text_description']))

    # Files to list
    video_files = []
    image_files = []

    # Build output lists
    prompts = []

    # Iterate identifiers
    for identifier, prompt in tqdm(prompt_map.items()):
        video_name, scene_number = identifier.split(':')
        src_folder = os.path.join(keyframe_dir, video_name, scene_number)
        # Crop video
        crop_name = f"{video_name}-Scene-{scene_number}-Crop.mp4"
        src_crop = os.path.join(src_folder, crop_name)        
        # Start keyframe (PNG)
        start_img = f"{video_name}-Scene-{scene_number}-Key_Start.png"
        src_start = os.path.join(src_folder, start_img)
        # End keyframe (PNG)
        end_img = f"{video_name}-Scene-{scene_number}-Key_End.png"
        src_end = os.path.join(src_folder, end_img)

        
        if prompt is None or not (os.path.exists(src_crop) and os.path.exists(src_start) and os.path.exists(src_end)): 
            continue

        dst_crop = os.path.join(out_videos, crop_name)
        shutil.copy2(src_crop, dst_crop)
        video_files.append(crop_name)

        dst_start = os.path.join(out_images, start_img)
        shutil.copy2(src_start, dst_start)
        image_files.append(start_img)

        dst_end = os.path.join(out_images, end_img)
        shutil.copy2(src_end, dst_end)
        image_files.append(end_img)

        # Collect prompt
        prompts.append(prompt)

    # Write prompts.txt
    with open(os.path.join(out_base, 'prompts.txt'), 'w', encoding='utf-8') as f:
        for p in prompts:
            if p is not None:
                f.write(p.replace('\n', ' ') + '\n')
            else:
                f.write('\n')

    # Write videos.txt
    with open(os.path.join(out_base, 'videos.txt'), 'w') as f:
        for vf in video_files:
            f.write(vf + '\n')

    # Write images.txt
    with open(os.path.join(out_base, 'images.txt'), 'w') as f:
        for imf in image_files:
            f.write(imf + '\n')

def process_split_with_sketch(input_dir, split_name, output_dir, parquet_filename, clean=False):
    # Paths
    keyframe_dir = os.path.join(input_dir, 'keyframe')
    parquet_path = os.path.join(input_dir, 'parquet', parquet_filename)
    sketch_dir = os.path.join(input_dir, "sketch")

    # Output structure
    out_base = output_dir
    out_videos = os.path.join(out_base, 'videos')
    out_images = os.path.join(out_base, 'images')
    out_sketches = os.path.join(out_base, "sketches")

    # Clean output if requested
    if clean:
        clean_dir(out_base)
        os.makedirs(out_videos, exist_ok=True)
        os.makedirs(out_images, exist_ok=True)
        os.makedirs(out_sketches, exist_ok=True)
    else:
        os.makedirs(out_videos, exist_ok=True)
        os.makedirs(out_images, exist_ok=True)
        os.makedirs(out_sketches, exist_ok=True)

    # Load prompts
    df = pd.read_parquet(parquet_path)
    # Expect columns: 'identifier' ("{video_name}:{scene_number}"), and 'prompt'
    prompt_map = dict(zip(df['identifier'], df['text_description']))

    # Files to list
    video_files = []
    image_files = []
    sketch_files = []

    # Build output lists
    prompts = []

    # Iterate identifiers
    for identifier, prompt in tqdm(prompt_map.items()):
        video_name, scene_number = identifier.split(':')
        src_folder = os.path.join(keyframe_dir, video_name, scene_number)
        sketch_src_folder = os.path.join(sketch_dir, video_name, scene_number)
        # Crop video
        crop_name = f"{video_name}-Scene-{scene_number}-Crop.mp4"
        src_crop = os.path.join(src_folder, crop_name)        
        # Start keyframe (PNG)
        start_img = f"{video_name}-Scene-{scene_number}-Key_Start.png"
        src_start = os.path.join(src_folder, start_img)
        # End keyframe (PNG)
        end_img = f"{video_name}-Scene-{scene_number}-Key_End.png"
        src_end = os.path.join(src_folder, end_img)
        #sketches MP4
        sketch_name = f"{video_name}-Scene-{scene_number}-Crop_sketch.mp4"
        src_sketch = os.path.join(sketch_src_folder, sketch_name) 

        
        if prompt is None or not (os.path.exists(src_crop) and \
                                  os.path.exists(src_start) and \
                                    os.path.exists(src_end) and \
                                        os.path.exists(src_sketch)): 
            continue

        dst_crop = os.path.join(out_videos, crop_name)
        shutil.copy2(src_crop, dst_crop)
        video_files.append(crop_name)

        dst_start = os.path.join(out_images, start_img)
        shutil.copy2(src_start, dst_start)
        image_files.append(start_img)

        dst_end = os.path.join(out_images, end_img)
        shutil.copy2(src_end, dst_end)
        image_files.append(end_img)

        dst_sketch = os.path.join(out_sketches, sketch_name)
        shutil.copy2(src_sketch, dst_sketch)
        sketch_files.append(sketch_name)

        # Collect prompt
        prompts.append(prompt)

    # Write prompts.txt
    with open(os.path.join(out_base, 'prompts.txt'), 'w', encoding='utf-8') as f:
        for p in prompts:
            if p is not None:
                f.write(p.replace('\n', ' ') + '\n')
            else:
                f.write('\n')

    # Write videos.txt
    with open(os.path.join(out_base, 'videos.txt'), 'w') as f:
        for vf in video_files:
            f.write(vf + '\n')

    # Write images.txt
    with open(os.path.join(out_base, 'images.txt'), 'w') as f:
        for imf in image_files:
            f.write(imf + '\n')
    with open(os.path.join(out_base, 'sketches.txt'), 'w') as f:
        for skt in sketch_files:
            f.write(skt + '\n')

def main():
    parser = argparse.ArgumentParser(description='Prepare Sakuga Dataset for training and validation')
    parser.add_argument('--train_input', default='dataset/sampled_train_80', help='Input directory for training split')
    parser.add_argument('--val_input', default='dataset/sampled_val', help='Input directory for validation split')
    parser.add_argument('--train_output', 
                        #default='dataset/cogvideox_train', 
                        default='dataset/finetrainers_train_80', 
                        help='Output directory for training')
    parser.add_argument('--val_output', 
                        #default='dataset/cogvideox_val', 
                        default='dataset/finetrainers_val', 
                        help='Output directory for validation')
    parser.add_argument('--train_parquet_filename', default='train_80000_16_90.parquet')
    parser.add_argument('--val_parquet_filename', default='val_1000_16_32.parquet')
    parser.add_argument('--clean', action='store_true', help='Clean output folders before generating')
    args = parser.parse_args()

    process_split_with_sketch(args.train_input, 'train', args.train_output, args.train_parquet_filename, clean=args.clean)
    #process_split_with_sketch(args.val_input, 'val', args.val_output, args.val_parquet_filename, clean=args.clean)

if __name__ == '__main__':
    main()
