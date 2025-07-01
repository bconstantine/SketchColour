import os
import argparse
import cv2
from tqdm import tqdm
import pandas as pd
from scenedetect import split_video_ffmpeg, FrameTimecode
from scenedetect import detect, AdaptiveDetector
from collections import defaultdict
from tqdm import tqdm
import shutil
import multiprocessing
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
import subprocess
import math

def resize_and_crop(img, target_w=720, target_h=480):
    """
    Return an image with exact size (target_h, target_w).
    • If the input is larger on both axes → centre-crop.
    • Otherwise → scale up so the *smaller* axis fits, then centre-crop.
    Keeps aspect-ratio; uses bilinear interpolation for up/down-scaling.
    """
    h, w = img.shape[:2]
    if w == target_w and h == target_h:
        return img

    # ----- 1) scale (if needed) without distorting aspect ---------------
    scale = max(target_w / w, target_h / h)      # ≥ 1 when up-scaling
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # ----- 2) centre-crop to target size --------------------------------
    x0 = (new_w - target_w) // 2
    y0 = (new_h - target_h) // 2
    return img_resized[y0:y0 + target_h, x0:x0 + target_w]

def get_fps(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        print("Error: Could not open video.")
        return
    
    # Get frames per second from the video file
    fps = video.get(cv2.CAP_PROP_FPS)
    
    # Release the video capture object
    video.release()
    
    return fps

def delete_folder_contents(folder_path, spare_gitkeep= True):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            if filename == '.gitkeep' and spare_gitkeep:
                continue
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
        print("All contents deleted.")

def detect_crop_worker(task):
    """
    Pick a random max_frames window from `path_clip`, normalise every
    frame to 480×720, save key-frames, and emit an H.264 clip.
    """
    path_clip, path_keyframe, max_frames, args = task
    os.makedirs(os.path.dirname(path_keyframe), exist_ok=True)

    # 1) open input video
    cap = cv2.VideoCapture(path_clip)
    if not cap.isOpened():
        print(f"Error: Could not open input video {path_clip}")
        return

    # 2) read video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or math.isnan(fps):
        fps = 30.0

    # --- CogVideoX target size ---
    target_w, target_h = args.target_width, args.target_height          # (w, h)

    # 3) choose window
    if total_frames > max_frames:
        start_idx = random.randint(0, total_frames - max_frames)
        end_idx   = start_idx + max_frames - 1
    else:
        start_idx, end_idx = 0, total_frames - 1

    base, _ = os.path.splitext(path_keyframe)
    temp_avi = f"{base}-temp.avi"

    # 4) MJPG writer with exact target size
    writer = cv2.VideoWriter(
        temp_avi,
        cv2.VideoWriter_fourcc(*'MJPG'),
        fps,
        (target_w, target_h),
        True
    )
    if not writer.isOpened():
        print(f"Error: Could not open temporary AVI writer for {temp_avi}")
        cap.release()
        return

    # 5) iterate over the selected window
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    for fid in range(start_idx, end_idx + 1):
        ret, frame = cap.read()
        if not ret:
            break
        frame_proc = resize_and_crop(frame, target_w, target_h)
        writer.write(frame_proc)

        if fid == start_idx:
            cv2.imwrite(f"{base}-Key_Start.png", frame_proc)
        if fid == end_idx:
            cv2.imwrite(f"{base}-Key_End.png", frame_proc)

    writer.release()
    cap.release()

    # 6) re-encode to H.264 MP4 (silent)
    mp4_out = f"{base}-Crop.mp4"
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-i', temp_avi,
        '-c:v', 'libx264',
        '-profile:v', 'baseline', '-level', '3.0',
        '-pix_fmt', 'yuv420p', '-movflags', 'faststart',
        mp4_out
    ]
    subprocess.run(ffmpeg_cmd,
                   check=True,
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)

    os.remove(temp_avi)


def run_parallel_crop(args, max_frames, num_workers, seed=42):
    """
    Run detect_crop_worker over tasks in parallel with a video‐level progress bar.
    
    Args:
        args:                Namespace with `output_dir_clip` and `output_dir_keyframe`.
        max_frames (int):    Number of frames to crop.
        num_workers (int):   Number of parallel worker processes.
        seed (int, optional): Random seed for reproducibility.
    """
    # 1) seed RNG
    if seed is not None:
        random.seed(seed)

    # 2) build task list
    tasks = []
    for video in os.listdir(args.output_dir_clip):
        if video.startswith('.'):  # skip hidden files like .gitkeep
            continue
        clip_folder = os.path.join(args.output_dir_clip, video)
        for clip in os.listdir(clip_folder):
            path_clip = os.path.join(clip_folder, clip)
            # extract scene token from filename
            scene = os.path.splitext(clip)[0].split('-')[2]
            path_keyframe = os.path.join(args.output_dir_keyframe, video, scene, clip)

            # skip if we've already produced a -Crop-*.mp4 in that folder
            base = os.path.splitext(path_keyframe)[0]
            out_dir = os.path.dirname(path_keyframe)
            if os.path.isdir(out_dir):
                for f in os.listdir(out_dir):
                    if f.startswith(os.path.basename(base) + "-Crop") and f.endswith(".mp4"):
                        break
                else:
                    tasks.append((path_clip, path_keyframe, max_frames, args))
            else:
                tasks.append((path_clip, path_keyframe, max_frames, args))

    if not tasks:
        print("✅ All videos have already been processed.")
        return

    # 3) process with a video‐level tqdm bar
    with Pool(processes=num_workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(detect_crop_worker, tasks),
            total=len(tasks),
            desc="Processing videos",
            unit="video"
        ):
            pass


def fetch_boundaries(df):
    df['identifier_video'] = df['identifier'].apply(lambda x: int(x.split(':')[0]))
    df['identifier_clip'] = df['identifier'].apply(lambda x: int(x.split(':')[1]))
    df = df.sort_values(by=['identifier_video', 'identifier_clip'])
    
    boundaries = defaultdict(dict)
    for i in range(len(df)):
        row = df.iloc[i]
        identifier_video = row['identifier_video']
        identifier_clip = row['identifier_clip']
        start_time = row['scene_start_time']
        end_time = row['scene_end_time']
        fps = row['fps']
        
        boundaries[identifier_video][identifier_clip] = (FrameTimecode(start_time, fps), FrameTimecode(end_time, fps))
    
    return boundaries

def fetch_metadata_for_cache_creation(df):
    df['identifier_video'] = df['identifier'].apply(lambda x: int(x.split(':')[0]))
    df['identifier_clip'] = df['identifier'].apply(lambda x: int(x.split(':')[1]))
    df = df.sort_values(by=['identifier_video', 'identifier_clip'])

    identifier_video_to_data = {}
    for i in range(len(df)):
        row = df.iloc[i]
        identifier_video_to_data[row['identifier_video']] = {"fps": row["fps"]}
    return identifier_video_to_data

def _process_single_video(video,
                          input_dir_video,
                          output_dir_clip,
                          split_mode,
                          boundaries,
                          fps_dict,
                          args):
    if video.endswith('.gitkeep'):
        return
    vid_id = os.path.splitext(video)[0]
    src = os.path.join(input_dir_video, video)
    dst_dir = os.path.join(output_dir_clip, vid_id)
    os.makedirs(dst_dir, exist_ok=True)

    # Decide which splitting logic to run
    if split_mode == 'cached_skip':
        b = boundaries.get(int(vid_id))
        if not b:
            return
        if len(os.listdir(dst_dir)) != len(b):
            for scene_number in b:
                split_video_ffmpeg(src, [b[scene_number]],
                    output_file_template=f'{dst_dir}/$VIDEO_NAME-Scene-{scene_number}.mp4')

    elif split_mode == 'cached_split':
        b = boundaries.get(int(vid_id))
        if not b:
            detect_boundary_and_do_splits(src, dst_dir, args)
        elif len(os.listdir(dst_dir)) != len(b):
            split_video_ffmpeg(src, b,
                output_file_template=f'{dst_dir}/$VIDEO_NAME-Scene-$SCENE_NUMBER.mp4')

    elif split_mode == 'split_save':
        fps = fps_dict.get(int(vid_id), {}).get("fps")
        if fps is None:
            fps = get_fps(src)
        detect_boundary_and_do_splits(src, dst_dir, fps, args)


def split_video_from_parquet(args):
    # loop over each parquet, build metadata
    for pq in os.listdir(args.input_dir_parquet):
        if not pq.endswith('.parquet'):
            continue
        print(f'Processing {pq}')
        df = pd.read_parquet(os.path.join(args.input_dir_parquet, pq))
        df = df[['identifier','scene_start_time','scene_end_time','fps']]

        if args.split_mode in ('cached_skip','cached_split'):
            df_meta = df.dropna(subset=['fps'] if args.split_mode=='cached_split'
                                 else ['scene_start_time','scene_end_time','fps'])
            boundaries = fetch_boundaries(df_meta)
            fps_dict = {}
        else:  # split_save
            df_meta = df.dropna(subset=['fps'])
            boundaries = {}
            fps_dict = fetch_metadata_for_cache_creation(df_meta)

        videos = os.listdir(args.input_dir_video)
        with ProcessPoolExecutor() as exe:
            futures = {
                exe.submit(_process_single_video,
                           video,
                           args.input_dir_video,
                           args.output_dir_clip,
                           args.split_mode,
                           boundaries,
                           fps_dict,
                           args): video
                for video in videos
            }
            for _ in tqdm(as_completed(futures), total=len(futures),
                          desc="Splitting videos"):
                pass


def detect_boundary_and_do_splits(path_video, clip_dir_save, video_fps, args):
    boundary =  detect(path_video, 
                       AdaptiveDetector(adaptive_threshold=args.detector_adaptive_threshold, 
                                                               min_scene_len=args.detector_min_scene_len, 
                                                               min_content_val=args.detector_min_content_val), 
                        start_in_scene=True)
    # boundary =  detect(path_video, AdaptiveDetector())
    if len(boundary):
        if args.enforce_max_content_len and args.enforce_max_content_len > 0:
            new_boundary = []
            for subvideo_idx in range(len(boundary)):
                start_frame = boundary[subvideo_idx][0].frame_num
                end_frame = boundary[subvideo_idx][1].frame_num
                while start_frame < end_frame:
                    next_endframe = start_frame + args.enforce_max_content_len -1
                    if next_endframe > end_frame:
                        break
                    start_frametimecode = FrameTimecode(start_frame, fps=video_fps)
                    end_frametimecode = FrameTimecode(next_endframe, fps=video_fps)
                    new_boundary.append((start_frametimecode, end_frametimecode))
                    start_frame = next_endframe+1
            boundary = new_boundary
    if len(boundary) and len(os.listdir(clip_dir_save)) != len(boundary):  # if the video is already split, skip
        split_video_ffmpeg(path_video, boundary, output_file_template=f'{clip_dir_save}/$VIDEO_NAME-Scene-$SCENE_NUMBER.mp4')
        

    return boundary


def clean_existing_videos(args):
    if args.clean_existing_videos:
        delete_folder_contents(args.output_dir_clip)
        delete_folder_contents(args.output_dir_keyframe)


def main(args):
    multiprocessing.set_start_method('spawn')  # Set spawn method
    clean_existing_videos(args)
    split_video_from_parquet(args)
    os.makedirs(args.output_dir_keyframe, exist_ok=True)
    run_parallel_crop(args, max_frames=args.frame_number, num_workers=4, seed=42)
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir_parquet', type=str, default=r'dataset/val/parquet', help='Directory to the parquet files')
    parser.add_argument('--input_dir_video', type=str, default=r'dataset/val/download', help='Directory to the downloaded videos')
    parser.add_argument('--split_mode', type=str, default='cached_skip', choices={"cached_skip", "cached_split", "split_save"}, help='use cached duration from the parquet file or run split')
    parser.add_argument('--frame_number', type=int, default=None, help="Enforce maximum content length in frames", required=True)
    parser.add_argument('--clean_existing_videos', action='store_true', help='clean existing videos in the output directory')
    parser.add_argument('--output_dir_clip', type=str, default=r'dataset/val/split', help='Directory to save the split videos')
    parser.add_argument('--output_dir_keyframe', type=str, default=r'dataset/val/keyframe')
    parser.add_argument(
        '--target_width', type=int, default=720,
        help='If set, width (in px) of the centre-crop written to disk')
    parser.add_argument(
        '--target_height', type=int, default=480,
        help='If set, height (in px) of the centre-crop written to disk')
    args = parser.parse_args()
    
    main(args)
