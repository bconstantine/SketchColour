CUDA_DEVICES="2"
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

python preprocess/preprocess_video_split_sampled_trim.py \
    --input_dir_parquet '../dataset/sampled_val/parquet' \
    --input_dir_video '../dataset/sampled_val/download' \
    --split_mode 'cached_skip' \
    --frame_number 17 \
    --output_dir_clip '../dataset/sampled_val/split' \
    --output_dir_keyframe '../dataset/sampled_val/keyframe' \
    --clean_existing_videos