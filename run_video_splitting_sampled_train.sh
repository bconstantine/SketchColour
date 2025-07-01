CUDA_DEVICES="2"
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

python preprocess/preprocess_video_split_sampled_trim.py \
    --input_dir_parquet '../dataset/sampled_train/parquet' \
    --input_dir_video '../dataset/sampled_train/download' \
    --split_mode 'cached_skip' \
    --frame_number 17 \
    --output_dir_clip '../dataset/sampled_train/split' \
    --output_dir_keyframe '../dataset/sampled_train/keyframe' \
    --clean_existing_videos