CUDA_DEVICES="0"
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

python preprocess/preprocess_keyframe_sketch.py \
    --dataroot '../dataset/sampled_test/keyframe' \
    --output_folder '../dataset/sampled_test/sketch'\
    --clean_existing_sketches \
    --input_type 'video' \
    --gpu_ids $CUDA_DEVICES \
    --binarize_threshold 250