#!/bin/bash

set -e -x

export WANDB_MODE="offline"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export FINETRAINERS_LOG_LEVEL="INFO"
export TORCH_LOGS="recompiles"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

#Activate attn checks
export FINETRAINERS_ATTN_CHECKS="1"


# Finetrainers supports multiple backends for distributed training. Select your favourite and benchmark the differences!
BACKEND="ptd"

# In this setting, I'm using 1 GPU on 4-GPU node for training
NUM_GPUS=2
CUDA_VISIBLE_DEVICES="0,9"



# Check the JSON files for the expected JSON format
TRAINING_DATASET_CONFIG="./src/examples/training/control/cogvideox/i2v-control/training.json"
VALIDATION_DATASET_FILE="./src/examples/training/control/cogvideox/i2v-control/example_validation.json" 

# Depending on how many GPUs you have available, choose your degree of parallelism and technique!
DDP_1="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 1 --dp_shards 1 --cp_degree 1 --tp_degree 1"
DDP_2="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 2 --dp_shards 1 --cp_degree 1 --tp_degree 1"
DDP_4="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 4 --dp_shards 1 --cp_degree 1 --tp_degree 1"
DDP_5="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 5 --dp_shards 1 --cp_degree 1 --tp_degree 1"
DDP_6="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 6 --dp_shards 1 --cp_degree 1 --tp_degree 1"
DDP_8="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 8 --dp_shards 1 --cp_degree 1 --tp_degree 1"
FSDP_2="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 1 --dp_shards 2 --cp_degree 1 --tp_degree 1"
FSDP_4="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 1 --dp_shards 4 --cp_degree 1 --tp_degree 1"
HSDP_2_2="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 2 --dp_shards 2 --cp_degree 1 --tp_degree 1"

# Parallel arguments
parallel_cmd=(
  $DDP_2
)

# Model arguments
model_cmd=(
  --model_name "cogvideox"
  --pretrained_model_name_or_path "THUDM/CogVideoX-5b-I2V"
  --compile_modules transformer
  --cache_dir /gpfs/junlab/wangwanding24/hub #Your Hugingface Hub cache datapath, for CogVideoX-5b-I2V Model
  --image_pipeline
)

# Control arguments
control_cmd=(
  --control_type custom
  --rank 192
  --lora_alpha 192
  --target_modules "blocks.*(to_q|to_k|to_v|to_out.0|ff.net.0.proj|ff.net.2)"
  --frame_conditioning_type full
  --frame_conditioning_index 0
)

# Dataset arguments
dataset_cmd=(
  --dataset_config $TRAINING_DATASET_CONFIG
  --dataset_shuffle_buffer_size 32
  --enable_precomputation
  --precomputation_items 250
)

# Dataloader arguments
dataloader_cmd=(
  --dataloader_num_workers 0
)

# Diffusion arguments
diffusion_cmd=(
  --flow_weighting_scheme "logit_normal"
)

training_cmd=(
  --training_type control-lora
  --seed 42
  --batch_size 1
  --train_steps 40000 
  --gradient_accumulation_steps 1
  --gradient_checkpointing
  --checkpointing_steps 5000 
  --checkpointing_limit 5
  --resume_from_checkpoint 5000
  --enable_slicing
  --enable_tiling
)

optimizer_cmd=(
  --optimizer "adamw"
  --lr 2e-5
  --lr_scheduler "constant_with_warmup"
  --lr_warmup_steps 1000
  --lr_num_cycles 1
  --beta1 0.9
  --beta2 0.99
  --weight_decay 1e-4
  --epsilon 1e-8
  --max_grad_norm 1.0
)

# Disable validation to speed up training
validation_cmd=(
  --validation_dataset_file "$VALIDATION_DATASET_FILE"
  --validation_steps 5000
  --disable_validation
)

miscellaneous_cmd=(
  --tracker_name "finetrainers-cogvideox-control"
  --output_dir "src/train_logs/final_model"
  --init_timeout 600
  --nccl_timeout 600
  --report_to "wandb"
)

# Execute the training script
if [ "$BACKEND" == "accelerate" ]; then

  ACCELERATE_CONFIG_FILE=""
  if [ "$NUM_GPUS" == 1 ]; then
    ACCELERATE_CONFIG_FILE="accelerate_configs/uncompiled_1.yaml"
  elif [ "$NUM_GPUS" == 2 ]; then
    ACCELERATE_CONFIG_FILE="accelerate_configs/uncompiled_2.yaml"
  elif [ "$NUM_GPUS" == 4 ]; then
    ACCELERATE_CONFIG_FILE="accelerate_configs/uncompiled_4.yaml"
  elif [ "$NUM_GPUS" == 8 ]; then
    ACCELERATE_CONFIG_FILE="accelerate_configs/uncompiled_8.yaml"
  fi
  
  accelerate launch --config_file "$ACCELERATE_CONFIG_FILE" --gpu_ids $CUDA_VISIBLE_DEVICES train.py \
    "${parallel_cmd[@]}" \
    "${model_cmd[@]}" \
    "${control_cmd[@]}" \
    "${dataset_cmd[@]}" \
    "${dataloader_cmd[@]}" \
    "${diffusion_cmd[@]}" \
    "${training_cmd[@]}" \
    "${optimizer_cmd[@]}" \
    "${validation_cmd[@]}" \
    "${miscellaneous_cmd[@]}"

elif [ "$BACKEND" == "ptd" ]; then

  export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
  
  torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    --rdzv_backend c10d \
    --rdzv_endpoint="localhost:19242" \
    ./src/train.py \
      "${parallel_cmd[@]}" \
      "${model_cmd[@]}" \
      "${control_cmd[@]}" \
      "${dataset_cmd[@]}" \
      "${dataloader_cmd[@]}" \
      "${diffusion_cmd[@]}" \
      "${training_cmd[@]}" \
      "${optimizer_cmd[@]}" \
      "${validation_cmd[@]}" \
      "${miscellaneous_cmd[@]}"
fi

echo -ne "-------------------- Finished executing script --------------------\n\n"
