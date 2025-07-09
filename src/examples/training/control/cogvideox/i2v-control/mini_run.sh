#!/usr/bin/env bash
set -euo pipefail
VAL_FILE=$1  
CKPT=$2     
OUTDIR=$3


NUM_GPUS=1
BACKEND="ptd"                 # keep identical to training code
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES not set — shard launcher aborted}
TRAINING_DATASET_CONFIG="./src/examples/training/control/cogvideox/i2v-control/training.json" 

DDP_1="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 1 --dp_shards 1 \
       --cp_degree 1 --tp_degree 1"

parallel_cmd=( $DDP_1 )

model_cmd=(
  --model_name "cogvideox"
  --pretrained_model_name_or_path "THUDM/CogVideoX-5b-I2V"
  --compile_modules transformer
  --cache_dir /gpfs/junlab/wangwanding24/hub #Your Hugingface Hub cache datapath, for CogVideoX-5b-I2V Model
  --image_pipeline
)

control_cmd=(
  --control_type custom
  --rank 192
  --lora_alpha 192
  --target_modules "blocks.*(to_q|to_k|to_v|to_out.0|ff.net.0.proj|ff.net.2)"
  --frame_conditioning_type full
  --frame_conditioning_index 0
)

dataset_cmd=(
  --dataset_config $TRAINING_DATASET_CONFIG
  --dataset_shuffle_buffer_size 32
  --enable_precomputation
  --precomputation_items 100
)

dataloader_cmd=(
  --dataloader_num_workers 0
)

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
  --checkpointing_limit 40
  --resume_from_checkpoint 40000
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

validation_cmd=(
  --validation_dataset_file "$VAL_FILE"
  --resume_from_checkpoint "$CKPT"
  --enable_model_cpu_offload 
  --force_every_validation_to_be_bfloat16
)
miscellaneous_cmd=(
  --tracker_name "finetrainers-cogvideox-control"
  --output_dir "$OUTDIR"
)
# --------------------------------------------------------------------------- #
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=$NUM_GPUS \
  --rdzv_backend c10d \
  --rdzv_endpoint="localhost:29${CUDA_VISIBLE_DEVICES}42" \
  ./src/run.py \
    "${parallel_cmd[@]}" \
    "${model_cmd[@]}"     \
    "${control_cmd[@]}"   \
    "${dataset_cmd[@]}"   \
    "${dataloader_cmd[@]}"\
    "${diffusion_cmd[@]}" \
    "${training_cmd[@]}"  \
    "${optimizer_cmd[@]}" \
    "${validation_cmd[@]}"\
    "${miscellaneous_cmd[@]}"

echo "[minirun] GPU $CUDA_VISIBLE_DEVICES finished ✓"