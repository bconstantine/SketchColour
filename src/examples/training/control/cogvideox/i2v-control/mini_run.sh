#!/usr/bin/env bash
set -euo pipefail
VAL_FILE=$1          # e.g. validation_part_0.json
CKPT=$2              # e.g. 10000 or /path/to/ckpt
OUTDIR=$3

# Re-use almost everything from your original run.sh.
# We just override the bits that must be different for inference-only, 1-GPU.

NUM_GPUS=1
BACKEND="ptd"                 # keep identical to training code
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES not set — shard launcher aborted}
TRAINING_DATASET_CONFIG="examples/training/control/cogvideox/i2v-control/training_real.json"
# --------------------------------------------------------------------------- #
# ---- the block below is *copy-pasted* from run.sh and lightly trimmed ----  #
# --------------------------------------------------------------------------- #
DDP_1="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 1 --dp_shards 1 \
       --cp_degree 1 --tp_degree 1"

parallel_cmd=( $DDP_1 )

# (model_cmd, control_cmd, etc. … keep exactly as in run.sh) ------------------
model_cmd=(
  --model_name "cogvideox"
  --pretrained_model_name_or_path "THUDM/CogVideoX-5b-I2V"
  --compile_modules transformer
  --cache_dir /gpfs/junlab/wangwanding24/hub
  --image_pipeline
)

# Control arguments
control_cmd=(
  --control_type custom
  --rank 128
  --lora_alpha 128
  --target_modules "blocks.*(to_q|to_k|to_v|to_out.0|ff.net.0.proj|ff.net.2)"
  --frame_conditioning_type full
  --frame_conditioning_index 0
)

# Dataset arguments
dataset_cmd=(
  --dataset_config $TRAINING_DATASET_CONFIG
  --dataset_shuffle_buffer_size 32
  --enable_precomputation
  --precomputation_items 100
  #--enable_reuse #if the precomputed data already exists
  #--precomputation_reuse
)

# Dataloader arguments
dataloader_cmd=(
  --dataloader_num_workers 0
  #--precomputation_once
)

# Diffusion arguments
diffusion_cmd=(
  --flow_weighting_scheme "logit_normal"
)

# Training arguments
# We target just the attention projections layers for LoRA training here.
# You can modify as you please and target any layer (regex is supported)
training_cmd=(
  --training_type control-lora
  --seed 42
  --batch_size 1
  --train_steps 200000
  --gradient_accumulation_steps 1
  --gradient_checkpointing
  #--checkpointing_steps 5000
  --checkpointing_limit 40
  #--resume_from_checkpoint 55000
  --enable_slicing
  #--enable_tiling
  #--attn_provider_training "_native_flash"
)

# Optimizer arguments
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
  # NOTE:  ↓ turn OFF offloading because we are single GPU
  #        If you want *on-GPU* bfloat16, leave the flag out.
  --enable_model_cpu_offload        # <-- remove / comment
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
  run.py \
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