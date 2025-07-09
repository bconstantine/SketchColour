#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"



NUM_GPUS=1   
GPU_IDS="0"  
CKPT_LIST="40000" #CKPT to use from the OUT_ROOT directory
VAL_JSON="./src/examples/training/control/cogvideox/i2v-control/example_validation.json"
OUT_ROOT="src/train_logs/final_model" 
RUN_ID=$(date +"%Y%m%d_%H%M%S")
echo "[dist_infer] Run ID      : $RUN_ID"s
IFS=',' read -ra IDS   <<< "$GPU_IDS"
IFS=',' read -ra CKPTS <<< "$CKPT_LIST"

[[ ${#IDS[@]} -ge $NUM_GPUS ]] || { echo "[dist_infer] ERROR: fewer GPU IDs than NUM_GPUS" >&2; exit 1; }

mkdir -p "$OUT_ROOT"

echo "[dist_infer] GPUs        : $GPU_IDS"
echo "[dist_infer] Checkpoints : $CKPT_LIST"
echo "[dist_infer] Out root    : $OUT_ROOT"
echo "====================================================================="


readarray -t RAW_SHARDS < <(
    python ./src/examples/training/control/cogvideox/i2v-control/split_validation_json.py "$VAL_JSON" "$NUM_GPUS"
)

SPLITS=()
for f in "${RAW_SHARDS[@]}"; do
  dir=$(dirname "$f") 
  base=$(basename "$f") 
  newf="${dir}/${RUN_ID}_${base}" 
  mv "$f" "$newf"       
  SPLITS+=("$newf")
done

auto_clean() { rm -f "${SPLITS[@]}"; }
trap auto_clean EXIT

echo "[dist_infer] Created ${#SPLITS[@]} validation shards"
for CKPT in "${CKPTS[@]}"; do
  echo "[dist_infer] >>> Starting checkpoint $CKPT"

  for idx in $(seq 0 $((NUM_GPUS-1))); do
      GPU=${IDS[$idx]}
      SPLIT=${SPLITS[$idx]}
      OUTDIR="${OUT_ROOT}"
      mkdir -p "$OUTDIR"

      echo "[dist_infer]   → shard $idx  GPU:$GPU  split:$SPLIT  out:$OUTDIR"
      (
          export CUDA_VISIBLE_DEVICES=$GPU
          ./src/examples/training/control/cogvideox/i2v-control/mini_run.sh "$SPLIT" "$CKPT" "$OUTDIR"
      ) &
  done

  wait
  echo "[dist_infer] <<< Checkpoint $CKPT finished ✓"
  echo "---------------------------------------------------------------------"
done

echo "[dist_infer] All checkpoints completed successfully ✓"