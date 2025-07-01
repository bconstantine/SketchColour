#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Resolve directory of THIS script
###############################################################################
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"


###############################################################################
#                               USER SETTINGS
###############################################################################
NUM_GPUS=5                                # number of parallel shards
GPU_IDS="1,2,3,4,9"                            # comma-separated CUDA IDs
CKPT_LIST="40000"         # GPU 39, 1,2,3,4,9
VAL_JSON="src/examples/training/control/cogvideox/i2v-control/example_validation.json"
OUT_ROOT="src/train_logs/final_model" 
###############################################################################
# unique prefix for this run
RUN_ID=$(date +"%Y%m%d_%H%M%S")
echo "[dist_infer] Run ID      : $RUN_ID"s
# Convert comma‑separated strings → arrays -----------------------------------
IFS=',' read -ra IDS   <<< "$GPU_IDS"
IFS=',' read -ra CKPTS <<< "$CKPT_LIST"

[[ ${#IDS[@]} -ge $NUM_GPUS ]] || { echo "[dist_infer] ERROR: fewer GPU IDs than NUM_GPUS" >&2; exit 1; }

mkdir -p "$OUT_ROOT"

echo "[dist_infer] GPUs        : $GPU_IDS"
echo "[dist_infer] Checkpoints : $CKPT_LIST"
echo "[dist_infer] Out root    : $OUT_ROOT"
echo "====================================================================="

###############################################################################
# 1) Split validation.json ONCE → validation_part_<idx>.json                 #
###############################################################################
readarray -t RAW_SHARDS < <(
    split_validation_json.py "$VAL_JSON" "$NUM_GPUS"
)

# rename each into $RUN_ID_<original> and collect into SPLITS[]
SPLITS=()
for f in "${RAW_SHARDS[@]}"; do
  dir=$(dirname "$f")                   # e.g. /gpfs/junlab/.../i2v-control
  base=$(basename "$f")                 # e.g. validation_part_0.json
  newf="${dir}/${RUN_ID}_${base}"       # e.g. /gpfs/.../20250526_120705_validation_part_0.json
  mv "$f" "$newf"                       # rename in place
  SPLITS+=("$newf")                     # record the new absolute path
done

# Ensure shards are deleted when everything finishes or script is interrupted
auto_clean() { rm -f "${SPLITS[@]}"; }
trap auto_clean EXIT

echo "[dist_infer] Created ${#SPLITS[@]} validation shards"

###############################################################################
# 2) Sweep over checkpoints                                                   #
###############################################################################
for CKPT in "${CKPTS[@]}"; do
  echo "[dist_infer] >>> Starting checkpoint $CKPT"

  # ── Launch one background shard per GPU ───────────────────────────────────
  for idx in $(seq 0 $((NUM_GPUS-1))); do
      GPU=${IDS[$idx]}
      SPLIT=${SPLITS[$idx]}
      OUTDIR="${OUT_ROOT}"
      mkdir -p "$OUTDIR"

      echo "[dist_infer]   → shard $idx  GPU:$GPU  split:$SPLIT  out:$OUTDIR"
      (
          export CUDA_VISIBLE_DEVICES=$GPU
          mini_run.sh "$SPLIT" "$CKPT" "$OUTDIR"
      ) &
  done

  wait
  echo "[dist_infer] <<< Checkpoint $CKPT finished ✓"
  echo "---------------------------------------------------------------------"
done

echo "[dist_infer] All checkpoints completed successfully ✓"