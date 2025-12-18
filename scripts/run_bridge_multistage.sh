#!/usr/bin/env bash
# ============================================================================
# Bridge-LeRobot :: Multi-Stage Training Launcher
# Runs Stage 1 -> Stage 4 sequentially. Each stage loads the previous stage's
# checkpoint (configurable step) and adjusts thinking-token settings.
# ----------------------------------------------------------------------------
# Stage semantics for this script:
#   Stage 1 : Full CoT (bridge_reasoning.stage = 0)
#   Stage 2 : +Subtask latent  (bridge_reasoning.stage = 2)
#   Stage 3 : +Reason latent   (bridge_reasoning.stage = 3)
#   Stage 4 : +BBox latent     (bridge_reasoning.stage = 4)
# ============================================================================
set -euo pipefail

# ----------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/share/project/lvjing/starVLA/qwen_cache}"
export WANDB_API_KEY="${WANDB_API_KEY:-a8989c35c0573184da807b8a781d72936fe7e379}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# ----------------------------------------------------------------------------
# Base configuration
# ----------------------------------------------------------------------------
CONFIG_PATH="${CONFIG_PATH:-/share/project/lvjing/starVLA/starVLA/config/training/bridge_lerobot_stage2.yaml}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-results/BridgeLeRobot_VLM2}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-bridge_multistage_vlm2}"
WANDB_PROJECT="${WANDB_PROJECT:-bridge_multistage_vlm2}"
WANDB_ENTITY="${WANDB_ENTITY:-lvj2114-beijing-academy-of-artificial-intelligence}"
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29512}"
ACTION_TOKENS_IN_STAGE1="${ACTION_TOKENS_IN_STAGE1:-true}"
USE_REASONING_SUMMARY="${USE_REASONING_SUMMARY:-false}"
REASONING_SUMMARY_TOKENS="${REASONING_SUMMARY_TOKENS:-2}"
REASONING_SUMMARY_HEADS="${REASONING_SUMMARY_HEADS:-4}"
REASONING_SUMMARY_DROPOUT="${REASONING_SUMMARY_DROPOUT:-0.1}"
USE_REASONING_FILM="${USE_REASONING_FILM:-false}"
REASONING_FILM_FIRST_K="${REASONING_FILM_FIRST_K:-0}"
REASONING_FILM_DROPOUT="${REASONING_FILM_DROPOUT:-0.1}"
REASONING_FILM_HIDDEN="${REASONING_FILM_HIDDEN:-1024}"

# Staging control
# STAGES=(1 2 3 4)
STAGES=(1)
START_STAGE="${START_STAGE:-1}"
INITIAL_PRETRAINED_CKPT="${INITIAL_PRETRAINED_CKPT:-}"
CKPT_PRE="${CKPT_PRE:-}"
RELOAD_MODULES="${RELOAD_MODULES:-}"
GLOBAL_STEPS_CACHE_PATH="${STEPS_CACHE_PATH:-}"

# Stage-specific configs (can be adjusted as needed)
declare -A STAGE_BRIDGE_STAGE=(
  [1]=1   # full CoT
  [2]=2   # subtask latent
  [3]=3   # subtask+reason latent
  [4]=4   # subtask+reason+bbox latent
)
declare -A STAGE_SCHEDULED_STAGE=(
  [1]=1
  [2]=2
  [3]=3
  [4]=4
)
declare -A STAGE_TRAINING_STAGE=(
  [1]="reasoning_only"
  [2]="reasoning_only"
  [3]="reasoning_only"
  [4]="full"
)
declare -A STAGE_VLM_LOSS_WEIGHT=(
  [1]=1.0
  [2]=1.0
  [3]=1.0
  [4]=0.5
)
declare -A STAGE_BATCH_SIZE=(
  [1]=12
  [2]=12
  [3]=16
  [4]=16
)
declare -A STAGE_MAX_STEPS=(
  [1]=10000
  [2]=5000
  [3]=5000
  [4]=10000
)
declare -A STAGE_SAVE_INTERVAL=(
  [1]=10000
  [2]=5000
  [3]=5000
  [4]=5000
)
declare -A STAGE_CHECKPOINT_EXPORT=(
  [1]=10000
  [2]=5000
  [3]=5000
  [4]=5000
)

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
default_steps_cache_for_stage() {
  local scheduled_stage="$1"
  if (( scheduled_stage <= 3 )); then
    echo "/share/project/baishuanghao/data/bridge_orig_lerobot/meta/steps_9f926a41b0ba.pkl"
  else
    echo "/share/project/baishuanghao/data/bridge_orig_lerobot/meta/steps_45cc68a6124a.pkl"
  fi
}

ensure_output_dir() {
  mkdir -p "$1"
}

LAST_STAGE_CKPT=""

run_stage() {
  local stage="$1"
  local prev_ckpt="$2"

  local run_id="${RUN_ID_PREFIX}_stage_${stage}"
  local output_dir="${RUN_ROOT_DIR}/${run_id}"
  ensure_output_dir "${output_dir}"
  cp "$0" "${output_dir}/run_command.sh"

  local bridge_stage="${STAGE_BRIDGE_STAGE[$stage]}"
  local scheduled_stage="${STAGE_SCHEDULED_STAGE[$stage]}"
  local training_stage="${STAGE_TRAINING_STAGE[$stage]}"
  local vlm_loss_weight="${STAGE_VLM_LOSS_WEIGHT[$stage]}"
  local per_device_batch="${STAGE_BATCH_SIZE[$stage]}"
  local max_steps="${STAGE_MAX_STEPS[$stage]}"
  local save_interval="${STAGE_SAVE_INTERVAL[$stage]}"
  local component_order="SUBTASK,BBOX,REASON"

  local steps_cache_path
  if [[ -n "${GLOBAL_STEPS_CACHE_PATH}" ]]; then
    steps_cache_path="${GLOBAL_STEPS_CACHE_PATH}"
  else
    steps_cache_path="$(default_steps_cache_for_stage "${scheduled_stage}")"
  fi

  echo "============================================================================"
  echo "Stage ${stage} | bridge_stage=${bridge_stage} | scheduled_stage=${scheduled_stage}"
  echo " Run ID : ${run_id}"
  echo " Output : ${output_dir}"
  echo "============================================================================"

  TRAIN_CONFIG_ARGS=(
    --trainer.max_train_steps "${max_steps}"
    --trainer.save_interval "${save_interval}"
    --trainer.eval_interval 50000000
    --trainer.logging_frequency 20
    --trainer.warmup_ratio 0.1
    --trainer.learning_rate.base 3.0e-5
    --trainer.learning_rate.action_model 1.0e-4
    --datasets.vla_data.per_device_batch_size "${per_device_batch}"
    --datasets.vla_data.bridge_reasoning.stage "${bridge_stage}"
    --datasets.vla_data.ecot.scheduled_stage "${scheduled_stage}"
    --datasets.vla_data.bridge_reasoning.include_action_tokens "true"
    --datasets.vla_data.bridge_reasoning.component_order "${component_order}"
    --datasets.vla_data.bridge_annotations.steps_cache_path "${steps_cache_path}"
    --framework.action_model.use_reasoning_film "${USE_REASONING_FILM}"
    --framework.action_model.reasoning_film_first_k "${REASONING_FILM_FIRST_K}"
    --framework.action_model.reasoning_film_dropout "${REASONING_FILM_DROPOUT}"
    --framework.action_model.reasoning_film_hidden "${REASONING_FILM_HIDDEN}"
    --framework.action_model.use_reasoning_summary "${USE_REASONING_SUMMARY}"
    --framework.action_model.reasoning_summary_tokens "${REASONING_SUMMARY_TOKENS}"
    --framework.action_model.reasoning_summary_heads "${REASONING_SUMMARY_HEADS}"
    --framework.action_model.reasoning_summary_dropout "${REASONING_SUMMARY_DROPOUT}"
    --framework.training_stage "${training_stage}"
    --framework.latent_reasoning.vlm_loss_weight "${vlm_loss_weight}"
  )

  if [[ -n "${prev_ckpt}" ]]; then
    echo "Loading checkpoint: ${prev_ckpt}"
    TRAIN_CONFIG_ARGS+=( --trainer.pretrained_checkpoint "${prev_ckpt}" )
    if [[ -n "${RELOAD_MODULES}" ]]; then
      TRAIN_CONFIG_ARGS+=( --trainer.reload_modules "${RELOAD_MODULES}" )
    fi
  fi

  cmd=(
    torchrun
    --nproc_per_node="${NUM_GPUS}"
    --master_port="${MASTER_PORT}"
    starVLA/training/train_ecot.py
    --config_yaml "${CONFIG_PATH}"
    --run_root_dir "${RUN_ROOT_DIR}"
    --run_id "${run_id}"
    --wandb_project "${WANDB_PROJECT}"
    --wandb_entity "${WANDB_ENTITY}"
    "${TRAIN_CONFIG_ARGS[@]}"
  )

  echo "Command:"
  printf '  %q' "${cmd[@]}"
  echo
  "${cmd[@]}"

  local export_step="${STAGE_CHECKPOINT_EXPORT[$stage]}"
  local next_ckpt="${output_dir}/checkpoints/steps_${export_step}_pytorch_model.pt"
  if [[ ! -f "${next_ckpt}" ]]; then
    echo "Expected checkpoint not found for stage ${stage}: ${next_ckpt}"
    echo "Please check save_interval/export step settings."
    next_ckpt=""
  fi
  LAST_STAGE_CKPT="${next_ckpt}"
}

# ----------------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------------

previous_ckpt="${CKPT_PRE:-${INITIAL_PRETRAINED_CKPT}}"
for stage in "${STAGES[@]}"; do
  if (( stage < START_STAGE )); then
    continue
  fi

  if (( stage > START_STAGE )) && [[ -z "${previous_ckpt}" ]]; then
    echo "❌ Missing checkpoint for Stage ${stage}. Set INITIAL_PRETRAINED_CKPT or ensure prior stage ran."
    exit 1
  fi

  run_stage "${stage}" "${previous_ckpt}"
  previous_ckpt="${LAST_STAGE_CKPT}"
done

echo "✅ Multi-stage pipeline completed."
