#!/usr/bin/env bash
# ============================================================================
# Agilex Cobot Magic (LeRobot) :: Multi-Stage Training Launcher
# Runs Stage 1 -> Stage 4 sequentially, loading the previous stage checkpoint.
#
# This script mirrors `scripts/run_libero_multistage.sh` and only changes:
# - CONFIG_PATH / RUN_ROOT_DIR / RUN_ID_PREFIX / WANDB_PROJECT defaults
# - steps-cache directory default
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
CONFIG_PATH="${CONFIG_PATH:-/share/project/lvjing/starVLA/starVLA/config/training/agilex_cobot_magic_action_only_50step.yaml}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-results/AgilexCobotMagic}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-agilex_cobot_magic_multistage}"
WANDB_PROJECT="${WANDB_PROJECT:-agilex_cobot_magic_ecot}"
WANDB_ENTITY="${WANDB_ENTITY:-lvj2114-beijing-academy-of-artificial-intelligence}"
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29523}"

# Keep parity with bridge/libero launcher knobs
USE_REASONING_SUMMARY="${USE_REASONING_SUMMARY:-false}"
REASONING_SUMMARY_TOKENS="${REASONING_SUMMARY_TOKENS:-2}"
REASONING_SUMMARY_HEADS="${REASONING_SUMMARY_HEADS:-4}"
REASONING_SUMMARY_DROPOUT="${REASONING_SUMMARY_DROPOUT:-0.1}"
USE_REASONING_FILM="${USE_REASONING_FILM:-false}"
REASONING_FILM_FIRST_K="${REASONING_FILM_FIRST_K:-0}"
REASONING_FILM_DROPOUT="${REASONING_FILM_DROPOUT:-0.1}"
REASONING_FILM_HIDDEN="${REASONING_FILM_HIDDEN:-1024}"
attention_implementation="${attention_implementation:-sdpa}"

# Common knobs
RELOAD_MODULES="${RELOAD_MODULES:-}"
GLOBAL_STEPS_CACHE_PATH="${STEPS_CACHE_PATH:-}"

# Local steps cache dir (directory path => one cache file per dataset/config_key)
STEPS_CACHE_DIR="${STEPS_CACHE_DIR:-${RUN_ROOT_DIR}/steps_cache/agilex_cobot_magic_real4}"

# Stage control
STAGES=(1 2 3 4)
START_STAGE="${START_STAGE:-1}"
INITIAL_PRETRAINED_CKPT="${INITIAL_PRETRAINED_CKPT:-}"
CKPT_PRE="${CKPT_PRE:-}"

# Stage semantics (match config stages; identity mapping)
declare -A STAGE_BRIDGE_STAGE=(
  [1]=1
  [2]=2
  [3]=3
  [4]=4
)
declare -A STAGE_SCHEDULED_STAGE=(
  [1]=1
  [2]=2
  [3]=3
  [4]=4
)
declare -A STAGE_TRAINING_STAGE=(
  [1]="${TRAINING_STAGE:-reasoning_only}"
  [2]="${TRAINING_STAGE:-reasoning_only}"
  [3]="${TRAINING_STAGE:-reasoning_only}"
  [4]="${TRAINING_STAGE:-reasoning_only}"
)
declare -A STAGE_VLM_LOSS_WEIGHT=(
  [1]=1.0
  [2]=1.0
  [3]=1.0
  [4]=1.0
)
IMG_NEXT_LOSS_WEIGHT_STAGE1="${IMG_NEXT_LOSS_WEIGHT_STAGE1:-0.1}"
IMG_NEXT_LOSS_WEIGHT_STAGE2="${IMG_NEXT_LOSS_WEIGHT_STAGE2:-0.1}"
IMG_NEXT_LOSS_WEIGHT_STAGE3="${IMG_NEXT_LOSS_WEIGHT_STAGE3:-0.2}"
IMG_NEXT_LOSS_WEIGHT_STAGE4="${IMG_NEXT_LOSS_WEIGHT_STAGE4:-0.2}"
declare -A STAGE_IMG_NEXT_LOSS_WEIGHT=(
  [1]="${IMG_NEXT_LOSS_WEIGHT_STAGE1}"
  [2]="${IMG_NEXT_LOSS_WEIGHT_STAGE2}"
  [3]="${IMG_NEXT_LOSS_WEIGHT_STAGE3}"
  [4]="${IMG_NEXT_LOSS_WEIGHT_STAGE4}"
)
declare -A STAGE_BATCH_SIZE=(
  [1]=12
  [2]=12
  [3]=12
  [4]=16
)
declare -A STAGE_MAX_STEPS=(
  [1]=5000
  [2]=2000
  [3]=2000
  [4]=2000
)
declare -A STAGE_SAVE_INTERVAL=(
  [1]=5000
  [2]=2000
  [3]=2000
  [4]=2000
)
declare -A STAGE_EXPORT_STEP=(
  # The checkpoint step to load for the next stage. By default, match save_interval.
  [1]=5000
  [2]=2000
  [3]=2000
  [4]=2000
)

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
  local img_next_loss_weight="${STAGE_IMG_NEXT_LOSS_WEIGHT[$stage]}"
  local per_device_batch="${STAGE_BATCH_SIZE[$stage]}"
  local max_steps="${STAGE_MAX_STEPS[$stage]}"
  local save_interval="${STAGE_SAVE_INTERVAL[$stage]}"
  local export_step="${STAGE_EXPORT_STEP[$stage]}"
  local cot_mode="${COT_MODE:-implicit}"
  local component_order='[SUBTASK,BBOX,REASON]'

  local steps_cache_path
  if [[ -n "${GLOBAL_STEPS_CACHE_PATH}" ]]; then
    steps_cache_path="${GLOBAL_STEPS_CACHE_PATH}"
  else
    steps_cache_path="${STEPS_CACHE_DIR}"
  fi

  echo "============================================================================"
  echo "Stage ${stage} | bridge_stage=${bridge_stage} | scheduled_stage=${scheduled_stage}"
  echo " Run ID : ${run_id}"
  echo " Output : ${output_dir}"
  echo " Steps cache : ${steps_cache_path}"
  echo "============================================================================"

  TRAIN_CONFIG_ARGS=(
    --framework.training_stage "${training_stage}"
    --framework.cot_mode "${cot_mode}"
    --framework.qwenvl.attn_implementation "${attention_implementation}"
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
    --datasets.vla_data.bridge_annotations.write_steps_cache "true"
    --framework.action_model.use_reasoning_film "${USE_REASONING_FILM}"
    --framework.action_model.reasoning_film_first_k "${REASONING_FILM_FIRST_K}"
    --framework.action_model.reasoning_film_dropout "${REASONING_FILM_DROPOUT}"
    --framework.action_model.reasoning_film_hidden "${REASONING_FILM_HIDDEN}"
    --framework.action_model.use_reasoning_summary "${USE_REASONING_SUMMARY}"
    --framework.action_model.reasoning_summary_tokens "${REASONING_SUMMARY_TOKENS}"
    --framework.action_model.reasoning_summary_heads "${REASONING_SUMMARY_HEADS}"
    --framework.action_model.reasoning_summary_dropout "${REASONING_SUMMARY_DROPOUT}"
    --framework.latent_reasoning.vlm_loss_weight "${vlm_loss_weight}"
    --framework.img_next.loss_weight "${img_next_loss_weight}"
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

  local next_ckpt="${output_dir}/checkpoints/steps_${export_step}_pytorch_model.pt"
  if [[ ! -f "${next_ckpt}" ]]; then
    echo "Expected checkpoint not found for stage ${stage}: ${next_ckpt}"
    next_ckpt=""
  fi
  LAST_STAGE_CKPT="${next_ckpt}"
}

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

echo "✅ Agilex Cobot Magic multi-stage pipeline completed."

