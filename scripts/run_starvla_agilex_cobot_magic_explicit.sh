#!/usr/bin/env bash
# ============================================================================
# Agilex Cobot Magic (LeRobot) :: ECOT Stage-1 Explicit CoT Training Launcher
#  - Fixed COT_MODE=explicit（显式 CoT 文本生成，不用 latent reasoning/FiLM/summary）
#  - Mirrors scripts/run_starvla_bridge_explicit.sh behavior
#  - Base config: starVLA/config/training/agilex_cobot_magic_action_only_50step.yaml
# ============================================================================
set -euo pipefail

# ----------------------------------------------------------------------------
# 环境变量
# ----------------------------------------------------------------------------
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/share/project/lvjing/starVLA/qwen_cache}"
export WANDB_API_KEY="${WANDB_API_KEY:-a8989c35c0573184da807b8a781d72936fe7e379}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ----------------------------------------------------------------------------
# 训练参数（可通过环境变量覆写）
# ----------------------------------------------------------------------------
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29524}"

MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-20000}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-12}"
LR_VLM="${LR_VLM:-1.2e-5}"
LR_ACTION="${LR_ACTION:-1.2e-4}"
LR_BASE="${LR_BASE:-3.0e-5}"
ACTION_DIT_TYPE="${ACTION_DIT_TYPE:-DiT-B}"
DIFFUSION_MODEL_DROPOUT="${DIFFUSION_MODEL_DROPOUT:-0.1}"

RUN_ID="${RUN_ID:-agilex_cobot_magic_explicit_cot_stage1}"
TRAINING_STAGE="${TRAINING_STAGE:-full}"

# 显式 CoT 固定使用 stage=1
SCHEDULED_STAGE=1

SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-20000000}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-20}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
MIN_SAVE_STEP="${MIN_SAVE_STEP:-10000}"

WANDB_PROJECT="${WANDB_PROJECT:-agilex_cobot_magic_explicit}"
WANDB_ENTITY="${WANDB_ENTITY:-lvj2114-beijing-academy-of-artificial-intelligence}"

CONFIG_PATH="${CONFIG_PATH:-/share/project/lvjing/starVLA/starVLA/config/training/agilex_cobot_magic_action_only_50step.yaml}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-results/AgilexCobotMagic_explicit}"

# Steps cache: use an absolute directory path (avoid writing under dataset root)
DEFAULT_STEPS_CACHE_PATH="${REPO_ROOT}/${RUN_ROOT_DIR}/steps_cache/agilex_cobot_magic_explicit"
STEPS_CACHE_PATH="${STEPS_CACHE_PATH:-${DEFAULT_STEPS_CACHE_PATH}}"
WRITE_STEPS_CACHE="${WRITE_STEPS_CACHE:-true}"

# Optional: load from checkpoint
PRETRAINED_CKPT="${PRETRAINED_CKPT:-}"
RELOAD_MODULES="${RELOAD_MODULES:-}"

OUTPUT_DIR="${RUN_ROOT_DIR}/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"
cp "$0" "${OUTPUT_DIR}/run_command.sh"

# ----------------------------------------------------------------------------
# 训练配置覆盖项
# ----------------------------------------------------------------------------
TRAIN_CONFIG_ARGS=(
  --trainer.max_train_steps "${MAX_TRAIN_STEPS}"
  --trainer.save_interval "${SAVE_INTERVAL}"
  --trainer.eval_interval "${EVAL_INTERVAL}"
  --trainer.logging_frequency "${LOGGING_FREQUENCY}"
  --trainer.warmup_ratio "${WARMUP_RATIO}"
  --trainer.min_save_step "${MIN_SAVE_STEP}"
  --trainer.learning_rate.base "${LR_BASE}"
  --trainer.learning_rate.action_model "${LR_ACTION}"
  --trainer.learning_rate.qwen_vl_interface "${LR_VLM}"

  --framework.action_model.action_model_type "${ACTION_DIT_TYPE}"
  --framework.action_model.diffusion_model_cfg.dropout "${DIFFUSION_MODEL_DROPOUT}"
  --framework.action_model.use_reasoning_film "false"
  --framework.action_model.use_reasoning_summary "false"

  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH}"
  --datasets.vla_data.bridge_reasoning.stage "${SCHEDULED_STAGE}"
  --datasets.vla_data.ecot.scheduled_stage "${SCHEDULED_STAGE}"
  --datasets.vla_data.bridge_reasoning.include_action_tokens "false"

  --datasets.vla_data.bridge_annotations.steps_cache_path "${STEPS_CACHE_PATH}"
  --datasets.vla_data.bridge_annotations.write_steps_cache "${WRITE_STEPS_CACHE}"

  # Explicit baseline: do NOT append img_next tokens in formatter or enable img_next in the model
  --framework.img_next.enable "false"
  --datasets.vla_data.bridge_reasoning.include_img_next "false"
  --datasets.vla_data.bridge_reasoning.img_next_count 0

  --framework.training_stage "${TRAINING_STAGE}"
  # 显式 CoT：固定 cot_mode=explicit，关闭 latent reasoning / emit_thinking_tokens
  --framework.cot_mode "explicit"
  --framework.enable_latent_reasoning "false"
)

if [[ -n "${PRETRAINED_CKPT}" ]]; then
  TRAIN_CONFIG_ARGS+=(--trainer.pretrained_checkpoint "${PRETRAINED_CKPT}")
  if [[ -n "${RELOAD_MODULES}" ]]; then
    TRAIN_CONFIG_ARGS+=(--trainer.reload_modules "${RELOAD_MODULES}")
  fi
fi

BASE_CONFIG_ARGS=(
  --run_root_dir "${RUN_ROOT_DIR}"
  --run_id "${RUN_ID}"
  --framework.training_stage "${TRAINING_STAGE}"
  --wandb_project "${WANDB_PROJECT}"
  --wandb_entity "${WANDB_ENTITY}"
)

echo "============================================================================"
echo " Agilex Cobot Magic ECOT Explicit CoT Training (Stage=1)"
echo " Config : ${CONFIG_PATH}"
echo " Run ID : ${RUN_ID}"
echo " Output : ${OUTPUT_DIR}"
echo " GPUs   : ${NUM_GPUS} (master_port=${MASTER_PORT})"
echo " Steps cache: ${STEPS_CACHE_PATH} (write=${WRITE_STEPS_CACHE})"
echo "============================================================================"

torchrun \
  --nproc_per_node="${NUM_GPUS}" \
  --master_port="${MASTER_PORT}" \
  starVLA/training/train_ecot.py \
  --config_yaml "${CONFIG_PATH}" \
  "${BASE_CONFIG_ARGS[@]}" \
  "${TRAIN_CONFIG_ARGS[@]}" \
  "$@"

echo "✅ Training finished. Check ${OUTPUT_DIR} for logs and checkpoints."

