#!/usr/bin/env bash
# ============================================================================
# Bridge-LeRobot ECOT Stage-2 Training Launcher
#  - Uses starVLA/config/training/bridge_ecot_stage2.yaml as base config
#  - Provides common overrides similar to scripts/run_ecot_8gpu.sh
# ============================================================================
set -euo pipefail

# ----------------------------------------------------------------------------
# 环境变量（可按需修改）
# ----------------------------------------------------------------------------
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/share/project/lvjing/starVLA/qwen_cache}"
export WANDB_API_KEY="${WANDB_API_KEY:-a8989c35c0573184da807b8a781d72936fe7e379}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# export WANDB_MODE=offline

# ----------------------------------------------------------------------------
# 训练参数（可通过环境变量覆写）
# ----------------------------------------------------------------------------
NUM_GPUS="${NUM_GPUS:-8}"

MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-60000}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-24}"
LR_VLM="${LR_VLM:-1e-5}"
LR_ACTION="${LR_ACTION:-1e-4}"
ACTION_DIT_TYPE="${ACTION_DIT_TYPE:-DiT-B}"
RUN_ID="${RUN_ID:-bridge_lerobot_DITB_LR1E-4_LR1E-5_BTS24_60K}"
DIFFUSION_MODEL_DROPOUT="${DIFFUSION_MODEL_DROPOUT:-0.2}"
# Reasoning summary (latent summarizer) controls
USE_REASONING_SUMMARY="${USE_REASONING_SUMMARY:-false}"
REASONING_SUMMARY_TOKENS="${REASONING_SUMMARY_TOKENS:-2}"
REASONING_SUMMARY_HEADS="${REASONING_SUMMARY_HEADS:-4}"
REASONING_SUMMARY_DROPOUT="${REASONING_SUMMARY_DROPOUT:-0.1}"
# FiLM controls
USE_REASONING_FILM="${USE_REASONING_FILM:-true}"
REASONING_FILM_FIRST_K="${REASONING_FILM_FIRST_K:-4}"
REASONING_FILM_DROPOUT="${REASONING_FILM_DROPOUT:-0.1}"
REASONING_FILM_HIDDEN="${REASONING_FILM_HIDDEN:-1024}"






TRAINING_STAGE="${TRAINING_STAGE:-full}"

# Action-only run: use stage 4 (all latents) and keep only action tokens
SCHEDULED_STAGE="${SCHEDULED_STAGE:-4}"

MIN_SAVE_STEP="${MIN_SAVE_STEP:-15000}"
LR_BASE="${LR_BASE:-3.0e-5}"

WANDB_PROJECT="${WANDB_PROJECT:-bridge_lerobot_final}"


if (( SCHEDULED_STAGE <= 2 )); then
  DEFAULT_STEPS_CACHE="/share/project/baishuanghao/data/bridge_orig_lerobot/meta/steps_9f926a41b0ba.pkl"
else
  DEFAULT_STEPS_CACHE="/share/project/baishuanghao/data/bridge_orig_lerobot/meta/steps_45cc68a6124a.pkl"
fi
STEPS_CACHE_PATH="${STEPS_CACHE_PATH:-${DEFAULT_STEPS_CACHE}}"


CONFIG_PATH="${CONFIG_PATH:-/share/project/lvjing/starVLA/starVLA/config/training/bridge_lerobot_stage2.yaml}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-results/BridgeFinal_Action}"
MASTER_PORT="${MASTER_PORT:-29512}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-20000000}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-20}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-/share/project/lvjing/starVLA/results/BridgeLeRobot_VLM2/bridge_multistage_vlm2_stage_4/checkpoints/steps_10000_pytorch_model.pt}"
# PRETRAINED_CKPT="${PRETRAINED_CKPT:-}"


RELOAD_MODULES="${RELOAD_MODULES:-qwen_vl_interface}"


OUTPUT_DIR="${RUN_ROOT_DIR}/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"
cp "$0" "${OUTPUT_DIR}/run_command.sh"

# 训练配置覆盖项
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
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH}"
  --datasets.vla_data.bridge_reasoning.stage "${SCHEDULED_STAGE}"
  --datasets.vla_data.ecot.scheduled_stage "${SCHEDULED_STAGE}"
  --datasets.vla_data.bridge_reasoning.include_action_tokens "false"
  --datasets.vla_data.bridge_annotations.steps_cache_path "${STEPS_CACHE_PATH}"
  --framework.action_model.diffusion_model_cfg.dropout "${DIFFUSION_MODEL_DROPOUT}"
  --framework.action_model.use_reasoning_film "${USE_REASONING_FILM}"
  --framework.action_model.reasoning_film_first_k "${REASONING_FILM_FIRST_K}"
  --framework.action_model.reasoning_film_dropout "${REASONING_FILM_DROPOUT}"
  --framework.action_model.reasoning_film_hidden "${REASONING_FILM_HIDDEN}"
  --framework.action_model.use_reasoning_summary "${USE_REASONING_SUMMARY}"
  --framework.action_model.reasoning_summary_tokens "${REASONING_SUMMARY_TOKENS}"
  --framework.action_model.reasoning_summary_heads "${REASONING_SUMMARY_HEADS}"
  --framework.action_model.reasoning_summary_dropout "${REASONING_SUMMARY_DROPOUT}"
  --framework.training_stage "full"
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
  --wandb_entity "${WANDB_ENTITY:-lvj2114-beijing-academy-of-artificial-intelligence}"
)

echo "============================================================================"
echo " Bridge-LeRobot ECOT Stage-2 Training"
echo " Config : ${CONFIG_PATH}"
echo " Run ID : ${RUN_ID}"
echo " Output : ${OUTPUT_DIR}"
echo " GPUs   : ${NUM_GPUS} (master_port=${MASTER_PORT})"
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
