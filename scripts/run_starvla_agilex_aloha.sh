#!/usr/bin/env bash
# ============================================================================
# Agilex Cobot Magic (LeRobot) :: ECOT Training Launcher
#  - Uses starVLA/config/training/agilex_cobot_magic_action_only_50step.yaml as base config
#  - Mirrors scripts/run_starvla_libero.sh (same knobs / overrides layout)
#  - Uses a local steps-cache directory (required for dataset mixtures)
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

MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-10000}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-16}"
LR_VLM="${LR_VLM:-1.2e-5}"
LR_ACTION="${LR_ACTION:-1.2e-4}"
LR_BASE="${LR_BASE:-3.0e-5}"
ACTION_DIT_TYPE="${ACTION_DIT_TYPE:-DiT-B}"
DIFFUSION_MODEL_DROPOUT="${DIFFUSION_MODEL_DROPOUT:-0.1}"

RUN_ID="${RUN_ID:-agilex_aloha1}"
TRAINING_STAGE="${TRAINING_STAGE:-full}"     # full / reasoning_only / action_only
COT_MODE="${COT_MODE:-implicit}"             # none / vlm_seen_no_out / explicit / implicit

# 默认随 COT_MODE 重写；也可手动指定
SCHEDULED_STAGE="${SCHEDULED_STAGE:-4}"

SAVE_INTERVAL="${SAVE_INTERVAL:-3000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-20000000}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-10}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
MIN_SAVE_STEP="${MIN_SAVE_STEP:-7000}"

# Reasoning summary (latent summarizer) controls
USE_REASONING_SUMMARY="${USE_REASONING_SUMMARY:-false}"
REASONING_SUMMARY_TOKENS="${REASONING_SUMMARY_TOKENS:-2}"
REASONING_SUMMARY_HEADS="${REASONING_SUMMARY_HEADS:-4}"
REASONING_SUMMARY_DROPOUT="${REASONING_SUMMARY_DROPOUT:-0.1}"

# FiLM controls
USE_REASONING_FILM="${USE_REASONING_FILM:-false}"
REASONING_FILM_FIRST_K="${REASONING_FILM_FIRST_K:-4}"
REASONING_FILM_DROPOUT="${REASONING_FILM_DROPOUT:-0.1}"
REASONING_FILM_HIDDEN="${REASONING_FILM_HIDDEN:-1024}"

# Img-next controls
ENABLE_IMG_NEXT="${ENABLE_IMG_NEXT:-true}"
IMG_NEXT_USE_TEACHER="${IMG_NEXT_USE_TEACHER:-false}"
IMG_NEXT_LOSS_WEIGHT="${IMG_NEXT_LOSS_WEIGHT:-0.1}"
USE_IMG_NEXT_MLP="${USE_IMG_NEXT_MLP:-false}"

# Language/VLM loss weight (set 0 to mimic action-only/no-imgloss runs)
VLM_LOSS_WEIGHT="${VLM_LOSS_WEIGHT:-0}"

WANDB_PROJECT="${WANDB_PROJECT:-agilex_aloha_final}"
WANDB_ENTITY="${WANDB_ENTITY:-lvj2114-beijing-academy-of-artificial-intelligence}"

CONFIG_PATH="${CONFIG_PATH:-/share/project/lvjing/starVLA/starVLA/config/training/agilex_aloha_action_only_50step.yaml}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-results/AgilexAloha_final}"
MASTER_PORT="${MASTER_PORT:-29523}"

# Optional: load from checkpoint
PRETRAINED_CKPT="${PRETRAINED_CKPT:-/share/project/lvjing/starVLA/results/AgilexAloha_final/agilex_aloha_final_multistage_stage_4/checkpoints/steps_2000_pytorch_model.pt}"
RELOAD_MODULES="${RELOAD_MODULES:-qwen_vl_interface}"

# Steps cache: directory path recommended for multi-dataset mixtures.
STEPS_CACHE_PATH="${STEPS_CACHE_PATH:-${RUN_ROOT_DIR}/steps_cache/agilex_aloha_final}"
WRITE_STEPS_CACHE="${WRITE_STEPS_CACHE:-true}"

# ----------------------------------------------------------------------------
# 根据 COT_MODE 派生开关
# ----------------------------------------------------------------------------
ENABLE_LATENT_REASONING="${ENABLE_LATENT_REASONING:-true}"

case "${COT_MODE}" in
  none)
    SCHEDULED_STAGE=0
    ENABLE_LATENT_REASONING="false"
    USE_REASONING_FILM="false"
    USE_REASONING_SUMMARY="false"
    ;;
  vlm_seen_no_out)
    SCHEDULED_STAGE=1
    ENABLE_LATENT_REASONING="false"
    USE_REASONING_FILM="false"
    USE_REASONING_SUMMARY="false"
    ;;
  explicit)
    SCHEDULED_STAGE=1
    ENABLE_LATENT_REASONING="false"
    USE_REASONING_FILM="false"
    USE_REASONING_SUMMARY="false"
    ;;
  implicit)
    SCHEDULED_STAGE=4
    ENABLE_LATENT_REASONING="true"
    # USE_REASONING_FILM/SUMMARY 保持外部传入默认
    ;;
  *)
    echo "❌ 无效的 COT_MODE=${COT_MODE}，可选：none/vlm_seen_no_out/explicit/implicit"
    exit 1
    ;;
esac

if [[ -n "${PRETRAINED_CKPT}" ]]; then
  while [[ ! -f "${PRETRAINED_CKPT}" ]]; do
    echo "⏳ PRETRAINED_CKPT 不存在，等待生成: ${PRETRAINED_CKPT}"
    echo "   每 3 分钟检查一次..."
    sleep 180
  done
  echo "✅ 检测到 PRETRAINED_CKPT: ${PRETRAINED_CKPT}"
fi

OUTPUT_DIR="${RUN_ROOT_DIR}/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"
cp "$0" "${OUTPUT_DIR}/run_command.sh"

# ----------------------------------------------------------------------------
# 训练配置覆盖项（与 run_starvla_libero.sh 对齐）
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

  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH}"
  --datasets.vla_data.bridge_reasoning.stage "${SCHEDULED_STAGE}"
  --datasets.vla_data.ecot.scheduled_stage "${SCHEDULED_STAGE}"

  --datasets.vla_data.bridge_annotations.steps_cache_path "${STEPS_CACHE_PATH}"
  --datasets.vla_data.bridge_annotations.write_steps_cache "${WRITE_STEPS_CACHE}"

  --framework.training_stage "${TRAINING_STAGE}"
  --framework.cot_mode "${COT_MODE}"
  --framework.enable_latent_reasoning "${ENABLE_LATENT_REASONING}"

  --framework.action_model.action_model_type "${ACTION_DIT_TYPE}"
  --framework.action_model.diffusion_model_cfg.dropout "${DIFFUSION_MODEL_DROPOUT}"
  --framework.action_model.use_reasoning_film "${USE_REASONING_FILM}"
  --framework.action_model.use_img_next_mlp_compress "${USE_IMG_NEXT_MLP}"
  --framework.action_model.reasoning_film_first_k "${REASONING_FILM_FIRST_K}"
  --framework.action_model.reasoning_film_dropout "${REASONING_FILM_DROPOUT}"
  --framework.action_model.reasoning_film_hidden "${REASONING_FILM_HIDDEN}"
  --framework.action_model.use_reasoning_summary "${USE_REASONING_SUMMARY}"
  --framework.action_model.reasoning_summary_tokens "${REASONING_SUMMARY_TOKENS}"
  --framework.action_model.reasoning_summary_heads "${REASONING_SUMMARY_HEADS}"
  --framework.action_model.reasoning_summary_dropout "${REASONING_SUMMARY_DROPOUT}"

  --framework.img_next.enable "true"
  --framework.img_next.use_teacher "false"
  --framework.img_next.loss_weight "0.1"
  --datasets.vla_data.bridge_reasoning.include_action_tokens "false"
  --framework.latent_reasoning.vlm_loss_weight "0"
  --datasets.vla_data.bridge_reasoning.vlm_loss_weight "0"
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
  --wandb_project "${WANDB_PROJECT}"
  --wandb_entity "${WANDB_ENTITY}"
)

echo "============================================================================"
echo " Agilex Cobot Magic ECOT Training"
echo " Config : ${CONFIG_PATH}"
echo " Run ID : ${RUN_ID}"
echo " Output : ${OUTPUT_DIR}"
echo " GPUs   : ${NUM_GPUS} (master_port=${MASTER_PORT})"
echo " Cache  : ${STEPS_CACHE_PATH} (write=${WRITE_STEPS_CACHE})"
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
