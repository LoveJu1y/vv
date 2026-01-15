#!/usr/bin/env bash
# ============================================================================
# Batch launcher to evaluate every checkpoint in a directory using
# examples/LIBERO/eval_libero_parallel_job.sh (multi-server, implicit latent).
#
# Usage:
#   bash examples/LIBERO/run_all_ckpts_libero_latent.sh /path/to/checkpoints [MIN_STEP]
#
# Notes:
#   - Checkpoints are evaluated SERIALly (one ckpt after another).
#   - Within each ckpt, suites may still be evaluated in parallel depending on
#     eval_libero_parallel_job.sh behavior.
#
# Environment overrides:
#   TASK_SUITES            Comma-separated suites to evaluate (default 4)
#   NUM_TRIALS_PER_TASK    Rollouts per task (default 50)
#   CUDA_VISIBLE_DEVICES   GPU pool for per-suite servers (comma-separated)
#   GPU_ID                Used if CUDA_VISIBLE_DEVICES is unset (default 0)
#   BASE_PORT              Starting port for first ckpt (default 10093)
#   PORT_STRIDE            Port offset per ckpt (default 50)
#   MIN_STEP               Only evaluate ckpts with steps >= MIN_STEP (default 20000)
#   MAX_STEP               Only evaluate ckpts with steps <= MAX_STEP (default unset)
#   SAVE_VIDEOS            true/false (default false)
#   STOP_ON_FAIL           true/false (default false; continue on failure)
# ============================================================================
set -euo pipefail

CKPT_DIR="${1:-/share/project/lvjing/starVLA/results/LiberoECOT_final/libero_all_DITB_LR1E-4_LR1E-5_BTS14_40K_1lr/checkpoints}"
MIN_STEP_ARG="${2:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}" || exit 1

if [[ -z "${CKPT_DIR}" ]]; then
  echo "❌ 请提供 checkpoints 目录，例如：bash $0 /share/.../checkpoints [MIN_STEP]" >&2
  exit 1
fi
if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "❌ checkpoint 目录不存在: ${CKPT_DIR}" >&2
  exit 1
fi

MIN_STEP="${MIN_STEP:-24000}"
MAX_STEP="${MAX_STEP:-}"
if [[ -n "${MIN_STEP_ARG}" ]]; then
  MIN_STEP="${MIN_STEP_ARG}"
fi
if [[ -n "${MIN_STEP}" ]] && ! [[ "${MIN_STEP}" =~ ^[0-9]+$ ]]; then
  echo "❌ MIN_STEP 必须是非负整数，当前为: ${MIN_STEP}" >&2
  exit 1
fi
if [[ -n "${MAX_STEP}" ]] && ! [[ "${MAX_STEP}" =~ ^[0-9]+$ ]]; then
  echo "❌ MAX_STEP 必须是非负整数（或留空），当前为: ${MAX_STEP}" >&2
  exit 1
fi

BASE_PORT="${BASE_PORT:-10093}"
PORT_STRIDE="${PORT_STRIDE:-50}"
SAVE_VIDEOS="${SAVE_VIDEOS:-false}"
STOP_ON_FAIL="${STOP_ON_FAIL:-false}"

if ! [[ "${BASE_PORT}" =~ ^[0-9]+$ ]]; then
  echo "❌ BASE_PORT 必须是非负整数，当前为: ${BASE_PORT}" >&2
  exit 1
fi
if ! [[ "${PORT_STRIDE}" =~ ^[0-9]+$ ]]; then
  echo "❌ PORT_STRIDE 必须是非负整数，当前为: ${PORT_STRIDE}" >&2
  exit 1
fi

echo "======================================================"
echo "📊 Batch LIBERO Eval (serial per-ckpt)"
echo "Checkpoint dir : ${CKPT_DIR}"
echo "Base port      : ${BASE_PORT} (stride ${PORT_STRIDE})"
echo "Min step       : ${MIN_STEP}"
echo "Max step       : ${MAX_STEP:-<unset>}"
echo "Save videos    : ${SAVE_VIDEOS}"
echo "Stop on fail   : ${STOP_ON_FAIL}"
echo "======================================================"

shopt -s nullglob
mapfile -t CKPTS < <(printf '%s\n' "${CKPT_DIR}"/steps_*_pytorch_model.pt | sort -V)
if (( ${#CKPTS[@]} == 0 )); then
  echo "❌ 未找到 steps_*_pytorch_model.pt 文件: ${CKPT_DIR}" >&2
  exit 1
fi

FILTERED_CKPTS=()
for ckpt in "${CKPTS[@]}"; do
  base="$(basename "${ckpt}")"
  if [[ "${base}" =~ ^steps_([0-9]+)_pytorch_model\.pt$ ]]; then
    step="${BASH_REMATCH[1]}"
    if (( step < MIN_STEP )); then
      continue
    fi
    if [[ -n "${MAX_STEP}" ]] && (( step > MAX_STEP )); then
      continue
    fi
    FILTERED_CKPTS+=("${ckpt}")
  else
    echo "⚠️ 跳过非标准命名 ckpt: ${ckpt}" >&2
  fi
done

if (( ${#FILTERED_CKPTS[@]} == 0 )); then
  echo "❌ 未找到满足 step 范围的 ckpt（MIN_STEP=${MIN_STEP}, MAX_STEP=${MAX_STEP:-<unset>}）" >&2
  exit 1
fi

idx=0
for ckpt in "${FILTERED_CKPTS[@]}"; do
  ckpt_name="$(basename "${ckpt%.*}")"
  ckpt_base_port=$((BASE_PORT + idx * PORT_STRIDE))
  echo "------------------------------------------------------"
  echo "▶️  Eval ${ckpt_name} | base_port=${ckpt_base_port}"

  if ! (
    YOUR_CKPT="${ckpt}" \
    BASE_PORT="${ckpt_base_port}" \
    SAVE_VIDEOS="${SAVE_VIDEOS}" \
      bash examples/LIBERO/eval_libero_parallel_job.sh "${ckpt}"
  ); then
    echo "⚠️  Eval failed: ${ckpt}" >&2
    if [[ "${STOP_ON_FAIL}" == "true" ]]; then
      exit 1
    fi
  fi

  idx=$((idx + 1))
done

echo "✅ 所有 checkpoints 测评完成：${CKPT_DIR}"
