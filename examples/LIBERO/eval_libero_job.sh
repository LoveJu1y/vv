#!/bin/bash

# cd /share/project/baishuanghao/code/starVLA && ./scripts/few-shot/libero/eval/eval_libero_job_gr00t.sh

# ==== start inference server (background) ====
STAR_VLA_PYTHON="${STAR_VLA_PYTHON:-/share/project/lvjing/miniconda3/envs/starVLA/bin/python}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/share/project/lvjing/miniconda3/envs/libero/bin/python}"

require_python() {
  local label="$1"
  local value="$2"
  if [[ -x "${value}" ]]; then
    return 0
  fi
  if command -v "${value}" >/dev/null 2>&1; then
    return 0
  fi
  echo "❌ ${label} 不可用: ${value}" >&2
  return 1
}

require_python "STAR_VLA_PYTHON" "${STAR_VLA_PYTHON}"
require_python "LIBERO_PYTHON" "${LIBERO_PYTHON}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Prefer passing an absolute checkpoint path.
# Example:
#   YOUR_CKPT=/share/project/lvjing/starVLA/results/LiberoECOT_final/.../checkpoints/steps_xxx_pytorch_model.pt bash examples/LIBERO/eval_libero_job.sh
YOUR_CKPT="${YOUR_CKPT:-}"
if [[ -n "$YOUR_CKPT" ]]; then
  your_ckpt="$YOUR_CKPT"
else
  ckpt_root=/share/project/baishuanghao/code/starVLA/pretrained_models_few_shot/libero_10_2B_0.5
  ckpt_path=starvla_qwen_gr00t/final_model/pytorch_model.pt
  your_ckpt="$ckpt_root/$ckpt_path"
fi

run_dir="$(dirname "$(dirname "$your_ckpt")")"

# Output layout (A):
#   CKPT_DIR=.../checkpoints
#   CKPT_BASENAME=steps_15000_pytorch_model
#   EVAL_DIR=${CKPT_DIR}/eval_libero_implicit/${CKPT_BASENAME}/
#     ├── server_logs/server_${port}.log
#     ├── logs/libero_goal.log ...
#     └── videos/libero_goal/...
CKPT_DIR="$(cd "$(dirname "$your_ckpt")" && pwd)"
CKPT_BASENAME="$(basename "${your_ckpt%.pt}")"
EVAL_DIR="${EVAL_DIR:-${CKPT_DIR}/eval_libero_implicit/${CKPT_BASENAME}}"
SERVER_LOG_DIR="${EVAL_DIR}/server_logs"
log_path="${LOG_PATH:-${EVAL_DIR}/logs}"
mkdir -p "${SERVER_LOG_DIR}" "${log_path}"

# Task suites to evaluate (default: 4-in-1).
# Override with: TASK_SUITES="libero_goal" or TASK_SUITES="libero_goal,libero_10"
TASK_SUITES="${TASK_SUITES:-libero_goal,libero_spatial,libero_object,libero_10}"
IFS=',' read -r -a task_suite_names <<< "${TASK_SUITES}"
echo "task_suite_names=${task_suite_names[*]}"

base_port="${BASE_PORT:-10093}"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

"${STAR_VLA_PYTHON}" deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${base_port} \
    --use_bf16 \
    > "${SERVER_LOG_DIR}/server_${base_port}.log" 2>&1 &

SERVER_PID=$!
echo ">>> server started, pid = $SERVER_PID"

# Wait for port to be ready.
"${STAR_VLA_PYTHON}" - <<PY
import os
import sys
import time

import websockets.sync.client

host="127.0.0.1"
port=int("${base_port}")
timeout_s=180
start=time.time()

for k in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(k, None)

while True:
    if time.time() - start > timeout_s:
        print(f"server not ready after {timeout_s}s: {host}:{port}", file=sys.stderr)
        sys.exit(1)
    try:
        uri = f"ws://{host}:{port}"
        conn = websockets.sync.client.connect(
            uri,
            compression=None,
            max_size=None,
            open_timeout=5,
            ping_interval=None,
        )
        conn.close()
        print(f"server ready: {host}:{port}")
        sys.exit(0)
    except Exception:
        time.sleep(2)
PY

# ==== start eval ====
export LIBERO_HOME="${LIBERO_HOME:-/share/project/baishuanghao/code/LIBERO}"
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
EVAL_PYTHONPATH="${REPO_ROOT}:${LIBERO_HOME}"
if [[ -n "${PYTHONPATH:-}" ]]; then
  EVAL_PYTHONPATH="${EVAL_PYTHONPATH}:${PYTHONPATH}"
fi

num_trials_per_task="${NUM_TRIALS_PER_TASK:-50}"
SAVE_VIDEOS="${SAVE_VIDEOS:-false}"

video_args=()
if [[ "${SAVE_VIDEOS}" == "true" ]]; then
  video_args+=(--args.save-videos)
fi

host="127.0.0.1"

for task_suite_name in "${task_suite_names[@]}"; do
    video_out_path="${EVAL_DIR}/videos/${task_suite_name}"
    LIBERO_HOME="${LIBERO_HOME}" \
    LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH}" \
    PYTHONPATH="${EVAL_PYTHONPATH}" \
    "${LIBERO_PYTHON}" ./examples/LIBERO/eval_libero1.py \
        --args.pretrained-path ${your_ckpt} \
        --args.host "$host" \
        --args.port $base_port \
        --args.task-suite-name "$task_suite_name" \
        --args.num-trials-per-task "$num_trials_per_task" \
        --args.video-out-path "${video_out_path}" \
        "${video_args[@]}" \
        --args.enable-latent-reasoning \
        --args.cot-mode implicit \
        --args.log_path ${log_path}
done

echo ">>> job finished"
