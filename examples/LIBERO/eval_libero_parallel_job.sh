#!/usr/bin/env bash
# ============================================================================
# LIBERO 并行评测脚本（implicit latent reasoning / 多 server 多端口）
#
# 目标：参考 examples/SimplerEnv/star_bridge_lerobot_latent.sh 的模式：
#   - 为每个 suite 启动一个独立的 websocket server（不同 port）
#   - 同时启动多个 eval 进程（每个 suite 连接自己的 port）
#
# 用法：
#   YOUR_CKPT=/abs/path/to/steps_xxx_pytorch_model.pt bash examples/LIBERO/eval_libero_parallel_job.sh
# 或：
#   bash examples/LIBERO/eval_libero_parallel_job.sh /abs/path/to/steps_xxx_pytorch_model.pt
#
# 可用环境变量：
#   TASK_SUITES            逗号分隔，默认 libero_goal,libero_spatial,libero_object,libero_10
#   NUM_TRIALS_PER_TASK    每个任务 rollout 次数，默认 50
#   BASE_PORT              起始端口，默认 10093（会占用 BASE_PORT..BASE_PORT+N-1）
#   CUDA_VISIBLE_DEVICES   GPU 池（逗号分隔）；server 会轮询分配到这些 GPU
#   GPU_ID                 未设置 CUDA_VISIBLE_DEVICES 时使用的单个 GPU（默认 0）
#   STAR_VLA_PYTHON         server 端 python（默认 /share/project/lvjing/miniconda3/envs/starVLA/bin/python）
#   LIBERO_PYTHON           eval 端 python（默认 /share/project/lvjing/miniconda3/envs/libero/bin/python）
#   LIBERO_HOME            默认 /share/project/baishuanghao/code/LIBERO
#   SAVE_VIDEOS            true/false，默认 false（仅 true 时保存 mp4）
#   EVAL_DIR               覆写输出目录（默认见下方 A 结构）
#
# 输出目录（A）：
#   CKPT_DIR=.../checkpoints
#   CKPT_BASENAME=steps_15000_pytorch_model
#   EVAL_DIR=${CKPT_DIR}/eval_libero_implicit_parallel/${CKPT_BASENAME}/
#     ├── server_logs/server_${port}.log
#     ├── logs/${suite}.log            (eval_libero1.py 写)
#     ├── logs/${suite}.stdout.log     (脚本重定向)
#     └── videos/${suite}/...          (SAVE_VIDEOS=true 才写)
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}" || exit 1

# Default python interpreters (override via env vars).
STAR_VLA_PYTHON="${STAR_VLA_PYTHON:-/share/project/lvjing/miniconda3/envs/starVLA/bin/python}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/share/project/lvjing/miniconda3/envs/libero/bin/python}"

# Default checkpoint (override by arg $1 or YOUR_CKPT).
DEFAULT_CKPT_PATH="${DEFAULT_CKPT_PATH:-/share/project/lvjing/starVLA/results/LiberoECOT_final/libero_all_DITB_LR1E-4_LR1E-5_BTS16_40K_1lr/checkpoints/steps_40000_pytorch_model.pt}"

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

CKPT_PATH="${1:-${YOUR_CKPT:-${DEFAULT_CKPT_PATH:-}}}"
if [[ -z "${CKPT_PATH}" ]]; then
  echo "❌ 请提供 checkpoint 路径（建议绝对路径），例如："
  echo "   YOUR_CKPT=/share/.../steps_15000_pytorch_model.pt bash $0"
  echo "或在脚本/环境变量里设置 DEFAULT_CKPT_PATH"
  exit 1
fi
if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "❌ 找不到模型文件: ${CKPT_PATH}"
  exit 1
fi

TASK_SUITES="${TASK_SUITES:-libero_goal,libero_spatial,libero_object,libero_10}"
IFS=',' read -r -a SUITES <<< "${TASK_SUITES}"
if [[ "${#SUITES[@]}" -eq 0 ]]; then
  echo "❌ TASK_SUITES 为空"
  exit 1
fi

BASE_PORT="${BASE_PORT:-10093}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
SAVE_VIDEOS="${SAVE_VIDEOS:-false}"

GPU_ID="${GPU_ID:-0}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi
IFS=',' read -r -a CUDA_DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#CUDA_DEVICES[@]}"

export LIBERO_HOME="${LIBERO_HOME:-/share/project/baishuanghao/code/LIBERO}"
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"

EVAL_PYTHONPATH="${REPO_ROOT}:${LIBERO_HOME}"
if [[ -n "${PYTHONPATH:-}" ]]; then
  EVAL_PYTHONPATH="${EVAL_PYTHONPATH}:${PYTHONPATH}"
fi

CKPT_DIR="$(cd "$(dirname "${CKPT_PATH}")" && pwd)"
CKPT_BASENAME="$(basename "${CKPT_PATH%.pt}")"
EVAL_DIR="${EVAL_DIR:-${CKPT_DIR}/eval_libero_implicit_parallel/${CKPT_BASENAME}}"
SERVER_LOG_DIR="${EVAL_DIR}/server_logs"
LOG_DIR="${EVAL_DIR}/logs"
VIDEO_DIR="${EVAL_DIR}/videos"
mkdir -p "${SERVER_LOG_DIR}" "${LOG_DIR}" "${VIDEO_DIR}"

video_args=()
if [[ "${SAVE_VIDEOS}" == "true" ]]; then
  video_args+=(--args.save-videos)
fi

echo "======================================================"
echo "📊 LIBERO 并行评测（implicit latent reasoning）"
echo "------------------------------------------------------"
echo "Checkpoint : ${CKPT_PATH}"
echo "Suites     : ${SUITES[*]}"
echo "Ports      : from ${BASE_PORT} (N=${#SUITES[@]})"
echo "GPU Pool   : ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS} GPUs)"
echo "Trials     : ${NUM_TRIALS_PER_TASK}"
echo "Videos     : ${SAVE_VIDEOS}"
echo "Output Dir : ${EVAL_DIR}"
echo "======================================================"

server_pids=()
server_ports=()
eval_pids=()

cleanup_port() {
  local port="$1"
  local stale_pids=""
  if command -v pgrep >/dev/null 2>&1; then
    stale_pids="$(pgrep -f "deployment/model_server/server_policy.py.*--port ${port}" || true)"
  else
    stale_pids="$(ps aux | grep "deployment/model_server/server_policy.py" | grep "--port ${port}" | grep -v grep | awk '{print $2}' || true)"
  fi
  if [[ -n "${stale_pids}" ]]; then
    echo "🧹 清理端口 ${port} 上的旧 server: ${stale_pids}"
    kill ${stale_pids} 2>/dev/null || true
    sleep 2
    kill -9 ${stale_pids} 2>/dev/null || true
  fi
}

cleanup() {
  for pid in "${eval_pids[@]}"; do
    if ps -p "${pid}" > /dev/null 2>&1; then
      wait "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${server_pids[@]}"; do
    if ps -p "${pid}" > /dev/null 2>&1; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  sleep 2
  for pid in "${server_pids[@]}"; do
    if ps -p "${pid}" > /dev/null 2>&1; then
      kill -9 "${pid}" 2>/dev/null || true
    fi
  done
  for port in "${server_ports[@]}"; do
    cleanup_port "${port}"
  done
}
trap cleanup EXIT

wait_port_ready() {
  local port="$1"
  echo "⏳ 等待 server 就绪: 127.0.0.1:${port}"
  "${STAR_VLA_PYTHON}" - <<PY
import os
import sys
import time

try:
    import websockets.sync.client
except Exception as e:
    print(
        "missing python dependency for port check: websockets (sync client). "
        "Install it in STAR_VLA_PYTHON env, e.g. `pip install websockets>=11`.",
        file=sys.stderr,
    )
    raise

host="127.0.0.1"
port=int("${port}")
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
        # If handshake succeeded, the server is ready. Close immediately.
        conn.close()
        print(f"server ready: {host}:{port}")
        sys.exit(0)
    except Exception:
        time.sleep(2)
PY
}

start_server() {
  local gpu_id="$1"
  local port="$2"
  local log_file="${SERVER_LOG_DIR}/server_${port}.log"
  cleanup_port "${port}"
  echo "▶️  启动 server: GPU=${gpu_id} port=${port} log=${log_file}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${STAR_VLA_PYTHON}" deployment/model_server/server_policy.py \
    --ckpt_path "${CKPT_PATH}" \
    --port "${port}" \
    --use_bf16 \
    > "${log_file}" 2>&1 &
  server_pids+=("$!")
  server_ports+=("${port}")
}

run_suite_eval() {
  local suite="$1"
  local port="$2"
  local suite_video_dir="${VIDEO_DIR}/${suite}"
  local stdout_log="${LOG_DIR}/${suite}.stdout.log"

  mkdir -p "${suite_video_dir}"
  echo "🧪 启动评测: suite=${suite} port=${port} stdout_log=${stdout_log}"
  LIBERO_HOME="${LIBERO_HOME}" \
  LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH}" \
  PYTHONPATH="${EVAL_PYTHONPATH}" \
  "${LIBERO_PYTHON}" ./examples/LIBERO/eval_libero1.py \
    --args.pretrained-path "${CKPT_PATH}" \
    --args.host "127.0.0.1" \
    --args.port "${port}" \
    --args.task-suite-name "${suite}" \
    --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
    --args.video-out-path "${suite_video_dir}" \
    "${video_args[@]}" \
    --args.enable-latent-reasoning \
    --args.cot-mode implicit \
    --args.log_path "${LOG_DIR}" \
    > "${stdout_log}" 2>&1 &
  eval_pids+=("$!")
}

for idx in "${!SUITES[@]}"; do
  suite="${SUITES[$idx]}"
  port=$((BASE_PORT + idx))
  gpu_id="${CUDA_DEVICES[$((idx % NUM_GPUS))]}"

  start_server "${gpu_id}" "${port}"
done

if (( NUM_GPUS < ${#SUITES[@]} )); then
  echo "⚠️  当前 GPU 池只有 ${NUM_GPUS} 张卡，但 suite 数为 ${#SUITES[@]}，将会在同一张 GPU 上启动多个 server，可能导致 OOM/变慢。"
fi

echo ""
echo "⏳ 等待所有 servers 就绪..."
for idx in "${!SUITES[@]}"; do
  port=$((BASE_PORT + idx))
  wait_port_ready "${port}"
done

for idx in "${!SUITES[@]}"; do
  suite="${SUITES[$idx]}"
  port=$((BASE_PORT + idx))
  run_suite_eval "${suite}" "${port}"
done

echo ""
echo "🚀 已启动 ${#server_pids[@]} 个 server + ${#eval_pids[@]} 个 eval，等待完成..."
for pid in "${eval_pids[@]}"; do
  wait "${pid}"
done

echo ""
echo "✅ LIBERO 并行评测完成：${EVAL_DIR}"
