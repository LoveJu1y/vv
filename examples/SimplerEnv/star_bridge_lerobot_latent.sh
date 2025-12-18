#!/usr/bin/env bash
# ============================================================================
# SimplerEnv 并行评测脚本（Stage-4 action-only / latent reasoning）
# 使用方法：
#   bash star_bridge_lerobot_latent.sh /path/to/steps_xxxxx_pytorch_model.pt
# 可用环境变量：
#   TSET_NUM              每个任务重复次数，默认 1
#   NUM_EPISODES          每个任务的 episode 数，默认 24
#   BASE_PORT             起始端口，默认 10120
#   GPU_ID                未设置 CUDA_VISIBLE_DEVICES 时使用的 GPU（默认 0）
#   CUDA_VISIBLE_DEVICES  可选，逗号分隔；脚本会在这些 GPU 上并行分配任务
#   LOG_DIR               日志目录（默认 ckpt_dir/eval_stage4_parallel）
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export star_vla_python=/share/project/lvjing/miniconda3/envs/starVLA/bin/python
export sim_python=/share/project/lvjing/miniconda3/envs/simpler_env/bin/python
export SimplerEnv_PATH=${SimplerEnv_PATH:-/share/project/lvjing/SimplerEnv}
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/share/project/lvjing/starVLA/qwen_cache

CKPT_PATH="${1:-/share/project/lvjing/starVLA/results/BridgeFinal_Action/bridge_lerobot_DITB_LR1E-4_LR1E-5_BTS16_60K_SUM_DR01/checkpoints/steps_60000_pytorch_model.pt}"
if [[ -z "${CKPT_PATH}" ]]; then
  echo "❌ 请提供模型路径，例如：bash $0 /share/.../steps_10000_pytorch_model.pt"
  exit 1
fi
if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "❌ 找不到模型文件: ${CKPT_PATH}"
  exit 1
fi

TSET_NUM="${TSET_NUM:-1}"
NUM_EPISODES="${NUM_EPISODES:-24}"
BASE_PORT="${BASE_PORT:-10220}"
GPU_ID="${GPU_ID:-0}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi
IFS=',' read -r -a CUDA_DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#CUDA_DEVICES[@]}"

CKPT_DIR="$(cd "$(dirname "${CKPT_PATH}")" && pwd)"
CKPT_BASENAME="$(basename "${CKPT_PATH%.*}")"
LOG_DIR="${LOG_DIR:-${CKPT_DIR}/eval_stage4_parallel60000}"
mkdir -p "${LOG_DIR}"

echo "======================================================"
echo "📊 Stage-4 SimplerEnv 并行评测"
echo "------------------------------------------------------"
echo "Checkpoint : ${CKPT_PATH}"
echo "Logs       : ${LOG_DIR}"
echo "Repeat     : ${TSET_NUM}"
echo "Episodes   : ${NUM_EPISODES}"
echo "Ports      : from ${BASE_PORT}"
echo "GPU Pool   : ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS} GPUs)"
echo "======================================================"

policyserver_pids=()
server_ports=()
eval_pids=()

cleanup_port() {
  local port="$1"
  local stale
  stale=$(ps aux | grep "server_policy.py" | grep "--port ${port}" | grep -v grep | awk '{print $2}' || true)
  if [[ -n "${stale}" ]]; then
    echo "   清理端口 ${port} 上的旧 server: ${stale}"
    kill -9 ${stale} 2>/dev/null || true
    sleep 1
  fi
}

start_server() {
  local gpu_id="$1"
  local port="$2"
  local server_logs="${LOG_DIR}/server_logs"
  mkdir -p "${server_logs}"
  local log_file="${server_logs}/${CKPT_BASENAME}_server_${port}.log"

  cleanup_port "${port}"
  echo "▶️  启动策略服务器 (GPU ${gpu_id}, port ${port})"
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${star_vla_python}" deployment/model_server/server_policy.py \
    --ckpt_path "${CKPT_PATH}" \
    --port "${port}" \
    --use_bf16 \
    > "${log_file}" 2>&1 &

  local pid=$!
  policyserver_pids+=("${pid}")
  server_ports+=("${port}")
  sleep 8

  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "⚠️  Server (port ${port}) 可能启动失败，请检查 ${log_file}"
  fi
}

stop_all_servers() {
  echo ""
  echo "⏳ 等待所有评测任务结束..."
  for pid in "${eval_pids[@]}"; do
    if ps -p "${pid}" > /dev/null 2>&1; then
      wait "${pid}" || true
    fi
  done

  echo "⏹  停止策略服务器..."
  for pid in "${policyserver_pids[@]}"; do
    if ps -p "${pid}" > /dev/null 2>&1; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  sleep 2

  for pid in "${policyserver_pids[@]}"; do
    if ps -p "${pid}" > /dev/null 2>&1; then
      kill -9 "${pid}" 2>/dev/null || true
    fi
  done

  for port in "${server_ports[@]}"; do
    cleanup_port "${port}"
  done

  eval_pids=()
  policyserver_pids=()
  server_ports=()
}

run_task() {
  local gpu_id="$1"
  local env_name="$2"
  local scene="$3"
  local robot="$4"
  local rgb_overlay="$5"
  local robot_x="$6"
  local robot_y="$7"
  local run_idx="$8"
  local port="$9"

  start_server "${gpu_id}" "${port}"

  local tag="run${run_idx}"
  local log_file="${LOG_DIR}/${CKPT_BASENAME}_stage4_${env_name}.log.${tag}"
  echo "🧪 任务 ${env_name} | 第 ${run_idx}/${TSET_NUM} 次 | GPU ${gpu_id} | 端口 ${port}"
  echo "   日志: ${log_file}"

  CUDA_VISIBLE_DEVICES="${gpu_id}" "${sim_python}" examples/SimplerEnv/start_simpler_env.py \
    --port "${port}" \
    --ckpt-path "${CKPT_PATH}" \
    --policy-setup widowx_bridge \
    --robot "${robot}" \
    --control-freq 5 \
    --sim-freq 500 \
    --max-episode-steps 120 \
    --env-name "${env_name}" \
    --scene-name "${scene}" \
    --rgb-overlay-path "${rgb_overlay}" \
    --robot-init-x "${robot_x}" "${robot_x}" 1 \
    --robot-init-y "${robot_y}" "${robot_y}" 1 \
    --obj-variation-mode episode \
    --obj-episode-range 0 "${NUM_EPISODES}" \
    --robot-init-rot-quat-center 0 0 0 1 \
    --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
    --enable-latent-reasoning \
    --thinking-token-count 3 \
    --logging-dir "${LOG_DIR}" \
    > "${log_file}" 2>&1 &

  eval_pids+=("$!")
}

trap stop_all_servers EXIT

declare -a TASKS=(
  "StackGreenCubeOnYellowCubeBakedTexInScene-v0|bridge_table_1_v1|widowx|${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png|0.147|0.028"
  "PutCarrotOnPlateInScene-v0|bridge_table_1_v1|widowx|${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png|0.147|0.028"
  "PutSpoonOnTableClothInScene-v0|bridge_table_1_v1|widowx|${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png|0.147|0.028"
  "PutEggplantInBasketScene-v0|bridge_table_1_v2|widowx_sink_camera_setup|${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png|0.127|0.06"
)

task_index=0
for task_spec in "${TASKS[@]}"; do
  IFS='|' read -r ENV_NAME SCENE_NAME ROBOT_NAME RGB_PATH RX RY <<< "${task_spec}"
  for ((run_idx=1; run_idx<=TSET_NUM; run_idx++)); do
    gpu_id="${CUDA_DEVICES[$((task_index % NUM_GPUS))]}"
    port=$((BASE_PORT + task_index))
    run_task "${gpu_id}" "${ENV_NAME}" "${SCENE_NAME}" "${ROBOT_NAME}" "${RGB_PATH}" "${RX}" "${RY}" "${run_idx}" "${port}"
    task_index=$((task_index + 1))
  done
done

echo ""
echo "🚀 已启动 ${task_index} 个任务，等待完成..."
stop_all_servers

echo ""
echo "✅ Stage-4 并行评测完成，日志位于: ${LOG_DIR}"
