#!/usr/bin/env bash
# ==========================================================================
# Batch launcher to evaluate every checkpoint in a directory using
# star_bridge_lerobot_latent.sh. Designed for multi-GPU servers (default 8).
# Supports running up to NUM_SLOTS concurrent evaluations (default = #GPUs).
# Usage:
#   bash run_all_ckpts_lerobot_latent.sh \
#       /share/.../bridge_lerobot_fullcot_stage_bt164/checkpoints [MIN_STEP]
# Environment overrides:
#   GPU_LIST       Comma list of GPU ids to use (default "0,1,2,3,4,5,6,7")
#   NUM_SLOTS      Max concurrent ckpt evaluations (default = len(GPU_LIST))
#   BASE_PORT      Starting port offset (default 25000)
#   PORT_STRIDE    Per-job port spacing (default 500)
#   TSET_NUM       Repeats per task (default 1)
#   NUM_EPISODES   Episodes per task (default 24)
#   MIN_STEP       Only evaluate ckpts with steps >= MIN_STEP (default 0)
#   MAX_STEP       Only evaluate ckpts with steps <= MAX_STEP (default unset)
#   LOG_ROOT       Root dir for logs (default <ckpt_dir>/eval_all_stage4_parallel)
#   OMP_NUM_THREADS / MKL_NUM_THREADS  Thread limits to avoid CPU thrashing (default 1 here)
# ==========================================================================
set -euo pipefail

# Limit per-process CPU threads unless caller overrides
# export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
# export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
# export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

CKPT_DIR=${1:-/share/project/lvjing/starVLA/results/BridgeFinal_Action/bridge_none_stage0_final1/checkpoints}
MIN_STEP_ARG=${2:-}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
COT_MODE=${COT_MODE:-none}
IMG_NEXT_COUNT=${IMG_NEXT_COUNT:-0}
MIN_STEP=${MIN_STEP:-20000}
MAX_STEP=${MAX_STEP:-}

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
if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "❌ checkpoint 目录不存在: ${CKPT_DIR}" >&2
  exit 1
fi

GPU_LIST=${GPU_LIST:-"0"}
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_LIST}"
NUM_GPUS=${#GPU_ARRAY[@]}
if (( NUM_GPUS == 0 )); then
  echo "❌ GPU_LIST 为空" >&2
  exit 1
fi
NUM_SLOTS=${NUM_SLOTS:-${NUM_GPUS}}
BASE_PORT=${BASE_PORT:-25000}
PORT_STRIDE=${PORT_STRIDE:-500}
TSET_NUM=${TSET_NUM:-1}
NUM_EPISODES=${NUM_EPISODES:-24}
LOG_ROOT=${LOG_ROOT:-${CKPT_DIR}/eval_all_lerobot_latent_parallel}
mkdir -p "${LOG_ROOT}"

echo "======================================================"
echo "📊 Batch Stage-4 Eval"
echo "Checkpoint dir : ${CKPT_DIR}"
echo "GPU list       : ${GPU_LIST}"
echo "Slots          : ${NUM_SLOTS}"
echo "Base port      : ${BASE_PORT} (stride ${PORT_STRIDE})"
echo "Log root       : ${LOG_ROOT}"
echo "Min step       : ${MIN_STEP}"
echo "Max step       : ${MAX_STEP:-<unset>}"
echo "======================================================"

CKPTS=$(ls "${CKPT_DIR}"/steps_*_pytorch_model.pt 2>/dev/null | sort -V)
if [[ -z "${CKPTS}" ]]; then
  echo "❌ 未找到 steps_*_pytorch_model.pt 文件" >&2
  exit 1
fi

# Filter ckpts by step range (steps_XXXXX_pytorch_model.pt)
FILTERED_CKPTS=()
for ckpt in ${CKPTS}; do
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

pids=()
running=0
idx=0
ckpt_dirs=()
for ckpt in "${FILTERED_CKPTS[@]}"; do
  gpu="${GPU_ARRAY[$((idx % NUM_GPUS))]}"
  job_port=$((BASE_PORT + idx * PORT_STRIDE))
  ckpt_name="$(basename "${ckpt%.*}")"
  ckpt_log_dir="${LOG_ROOT}/${ckpt_name}"
  mkdir -p "${ckpt_log_dir}"
  ckpt_dirs+=("${ckpt_log_dir}")

  echo "------------------------------------------------------"
  echo "▶️  Launch ${ckpt_name} on GPU ${gpu} | port ${job_port}"
  echo "    Logs -> ${ckpt_log_dir}"

  (
    TSET_NUM=${TSET_NUM} \
    NUM_EPISODES=${NUM_EPISODES} \
    BASE_PORT=${job_port} \
    LOG_DIR="${ckpt_log_dir}" \
    CUDA_VISIBLE_DEVICES=${gpu} \
    COT_MODE=${COT_MODE} \
    IMG_NEXT_COUNT=${IMG_NEXT_COUNT} \
      bash examples/SimplerEnv/star_bridge_lerobot_latent.sh "${ckpt}"
  ) &
  pids+=($!)
  running=$((running + 1))
  idx=$((idx + 1))

  if (( running >= NUM_SLOTS )); then
    wait -n
    running=$((running - 1))
  fi
  sleep 2
done

wait

# ---------------------------------------------------------------------------
# Compute per-ckpt success summary (scan ALL ckpt dirs under LOG_ROOT)
# This makes the script resumable: reruns will still summarise earlier ckpts.
# ---------------------------------------------------------------------------
for ckpt_log_dir in "${LOG_ROOT}"/steps_*_pytorch_model; do
  if [[ ! -d "${ckpt_log_dir}" ]]; then
    continue
  fi
  python - "${ckpt_log_dir}" <<'PY' || true
import glob, os, sys, re
log_dir = sys.argv[1]
logs = sorted(glob.glob(os.path.join(log_dir, "*stage4_*.log.*")))
values = []
for path in logs:
    val = None
    with open(path, "r") as f:
        for line in f:
            if "Average success" in line:
                parts = line.strip().split()
                if parts:
                    try:
                        val = float(parts[-1])
                    except ValueError:
                        pass
    if val is not None:
        values.append((os.path.basename(path), val))

if not values:
    print(f"[Summary] {log_dir}: 未找到 Average success 记录")
    sys.exit(0)

mean = sum(v for _, v in values) / len(values)
summary_path = os.path.join(log_dir, "success_summary.txt")
with open(summary_path, "w") as f:
    f.write("Average success per log:\n")
    for name, val in values:
        f.write(f"{name}: {val:.6f}\n")
    f.write(f"\nMean success across {len(values)} logs: {mean:.6f}\n")

print(f"[Summary] {log_dir}: mean success = {mean:.6f} (details -> success_summary.txt)")
PY
done

# ---------------------------------------------------------------------------
# Aggregate ckpt-wise mean success into a global summary
# ---------------------------------------------------------------------------
OVERALL_SUMMARY="${LOG_ROOT}/overall_success_summary.txt"
python - "${LOG_ROOT}" "${OVERALL_SUMMARY}" <<'PY' || true
import os, sys, re
root, out_path = sys.argv[1], sys.argv[2]
entries = []
for name in sorted(os.listdir(root)):
    ckpt_dir = os.path.join(root, name)
    summary_path = os.path.join(ckpt_dir, "success_summary.txt")
    if not os.path.isdir(ckpt_dir) or not os.path.exists(summary_path):
        continue
    mean_val = None
    with open(summary_path, "r") as f:
        for line in f:
            m = re.search(r"Mean success.*:\s*([0-9.+-eE]+)", line)
            if m:
                try:
                    mean_val = float(m.group(1))
                except ValueError:
                    mean_val = None
                break
    if mean_val is not None:
        entries.append((name, mean_val))

if not entries:
    print(f"[Overall] 未找到 success_summary.txt，跳过全局汇总")
    sys.exit(0)

with open(out_path, "w") as f:
    f.write("Checkpoint\tMeanSuccess\n")
    for name, val in entries:
        f.write(f"{name}\t{val:.6f}\n")

print(f"[Overall] 成功写入 {len(entries)} 条记录 -> {out_path}")
PY

echo "✅ 所有 checkpoints 测评完成。日志位于 ${LOG_ROOT}"
