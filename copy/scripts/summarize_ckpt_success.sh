#!/usr/bin/env bash
# ==========================================================================
# Summarize eval logs for all checkpoints in a directory.
#  - For each ckpt subdir under LOG_ROOT, parse *stage4_*.log.* for
#    "Average success" and write success_summary.txt.
#  - Aggregate per-ckpt均值到 overall_success_summary.txt.
# Usage:
#   bash summarize_ckpt_success.sh /path/to/checkpoints [LOG_ROOT]
#    - If LOG_ROOT 未指定，默认 <ckpt_dir>/eval_all_lerobot_latent_parallel
# ==========================================================================
set -euo pipefail

CKPT_DIR=${1:-}
if [[ -z "${CKPT_DIR}" ]]; then
  echo "用法: $0 /path/to/checkpoints [LOG_ROOT]" >&2
  exit 1
fi

LOG_ROOT=${2:-${CKPT_DIR}/eval_all_lerobot_latent_parallel}
if [[ ! -d "${LOG_ROOT}" ]]; then
  echo "❌ LOG_ROOT 不存在: ${LOG_ROOT}" >&2
  exit 1
fi

echo "======================================================"
echo "📊 Summarize ckpt eval logs"
echo "Checkpoint dir : ${CKPT_DIR}"
echo "Log root       : ${LOG_ROOT}"
echo "======================================================"

ckpt_dirs=()
while IFS= read -r -d '' d; do
  ckpt_dirs+=("$d")
done < <(find "${LOG_ROOT}" -maxdepth 1 -mindepth 1 -type d -print0 | sort -z)

if [[ ${#ckpt_dirs[@]} -eq 0 ]]; then
  echo "❌ 在 ${LOG_ROOT} 下未找到 ckpt 子目录" >&2
  exit 1
fi

# Per-ckpt summary
for ckpt_log_dir in "${ckpt_dirs[@]}"; do
  python - "${ckpt_log_dir}" <<'PY' || true
import glob, os, sys
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

# Global summary
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

echo "✅ 汇总完成。全局结果: ${OVERALL_SUMMARY}"
