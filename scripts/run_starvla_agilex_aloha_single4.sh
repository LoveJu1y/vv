#!/usr/bin/env bash
# ============================================================================
# Agilex Aloha (LeRobot) :: Single-task ECOT Launcher (4 runs sequential)
# Runs 4 single-directory experiments in order:
#   stack_bowl_1110 -> storage_building_blocks_1109 -> storage_fruits_1114 -> storage_item_1124
#
# Uses scripts/run_starvla_agilex_aloha.sh as the base launcher.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RUN_ROOT_DIR="${RUN_ROOT_DIR:-results/AgilexAloha_single}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-agilex_aloha_single}"

TASKS=(stack_bowl_1110 storage_building_blocks_1109 storage_fruits_1114 storage_item_1124)
FAILED_TASKS=()

for task in "${TASKS[@]}"; do
  case "${task}" in
    stack_bowl_1110)               mix_key="agilex_aloha_stack_bowl_1110" ;;
    storage_building_blocks_1109)  mix_key="agilex_aloha_storage_building_blocks_1109" ;;
    storage_fruits_1114)           mix_key="agilex_aloha_storage_fruits_1114" ;;
    storage_item_1124)             mix_key="agilex_aloha_storage_item_1124" ;;
    *) echo "Unknown task: ${task}"; exit 1 ;;
  esac

  export RUN_ROOT_DIR
  export RUN_ID="${RUN_ID_PREFIX}_${task}"
  export STEPS_CACHE_PATH="${REPO_ROOT}/${RUN_ROOT_DIR}/steps_cache/${mix_key}"

  echo "============================================================================"
  echo "[Agilex Aloha Single4] task=${task}"
  echo "  data_mix        : ${mix_key}"
  echo "  RUN_ROOT_DIR    : ${RUN_ROOT_DIR}"
  echo "  RUN_ID          : ${RUN_ID}"
  echo "  STEPS_CACHE_PATH: ${STEPS_CACHE_PATH}"
  echo "============================================================================"

  if bash scripts/run_starvla_agilex_aloha.sh \
      "$@" \
      --datasets.vla_data.data_mix "${mix_key}"; then
    echo "✅ [Agilex Aloha Single4] task=${task} finished"
  else
    echo "❌ [Agilex Aloha Single4] task=${task} failed (continuing)"
    FAILED_TASKS+=("${task}")
  fi
done

if (( ${#FAILED_TASKS[@]} > 0 )); then
  echo "============================================================================"
  echo "❌ [Agilex Aloha Single4] Some tasks failed:"
  printf '  - %s\n' "${FAILED_TASKS[@]}"
  echo "============================================================================"
  exit 1
fi

echo "============================================================================"
echo "✅ [Agilex Aloha Single4] All tasks finished successfully"
echo "============================================================================"