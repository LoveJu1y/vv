#!/usr/bin/env bash
# ============================================================================
# Agilex Cobot Magic (LeRobot) :: Single-task ECOT Launcher (4 runs sequential)
# Runs 4 single-directory experiments in order:
#   fruit -> pour -> stack -> storage
#
# Uses scripts/run_starvla_agilex_cobot_magic.sh as the base launcher.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RUN_ROOT_DIR="${RUN_ROOT_DIR:-results/AgilexCobotMagic_single}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-agilex_cobot_magic_single}"

TASKS=(fruit pour stack storage)

for task in "${TASKS[@]}"; do
  case "${task}" in
    fruit)   mix_key="agilex_cobot_magic_fruit" ;;
    pour)    mix_key="agilex_cobot_magic_pour" ;;
    stack)   mix_key="agilex_cobot_magic_stack" ;;
    storage) mix_key="agilex_cobot_magic_storage" ;;
    *) echo "Unknown task: ${task}"; exit 1 ;;
  esac

  export RUN_ROOT_DIR
  export RUN_ID="${RUN_ID_PREFIX}_${task}"
  export STEPS_CACHE_PATH="${REPO_ROOT}/${RUN_ROOT_DIR}/steps_cache/${mix_key}"

  echo "============================================================================"
  echo "[Agilex Single4] task=${task}"
  echo "  data_mix        : ${mix_key}"
  echo "  RUN_ROOT_DIR    : ${RUN_ROOT_DIR}"
  echo "  RUN_ID          : ${RUN_ID}"
  echo "  STEPS_CACHE_PATH: ${STEPS_CACHE_PATH}"
  echo "============================================================================"

  bash scripts/run_starvla_agilex_cobot_magic.sh \
    "$@" \
    --datasets.vla_data.data_mix "${mix_key}"
done

