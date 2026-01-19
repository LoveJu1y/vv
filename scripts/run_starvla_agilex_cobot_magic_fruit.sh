#!/usr/bin/env bash
# ============================================================================
# Agilex Cobot Magic (LeRobot) :: Single-task (fruit) ECOT Training Launcher
#  - Wrapper around scripts/run_starvla_agilex_cobot_magic.sh
#  - Fixes datasets.vla_data.data_mix=agilex_cobot_magic_fruit
# ============================================================================
set -euo pipefail

RUN_ROOT_DIR="${RUN_ROOT_DIR:-results/AgilexCobotMagic_single}"
RUN_ID="${RUN_ID:-agilex_cobot_magic_single_fruit}"
STEPS_CACHE_PATH="${STEPS_CACHE_PATH:-${RUN_ROOT_DIR}/steps_cache/agilex_cobot_magic_fruit}"

export RUN_ROOT_DIR
export RUN_ID
export STEPS_CACHE_PATH

bash scripts/run_starvla_agilex_cobot_magic.sh \
  --datasets.vla_data.data_mix agilex_cobot_magic_fruit \
  "$@"

