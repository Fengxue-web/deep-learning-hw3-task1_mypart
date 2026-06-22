#!/usr/bin/env bash
set -euo pipefail

ASSET_ROOT="${HW3_ASSET_ROOT:-/root/autodl-tmp/hw3_task1_assets}"
LOG_DIR="${ASSET_ROOT}/logs"
LOG_FILE="${1:-${LOG_DIR}/gpu_monitor.log}"
INTERVAL_SECONDS="${GPU_MONITOR_INTERVAL:-10}"

mkdir -p "${LOG_DIR}"

echo "== GPU monitor =="
echo "Log file: ${LOG_FILE}"
echo "Interval seconds: ${INTERVAL_SECONDS}"
echo "Stop with Ctrl+C."
echo

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found."
  exit 1
fi

while true; do
  nvidia-smi --query-gpu=timestamp,name,index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv | tee -a "${LOG_FILE}"
  sleep "${INTERVAL_SECONDS}"
done
