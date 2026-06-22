#!/usr/bin/env bash
set -euo pipefail

# Purpose:
#   Check a fresh cloud GPU environment before running the HW3 Task 1 asset
#   preparation pipeline. This script is read-only: it does not install
#   packages, create environments, download models, or delete files.

ASSET_ROOT="${HW3_ASSET_ROOT:-/root/autodl-tmp/hw3_task1_assets}"

echo "== HW3 Task 1 server check =="
echo "This script only prints environment diagnostics."
echo "Recommended asset root: ${ASSET_ROOT}"
echo

echo "== Current directory =="
pwd
echo

echo "== Current user =="
whoami
echo

echo "== Disk usage =="
df -h
echo

echo "== Recommended data disk =="
if [[ -d /root/autodl-tmp ]]; then
  ls -ld /root/autodl-tmp
else
  echo "/root/autodl-tmp not found. Check the cloud instance data disk before saving large files."
fi
echo

echo "== GPU =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found."
fi
echo

echo "== Python =="
if command -v python >/dev/null 2>&1; then
  python --version
  which python
else
  echo "python not found."
fi
echo

echo "== CUDA_HOME =="
echo "${CUDA_HOME:-CUDA_HOME is not set}"
echo

echo "== Conda =="
if command -v conda >/dev/null 2>&1; then
  conda --version
  conda info --envs
else
  echo "conda not found."
fi
echo

echo "== CUDA compiler =="
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version
else
  echo "nvcc not found."
  echo "3DGS CUDA extensions need nvcc. Use a CUDA Toolkit image or install the toolkit before building 3DGS."
fi
echo

echo "== Common tools =="
for cmd in git git-lfs ffmpeg colmap tmux cmake ninja; do
  if command -v "${cmd}" >/dev/null 2>&1; then
    printf "%-10s %s\n" "${cmd}" "$(command -v "${cmd}")"
  else
    printf "%-10s %s\n" "${cmd}" "not found"
  fi
done
echo

echo "== Network acceleration hint =="
if [[ -f /etc/network_turbo ]]; then
  echo "/etc/network_turbo exists. You can run 'source /etc/network_turbo' before GitHub or model downloads."
else
  echo "/etc/network_turbo not found. If GitHub or model downloads are slow, check the cloud platform's network options."
fi
