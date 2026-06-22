#!/usr/bin/env bash
set -euo pipefail

# Purpose:
#   Prepare the basic Linux tools and directory layout for HW3 Task 1 on a
#   fresh cloud GPU instance. This script creates folders and installs common
#   packages needed by COLMAP, 3DGS, video processing, and long-running jobs.
#   It never deletes files.

ASSET_ROOT="${HW3_ASSET_ROOT:-/root/autodl-tmp/hw3_task1_assets}"

echo "== HW3 Task 1 system setup =="
echo "Cloud asset root: ${ASSET_ROOT}"
echo "This script creates folders and installs common system packages."
echo "It does not delete files or modify Git history."
echo

read -r -p "Continue with folder creation and apt package installation? [y/N] " CONFIRM
if [[ "${CONFIRM}" != "y" && "${CONFIRM}" != "Y" ]]; then
  echo "Cancelled."
  exit 0
fi

echo
echo "== Create cloud project folders =="
mkdir -p \
  "${ASSET_ROOT}/repo" \
  "${ASSET_ROOT}/third_party" \
  "${ASSET_ROOT}/data/object_A_raw/input" \
  "${ASSET_ROOT}/data/object_C_raw" \
  "${ASSET_ROOT}/outputs/A_3dgs" \
  "${ASSET_ROOT}/outputs/B_threestudio" \
  "${ASSET_ROOT}/outputs/C_zero123" \
  "${ASSET_ROOT}/final_assets/object_A" \
  "${ASSET_ROOT}/final_assets/object_B" \
  "${ASSET_ROOT}/final_assets/object_C" \
  "${ASSET_ROOT}/logs/gpu" \
  "${ASSET_ROOT}/report_materials/screenshots" \
  "${ASSET_ROOT}/report_materials/videos" \
  "${ASSET_ROOT}/report_materials/tables"

find "${ASSET_ROOT}" -maxdepth 2 -type d | sort
echo

if command -v sudo >/dev/null 2>&1; then
  SUDO="sudo"
else
  SUDO=""
fi

echo "== Install common system packages =="
${SUDO} apt-get update
${SUDO} apt-get install -y \
  git \
  git-lfs \
  wget \
  curl \
  unzip \
  zip \
  build-essential \
  cmake \
  ninja-build \
  ffmpeg \
  imagemagick \
  tmux \
  htop \
  libgl1 \
  libglib2.0-0 \
  libegl1 \
  libgles2

if command -v git-lfs >/dev/null 2>&1; then
  git lfs install
else
  echo "git-lfs command not found after apt install. Continue, but model/data downloads may need manual Git LFS setup."
fi

echo
echo "== Install COLMAP if available from apt =="
if command -v colmap >/dev/null 2>&1; then
  echo "COLMAP already exists: $(command -v colmap)"
else
  set +e
  ${SUDO} apt-get install -y colmap
  COLMAP_STATUS=$?
  set -e
  if [[ "${COLMAP_STATUS}" -ne 0 ]]; then
    echo "apt could not install COLMAP."
    echo "Fallback option after activating the target conda environment:"
    echo "conda install -c conda-forge colmap -y"
  fi
fi

echo
echo "System package setup finished."
echo "Recommended next checks:"
echo "  bash scripts/00_check_server.sh"
echo "  conda create -n hw3_3dgs python=3.10 -y"
echo "  conda create -n hw3_threestudio python=3.10 -y"
