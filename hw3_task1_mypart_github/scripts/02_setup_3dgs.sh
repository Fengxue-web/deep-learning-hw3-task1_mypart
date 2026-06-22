#!/usr/bin/env bash
set -euo pipefail

# Purpose:
#   Install the official gaussian-splatting repository for HW3 Task 1 Object A.
#   Run this after the hw3_3dgs conda environment already has a working
#   PyTorch CUDA 12.8 installation. The script is designed for fresh cloud GPU
#   instances where GitHub access may be unstable.

ASSET_ROOT="${HW3_ASSET_ROOT:-/root/autodl-tmp/hw3_task1_assets}"
THIRD_PARTY_DIR="${ASSET_ROOT}/third_party"
REPO_DIR="${THIRD_PARTY_DIR}/gaussian-splatting"
ENV_NAME="${HW3_3DGS_ENV_NAME:-hw3_3dgs}"
PYPI_MIRROR="${HW3_PYPI_MIRROR:-https://mirrors.aliyun.com/pypi/simple}"

echo "== 3D Gaussian Splatting setup =="
echo "Cloud root: ${ASSET_ROOT}"
echo "Target repo: ${REPO_DIR}"
echo "Conda env: ${ENV_NAME}"
echo

read -r -p "Install official gaussian-splatting into the existing conda environment? [y/N] " CONFIRM
if [[ "${CONFIRM}" != "y" && "${CONFIRM}" != "Y" ]]; then
  echo "Cancelled."
  exit 0
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Please initialize conda before running this script."
  exit 1
fi

CONDA_BASE="$(conda info --base)"
if [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  echo "Cannot find conda activation script: ${CONDA_BASE}/etc/profile.d/conda.sh"
  exit 1
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo
echo "== Verify PyTorch CUDA before building 3DGS extensions =="
python - <<'PY'
import sys
try:
    import torch
except Exception as exc:
    print("Failed to import torch:", repr(exc))
    sys.exit(1)

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("CUDA is not available in this environment.")
    sys.exit(1)
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
PY

echo
echo "== Optional network acceleration =="
if [[ -f /etc/network_turbo ]]; then
  # This is common on some cloud images and usually only affects the current shell.
  source /etc/network_turbo
  echo "Loaded /etc/network_turbo"
else
  echo "/etc/network_turbo not found. Continue without it."
fi
echo "http_proxy=${http_proxy:-}"
echo "https_proxy=${https_proxy:-}"
echo "HTTP_PROXY=${HTTP_PROXY:-}"
echo "HTTPS_PROXY=${HTTPS_PROXY:-}"

mkdir -p "${THIRD_PARTY_DIR}" "${ASSET_ROOT}/logs"
cd "${THIRD_PARTY_DIR}"

echo
echo "== Clone official gaussian-splatting =="
if [[ -d "${REPO_DIR}" ]]; then
  if [[ -f "${REPO_DIR}/train.py" \
        && -d "${REPO_DIR}/submodules/diff-gaussian-rasterization" \
        && -d "${REPO_DIR}/submodules/simple-knn" ]]; then
    echo "Existing repo looks complete, skip clone: ${REPO_DIR}"
  else
    BACKUP_DIR="${REPO_DIR}_failed_$(date +%Y%m%d_%H%M%S)"
    echo "Existing repo looks incomplete."
    echo "Rename it instead of deleting: ${BACKUP_DIR}"
    mv "${REPO_DIR}" "${BACKUP_DIR}"
  fi
fi

if [[ ! -d "${REPO_DIR}" ]]; then
  git clone --depth 1 --recursive --shallow-submodules \
    https://github.com/graphdeco-inria/gaussian-splatting.git "${REPO_DIR}"
fi

cd "${REPO_DIR}"
git submodule update --init --recursive --depth 1

echo
echo "== Repository check =="
pwd
ls -lah
ls -lah submodules

if [[ ! -d submodules/diff-gaussian-rasterization || ! -d submodules/simple-knn ]]; then
  echo "Required submodules are missing. Check GitHub/network access and retry."
  exit 1
fi

echo
echo "== Install Python dependencies =="
python -m pip install -i "${PYPI_MIRROR}" \
  plyfile tqdm opencv-python scipy matplotlib pillow tensorboard ninja cmake setuptools wheel \
  --timeout 120 --retries 10

echo
echo "== Build CUDA extensions =="
export TORCH_CUDA_ARCH_LIST="$(python - <<'PY'
import torch
major, minor = torch.cuda.get_device_capability(0)
print(f"{major}.{minor}")
PY
)"
echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

# --no-build-isolation is important because these submodules import torch in
# setup/build code. A temporary isolated build environment may not contain torch.
python -m pip install --no-build-isolation ./submodules/diff-gaussian-rasterization
python -m pip install --no-build-isolation ./submodules/simple-knn

echo
echo "== Import check =="
python - <<'PY'
import torch
import diff_gaussian_rasterization
import simple_knn._C
print("3DGS extensions import OK")
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("device:", torch.cuda.get_device_name(0))
PY

echo
echo "3DGS setup finished."
echo "Repo: ${REPO_DIR}"
echo "Conda env: ${ENV_NAME}"
