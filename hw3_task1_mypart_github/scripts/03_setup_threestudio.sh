#!/usr/bin/env bash
set -euo pipefail

# Reproducible cloud environment preparation for HW3 Task 1 Object B/C.
#
# The script is intentionally command-based. Running it without a subcommand
# prints help and does not install packages or start long jobs.
#
# The commands below are implemented by this script and are intended for
# rerunning the threestudio setup on a fresh AutoDL RTX 5090 cloud instance.
# The submitted GitHub repository keeps the copied final results under
# result_preview/object_B and result_preview/object_C.
#
# Typical cloud setup order:
#   bash scripts/03_setup_threestudio.sh torch
#   bash scripts/03_setup_threestudio.sh repo
#   bash scripts/03_setup_threestudio.sh core-deps
#   bash scripts/03_setup_threestudio.sh zero123-deps
#   bash scripts/03_setup_threestudio.sh check
#
# Before rerunning Object B text-to-3D generation, also run:
#   bash scripts/03_setup_threestudio.sh object-b-check

ASSET_ROOT="${HW3_ASSET_ROOT:-/root/autodl-tmp/hw3_task1_assets}"
THIRD_PARTY_DIR="${ASSET_ROOT}/third_party"
REPO_DIR="${THIRD_PARTY_DIR}/threestudio"
TAMING_DIR="${THIRD_PARTY_DIR}/taming-transformers"
ENV_NAME="${THREESTUDIO_ENV_NAME:-hw3_threestudio}"
CONDA_SH="${HW3_CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
ENV_PY="${HW3_THREESTUDIO_PYTHON:-/root/miniconda3/envs/${ENV_NAME}/bin/python}"
PYPI_MIRROR="${HW3_PYPI_MIRROR:-https://mirrors.aliyun.com/pypi/simple}"
HF_HOME="${HF_HOME:-${ASSET_ROOT}/third_party/huggingface_cache}"
SD15_SNAPSHOT_ROOT="${HF_HOME}/hub/models--runwayml--stable-diffusion-v1-5/snapshots"
OBJECT_B_SD15_PATH="${HW3_OBJECT_B_SD15_PATH:-${SD15_SNAPSHOT_ROOT}/451f4fe16113bff5a5d2269ed5ad43b0592e9a14}"
OBJECT_B_PROMPT="${HW3_OBJECT_B_PROMPT:-a small blue ceramic coffee mug with a large handle, full object, centered}"

usage() {
  cat <<EOF
Usage:
  bash scripts/03_setup_threestudio.sh <command>

Commands:
  help          Show this message.
  torch         Install or verify PyTorch CUDA 12.8 in ${ENV_NAME}.
  repo          Clone or verify the threestudio repository.
  core-deps     Install core Python/CUDA dependencies used by threestudio.
  zero123-deps  Install Stable Zero123 runtime dependencies.
  check         Verify the environment without modifying files.
  object-b-check
                Verify Object B LatentNeRF config and local Stable Diffusion v1-5 files.
  all           Run torch, repo, core-deps, zero123-deps, and check.

Environment overrides:
  HW3_ASSET_ROOT          Default: ${ASSET_ROOT}
  THREESTUDIO_ENV_NAME    Default: ${ENV_NAME}
  HW3_THREESTUDIO_PYTHON  Default: ${ENV_PY}
  HW3_PYPI_MIRROR         Default: ${PYPI_MIRROR}
  HW3_OBJECT_B_SD15_PATH  Default: ${OBJECT_B_SD15_PATH}
  HW3_OBJECT_B_PROMPT     Default: ${OBJECT_B_PROMPT}
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

load_conda() {
  [[ -f "${CONDA_SH}" ]] || die "Cannot find conda activation script: ${CONDA_SH}"
  # shellcheck source=/dev/null
  source "${CONDA_SH}"
}

ensure_env() {
  load_conda
  if ! conda info --envs | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    conda create -n "${ENV_NAME}" python=3.10 -y
  fi
  conda activate "${ENV_NAME}"
  [[ -x "${ENV_PY}" ]] || die "Environment Python not found: ${ENV_PY}"
}

network_hint() {
  if [[ -f /etc/network_turbo ]]; then
    # Common on AutoDL images; affects only the current shell.
    # shellcheck source=/dev/null
    source /etc/network_turbo
    echo "Loaded /etc/network_turbo"
  else
    echo "No /etc/network_turbo found; continue without it."
  fi
}

command_torch() {
  ensure_env
  network_hint

  echo "== PyTorch CUDA 12.8 check =="
  if "${ENV_PY}" - <<'PY'
import sys
try:
    import torch
    ok = (
        "+cu128" in torch.__version__
        and torch.version.cuda == "12.8"
        and torch.cuda.is_available()
    )
    print("torch:", torch.__version__)
    print("cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
        print("capability:", torch.cuda.get_device_capability(0))
    sys.exit(0 if ok else 1)
except Exception as exc:
    print("PyTorch check failed:", repr(exc))
    sys.exit(1)
PY
  then
    echo "PyTorch CUDA 12.8 already works in ${ENV_NAME}."
    return
  fi

  echo "Installing PyTorch CUDA 12.8 into ${ENV_NAME}..."
  "${ENV_PY}" -m pip install -i "${PYPI_MIRROR}" \
    --upgrade pip setuptools wheel \
    --timeout 120 --retries 10

  "${ENV_PY}" -m pip install -i "${PYPI_MIRROR}" \
    sympy mpmath typing-extensions filelock fsspec jinja2 MarkupSafe networkx numpy pillow \
    --timeout 120 --retries 10

  "${ENV_PY}" -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url "${PYPI_MIRROR}" \
    --timeout 120 --retries 10

  "${ENV_PY}" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
assert torch.cuda.is_available()
assert torch.version.cuda == "12.8"
PY
}

command_repo() {
  ensure_env
  network_hint
  mkdir -p "${THIRD_PARTY_DIR}" "${ASSET_ROOT}/logs"

  if [[ -d "${REPO_DIR}" ]]; then
    if [[ -f "${REPO_DIR}/launch.py" && -d "${REPO_DIR}/configs" ]]; then
      echo "Existing threestudio repository looks usable: ${REPO_DIR}"
      return
    fi
    die "Existing repository is incomplete: ${REPO_DIR}. Rename it manually before retrying."
  fi

  git clone --depth 1 https://github.com/threestudio-project/threestudio.git "${REPO_DIR}"
  [[ -f "${REPO_DIR}/launch.py" ]] || die "launch.py not found after clone."
  echo "Cloned threestudio: ${REPO_DIR}"
}

command_core_deps() {
  ensure_env
  [[ -f "${REPO_DIR}/launch.py" ]] || die "Run 'repo' first: ${REPO_DIR}"
  network_hint
  cd "${REPO_DIR}"

  echo "== Basic Python dependencies =="
  "${ENV_PY}" -m pip install -i "${PYPI_MIRROR}" \
    ninja cmake imageio imageio-ffmpeg opencv-python opencv-python-headless \
    huggingface_hub omegaconf jaxtyping typeguard rich tqdm matplotlib tensorboard \
    scipy scikit-image packaging wheel \
    --timeout 120 --retries 10

  echo "== nerfacc =="
  export TORCH_CUDA_ARCH_LIST="$("${ENV_PY}" - <<'PY'
import torch
major, minor = torch.cuda.get_device_capability(0)
print(f"{major}.{minor}")
PY
)"
  "${ENV_PY}" -m pip install --no-build-isolation \
    "git+https://github.com/KAIR-BAIR/nerfacc.git@v0.5.2"

  echo "== tiny-cuda-nn =="
  "${ENV_PY}" -m pip install -i "${PYPI_MIRROR}" \
    "setuptools==69.5.1" wheel packaging ninja cmake \
    --timeout 120 --retries 10

  export TCNN_CUDA_ARCHITECTURES="$("${ENV_PY}" - <<'PY'
import torch
major, minor = torch.cuda.get_device_capability(0)
print(f"{major}{minor}")
PY
)"
  "${ENV_PY}" -m pip install --no-build-isolation \
    "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
}

command_zero123_deps() {
  ensure_env
  [[ -f "${REPO_DIR}/launch.py" ]] || die "Run 'repo' first: ${REPO_DIR}"
  network_hint
  cd "${REPO_DIR}"

  echo "== Stable Zero123 runtime dependencies =="
  "${ENV_PY}" -m pip install -i "${PYPI_MIRROR}" \
    "numpy==1.26.4" \
    "opencv-python==4.9.0.80" \
    "opencv-python-headless==4.9.0.80" \
    "pytorch-lightning==2.0.0" \
    "torchmetrics==0.11.4" \
    "lightning-utilities>=0.7.0" \
    "libigl==2.4.1" \
    diffusers==0.19.3 \
    transformers==4.28.1 \
    accelerate einops kornia xatlas trimesh PyMCubes safetensors sentencepiece \
    "controlnet_aux==0.0.7" \
    "timm==0.6.13" \
    "open-clip-torch==2.20.0" \
    ftfy regex \
    "protobuf==3.20.3" \
    "wandb==0.15.12" \
    --timeout 120 --retries 10

  echo "== nvdiffrast =="
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
  export TORCH_CUDA_ARCH_LIST="$("${ENV_PY}" - <<'PY'
import torch
major, minor = torch.cuda.get_device_capability(0)
print(f"{major}.{minor}")
PY
)"
  export FORCE_CUDA=1
  "${ENV_PY}" -m pip install --no-cache-dir --no-build-isolation \
    "git+https://github.com/NVlabs/nvdiffrast.git" \
    --timeout 120 --retries 10

  echo "== envlight =="
  "${ENV_PY}" -m pip install --no-cache-dir \
    "git+https://github.com/ashawkey/envlight.git" \
    --timeout 120 --retries 10

  echo "== pysdf =="
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update
    apt-get install -y libeigen3-dev
  else
    echo "apt-get not found; pysdf may need Eigen headers installed manually."
  fi
  export CPATH="/usr/include/eigen3:${CPATH:-}"
  export CPLUS_INCLUDE_PATH="/usr/include/eigen3:${CPLUS_INCLUDE_PATH:-}"
  "${ENV_PY}" -m pip install -i "${PYPI_MIRROR}" pybind11 pysdf \
    --timeout 120 --retries 10

  echo "== taming-transformers source tree =="
  mkdir -p "${THIRD_PARTY_DIR}"
  if [[ -d "${TAMING_DIR}/taming" ]]; then
    echo "Existing taming-transformers source tree: ${TAMING_DIR}"
  else
    git clone --depth 1 https://github.com/CompVis/taming-transformers.git "${TAMING_DIR}"
  fi

  echo "Stable Zero123 dependencies installed."
}

command_check() {
  ensure_env
  [[ -f "${REPO_DIR}/launch.py" ]] || die "Missing threestudio repository: ${REPO_DIR}"
  cd "${REPO_DIR}"
  export PYTHONPATH="${TAMING_DIR}:${REPO_DIR}/extern:${PYTHONPATH:-}"

  "${ENV_PY}" - <<'PY'
import importlib
import sys
import torch

checks = [
    "torch",
    "nerfacc",
    "tinycudann",
    "cv2",
    "imageio",
    "omegaconf",
    "jaxtyping",
    "typeguard",
    "rich",
    "tqdm",
    "matplotlib",
    "PIL",
    "pytorch_lightning",
    "igl",
    "envlight",
    "nvdiffrast.torch",
    "diffusers",
    "transformers",
    "accelerate",
    "controlnet_aux",
    "taming",
    "open_clip",
    "wandb",
]

failed = []
for name in checks:
    try:
        importlib.import_module(name)
        print(f"{name}: OK")
    except Exception as exc:
        print(f"{name}: FAIL -> {exc!r}")
        failed.append(name)

try:
    ldm_zero123 = importlib.import_module("ldm_zero123")
    sys.modules["ldm"] = ldm_zero123
    importlib.import_module("ldm.models.diffusion.ddpm")
    print("ldm_zero123 alias check: OK")
except Exception as exc:
    print("ldm_zero123 alias check: FAIL ->", repr(exc))
    failed.append("ldm_zero123")

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))

if failed:
    raise SystemExit("FAILED_IMPORTS=" + ",".join(failed))

print("threestudio environment check OK")
PY

  "${ENV_PY}" launch.py --help >/dev/null
  echo "launch.py help check OK"
  ls configs | grep -E "dreamfusion|latentnerf|sdi|magic123|stable-zero123" || true
}

command_object_b_check() {
  ensure_env
  [[ -f "${REPO_DIR}/launch.py" ]] || die "Missing threestudio repository: ${REPO_DIR}"
  [[ -f "${REPO_DIR}/configs/latentnerf.yaml" ]] || die "Missing Object B config: ${REPO_DIR}/configs/latentnerf.yaml"

  echo "== Object B LatentNeRF / Stable Diffusion v1-5 check =="
  echo "threestudio repo: ${REPO_DIR}"
  echo "Stable Diffusion v1-5 path: ${OBJECT_B_SD15_PATH}"
  echo "Prompt: ${OBJECT_B_PROMPT}"

  local required_model_files=(
    "model_index.json"
    "scheduler/scheduler_config.json"
    "tokenizer/tokenizer_config.json"
    "text_encoder/model.safetensors"
    "unet/diffusion_pytorch_model.safetensors"
    "vae/diffusion_pytorch_model.safetensors"
  )

  [[ -d "${OBJECT_B_SD15_PATH}" ]] || die "Local Stable Diffusion v1-5 snapshot not found."
  for item in "${required_model_files[@]}"; do
    [[ -f "${OBJECT_B_SD15_PATH}/${item}" ]] || die "Missing Stable Diffusion v1-5 file: ${OBJECT_B_SD15_PATH}/${item}"
    echo "OK: ${OBJECT_B_SD15_PATH}/${item}"
  done

  cd "${REPO_DIR}"
  export HF_HOME="${HF_HOME}"
  export TRANSFORMERS_CACHE="${HF_HOME}/hub"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export DIFFUSERS_OFFLINE=1
  export OBJECT_B_SD15_PATH OBJECT_B_PROMPT

  "${ENV_PY}" - <<'PY'
import importlib
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

model_path = Path(os.environ["OBJECT_B_SD15_PATH"])
prompt = os.environ["OBJECT_B_PROMPT"]

checks = [
    "torch",
    "nerfacc",
    "tinycudann",
    "cv2",
    "imageio",
    "omegaconf",
    "pytorch_lightning",
    "diffusers",
    "transformers",
    "PIL",
    "trimesh",
    "xatlas",
]

failed = []
for name in checks:
    try:
        importlib.import_module(name)
        print(f"{name}: OK")
    except Exception as exc:
        print(f"{name}: FAIL -> {exc!r}")
        failed.append(name)

tokenizer = AutoTokenizer.from_pretrained(str(model_path / "tokenizer"), local_files_only=True)
token_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
print("prompt_token_count:", len(token_ids))
if len(token_ids) > 77:
    failed.append("prompt_too_long")

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
else:
    failed.append("cuda")

if failed:
    raise SystemExit("OBJECT_B_CHECK_FAILED=" + ",".join(failed))
PY

  "${ENV_PY}" launch.py --help >/dev/null
  echo "OBJECT_B_LATENTNERF_SD15_CHECK_OK"
}

command_all() {
  command_torch
  command_repo
  command_core_deps
  command_zero123_deps
  command_check
}

main() {
  local command="${1:-help}"
  case "${command}" in
    help|-h|--help) usage ;;
    torch) command_torch ;;
    repo) command_repo ;;
    core-deps) command_core_deps ;;
    zero123-deps) command_zero123_deps ;;
    check) command_check ;;
    object-b-check) command_object_b_check ;;
    all) command_all ;;
    *) usage; die "Unknown command: ${command}" ;;
  esac
}

main "$@"
