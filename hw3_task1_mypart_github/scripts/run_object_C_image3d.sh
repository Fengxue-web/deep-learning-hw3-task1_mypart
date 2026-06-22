#!/usr/bin/env bash
set -euo pipefail

# Object C reproducible cloud single-image-to-3D workflow.
#
# This script records the Stable Zero123 route used to produce the submitted
# Object C result. It operates in the cloud workspace defined by HW3_ASSET_ROOT.
# The submitted repository stores the copied final result under
# result_preview/object_C. The script never starts a long job unless an explicit
# subcommand is provided, and it does not delete training outputs.
#
# Cloud reproduction order:
#   bash scripts/run_object_C_image3d.sh prepare-input
#   bash scripts/run_object_C_image3d.sh preflight
#   bash scripts/run_object_C_image3d.sh train-400
#   bash scripts/run_object_C_image3d.sh continue-1200
#   bash scripts/run_object_C_image3d.sh sweep-1200
#   bash scripts/run_object_C_image3d.sh timed-final-1200-threshold20
#   bash scripts/run_object_C_image3d.sh collect-threshold20

ASSET_ROOT="${HW3_ASSET_ROOT:-/root/autodl-tmp/hw3_task1_assets}"
ENV_NAME="${THREESTUDIO_ENV_NAME:-hw3_threestudio}"
CONDA_SH="${HW3_CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
ENV_PY="${HW3_THREESTUDIO_PYTHON:-/root/miniconda3/envs/${ENV_NAME}/bin/python}"
THREESTUDIO_DIR="${HW3_THREESTUDIO_DIR:-${ASSET_ROOT}/third_party/threestudio}"
TAMING_DIR="${HW3_TAMING_DIR:-${ASSET_ROOT}/third_party/taming-transformers}"

RAW_INPUT="${HW3_OBJECT_C_ORIGINAL:-${ASSET_ROOT}/data/object_C_raw/input.jpg}"
RGBA_INPUT="${HW3_OBJECT_C_RGBA:-${ASSET_ROOT}/data/object_C_raw/object_C_rgba.png}"
THREESTUDIO_INPUT="${THREESTUDIO_DIR}/load/images/object_C_rgba.png"
ZERO123_DIR="${THREESTUDIO_DIR}/load/zero123"
LOG_DIR="${ASSET_ROOT}/logs"
GPU_LOG_DIR="${LOG_DIR}/gpu"
FINAL_DIR="${ASSET_ROOT}/final_assets/object_C"
BASELINE_DIR="${FINAL_DIR}/object_C_baseline_stable_zero123_400step_pytorch_encoding"
SWEEP_DIR="${FINAL_DIR}/object_C_candidate_1200_threshold_sweep_float"
TIMED_FINAL_PREFIX="object_C_timed_1200_threshold20"

usage() {
  cat <<EOF
Usage:
  bash scripts/run_object_C_image3d.sh <command>

Commands:
  help                 Show this message.
  check-input          Verify the RGBA foreground image.
  prepare-input        Copy the RGBA image into threestudio/load/images.
  preflight            Verify image, weights, Python imports, and GPU.
  train-400            Train the low-memory 400-step baseline.
  continue-1200        Continue the 400-step run to 1200 steps.
  export-400-auto      Export the 400-step baseline mesh with auto threshold.
  sweep-1200           Export 1200-step meshes with fixed thresholds.
  timed-final-1200-threshold20
                       Reproduce the final timed Object C result: 1200 continuation
                       plus threshold_20 mesh export, with wall time and GPU logs.
  collect-threshold20  Print and verify the recommended threshold_20 result.

Important outputs:
  400-step baseline: ${BASELINE_DIR}
  1200 threshold sweep: ${SWEEP_DIR}
  Recommended Object C result: ${SWEEP_DIR}/threshold_20
  Timed final result prefix: ${FINAL_DIR}/${TIMED_FINAL_PREFIX}_<timestamp>
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

activate_env() {
  load_conda
  conda activate "${ENV_NAME}"
  [[ -x "${ENV_PY}" ]] || die "Environment Python not found: ${ENV_PY}"
}

runtime_env() {
  export PYTHONPATH="${TAMING_DIR}:${THREESTUDIO_DIR}/extern:${THREESTUDIO_DIR}/extern/zero123:${THREESTUDIO_DIR}:${PYTHONPATH:-}"
  export PYTHONUNBUFFERED=1
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export WANDB_MODE="${WANDB_MODE:-offline}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
}

require_repo() {
  [[ -f "${THREESTUDIO_DIR}/launch.py" ]] || die "Missing threestudio launch.py: ${THREESTUDIO_DIR}"
  [[ -f "${THREESTUDIO_DIR}/configs/stable-zero123.yaml" ]] || die "Missing stable-zero123.yaml"
}

latest_trial() {
  local pattern="$1"
  local root="${THREESTUDIO_DIR}/outputs/zero123-sai-pytorch-encoding"
  [[ -d "${root}" ]] || return 0
  find "${root}" -maxdepth 1 -type d -name "${pattern}" 2>/dev/null | sort | tail -1
}

start_gpu_monitor() {
  local gpu_log="$1"
  mkdir -p "$(dirname "${gpu_log}")"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu \
      --format=csv -l 10 > "${gpu_log}" &
    GPU_MON_PID=$!
    echo "GPU monitor PID: ${GPU_MON_PID}"
    echo "GPU log: ${gpu_log}"
  else
    GPU_MON_PID=""
    echo "nvidia-smi not found; skip GPU monitor."
  fi
}

stop_gpu_monitor() {
  if [[ -n "${GPU_MON_PID:-}" ]] && kill -0 "${GPU_MON_PID}" >/dev/null 2>&1; then
    kill "${GPU_MON_PID}" || true
  fi
}

run_python_block() {
  local log_file="$1"
  shift
  mkdir -p "$(dirname "${log_file}")"
  echo "Log file: ${log_file}"
  set +e
  {
    if [[ -x /usr/bin/time ]]; then
      /usr/bin/time -v "${ENV_PY}" "$@"
    else
      echo "WARNING: /usr/bin/time is not available."
      "${ENV_PY}" "$@"
    fi
  } 2>&1 | tee "${log_file}"
  local status=${PIPESTATUS[0]}
  set -e
  return "${status}"
}

command_check_input() {
  activate_env
  [[ -f "${RGBA_INPUT}" ]] || die "Missing RGBA input: ${RGBA_INPUT}"
  "${ENV_PY}" - <<PY
from PIL import Image
from pathlib import Path

p = Path("${RGBA_INPUT}")
img = Image.open(p)
print("path:", p)
print("format:", img.format)
print("mode:", img.mode)
print("size:", img.size)
assert img.mode == "RGBA", "Object C input must be an RGBA PNG"
alpha = img.getchannel("A")
print("alpha bbox:", alpha.getbbox())
print("RGBA input check OK")
PY
}

command_prepare_input() {
  activate_env
  command_check_input
  mkdir -p "${THREESTUDIO_DIR}/load/images"
  cp -f "${RGBA_INPUT}" "${THREESTUDIO_INPUT}"
  echo "Copied RGBA input to: ${THREESTUDIO_INPUT}"
  "${ENV_PY}" - <<PY
from PIL import Image
p = "${THREESTUDIO_INPUT}"
img = Image.open(p)
print("format:", img.format)
print("mode:", img.mode)
print("size:", img.size)
assert img.mode == "RGBA"
print("threestudio input image OK")
PY
}

command_preflight() {
  activate_env
  runtime_env
  require_repo
  cd "${THREESTUDIO_DIR}"

  [[ -f "${THREESTUDIO_INPUT}" ]] || die "Run prepare-input first: ${THREESTUDIO_INPUT}"
  [[ -f "${ZERO123_DIR}/stable_zero123.ckpt" ]] || die "Missing Stable Zero123 checkpoint"
  [[ -f "${ZERO123_DIR}/sd-objaverse-finetune-c_concat-256.yaml" ]] || die "Missing Stable Zero123 yaml"

  "${ENV_PY}" - <<'PY'
from pathlib import Path
from PIL import Image
import importlib
import sys
import torch

img = Image.open("load/images/object_C_rgba.png")
print("image:", img.format, img.mode, img.size)
assert img.mode == "RGBA"

ckpt = Path("load/zero123/stable_zero123.ckpt")
yaml = Path("load/zero123/sd-objaverse-finetune-c_concat-256.yaml")
print("ckpt GiB:", round(ckpt.stat().st_size / 1024**3, 2))
print("yaml bytes:", yaml.stat().st_size)
assert ckpt.stat().st_size > int(7.9 * 1024**3)
assert yaml.stat().st_size > 1000

def prepare_ldm_alias():
    for name in ("ldm_zero123", "ldm"):
        try:
            module = importlib.import_module(name)
            if name == "ldm_zero123":
                sys.modules["ldm"] = module
            return
        except ModuleNotFoundError as exc:
            if exc.name != name:
                raise
    raise ModuleNotFoundError("Cannot import ldm_zero123 or ldm")

prepare_ldm_alias()
importlib.import_module("ldm.models.diffusion.ddpm")
importlib.import_module("taming")
importlib.import_module("open_clip")
importlib.import_module("pytorch_lightning")
importlib.import_module("nvdiffrast.torch")
importlib.import_module("envlight")

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
print("Stable Zero123 preflight OK")
PY
}

command_train_400() {
  activate_env
  runtime_env
  require_repo
  mkdir -p "${LOG_DIR}" "${GPU_LOG_DIR}"
  cd "${THREESTUDIO_DIR}"

  local train_log="${LOG_DIR}/C_stable_zero123_pytorch_400.log"
  local time_log="${LOG_DIR}/C_stable_zero123_pytorch_400_time.txt"
  local gpu_log="${GPU_LOG_DIR}/C_stable_zero123_pytorch_400_gpu.csv"

  echo "Stable Zero123 PyTorch-encoding 400-step training started at: $(date)" | tee "${time_log}"
  start_gpu_monitor "${gpu_log}"

  set +e
  {
    if [[ -x /usr/bin/time ]]; then
      /usr/bin/time -v "${ENV_PY}" - <<'PY'
import sys, runpy, importlib, torch

torch.set_float32_matmul_precision("high")
def prepare_ldm_alias():
    for name in ("ldm_zero123", "ldm"):
        try:
            module = importlib.import_module(name)
            if name == "ldm_zero123":
                sys.modules["ldm"] = module
            return
        except ModuleNotFoundError as exc:
            if exc.name != name:
                raise
    raise ModuleNotFoundError("Cannot import ldm_zero123 or ldm")

prepare_ldm_alias()

sys.argv = [
    "launch.py",
    "--config", "configs/stable-zero123.yaml",
    "--train",
    "--gpu", "0",
    "seed=42",
    "name=zero123-sai-pytorch-encoding",
    "tag=object_C_pytorch_400",
    "data.image_path=./load/images/object_C_rgba.png",
    "data.height=128",
    "data.width=128",
    "data.batch_size=1",
    "data.resolution_milestones=[]",
    "data.random_camera.height=[64]",
    "data.random_camera.width=[64]",
    "data.random_camera.batch_size=[1]",
    "data.random_camera.resolution_milestones=[]",
    "data.random_camera.eval_height=128",
    "data.random_camera.eval_width=128",
    "data.random_camera.eval_batch_size=1",
    "data.random_camera.n_val_views=8",
    "data.random_camera.n_test_views=16",
    "system.renderer.num_samples_per_ray=64",
    "system.geometry.pos_encoding_config.otype=ProgressiveBandFrequency",
    "system.geometry.pos_encoding_config.n_frequencies=6",
    "system.geometry.pos_encoding_config.n_masking_step=0",
    "system.geometry.pos_encoding_config.include_xyz=True",
    "system.geometry.mlp_network_config.otype=VanillaMLP",
    "system.geometry.mlp_network_config.n_neurons=64",
    "system.geometry.mlp_network_config.n_hidden_layers=3",
    "trainer.max_steps=400",
    "trainer.val_check_interval=100",
    "checkpoint.every_n_train_steps=100",
]
runpy.run_path("launch.py", run_name="__main__")
PY
    else
      "${ENV_PY}" - <<'PY'
import sys, runpy, importlib, torch

torch.set_float32_matmul_precision("high")
def prepare_ldm_alias():
    for name in ("ldm_zero123", "ldm"):
        try:
            module = importlib.import_module(name)
            if name == "ldm_zero123":
                sys.modules["ldm"] = module
            return
        except ModuleNotFoundError as exc:
            if exc.name != name:
                raise
    raise ModuleNotFoundError("Cannot import ldm_zero123 or ldm")

prepare_ldm_alias()
sys.argv = [
    "launch.py", "--config", "configs/stable-zero123.yaml",
    "--train", "--gpu", "0", "seed=42",
    "name=zero123-sai-pytorch-encoding",
    "tag=object_C_pytorch_400",
    "data.image_path=./load/images/object_C_rgba.png",
    "data.height=128", "data.width=128", "data.batch_size=1",
    "data.resolution_milestones=[]",
    "data.random_camera.height=[64]",
    "data.random_camera.width=[64]",
    "data.random_camera.batch_size=[1]",
    "data.random_camera.resolution_milestones=[]",
    "data.random_camera.eval_height=128",
    "data.random_camera.eval_width=128",
    "data.random_camera.eval_batch_size=1",
    "data.random_camera.n_val_views=8",
    "data.random_camera.n_test_views=16",
    "system.renderer.num_samples_per_ray=64",
    "system.geometry.pos_encoding_config.otype=ProgressiveBandFrequency",
    "system.geometry.pos_encoding_config.n_frequencies=6",
    "system.geometry.pos_encoding_config.n_masking_step=0",
    "system.geometry.pos_encoding_config.include_xyz=True",
    "system.geometry.mlp_network_config.otype=VanillaMLP",
    "system.geometry.mlp_network_config.n_neurons=64",
    "system.geometry.mlp_network_config.n_hidden_layers=3",
    "trainer.max_steps=400",
    "trainer.val_check_interval=100",
    "checkpoint.every_n_train_steps=100",
]
runpy.run_path("launch.py", run_name="__main__")
PY
    fi
  } 2>&1 | tee "${train_log}"
  local status=${PIPESTATUS[0]}
  set -e

  stop_gpu_monitor
  echo "Stable Zero123 PyTorch-encoding 400-step training finished at: $(date)" | tee -a "${time_log}"
  echo "RUN_STATUS=${status}" | tee -a "${time_log}"
  return "${status}"
}

command_continue_1200() {
  activate_env
  runtime_env
  require_repo
  mkdir -p "${LOG_DIR}" "${GPU_LOG_DIR}"
  cd "${THREESTUDIO_DIR}"

  local base_trial
  base_trial="$(latest_trial "object_C_pytorch_400@*")"
  [[ -n "${base_trial}" ]] || die "No object_C_pytorch_400 trial found."
  [[ -f "${base_trial}/ckpts/last.ckpt" ]] || die "Missing base checkpoint: ${base_trial}/ckpts/last.ckpt"

  export BASE_TRIAL_DIR="${base_trial}"
  local train_log="${LOG_DIR}/C_candidate_1200_from400_train.log"
  local gpu_log="${GPU_LOG_DIR}/C_stable_zero123_pytorch_1200_gpu.csv"

  echo "BASE_TRIAL_DIR=${BASE_TRIAL_DIR}"
  start_gpu_monitor "${gpu_log}"

  set +e
  {
    if [[ -x /usr/bin/time ]]; then
      /usr/bin/time -v "${ENV_PY}" - <<'PY'
import os, sys, runpy, importlib, torch

base_trial_dir = os.environ["BASE_TRIAL_DIR"]
_original_torch_load = torch.load
def torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = torch_load_compat

def prepare_ldm_alias():
    for name in ("ldm_zero123", "ldm"):
        try:
            module = importlib.import_module(name)
            if name == "ldm_zero123":
                sys.modules["ldm"] = module
            return
        except ModuleNotFoundError as exc:
            if exc.name != name:
                raise
    raise ModuleNotFoundError("Cannot import ldm_zero123 or ldm")

prepare_ldm_alias()

sys.argv = [
    "launch.py",
    "--config", f"{base_trial_dir}/configs/parsed.yaml",
    "--train",
    "--gpu", "0",
    f"resume={base_trial_dir}/ckpts/last.ckpt",
    "tag=object_C_pytorch_1200_from400_sameconfig",
    "trainer.max_steps=1200",
]
runpy.run_path("launch.py", run_name="__main__")
PY
    else
      "${ENV_PY}" - <<'PY'
import os, sys, runpy, importlib, torch
base_trial_dir = os.environ["BASE_TRIAL_DIR"]
_original_torch_load = torch.load
def torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = torch_load_compat
def prepare_ldm_alias():
    for name in ("ldm_zero123", "ldm"):
        try:
            module = importlib.import_module(name)
            if name == "ldm_zero123":
                sys.modules["ldm"] = module
            return
        except ModuleNotFoundError as exc:
            if exc.name != name:
                raise
    raise ModuleNotFoundError("Cannot import ldm_zero123 or ldm")

prepare_ldm_alias()
sys.argv = [
    "launch.py", "--config", f"{base_trial_dir}/configs/parsed.yaml",
    "--train", "--gpu", "0", f"resume={base_trial_dir}/ckpts/last.ckpt",
    "tag=object_C_pytorch_1200_from400_sameconfig", "trainer.max_steps=1200",
]
runpy.run_path("launch.py", run_name="__main__")
PY
    fi
  } 2>&1 | tee "${train_log}"
  local status=${PIPESTATUS[0]}
  set -e
  stop_gpu_monitor

  echo "TRAIN_STATUS=${status}"
  latest_trial "object_C_pytorch_1200_from400_sameconfig@*" || true
  return "${status}"
}

export_mesh() {
  local trial_dir="$1"
  local threshold_value="$2"
  local log_file="$3"
  export TRIAL_DIR="${trial_dir}"
  export THRESHOLD_VALUE="${threshold_value}"

  set +e
  "${ENV_PY}" - <<'PY' 2>&1 | tee "${log_file}"
import os, sys, runpy, importlib, torch

trial_dir = os.environ["TRIAL_DIR"]
threshold_value = os.environ["THRESHOLD_VALUE"]

_original_torch_load = torch.load
def torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = torch_load_compat

def prepare_ldm_alias():
    for name in ("ldm_zero123", "ldm"):
        try:
            module = importlib.import_module(name)
            if name == "ldm_zero123":
                sys.modules["ldm"] = module
            return
        except ModuleNotFoundError as exc:
            if exc.name != name:
                raise
    raise ModuleNotFoundError("Cannot import ldm_zero123 or ldm")

prepare_ldm_alias()

sys.argv = [
    "launch.py",
    "--config", f"{trial_dir}/configs/parsed.yaml",
    "--export",
    "--gpu", "0",
    f"resume={trial_dir}/ckpts/last.ckpt",
    "system.exporter_type=mesh-exporter",
    f"system.geometry.isosurface_threshold={threshold_value}",
    "system.geometry.isosurface_resolution=128",
]
runpy.run_path("launch.py", run_name="__main__")
PY
  local status=${PIPESTATUS[0]}
  set -e
  return "${status}"
}

command_export_400_auto() {
  activate_env
  runtime_env
  require_repo
  mkdir -p "${BASELINE_DIR}" "${LOG_DIR}"
  cd "${THREESTUDIO_DIR}"

  local trial_dir
  trial_dir="$(latest_trial "object_C_pytorch_400@*")"
  [[ -n "${trial_dir}" ]] || die "No object_C_pytorch_400 trial found."

  local log_file="${LOG_DIR}/C_baseline_400step_export_auto_threshold.log"
  set +e
  export_mesh "${trial_dir}" "auto" "${log_file}"
  local status=$?
  set -e

  local export_dir
  export_dir="$(find "${trial_dir}/save" -maxdepth 2 -type d -name "*export*" 2>/dev/null | sort | tail -1)"
  if [[ "${status}" -eq 0 && -n "${export_dir}" && -f "${export_dir}/model.obj" ]]; then
    cp -f "${log_file}" "${BASELINE_DIR}/C_baseline_400step_export_auto_threshold.log"
    cp -f "${trial_dir}/ckpts/last.ckpt" "${BASELINE_DIR}/last.ckpt"
    cp -f "${trial_dir}/configs/parsed.yaml" "${BASELINE_DIR}/parsed.yaml"
    cp -f "${THREESTUDIO_INPUT}" "${BASELINE_DIR}/object_C_rgba.png"
    cp -f "${trial_dir}/save/all_training_images.png" "${BASELINE_DIR}/" 2>/dev/null || true
    cp -f "${trial_dir}/save/all_training_images-checkpoint.png" "${BASELINE_DIR}/" 2>/dev/null || true
    cp -f "${trial_dir}/save/it400-test.mp4" "${BASELINE_DIR}/" 2>/dev/null || true
    cp -f "${trial_dir}/save/it400-val.mp4" "${BASELINE_DIR}/" 2>/dev/null || true
    cp -f "${export_dir}/model.obj" "${BASELINE_DIR}/model.obj"
    cp -f "${export_dir}/model.mtl" "${BASELINE_DIR}/model.mtl"
    cp -f "${export_dir}/texture_kd.jpg" "${BASELINE_DIR}/texture_kd.jpg" 2>/dev/null || true
    echo "400-step baseline export collected: ${BASELINE_DIR}"
  else
    die "400-step baseline export failed or model.obj is missing."
  fi
}

command_sweep_1200() {
  activate_env
  runtime_env
  require_repo
  mkdir -p "${SWEEP_DIR}" "${LOG_DIR}"
  cd "${THREESTUDIO_DIR}"

  local trial_dir
  trial_dir="$(latest_trial "object_C_pytorch_1200_from400_sameconfig@*")"
  [[ -n "${trial_dir}" ]] || die "No object_C_pytorch_1200_from400_sameconfig trial found."
  [[ -f "${trial_dir}/ckpts/last.ckpt" ]] || die "Missing 1200 checkpoint."

  echo "TRIAL_DIR=${trial_dir}"
  echo "SWEEP_DIR=${SWEEP_DIR}"

  for th in 8 12 16 20 24 28; do
    local th_float="${th}.0"
    local outdir="${SWEEP_DIR}/threshold_${th}"
    local log_file="${outdir}/export.log"
    mkdir -p "${outdir}"

    echo "== Export threshold ${th_float} =="
    if export_mesh "${trial_dir}" "${th_float}" "${log_file}"; then
      local export_dir
      export_dir="$(find "${trial_dir}/save" -maxdepth 2 -type d -name "*export*" 2>/dev/null | sort | tail -1)"
      if [[ -n "${export_dir}" && -f "${export_dir}/model.obj" ]]; then
        cp -f "${export_dir}/model.obj" "${outdir}/model.obj"
        cp -f "${export_dir}/model.mtl" "${outdir}/model.mtl"
        cp -f "${export_dir}/texture_kd.jpg" "${outdir}/texture_kd.jpg" 2>/dev/null || true
        cp -f "${trial_dir}/save/it1200-test.mp4" "${outdir}/it1200-test.mp4" 2>/dev/null || true
        echo "THRESHOLD_${th}_OK"
      else
        echo "THRESHOLD_${th}_MISSING_MODEL"
      fi
    else
      echo "THRESHOLD_${th}_FAILED"
    fi
  done

  echo "== threshold sweep result =="
  find "${SWEEP_DIR}" -maxdepth 2 -type f | sort | xargs -r ls -lh
}

command_collect_threshold20() {
  local target="${SWEEP_DIR}/threshold_20"
  echo "Recommended Object C candidate: ${target}"
  [[ -f "${target}/model.obj" ]] || die "Missing model.obj in threshold_20"
  [[ -f "${target}/model.mtl" ]] || die "Missing model.mtl in threshold_20"
  [[ -f "${target}/texture_kd.jpg" ]] || echo "WARNING: texture_kd.jpg is missing."
  [[ -f "${target}/it1200-test.mp4" ]] || echo "WARNING: it1200-test.mp4 is missing."
  find "${target}" -maxdepth 1 -type f | sort | xargs -r ls -lh
}

command_timed_final_1200_threshold20() {
  activate_env
  runtime_env
  require_repo
  mkdir -p "${FINAL_DIR}" "${LOG_DIR}" "${GPU_LOG_DIR}"
  cd "${THREESTUDIO_DIR}"

  [[ -x /usr/bin/time ]] || die "/usr/bin/time is required for final runtime recording."

  local old_backup="${SWEEP_DIR}/threshold_20"
  local stamp="${HW3_OBJECT_C_TIMED_STAMP:-$(date +%Y%m%d_%H%M%S)}"
  local train_tag="${HW3_OBJECT_C_TIMED_TRAIN_TAG:-object_C_final_1200_timed_${stamp}}"
  local final_dir="${HW3_OBJECT_C_TIMED_FINAL_DIR:-${FINAL_DIR}/${TIMED_FINAL_PREFIX}_${stamp}}"
  local train_log="${LOG_DIR}/C_final_1200_train_${stamp}.log"
  local export_log="${LOG_DIR}/C_final_threshold20_export_${stamp}.log"
  local train_gpu_log="${GPU_LOG_DIR}/C_final_1200_train_gpu_${stamp}.csv"
  local export_gpu_log="${GPU_LOG_DIR}/C_final_threshold20_export_gpu_${stamp}.csv"

  case "${final_dir}" in
    "${FINAL_DIR}/${TIMED_FINAL_PREFIX}_"*) ;;
    *) die "Refuse unsafe timed final directory: ${final_dir}" ;;
  esac
  [[ "${final_dir}" != "${old_backup}" ]] || die "Refuse to touch old threshold_20 backup."

  echo "Protected old backup:"
  echo "  ${old_backup}"
  echo "Timed final directory:"
  echo "  ${final_dir}"

  if [[ ! -f "${THREESTUDIO_INPUT}" && -f "${RGBA_INPUT}" ]]; then
    mkdir -p "$(dirname "${THREESTUDIO_INPUT}")"
    cp -f "${RGBA_INPUT}" "${THREESTUDIO_INPUT}"
  fi

  "${ENV_PY}" - <<'PY'
from pathlib import Path
from PIL import Image
import importlib
import sys
import torch

def prepare_ldm_alias():
    for name in ("ldm_zero123", "ldm"):
        try:
            module = importlib.import_module(name)
            if name == "ldm_zero123":
                sys.modules["ldm"] = module
            return
        except ModuleNotFoundError as exc:
            if exc.name != name:
                raise
    raise ModuleNotFoundError("Cannot import ldm_zero123 or ldm")

img = Image.open("load/images/object_C_rgba.png")
print("image:", img.mode, img.size)
assert img.mode == "RGBA"
ckpt = Path("load/zero123/stable_zero123.ckpt")
assert ckpt.exists(), ckpt
assert ckpt.stat().st_size > 7_900_000_000
assert Path("load/tets/128_tets.npz").exists()
assert torch.cuda.is_available()
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
prepare_ldm_alias()
importlib.import_module("ldm.models.diffusion.ddpm")
print("timed final preflight OK")
PY

  local base_trial
  base_trial="$(latest_trial "object_C_pytorch_400@*")"
  [[ -n "${base_trial}" ]] || die "No object_C_pytorch_400 trial found."
  [[ -f "${base_trial}/ckpts/last.ckpt" ]] || die "Missing base checkpoint."
  [[ -f "${base_trial}/configs/parsed.yaml" ]] || die "Missing base parsed.yaml."

  export BASE_TRIAL_DIR="${base_trial}"
  export TRAIN_TAG="${train_tag}"

  echo "BASE_TRIAL_DIR=${BASE_TRIAL_DIR}"
  echo "TRAIN_TAG=${TRAIN_TAG}"
  start_gpu_monitor "${train_gpu_log}"
  set +e
  /usr/bin/time -v "${ENV_PY}" - <<'PY' 2>&1 | tee "${train_log}"
import os
import sys
import runpy
import importlib
import torch

def prepare_ldm_alias():
    for name in ("ldm_zero123", "ldm"):
        try:
            module = importlib.import_module(name)
            if name == "ldm_zero123":
                sys.modules["ldm"] = module
            return
        except ModuleNotFoundError as exc:
            if exc.name != name:
                raise
    raise ModuleNotFoundError("Cannot import ldm_zero123 or ldm")

base_trial_dir = os.environ["BASE_TRIAL_DIR"]
train_tag = os.environ["TRAIN_TAG"]

torch.set_float32_matmul_precision("high")
_original_torch_load = torch.load
def torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = torch_load_compat
prepare_ldm_alias()

sys.argv = [
    "launch.py",
    "--config", f"{base_trial_dir}/configs/parsed.yaml",
    "--train",
    "--gpu", "0",
    f"resume={base_trial_dir}/ckpts/last.ckpt",
    f"tag={train_tag}",
    "trainer.max_steps=1200",
]
runpy.run_path("launch.py", run_name="__main__")
PY
  local train_status=${PIPESTATUS[0]}
  set -e
  stop_gpu_monitor
  echo "TRAIN_STATUS=${train_status}"
  [[ "${train_status}" -eq 0 ]] || die "1200 continuation failed."

  local timed_trial
  timed_trial="$(latest_trial "${train_tag}@*")"
  [[ -n "${timed_trial}" ]] || die "Timed 1200 trial not found."
  [[ -f "${timed_trial}/ckpts/last.ckpt" ]] || die "Timed checkpoint missing."
  export TIMED_TRIAL_DIR="${timed_trial}"
  echo "TIMED_TRIAL_DIR=${TIMED_TRIAL_DIR}"

  local export_marker
  export_marker="$(mktemp /tmp/object_c_export_marker.XXXXXX)"
  touch "${export_marker}"

  start_gpu_monitor "${export_gpu_log}"
  set +e
  /usr/bin/time -v "${ENV_PY}" - <<'PY' 2>&1 | tee "${export_log}"
import os
import sys
import runpy
import importlib
import torch

def prepare_ldm_alias():
    for name in ("ldm_zero123", "ldm"):
        try:
            module = importlib.import_module(name)
            if name == "ldm_zero123":
                sys.modules["ldm"] = module
            return
        except ModuleNotFoundError as exc:
            if exc.name != name:
                raise
    raise ModuleNotFoundError("Cannot import ldm_zero123 or ldm")

trial_dir = os.environ["TIMED_TRIAL_DIR"]
_original_torch_load = torch.load
def torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = torch_load_compat
prepare_ldm_alias()

sys.argv = [
    "launch.py",
    "--config", f"{trial_dir}/configs/parsed.yaml",
    "--export",
    "--gpu", "0",
    f"resume={trial_dir}/ckpts/last.ckpt",
    "system.exporter_type=mesh-exporter",
    "system.geometry.isosurface_threshold=20.0",
    "system.geometry.isosurface_resolution=128",
]
runpy.run_path("launch.py", run_name="__main__")
PY
  local export_status=${PIPESTATUS[0]}
  set -e
  stop_gpu_monitor
  echo "EXPORT_STATUS=${export_status}"
  [[ "${export_status}" -eq 0 ]] || die "threshold_20 export failed."

  local model_obj
  model_obj="$(find "${timed_trial}/save" -type f -name model.obj -newer "${export_marker}" | sort | tail -1)"
  [[ -n "${model_obj}" ]] || die "No fresh model.obj found."
  local export_dir
  export_dir="$(dirname "${model_obj}")"

  mkdir -p "${final_dir}/logs" "${final_dir}/gpu"
  cp -f "${export_dir}/model.obj" "${final_dir}/model.obj"
  cp -f "${export_dir}/model.mtl" "${final_dir}/model.mtl"
  cp -f "${export_dir}/texture_kd.jpg" "${final_dir}/texture_kd.jpg" 2>/dev/null || true
  cp -f "${timed_trial}/save/it1200-test.mp4" "${final_dir}/it1200-test.mp4" 2>/dev/null || true
  cp -f "${timed_trial}/configs/parsed.yaml" "${final_dir}/parsed.yaml"
  cp -f "${THREESTUDIO_INPUT}" "${final_dir}/input_rgba.png"
  cp -f "${train_log}" "${final_dir}/logs/"
  cp -f "${export_log}" "${final_dir}/logs/"
  cp -f "${train_gpu_log}" "${final_dir}/gpu/"
  cp -f "${export_gpu_log}" "${final_dir}/gpu/"

  cat > "${final_dir}/notes.md" <<EOF
# Object C Timed Final Candidate

- Method: Stable Zero123 through threestudio.
- Input: one RGBA foreground image.
- Final candidate: 1200-step continuation exported with fixed threshold 20.0.
- Runtime definition: 1200 continuation + threshold_20 mesh export.
- The 400-step baseline is only the resume source and is not counted as final runtime.
- Old backup kept untouched: ${old_backup}
- Source trial: ${timed_trial}
- Training log: $(basename "${train_log}")
- Export log: $(basename "${export_log}")
- Training GPU log: $(basename "${train_gpu_log}")
- Export GPU log: $(basename "${export_gpu_log}")
EOF

  echo "== Runtime summary =="
  grep -nE "Elapsed \\(wall clock\\) time|Exit status|max_steps=1200|Training complete|Traceback|ValidationError|Error" "${train_log}" "${export_log}" || true
  echo "== Timed final Object C folder =="
  find "${final_dir}" -maxdepth 2 -type f | sort | xargs -r ls -lh
  echo "TIMED_OBJECT_C_FINAL_DIR=${final_dir}"
  echo "TIMED_OBJECT_C_DONE"
}

main() {
  local command="${1:-help}"
  case "${command}" in
    help|-h|--help) usage ;;
    check-input) command_check_input ;;
    prepare-input) command_prepare_input ;;
    preflight) command_preflight ;;
    train-400) command_train_400 ;;
    continue-1200) command_continue_1200 ;;
    export-400-auto) command_export_400_auto ;;
    sweep-1200) command_sweep_1200 ;;
    timed-final-1200-threshold20) command_timed_final_1200_threshold20 ;;
    collect-threshold20) command_collect_threshold20 ;;
    *) usage; die "Unknown command: ${command}" ;;
  esac
}

main "$@"
