#!/usr/bin/env bash
set -euo pipefail

# HW3 Task 1 / Object A reproducible pipeline.
#
# Purpose:
#   Reconstruct a real object from a phone video or multi-view photos using
#   COLMAP and the official 3D Gaussian Splatting implementation.
#
# Pipeline:
#   frames -> COLMAP camera poses -> 3DGS training -> preview rendering ->
#   cloud final asset folder.
#
# Safety:
#   - This script does not delete files.
#   - The default action is "help" so that opening or running the script
#     without arguments will not launch a long cloud job.
#   - Existing final asset folders are not overwritten automatically.

ASSET_ROOT="${HW3_ASSET_ROOT:-/root/autodl-tmp/hw3_task1_assets}"
ENV_NAME="${HW3_3DGS_ENV_NAME:-hw3_3dgs}"
CONDA_SH="${HW3_CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
ENV_PY="${HW3_3DGS_PYTHON:-/root/miniconda3/envs/${ENV_NAME}/bin/python}"

REPO_DIR="${HW3_GAUSSIAN_REPO:-${ASSET_ROOT}/third_party/gaussian-splatting}"
DATA_DIR="${HW3_OBJECT_A_DATA:-${ASSET_ROOT}/data/object_A_raw}"
VIDEO_PATH="${HW3_OBJECT_A_VIDEO:-${DATA_DIR}/video.mp4}"
OUTPUT_DIR="${HW3_OBJECT_A_OUTPUT:-${ASSET_ROOT}/outputs/A_3dgs}"
FINAL_DIR="${HW3_OBJECT_A_FINAL:-${ASSET_ROOT}/final_assets/object_A}"
LOG_DIR="${ASSET_ROOT}/logs"
GPU_LOG_DIR="${LOG_DIR}/gpu"

ITERATIONS="${HW3_OBJECT_A_ITERATIONS:-7000}"
FRAME_FPS="${HW3_OBJECT_A_FRAME_FPS:-2}"
FRAME_WIDTH="${HW3_OBJECT_A_FRAME_WIDTH:-1280}"

usage() {
  cat <<EOF
Usage:
  bash scripts/run_object_A_3dgs.sh <command>

Commands:
  check      Check paths, conda environment, PyTorch/CUDA, inputs, and outputs.
  frames     Extract frames from DATA_DIR/video.mp4 into DATA_DIR/input.
  convert    Run COLMAP conversion for Object A with CPU feature extraction.
  train      Train 3DGS for Object A, default ${ITERATIONS} iterations.
  render     Render preview images from the trained 3DGS model.
  handoff    Copy the trained model to final_assets/object_A without overwrite.
  all        Run check, frames, convert, train, render, and handoff in order.
  help       Show this message.

Environment overrides:
  HW3_ASSET_ROOT          Default: ${ASSET_ROOT}
  HW3_3DGS_ENV_NAME       Default: ${ENV_NAME}
  HW3_3DGS_PYTHON         Default: ${ENV_PY}
  HW3_GAUSSIAN_REPO       Default: ${REPO_DIR}
  HW3_OBJECT_A_DATA       Default: ${DATA_DIR}
  HW3_OBJECT_A_OUTPUT     Default: ${OUTPUT_DIR}
  HW3_OBJECT_A_FINAL      Default: ${FINAL_DIR}
  HW3_OBJECT_A_ITERATIONS Default: ${ITERATIONS}

Example:
  bash scripts/run_object_A_3dgs.sh check
  bash scripts/run_object_A_3dgs.sh frames
  bash scripts/run_object_A_3dgs.sh convert
  bash scripts/run_object_A_3dgs.sh train
  bash scripts/run_object_A_3dgs.sh render
  bash scripts/run_object_A_3dgs.sh handoff
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

need_cmd() {
  local cmd="$1"
  command -v "${cmd}" >/dev/null 2>&1 || die "Required command not found: ${cmd}"
}

activate_env() {
  [[ -f "${CONDA_SH}" ]] || die "Cannot find conda activation script: ${CONDA_SH}"
  # shellcheck source=/dev/null
  source "${CONDA_SH}"
  conda activate "${ENV_NAME}"
  [[ -x "${ENV_PY}" ]] || die "Environment Python not found or not executable: ${ENV_PY}"
}

run_with_log() {
  local log_file="$1"
  shift

  mkdir -p "$(dirname "${log_file}")"
  echo "== Command =="
  printf '%q ' "$@"
  echo
  echo "== Log file: ${log_file} =="

  if [[ -x /usr/bin/time ]]; then
    /usr/bin/time -v "$@" 2>&1 | tee "${log_file}"
  else
    echo "WARNING: /usr/bin/time is not available; runtime details will be incomplete." | tee "${log_file}"
    "$@" 2>&1 | tee -a "${log_file}"
  fi
}

print_config() {
  cat <<EOF
== Object A configuration ==
Asset root:    ${ASSET_ROOT}
Conda env:     ${ENV_NAME}
Env python:    ${ENV_PY}
3DGS repo:     ${REPO_DIR}
Data dir:      ${DATA_DIR}
Video path:    ${VIDEO_PATH}
Output dir:    ${OUTPUT_DIR}
Final dir:     ${FINAL_DIR}
Log dir:       ${LOG_DIR}
Iterations:    ${ITERATIONS}
Frame FPS:     ${FRAME_FPS}
Frame width:   ${FRAME_WIDTH}
EOF
}

check_repo() {
  [[ -d "${REPO_DIR}" ]] || die "3DGS repository does not exist: ${REPO_DIR}"
  [[ -f "${REPO_DIR}/convert.py" ]] || die "Missing convert.py in ${REPO_DIR}"
  [[ -f "${REPO_DIR}/train.py" ]] || die "Missing train.py in ${REPO_DIR}"
  [[ -f "${REPO_DIR}/render.py" ]] || die "Missing render.py in ${REPO_DIR}"
}

check_data_dir() {
  [[ -d "${DATA_DIR}" ]] || die "Object A data directory does not exist: ${DATA_DIR}"
  [[ -d "${DATA_DIR}/input" ]] || die "Input frame directory does not exist: ${DATA_DIR}/input"
}

check_extensions() {
  "${ENV_PY}" - <<'PY'
import diff_gaussian_rasterization
import simple_knn._C
print("3DGS extensions import OK")
PY
}

ensure_extensions() {
  echo "== Check 3DGS CUDA extensions =="
  if check_extensions; then
    return
  fi

  echo "3DGS extensions are missing in the selected Python environment."
  echo "Building them with --no-build-isolation..."
  cd "${REPO_DIR}"

  export TORCH_CUDA_ARCH_LIST="$("${ENV_PY}" - <<'PY'
import torch
major, minor = torch.cuda.get_device_capability(0)
print(f"{major}.{minor}")
PY
)"
  echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

  "${ENV_PY}" -m pip install --no-build-isolation ./submodules/diff-gaussian-rasterization
  "${ENV_PY}" -m pip install --no-build-isolation ./submodules/simple-knn
  check_extensions
}

command_check() {
  print_config
  activate_env
  check_repo

  echo
  echo "== Basic tools =="
  need_cmd colmap
  need_cmd ffmpeg
  need_cmd nvidia-smi
  command -v mogrify >/dev/null 2>&1 || echo "WARNING: mogrify not found. convert.py --resize may fail."
  [[ -x /usr/bin/time ]] || echo "WARNING: /usr/bin/time not found. Install package 'time' for detailed runtime logs."

  echo
  echo "== Python / CUDA =="
  echo "which python: $(which python)"
  echo "which pip:    $(which pip)"
  "${ENV_PY}" - <<'PY'
import sys
import torch
print("env python:", sys.executable)
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY

  echo
  echo "== Input data =="
  if [[ -d "${DATA_DIR}" ]]; then
    ls -lah "${DATA_DIR}"
  else
    echo "Data directory not found yet: ${DATA_DIR}"
  fi

  if [[ -d "${DATA_DIR}/input" ]]; then
    echo "Frame count:"
    find "${DATA_DIR}/input" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l
  fi

  echo
  echo "== Existing COLMAP / 3DGS outputs =="
  [[ -d "${DATA_DIR}/sparse/0" ]] && ls -lah "${DATA_DIR}/sparse/0" || echo "No sparse/0 directory yet."
  [[ -d "${OUTPUT_DIR}" ]] && ls -lah "${OUTPUT_DIR}" || echo "No 3DGS output directory yet."
}

command_frames() {
  activate_env
  need_cmd ffmpeg
  [[ -f "${VIDEO_PATH}" ]] || die "Video file not found: ${VIDEO_PATH}"

  mkdir -p "${DATA_DIR}/input"
  if find "${DATA_DIR}/input" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | grep -q .; then
    echo "Existing frames found in ${DATA_DIR}/input; skip extraction to avoid overwriting."
    echo "Use a fresh data directory if you need to re-extract frames cleanly."
    return
  fi

  echo "== Extract frames =="
  echo "Video: ${VIDEO_PATH}"
  echo "Output: ${DATA_DIR}/input/frame_%04d.jpg"
  ffmpeg -i "${VIDEO_PATH}" -vf "fps=${FRAME_FPS},scale=${FRAME_WIDTH}:-1" \
    "${DATA_DIR}/input/frame_%04d.jpg"

  echo "Frame count:"
  find "${DATA_DIR}/input" -maxdepth 1 -type f -iname "*.jpg" | wc -l
}

command_convert() {
  activate_env
  check_repo
  check_data_dir
  need_cmd colmap
  command -v mogrify >/dev/null 2>&1 || echo "WARNING: mogrify not found. convert.py --resize may fail."

  mkdir -p "${LOG_DIR}"
  cd "${REPO_DIR}"

  local frame_count
  frame_count="$(find "${DATA_DIR}/input" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l)"
  [[ "${frame_count}" -gt 0 ]] || die "No frames found in ${DATA_DIR}/input"

  echo "== COLMAP conversion =="
  echo "Input frame count: ${frame_count}"
  echo "This step uses --no_gpu for COLMAP feature extraction/matching to work reliably in headless cloud terminals."

  local log_file="${LOG_DIR}/A_convert_cpu_$(date +%Y%m%d_%H%M%S).log"
  run_with_log "${log_file}" "${ENV_PY}" convert.py \
    -s "${DATA_DIR}" \
    --resize \
    --no_gpu

  echo
  echo "== Conversion checks =="
  ls -lah "${DATA_DIR}/sparse/0"
  for dir_name in images images_2 images_4 images_8; do
    [[ -d "${DATA_DIR}/${dir_name}" ]] || die "Missing directory after convert: ${DATA_DIR}/${dir_name}"
    printf "%-10s " "${dir_name}"
    find "${DATA_DIR}/${dir_name}" -maxdepth 1 -type f | wc -l
  done
  echo "COLMAP conversion finished. Log: ${log_file}"
}

start_gpu_monitor() {
  local gpu_log="$1"
  mkdir -p "$(dirname "${gpu_log}")"
  nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu \
    --format=csv -l 10 > "${gpu_log}" &
  GPU_MON_PID=$!
  echo "GPU monitor PID: ${GPU_MON_PID}"
  echo "GPU log: ${gpu_log}"
}

stop_gpu_monitor() {
  if [[ -n "${GPU_MON_PID:-}" ]] && kill -0 "${GPU_MON_PID}" >/dev/null 2>&1; then
    kill "${GPU_MON_PID}" || true
  fi
}

command_train() {
  activate_env
  check_repo
  check_data_dir
  ensure_extensions

  [[ -f "${DATA_DIR}/sparse/0/cameras.bin" ]] || die "Missing COLMAP file: sparse/0/cameras.bin"
  [[ -f "${DATA_DIR}/sparse/0/images.bin" ]] || die "Missing COLMAP file: sparse/0/images.bin"
  [[ -f "${DATA_DIR}/sparse/0/points3D.bin" ]] || die "Missing COLMAP file: sparse/0/points3D.bin"

  mkdir -p "${OUTPUT_DIR}" "${GPU_LOG_DIR}"
  cd "${REPO_DIR}"

  local train_log="${LOG_DIR}/A_train_${ITERATIONS}.log"
  local gpu_log="${GPU_LOG_DIR}/A_train_gpu.csv"

  echo "== 3DGS training =="
  echo "Iterations: ${ITERATIONS}"
  start_gpu_monitor "${gpu_log}"

  set +e
  run_with_log "${train_log}" "${ENV_PY}" train.py \
    -s "${DATA_DIR}" \
    -m "${OUTPUT_DIR}" \
    --iterations "${ITERATIONS}" \
    --test_iterations -1 \
    --data_device cpu
  local train_status=$?
  set -e

  stop_gpu_monitor
  [[ "${train_status}" -eq 0 ]] || exit "${train_status}"

  echo
  echo "== Training checks =="
  ls -lah "${OUTPUT_DIR}"
  find "${OUTPUT_DIR}" -maxdepth 4 -type f | head -50
  echo "Training finished. Log: ${train_log}"
  echo "GPU log: ${gpu_log}"
}

command_render() {
  activate_env
  check_repo

  [[ -d "${OUTPUT_DIR}" ]] || die "3DGS output directory does not exist: ${OUTPUT_DIR}"
  cd "${REPO_DIR}"

  local render_log="${LOG_DIR}/A_render.log"

  echo "== Render preview images =="
  run_with_log "${render_log}" "${ENV_PY}" render.py -m "${OUTPUT_DIR}"

  echo
  echo "== Render checks =="
  find "${OUTPUT_DIR}" -maxdepth 6 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.ply" \) | head -80
  echo "Render finished. Log: ${render_log}"
}

command_handoff() {
  [[ -d "${OUTPUT_DIR}" ]] || die "3DGS output directory does not exist: ${OUTPUT_DIR}"

  mkdir -p "${FINAL_DIR}"

  echo "== Prepare final Object A folder =="
  if [[ ! -e "${FINAL_DIR}/A_3dgs_model" ]]; then
    cp -r "${OUTPUT_DIR}" "${FINAL_DIR}/A_3dgs_model"
    echo "Copied model directory to: ${FINAL_DIR}/A_3dgs_model"
  else
    echo "A_3dgs_model already exists; skip copying to avoid overwriting."
  fi

  if [[ ! -e "${FINAL_DIR}/notes.md" ]]; then
    cat > "${FINAL_DIR}/notes.md" <<EOF
# Object A notes

Method: Real multi-view reconstruction + COLMAP + 3DGS

Input: phone video or multi-view images in data/object_A_raw

Output:
- 3DGS model directory

Recommended file for downstream use:
- A_3dgs_model/point_cloud/iteration_${ITERATIONS}/point_cloud.ply
- Or the whole A_3dgs_model directory

Runtime and hardware:
- GPU: NVIDIA GeForce RTX 5090
- Training iterations: ${ITERATIONS}
- Training log: logs/A_train_${ITERATIONS}.log
- GPU log: logs/gpu/A_train_gpu.csv

Known limitations:
- The reconstructed 3DGS may include background visible in the input video.
- This is a 3DGS representation, not a traditional mesh.
EOF
    echo "Created: ${FINAL_DIR}/notes.md"
  else
    echo "notes.md already exists; skip writing to avoid overwriting."
  fi

  echo
  echo "== Final folder check =="
  find "${FINAL_DIR}" -maxdepth 4 -type f | head -80
}

command_all() {
  command_check
  command_frames
  command_convert
  command_train
  command_render
  command_handoff
}

main() {
  local command="${1:-help}"

  case "${command}" in
    help|-h|--help)
      usage
      ;;
    check)
      command_check
      ;;
    frames)
      command_frames
      ;;
    convert)
      command_convert
      ;;
    train)
      command_train
      ;;
    render)
      command_render
      ;;
    handoff)
      command_handoff
      ;;
    all)
      command_all
      ;;
    *)
      usage
      die "Unknown command: ${command}"
      ;;
  esac
}

main "$@"
