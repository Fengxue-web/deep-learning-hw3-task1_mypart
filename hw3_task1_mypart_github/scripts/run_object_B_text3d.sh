#!/usr/bin/env bash
set -euo pipefail

# HW3 Task 1 / Object B reproducible text-to-3D workflow.
#
# It uses threestudio LatentNeRF with a pretrained
# Stable Diffusion v1-5 model and SDS-style optimization. The only semantic
# input is a text prompt; no Object A/C images or videos are used.
#
# The script is command-based on purpose. Running it without a command prints
# help and does not start a long GPU job. It also avoids deleting previous
# outputs. Final cloud asset collection uses non-destructive copies by default.

ASSET_ROOT="${HW3_ASSET_ROOT:-/root/autodl-tmp/hw3_task1_assets}"
ENV_NAME="${THREESTUDIO_ENV_NAME:-hw3_threestudio}"
CONDA_SH="${HW3_CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
ENV_PY="${HW3_THREESTUDIO_PYTHON:-/root/miniconda3/envs/${ENV_NAME}/bin/python}"
THREESTUDIO_DIR="${HW3_THREESTUDIO_DIR:-${ASSET_ROOT}/third_party/threestudio}"

HF_HOME="${HF_HOME:-${ASSET_ROOT}/third_party/huggingface_cache}"
SD15_SNAPSHOT_ROOT="${HF_HOME}/hub/models--runwayml--stable-diffusion-v1-5/snapshots"
MODEL_PATH="${HW3_OBJECT_B_SD15_PATH:-${SD15_SNAPSHOT_ROOT}/451f4fe16113bff5a5d2269ed5ad43b0592e9a14}"

PROMPT="${HW3_OBJECT_B_PROMPT:-a small blue ceramic coffee mug with a large handle, full object, centered}"
NAME="${HW3_OBJECT_B_NAME:-object_B_latentnerf}"
TAG="${HW3_OBJECT_B_TAG:-object_B_mug_20260613_latentnerf_repro_p2_blue_full_handle_1000}"
SEED="${HW3_OBJECT_B_SEED:-11}"
STEPS="${HW3_OBJECT_B_STEPS:-1000}"

LOG_DIR="${ASSET_ROOT}/logs"
GPU_LOG_DIR="${LOG_DIR}/gpu"
FINAL_ROOT="${ASSET_ROOT}/final_assets/object_B"
FINAL_DIR="${HW3_OBJECT_B_FINAL_DIR:-${FINAL_ROOT}/object_B_latentnerf_p2_blue_20260613_final}"

ADOPTED_TRIAL_REL="outputs/object_B_latentnerf/object_B_mug_20260613_latentnerf_sweep_p2_blue_full_handle_1000@20260613-182321"
ADOPTED_TRIAL="${HW3_OBJECT_B_ADOPTED_TRIAL:-${THREESTUDIO_DIR}/${ADOPTED_TRIAL_REL}}"
ADOPTED_EXPORT_DIR="${HW3_OBJECT_B_EXPORT_DIR:-${ADOPTED_TRIAL}/save/it1000-export}"
ADOPTED_TRAIN_LOG="${HW3_OBJECT_B_TRAIN_LOG:-${LOG_DIR}/B_latentnerf_sweep_p2_blue_full_handle_20260613_182040.log}"
ADOPTED_TRAIN_GPU_CSV="${HW3_OBJECT_B_TRAIN_GPU_CSV:-${GPU_LOG_DIR}/B_latentnerf_sweep_p2_blue_full_handle_gpu_20260613_182040.csv}"
ADOPTED_EXPORT_LOG="${HW3_OBJECT_B_EXPORT_LOG:-${LOG_DIR}/B_latentnerf_p2_export_retry_20260613_194005.log}"
ADOPTED_EXPORT_GPU_CSV="${HW3_OBJECT_B_EXPORT_GPU_CSV:-${GPU_LOG_DIR}/B_latentnerf_p2_export_retry_gpu_20260613_194005.csv}"

usage() {
  cat <<EOF
Usage:
  bash scripts/run_object_B_text3d.sh <command>

Commands:
  help              Show this message.
  preflight         Verify GPU, threestudio, LatentNeRF config, and local SD v1-5 files.
  train             Run a reproducible Object B LatentNeRF training job.
  export-latest     Export OBJ/MTL/texture from the latest matching training run.
  collect-adopted   Collect the adopted successful Object B result into final_assets/object_B.
  sanity-final      Check the final Object B handoff folder and texture references.
  all               Run preflight, train, and export-latest for a new candidate trial.

Default Object B prompt:
  ${PROMPT}

Default adopted final folder:
  ${FINAL_DIR}

Important overrides:
  HW3_ASSET_ROOT
  HW3_THREESTUDIO_DIR
  HW3_OBJECT_B_SD15_PATH
  HW3_OBJECT_B_PROMPT
  HW3_OBJECT_B_TAG
  HW3_OBJECT_B_STEPS
  HW3_OBJECT_B_ADOPTED_TRIAL
  HW3_OBJECT_B_FINAL_DIR
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
  conda activate "${ENV_NAME}"
  [[ -x "${ENV_PY}" ]] || die "Environment Python not found: ${ENV_PY}"
}

configure_offline_model_env() {
  export HF_HOME="${HF_HOME}"
  export TRANSFORMERS_CACHE="${HF_HOME}/hub"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export DIFFUSERS_OFFLINE=1
  export WANDB_MODE=offline
  export PYTHONUNBUFFERED=1
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
}

require_threestudio_repo() {
  [[ -f "${THREESTUDIO_DIR}/launch.py" ]] || die "Missing threestudio launch.py: ${THREESTUDIO_DIR}"
  [[ -f "${THREESTUDIO_DIR}/configs/latentnerf.yaml" ]] || die "Missing configs/latentnerf.yaml in ${THREESTUDIO_DIR}"
}

require_sd15_snapshot() {
  local required=(
    "model_index.json"
    "scheduler/scheduler_config.json"
    "tokenizer/tokenizer_config.json"
    "text_encoder/model.safetensors"
    "unet/diffusion_pytorch_model.safetensors"
    "vae/diffusion_pytorch_model.safetensors"
  )

  [[ -d "${MODEL_PATH}" ]] || die "Stable Diffusion v1-5 local snapshot not found: ${MODEL_PATH}"
  for item in "${required[@]}"; do
    [[ -f "${MODEL_PATH}/${item}" ]] || die "Missing SD v1-5 file: ${MODEL_PATH}/${item}"
  done
}

command_preflight() {
  echo "== Object B preflight =="
  echo "Asset root: ${ASSET_ROOT}"
  echo "threestudio: ${THREESTUDIO_DIR}"
  echo "Conda env: ${ENV_NAME}"
  echo "Model path: ${MODEL_PATH}"
  echo "Prompt: ${PROMPT}"

  load_conda
  configure_offline_model_env
  require_threestudio_repo
  require_sd15_snapshot

  nvidia-smi

  export MODEL_PATH PROMPT
  "${ENV_PY}" - <<'PY'
import importlib
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

model_path = Path(os.environ["MODEL_PATH"])
prompt = os.environ["PROMPT"]

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

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))

if len(token_ids) > 77:
    failed.append("prompt_too_long")
if not torch.cuda.is_available():
    failed.append("cuda")
if failed:
    raise SystemExit("FAILED_PREFLIGHT=" + ",".join(failed))
PY

  cd "${THREESTUDIO_DIR}"
  "${ENV_PY}" launch.py --help >/dev/null
  echo "OBJECT_B_PREFLIGHT_OK"
}

start_gpu_monitor() {
  local gpu_csv="$1"
  mkdir -p "$(dirname "${gpu_csv}")"
  nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu \
    --format=csv -l 5 > "${gpu_csv}" &
  GPU_MON_PID=$!
}

stop_gpu_monitor() {
  if [[ -n "${GPU_MON_PID:-}" ]]; then
    kill "${GPU_MON_PID}" 2>/dev/null || true
    wait "${GPU_MON_PID}" 2>/dev/null || true
    GPU_MON_PID=""
  fi
}

trap stop_gpu_monitor EXIT

run_with_gpu_log() {
  local label="$1"
  local log_path="$2"
  local gpu_csv="$3"
  shift 3

  mkdir -p "$(dirname "${log_path}")" "$(dirname "${gpu_csv}")"
  echo "== ${label} =="
  echo "Log: ${log_path}"
  echo "GPU CSV: ${gpu_csv}"
  echo "Command: $*"

  start_gpu_monitor "${gpu_csv}"
  set +e
  /usr/bin/time -v "$@" 2>&1 | tee "${log_path}"
  local status=${PIPESTATUS[0]}
  set -e
  stop_gpu_monitor

  echo "== ${label} status: ${status} =="
  return "${status}"
}

command_train() {
  load_conda
  configure_offline_model_env
  require_threestudio_repo
  require_sd15_snapshot

  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  local train_log="${LOG_DIR}/B_latentnerf_repro_p2_blue_full_handle_${stamp}.log"
  local train_gpu_csv="${GPU_LOG_DIR}/B_latentnerf_repro_p2_blue_full_handle_gpu_${stamp}.csv"

  cd "${THREESTUDIO_DIR}"
  local cmd=(
    "${ENV_PY}" launch.py
    --config configs/latentnerf.yaml
    --train
    --gpu 0
    "seed=${SEED}"
    "name=${NAME}"
    "tag=${TAG}"
    "trainer.max_steps=${STEPS}"
    "trainer.precision=32"
    "data.batch_size=1"
    "system.prompt_processor.pretrained_model_name_or_path=${MODEL_PATH}"
    "system.guidance.pretrained_model_name_or_path=${MODEL_PATH}"
    "system.guidance.half_precision_weights=false"
    "system.prompt_processor.prompt=${PROMPT}"
    "system.geometry.pos_encoding_config.otype=ProgressiveBandFrequency"
    "system.geometry.pos_encoding_config.n_frequencies=6"
    "system.geometry.mlp_network_config.otype=VanillaMLP"
    "system.geometry.mlp_network_config.n_neurons=64"
    "system.geometry.mlp_network_config.n_hidden_layers=3"
  )

  if run_with_gpu_log "Object B LatentNeRF training" "${train_log}" "${train_gpu_csv}" "${cmd[@]}"; then
    local trial
    trial="$(latest_trial || true)"
    echo "TRAIN_STATUS=0"
    echo "TRAIN_LOG=${train_log}"
    echo "TRAIN_GPU_CSV=${train_gpu_csv}"
    echo "LATEST_TRIAL=${trial}"
  else
    local status=$?
    echo "TRAIN_STATUS=${status}"
    echo "TRAIN_LOG=${train_log}"
    echo "TRAIN_GPU_CSV=${train_gpu_csv}"
    die "Object B training failed. Do not export this failed trial."
  fi
}

latest_trial() {
  find "${THREESTUDIO_DIR}/outputs/${NAME}" -maxdepth 1 -type d -name "${TAG}@*" 2>/dev/null | sort | tail -1
}

resolve_export_trial() {
  if [[ -n "${HW3_OBJECT_B_TRIAL:-}" ]]; then
    echo "${HW3_OBJECT_B_TRIAL}"
    return
  fi

  local trial
  trial="$(latest_trial || true)"
  [[ -n "${trial}" ]] || die "No matching Object B trial found. Set HW3_OBJECT_B_TRIAL explicitly."
  echo "${trial}"
}

command_export_latest() {
  load_conda
  configure_offline_model_env
  require_threestudio_repo

  local trial
  trial="$(resolve_export_trial)"
  [[ -f "${trial}/configs/parsed.yaml" ]] || die "Missing parsed config: ${trial}/configs/parsed.yaml"
  [[ -f "${trial}/ckpts/last.ckpt" ]] || die "Missing checkpoint: ${trial}/ckpts/last.ckpt"

  export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  local export_log="${LOG_DIR}/B_latentnerf_repro_export_${stamp}.log"
  local export_gpu_csv="${GPU_LOG_DIR}/B_latentnerf_repro_export_gpu_${stamp}.csv"

  cd "${THREESTUDIO_DIR}"
  local cmd=(
    "${ENV_PY}" launch.py
    --config "${trial}/configs/parsed.yaml"
    --export
    --gpu 0
    "resume=${trial}/ckpts/last.ckpt"
  )

  if run_with_gpu_log "Object B OBJ export" "${export_log}" "${export_gpu_csv}" "${cmd[@]}"; then
    echo "EXPORT_STATUS=0"
    echo "TRIAL_DIR=${trial}"
    echo "EXPORT_LOG=${export_log}"
    echo "EXPORT_GPU_CSV=${export_gpu_csv}"
    echo "Recent exported files:"
    find "${trial}/save" -maxdepth 3 \( -name "model.obj" -o -name "model.mtl" -o -name "texture_kd.jpg" \) -print 2>/dev/null | sort
  else
    local status=$?
    echo "EXPORT_STATUS=${status}"
    echo "EXPORT_LOG=${export_log}"
    echo "EXPORT_GPU_CSV=${export_gpu_csv}"
    die "Object B export failed."
  fi
}

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -f "${src}" ]]; then
    if [[ -f "${dst}" ]]; then
      echo "Exists, skip: ${dst}"
    else
      cp -a "${src}" "${dst}"
      echo "Copied: ${dst}"
    fi
  else
    echo "Optional file missing, skip: ${src}"
  fi
}

copy_required() {
  local src="$1"
  local dst="$2"
  [[ -f "${src}" ]] || die "Required file missing: ${src}"
  if [[ -f "${dst}" ]]; then
    echo "Exists, skip: ${dst}"
  else
    cp -a "${src}" "${dst}"
    echo "Copied: ${dst}"
  fi
}

write_notes_if_missing() {
  local notes_path="${FINAL_DIR}/notes.md"
  if [[ -f "${notes_path}" ]]; then
    echo "Exists, skip: ${notes_path}"
    return
  fi

  cat > "${notes_path}" <<EOF
# Object B Notes

## Method

- Object B was generated with threestudio LatentNeRF.
- The guidance model is a local pretrained Stable Diffusion v1-5 snapshot.
- The optimization follows the text-to-3D/SDS-style route required by HW3 Task 1.
- No real Object A/C image, video, or 3D geometry is used as input.

## Selected Prompt

${PROMPT}

## Selected Run

- Selected trial: ${ADOPTED_TRIAL_REL}
- Exported asset directory: ${ADOPTED_EXPORT_DIR#${THREESTUDIO_DIR}/}
- Final handoff directory: ${FINAL_DIR}
- Recommended asset: asset/model.obj
- Texture files: asset/model.mtl and asset/texture_kd.jpg

## Known Visual Issues

- The result is recognizable as a blue mug-like object, but the geometry is coarse.
- The handle and side protrusions are imperfect and may need manual adjustment in scene fusion.
- Texture is generated and blurry, not faithful to a real photographed mug.

## Runtime And Hardware

- Training runtime observed in the adopted run: about 2:45.00 on RTX 5090.
- Training peak GPU memory observed: about 7739 MiB.
- Export runtime observed: about 1:34.06.
- Export peak GPU memory observed: about 5199 MiB.
- Adopted total train-plus-export runtime: about 4:19.06.

## Handoff Recommendation

Use asset/model.obj together with asset/model.mtl and asset/texture_kd.jpg. If OBJ import is unstable in the downstream scene-fusion pipeline, use the preview video, preview image, and checkpoint files as backup evidence for Object B completion.
EOF
  echo "Wrote: ${notes_path}"
}

write_runtime_summary_if_missing() {
  local summary_path="${FINAL_DIR}/report_materials/runtime_summary.txt"
  if [[ -f "${summary_path}" ]]; then
    echo "Exists, skip: ${summary_path}"
    return
  fi

  cat > "${summary_path}" <<EOF
Object B runtime summary

Selected trial: ${ADOPTED_TRIAL_REL}
Exported asset dir: ${ADOPTED_EXPORT_DIR#${THREESTUDIO_DIR}/}
Final handoff dir: ${FINAL_DIR}

Training log: ${ADOPTED_TRAIN_LOG}
Training GPU CSV: ${ADOPTED_TRAIN_GPU_CSV}
Export log: ${ADOPTED_EXPORT_LOG}
Export GPU CSV: ${ADOPTED_EXPORT_GPU_CSV}

Training runtime: about 2:45.00
Training peak GPU memory: about 7739 MiB
Export runtime: about 1:34.06
Export peak GPU memory: about 5199 MiB
Adopted total train-plus-export runtime: about 4:19.06
EOF
  echo "Wrote: ${summary_path}"
}

command_collect_adopted() {
  echo "== Collect adopted Object B result =="
  echo "Adopted trial: ${ADOPTED_TRIAL}"
  echo "Adopted export: ${ADOPTED_EXPORT_DIR}"
  echo "Final dir: ${FINAL_DIR}"

  [[ -d "${ADOPTED_TRIAL}" ]] || die "Adopted trial folder not found: ${ADOPTED_TRIAL}"
  [[ -d "${ADOPTED_EXPORT_DIR}" ]] || die "Adopted export folder not found: ${ADOPTED_EXPORT_DIR}"

  mkdir -p \
    "${FINAL_DIR}/asset" \
    "${FINAL_DIR}/previews/test_views" \
    "${FINAL_DIR}/configs" \
    "${FINAL_DIR}/checkpoints" \
    "${FINAL_DIR}/logs" \
    "${FINAL_DIR}/gpu" \
    "${FINAL_DIR}/report_materials"

  echo "== Copy exported asset =="
  copy_required "${ADOPTED_EXPORT_DIR}/model.obj" "${FINAL_DIR}/asset/model.obj"
  copy_required "${ADOPTED_EXPORT_DIR}/model.mtl" "${FINAL_DIR}/asset/model.mtl"
  copy_required "${ADOPTED_EXPORT_DIR}/texture_kd.jpg" "${FINAL_DIR}/asset/texture_kd.jpg"

  echo "== Copy previews =="
  copy_if_exists "${ADOPTED_TRIAL}/save/it1000-0.png" "${FINAL_DIR}/previews/preview_it1000.png"
  copy_if_exists "${ADOPTED_TRIAL}/save/it1000-test.mp4" "${FINAL_DIR}/previews/preview_turntable_it1000.mp4"
  copy_if_exists "${ADOPTED_TRIAL}/save/it1000-test/0.png" "${FINAL_DIR}/previews/test_views/view_0.png"
  copy_if_exists "${ADOPTED_TRIAL}/save/it1000-test/30.png" "${FINAL_DIR}/previews/test_views/view_30.png"
  copy_if_exists "${ADOPTED_TRIAL}/save/it1000-test/60.png" "${FINAL_DIR}/previews/test_views/view_60.png"
  copy_if_exists "${ADOPTED_TRIAL}/save/it1000-test/90.png" "${FINAL_DIR}/previews/test_views/view_90.png"

  echo "== Copy configs and command records =="
  copy_if_exists "${ADOPTED_TRIAL}/configs/parsed.yaml" "${FINAL_DIR}/configs/parsed.yaml"
  copy_if_exists "${ADOPTED_TRIAL}/configs/raw.yaml" "${FINAL_DIR}/configs/raw.yaml"
  copy_if_exists "${ADOPTED_TRIAL}/cmd.txt" "${FINAL_DIR}/configs/cmd.txt"
  copy_if_exists "${ADOPTED_TRIAL}/csv_logs/version_0/hparams.yaml" "${FINAL_DIR}/configs/hparams.yaml"
  copy_if_exists "${ADOPTED_TRIAL}/csv_logs/version_0/metrics.csv" "${FINAL_DIR}/report_materials/metrics.csv"

  echo "== Copy checkpoints =="
  copy_if_exists "${ADOPTED_TRIAL}/ckpts/last.ckpt" "${FINAL_DIR}/checkpoints/last.ckpt"
  copy_if_exists "${ADOPTED_TRIAL}/ckpts/epoch=0-step=1000.ckpt" "${FINAL_DIR}/checkpoints/epoch=0-step=1000.ckpt"

  echo "== Copy logs if available =="
  copy_if_exists "${ADOPTED_TRAIN_LOG}" "${FINAL_DIR}/logs/train.log"
  copy_if_exists "${ADOPTED_EXPORT_LOG}" "${FINAL_DIR}/logs/export.log"
  copy_if_exists "${ADOPTED_TRAIN_GPU_CSV}" "${FINAL_DIR}/gpu/train_gpu.csv"
  copy_if_exists "${ADOPTED_EXPORT_GPU_CSV}" "${FINAL_DIR}/gpu/export_gpu.csv"

  write_notes_if_missing
  write_runtime_summary_if_missing

  command_sanity_final
}

command_sanity_final() {
  echo "== Object B final sanity check =="
  echo "Final dir: ${FINAL_DIR}"

  local required=(
    "${FINAL_DIR}/asset/model.obj"
    "${FINAL_DIR}/asset/model.mtl"
    "${FINAL_DIR}/asset/texture_kd.jpg"
    "${FINAL_DIR}/previews/preview_it1000.png"
    "${FINAL_DIR}/previews/preview_turntable_it1000.mp4"
    "${FINAL_DIR}/checkpoints/last.ckpt"
    "${FINAL_DIR}/configs/parsed.yaml"
    "${FINAL_DIR}/configs/raw.yaml"
    "${FINAL_DIR}/logs/train.log"
    "${FINAL_DIR}/logs/export.log"
    "${FINAL_DIR}/gpu/train_gpu.csv"
    "${FINAL_DIR}/gpu/export_gpu.csv"
    "${FINAL_DIR}/notes.md"
    "${FINAL_DIR}/report_materials/runtime_summary.txt"
  )

  for item in "${required[@]}"; do
    [[ -s "${item}" ]] || die "Missing or empty final file: ${item}"
    echo "OK $(stat -c '%s bytes' "${item}") ${item}"
  done

  local texture_ref
  texture_ref="$(awk '/^map_Kd / {print $2; exit}' "${FINAL_DIR}/asset/model.mtl")"
  [[ -n "${texture_ref}" ]] || die "model.mtl does not contain map_Kd texture reference."
  [[ -f "${FINAL_DIR}/asset/${texture_ref}" ]] || die "MTL texture reference is missing: ${texture_ref}"
  echo "MTL_TEXTURE_REFERENCE_OK=${texture_ref}"

  echo "== OBJ / MTL basic stats =="
  wc -l "${FINAL_DIR}/asset/model.obj" "${FINAL_DIR}/asset/model.mtl"

  echo "== Runtime summary =="
  sed -n '1,120p' "${FINAL_DIR}/report_materials/runtime_summary.txt"

  echo "SANITY_CHECK_DONE"
}

command_all() {
  command_preflight
  command_train
  command_export_latest
  echo "ALL_DONE_FOR_NEW_TRIAL"
  echo "Inspect the latest preview PNG/MP4 before collecting it as a final Object B asset."
  echo "For the adopted 2026-06-13 result, run: bash scripts/run_object_B_text3d.sh collect-adopted"
  echo "Then run: bash scripts/run_object_B_text3d.sh sanity-final"
}

main() {
  local command="${1:-help}"
  case "${command}" in
    help|-h|--help) usage ;;
    preflight) command_preflight ;;
    train) command_train ;;
    export-latest) command_export_latest ;;
    collect-adopted) command_collect_adopted ;;
    sanity-final) command_sanity_final ;;
    all) command_all ;;
    *) usage; die "Unknown command: ${command}" ;;
  esac
}

main "$@"
