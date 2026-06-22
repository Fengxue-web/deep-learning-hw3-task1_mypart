#!/usr/bin/env bash
set -euo pipefail

# HW3 Task 1 cloud final asset collection and verification.
#
# This script operates in the cloud workspace defined by HW3_ASSET_ROOT. It
# prepares or verifies compact result folders under ASSET_ROOT/final_assets.
# In the submitted GitHub repository, the copied inspection copy of these
# results is stored under result_preview/. The script intentionally does not
# delete raw training outputs, and it checks final mesh texture references when
# applicable.

ASSET_ROOT="${HW3_ASSET_ROOT:-/root/autodl-tmp/hw3_task1_assets}"
FINAL_DIR="${ASSET_ROOT}/final_assets"
OBJECT_A_DIR="${FINAL_DIR}/object_A"
OBJECT_B_DIR="${FINAL_DIR}/object_B"
OBJECT_C_DIR="${FINAL_DIR}/object_C"

OBJECT_B_FINAL_DEFAULT="${OBJECT_B_DIR}/object_B_latentnerf_p2_blue_20260613_final"
OBJECT_B_HANDOFF="${HW3_OBJECT_B_HANDOFF:-${OBJECT_B_FINAL_DEFAULT}}"

OBJECT_C_LEGACY_SOURCE="${OBJECT_C_DIR}/object_C_candidate_1200_threshold_sweep_float/threshold_20"
OBJECT_C_HANDOFF="${HW3_OBJECT_C_HANDOFF:-${OBJECT_C_DIR}/recommended_timed_threshold20}"

usage() {
  cat <<EOF
Usage:
  bash scripts/collect_final_assets.sh <command>

Commands:
  help        Show this message.
  prepare     Create final asset folders and notes files when missing.
  object-b    Verify the adopted Object B final handoff folder.
  object-c    Copy the recommended Object C timed threshold_20 asset files.
  check       Print current final asset files.
  all         Run prepare, object-b, object-c, and check.

Object B final folder:
  HW3_OBJECT_B_HANDOFF if set; otherwise:
  ${OBJECT_B_HANDOFF}

Object C source:
  HW3_OBJECT_C_SOURCE if set; otherwise the latest ${OBJECT_C_DIR}/object_C_timed_1200_threshold20_* folder.
  If no timed folder exists, the script falls back to:
  ${OBJECT_C_LEGACY_SOURCE}

Object C handoff folder:
  ${OBJECT_C_HANDOFF}
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

copy_optional_file() {
  local src="$1"
  local dst="$2"
  if [[ -f "${src}" ]]; then
    if [[ -f "${dst}" ]]; then
      echo "Exists, skip: ${dst}"
    else
      mkdir -p "$(dirname "${dst}")"
      cp -a "${src}" "${dst}"
      echo "Copied: ${dst}"
    fi
  else
    echo "Optional file missing, skip: ${src}"
  fi
}

copy_required_file() {
  local src="$1"
  local dst="$2"
  [[ -f "${src}" ]] || die "Required file missing: ${src}"
  if [[ -f "${dst}" ]]; then
    echo "Exists, skip: ${dst}"
  else
    mkdir -p "$(dirname "${dst}")"
    cp -a "${src}" "${dst}"
    echo "Copied: ${dst}"
  fi
}

validate_mtl_texture() {
  local asset_dir="$1"
  local mtl_path="${asset_dir}/model.mtl"
  [[ -f "${mtl_path}" ]] || die "Missing model.mtl: ${mtl_path}"

  local texture_ref
  texture_ref="$(awk '/^map_Kd / {print $2; exit}' "${mtl_path}")"
  [[ -n "${texture_ref}" ]] || die "No map_Kd texture reference found in ${mtl_path}"
  [[ -f "${asset_dir}/${texture_ref}" ]] || die "Texture referenced by model.mtl is missing: ${asset_dir}/${texture_ref}"
  echo "MTL texture reference OK: ${texture_ref}"
}

resolve_object_c_source() {
  if [[ -n "${HW3_OBJECT_C_SOURCE:-}" ]]; then
    echo "${HW3_OBJECT_C_SOURCE}"
    return
  fi

  local latest_timed
  latest_timed="$(find "${OBJECT_C_DIR}" -maxdepth 1 -type d -name "object_C_timed_1200_threshold20_*" 2>/dev/null | sort | tail -1)"
  if [[ -n "${latest_timed}" ]]; then
    echo "${latest_timed}"
    return
  fi

  echo "${OBJECT_C_LEGACY_SOURCE}"
}

create_notes_if_missing() {
  local object_dir="$1"
  local method_name="$2"
  local notes_path="${object_dir}/notes.md"

  if [[ -f "${notes_path}" ]]; then
    echo "notes.md already exists, skip: ${notes_path}"
    return
  fi

  mkdir -p "${object_dir}"
  cat > "${notes_path}" <<EOF
# Asset Notes

- Method used: ${method_name}
- Input used:
- Output format:
- Recommended file for fusion teammate:
- Texture files, if any:
- Preview image:
- Preview video:
- Scale / orientation notes:
- Known visual problems:
- Runtime and hardware notes:
- Related log path:
EOF

  echo "Created: ${notes_path}"
}

command_prepare() {
  echo "== Final asset folder preparation =="
  echo "Final asset root: ${FINAL_DIR}"
  mkdir -p "${OBJECT_A_DIR}" "${OBJECT_B_DIR}" "${OBJECT_C_DIR}"

  create_notes_if_missing "${OBJECT_A_DIR}" "multi-view reconstruction + 3DGS"
  create_notes_if_missing "${OBJECT_B_DIR}" "text-to-3D generation with threestudio"
  create_notes_if_missing "${OBJECT_C_DIR}" "single-image-to-3D generation with Stable Zero123"
}

command_object_b() {
  echo "== Verify Object B adopted final asset =="
  echo "Object B handoff: ${OBJECT_B_HANDOFF}"

  [[ -d "${OBJECT_B_HANDOFF}" ]] || die "Object B final handoff folder not found."

  local required=(
    "${OBJECT_B_HANDOFF}/asset/model.obj"
    "${OBJECT_B_HANDOFF}/asset/model.mtl"
    "${OBJECT_B_HANDOFF}/asset/texture_kd.jpg"
    "${OBJECT_B_HANDOFF}/previews/preview_it1000.png"
    "${OBJECT_B_HANDOFF}/previews/preview_turntable_it1000.mp4"
    "${OBJECT_B_HANDOFF}/checkpoints/last.ckpt"
    "${OBJECT_B_HANDOFF}/configs/parsed.yaml"
    "${OBJECT_B_HANDOFF}/configs/raw.yaml"
    "${OBJECT_B_HANDOFF}/logs/train.log"
    "${OBJECT_B_HANDOFF}/logs/export.log"
    "${OBJECT_B_HANDOFF}/gpu/train_gpu.csv"
    "${OBJECT_B_HANDOFF}/gpu/export_gpu.csv"
    "${OBJECT_B_HANDOFF}/notes.md"
    "${OBJECT_B_HANDOFF}/report_materials/runtime_summary.txt"
  )

  for item in "${required[@]}"; do
    [[ -s "${item}" ]] || die "Missing or empty Object B file: ${item}"
    echo "OK $(stat -c '%s bytes' "${item}") ${item}"
  done

  validate_mtl_texture "${OBJECT_B_HANDOFF}/asset"

  echo "== Object B recommended files for handoff =="
  echo "Recommended mesh: ${OBJECT_B_HANDOFF}/asset/model.obj"
  echo "Material file: ${OBJECT_B_HANDOFF}/asset/model.mtl"
  echo "Texture file: ${OBJECT_B_HANDOFF}/asset/texture_kd.jpg"
  echo "Preview image: ${OBJECT_B_HANDOFF}/previews/preview_it1000.png"
  echo "Preview video: ${OBJECT_B_HANDOFF}/previews/preview_turntable_it1000.mp4"

  echo "== Object B folder overview =="
  find "${OBJECT_B_HANDOFF}" -maxdepth 3 -type f | sort | xargs -r ls -lh
}

command_object_c() {
  echo "== Collect Object C recommended timed threshold_20 result =="
  local object_c_source
  object_c_source="$(resolve_object_c_source)"
  echo "Source: ${object_c_source}"
  echo "Handoff: ${OBJECT_C_HANDOFF}"

  [[ -d "${object_c_source}" ]] || die "Object C source folder not found."
  [[ -f "${object_c_source}/model.obj" ]] || die "Missing model.obj in Object C source."
  [[ -f "${object_c_source}/model.mtl" ]] || die "Missing model.mtl in Object C source."

  mkdir -p "${OBJECT_C_HANDOFF}" "${OBJECT_C_HANDOFF}/logs" "${OBJECT_C_HANDOFF}/gpu"

  copy_required_file "${object_c_source}/model.obj" "${OBJECT_C_HANDOFF}/model.obj"
  copy_required_file "${object_c_source}/model.mtl" "${OBJECT_C_HANDOFF}/model.mtl"
  copy_optional_file "${object_c_source}/texture_kd.jpg" "${OBJECT_C_HANDOFF}/texture_kd.jpg"
  copy_optional_file "${object_c_source}/it1200-test.mp4" "${OBJECT_C_HANDOFF}/it1200-test.mp4"
  copy_optional_file "${object_c_source}/input_rgba.png" "${OBJECT_C_HANDOFF}/input_rgba.png"
  copy_optional_file "${object_c_source}/parsed.yaml" "${OBJECT_C_HANDOFF}/parsed.yaml"
  copy_optional_file "${object_c_source}/export.log" "${OBJECT_C_HANDOFF}/export.log"

  if [[ -d "${object_c_source}/logs" ]]; then
    find "${object_c_source}/logs" -maxdepth 1 -type f | while read -r file; do
      copy_optional_file "${file}" "${OBJECT_C_HANDOFF}/logs/$(basename "${file}")"
    done
  fi
  if [[ -d "${object_c_source}/gpu" ]]; then
    find "${object_c_source}/gpu" -maxdepth 1 -type f | while read -r file; do
      copy_optional_file "${file}" "${OBJECT_C_HANDOFF}/gpu/$(basename "${file}")"
    done
  fi

  if [[ ! -f "${OBJECT_C_HANDOFF}/input_rgba.png" && -f "${ASSET_ROOT}/data/object_C_raw/object_C_rgba.png" ]]; then
    copy_optional_file "${ASSET_ROOT}/data/object_C_raw/object_C_rgba.png" "${OBJECT_C_HANDOFF}/input_rgba.png"
  fi
  if [[ -f "${ASSET_ROOT}/data/object_C_raw/input.jpg" ]]; then
    copy_optional_file "${ASSET_ROOT}/data/object_C_raw/input.jpg" "${OBJECT_C_HANDOFF}/input.jpg"
  fi

  if [[ -f "${object_c_source}/notes.md" ]]; then
    copy_optional_file "${object_c_source}/notes.md" "${OBJECT_C_HANDOFF}/notes.md"
  elif [[ ! -f "${OBJECT_C_HANDOFF}/notes.md" ]]; then
    cat > "${OBJECT_C_HANDOFF}/notes.md" <<'EOF'
# Object C Notes

- Method used: single-image-to-3D generation with Stable Zero123.
- Input used: one RGBA foreground image of the mug.
- Output format: OBJ mesh with MTL and texture image.
- Recommended file for fusion teammate: model.obj, together with model.mtl and texture_kd.jpg.
- Preview video: it1200-test.mp4, if present.
- Selected candidate: timed 1200-step continuation run, fixed isosurface threshold 20.0.
- Scale / orientation notes: downstream scene fusion may need manual scale and orientation adjustment.
- Known visual problems: the cup opening is not physically accurate, geometry is coarse, and backside details are hallucinated from one image.
- Runtime and hardware notes: RTX 5090 was used. The final adopted runtime should be read from the copied training/export logs.
- Related log path: logs/ and gpu/ in this folder, if the source timed folder includes them.
EOF
    echo "Created: ${OBJECT_C_HANDOFF}/notes.md"
  else
    echo "notes.md already exists, skip: ${OBJECT_C_HANDOFF}/notes.md"
  fi

  local object_c_texture_ref
  object_c_texture_ref="$(awk '/^map_Kd / {print $2; exit}' "${OBJECT_C_HANDOFF}/model.mtl")"
  if [[ -n "${object_c_texture_ref}" && -f "${OBJECT_C_HANDOFF}/${object_c_texture_ref}" ]]; then
    echo "MTL texture reference OK: ${object_c_texture_ref}"
  else
    echo "WARNING: Object C model.mtl texture reference could not be fully verified."
  fi

  echo "Object C handoff folder:"
  find "${OBJECT_C_HANDOFF}" -maxdepth 2 -type f | sort | xargs -r ls -lh
}

command_check() {
  echo "== Final asset file overview =="
  mkdir -p "${FINAL_DIR}"
  find "${FINAL_DIR}" -maxdepth 4 -type f | sort | xargs -r ls -lh
}

command_all() {
  command_prepare
  command_object_b
  command_object_c
  command_check
}

main() {
  local command="${1:-help}"
  case "${command}" in
    help|-h|--help) usage ;;
    prepare) command_prepare ;;
    object-b) command_object_b ;;
    object-c) command_object_c ;;
    check) command_check ;;
    all) command_all ;;
    *) usage; die "Unknown command: ${command}" ;;
  esac
}

main "$@"
