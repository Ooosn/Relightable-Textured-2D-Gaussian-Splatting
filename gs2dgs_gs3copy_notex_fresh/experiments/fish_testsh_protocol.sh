#!/usr/bin/env bash
set -euo pipefail

# Fish comparison protocol.
#
# This script is intentionally the single source of truth for Fish no-texture /
# texture comparisons in this copy.  Keep the common optimization arguments
# identical to gs3/test.sh; texture experiments may only add texture flags and,
# for resume comparisons, a start checkpoint.

ROOT="${ROOT:-/home/wangyy/RTS/gs2dgs_gs3copy_notex_fresh}"
DATA_ROOT="${DATA_ROOT:-/home/wangyy/data_download/gsrelight}"
RUN_ROOT="${RUN_ROOT:-/home/wangyy/data_download/gsrelight_runs/gs2dgs_gs3copy_texture_validation}"
SUBTASK="${SUBTASK:-NRHints}"
MODEL="${MODEL:-Fish}"
VIEW_NUM="${VIEW_NUM:-2000}"
ITERATIONS="${ITERATIONS:-100000}"
CONDA_ENV="${CONDA_ENV:-gs3}"
PYTHON_BIN="${PYTHON_BIN:-/home/wangyy/miniconda3/envs/${CONDA_ENV}/bin/python -u}"
read -r -a PYTHON_CMD <<< "${PYTHON_BIN}"

export PYTHONPATH="${ROOT}/submodules/simple-knn:${ROOT}/submodules/diff-gaussian-rasterization:${ROOT}/submodules/diff-gaussian-rasterization_light:${ROOT}/submodules/diff-gaussian-rasterization_hgs:${ROOT}/submodules/v_3dgs:${ROOT}/submodules/v_3dgs_ortho:${ROOT}/../2dgs/submodules/surfel-texture:${ROOT}/../2dgs/submodules/surfel-texture-deferred:${ROOT}/../2dgs/submodules/diff-surfel-rasterization-shadow:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

COMMON_ARGS=(
  -s "${DATA_ROOT}/${SUBTASK}/${MODEL}/${MODEL}"
  --data_device cpu
  --view_num "${VIEW_NUM}"
  --iterations "${ITERATIONS}"
  --rasterizer 2dgs
  --sh_degree 0
  --resolution 1
  --asg_freeze_step 22000
  --spcular_freeze_step 9000
  --fit_linear_step 7000
  --asg_lr_freeze_step 40000
  --asg_lr_max_steps 50000
  --asg_lr_init 0.01
  --asg_lr_final 0.0001
  --local_q_lr_freeze_step 40000
  --local_q_lr_init 0.01
  --local_q_lr_final 0.0001
  --local_q_lr_max_steps 50000
  --neural_phasefunc_lr_init 0.001
  --neural_phasefunc_lr_final 0.00001
  --freeze_phasefunc_steps 50000
  --neural_phasefunc_lr_max_steps 50000
  --position_lr_max_steps 70000
  --densify_until_iter 90000
  --test_iterations 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000
  --save_iterations 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000
  --checkpoint_iterations 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000
  --unfreeze_iterations 5000
  --use_nerual_phasefunc
  --cam_opt
  --pl_opt
  --densify_grad_threshold 0.00015
  --eval
)

usage() {
  echo "Usage:"
  echo "  $0 notex-full RUN_NAME"
  echo "  $0 tex-full RUN_NAME"
  echo "  $0 notex-from-30k RUN_NAME /path/to/chkpnt30000.pth"
  echo "  $0 tex-from-30k RUN_NAME /path/to/chkpnt30000.pth"
}

mode="${1:-}"
run_name="${2:-}"
if [[ -z "${mode}" || -z "${run_name}" ]]; then
  usage
  exit 2
fi

cd "${ROOT}"

case "${mode}" in
  notex-full)
    "${PYTHON_CMD[@]}" train.py \
      "${COMMON_ARGS[@]}" \
      -m "${RUN_ROOT}/${run_name}"
    ;;
  tex-full)
    "${PYTHON_CMD[@]}" train.py \
      "${COMMON_ARGS[@]}" \
      -m "${RUN_ROOT}/${run_name}" \
      --use_textures \
      --texture_resolution 4 \
      --texture_effect_mode per_uv_micro_normal \
      --texture_start_iter 30000
    ;;
  notex-from-30k)
    start_checkpoint="${3:-}"
    if [[ -z "${start_checkpoint}" ]]; then
      usage
      exit 2
    fi
    if [[ "${start_checkpoint}" != *chkpnt30000.pth ]]; then
      echo "Refusing no-texture comparison: start checkpoint must be chkpnt30000.pth" >&2
      exit 2
    fi
    "${PYTHON_CMD[@]}" train.py \
      "${COMMON_ARGS[@]}" \
      -m "${RUN_ROOT}/${run_name}" \
      --start_checkpoint "${start_checkpoint}"
    ;;
  tex-from-30k)
    start_checkpoint="${3:-}"
    if [[ -z "${start_checkpoint}" ]]; then
      usage
      exit 2
    fi
    if [[ "${start_checkpoint}" != *chkpnt30000.pth ]]; then
      echo "Refusing texture comparison: start checkpoint must be chkpnt30000.pth" >&2
      exit 2
    fi
    "${PYTHON_CMD[@]}" train.py \
      "${COMMON_ARGS[@]}" \
      -m "${RUN_ROOT}/${run_name}" \
      --start_checkpoint "${start_checkpoint}" \
      --use_textures \
      --texture_resolution 4 \
      --texture_effect_mode per_uv_micro_normal \
      --texture_start_iter 30000
    ;;
  *)
    usage
    exit 2
    ;;
esac
