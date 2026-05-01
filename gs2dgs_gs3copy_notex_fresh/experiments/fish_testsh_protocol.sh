#!/usr/bin/env bash
set -euo pipefail

# Fish comparison protocol.
#
# This script is intentionally the single source of truth for Fish no-texture /
# texture comparisons in this copy.  Keep the common optimization arguments
# identical to gs3/test.sh; texture experiments may only add texture flags and,
# for resume comparisons, a start checkpoint.
#
# Launch rule: use the environment Python directly, not `conda run`, for
# training.  If CUDA imports need a smoke test, import torch first; bare
# extension imports can fail on libc10.so even when training imports are fine.

ROOT="${ROOT:-/home/wangyy/RTS/gs2dgs_gs3copy_notex_fresh}"
DATA_ROOT="${DATA_ROOT:-/home/wangyy/data_download/gsrelight}"
RUN_ROOT="${RUN_ROOT:-/home/wangyy/data_download/gsrelight_runs/gs2dgs_gs3copy_texture_validation}"
SUBTASK="${SUBTASK:-NRHints}"
MODEL="${MODEL:-Fish}"
VIEW_NUM="${VIEW_NUM:-2000}"
ITERATIONS="${ITERATIONS:-100000}"
CONDA_ENV="${CONDA_ENV:-gs3}"
PYTHON_BIN="${PYTHON_BIN:-/home/wangyy/miniconda3/envs/${CONDA_ENV}/bin/python -u}"
TEXTURE_EFFECT_MODE="${TEXTURE_EFFECT_MODE:-uvshadow_specular_residual}"
TEXTURE_RESOLUTION="${TEXTURE_RESOLUTION:-4}"
TEXTURE_START_ITER="${TEXTURE_START_ITER:-30000}"
TEXTURE_SPECULAR_LR_SCALE="${TEXTURE_SPECULAR_LR_SCALE:-1.0}"
TEXTURE_NORMAL_LR_SCALE="${TEXTURE_NORMAL_LR_SCALE:-1.0}"
MBRDF_NORMAL_SOURCE="${MBRDF_NORMAL_SOURCE:-local_q}"
TEXTURE_FREEZE_GAUSSIAN_DENSIFY="${TEXTURE_FREEZE_GAUSSIAN_DENSIFY:-0}"
TEXTURE_REFINE="${TEXTURE_REFINE:-0}"
TEXTURE_MIN_RESOLUTION="${TEXTURE_MIN_RESOLUTION:-4}"
TEXTURE_MAX_RESOLUTION="${TEXTURE_MAX_RESOLUTION:-64}"
TEXTURE_RTG_REFINE_FROM_ITER="${TEXTURE_RTG_REFINE_FROM_ITER:-30000}"
TEXTURE_RTG_REFINE_UNTIL_ITER="${TEXTURE_RTG_REFINE_UNTIL_ITER:-100000}"
TEXTURE_RTG_REFINE_INTERVAL="${TEXTURE_RTG_REFINE_INTERVAL:-1000}"
TEXTURE_RTG_REFINE_FRACTION="${TEXTURE_RTG_REFINE_FRACTION:-0.02}"
TEXTURE_RTG_MIN_SCORE="${TEXTURE_RTG_MIN_SCORE:-1e-10}"
TEXTURE_RTG_RESOLUTION_GAMMA="${TEXTURE_RTG_RESOLUTION_GAMMA:-1.0}"
TEXTURE_RTG_ALPHA_WEIGHT="${TEXTURE_RTG_ALPHA_WEIGHT:-0.0}"
TEXTURE_RTG_OPTIMIZER_STATE_SCALE="${TEXTURE_RTG_OPTIMIZER_STATE_SCALE:-0.5}"
read -r -a PYTHON_CMD <<< "${PYTHON_BIN}"

export PYTHONPATH="${ROOT}/submodules/simple-knn:${ROOT}/submodules/diff-gaussian-rasterization:${ROOT}/submodules/diff-gaussian-rasterization_light:${ROOT}/submodules/diff-gaussian-rasterization_hgs:${ROOT}/submodules/v_3dgs:${ROOT}/submodules/v_3dgs_ortho:${ROOT}/../gs3/submodules/diff-gaussian-rasterization:${ROOT}/../gs3/submodules/diff-gaussian-rasterization_light:${ROOT}/../gs3/submodules/diff-gaussian-rasterization_hgs:${ROOT}/../gs3/submodules/v_3dgs:${ROOT}/../gs3/submodules/v_3dgs_ortho:${ROOT}/submodules/surfel-texture:${ROOT}/submodules/surfel-texture-deferred:${ROOT}/submodules/diff-surfel-rasterization-shadow:${ROOT}/../2dgs/submodules/surfel-texture:${ROOT}/../2dgs/submodules/surfel-texture-deferred:${ROOT}/../2dgs/submodules/diff-surfel-rasterization-shadow:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

COMMON_ARGS=(
  -s "${DATA_ROOT}/${SUBTASK}/${MODEL}/${MODEL}"
  --data_device cpu
  --view_num "${VIEW_NUM}"
  --iterations "${ITERATIONS}"
  --rasterizer 2dgs
  --mbrdf_normal_source "${MBRDF_NORMAL_SOURCE}"
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
  --test_iterations 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000 "${ITERATIONS}"
  --save_iterations 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000 "${ITERATIONS}"
  --checkpoint_iterations 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000 "${ITERATIONS}"
  --unfreeze_iterations 5000
  --use_nerual_phasefunc
  --cam_opt
  --pl_opt
  --densify_grad_threshold 0.00015
  --eval
)

TEXTURE_ARGS=(
  --use_textures
  --texture_resolution "${TEXTURE_RESOLUTION}"
  --texture_effect_mode "${TEXTURE_EFFECT_MODE}"
  --texture_specular_lr_scale "${TEXTURE_SPECULAR_LR_SCALE}"
  --texture_normal_lr_scale "${TEXTURE_NORMAL_LR_SCALE}"
  --texture_start_iter "${TEXTURE_START_ITER}"
)
if [[ "${TEXTURE_FREEZE_GAUSSIAN_DENSIFY}" == "1" || "${TEXTURE_FREEZE_GAUSSIAN_DENSIFY}" == "true" ]]; then
  TEXTURE_ARGS+=(--texture_freeze_gaussian_densify)
fi
if [[ "${TEXTURE_REFINE}" == "1" || "${TEXTURE_REFINE}" == "true" ]]; then
  TEXTURE_ARGS+=(
    --texture_dynamic_resolution
    --texture_min_resolution "${TEXTURE_MIN_RESOLUTION}"
    --texture_max_resolution "${TEXTURE_MAX_RESOLUTION}"
    --texture_rtg_enabled
    --texture_rtg_refine_from_iter "${TEXTURE_RTG_REFINE_FROM_ITER}"
    --texture_rtg_refine_until_iter "${TEXTURE_RTG_REFINE_UNTIL_ITER}"
    --texture_rtg_refine_interval "${TEXTURE_RTG_REFINE_INTERVAL}"
    --texture_rtg_refine_fraction "${TEXTURE_RTG_REFINE_FRACTION}"
    --texture_rtg_min_score "${TEXTURE_RTG_MIN_SCORE}"
    --texture_rtg_resolution_gamma "${TEXTURE_RTG_RESOLUTION_GAMMA}"
    --texture_rtg_alpha_weight "${TEXTURE_RTG_ALPHA_WEIGHT}"
    --texture_rtg_optimizer_state_scale "${TEXTURE_RTG_OPTIMIZER_STATE_SCALE}"
  )
fi

usage() {
  echo "Usage:"
  echo "  $0 notex-full RUN_NAME"
  echo "  $0 tex-full RUN_NAME"
  echo "  $0 notex-from-30k RUN_NAME /path/to/chkpnt30000.pth"
  echo "  $0 tex-from-30k RUN_NAME /path/to/chkpnt30000.pth"
  echo "  $0 notex-resume RUN_NAME /path/to/chkpntITER.pth"
  echo "  $0 tex-resume RUN_NAME /path/to/chkpntITER.pth"
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
      "${TEXTURE_ARGS[@]}"
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
      "${TEXTURE_ARGS[@]}"
    ;;
  notex-resume)
    start_checkpoint="${3:-}"
    if [[ -z "${start_checkpoint}" ]]; then
      usage
      exit 2
    fi
    "${PYTHON_CMD[@]}" train.py \
      "${COMMON_ARGS[@]}" \
      -m "${RUN_ROOT}/${run_name}" \
      --start_checkpoint "${start_checkpoint}"
    ;;
  tex-resume)
    start_checkpoint="${3:-}"
    if [[ -z "${start_checkpoint}" ]]; then
      usage
      exit 2
    fi
    "${PYTHON_CMD[@]}" train.py \
      "${COMMON_ARGS[@]}" \
      -m "${RUN_ROOT}/${run_name}" \
      --start_checkpoint "${start_checkpoint}" \
      "${TEXTURE_ARGS[@]}"
    ;;
  *)
    usage
    exit 2
    ;;
esac
