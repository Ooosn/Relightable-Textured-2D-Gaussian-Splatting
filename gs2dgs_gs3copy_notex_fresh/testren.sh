#!/usr/bin/env bash

export CUDA_HOME="/opt/cuda/current"
export TORCH_CUDA_ARCH_LIST="9.0"
export PATH="${CUDA_HOME}/bin:/home/wangyy/miniconda3/envs/gs3/bin:/home/wangyy/miniconda3/bin:${PATH}"
export LD_LIBRARY_PATH="/home/wangyy/miniconda3/envs/gs3/lib:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONUNBUFFERED=1
export SSGS_SERIAL_STREAM=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

set_visible_gpu() {
  local fallback_gpu="$1"
  if [[ -n "${SGE_HGR_h100:-}" || -n "${SGE_HGR_TASK_h100:-}" ]]; then
    unset CUDA_VISIBLE_DEVICES
    echo "SGE allocated h100=${SGE_HGR_h100:-${SGE_HGR_TASK_h100:-}}; leaving CUDA_VISIBLE_DEVICES unset for scheduler GPU remapping"
  elif [[ -n "${RTS_GPU_IDS:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$RTS_GPU_IDS"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  else
    export CUDA_VISIBLE_DEVICES="$fallback_gpu"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  fi
}

export PYTHONPATH="/home/wangyy/RTS/gs3/submodules/gsplat-1.1.1:/home/wangyy/RTS/gs3/submodules/simple-knn:/home/wangyy/RTS/gs3/submodules/diff-gaussian-rasterization:/home/wangyy/RTS/gs3/submodules/diff-gaussian-rasterization_light:/home/wangyy/RTS/gs3/submodules/diff-gaussian-rasterization_hgs:/home/wangyy/RTS/gs3/submodules/v_3dgs:/home/wangyy/RTS/gs3/submodules/v_3dgs_ortho:/home/wangyy/RTS/gs3${PYTHONPATH:+:${PYTHONPATH}}"
set_visible_gpu 5
python render.py -m /home/wangyy/data_download/gsrelight_runs/gs3/NRHints/20260419_100k/Pixiu \
                --load_iteration -1 \
                --skip_train \
                --use_nerual_phasefunc \
                --write_images \
                --offset 0.015 \
                --load_num 400 \
                --force_save \
                        
: <<EOF

                --valid \
--gamma

                --skip_test \
                --calculate_fps \
--synthesize_video  


                
EOF





