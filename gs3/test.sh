date="gogo"
subtask="NRHints"
data_root="/home/wangyy/data_download/gsrelight"
view_num=2000
output_dir="/home/wangyy/data_download/gsrelight_runs/gs3"
model="Pixiu"

export PYTHONPATH="/home/wangyy/RTS/gs3/submodules/simple-knn:/home/wangyy/RTS/gs3/submodules/diff-gaussian-rasterization:/home/wangyy/RTS/gs3/submodules/diff-gaussian-rasterization_light:/home/wangyy/RTS/gs3/submodules/diff-gaussian-rasterization_hgs:/home/wangyy/RTS/gs3/submodules/v_3dgs:/home/wangyy/RTS/gs3/submodules/v_3dgs_ortho:/home/wangyy/RTS/gs3${PYTHONPATH:+:${PYTHONPATH}}"

## ==========================================================
## =======================LightStage=========================

python train.py -s $data_root/$subtask/$model/$model  \
                --data_device "cpu" \
                --view_num $view_num \
                --iterations 30000 \
                --asg_freeze_step 22000 \
                --spcular_freeze_step 9000 \
                --fit_linear_step 7000 \
                --asg_lr_freeze_step 40000 \
                --asg_lr_max_steps 50000 \
                --asg_lr_init 0.01 \
                --asg_lr_final 0.0001 \
                --local_q_lr_freeze_step 40000 \
                --local_q_lr_init 0.01 \
                --local_q_lr_final 0.0001 \
                --local_q_lr_max_steps 50000 \
                --neural_phasefunc_lr_init 0.001 \
                --neural_phasefunc_lr_final 0.00001 \
                --freeze_phasefunc_steps 50000 \
                --neural_phasefunc_lr_max_steps 50000 \
                --position_lr_max_steps 70000 \
                --densify_until_iter 90000 \
                --test_iterations 2000 4000 7000 10000 15000 20000 25000 30000 \
                --save_iterations 7000 10000 15000 20000 30000 \
                --checkpoint_iterations 7000 10000 15000 20000 30000 \
                --unfreeze_iterations 5000 \
                -m "$output_dir/$subtask/$date/$model" \
                --use_nerual_phasefunc \
                --cam_opt \
                --pl_opt \
                --densify_grad_threshold 0.00015 \
                --eval \
                

 : <<EOF
--gamma_change
EOF               
