date="250315"
subtask="NRHints"
data_root="E:\\gs3_Dataset"
view_num=2000
data_device="cpu"


# -------------------------------------------------- mlp_zero=False --------------------------------------------------
python train.py -s $data_root/NRHints/Cup-Fabric  \
                --data_device $data_device \
                --view_num $view_num \
                --iterations 100000 \
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
                --densify_until_iter 100000 \
                --test_iterations 2000 7000 10000 15000 20000 25000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --save_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --checkpoint_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --unfreeze_iterations 5000 \
                -m "E:\\asg\\Cup-Fabric\\channel1" \
                --use_nerual_phasefunc \
                --cam_opt \
                --pl_opt \
                --densify_grad_threshold 0.00012 \
                --eval \
                --asg_channel_num 1 

python train.py -s $data_root/NRHints/Fish  \
                --data_device $data_device \
                --view_num $view_num \
                --iterations 100000 \
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
                --densify_until_iter 100000 \
                --test_iterations 2000 7000 10000 15000 20000 25000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --save_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --checkpoint_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --unfreeze_iterations 5000 \
                -m "E:\\asg\\fish\\mlp" \
                --use_nerual_phasefunc \
                --cam_opt \
                --pl_opt \
                --densify_grad_threshold 0.00012 \
                --eval \
                --asg_channel_num 1 \
                --asg_mlp 


python train.py -s $data_root/NRHints/Cup-Fabric  \
                --data_device $data_device \
                --view_num $view_num \
                --iterations 100000 \
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
                --densify_until_iter 100000 \
                --test_iterations 2000 7000 10000 15000 20000 25000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --save_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --checkpoint_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --unfreeze_iterations 5000 \
                -m "E:\\asg\\Cup-Fabric\\mlp" \
                --use_nerual_phasefunc \
                --cam_opt \
                --pl_opt \
                --densify_grad_threshold 0.00012 \
                --eval \
                --asg_channel_num 1 \
                --asg_mlp 


# -------------------------------------------------- zero start --------------------------------------------------

python train.py -s $data_root/NRHints/Fish  \
                --data_device $data_device \
                --view_num $view_num \
                --iterations 100000 \
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
                --densify_until_iter 100000 \
                --test_iterations 2000 7000 10000 15000 20000 25000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --save_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --checkpoint_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --unfreeze_iterations 5000 \
                -m "E:\\asg\\fish\\zerostart" \
                --use_nerual_phasefunc \
                --cam_opt \
                --pl_opt \
                --densify_grad_threshold 0.00012 \
                --eval \
                --asg_channel_num 1 \
                --asg_mlp \
                --mlp_zero


python train.py -s $data_root/NRHints/Cup-Fabric  \
                --data_device $data_device \
                --view_num $view_num \
                --iterations 100000 \
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
                --densify_until_iter 100000 \
                --test_iterations 2000 7000 10000 15000 20000 25000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --save_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --checkpoint_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --unfreeze_iterations 5000 \
                -m "E:\\asg\\Cup-Fabric\\zerostart" \
                --use_nerual_phasefunc \
                --cam_opt \
                --pl_opt \
                --densify_grad_threshold 0.00012 \
                --eval \
                --asg_channel_num 1 \
                --asg_mlp \
                --mlp_zero


