date="250317"
subtask="NRHints"
data_root="E:\\gs3_Dataset"
view_num=20
data_device="cpu"
python finetune.py -s $data_root/NRHints/Fish  \
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
                --test_iterations 2000 7000 10000 12000 15000 18000 20000 25000 30000 35000 40000 45000 50000 55000 60000 65000 70000 75000 80000 85000 90000 95000 100000 \
                --save_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 95000 100000 \
                --checkpoint_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 95000 100000 \
                --unfreeze_iterations 5000 \
                -m "E:\\asg\\fish\\reset_opacity" \
                --use_nerual_phasefunc \
                --cam_opt \
                --pl_opt \
                --densify_grad_threshold 0.00012 \
                --eval \
                --asg_channel_num 1 \
                --asg_alpha_num 3 \
                --start_checkpoint "E:\\asg\\fish\\alpha1-3\\chkpnt100000.pth" \
                --use_hgs_finetune \
