date="250317"
subtask="NRHints"
data_root="E:\\gs3_Dataset"
view_num=2000
data_device="cpu"
add_iter=1.2


iterations=60000
asg_freeze_step=26400
spcular_freeze_step=10800
fit_linear_step=8400
asg_lr_freeze_step=48000
asg_lr_max_steps=60000
train_cam_freeze_step=6000
opt_cam_lr_max_steps=96000
train_pl_freeze_step=18000
opt_pl_lr_max_steps=96000
local_q_lr_freeze_step=48000
local_q_lr_max_steps=60000
freeze_phasefunc_steps=60000
neural_phasefunc_lr_max_steps=60000
position_lr_max_steps=84000
densify_until_iter=120000
unfreeze_iterations=6000
end_iterations=120000


: << option
iterations=$(python3 -c "print(int(50000 * $add_iter))"| tr -d '\n')
asg_freeze_step=$(python3 -c "print(int(22000 * $add_iter))"| tr -d '\n')
spcular_freeze_step=$(python3 -c "print(int(9000 * $add_iter))"| tr -d '\n')
fit_linear_step=$(python3 -c "print(int(7000 * $add_iter))"| tr -d '\n')
asg_lr_freeze_step=$(python3 -c "print(int(40000 * $add_iter))"| tr -d '\n')
asg_lr_max_steps=$(python3 -c "print(int(50000 * $add_iter))"| tr -d '\n')
train_cam_freeze_step=$(python3 -c "print(int(5000 * $add_iter))"| tr -d '\n')
opt_cam_lr_max_steps=$(python3 -c "print(int(80000 * $add_iter))"| tr -d '\n')
train_pl_freeze_step=$(python3 -c "print(int(15000 * $add_iter))"| tr -d '\n')
opt_pl_lr_max_steps=$(python3 -c "print(int(80000 * $add_iter))"| tr -d '\n')
local_q_lr_freeze_step=$(python3 -c "print(int(40000 * $add_iter))"| tr -d '\n')
local_q_lr_max_steps=$(python3 -c "print(int(50000 * $add_iter))"| tr -d '\n')
freeze_phasefunc_steps=$(python3 -c "print(int(50000 * $add_iter))"| tr -d '\n')
neural_phasefunc_lr_max_steps=$(python3 -c "print(int(50000 * $add_iter))"| tr -d '\n')
position_lr_max_steps=$(python3 -c "print(int(70000 * $add_iter))"| tr -d '\n')
densify_until_iter=$(python3 -c "print(int(100000 * $add_iter))"| tr -d '\n')
unfreeze_iterations=$(python3 -c "print(int(5000 * $add_iter))"| tr -d '\n')
end_iterations=$(python3 -c "print(int(100000 * $add_iter))"| tr -d '\n')




python train.py -s $data_root/NRHints/Fish  \
                --data_device $data_device \
                --view_num $view_num \
                --iterations 50000 \
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
                --test_iterations 2000 4000 7000 10000 12000 15000 18000 20000 25000 30000 35000 40000 45000 50000 55000 60000 65000 70000 75000 80000 85000 90000 95000 100000 \
                --save_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 95000 100000 \
                --checkpoint_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 95000 100000 \
                --unfreeze_iterations 5000 \
                -m "E:\\why\\why1" \
                --use_nerual_phasefunc \
                --cam_opt \
                --pl_opt \
                --densify_grad_threshold 0.00012 \
                --eval \
                --asg_channel_num 1 \
                --asg_alpha_num 3 \
                #--rasterizer 3dgs

option



python train.py -s $data_root/NRHints/Fish  \
                --data_device $data_device \
                --view_num $view_num \
                --iterations $iterations \
                --asg_freeze_step $asg_freeze_step \
                --spcular_freeze_step $spcular_freeze_step \
                --fit_linear_step $fit_linear_step \
                --asg_lr_freeze_step $asg_lr_freeze_step \
                --asg_lr_max_steps $asg_lr_max_steps \
                --train_cam_freeze_step $train_cam_freeze_step \
                --opt_cam_lr_max_steps $opt_cam_lr_max_steps \
                --train_pl_freeze_step $train_pl_freeze_step \
                --opt_pl_lr_max_steps $opt_pl_lr_max_steps \
                --asg_lr_init 0.01 \
                --asg_lr_final 0.0001 \
                --local_q_lr_freeze_step $local_q_lr_freeze_step \
                --local_q_lr_init 0.01 \
                --local_q_lr_final 0.0001 \
                --local_q_lr_max_steps $local_q_lr_max_steps \
                --neural_phasefunc_lr_init 0.001 \
                --neural_phasefunc_lr_final 0.00001 \
                --freeze_phasefunc_steps $freeze_phasefunc_steps \
                --neural_phasefunc_lr_max_steps $neural_phasefunc_lr_max_steps \
                --position_lr_max_steps $position_lr_max_steps \
                --densify_until_iter $densify_until_iter \
                --test_iterations $(seq 2000 2000 $end_iterations) \
                --save_iterations $(seq 10000 10000 $end_iterations) \
                --checkpoint_iterations $(seq 10000 10000 $end_iterations) \
                --unfreeze_iterations $unfreeze_iterations \
                -m "E:\\why\\why4" \
                --use_nerual_phasefunc \
                --cam_opt \
                --pl_opt \
                --densify_grad_threshold 0.00012 \
                --eval \
                --asg_channel_num 1 \
                --asg_alpha_num 3 \
                --rasterizer 3dgs

