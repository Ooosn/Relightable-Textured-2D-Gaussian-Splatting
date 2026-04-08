@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
set PATH=D:\Users\namew\miniconda3\envs\mygs;D:\Users\namew\miniconda3\envs\mygs\Scripts;%PATH%
set CUDA_LAUNCH_BLOCKING=0

D:\Users\namew\miniconda3\envs\mygs\python.exe D:\RTS\2dgs\train.py ^
  -s E:\gsrelight-data\NRHints\Pixiu ^
  -m D:\RTS\output\2dgs_NRHints_Pixiu_v7 ^
  --use_mbrdf ^
  --shadow_pass ^
  --cam_opt ^
  --pl_opt ^
  --view_num 2000 ^
  --iterations 30000 ^
  --unfreeze_iterations 5000 ^
  --spcular_freeze_step 9000 ^
  --fit_linear_step 7000 ^
  --asg_freeze_step 22000 ^
  --asg_lr_init 0.01 ^
  --asg_lr_final 0.0001 ^
  --asg_lr_max_steps 50000 ^
  --local_q_lr_init 0.01 ^
  --local_q_lr_final 0.0001 ^
  --local_q_lr_max_steps 50000 ^
  --neural_phasefunc_lr_init 0.001 ^
  --neural_phasefunc_lr_final 0.00001 ^
  --neural_phasefunc_lr_max_steps 50000 ^
  --position_lr_max_steps 70000 ^
  --densify_until_iter 90000 ^
  --test_iterations 2000 4000 7000 10000 15000 20000 25000 30000 ^
  --save_iterations 7000 10000 15000 20000 30000 ^
  --checkpoint_iterations 7000 10000 15000 20000 30000 ^
  --densify_grad_threshold 0.00015 ^
  --opacity_cull 0.005 ^
  --data_device cpu ^
  --eval
