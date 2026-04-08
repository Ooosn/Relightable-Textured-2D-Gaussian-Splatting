@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
set PATH=D:\Users\namew\miniconda3\envs\mygs;D:\Users\namew\miniconda3\envs\mygs\Scripts;%PATH%

D:\Users\namew\miniconda3\envs\mygs\python.exe D:\RTS\gs3\train.py ^
  -s E:\gsrelight-data\NRHints\Pixiu ^
  -m D:\RTS\output\gs3_NRHints_Pixiu ^
  --data_device cpu ^
  --view_num 2000 ^
  --iterations 30000 ^
  --asg_freeze_step 22000 ^
  --spcular_freeze_step 9000 ^
  --fit_linear_step 7000 ^
  --asg_lr_freeze_step 40000 ^
  --asg_lr_max_steps 50000 ^
  --asg_lr_init 0.01 ^
  --asg_lr_final 0.0001 ^
  --local_q_lr_freeze_step 40000 ^
  --local_q_lr_init 0.01 ^
  --local_q_lr_final 0.0001 ^
  --local_q_lr_max_steps 50000 ^
  --neural_phasefunc_lr_init 0.001 ^
  --neural_phasefunc_lr_final 0.00001 ^
  --freeze_phasefunc_steps 50000 ^
  --neural_phasefunc_lr_max_steps 50000 ^
  --position_lr_max_steps 70000 ^
  --densify_until_iter 90000 ^
  --test_iterations 2000 4000 7000 10000 15000 20000 25000 30000 ^
  --save_iterations 7000 10000 15000 20000 30000 ^
  --checkpoint_iterations 7000 10000 15000 20000 30000 ^
  --unfreeze_iterations 5000 ^
  --use_nerual_phasefunc ^
  --cam_opt ^
  --pl_opt ^
  --densify_grad_threshold 0.00015 ^
  --eval
