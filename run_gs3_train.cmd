@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
set PATH=D:\Users\namew\miniconda3\envs\mygs;D:\Users\namew\miniconda3\envs\mygs\Scripts;%PATH%
D:\Users\namew\miniconda3\envs\mygs\python.exe d:\RTS\gs3\train.py -s d:\RTS\data\synthetic_shadow_single_object -m d:\RTS\output\gs3_synth_single_object_compare_7000 --white_background --iterations 7000
