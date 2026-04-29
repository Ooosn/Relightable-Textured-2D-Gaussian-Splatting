#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

cxx_compiler_flags = []

# 让编译器安静一点
if os.name == 'nt':
    cxx_compiler_flags.append("/wd4624")


# 设置编译参数
setup(
    name="simple_knn",
    ext_modules=[
        CUDAExtension(
            name="simple_knn._C",
            sources=[
            "spatial.cu", 
            "simple_knn.cu",
            "ext.cpp"],
            # xtra_compile_args ————设置显卡编译参数（对应显卡的架构）
                # 例如：extra_compile_args={"nvcc": ["-arch=sm_75"]（对应 30 系显卡，40 系显卡为 sm_86）, 
            # "cxx" ———— 设置 C++ 编译参数
            extra_compile_args={"nvcc": [], 
            "cxx": cxx_compiler_flags})
        ],
    # 使用 torch 的 build_ext 扩展，setup本身的设置不支持 CUDA 的编译，因此需要使用 torch 的 build_ext 扩展
    cmdclass={
        'build_ext': BuildExtension
    }
)
