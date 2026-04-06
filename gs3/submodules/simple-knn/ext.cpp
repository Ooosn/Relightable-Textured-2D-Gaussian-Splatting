/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "spatial.h"



/* 
 * 注册一个 PyTorch 扩展模块（使用 PYBIND11_MODULE）
 * 提供一个 Python 可以调用的 C++/CUDA 函数（distCUDA2）
 * 引入 torch/extension.h，提供 PyTorch 的扩展接口，即 PYBIND11_MODULE
 * 引入 spatial.h，里面定义了 distCUDA2
 * TORCH_EXTENSION_NAME 为模块名，后续可以用 import 导入 (BuildExtension 在编译时会自动给 TORCH_EXTENSION_NAME 赋值)
 * m 为模块对象，可以向其中添加函数，.def() 添加函数
*/
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("distCUDA2", &distCUDA2);  
}
