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

import torch
import sys
from datetime import datetime
import numpy as np
import random
import cv2 as cv

import torch.nn as nn
import torch.nn.functional as F

"""
作用：
    求函数 Sigmoid 'y = 1/(1+e^-x)' 的反函数
"""
def inverse_sigmoid(x):        #Sigmoid y = 1/(1+e^-x) 的反函数  
    return torch.log(x/(1-x))

"""
Bonus：
    EXR文件，可以拥有16位或32位通道，数据格式是浮点数，拥有更大的动态范围，而不是专注于极端精确。
    通道信息可不止包含颜色，还可包含深度，法线等信息，因此使用时，需要转换。
"""
"""
作用：
    这里默认是exr格式的RGB图像，通道是(H, W, C)，因此需要转换成torch处理文件
参数：
    cv2_image: cv2读取的图像，通道是(H, W, C)
"""
def ExrtoTorch(cv2_image, resolution):   #exr文件转torch处理文件
    resized_image_cv2 = cv.resize(cv2_image, resolution)
    resized_image = torch.from_numpy(resized_image_cv2)
    if len(resized_image.shape) == 3:          #如果是3通道，RGB图像默认通道是(H, W, C)
        return resized_image.permute(2, 0, 1)  #重新排序 # 将张量维度从 (H, W, C) 转换为 (C, H, W)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)  #在最后插入第三个维度，数值为1，进行对齐，并且重新排序

"""
Pillow 是一个 Python 图像处理库，可以处理多种图像格式，
如 JPEG、PNG、TIFF、BMP、GIF 等。通过 Pillow 打开的图像会成为 Pillow 的图像对象
图像对象中封装了多种对图片处理的函数，比如调整亮度、对比度、旋转等，以及代码中用到的resize。
"""
def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)
    
"""
作用：
    返回闭包函数，计算并返回给定 step 的学习率。
    调用时根据传入的参数创建调度器函数，后续可以动态地根据step计算学习率

参数：
    lr_init (float): 初始学习率。
    lr_final (float): 最终学习率。
    lr_delay_steps (int): 延迟步数，即学习率从初始值到最终值之间的过渡阶段。
    lr_delay_mult (float): 延迟系数，用于控制学习率在延迟阶段的衰减速度。
    max_steps (int): 最大迭代步数。
"""
def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        #判断step
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        #判断是否存在延迟步数
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )    #判断当前步数延迟系数，随着步数接近延迟步数逐渐增大，最终变为1
        else:
            delay_rate = 1.0
        #开始衰减
        t = np.clip(step / max_steps, 0, 1)   #用于学习率指数衰减(1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)  #用于学习率指数衰减(2)
        return delay_rate * log_lerp   #延迟系数乘以当前学习率

    return helper

"""
作用：
    直接计算并返回给定 step 的学习率。
"""
def get_expon_lr(step, lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000):
    if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
    #判断是否存在延迟步数
    if lr_delay_steps > 0:
        # A kind of reverse cosine decay.
        delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
            0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
        )    #判断当前步数延迟系数，随着步数接近延迟步数逐渐增大，最终变为1
    else:
        delay_rate = 1.0
    #开始衰减
    t = np.clip(step / max_steps, 0, 1)   #用于学习率指数衰减(1)
    log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)  #用于学习率指数衰减(2)
    return delay_rate * log_lerp   #延迟系数乘以当前学习率

"""
从Lbatch中提取每一个矩阵的上三角部分，按顺序存入每一行。

参数：
    L.shape[0]: Batch_size
    数据类型是 torch.float
    存储设备是 GPU (device="cuda")

返回：
    torch.Tensor: 形状为 (LBatch.shape[0], 6) 的矩阵，其中每一行包含 LBatch 中每个矩阵的上三角元素。

注意：
    因为这里是三维，所以矩阵是3*3的矩阵。
"""
def strip_lowerdiag(L):                  #去除下三角元素（不包括对角线）
    #创建 batch_size*6 的0值矩阵
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    #按顺序提取 LBatch 中每一个矩阵的上对角元素，存入 uncertainty矩阵 的每一行
    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

"""
优化对称矩阵，用下三角元素对对称矩阵进行压缩。

参数：
    sym：torch.Tensor: 形状为 (N, 3, 3) 的对称矩阵张量，其中 N 是矩阵的数量，Batch_size。
"""
def strip_symmetric(sym):
    return strip_lowerdiag(sym)

"""
根据四元数构建旋转矩阵。

参数:
    r (torch.Tensor): 形状为 (N, 4) 的四元数张量，其中 N 是四元数的数量，Batch_size。

返回:
    torch.Tensor: 形状为 (N, 3, 3) 的旋转矩阵张量。

此函数将输入的四元数转换为对应的旋转矩阵。首先计算四元数的范数，
然后将四元数归一化得到单位四元数。接着根据单位四元数的分量计算
旋转矩阵的各个元素。

注意:
    - 输入张量应在 GPU 上。
    - 此函数假设输入的四元数是规范化的。
"""
def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    # 归一化 四元数 ，使最终的旋转矩阵是 严格的正交矩阵
    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

"""
根据给定的缩放因子和旋转角度构建缩放旋转矩阵。

参数:
    s (torch.Tensor): 形状为 (N, 3) 的张量，表示 N 个物体的缩放因子。
    r (float 或 torch.Tensor): 旋转角度，可以是标量或形状为 (N,) 的张量。

返回:
    torch.Tensor: 形状为 (N, 3, 3) 的张量，表示 N 个物体的缩放旋转矩阵。

示例:
    >>> s = torch.tensor([[1.0, 2.0, 3.0]])
    >>> r = torch.tensor([0.7854])  # 45度
    >>> build_scaling_rotation(s, r)
"""
def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L    #torch的默认调用 @：torch.matmul
    return L

"""
此函数用于设置程序的安全状态。

它首先保存当前的标准输出流。然后定义了一个内部类 F ，用于重写标准输出的行为。当 silent 参数为 False 时，会在输出的每一行末尾添加当前的时间戳。类 F 还实现了 flush 方法以确保输出的及时刷新。

接着，函数设置了随机数种子以保证结果的可重复性，并配置了 PyTorch 的相关设置，包括设置使用的 CUDA 设备以及对应的随机数种子。

此函数主要用于控制程序的输出格式和确保一些随机操作的可重复性。
"""
def safe_state(silent):
    old_f = sys.stdout  #记录原始的stdout

    #自定义条件拦截输出
    #每次调用 print() 或 sys.stdout.write() 时，都会调用 F.write(x)
    class F:
        def __init__(self, silent):
            self.silent = silent
        #利用原始stdout的write函数输出
        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    #如果以换行为结尾，则添加时间戳
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    #重定向stdout
    sys.stdout = F(silent)

    #固定种子为0
    random.seed(0)  #设置 Python 标准库的随机数生成器的种子。
    np.random.seed(0)   #设置 NumPy 随机数生成器的种子
    torch.manual_seed(0)    #设置 PyTorch 随机数生成器的种子
    torch.cuda.set_device(torch.device("cuda:0"))   #设置 PyTorch 的默认 GPU 设备为 cuda:0（第一个 GPU）
    torch.cuda.manual_seed(0)   #设置 PyTorch 在当前 GPU 设备上的随机数生成器种子
    torch.cuda.manual_seed_all(0)   #如果使用多 GPU，这个操作会为每个 GPU 设备设置相同的随机数种子


"""
冻结所有参数
"""
def freeze_anything(gaussians):
    for name, item in vars(gaussians).items():
        if isinstance(item, nn.Module):
            for param in item.parameters():
                param.requires_grad = False
            print(f"froze module {name}")
        elif isinstance(item, nn.Parameter):
            item.requires_grad = False
            print(f"froze parameter {name}")

"""
手动调整梯度
"""
class GradientScaler(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input1 = None, input2 = None, input3 = None, input4 = None, input5 = None):
        return input1, input2, input3, input4, input5
    
    @staticmethod
    def backward(ctx, grad_output_input1 = None, grad_output_input2 = None, grad_output_input3 = None, grad_output_input4 = None, grad_output_input5 = None):
        return grad_output_input1, grad_output_input2, grad_output_input3, grad_output_input4, grad_output_input5