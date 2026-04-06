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
necessary = False


# 高斯对象模块
# 核心函数：densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size)

import torch
import math
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
import torch.nn.functional as F
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.neural_phase_function import Neural_phase
from scene.mixture_ASG import Mixture_of_ASG

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
        
        self.kd_activation = torch.nn.Softplus()
        self.mm_asg_aplha_activation = torch.nn.Softplus()


    def __init__(self, modelset, opt=None):

        # passed parameters <- modelset
        sh_degree            = getattr(modelset, "sh_degree", 0)
        use_MBRDF            = getattr(modelset, "use_nerual_phasefunc", False)
        basis_asg_num        = getattr(modelset, "basis_asg_num", 8)
        hidden_feature_size  = getattr(modelset, "phasefunc_hidden_size", 32)
        hidden_feature_layer = getattr(modelset, "phasefunc_hidden_layers", 3)
        phase_frequency      = getattr(modelset, "phasefunc_frequency", 4)
        neural_material_size = getattr(modelset, "neural_material_size", 6)
        maximum_gs           = getattr(modelset, "maximum_gs", 1_000_000)

        asg_channel_num      = getattr(modelset, "asg_channel_num", 1)  # 控制 ASG 的高光基函数通道数量
        asg_mlp              = getattr(modelset, "asg_mlp", False)  # 控制是否使用 ASG MLP，通过神经材料参数修正最终高光通道
        alpha_change         = getattr(modelset, "alpha_change", False)  # 控制是否在训练过程中改变 alpha_asg 的维度，从1变为3
        asg_alpha_num        = getattr(modelset, "asg_alpha_num", 1)  # 控制开始的 alpha_asg 的维度，从1变为3


        use_hgs              = getattr(modelset, "use_hgs", False)
        use_hgs_finetune     = getattr(modelset, "use_hgs_finetune", False)
        
        rasterizer           = getattr(modelset, "rasterizer", "3dgs")  # 3dgs 或者 gsplat 
        
        # 检查 alpha 通道参数设置
        if alpha_change and asg_alpha_num == 3:
            assert asg_channel_num == 1, "alpha_change 和 asg_alpha_num 不能同时启动"
        if asg_channel_num == 3 and asg_alpha_num == 3:
            print("attention: alpha 通道和 alpha_asg 基函数维度同时为 3")
        elif asg_alpha_num == 3:
            print("attention: 仅 alpha_asg 基函数维度为 3")
        elif asg_channel_num == 3:
            print("attention: 仅 alpha 通道为 3")

        # passed parameters <- opt
        mlp_zero            = getattr(opt, "mlp_zero", False)
        
        # 检查 asg 参数设置
        if asg_mlp:
            print("attention: 使用 ASG MLP 修正高光通道")
        if mlp_zero:
            print("attention: 所有 MLP 采用 zero 初始化")

        # 检查渲染器参数设置
        if rasterizer == "gsplat":
            print("attention: 使用 gsplat 渲染器")
        elif rasterizer == "3dgs":
            print("attention: 使用 3dgs 渲染器")
            
        # 初始化渲染器
        self.rasterizer = rasterizer

        # initialize parameters
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        
        # 高斯点云坐标
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)  
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.out_weights_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        
        # maximum Gaussian number
        self.maximum_gs = maximum_gs
        # debug 专用
        self.global_call_counter = 0

        # MBRDF
        self.use_MBRDF = use_MBRDF
        if self.use_MBRDF:
            self.kd = torch.empty(0)
            self.ks = torch.empty(0)
        
            # ASG params
            self.basis_asg_num = basis_asg_num
            self.asg_channel_num = asg_channel_num
            self.asg_mlp = asg_mlp
            self.asg_alpha_num = asg_alpha_num
            self.alpha_change = alpha_change
            self.alpha_asg = torch.empty(0)
            self.asg_func = torch.empty(0)

            # neural phase function
            self.mlp_zero = mlp_zero

            # local frame
            self.local_q = torch.empty(0)
            
            # latent params
            self.neural_material = torch.empty(0)
            self.neural_material_size = neural_material_size
            self.hidden_feature_size = hidden_feature_size
            self.hidden_feature_layer = hidden_feature_layer
            self.phase_frequency = phase_frequency
            self.neural_phasefunc = torch.empty(0)
            
            print("Use MBRDF!")
        
        # hgs
        self.use_hgs = use_hgs
        self.use_hgs_finetune = use_hgs_finetune
        self.hgs_normals = torch.empty(0)
        self.hgs_opacities = torch.empty(0)
        self.small_gaussian = torch.empty(0)
        
    def capture(self):
        return {
        "active_sh_degree": self.active_sh_degree,
        "_xyz": self._xyz,
        "_features_dc": self._features_dc,
        "_features_rest": self._features_rest,  
        "_scaling": self._scaling,
        "_rotation": self._rotation,
        "_opacity": self._opacity,
        "max_radii2D": self.max_radii2D,
        "spatial_lr_scale": self.spatial_lr_scale,
        "maximum_gs": self.maximum_gs,
        
        # MBRDF
        "kd": self.kd if self.use_MBRDF else None,
        "ks": self.ks if self.use_MBRDF else None,
        "basis_asg_num": self.basis_asg_num if self.use_MBRDF else None,
        "alpha_asg": self.alpha_asg if self.use_MBRDF else None,
        "asg_sigma": self.asg_func.asg_sigma if self.use_MBRDF else None,
        "asg_rotation": self.asg_func.asg_rotation if self.use_MBRDF else None,
        "asg_scales": self.asg_func.asg_scales if self.use_MBRDF else None,
        "local_q": self.local_q if self.use_MBRDF else None,
        "neural_material": self.neural_material if self.use_MBRDF else None,
        "neural_material_size": self.neural_material_size if self.use_MBRDF else None,
        "neural_phasefunc": self.neural_phasefunc.state_dict() if self.use_MBRDF else None,
        "asg_mlp": self.asg_mlp if self.use_MBRDF else None,
        "asg_alpha_num": self.asg_alpha_num if self.use_MBRDF else None,

        # hgs
        "hgs_normals": self.hgs_normals if self.use_hgs else None,
        "hgs_opacities": self.hgs_opacities if self.use_hgs else None,
        "small_gaussian": self.small_gaussian if self.use_hgs else None,
        
        # 优化器 这四个按照这个顺序
        "optimizer": self.optimizer.state_dict(),
        "xyz_gradient_accum": self.xyz_gradient_accum,
        "out_weights_accum": self.out_weights_accum,
        "denom": self.denom,
        }
    
    def restore(self, model_args, training_args):
        if isinstance(model_args, dict):
            for key, value in model_args.items():
                if key == "neural_phasefunc":
                    if self.use_MBRDF:
                        self.neural_phasefunc.load_state_dict(value)
                    else:
                        self.neural_phasefunc = value
                elif key in {"shadow_opacity", "start_shadow_opacity", "use_shadow_gs"}:
                    # Ignore legacy shadow-gs fields. The main shadow path now uses
                    # the analytic light rasterizer directly.
                    continue
                elif key.startswith("asg_func"):
                    setattr(self.asg_func, key, value)
                    print("key", key)
                    print(print(value.shape))
                elif key == "optimizer":
                    self.training_setup(training_args)
                else:
                    setattr(self, key, value)  # 直接赋值
                    if isinstance(value, torch.Tensor):
                        print("key", key)
                        print(value.shape)
            if not self.use_hgs_finetune:
                try:
                    self.optimizer.load_state_dict(model_args["optimizer"])
                except ValueError as exc:
                    print(f"Warning: skip loading legacy optimizer state: {exc}")
            
        else: # 加载旧模型
            (self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            self.kd,
            self.ks,
            self.basis_asg_num,
            self.alpha_asg,
            self.asg_func.asg_sigma,
            self.asg_func.asg_rotation,
            self.asg_func.asg_scales,
            self.local_q,
            self.neural_material,
            self.neural_material_size,
            neural_phasefunc_param,
            self._scaling,
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum, 
            out_weights_accum,
            denom,
            opt_dict, 
            self.spatial_lr_scale,
            self.maximum_gs,
            self.asg_mlp) = model_args
            self.training_setup(training_args)
            self.xyz_gradient_accum = xyz_gradient_accum
            self.out_weights_accum = out_weights_accum
            self.denom = denom
            if self.use_MBRDF:
                self.neural_phasefunc.load_state_dict(neural_phasefunc_param)
            else:
                self.neural_phasefunc = neural_phasefunc_param
            self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_kd(self):
        return self.kd_activation(self.kd)
    
    @property
    def get_ks(self):
        return self.kd_activation(self.ks)

    @property
    def get_alpha_asg(self):
        return self.mm_asg_aplha_activation(self.alpha_asg)

    @property
    def get_local_axis(self):
        return build_rotation(self.local_q) # (K, 3, 3)
    
    @property
    def get_local_z(self):
        return self.get_local_axis[:, :, 2] # (K, 3)

    @property
    def get_neural_material(self):
        return self.neural_material
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_hgs_normals(self):
        return self.hgs_normals
    
    @property
    def get_hgs_opacities(self):
        return self.opacity_activation(self.hgs_opacities)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1


    # pcd: point cloud data
    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        # lr：local radius，这里的 spatial_lr_scale，是一个半径，以相机中心为原点，计算相加距离中心的最大半径，也就是整个场景的半径
        self.spatial_lr_scale = spatial_lr_scale
        """
        np.array: 无论接收什么，都会创建 Numpy 数组，然后根据需要进行转换并复制数据
        np.asarray: 对于 Numpy 数组，会直接返回，对于其他的依然也会复制数据
        torch.tensor(): 转换格式，能转换 Python 和 Numpy 数组
        torch.float()：转换为 torch.float32，因为直接转换数据，会尽可能维持原数据格式，因此需要利用 torch.float() 转化为 torch 默认的数据格式
        torch.cuda()：数据移动到 GPU
        """
        # 初始化坐标：
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()

        # 初始化球谐函数：
        # RGB2SH() 负责将 标准 RGB 颜色转换为 SH 0阶项，即高斯点的基础颜色
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        # 创建高斯 SH 属性，fused_color.shape[0] 表示点云的数量，3 表示 RGB 颜色三通道，(self.max_sh_degree + 1) ** 2 计算当前 SH 级别所需的球谐系数总数
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        # 将 SH 0阶项 也就是基础颜色导入高斯点的颜色信息中
        features[:, :3, 0 ] = fused_color
        # SH 的高阶项初始化为 0
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        # 初始化尺度：
        # 计算每个点到最近点的距离，并且保证最小值大于 0.0000001，较小的 dist2 表明 高斯点之间距离近，密度较高，因此可能需要更小的尺度
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        # 利用 dist2 算出每个点的尺度信息，先通过 sqrt 取平方根，再用 log 缩小/平滑尺度变化范围，随后将每个点的尺度复制为 3 维，即 xyz 初始化为同一个尺度
        """
        torch.repeat(1,3): 沿 dim = 1 重复三次
        [...,None]: 等于 torch.unsqueeze(-1)
        """
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)

        # 初始化旋转矩阵：
        # 采用 4元数（quaternion） 代替旋转矩阵
        # 所有点的四元数设置为，[1, 0, 0, 0]，表明没有旋转
        """
        四元数是一种表示 3D 旋转 的数学工具，定义为：q = [w, x, y, z]
            - w 是实部，用于表示旋转角度。
            - x, y, z 是虚部，表示旋转轴方向。
        """
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        # 初始化透明度：
        # 生成一个 形状为 (N,1) 的张量，N 为 点云/高斯点 数量，随后乘以 0.1 ， 使所有点透明度初始化为 0.1 
        # 不直接存储 opacity ， 而是存储 inverse_sigmoid（opacity）
        """
        这是一个常见技巧，用于控制参数范围:
        神经网络优化本身不会自动保持参数在 [0,1] 之内，优化过程中可能会超出这个范围
        如果直接对 opacity 进行优化，因为梯度下降优化是无约束的，它可能会跑出 [0,1]
        inverse_sigmoid 让优化参数可以自由更新，同时保证最终的 opacity 仍然落在 [0,1] 之内，这是一种 “间接控制参数范围” 的方法，解决了神经网络无法直接限制参数范围的问题
        """
        if self.use_hgs:
            hgs_normals =  0.6*torch.ones((fused_point_cloud.shape[0], 3), dtype=torch.float, device="cuda")
            hgs_opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 2), dtype=torch.float, device="cuda"))
        else:
            opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # 场景参数设置
        # 将高斯坐标、高斯特征、高斯透明度设置为神经网络参数，开启梯度计算
        """ 传入 nn 前，需要将数据转换为 tensor 类型 """
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))    # 位置信息

        # 分离 SH 0阶项（Direct Component） 和 其他 SH 高阶项（方向相关颜色） 
        """
        因为这里 RGB 三通道作为最后一个维度来计算，所以需要交换维度:
        1. torch.transpose(dim1, dim2): 只能调整两个维度之间的顺序，适用于简单情况
        2. torch.permute(...): 能调整多个维度之间的顺序，但需要显式指定所有维度顺序
        两种方式都是返回视图，并不改变内存，所以为了保证计算正确时往往还需要 contiguous() 使内存连续化
        """
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        
        # MBRDF 相关
        if self.use_MBRDF:
            
            # 漫反射颜色系数
            kd = torch.ones((features.shape[0], 3), dtype=torch.float, device="cuda")*0.5
            self.kd = nn.Parameter(kd.requires_grad_(True))

            # 镜面反射系数
            ks = torch.ones((features.shape[0], 3), dtype=torch.float, device="cuda")*0.5
            self.ks = nn.Parameter(ks.requires_grad_(True))

            # 基于 ASG (Anisotropic Spherical Gaussians)，混合多个 ASG 以逼近复杂的高光形状
            # 这里 channel_num 是 ASG 的通道数量，用于控制 ASG 的输出维度
            alpha_asg = torch.zeros((features.shape[0], self.basis_asg_num, self.asg_alpha_num), dtype=torch.float, device="cuda")
            self.alpha_asg = nn.Parameter(alpha_asg.requires_grad_(True))
            
            self.asg_func = Mixture_of_ASG(self.basis_asg_num, self.asg_channel_num)  # forward(self, wi, wo, alpha_asg, asg_scales, asg_axises):
            
            # 计算 高斯局部坐标系，用于 asg 相关计算
            local_rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
            local_rots[:, 0] = 1
            self.local_q = nn.Parameter(local_rots.requires_grad_(True))
            
            # 神经材质参数/高斯点可学习的隐变量
            # 相位函数：建立两个 MLP 分别用来优化 shadow 和 添加其他效果颜色信息
            neural_materials = torch.ones((features.shape[0], self.neural_material_size), dtype=torch.float, device="cuda")
            self.neural_material = nn.Parameter(neural_materials.requires_grad_(True))
            self.neural_phasefunc = Neural_phase(hidden_feature_size=self.hidden_feature_size, \
                                                hidden_feature_layers=self.hidden_feature_layer, \
                                                frequency=self.phase_frequency, \
                                                neural_material_size=self.neural_material_size, \
                                                asg_mlp=self.asg_mlp, \
                                                mlp_zero=self.mlp_zero).to(device="cuda")
           
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        if self.use_hgs:
            self.hgs_normals = nn.Parameter(hgs_normals.requires_grad_(True))
            self.hgs_opacities = nn.Parameter(hgs_opacities.requires_grad_(True))

    """
    函数：
        用于将 asg 的 alpha 参数 由 1维 转换为 3 维
    """
    def change_alpha_asg(self, alpha_asg):
        alpha_asg_3 = alpha_asg.repeat(1, 1, 3)
        self.alpha_asg = nn.Parameter(alpha_asg_3.requires_grad_(True))
        self.replace_or_add_tensor_to_optimizer(self.alpha_asg, "alpha_asg")
        self.asg_alpha_num = 3
    """
    函数：
        根据传入的 args 初始化可学习参数以及训练中要用到的参数
    """
    def training_setup(self, training_args):


        if self.use_hgs_finetune:
            for i in vars(self):
                param = getattr(self, i)
                if isinstance(param, nn.Parameter):
                    param.requires_grad = False
                    
            self.hgs_normals.requires_grad = True
            self.hgs_opacities.requires_grad = False

            for i in vars(self.asg_func):
                param = getattr(self.asg_func, i)
                if isinstance(param, nn.Parameter):
                    param.requires_grad = False
            self.neural_phasefunc.freeze()

        # 修建参数初始化
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.out_weights_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # 设置参数组
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "features_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "features_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.use_MBRDF:
            l.append({'params': [self.kd], 'lr': training_args.kd_lr, "name": "kd"})
            l.append({'params': [self.ks], 'lr': training_args.ks_lr, "name": "ks"})
            l.append({'params': [self.alpha_asg], 'lr': training_args.asg_lr_init, "name": "alpha_asg"})
            l.append({'params': [self.asg_func.asg_sigma], 'lr': training_args.asg_lr_init, "name": "asg_sigma"})
            l.append({'params': [self.asg_func.asg_scales], 'lr': training_args.asg_lr_init, "name": "asg_scales"})
            l.append({'params': [self.asg_func.asg_rotation], 'lr': training_args.asg_lr_init, "name": "asg_rotation"})
            l.append({'params': [self.local_q], 'lr': training_args.local_q_lr_init, "name": "local_q"})
            l.append({'params': [self.neural_material], 'lr': training_args.neural_phasefunc_lr_init, "name": "neural_material"})
            l.append({'params': list(self.neural_phasefunc.parameters()), 'lr': training_args.neural_phasefunc_lr_init, "name": "neural_phasefunc"})
        
        if self.use_hgs:
            l.append({'params': [self.hgs_normals], 'lr': 0.003, "name": "hgs_normals"})
            l.append({'params': [self.hgs_opacities], 'lr': training_args.opacity_lr, "name": "hgs_opacities"})

        for gi, g in enumerate(l):
            print(f"\nGROUP {gi}: keys = {list(g.keys())}")
            if "params" not in g:
                print("  [ERROR] NO 'params' KEY - this is not a valid param-group!")
                continue
            for pi, p in enumerate(g["params"]):
                if p is None:
                    print(f"  [ERROR] param {pi} is None")
                elif not isinstance(p, torch.Tensor):
                    print(f"  [ERROR] param {pi} is type {type(p)}")
                else:
                    print(f"  [OK] param {pi}: shape {tuple(p.shape)}, requires_grad={p.requires_grad}")
        # 利用 Adam 优化器优化参数组 l
        # 不同参数学习率不同，并且有些参数可能未设置学习率，因此 lr 设置为0，在 update_learning_rate 中更新
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        # 添加学习率调度器，用于控制起始学习率逐渐上升至最终学习率，决定 delay_rate 
        # 其中基础学习率会指数衰减，决定 log_lerp 
        # 最终 elay_rate * log_lerp 为当前学习率，最终会传递给 optimizer
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.asg_scheduler_args = get_expon_lr_func(lr_init=training_args.asg_lr_init,
                                                    lr_final=training_args.asg_lr_final,
                                                    lr_delay_mult=training_args.asg_lr_delay_mult,
                                                    max_steps=training_args.asg_lr_max_steps)
        
        self.local_q_scheduler_args = get_expon_lr_func(lr_init=training_args.local_q_lr_init,
                                                    lr_final=training_args.local_q_lr_final,
                                                    lr_delay_mult=training_args.local_q_lr_delay_mult,
                                                    max_steps=training_args.local_q_lr_max_steps)
        
        self.neural_phasefunc_scheduler_args = get_expon_lr_func(lr_init=training_args.neural_phasefunc_lr_init,
                                                    lr_final=training_args.neural_phasefunc_lr_final,
                                                    lr_delay_mult=training_args.neural_phasefunc_lr_delay_mult,
                                                    max_steps=training_args.neural_phasefunc_lr_max_steps)
        
        self.hgs_normals_scheduler_args = get_expon_lr_func(lr_init=0.003,
                                                    lr_final=0.00003,
                                                    lr_delay_mult=0.01,
                                                    max_steps=150_000)
        
        self.hgs_opacities_scheduler_args = get_expon_lr_func(lr_init=0.003,
                                                    lr_final=0.00003,
                                                    lr_delay_mult=0.01,
                                                    max_steps=150_000)

    # 根据当前迭代次数调整学习率
    def update_learning_rate(self, iteration, asg_freeze_step=0, local_q_freeze_step=0, freeze_phasefunc_steps=0):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] in ["alpha_asg", "asg_sigma", "asg_rotation", "asg_scales"] :
                lr = self.asg_scheduler_args(max(0, iteration - asg_freeze_step))
                param_group['lr'] = lr
            if param_group["name"] == "local_q":
                lr = self.local_q_scheduler_args(max(0, iteration - local_q_freeze_step))
                param_group['lr'] = lr
            if param_group["name"] == ["neural_phasefunc", "neural_material"]:
                lr = self.neural_phasefunc_scheduler_args(max(0, iteration - freeze_phasefunc_steps))
                param_group['lr'] = lr
            if param_group["name"] == "hgs_normals":
                lr = self.hgs_normals_scheduler_args(max(0, iteration - asg_freeze_step))
                param_group['lr'] = lr  

    # 创建了一个 属性索引列表，用于下文的 ply 读取和存储 ———— ！！！因此要求这里的顺序和后面存储时的数据组合顺序要一致
    # 但是这里只有 每个高斯点的属性，不包括神经网络参数和公用参数，比如asg基函数
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']

        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))

        l.append('opacity')

        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))

        if self.use_MBRDF:
            for i in range(self.kd.shape[1]):
                l.append('kd_{}'.format(i))
            for i in range(self.ks.shape[1]):
                l.append('ks_{}'.format(i))
            for channel in range(self.alpha_asg.shape[2]):
                for basis in range(self.alpha_asg.shape[1]):
                    l.append('alpha_asg_{}_{}'.format(basis, channel))
            for i in range(self.local_q.shape[1]):
                l.append('local_q_{}'.format(i))
            for i in range(self.neural_material.shape[1]):
                l.append('neural_material_{}'.format(i))
            
        if self.use_hgs:
            for i in range(self.hgs_normals.shape[1]):
                l.append('hgs_normals_{}'.format(i))
            for i in range(self.hgs_opacities.shape[1]):
                l.append('hgs_opacities_{}'.format(i))

        return l

    # 以 点云形式 储存场景中的高斯点 可训练参数
    def save_ply(self, path):

        """
        "/home/user/project/file.py"
        1. os.path.dirname(path) 只会 返回路径部分，不会包含文件名
            - '/home/user/project'
		2. os.path.basename(path) 才是 获取文件名
            - 'file.py'
        3. os.path.splitext(path) 返回值：一个元组 (filename, extension)
            - ('/home/user/project/file', '.py')
        4. os.path.splitext(os.path.basename(path)) 拆分文件名和扩展名
            - ('file', '.py')
        """
        mkdir_p(os.path.dirname(path))

        # 从 tensor 中导出数据:
        # detach() 创建一个新的 tensor，但不包含计算图信息，如果 grad 为 true 的 tensro 直接 .cpu().numpy()，会报错
        """
        .detach() 不共享数据
        .cpu() 共享数据
        .numpy() 不共享数据
        """
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)    # 占位，对齐 'nx', 'ny', 'nz' 部分，本项目其实并没有直接用到法线
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        if self.use_MBRDF:
            kd = self.kd.detach().cpu().numpy()
            ks = self.ks.detach().cpu().numpy()
            alpha_asg = self.alpha_asg.detach().cpu().numpy()
            local_q = self.local_q.detach().cpu().numpy()
            neural_material = self.neural_material.detach().cpu().numpy()
        
        if self.use_hgs:
            hgs_normals = self.hgs_normals.detach().cpu().numpy()
            hgs_opacities = self.hgs_opacities.detach().cpu().numpy()

        # dtype_full 生成了一个 NumPy 结构化数据类型 (dtype)，其中每个字段都是 float32 ('f4')
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        # xyz.shape[0] 是 点的数量（即 N 个点），dtype=dtype_full 指定了 每个点的数据结构
        # elements: N个元素，每个元素是 dtype_full 数据结构
        elements = np.empty(xyz.shape[0], dtype=dtype_full)

        # 根据是否使用mbdrf，组合不同的数据，！！！与 construct_list_of_attributes(self) 函数中的存储顺序一致：
        if self.use_MBRDF:
            """
            np.concatenate: 将数据组合按 axis 拼接
            """
            if self.asg_alpha_num == 1:
                attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, \
                                            kd, ks, alpha_asg[:, :, 0], local_q, neural_material), axis=1)
            else:
                attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, \
                                            kd, ks, alpha_asg[:, :, 0], alpha_asg[:, :, 1], alpha_asg[:, :, 2], local_q, neural_material), axis=1)
        else:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        
        if self.use_hgs:
            attributes = np.concatenate((attributes, \
                                        hgs_normals, hgs_opacities), axis=1)
        # 创建 ply 数据:
        # map(tuple, attributes) 返回一个迭代器，它会 逐行 把 attributes 的数据转换为 tuple 。
        # 即每个 tuple 是一个 高斯点 的 所有属性
        """
        map(tuple, attributes): [tuple(row) for row in attributes]
        list(...): 把迭代器展开成一个 Python 列表，使其可以被 elements[:] 正确赋值，列表 赋 列表
        """ 
        # elements[i] 需要和 dtype_full 结构匹配，因此 attributes[i]/attributes每一行 的元素个数以及顺序必须 完全等于 dtype_full 里定义的字段数量
        # Numpy 自定义数据结构赋值要求使用 tuple ，这样 NumPy 才能正确解析数据格式
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex') 
        # 写入:
        PlyData([el]).write(path)

    # 将不透明度过大的点重置透明度为回归近乎透明，重新训练，避免某些部分效果不好但梯度消失。
    # 当不透明度 小于 0.005 时，会删除这些点，因此我们只用 设置 0.01 作为阈值
    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_or_add_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_local_q(self, temp):
        local_q_new = temp*0.6 + self.local_q*0.4
        optimizable_tensors = self.replace_or_add_tensor_to_optimizer(local_q_new, "local_q")
        self.local_q = optimizable_tensors["local_q"]

    """
    函数：
        加载已存在的 ply 文件，加载 可训练参数
        plydata.elements[0]: 是 一个 PlyElement 对象，通常表示 “vertex” 这一部分，也就是存储点云数据的部分。
        plydata.elements[0]["x"] 这样访问时，表示获取 所有点的 x 坐标数据（Numpy 数组）。
    返回：
        无返回，加载储存的数据，并以 self 的形式储存，
    """
    def load_ply(self, path):
        """
        Plydata: 
        plydata.elements: 代表文件中所有元素，每个元素是一个 PlyElement 对象，代表一个数据表
        plydata.elements[0]: 则代表 第一个数据表，即 我们的点云数据表
        plydata.elements[0].properties: 则代表 点云数据表 中的 所有属性
        虽然存储时是按行存储的，但读取时，对于每一个属性，会自动读取所有点的该属性，并返回一个 Numpy 数组
            例如: plydata.elements[0]["x"] 会返回所有点的 x 坐标数据
        """
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        
        if self.use_MBRDF:
            kd_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("kd_")]
            kd_names = sorted(kd_names, key = lambda x: int(x.split('_')[-1]))
            kd = np.zeros((xyz.shape[0], len(kd_names)))
            for idx, attr_name in enumerate(kd_names):
                kd[:, idx] = np.asarray(plydata.elements[0][attr_name])
                
            ks_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("ks_")]
            ks_names = sorted(ks_names, key = lambda x: int(x.split('_')[-1]))
            ks = np.zeros((xyz.shape[0], len(ks_names)))
            for idx, attr_name in enumerate(ks_names):
                ks[:, idx] = np.asarray(plydata.elements[0][attr_name])
                
            alpha_asg_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("alpha_asg_")]
            # 先按 Basis dim = 1 排序，再按 channel dim = 2 排序
            alpha_asg_names = sorted(alpha_asg_names, key = lambda x: (int(x.split('_')[-2]), int(x.split('_')[-1])))
            if self.alpha_change:
                asg_alpha_num = 3
            else:
                asg_alpha_num = self.asg_alpha_num
            assert len(alpha_asg_names) == self.basis_asg_num * asg_alpha_num,  (
                    f"alpha_asg_names 数量与 basis_asg_num 和 asg_alpha_num 不匹配: "
                    f"{len(alpha_asg_names)} != {self.basis_asg_num} * {asg_alpha_num}"
                )
            alpha_asg = np.zeros((xyz.shape[0], self.basis_asg_num, asg_alpha_num))
            for attr_name in alpha_asg_names:
                basis, channel = map(int, attr_name.split('_')[-2:])  # 解析通道索引 j 和 Basis 索引 i
                alpha_asg[:, basis, channel] = np.asarray(plydata.elements[0][attr_name])  # 赋值到正确位置

                
            local_q_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("local_q_")]
            local_q_names = sorted(local_q_names, key = lambda x: int(x.split('_')[-1]))
            local_q = np.zeros((xyz.shape[0], len(local_q_names)))
            for idx, attr_name in enumerate(local_q_names):
                local_q[:, idx] = np.asarray(plydata.elements[0][attr_name])
            
            neural_material_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("neural_material_")]
            neural_material_names = sorted(neural_material_names, key = lambda x: int(x.split('_')[-1]))
            neural_materials = np.zeros((xyz.shape[0], len(neural_material_names)))
            for idx, attr_name in enumerate(neural_material_names):
                neural_materials[:, idx] = np.asarray(plydata.elements[0][attr_name])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        if self.use_hgs:
            hgs_normals_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("hgs_normals_")]
            hgs_normals_names = sorted(hgs_normals_names, key = lambda x: int(x.split('_')[-1]))
            hgs_opacities_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("hgs_opacities_")]
            hgs_opacities_names = sorted(hgs_opacities_names, key = lambda x: int(x.split('_')[-1]))
            hgs_normals = np.zeros((xyz.shape[0], len(hgs_normals_names)))
            hgs_opacities = np.zeros((xyz.shape[0], len(hgs_opacities_names)))
            for idx, attr_name in enumerate(hgs_normals_names):
                hgs_normals[:, idx] = np.asarray(plydata.elements[0][attr_name])    # [N, 3]
            for idx, attr_name in enumerate(hgs_opacities_names):
                hgs_opacities[:, idx] = np.asarray(plydata.elements[0][attr_name])    # [N, 2]
            if hgs_normals_names == []:
                print("*"*50)
                print("hgs_attributes are empty, initate with one")
                hgs_normals = np.random.randn(xyz.shape[0], 3)
                hgs_opacities = np.random.randn(xyz.shape[0], 2)
                print("*"*50)
        
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        if self.use_MBRDF:
            self.kd = nn.Parameter(torch.tensor(kd, dtype=torch.float, device="cuda").requires_grad_(True))
            self.ks = nn.Parameter(torch.tensor(ks, dtype=torch.float, device="cuda").requires_grad_(True))
            self.alpha_asg = nn.Parameter(torch.tensor(alpha_asg, dtype=torch.float, device="cuda").requires_grad_(True))
            self.local_q = nn.Parameter(torch.tensor(local_q, dtype=torch.float, device="cuda").requires_grad_(True))
            self.neural_material = nn.Parameter(torch.tensor(neural_materials, dtype=torch.float, device="cuda").requires_grad_(True))
            
            self.asg_func = Mixture_of_ASG(self.basis_asg_num, self.asg_channel_num)
            self.neural_phasefunc = Neural_phase(hidden_feature_size=self.hidden_feature_size, \
                                                hidden_feature_layers=self.hidden_feature_layer, \
                                                frequency=self.phase_frequency, \
                                                neural_material_size=self.neural_material_size, \
                                                asg_mlp=self.asg_mlp, \
                                                mlp_zero=self.mlp_zero).to(device="cuda")
            
            
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        if self.use_hgs:
            self.hgs_normals = nn.Parameter(torch.tensor(hgs_normals, dtype=torch.float, device="cuda").requires_grad_(True))
            self.hgs_opacities = nn.Parameter(torch.tensor(hgs_opacities, dtype=torch.float, device="cuda").requires_grad_(True))
            # torch 实现，暂时没有用到，作为占位罢了
            self.small_gaussian = None


        # 建立 Gaussian 实例时决定，所以如果是中途保存的 ply，max_sh_degree 可能还需要额外记忆，其实应该也保存的，但这里没有考虑，可能训练比较快吧
        # 或者是 直接默认已启动最高阶的 SH 基函数，但是我就怕这个暂时还没保存，也许这个ply文件一定是训练后才保存的
        # 如果是训练后才保存的，就说的通了，因为渲染阶段，还需要 active_sh_degree 来选择需要的常数进行计算
        self.active_sh_degree = self.max_sh_degree

    # render 用，合并多个高斯点
    def merge_gaussians(self, gaussians, max_sh_degree = 3):
        if not gaussians:
            return
        
        num_new_coeffs = (max_sh_degree + 1) ** 2 - 1

        old_params = self._features_rest.clone()
        num_old_coeffs = old_params.shape[1]   
        self._features_rest = nn.Parameter(torch.zeros(old_params.shape[0], num_new_coeffs, old_params.shape[2], device="cuda", dtype=torch.float))
        self._features_rest[:, :num_old_coeffs, :] = old_params
        
        for gaussian in gaussians:
            old_params = gaussian._features_rest.clone()
            num_old_coeffs = old_params.shape[1] 
            gaussian._features_rest = nn.Parameter(torch.zeros(old_params.shape[0], num_new_coeffs, old_params.shape[2], device="cuda", dtype=torch.float))
            gaussian._features_rest[:, :num_old_coeffs, :] = old_params
            
            
            self._xyz = torch.cat((self._xyz, gaussian._xyz), dim=0)
            self._features_dc = torch.cat((self._features_dc, gaussian._features_dc), dim=0)
            self._features_rest = torch.cat((self._features_rest, gaussian._features_rest), dim=0)
            self._opacity = torch.cat((self._opacity, gaussian._opacity), dim=0)
            self._scaling = torch.cat((self._scaling, gaussian._scaling), dim=0)
            self._rotation = torch.cat((self._rotation, gaussian._rotation), dim=0)
            try:
                self.object_id = torch.cat((self.object_id, gaussian.object_id), dim=0)
                print("object_id is defined")
            except:
                print("object_id is not defined")
                print("object_id is not defined")
                print("object_id is not defined")

    """
    函数: 因为有时会改动一些 训练过程中的参数，因此需要重新 nn.Parameter 对象 并更新优化器使数据对齐
    返回: optimizable_tensors (dict): 更新后的参数字典，键为参数名称，值为新的 nn.Parameter
    """
    def replace_or_add_tensor_to_optimizer(self, tensor, name, lr=0.001):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1 or "asg" == group["name"][:3]:
                continue
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is None:
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    return
                else:
                    del self.optimizer.state[group['params'][0]]
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state
                    optimizable_tensors[group["name"]] = group["params"][0]
                    return optimizable_tensors
        # 如果参数组不存在，则添加参数组
        
        param = nn.Parameter(tensor.requires_grad_(True))
        self.optimizer.add_param_group({
            "params": [param],
            "lr": lr,
            "name": name
        })
        optimizable_tensors[name] = param
        print("\n optimizer add optimizable_tensors", optimizable_tensors)
        return optimizable_tensors
    

    """
    函数：通过 mask 过滤掉需要修剪的点，减少后续运算
    返回：直接更新参数和优化器，并通过 optimizable_tensors 字典，返回所有拼接后的 非通用asg 高斯点参数属性
        - asg 参数组不涉及修剪，所有高斯点共享，只用考虑数值变化
    """
    def _prune_optimizer(self, mask):
        optimizable_tensors = {}

        for group in self.optimizer.param_groups:
            # 跳过 asg 参数组（因为他们是高斯共用的，所以不涉及修剪），以及参数组数量大于1的参数组（实际上好像没有，也可能是在不是高斯点的参数部分）
            if len(group["params"]) > 1 or "asg" == group["name"][:3]:
                continue
            # 获取参数组状态,因为只有一个参数组，所以直接通过 group["params"][0] 获取
            # 获取参数组状态，如果状态不存在，则返回 None，说明还没经过后向传播，不用考虑 optimizer 的更新
            stored_state = self.optimizer.state.get(group["params"][0], None)
            
            # 如果状态存在，则更新状态
            if stored_state is not None:
                # 指数移动平均，一阶动量
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                # 平方指数移动平均，二阶动量
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                # 当你对参数进行修剪，即 group["params"][0][mask]
                # 优化器在之前为这个参数创建了状态，需要删除旧的状态，
                """
                为什么需要删除旧的状态？  
                    1. 因为在 state 字典中，字典的 key 是参数的地址，value 是参数的状态信息，
                    2. group["params"][0] = nn.Parameter((group... 这一步，group["params"][0] 的地址发生了变化，
                    2. 因此即便后续我们通过 self.optimizer.state[group['params'][0]] = stored_state 更新状态，会添加新的键值对，
                    3. 字典中依然会存在旧的状态，这会浪费内存，甚至有可能导致一些更严重的后果（比如说遍历状态集，或者做一些运算用到状态集）
                """
                del self.optimizer.state[group['params'][0]]
                # 更新参数，链接新地址
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                # 对齐/更新状态，添加键值对：新地址：新状态
                self.optimizer.state[group['params'][0]] = stored_state

                # 将更新后的参数组添加到 optimizable_tensors 字典中
                optimizable_tensors[group["name"]] = group["params"][0]

            # 如果状态不存在，则直接初始化
            # 优化器还没有为这个参数创建任何状态，因此不需要清理旧的状态条目，只需更新参数即可
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    """
    函数：更新修剪后的高斯点参数
    返回：无返回，直接更新 self 中的属性
    """
    def prune_points(self, mask):
        # mask 取反，得到需要保留的点，之前是需要修剪的为 True，现在需要保留的为 True。
        valid_points_mask = ~mask
        # 更新优化器中的参数，返回所有拼接后的 非通用asg 高斯点参数属性
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        # 更新参数
        for key, value in optimizable_tensors.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif hasattr(self, "_" + key):
                setattr(self, "_" + key, value)
            else:
                assert False, f"参数 {key} 不存在，请检查参数是否正确"
        

        # 更新2D梯度累加器以及权重累加器
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.out_weights_accum = self.out_weights_accum[valid_points_mask]

        # 更新 denom 和 max_radii2D，用于对齐后续的计算
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    """
    函数：这个函数的作用是将新的张量（来自 tensors_dict）与优化器中的现有参数进行拼接
         与之前那个函数不同，这个函数是拼接，而之前那个函数是完成对高斯点修剪后的属性更新
    返回：直接更新参数和优化器，并通过 optimizable_tensors 字典，返回所有拼接后的 非通用asg 高斯点参数属性
    """
    def cat_tensors_to_optimizer(self, tensors_dict):
        # tensors_dict 传入的新的高斯点参数，字典形式
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # 依然跳过 asg 参数组
            if len(group["params"]) > 1 or "asg" == group["name"][:3]:
                continue
            # assert len(group["params"]) == 1
            # 找到需要拼接的张量
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                
                # 将新的张量与优化器中的现有参数进行拼接，并更新优化器的状态
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                """
                torch.zeros_like(tensor)：生成和 tensor 形状、设备、数据类型一致的零张量
                torch.cat(tensor1, tensor2, dim=0):
                    1. 将 tensor1 和 tensor2 沿着第一个维度拼接，维度为 (N+M, ?)
                    2. dim=0 不定义也可以，默认从 dim=0 即第一个维度开始拼接
                """
                # 删除 state 中旧的键值对
                del self.optimizer.state[group['params'][0]]
                # 更新参数：新的张量与旧的张量拼接，形成新地址，并重新链接 group["params"][0]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                # 对齐/更新状态，添加新的键值对
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]

            # 如果状态不存在，则直接初始化，不用考虑前文中的旧状态处理，因为旧状态不存在  
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]


        return optimizable_tensors

    """
    函数：密集化后，修正场景中的高斯点，以及它们的属性
         利用上一个函数，将新生成的点和之前的高斯点拼接，并更新优化器中的参数
    返回：无返回，直接更新 self 中的属性
    """
    def densification_postfix(self, params):  
        
        # 构建新的参数字典，只包含非 None 的参数
        d = {key: value for key, value in params.items() if value is not None}

        # 新生成的高斯点和之前的高斯点拼接，直接更新参数和优化器：
        
        # 1. 拼接高斯点，重新构建可学习神经参数，并更新优化器
        # cat_tensors_to_optimizer(d) 函数拼接原高斯点和新添加的 d 高斯点，并且同时更新优化器状态，并且返回 optimizable_tensors 所有拼接后的 非通用asg 高斯点参数属性
        optimizable_tensors = self.cat_tensors_to_optimizer(d)

        # 2. 更新 gaussian 参数
        # 利用返回的 optimizable_tensors 更新类内部的属性
        for key, value in optimizable_tensors.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif hasattr(self, "_" + key):
                setattr(self, "_" + key, value)
            else:
                assert False, f"参数 {key} 不存在，请检查参数是否正确"



        # 重置2D梯度累加器以及权重累加器，用于对齐后续的计算，之前的计算已经用过修剪和密集化了，所以需要重置
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.out_weights_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    """
    函数：对高斯点进行密集化，并通过分裂高斯点的方式
         - 使用 densification_postfix 函数，将新生成的点和之前的高斯点拼接，并更新优化器中的参数
         - 使用 prune_points 函数，修剪掉不满足条件的高斯点
    返回：用于后续的 densify_and_prune 函数，并返回包括布尔索引，其中分裂前的点为 True，其他点为 False，包括新生成/分裂后的点
    """
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        
        """
        !!! 这里和 clone 部分不太一样，因为 clone 部分已经添加了新的高斯点，所以不能再直接 用 grads 进行筛选
        """
        # grads 是累积的2D梯度张量，grad_threshold 是2D梯度阈值，scene_extent 是场景范围，N 是分裂数量
        # 获取当前高斯点数量
        num_points = self.get_xyz.shape[0]
        # 初始化一个 tensor张量 用于后续的填充
        padded_grad = torch.zeros((num_points), device="cuda")
        """
        grads.squeeze()：
        （在 tensor 向量中，二维和一维的维度是不同的，二维是 (N, 1)，一维是 (N,)）
        1. 将 grads 张量的维度从 (N, 1) 压缩到 (N,)，即去掉维度为1的维度，变为一维张量，用于和其他一维张量进行拼接，对齐。
        2. 只适用于 grads 张量维度为 (N, 1) 的情况，如果 grads 维度为 (N, 2)，则不变
        """
        padded_grad[:grads.shape[0]] = grads.squeeze() # 其实直接 grads(:,0)也可以
        # 通过2D梯度阈值，筛选出大于2D梯度阈值的点，得到一个布尔索引
        """
        torch.where(condition, x, y)：
        1. 根据 condition 条件，选择 x 或 y 的值，如果 condition 为 True，则选择 x 的值，如果 condition 为 False，则选择 y 的值
        2. 返回一个与 condition 形状相同的张量，用于标记满足条件的点
        """
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        
        """
        torch.logical_and 是逻辑与运算，用于筛选出同时满足两个条件的点，true and true 为 true
        torch.max(self.get_scaling, dim=1) 返回一个两个属性的对象，values 是最大值，indices 是最大值的索引
            - dim=1 表示在第二个维度上取最大值
        """
        # 因为该函数是 split 分支，因此需要第二次筛选，判断是否过大
        # percent_dense*scene_extent 代表 高斯点 的尺寸阈值
        # torch.max(self.get_scaling, dim=1).values，得出每个高斯点在 xyz 方向上的最大缩放因子，随后和尺寸阈值进行比较
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        
        num_selected_pts = selected_pts_mask.sum()
        
        # 生成新的高斯点，计算新的高斯点属性
        # N=2 表示生成 2 个新的高斯点
        # 缩放因子维度为（splnum, 3），splnum 为需要分裂的高斯点数量    
        # 以缩放因子作为标准差，以正态分布生成偏差，维度为 (splnum*N, 3)
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)   # get_scaling：torch.exp(self._scaling)
        """
        tensor.repeat(N,1) 将张量沿着第一个维度复制 N 次，沿着第二个维度复制 1 次  
        """
        means =torch.zeros((stds.size(0), 3),device="cuda")     
        samples = torch.normal(mean=means, std=stds)
        """
        torch.normal 是生成正态分布的函数，mean 是均值，std 是标准差
            - torch.normal 中，mean 和 std 储存方式需要相同，维度可通过广播对齐
        """
        # 旋转矩阵维度为 (splnum, 3, 3)，进行复制对齐，维度为 (splnum*N, 3, 3)
        # 本文采用的 四元数，因此需要构建旋转矩阵
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)

        # 1）生成新的高斯点
        # 旋转矩阵与偏差相乘 + 分裂前高斯点中心 = 分裂后的高斯点中心
        # unsqueeze(-1) 在最后一个维度上增加一个维度，维度为 (splnum*N, 4, 1)，用于 bmm 计算的维度对齐
        # squeeze(-1) 删除最后一个维度，维度为 (splnum*N, 4)，和 分裂前高斯点中心维度对齐
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        """
        torch.bmm 是 Batch Matrix Multiplication，计算批次中的每对矩阵乘法
            - 专门用于 3D 张量，不支持高维张量，形状严格为 (batch_size, n, m) 和 (batch_size, m, p) 的张量
        torch.unsqueeze 用于在张量的指定维度上增加一个维度，通常用于在指定位置，比如在最后一个维度上增加一个维度
        torch.squeeze 可指定维度，默认删除所有维度为 1 的维度
        """

        # 2）生成新的高斯点属性
        # 将缩放因子除以 0.8*N，考虑分裂适当缩小，并反激活，得到原缩放因子
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        # 剩下属性皆直接继承分裂前点的属性，旋转矩阵，基础颜色，其他效果，漫反射系数，镜面反射系数，alpha，asg系数，局部旋转矩阵，神经材质，透明度
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_kd = None if not self.use_MBRDF else self.kd[selected_pts_mask].repeat(N,1)
        new_ks = None if not self.use_MBRDF else self.ks[selected_pts_mask].repeat(N,1)
        new_alpha_asg = None if not self.use_MBRDF else self.alpha_asg[selected_pts_mask].repeat(N,1,1)
        new_local_q = None if not self.use_MBRDF else self.local_q[selected_pts_mask].repeat(N,1)
        new_neural_material = None if not self.use_MBRDF else self.get_neural_material[selected_pts_mask].repeat(N,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        
        new_hgs_normals = None if not self.use_hgs else self.hgs_normals[selected_pts_mask].repeat(N,1)
        new_hgs_opacities = None if not self.use_hgs else self.hgs_opacities[selected_pts_mask].repeat(N,1)

        
        params = {"xyz": new_xyz, "features_dc": new_features_dc, "features_rest": new_features_rest, "opacity": new_opacity, "scaling": new_scaling, "rotation": new_rotation, \
            "kd": new_kd, "ks": new_ks, "alpha_asg": new_alpha_asg, "local_q": new_local_q, "neural_material": new_neural_material,
            "hgs_normals": new_hgs_normals, "hgs_opacities": new_hgs_opacities}
        
        # 更新高斯状态，为了模块化，分为了两步，第一步就是直接拼接模块，不考虑其他删减，第二是，在通过删减模块修正场景中的高斯点
        # 1）densification_postfix 先将分裂后生成的高斯点与原高斯点拼接，并更新模型参数和优化器状态
        # 此时尚未删除被分裂的点，因此还有后续一个处理
        """
        你可能会觉得 densification_postfix函数多余处理优化器状态，其实这个函数 clone 也用，所以为了简单，就写在一起了。
        """
        self.densification_postfix(params)

        # 2）prune_points 删除分裂前的点，并更新参数和优化器状态
        # 参数和优化器已经添加了新生成的点，所以需要再删除分裂前的点的时候，要通过拼接 N * selected_pts_mask.sum() 来对齐维度。
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
        
        return prune_filter, num_selected_pts
    
    """
    函数：对高斯点进行密集化，并通过直接复制高斯点的方式
         - 使用 densification_postfix 函数，将新生成的点和之前的高斯点拼接，并更新优化器中的参数
    返回：无返回，用于后续的 densify_and_prune 函数
         
    """
    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient coendition
        # 求 x 和 y 方向的梯度，并求其范数，即最终梯度
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        # 二次筛选，判断是否过大，这里要的是过小的，过大去分裂
        selected_pts_mask = torch.logical_and(selected_pts_mask,      # percent_dense 默认0.01
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        num_selected_pts = selected_pts_mask.sum()
        
        # 直接继承高斯点属性
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacity = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_kd = None if not self.use_MBRDF else self.kd[selected_pts_mask]
        new_ks = None if not self.use_MBRDF else self.ks[selected_pts_mask]
        new_alpha_asg = None if not self.use_MBRDF else self.alpha_asg[selected_pts_mask]
        new_local_q = None if not self.use_MBRDF else self.local_q[selected_pts_mask]
        new_neural_material = None if not self.use_MBRDF else self.get_neural_material[selected_pts_mask]

        #new_opacities = inverse_sigmoid(
        #    1 - torch.sqrt(1 - torch.sigmoid(new_opacities)))  # self._opacity[selected_pts_mask])
        #self._opacity[selected_pts_mask].copy_(new_opacities.detach())
        new_hgs_normals = None if not self.use_hgs else self.hgs_normals[selected_pts_mask]
        new_hgs_opacities = None if not self.use_hgs else self.hgs_opacities[selected_pts_mask]

        params = {"xyz": new_xyz, "features_dc": new_features_dc, "features_rest": new_features_rest, "opacity": new_opacity, "scaling": new_scaling, "rotation": new_rotation, \
            "kd": new_kd, "ks": new_ks, "alpha_asg": new_alpha_asg, "local_q": new_local_q, "neural_material": new_neural_material,
            "hgs_normals": new_hgs_normals, "hgs_opacities": new_hgs_opacities}
        
        # 将新生成的高斯点与原高斯点拼接，并更新模型参数和优化器状态（分裂中是先添加两组分裂点，然后删去原点，这里直接添加一组相同属性的点作为复制）
        self.densification_postfix(params)
        
        return num_selected_pts

    """
    函数：真正的密集化函数与修剪函数
         - 中间调用 densify_and_clone 和 densify_and_split 函数用于密集化
         - 三重修剪：透明度，尺寸，权重，最终调用 prune_points 函数用于修剪
    返回：无返回，直接更新 self 中的属性
    """
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, bigsize_threshold, crop_extent):
        # 计算各点归一化后的累积2D梯度，即除以各自累积的次数
        grads = self.xyz_gradient_accum / self.denom

        # 将 grads 中 NaN 值变为 0
        grads[grads.isnan()] = 0.0


        """
        .isnan() 作为布尔操作，根据值是否为数字（NaN），返回一个与 grads 形状相同的布尔变量。
        # 变量使用布尔索引 用于 给令一个变量赋值时，是作为副本
        # 但直接，比如此处，结合布尔索引进行赋值时，是一种视图
        """
        # 获得权重累加器，在后续的 densify 过程中，权重会被和其他一些判断信息一起初始化，所以先拿出来，准备后面修剪使用
        out_weights_acc = self.out_weights_accum
       

        # 调用 densify_and_clone 和 densify_and_split 函数
        num_clone = self.densify_and_clone(grads, max_grad, extent)
        _prune_filter, num_split = self.densify_and_split(grads, max_grad, extent)
        
        # 更新权重累加器
        # 先复制，再分裂，在 _prune_filter 中，新加入的点都在后面，且除了需要删除的点为 True，其他点为 False
        # ~ 是取反操作，True 变为 False，False 变为 True
        # 通过 ~_prune_filter 保留应该留下的点的权重，但是 out_weights_acc 的维度尚未变化，所以需要先对齐
        out_weights_acc = out_weights_acc[~_prune_filter[:out_weights_acc.shape[0]]]
        # 初始化 padded_out_weights 张量，即新的权重累加器张量，维度为 (self.get_xyz.shape[0],)
        padded_out_weights = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        # 将原本的权重累加器张量赋值给 padded_out_weights 张量，
        padded_out_weights[:out_weights_acc.shape[0]] = out_weights_acc.squeeze()   
        # 将新加入的点权重赋值给 padded_out_weights 张量，统一赋值为原本权重累加器中的最大权重
        padded_out_weights[out_weights_acc.shape[0]:] = torch.max(out_weights_acc)

        # 1）透明度修剪掩码
        # 计算透明度过低所导致的修剪掩码，将所有点的透明度与最小透明度进行比较，小于最小透明度的点为 True，否则为 False
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        
        # 2）尺寸修剪掩码
        # 存在一个迭代范围，如果迭代次数大于该范围，则进行尺寸阈值的筛选，之前不需要，即 max_screen_size 为 None
        # 根据高斯的最大半径以及缩放比例进行筛选，此处是默认 percent_dense 为 0.1
        # 在 3dgs 早期阶段，模型主要用于调整高斯点位置和分布，后期趋于稳定，则需要进行尺寸的控制和剔除过大的高斯点，加速渲染
        if max_screen_size:
            if self.rasterizer == "3dgs":
                big_points_ws = self.get_scaling.max(dim=1).values > bigsize_threshold * extent  # revised in 5.12
                big_points_vs = self.max_radii2D > max_screen_size
            else:
                big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent  
                big_points_vs = self.max_radii2D > max_screen_size
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        
        if crop_extent is not None:
           # 筛选出所有坐标都在 [-extent, extent] 范围内的点
            mask = torch.all(torch.abs(self.get_xyz) > (extent + crop_extent), dim=1)
            prune_mask = torch.logical_or(prune_mask, mask)
            crop_extent = None

        # 3）权重修剪掩码
        out_weights_mask = self.prune_visibility_mask(padded_out_weights)

        # 混合修剪掩码，得到最终的修剪掩码
        prune_mask = torch.logical_or(prune_mask, out_weights_mask)
        """
        torch.logical_or 是逻辑或运算，用于筛选出满足至少一个条件的点，有 true 为 true
        """
        self.global_call_counter += 1
        # 根据最终的修剪掩码，修剪高斯点
        n_before = self.get_xyz.shape[0]
        self.prune_points(prune_mask)
        n_after = self.get_xyz.shape[0]
        if self.global_call_counter % 20 == 0:  
            print("densify_and_prune: ", n_before - n_after)
            print("current: ", n_after)
            self.global_call_counter = 0
        # 清空缓存
        # 这个密集化以及修剪过程中，产生了大量中间蟑螂，因此需要清空缓存
        torch.cuda.empty_cache()
        return num_clone, num_split
    
    """
    函数：记录高斯密集化统计信息
         - 用于记录训练过程中，每个高斯点的累积2D梯度，权重
         - 并通过高斯点的可见性，减少计算量
    """
    def add_densification_stats(self, viewspace_point_tensor, update_filter, width, height, out_weights):
        # 这里的梯度是 2D 梯度，即每一个视角高斯球映射后的椭圆的 x,y 方向的梯度 
        grad = viewspace_point_tensor.grad.squeeze(0) # [N, 2]

        # gsplat 修正
        """
        ① gsplat 中，means2D 的梯度是关于 normalized image plane(x/z, y/z)(透视除法后但尚未映射到像素坐标之前的，即 NDC 前两维) 的梯度，
        ② 3dgs 梯度是基于 像素空间的即 (x,y)
            - 因此需要根据 x 和 y 轴的 焦距 进行修正，但一般 fov = 90，所以近似认为 x 和 y 轴的焦距等于 width 和 height 的一半
        """
        if self.rasterizer == "gsplat":
            # 根据图片尺寸修正梯度
            grad[:, 0] *= width * 0.5
            grad[:, 1] *= height * 0.5

        elif self.rasterizer == "3dgs":
            # 根据图片尺寸修正梯度
            grad[:, 0] *= 1
            grad[:, 1] *= 1

        # 将梯度累加到累积梯度张量中，并更新累积次数，用于后续修建中的归一化
        self.xyz_gradient_accum[update_filter] += torch.norm(grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        """
        denom 只用于 2D梯度 的归一化，不用于权重归一化，
        原因：
            - 个人认为，每个视角中，并不是所有高斯点都参与，有的视角下高斯点动荡，有的视角下反而不可见，而有些高斯点可能动荡不大，但一直可见
            所以最终累积 2D梯度 可能一样大，显然这是不公平的，所以需要考虑高斯点参与的视角数量，除以累积次数归一化。
            - 而 权重累加器 看中的是整体，每个高斯点对整个场景的贡献，因此不需要归一化
        """
        # 将权重累加到累积权重张量中
        self.out_weights_accum += out_weights
    
    """
    函数：传入权重累加器，决定修剪哪些高斯点
    返回：n * 1 的张量，n 为高斯点数量，1 为 bool 类型，用于标记需要修剪的高斯点
    """    
    def prune_visibility_mask(self, out_weights_acc):
        n_before = self.get_xyz.shape[0]    # 当前高斯点数量，可能由于密集化产生变化
        n_after = self.maximum_gs   # 最大高斯点数量，由自己设置
        # 计算修剪数量
        n_prune = n_before - n_after
        # 创建一个与高斯点数量相同的布尔掩码，用于标记需要修剪的高斯点，并确保 prune_mask 张量与 _xyz 张量在储存方式上一致
        prune_mask = torch.zeros((self.get_xyz.shape[0],), dtype=torch.bool, device=self.get_xyz.device)

        # 决定是否修剪
        if n_prune > 0:
            # Find the mask of top n_prune smallest `self.out_weights_accum`
            # 找到权重值最小的 n_prune 个高斯点
            _, indices = torch.topk(out_weights_acc, n_prune, largest=False)
            # 权重值最小的 n_prune 的高斯点序号在布尔掩码张量中标记为 True，表示需要修剪
            prune_mask[indices] = True
            print("\nprune_visibility_mask: ", n_prune)
        return prune_mask
