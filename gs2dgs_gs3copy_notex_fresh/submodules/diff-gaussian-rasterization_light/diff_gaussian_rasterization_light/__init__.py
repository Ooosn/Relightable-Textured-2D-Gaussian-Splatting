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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, **kwargs):
        
        raster_settings = self.raster_settings
        kwargs["raster_settings"] = raster_settings


        shs = kwargs.get("shs", None)
        colors_precomp = kwargs.get("colors_precomp", None)
        means3D = kwargs.get("means3D", None)
        scales = kwargs.get("scales", None)
        rotations = kwargs.get("rotations", None)
        cov3Ds_precomp = kwargs.get("cov3Ds_precomp", None)
        hgs_normals = kwargs.get("hgs_normals", None)
        hgs_opacities = kwargs.get("hgs_opacities", None)
        hgs_opacities_shadow = kwargs.get("hgs_opacities_shadow", None)
        hgs_opacities_light = kwargs.get("hgs_opacities_light", None)


        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3Ds_precomp is None) or ((scales is not None or rotations is not None) and cov3Ds_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        if colors_precomp is not None and colors_precomp.numel() != 0:
            if colors_precomp.ndim != 2 or colors_precomp.size(0) != means3D.size(0) or colors_precomp.size(1) != 3:
                raise ValueError(
                    "colors_precomp must have shape (num_points, 3) for the light rasterizer, "
                    f"but got {tuple(colors_precomp.shape)} while means3D has shape {tuple(means3D.shape)}"
                )
        
        if shs is None:
            kwargs["shs"] = torch.Tensor([])
        if colors_precomp is None:
            kwargs["colors_precomp"] = torch.Tensor([])
        if scales is None:
            kwargs["scales"] = torch.Tensor([])
        if rotations is None:
            kwargs["rotations"] = torch.Tensor([])
        if cov3Ds_precomp is None:
            kwargs["cov3Ds_precomp"] = torch.Tensor([])
        if hgs_normals is None:
            kwargs["hgs_normals"] = torch.Tensor([])
        if hgs_opacities is None:
            kwargs["hgs_opacities"] = torch.Tensor([])
        if hgs_opacities_shadow is None:
            kwargs["hgs_opacities_shadow"] = torch.Tensor([])
        if hgs_opacities_light is None:
            kwargs["hgs_opacities_light"] = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(**kwargs)

def rasterize_gaussians(**kwargs):
    # apply 不支持 关键字参数，因此需要手动指定参数顺序
    # 顺便调整参数顺序，方便调试
    kwargs["projmatrix"] = kwargs["raster_settings"].projmatrix
    kwargs["viewmatrix"] = kwargs["raster_settings"].viewmatrix
    
    param_order = [
        # 高斯点相关
        "means3D", "means2D", "shs", "colors_precomp", "opacities",
        "scales", "rotations", "cov3Ds_precomp", 
        
        # 视角相关
        "projmatrix",
        "viewmatrix",
        
        # 配置相关
        "raster_settings",

        # 阴影相关
        "non_trans", "offset", "thres", 
        
        # prune 相关
        "is_train",

        # hgs 相关
        "hgs", "hgs_normals", "hgs_opacities",
        "hgs_opacities_shadow", "hgs_opacities_light",

        # 流
        "streams"
        

    ]
    args = [kwargs[k] for k in param_order]
    return _RasterizeGaussians.apply(*args)

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        shs,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        projmatrix,
        viewmatrix,
        
        raster_settings,

        # 阴影相关
        non_trans,
        offset,
        thres,

        # prune 相关
        is_train,

        # hgs 相关
        hgs,
        hgs_normals,
        hgs_opacities,
        hgs_opacities_shadow,
        hgs_opacities_light,

        # 流
        streams
    ):
        
        # Restructure arguments the way that the C++ lib expects them
        args = (
            # 高斯点/场景相关
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            viewmatrix,
            projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            shs,
            raster_settings.sh_degree,
            # 相机位置，在向光源泼溅中，则为光源位置
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
            raster_settings.low_pass_filter_radius,
            
            raster_settings.ortho,
            # opacity_light = torch.zeros(scales.shape[0], dtype=torch.float32, device=scales.device)
            # non_trans <- opacity_light，用于每个高斯阴影的标准化参数
            non_trans,  
            offset,
            thres,
            # is_train = prune_visibility: 
            is_train,

            # hgs 相关
            hgs,
            hgs_normals,
            hgs_opacities,
            hgs_opacities_shadow,
            hgs_opacities_light,

            # 流
            #streams
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, out_weight, radii, out_trans, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, out_weight, radii, out_trans, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        # 保存 非 tensor 参数
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.color = color.detach()
        # atomicAdd(&(non_trans[collected_id[j]]), exp(power_j))
        ctx.non_trans=non_trans.detach()
        # atomicAdd(&(out_trans[collected_id[j]]), exp(power_j)*(T));
        ctx.out_trans=out_trans.detach()
        ctx.offset=offset

        """
        在 autograd.Function.forward 中，使用 PyTorch 的操作（如 +, *, sin 等）仍然会构建计算图。
        如果你保留了某个 Tensor（没有 .detach()），它就会成为计算图的一部分，占用显存和内存。

        ctx.save_for_backward(...) 的作用：
            - 会自动对传入 Tensor 调用 .detach()，只保存数值副本，不保计算图。
            - 保存的中间变量可以在 backward 中安全使用。
            - 可以防止中间变量继续参与 autograd 图，从而节省显存、避免循环引用。

        如果你不使用 save_for_backward 而直接保留中间变量（比如 ctx.foo = some_tensor），
        那么计算图可能会一直保留在这些变量里，造成内存无法释放。
        """
        # 相当于使用 .detach() ，只保存 tensor 的值，不会存储梯度，计算图等
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, \
                              radii, shs, opacities, geomBuffer, binningBuffer, imgBuffer,\
                              projmatrix, viewmatrix)
        
        invdepths = None 
        # 返回 渲染结果，torch 会将他们作为计算图中的下一个节点
        # 这些梯度 会按照顺序 传入 backward 方法
        return color, out_weight, radii, out_trans, non_trans, invdepths

    @staticmethod
    def backward(ctx, grad_out_color, _, __, grad_out_trans, grad_non_trans, grad_out_depth):

        # Restore necessary values from context
        if grad_out_color is None:
            grad_out_color = torch.zeros_like(ctx.color)
        if grad_out_trans is None:
            grad_out_trans = torch.zeros_like(ctx.out_trans)
        if grad_non_trans is None:
            grad_non_trans = torch.zeros_like(ctx.non_trans)
        if grad_out_depth is None:
            grad_out_depth = torch.empty(0, device=ctx.color.device, dtype=ctx.color.dtype)
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        non_trans=ctx.non_trans
        out_trans=ctx.out_trans
        offset=ctx.offset
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer, projmatrix, viewmatrix= ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                opacities,
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                viewmatrix, 
                projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                
                grad_out_color, 
                grad_out_depth,
                
                out_trans,
                non_trans,
                grad_out_trans,
                grad_non_trans,
                
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug,
                raster_settings.low_pass_filter_radius,
                raster_settings.image_height,
                raster_settings.image_width,
                
                raster_settings.ortho,
                offset)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_projmatrix, grad_viewmatrix = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_projmatrix, grad_viewmatrix = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            grad_projmatrix,
            grad_viewmatrix,
            None,

            # 阴影相关
            None,
            None,
            None,

            # prune 相关
            None,

            # hgs 相关
            None,
            None,
            None,
            None,
            None,

            # 流
            None
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    low_pass_filter_radius : float
    ortho : bool

