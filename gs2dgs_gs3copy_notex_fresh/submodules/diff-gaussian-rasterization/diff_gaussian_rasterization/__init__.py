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
    param_order = [
        # 高斯点相关
        "means3D", "means2D", "shs", "colors_precomp", "opacities",
        "scales", "rotations", "cov3Ds_precomp", "raster_settings",

        # hgs 相关
        "hgs", "hgs_normals", "hgs_opacities"
    ]
    args = [kwargs[k] for k in param_order]
    return _RasterizeGaussians.apply(*args)

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,

        # hgs 相关
        hgs,
        hgs_normals,
        hgs_opacities
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
            raster_settings.low_pass_filter_radius,
            hgs,
            hgs_normals,
            hgs_opacities
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, radii, depth, alpha, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, radii, depth, alpha, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        return color, radii, depth, alpha

    @staticmethod
    def backward(ctx, grad_out_color, _, __, ___):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D,
                radii,
                colors_precomp,
                scales,
                rotations,
                raster_settings.scale_modifier,
                cov3Ds_precomp,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
                raster_settings.tanfovx,
                raster_settings.tanfovy,
                grad_out_color,
                sh,
                raster_settings.sh_degree,
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,

            None,
            None,
            None
        )
        return grads


# ---------------------------------------------------------------------------
# Differentiable camera-matrix wrapper
# ---------------------------------------------------------------------------
# The CUDA rasterizer treats viewmatrix / projmatrix as constants (they live
# inside the NamedTuple *raster_settings*).  This thin wrapper re-introduces
# them into the autograd graph so that cam_pose_adj (and any other parameter
# that feeds into the matrices) receives proper gradients through the
# projection path – matching what gsplat does natively via
# fully_fused_projection.
#
# Usage (drop-in replacement in the render function):
#   color, radii, depth, alpha = rasterizer(...)          # as before
#   color = CamGradBridge.apply(color, viewmatrix,
#                               projmatrix, means3D, means2D)
# ---------------------------------------------------------------------------

class CamGradBridge(torch.autograd.Function):
    """Zero-cost forward; backward injects viewmatrix gradient via mean2D."""

    @staticmethod
    def forward(ctx, rendered_image, viewmatrix, projmatrix, means3D, means2D_grad_holder):
        # rendered_image passes through unchanged
        ctx.save_for_backward(viewmatrix, projmatrix, means3D, means2D_grad_holder)
        return rendered_image

    @staticmethod
    def backward(ctx, grad_rendered_image):
        viewmatrix, projmatrix, means3D, means2D_grad_holder = ctx.saved_tensors

        # We need to propagate the CUDA-computed dL/d(mean2D) through a
        # differentiable projection to obtain dL/d(viewmatrix).
        #
        # means2D_grad_holder.grad is filled by the inner _RasterizeGaussians
        # backward (which ran first, since rendered_image depends on it).
        # It contains dL/d(screen_xy) with the 0.5*W factor baked in.

        grad_viewmatrix = None
        grad_projmatrix = None

        if viewmatrix.requires_grad and means2D_grad_holder.grad is not None:
            dL_dmean2D = means2D_grad_holder.grad[:, :2].detach()  # [N, 2]

            # Differentiable projection: world -> clip -> NDC -> pixel
            # viewmatrix is row-major [4,4] (already transposed in camera.py)
            means_h = torch.cat([means3D.detach(),
                                 torch.ones_like(means3D[:, :1])], dim=-1)  # [N, 4]
            # clip = means_h @ projmatrix  (row-vector convention)
            clip = means_h @ projmatrix           # [N, 4]
            w = clip[:, 3:4].clamp(min=1e-7)      # [N, 1]
            ndc_xy = clip[:, :2] / w               # [N, 2]
            H = viewmatrix.shape[0]  # just need image dims from raster_settings
            # ndc2Pix: pixel = ((ndc + 1) * S - 1) * 0.5
            # d(pixel)/d(ndc) = S * 0.5  (already baked into dL_dmean2D)
            # So dL_dmean2D is effectively dL/d(ndc_xy) when we account
            # for the 0.5*W factor the CUDA backward applied.

            # Compute scalar proxy = sum(ndc_xy * dL_dmean2D)
            # backward of this gives dL/d(projmatrix) and dL/d(viewmatrix)
            # through the chain: projmatrix depends on viewmatrix in
            # full_proj_transform = world_view_transform @ projection_matrix
            # But here projmatrix is the *full* proj transform, so its
            # gradient directly gives us what we need.
            proxy = (ndc_xy * dL_dmean2D).sum()
            # We only want gradients for viewmatrix/projmatrix, not means3D
            # proxy.backward() would work but we need to be careful with graphs.
            # Use torch.autograd.grad for targeted computation.
            grads_out = torch.autograd.grad(
                proxy, [v for v in [viewmatrix, projmatrix] if v.requires_grad],
                retain_graph=False, allow_unused=True
            )
            idx = 0
            if viewmatrix.requires_grad:
                grad_viewmatrix = grads_out[idx]; idx += 1
            if projmatrix.requires_grad:
                grad_projmatrix = grads_out[idx]; idx += 1

        return grad_rendered_image, grad_viewmatrix, grad_projmatrix, None, None


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
