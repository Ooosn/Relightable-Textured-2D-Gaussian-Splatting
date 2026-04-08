from typing import NamedTuple

import torch
import torch.nn as nn

from . import _C


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool
    low_pass_filter_radius: float = 0.3
    ortho: bool = False


def rasterize_gaussians(
    means3D,
    means2D,
    shs,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    texture_alpha,
    texture_sigma_factor,
    non_trans,
    offset,
    thres,
    is_train,
    raster_settings,
):
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
        shs,
        raster_settings.sh_degree,
        raster_settings.campos,
        texture_alpha,
        texture_sigma_factor,
        raster_settings.prefiltered,
        raster_settings.debug,
        non_trans,
        offset,
        thres,
        is_train,
    )

    if raster_settings.debug:
        cpu_args = cpu_deep_copy_tuple(args)
        try:
            num_rendered, color, out_weight, radii, out_trans, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
        except Exception as ex:
            torch.save(cpu_args, "snapshot_fw_shadow.dump")
            print("\nAn error occured in forward. Please forward snapshot_fw_shadow.dump for debugging.")
            raise ex
    else:
        num_rendered, color, out_weight, radii, out_trans, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

    _ = (num_rendered, geomBuffer, binningBuffer, imgBuffer)
    invdepths = None

    P = means3D.shape[0]
    # Reshape UV-indexed shadow buffers to [P, res, res] when texture was provided.
    tex_res = 0
    if texture_alpha is not None and texture_alpha.numel() > 0:
        tex_res = int(texture_alpha.shape[2])
    if tex_res > 0:
        out_trans = out_trans.view(P, tex_res, tex_res)
        non_trans = non_trans.view(P, tex_res, tex_res)

    return color, out_weight, radii, out_trans, non_trans, invdepths


def rasterize_color(
    means3D,
    means2D,
    shs,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    texture_color,
    texture_alpha,
    texture_shadow,
    texture_sigma_factor,
    raster_settings,
):
    """View-space color render with optional UV-level shadow modulation.

    ``texture_shadow`` should be the flat [N*R*R] (or [N,R,R] reshaped) tensor
    produced by rasterize_gaussians / shadow pass, already normalised to
    [0, 1] (i.e. out_trans / non_trans).  Pass an empty tensor to disable.

    Returns rendered image [C, H, W].
    """
    if texture_color is None:
        texture_color = torch.empty(0, device=means3D.device, dtype=means3D.dtype)
    if texture_alpha is None:
        texture_alpha = torch.empty(0, device=means3D.device, dtype=means3D.dtype)
    if texture_shadow is None:
        texture_shadow = torch.empty(0, device=means3D.device, dtype=means3D.dtype)
    # Flatten [N,R,R] → [N*R*R] if needed
    if texture_shadow.ndim == 3:
        texture_shadow = texture_shadow.reshape(-1)
    if shs is None:
        shs = torch.empty(0, device=means3D.device, dtype=means3D.dtype)
    if colors_precomp is None:
        colors_precomp = torch.empty(0, device=means3D.device, dtype=means3D.dtype)
    if scales is None:
        scales = torch.empty(0, device=means3D.device, dtype=means3D.dtype)
    if rotations is None:
        rotations = torch.empty(0, device=means3D.device, dtype=means3D.dtype)
    if cov3Ds_precomp is None:
        cov3Ds_precomp = torch.empty(0, device=means3D.device, dtype=means3D.dtype)

    return _C.rasterize_color(
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
        shs,
        raster_settings.sh_degree,
        raster_settings.campos,
        texture_color,
        texture_alpha,
        texture_shadow,
        texture_sigma_factor,
        raster_settings.prefiltered,
        raster_settings.debug,
    )


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
            )
        return visible

    def forward(
        self,
        means3D,
        means2D,
        opacities,
        shs=None,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3Ds_precomp=None,
        texture_alpha=None,
        texture_sigma_factor=3.0,
        non_trans=None,
        offset=0.015,
        thres=4.0,
        is_train=False,
        streams=None,
    ):
        del means2D, streams

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception("Please provide exactly one of either SHs or precomputed colors!")

        if ((scales is None or rotations is None) and cov3Ds_precomp is None) or (
            (scales is not None or rotations is not None) and cov3Ds_precomp is not None
        ):
            raise Exception("Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!")

        if shs is None:
            shs = torch.empty(0, device=means3D.device, dtype=means3D.dtype)
        if colors_precomp is None:
            colors_precomp = torch.empty(0, device=means3D.device, dtype=means3D.dtype)
        if scales is None:
            scales = torch.empty(0, device=means3D.device, dtype=means3D.dtype)
        if rotations is None:
            rotations = torch.empty(0, device=means3D.device, dtype=means3D.dtype)
        if cov3Ds_precomp is None:
            cov3Ds_precomp = torch.empty(0, device=means3D.device, dtype=means3D.dtype)
        if texture_alpha is None:
            texture_alpha = torch.empty(0, device=means3D.device, dtype=means3D.dtype)
        if non_trans is None:
            # Allocate per-UV-texel buffer when texture is provided; otherwise per-Gaussian.
            tex_res = int(texture_alpha.shape[2]) if texture_alpha.numel() > 0 else 0
            shadow_size = means3D.shape[0] * tex_res * tex_res if tex_res > 0 else means3D.shape[0]
            non_trans = torch.zeros(shadow_size, device=means3D.device, dtype=means3D.dtype)

        return rasterize_gaussians(
            means3D,
            None,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3Ds_precomp,
            texture_alpha,
            texture_sigma_factor,
            non_trans,
            offset,
            thres,
            is_train,
            self.raster_settings,
        )
