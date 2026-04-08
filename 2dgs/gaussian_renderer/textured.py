import math
import torch
from typing import NamedTuple

from surfel_texture import GaussianRasterizationSettings
from surfel_texture import GaussianRasterizer
from surfel_texture_deferred import GaussianRasterizer as GaussianRasterizer_deferred


class TextureRenderInputs(NamedTuple):
    mbrdf: dict
    texture_color: torch.Tensor
    enable_texture: bool


def _build_raster_settings(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, enable_texture):
    """Build GaussianRasterizationSettings + covariance tensors (shared by all paths)."""
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=getattr(pipe, "debug", False),
        enable_texture=enable_texture
    )
    scales, rotations, cov3D_precomp = None, None, None
    if pipe.compute_cov3D_python:
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W - 1) / 2],
            [0, H / 2, 0, (H - 1) / 2],
            [0, 0, far - near, near],
            [0, 0, 0, 1],
        ]).float().cuda().T
        splat2world = pc.get_covariance(scaling_modifier)
        cov3D_precomp = (
            splat2world[:, [0, 1, 3]] @ (viewpoint_camera.full_proj_transform @ ndc2pix)[:, [0, 1, 3]]
        ).permute(0, 2, 1).reshape(-1, 9)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    return raster_settings, scales, rotations, cov3D_precomp


def rasterize_with_texture_module(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, inputs):
    raster_settings, scales, rotations, cov3D_precomp = _build_raster_settings(viewpoint_camera, pc, pipe, bg_color, 
                                                                               scaling_modifier, inputs.enable_texture)
    mbrdf = inputs.mbrdf      
    deferred = inputs.deferred 
    colors_precomp = inputs.colors_precomp
    means2D = torch.zeros((pc.get_xyz.shape[0], 2), dtype=torch.float32, device=pc.get_xyz.device)  
    shs = None
           
    if not deferred:
        mbrdf_colors = mbrdf["basecolor"] * mbrdf["shadow"] + mbrdf["other_effects"] if mbrdf is not None else None
        # original 2dgs rasterizer
        if pc.use_textures:
            texture_color = mbrdf_colors if mbrdf_colors is not None else pc.get_texture_color
        else:
            texture_color = None
            shs = pc.get_features if colors_precomp is None else None

        rendered_image, radii, allmap = GaussianRasterizer(raster_settings)(means3D=pc.get_xyz, means2D=means2D,
                                                                                    shs=shs, colors_precomp=colors_precomp,
                                                                                    opacities=pc.get_opacity,
                                                                                    scales=scales, rotations=rotations, cov3D_precomp=cov3D_precomp,
                                                                                    texture_color=texture_color, texture_alpha=pc.get_texture_alpha,
                                                                                    )
    else:
        mbrdf_colors = torch.concat([mbrdf["basecolor"], mbrdf["shadow"], mbrdf["other_effects"]], dim=1) if mbrdf is not None else None
        # original 2dgs rasterizer
        if pc.use_textures:
            texture_color = mbrdf_colors if mbrdf_colors is not None else pc.get_texture_color
        else:
            texture_color = None
            shs = None
            if colors_precomp is None:
                assert False, "sh cannot be calculated in deferred mode when colors_precomp is None"

        rendered_image, radii, allmap = GaussianRasterizer_deferred(raster_settings)(means3D=pc.get_xyz, means2D=means2D,
                                                                                    shs=shs, colors_precomp=colors_precomp,
                                                                                    opacities=pc.get_opacity,
                                                                                    scales=scales, rotations=rotations, cov3D_precomp=cov3D_precomp,
                                                                                    texture_color=texture_color, texture_alpha=pc.get_texture_alpha,
                                                                                    )

    return rendered_image, radii, allmap, means2D