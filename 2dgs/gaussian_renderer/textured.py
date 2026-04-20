import math
import torch
from typing import NamedTuple

from surfel_texture import GaussianRasterizationSettings
from surfel_texture import GaussianRasterizer
from surfel_texture_deferred import GaussianRasterizer as GaussianRasterizer_deferred


class TextureRenderInputs(NamedTuple):
    deferred: bool
    mbrdf: dict
    colors_precomp: torch.Tensor
    return_split: bool = False


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
    deferred = inputs.deferred
    if deferred and bg_color.shape[0] == 3:
        # Deferred rasterizer needs 7ch bg, matching gs3: [0,0,0,0,0,0,0] for black
        bg_color = torch.cat([bg_color, torch.zeros(4, dtype=bg_color.dtype, device=bg_color.device)])
    raster_settings, scales, rotations, cov3D_precomp = _build_raster_settings(viewpoint_camera, pc, pipe, bg_color,
                                                                               scaling_modifier, bool(getattr(pc, "use_textures", False)))
    mbrdf = inputs.mbrdf      
    deferred = inputs.deferred 
    colors_precomp = inputs.colors_precomp
    means2D = torch.zeros_like(pc.get_xyz, dtype=torch.float32, requires_grad=True, device=pc.get_xyz.device)  
    transmat_grad_holder = None
    if getattr(viewpoint_camera, "cam_pose_adj", None) is not None and viewpoint_camera.cam_pose_adj.requires_grad:
        transmat_grad_holder = torch.zeros(
            (pc.get_xyz.shape[0], 9),
            dtype=torch.float32,
            device=pc.get_xyz.device,
            requires_grad=True,
        )
    shs = None
           
    use_tex = bool(getattr(pc, "use_textures", False))
    texture_alpha = pc.get_texture_alpha if use_tex else None

    if not deferred:
        mbrdf_colors = mbrdf["basecolor"] * mbrdf["shadow"] + mbrdf["other_effects"] if mbrdf is not None else None
        if use_tex:
            texture_color = mbrdf_colors if mbrdf_colors is not None else pc.get_texture_color
        else:
            texture_color = None
            colors_precomp = mbrdf_colors if mbrdf_colors is not None else colors_precomp
            shs = pc.get_features if colors_precomp is None else None

        rendered_image, radii, allmap = GaussianRasterizer(raster_settings)(
            means3D=pc.get_xyz, means2D=means2D,
            shs=shs, colors_precomp=colors_precomp,
            opacities=pc.get_opacity,
            scales=scales, rotations=rotations, cov3D_precomp=cov3D_precomp,
            texture_color=texture_color, texture_alpha=texture_alpha,
            use_textures=use_tex,
            transmat_grad_holder=transmat_grad_holder,
        )
    else:
        mbrdf_colors = torch.concat([mbrdf["basecolor"], mbrdf["shadow"], mbrdf["other_effects"]], dim=1) if mbrdf is not None else None
        if use_tex:
            texture_color = mbrdf_colors if mbrdf_colors is not None else pc.get_texture_color
        else:
            texture_color = None
            colors_precomp = mbrdf_colors if mbrdf_colors is not None else colors_precomp
            shs = None

        rendered_7ch, radii, allmap = GaussianRasterizer_deferred(raster_settings)(
            means3D=pc.get_xyz, means2D=means2D,
            shs=shs, colors_precomp=colors_precomp,
            opacities=pc.get_opacity,
            scales=scales, rotations=rotations, cov3D_precomp=cov3D_precomp,
            texture_color=texture_color, texture_alpha=texture_alpha,
            use_textures=use_tex,
            transmat_grad_holder=transmat_grad_holder,
        )
        # Screen-space deferred composition (matching gs3's train.py approach)
        rendered_image = rendered_7ch[0:3] * rendered_7ch[3:4] + rendered_7ch[4:7]
        rendered_split = rendered_7ch if inputs.return_split else None
        return rendered_image, radii, allmap, means2D, transmat_grad_holder, rendered_split

    return rendered_image, radii, allmap, means2D, transmat_grad_holder, None
