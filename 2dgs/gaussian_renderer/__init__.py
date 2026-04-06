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

import inspect
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.point_utils import depth_to_normal
from utils.sh_utils import eval_sh

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_GS3_ROOT = os.path.join(_REPO_ROOT, "gs3")
_GS3_SUBMODULES = [
    os.path.join(_GS3_ROOT, "submodules", "diff-gaussian-rasterization_light"),
    os.path.join(_GS3_ROOT, "submodules", "v_3dgs"),
]
for _path in reversed(_GS3_SUBMODULES):
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

from diff_gaussian_rasterization_light import (  # noqa: E402
    GaussianRasterizationSettings as LightRasterSettings,
)
from diff_gaussian_rasterization_light import GaussianRasterizer as LightRasterizer  # noqa: E402
import v_3dgs as _v3dgs_module  # noqa: E402
from v_3dgs import GaussianRasterizationSettings as V3DGSSettings  # noqa: E402
from v_3dgs import GaussianRasterizer as V3DGSRasterizer  # noqa: E402


def _patch_v3dgs_backward_signature():
    backward = _v3dgs_module._RasterizeGaussians.backward
    if len(inspect.signature(backward).parameters) != 4:
        return

    def _backward_with_final_t(ctx, grad_out_color, _, grad_out_depth, grad_out_final_t):
        del grad_out_final_t
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        (
            colors_precomp,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            opacities,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        ) = ctx.saved_tensors

        args = (
            raster_settings.bg,
            means3D,
            radii,
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
            grad_out_color,
            grad_out_depth,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            raster_settings.antialiasing,
            raster_settings.debug,
        )

        grads = _v3dgs_module._C.rasterize_gaussians_backward(*args)
        (
            grad_means2D,
            grad_colors_precomp,
            grad_opacities,
            grad_means3D,
            grad_cov3Ds_precomp,
            grad_sh,
            grad_scales,
            grad_rotations,
        ) = grads

        return (
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
        )

    _v3dgs_module._RasterizeGaussians.backward = staticmethod(_backward_with_final_t)


_patch_v3dgs_backward_signature()


def _get_orthographic_matrix_from_bounds(xmin, xmax, ymin, ymax, zmin, zmax):
    return torch.tensor(
        [
            [2 / (xmax - xmin), 0, 0, -(xmax + xmin) / (xmax - xmin)],
            [0, 2 / (ymax - ymin), 0, -(ymax + ymin) / (ymax - ymin)],
            [0, 0, 1 / (zmax - zmin), -zmin / (zmax - zmin)],
            [0, 0, 0, 1],
        ],
        dtype=xmin.dtype,
        device=xmin.device,
    )


def _look_at(camera_position, target_position, up_dir):
    camera_direction = camera_position - target_position
    camera_direction = camera_direction / np.linalg.norm(camera_direction)
    if abs(np.dot(up_dir, camera_direction)) > 0.9:
        up_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    camera_right = np.cross(up_dir, camera_direction)
    camera_right = camera_right / np.linalg.norm(camera_right)
    camera_up = np.cross(camera_direction, camera_right)
    camera_up = camera_up / np.linalg.norm(camera_up)

    rotation_transform = np.zeros((4, 4), dtype=np.float32)
    rotation_transform[0, :3] = camera_right
    rotation_transform[1, :3] = camera_up
    rotation_transform[2, :3] = camera_direction
    rotation_transform[3, 3] = 1.0

    translation_transform = np.eye(4, dtype=np.float32)
    translation_transform[:3, -1] = -np.asarray(camera_position, dtype=np.float32)

    look_at_transform = rotation_transform @ translation_transform
    look_at_transform[1:3, :] *= -1
    return look_at_transform.T


def _compute_shadow_pass(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0):
    if viewpoint_camera.pl_pos is None or pc.get_xyz.numel() == 0:
        return None

    means3d = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    fx_origin = viewpoint_camera.image_width / (2.0 * tanfovx)
    fy_origin = viewpoint_camera.image_height / (2.0 * tanfovy)

    object_center = means3d.mean(dim=0).detach().cpu().numpy()
    light_position = viewpoint_camera.pl_pos.detach().cpu().numpy()
    world_view_transform_light = _look_at(
        light_position,
        object_center,
        up_dir=np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )
    world_view_transform_light = torch.tensor(
        world_view_transform_light,
        device=viewpoint_camera.world_view_transform.device,
        dtype=viewpoint_camera.world_view_transform.dtype,
    )

    camera_position = viewpoint_camera.camera_center.detach().cpu().numpy() * pipe.shadow_light_scale
    light_norm = max(np.sum(light_position * light_position), 1e-8)
    camera_norm = max(np.sum(camera_position * camera_position), 1e-8)
    f_scale_ratio = math.sqrt(light_norm / camera_norm) * 0.2
    fx_far = fx_origin * f_scale_ratio
    fy_far = fy_origin * f_scale_ratio

    xyz_h = torch.cat([means3d, torch.ones_like(means3d[:, :1])], dim=1)
    xyz_ortho = xyz_h @ world_view_transform_light
    x_min, y_min, z_min = xyz_ortho[:, 0].min(), xyz_ortho[:, 1].min(), xyz_ortho[:, 2].min()
    x_max, y_max, z_max = xyz_ortho[:, 0].max(), xyz_ortho[:, 1].max(), xyz_ortho[:, 2].max()

    light_ortho_proj_matrix = _get_orthographic_matrix_from_bounds(
        xmin=x_min,
        xmax=x_max,
        ymin=y_min,
        ymax=y_max,
        zmin=z_min,
        zmax=z_max,
    ).transpose(0, 1).cuda()
    full_ortho_proj_transform_light = (
        world_view_transform_light.unsqueeze(0).bmm(light_ortho_proj_matrix.unsqueeze(0))
    ).squeeze(0)

    tanfovx_far = 0.5 * viewpoint_camera.image_width / fx_far
    tanfovy_far = 0.5 * viewpoint_camera.image_height / fy_far
    resolution_scale = max(float(getattr(pipe, "shadow_resolution_scale", 1.0)), 0.25)
    h_light = max(32, int(resolution_scale * viewpoint_camera.image_height))
    w_light = max(32, int(resolution_scale * viewpoint_camera.image_width))
    max_shadow_res = 2048
    h_light = min(h_light, max_shadow_res)
    w_light = min(w_light, max_shadow_res)

    opacity_light = torch.zeros((means3d.shape[0], 1), dtype=torch.float32, device="cuda")
    means2d_light = torch.zeros_like(means3d, dtype=means3d.dtype, requires_grad=True, device="cuda")

    light_settings = LightRasterSettings(
        image_height=h_light,
        image_width=w_light,
        tanfovx=tanfovx_far,
        tanfovy=tanfovy_far,
        bg=bg_color[:3],
        scale_modifier=scaling_modifier,
        viewmatrix=world_view_transform_light,
        projmatrix=full_ortho_proj_transform_light,
        sh_degree=pc.active_sh_degree,
        campos=torch.tensor(light_position, dtype=torch.float32, device="cuda"),
        prefiltered=False,
        debug=pipe.debug,
        low_pass_filter_radius=0.3,
        ortho=True,
    )
    light_rasterizer = LightRasterizer(raster_settings=light_settings)
    light_colors = torch.ones((means3d.shape[0], 3), dtype=torch.float32, device="cuda")
    _, _, _, out_trans, opacity_light, _ = light_rasterizer(
        means3D=means3d,
        means2D=means2d_light,
        shs=None,
        colors_precomp=light_colors,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3Ds_precomp=None,
        non_trans=opacity_light,
        offset=pipe.shadow_offset,
        thres=-1.0,
        is_train=False,
        hgs=False,
        hgs_normals=None,
        hgs_opacities=None,
        hgs_opacities_shadow=None,
        hgs_opacities_light=None,
        streams=None,
    )
    opacity_light = torch.clamp_min(opacity_light, 1e-6)
    shadow = torch.clamp(out_trans / opacity_light, 0.0, 1.0)

    view_settings = V3DGSSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=0,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=True,
        znear=viewpoint_camera.znear,
        zfar=viewpoint_camera.zfar,
    )
    view_rasterizer = V3DGSRasterizer(raster_settings=view_settings)
    means2d_view = torch.zeros_like(means3d, dtype=means3d.dtype, requires_grad=True, device="cuda")
    zero_colors = torch.zeros((means3d.shape[0], 3), dtype=torch.float32, device="cuda")
    shadow_rendered, _, _, _ = view_rasterizer(
        means3D=means3d,
        means2D=means2d_view,
        shs=None,
        colors_precomp=zero_colors,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
        shadow=shadow,
    )

    return {
        "per_point_shadow": shadow.detach(),
        "shadow_image": torch.clamp(shadow_rendered[3:4], 0.0, 1.0).detach(),
        "light_viewmatrix": world_view_transform_light,
        "light_projmatrix": full_ortho_proj_transform_light,
    }


def _safe_normalize(vec):
    return F.normalize(vec, p=2, dim=-1, eps=1e-6)


def _ndotwi(normals, wi):
    return F.elu(torch.sum(normals * wi, dim=-1, keepdim=True), alpha=0.01) + 0.01


def _compute_mbrdf_colors(viewpoint_camera, pc: GaussianModel, shadow_pkg):
    if viewpoint_camera.pl_pos is None:
        return None

    pl_pos = viewpoint_camera.pl_pos
    if pl_pos.ndim == 1:
        pl_pos = pl_pos.unsqueeze(0)
    pl_pos_expand = pl_pos[0].unsqueeze(0).expand(pc.get_xyz.shape[0], -1)
    wi_ray = pl_pos_expand - pc.get_xyz
    wi_dist2 = torch.sum(wi_ray * wi_ray, dim=-1, keepdim=True).clamp_min(1e-12)
    wi = wi_ray * torch.rsqrt(wi_dist2)
    wo = _safe_normalize(viewpoint_camera.camera_center - pc.get_xyz)

    local_axises = pc.get_local_axis
    local_z = local_axises[:, :, 2]
    wi_local = torch.einsum("ki,kij->kj", wi, local_axises)
    wo_local = torch.einsum("ki,kij->kj", wo, local_axises)
    asg_scales = pc.asg_func.get_asg_lam_miu
    asg_axises = pc.asg_func.get_asg_axis
    diffuse = pc.get_kd / math.pi
    asg_1 = pc.asg_func(wi_local, wo_local, pc.get_alpha_asg, asg_scales, asg_axises)
    shadow_hint = None if shadow_pkg is None else shadow_pkg["per_point_shadow"]
    decay, other_effects, asg_3, _ = pc.neural_phasefunc(
        wi,
        wo,
        pc.get_xyz,
        pc.get_neural_material,
        hint=shadow_hint,
        asg_1=asg_1,
        asg_mlp=pc.asg_mlp,
    )
    if decay is None:
        decay = torch.ones((pc.get_xyz.shape[0], 1), dtype=torch.float32, device="cuda")
    specular = pc.get_ks * asg_3
    colors = (diffuse + specular) * _ndotwi(local_z, wi) * (1.0 / wi_dist2)
    colors = colors * decay
    if other_effects is not None:
        colors = colors + other_effects * (1.0 / wi_dist2)
    return torch.clamp_min(colors, 0.0)


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

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
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor(
            [
                [W / 2, 0, 0, (W - 1) / 2],
                [0, H / 2, 0, (H - 1) / 2],
                [0, 0, far - near, near],
                [0, 0, 0, 1],
            ]
        ).float().cuda().T
        world2pix = viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (
            splat2world[:, [0, 1, 3]] @ world2pix[:, [0, 1, 3]]
        ).permute(0, 2, 1).reshape(-1, 9)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    pipe.convert_SHs_python = False
    shadow_pkg = None
    if getattr(pipe, "shadow_pass", False):
        shadow_pkg = _compute_shadow_pass(viewpoint_camera, pc, pipe, bg_color, scaling_modifier)

    shs = None
    colors_precomp = None
    if override_color is None:
        if getattr(pc, "use_mbrdf", False):
            colors_precomp = _compute_mbrdf_colors(viewpoint_camera, pc, shadow_pkg)
        if colors_precomp is None and pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        elif colors_precomp is None:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    rendered_image, radii, allmap = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    if shadow_pkg is not None:
        if not getattr(pc, "use_mbrdf", False):
            rendered_image = rendered_image * shadow_pkg["shadow_image"].repeat(3, 1, 1)

    rets = {
        "render": rendered_image,
        "viewspace_points": means2D,
        "visibility_filter": radii > 0,
        "radii": radii,
    }

    render_alpha = allmap[1:2]
    render_normal = allmap[2:5]
    render_normal = (
        render_normal.permute(1, 2, 0) @ (viewpoint_camera.world_view_transform[:3, :3].T)
    ).permute(2, 0, 1)

    render_depth_median = torch.nan_to_num(allmap[5:6], 0, 0)
    render_depth_expected = torch.nan_to_num(allmap[0:1] / render_alpha, 0, 0)
    render_dist = allmap[6:7]

    surf_depth = render_depth_expected * (1 - pipe.depth_ratio) + pipe.depth_ratio * render_depth_median
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth).permute(2, 0, 1)
    surf_normal = surf_normal * render_alpha.detach()

    rets.update(
        {
            "rend_alpha": render_alpha,
            "rend_normal": render_normal,
            "rend_dist": render_dist,
            "surf_depth": surf_depth,
            "surf_normal": surf_normal,
            "shadow": None if shadow_pkg is None else shadow_pkg["shadow_image"],
            "pre_shadow": None if shadow_pkg is None else shadow_pkg["per_point_shadow"],
        }
    )

    return rets
