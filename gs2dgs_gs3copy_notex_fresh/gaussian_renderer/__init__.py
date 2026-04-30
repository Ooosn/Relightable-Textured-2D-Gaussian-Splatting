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

import os
import sys
from contextlib import nullcontext
import torch
import torch.nn.functional as F
import numpy as np
import math
from scipy.spatial.transform import Rotation as Rot

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SUBMODULE_PATHS = [
    os.path.join(_REPO_ROOT, "submodules", "diff-gaussian-rasterization"),
    os.path.join(_REPO_ROOT, "submodules", "diff-gaussian-rasterization_light"),
    os.path.join(_REPO_ROOT, "submodules", "diff-gaussian-rasterization_hgs"),
    os.path.join(_REPO_ROOT, "submodules", "v_3dgs"),
    os.path.join(_REPO_ROOT, "submodules", "v_3dgs_ortho"),
    os.path.join(_REPO_ROOT, "..", "2dgs", "submodules", "surfel-texture"),
    os.path.join(_REPO_ROOT, "..", "2dgs", "submodules", "surfel-texture-deferred"),
    os.path.join(_REPO_ROOT, "..", "2dgs", "submodules", "diff-surfel-rasterization-shadow"),
]
for _path in reversed(_SUBMODULE_PATHS):
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

from gsplat import rasterization
from utils.graphics_utils import fov2focal
from diff_gaussian_rasterization import  GaussianRasterizationSettings as basic_settings
from diff_gaussian_rasterization import  GaussianRasterizer as basic_rasterizer
from diff_gaussian_rasterization import CamGradBridge as basic_cam_grad_bridge
from diff_gaussian_rasterization_light import GaussianRasterizationSettings as light_settings
from diff_gaussian_rasterization_light import  GaussianRasterizer as light_rasterizer
from v_3dgs import GaussianRasterizationSettings as v_3dgs_settings
from v_3dgs import GaussianRasterizer as persp_rasterizer
from v_3dgs_ortho import GaussianRasterizer as ortho_rasterizer
from surfel_texture import GaussianRasterizationSettings as surfel_settings
from surfel_texture import GaussianRasterizer as surfel_rasterizer
from surfel_texture_deferred import GaussianRasterizer as surfel_rasterizer_deferred
from diff_surfel_rasterization_shadow import GaussianRasterizationSettings as surfel_shadow_settings
from diff_surfel_rasterization_shadow import GaussianRasterizer as surfel_shadow_rasterizer
from gaussian_renderer.texture_branch import render_2dgs_texture_deferred

from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.graphics_utils import getProjectionMatrix
from utils.general_utils import build_scaling_rotation, strip_symmetric
from utils.graphics_utils import getProjectionMatrixWithPrincipalPoint, look_at,getProjectionMatrix, \
                                 getOrthographicMatrixFromBounds, computeCov2D_ortho_python, euler_to_matrix, \
                                 look_at_cu
import time
from utils.general_utils import GradientScaler


debug = False

def has_nan_or_inf(tensor, dim):
    return tensor.isnan().any(dim=dim) | tensor.isinf().any(dim=dim)


def _env_enabled(name, default="0"):
    return os.environ.get(name, default) == "1"


def _maybe_cuda_sync():
    if _env_enabled("SSGS_RENDER_SYNC"):
        torch.cuda.synchronize()


def _build_2dgs_raster_settings(viewpoint_camera, pipe, bg_color, scaling_modifier, sh_degree):
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    return surfel_settings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=getattr(pipe, "debug", False),
    )


def _look_at_2dgs(camera_position, target_position, up_dir):
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


def _build_light_transform_2dgs(viewpoint_camera, means3d, pipe):
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    fx_origin = viewpoint_camera.image_width / (2.0 * tanfovx)
    fy_origin = viewpoint_camera.image_height / (2.0 * tanfovy)

    object_center = means3d.mean(dim=0).detach().cpu().numpy()
    light_position = viewpoint_camera.pl_pos.detach().cpu().numpy().reshape(-1)
    world_view_transform_light = _look_at_2dgs(
        light_position,
        object_center,
        up_dir=np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )
    world_view_transform_light = torch.tensor(
        world_view_transform_light,
        device=viewpoint_camera.world_view_transform.device,
        dtype=viewpoint_camera.world_view_transform.dtype,
    )

    camera_position = viewpoint_camera.camera_center.detach().cpu().numpy() * getattr(pipe, "shadow_light_scale", 1.0)
    light_norm = max(float(np.sum(light_position * light_position)), 1e-8)
    camera_norm = max(float(np.sum(camera_position * camera_position)), 1e-8)
    f_scale_ratio = math.sqrt(light_norm / camera_norm)
    fx_far = fx_origin * f_scale_ratio
    fy_far = fy_origin * f_scale_ratio

    tanfovx_far = 0.5 * viewpoint_camera.image_width / fx_far
    tanfovy_far = 0.5 * viewpoint_camera.image_height / fy_far
    resolution_scale = max(float(getattr(pipe, "shadow_resolution_scale", 1.0)), 0.25)
    h_light = min(max(32, int(resolution_scale * viewpoint_camera.image_height)), 2048)
    w_light = min(max(32, int(resolution_scale * viewpoint_camera.image_width)), 2048)

    light_persp_proj_matrix = getProjectionMatrix(
        znear=viewpoint_camera.znear,
        zfar=viewpoint_camera.zfar,
        fovX=2.0 * math.atan(tanfovx_far),
        fovY=2.0 * math.atan(tanfovy_far),
    ).transpose(0, 1).cuda()
    light_projmatrix = (
        world_view_transform_light.unsqueeze(0).bmm(light_persp_proj_matrix.unsqueeze(0))
    ).squeeze(0)

    return dict(
        world_view_transform_light=world_view_transform_light,
        light_projmatrix=light_projmatrix,
        tanfovx_far=tanfovx_far,
        tanfovy_far=tanfovy_far,
        h_light=h_light,
        w_light=w_light,
        light_position=light_position,
    )


def _compute_shadow_pass_2dgs_native(viewpoint_camera, gau, pipe, bg_color, scaling_modifier=1.0):
    if viewpoint_camera.pl_pos is None or gau.get_xyz.numel() == 0:
        return None

    means3d = gau.get_xyz
    opacity = gau.get_opacity
    scales = gau.get_scaling
    rotations = gau.get_rotation
    lt = _build_light_transform_2dgs(viewpoint_camera, means3d, pipe)

    shadow_settings = surfel_shadow_settings(
        image_height=lt["h_light"],
        image_width=lt["w_light"],
        tanfovx=lt["tanfovx_far"],
        tanfovy=lt["tanfovy_far"],
        bg=bg_color[:3],
        scale_modifier=scaling_modifier,
        viewmatrix=lt["world_view_transform_light"],
        projmatrix=lt["light_projmatrix"],
        sh_degree=gau.active_sh_degree,
        campos=torch.tensor(lt["light_position"], dtype=torch.float32, device="cuda"),
        prefiltered=False,
        debug=getattr(pipe, "debug", False),
        low_pass_filter_radius=0.3,
        ortho=False,
        use_textures=False,
    )
    shadow_rasterizer = surfel_shadow_rasterizer(raster_settings=shadow_settings)
    light_colors = torch.ones((means3d.shape[0], 3), dtype=torch.float32, device="cuda")

    _, _, _, out_trans, non_trans, _ = shadow_rasterizer(
        means3D=means3d,
        means2D=torch.zeros_like(means3d, requires_grad=True),
        shs=None,
        colors_precomp=light_colors,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3Ds_precomp=None,
        texture_alpha=torch.empty(0, device="cuda"),
        texture_sigma_factor=3.0,
        non_trans=None,
        offset=getattr(pipe, "shadow_offset", 0.015),
        thres=-1.0,
        is_train=False,
    )
    per_point_shadow = (out_trans / torch.clamp_min(non_trans, 1e-6)).unsqueeze(-1)
    return {
        "per_point_shadow": per_point_shadow,
        "light_viewmatrix": lt["world_view_transform_light"],
        "light_projmatrix": lt["light_projmatrix"],
    }


def _compute_shadow_pass_2dgs_lifted(viewpoint_camera, gau, pipe, bg_color, modelset, scaling_modifier=1.0):
    if viewpoint_camera.pl_pos is None or gau.get_xyz.numel() == 0:
        return None

    means3d = gau.get_xyz
    opacity = gau.get_opacity
    scales3 = torch.cat([gau.get_scaling, torch.ones_like(gau.get_scaling[:, :1])], dim=-1)
    rotations = gau.get_rotation
    lt = _build_light_transform_2dgs(viewpoint_camera, means3d, pipe)

    L = build_scaling_rotation(scaling_modifier * scales3, rotations)
    cov3d = strip_symmetric(L @ L.transpose(1, 2))

    raster_settings_light = light_settings(
        image_height=lt["h_light"],
        image_width=lt["w_light"],
        tanfovx=lt["tanfovx_far"],
        tanfovy=lt["tanfovy_far"],
        bg=bg_color[:3],
        scale_modifier=scaling_modifier,
        viewmatrix=lt["world_view_transform_light"],
        projmatrix=lt["light_projmatrix"],
        sh_degree=gau.active_sh_degree,
        campos=viewpoint_camera.pl_pos[0],
        prefiltered=False,
        debug=getattr(pipe, "debug", False),
        low_pass_filter_radius=0.3,
        ortho=False,
    )
    rasterizer_light = light_rasterizer(raster_settings=raster_settings_light)
    light_inputs = {
        "means3D": means3d,
        "means2D": torch.zeros_like(means3d, dtype=means3d.dtype, requires_grad=True, device=means3d.device),
        "shs": None,
        "colors_precomp": torch.ones((means3d.shape[0], 3), dtype=torch.float32, device="cuda"),
        "opacities": opacity,
        "scales": None,
        "rotations": None,
        "cov3Ds_precomp": cov3d,
        "projmatrix": lt["light_projmatrix"],
        "viewmatrix": lt["world_view_transform_light"],
        "non_trans": torch.zeros((means3d.shape[0], 1), dtype=torch.float32, device="cuda"),
        "offset": getattr(modelset, "offset", 0.015),
        "thres": 4,
        "is_train": False,
        "hgs": False,
        "hgs_normals": None,
        "hgs_opacities": None,
        "hgs_opacities_shadow": None,
        "hgs_opacities_light": None,
        "streams": None,
    }
    _, _, _, shadow, non_trans, _ = rasterizer_light(**light_inputs)
    per_point_shadow = shadow / torch.clamp_min(non_trans, 1e-6)
    return {
        "per_point_shadow": per_point_shadow,
        "light_viewmatrix": lt["world_view_transform_light"],
        "light_projmatrix": lt["light_projmatrix"],
    }


def _render_2dgs_native_3ch(viewpoint_camera, gau, pipe, bg_color, modelset, scaling_modifier=1.0, fix_labert=False, shadow_map=False, iteration=0):
    means3D = gau.get_xyz
    means2D = torch.zeros_like(means3D, dtype=torch.float32, requires_grad=True, device=means3D.device)
    opacity = gau.get_opacity
    scales = gau.get_scaling
    rotations = gau.get_rotation
    transmat_grad_holder = None
    if getattr(viewpoint_camera, "cam_pose_adj", None) is not None and viewpoint_camera.cam_pose_adj.requires_grad:
        transmat_grad_holder = torch.zeros(
            (means3D.shape[0], 9),
            dtype=torch.float32,
            device=means3D.device,
            requires_grad=True,
        )

    if gau.use_MBRDF:
        if os.getenv("GS3_2DGS_USE_LIFTED_SHADOW", "0") == "1":
            shadow_pkg = _compute_shadow_pass_2dgs_lifted(viewpoint_camera, gau, pipe, bg_color, modelset, scaling_modifier)
        else:
            shadow_pkg = _compute_shadow_pass_2dgs_native(viewpoint_camera, gau, pipe, bg_color, scaling_modifier)
    else:
        shadow_pkg = None

    colors_precomp = None
    shs = None
    shadow_img = None
    other_img = None
    if gau.use_MBRDF:
        pl_pos_expand = viewpoint_camera.pl_pos.expand(gau.get_xyz.shape[0], -1)
        wi_ray = pl_pos_expand - gau.get_xyz
        wi_dist2 = torch.sum(wi_ray**2, dim=-1, keepdim=True).clamp_min(1e-12)
        dist_2_inv = 1.0 / wi_dist2
        wi = wi_ray * torch.sqrt(dist_2_inv)
        camera_center_for_brdf = viewpoint_camera.camera_center
        if os.getenv("GS3_2DGS_DETACH_VIEWDIR", "0") == "1":
            camera_center_for_brdf = camera_center_for_brdf.detach()
        wo = _safe_normalize(camera_center_for_brdf - gau.get_xyz)

        local_axises = gau.get_local_axis
        local_z = local_axises[:, :, 2]
        wi_local = torch.einsum('Ki,Kij->Kj', wi, local_axises)
        wo_local = torch.einsum('Ki,Kij->Kj', wo, local_axises)
        cosTheta = _NdotWi(local_z, wi, torch.nn.ELU(alpha=0.01), 0.01)
        diffuse = gau.get_kd / math.pi
        asg_scales = gau.asg_func.get_asg_lam_miu
        asg_axises = gau.asg_func.get_asg_axis
        asg_1 = gau.asg_func(wi_local, wo_local, gau.get_alpha_asg, asg_scales, asg_axises)
        shadow_hint = None if shadow_pkg is None else shadow_pkg["per_point_shadow"]
        decay, other_effects, asg_3, _ = gau.neural_phasefunc(
            wi, wo, gau.get_xyz, gau.get_neural_material,
            hint=shadow_hint, asg_1=asg_1, asg_mlp=gau.asg_mlp,
        )
        if decay is None:
            decay = torch.ones((means3D.shape[0], 1), dtype=torch.float32, device=means3D.device)
        if fix_labert:
            basecolor = diffuse * cosTheta * dist_2_inv
        else:
            specular = gau.get_ks * asg_3
            basecolor = (diffuse + specular) * cosTheta * dist_2_inv
        if other_effects is not None:
            other_effects = other_effects * dist_2_inv
        else:
            other_effects = torch.zeros_like(basecolor)
        colors_precomp = basecolor * decay + other_effects
    else:
        shs = gau.get_features

    raster_settings_2dgs = _build_2dgs_raster_settings(
        viewpoint_camera, pipe, bg_color[:3], scaling_modifier, gau.active_sh_degree
    )
    rasterizer_2dgs = surfel_rasterizer(raster_settings=raster_settings_2dgs)
    rendered_rgb, radii, allmap = rasterizer_2dgs(
        means3D=means3D,
        means2D=means2D,
        opacities=opacity,
        shs=shs if colors_precomp is None else None,
        colors_precomp=colors_precomp,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
        texture_color=None,
        texture_alpha=None,
        use_textures=False,
        transmat_grad_holder=transmat_grad_holder,
    )
    try:
        means2D.retain_grad()
    except:
        pass

    render_alpha = allmap[1:2].clamp_min(1e-8)
    expected_depth = torch.nan_to_num(allmap[0:1] / render_alpha, 0, 0)
    shadow_img = torch.ones((1, rendered_rgb.shape[1], rendered_rgb.shape[2]), dtype=rendered_rgb.dtype, device=rendered_rgb.device)
    other_img = torch.zeros((3, rendered_rgb.shape[1], rendered_rgb.shape[2]), dtype=rendered_rgb.dtype, device=rendered_rgb.device)
    return {
        "render": rendered_rgb,
        "shadow": shadow_img,
        "other_effects": other_img,
        "viewspace_points": means2D,
        "visibility_filter": radii > 0,
        "radii": radii,
        "out_weight": torch.zeros((means3D.shape[0], 1), dtype=torch.float32, device="cuda"),
        "backward_info": {},
        "shadow_stage": get_shadow_backward_stage(modelset, iteration),
        "transmat_grad_holder": transmat_grad_holder,
        "expected_depth": expected_depth,
    }


def _render_2dgs_native_deferred(viewpoint_camera, gau, pipe, bg_color, modelset, scaling_modifier=1.0, fix_labert=False, shadow_map=False, iteration=0):
    means3D = gau.get_xyz
    means2D = torch.zeros_like(means3D, dtype=torch.float32, requires_grad=True, device=means3D.device)
    opacity = gau.get_opacity
    scales = gau.get_scaling
    rotations = gau.get_rotation
    transmat_grad_holder = None
    if getattr(viewpoint_camera, "cam_pose_adj", None) is not None and viewpoint_camera.cam_pose_adj.requires_grad:
        transmat_grad_holder = torch.zeros(
            (means3D.shape[0], 9),
            dtype=torch.float32,
            device=means3D.device,
            requires_grad=True,
        )

    if gau.use_MBRDF:
        if os.getenv("GS3_2DGS_USE_LIFTED_SHADOW", "0") == "1":
            shadow_pkg = _compute_shadow_pass_2dgs_lifted(viewpoint_camera, gau, pipe, bg_color, modelset, scaling_modifier)
        else:
            shadow_pkg = _compute_shadow_pass_2dgs_native(viewpoint_camera, gau, pipe, bg_color, scaling_modifier)
    else:
        shadow_pkg = None

    colors_precomp = None
    if gau.use_MBRDF:
        pl_pos_expand = viewpoint_camera.pl_pos.expand(gau.get_xyz.shape[0], -1)
        wi_ray = pl_pos_expand - gau.get_xyz
        wi_dist2 = torch.sum(wi_ray**2, dim=-1, keepdim=True).clamp_min(1e-12)
        dist_2_inv = 1.0 / wi_dist2
        wi = wi_ray * torch.sqrt(dist_2_inv)
        camera_center_for_brdf = viewpoint_camera.camera_center
        if os.getenv("GS3_2DGS_DETACH_VIEWDIR", "0") == "1":
            camera_center_for_brdf = camera_center_for_brdf.detach()
        wo = _safe_normalize(camera_center_for_brdf - gau.get_xyz)

        local_axises = gau.get_local_axis
        local_z = local_axises[:, :, 2]
        wi_local = torch.einsum('Ki,Kij->Kj', wi, local_axises)
        wo_local = torch.einsum('Ki,Kij->Kj', wo, local_axises)
        cosTheta = _NdotWi(local_z, wi, torch.nn.ELU(alpha=0.01), 0.01)
        diffuse = gau.get_kd / math.pi
        asg_scales = gau.asg_func.get_asg_lam_miu
        asg_axises = gau.asg_func.get_asg_axis
        asg_1 = gau.asg_func(wi_local, wo_local, gau.get_alpha_asg, asg_scales, asg_axises)
        shadow_hint = None if shadow_pkg is None else shadow_pkg["per_point_shadow"]
        decay, other_effects, asg_3, _ = gau.neural_phasefunc(
            wi, wo, gau.get_xyz, gau.get_neural_material,
            hint=shadow_hint, asg_1=asg_1, asg_mlp=gau.asg_mlp,
        )
        if decay is None:
            decay = torch.ones((means3D.shape[0], 1), dtype=torch.float32, device=means3D.device)
        if fix_labert:
            basecolor = diffuse * cosTheta * dist_2_inv
        else:
            specular = gau.get_ks * asg_3
            basecolor = (diffuse + specular) * cosTheta * dist_2_inv
        if other_effects is None:
            other_effects = torch.zeros_like(basecolor)
        else:
            other_effects = other_effects * dist_2_inv
        colors_precomp = torch.cat([basecolor, decay, other_effects], dim=1)
    else:
        colors_precomp = None

    surfel_bg = bg_color
    if surfel_bg.shape[0] == 3:
        surfel_bg = torch.cat(
            [surfel_bg, torch.zeros(4, dtype=surfel_bg.dtype, device=surfel_bg.device)],
            dim=0,
        )
    raster_settings_2dgs = _build_2dgs_raster_settings(
        viewpoint_camera, pipe, surfel_bg, scaling_modifier, gau.active_sh_degree
    )
    rasterizer_2dgs = surfel_rasterizer_deferred(raster_settings=raster_settings_2dgs)
    rendered_7ch, radii, allmap = rasterizer_2dgs(
        means3D=means3D,
        means2D=means2D,
        opacities=opacity,
        shs=None,
        colors_precomp=colors_precomp,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
        texture_color=None,
        texture_alpha=None,
        use_textures=False,
        transmat_grad_holder=transmat_grad_holder,
    )
    try:
        means2D.retain_grad()
    except:
        pass

    render_alpha = allmap[1:2].clamp_min(1e-8)
    expected_depth = torch.nan_to_num(allmap[0:1] / render_alpha, 0, 0)
    return {
        "render": rendered_7ch[0:3],
        "shadow": rendered_7ch[3:4],
        "other_effects": rendered_7ch[4:7],
        "viewspace_points": means2D,
        "visibility_filter": radii > 0,
        "radii": radii,
        "out_weight": torch.zeros((means3D.shape[0], 1), dtype=torch.float32, device="cuda"),
        "backward_info": {},
        "shadow_stage": get_shadow_backward_stage(modelset, iteration),
        "transmat_grad_holder": transmat_grad_holder,
        "expected_depth": expected_depth,
    }


def get_shadow_backward_stage(modelset, iteration: int):
    if getattr(modelset, "detach_shadow", False):
        return {
            "xyz": False,
            "opacity": False,
            "scaling": False,
            "rotation": False,
        }
    if not getattr(modelset, "shadow_backward_stage_enabled", False):
        return {
            "xyz": True,
            "opacity": True,
            "scaling": True,
            "rotation": True,
        }
    return {
        "xyz": iteration >= int(getattr(modelset, "shadow_backward_xyz_from_iter", 0)),
        "opacity": iteration >= int(getattr(modelset, "shadow_backward_opacity_from_iter", 0)),
        "scaling": iteration >= int(getattr(modelset, "shadow_backward_scaling_from_iter", 0)),
        "rotation": iteration >= int(getattr(modelset, "shadow_backward_rotation_from_iter", 0)),
    }


def _shadow_stage_tensor(tensor, enabled: bool):
    if tensor is None or enabled:
        return tensor
    return tensor.detach()

def render(viewpoint_camera, 
           gau : GaussianModel, 
           light_stream, 
           calc_stream, 
           local_axises, 
           asg_scales, 
           asg_axises, 
           pipe, 
           bg_color : torch.Tensor, 
           modelset,
           shadowmap_render = False,
           scaling_modifier = 1.0,  # 高斯点的缩放因子，默认为 1.0，用于二次调整整体的高斯点大小
           override_color = None,   # 覆盖学习到的颜色，默认为 None，即使用学习到的颜色
           fix_labert = False,  # 是否只考虑漫反射，根据当前迭代次数来决定
           inten_scale = 1.0,   # 颜色强度缩放，但是感觉被弃用了
           is_train = False,    # 根据 prune_visibility
           asg_mlp = False,
           iteration = 0): 
    if (
        str(getattr(gau, "rasterizer", "")) == "2dgs_3ch"
        and hasattr(gau, "get_scaling")
        and gau.get_scaling.shape[-1] == 2
    ):
        return _render_2dgs_native_3ch(
            viewpoint_camera,
            gau,
            pipe,
            bg_color,
            modelset,
            scaling_modifier=scaling_modifier,
            fix_labert=fix_labert,
            shadow_map=shadowmap_render,
            iteration=iteration,
        )
    if (
        str(getattr(gau, "rasterizer", "")) == "2dgs"
        and hasattr(gau, "get_scaling")
        and gau.get_scaling.shape[-1] == 2
    ):
        if bool(getattr(gau, "use_textures", False)):
            return render_2dgs_texture_deferred(
                viewpoint_camera,
                gau,
                pipe,
                bg_color,
                modelset,
                scaling_modifier=scaling_modifier,
                fix_labert=fix_labert,
                iteration=iteration,
            )
        return _render_2dgs_native_deferred(
            viewpoint_camera,
            gau,
            pipe,
            bg_color,
            modelset,
            scaling_modifier=scaling_modifier,
            fix_labert=fix_labert,
            shadow_map=shadowmap_render,
            iteration=iteration,
        )
    
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # 1）光源方向的高斯泼溅 ———— 光栅化参数计算工作:

    """ 
    1. 焦距随着距离成比例缩放
    2. 这里将光源作为一个相机
    3. 因此通过计算光源和相机的距离比，来缩放得到光源的 焦距
    """
    # 计算相机的原始焦距: f = \frac{W}{2 \tan(\frac{\text{FoV}}{2})}
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    # calculate the fov and projmatrix of light
    fx_origin = viewpoint_camera.image_width / (2. * tanfovx)
    fy_origin = viewpoint_camera.image_height / (2. * tanfovy)

    # calculate the fov for shadow splatting:
    # 计算光源和相机的距离比 f_scale_ratio
    light_position = viewpoint_camera.pl_pos[0].detach().clone()
    camera_position = viewpoint_camera.camera_center.detach().clone()
    f_scale_ratio = torch.sqrt(torch.sum(light_position * light_position) / torch.sum(camera_position * camera_position))
    
    # 计算光源的焦距
    fx_far = fx_origin * f_scale_ratio
    fy_far = fy_origin * f_scale_ratio
    cx = viewpoint_camera.image_width / 2.0
    cy = viewpoint_camera.image_height / 2.0
    
    # 计算光源的视场角，即 FoV
    """
    相机的视锥体（View Frustum）决定了相机可以看到的空间范围：
        - 相机的视锥体是一个四棱锥：
            - 顶点 是相机的光心。
            - 成像平面（传感器 / near 平面），它始终与图像长宽一致，是固定的
            - 远平面 (far 平面)：透视扩展出来的“虚拟底”，它的大小不是固定的，而是透视投影的结果。
            - 图形学中，far 平面的距离可以受到人为规定，对场景进行裁剪，因此根据 fov 最后得到 far 平面的大小
            - 高 是焦距 f（光心到成像平面的距离），即定义了 near 平面。
        - 因此呈现出焦距大，范围小，焦距小，范围大
        - 焦距 和 FoV 是视野大小的不同表达方式，一个用距离，一个用角度，在确定 fov 形式（水平、垂直、对角）的情况下可以互相换算
    """
    # 先计算 FoV 对应的正切值，然后通过 arctan 得出 FoV
    tanfovx_far = 0.5 * viewpoint_camera.image_width / fx_far
    tanfovy_far = 0.5 * viewpoint_camera.image_height / fy_far
    # 将焦距反推回视场角（FoV）
    fovx_far = 2 * math.atan(tanfovx_far)
    fovy_far = 2 * math.atan(tanfovy_far)

    use_customized_light_direction = False
    if use_customized_light_direction:
        # 把面板里的度数填进来
        euler_deg = gau.light_direction
        r = euler_to_matrix(euler_deg)
        local_forward = torch.tensor([0, 0, 1], dtype=torch.float32, device='cuda')
        light_dir = (r @ local_forward)
    else:
        light_dir = None
    
    
    # 2）光源方向的高斯泼溅 ———— 视角转换准备工作: 

    # 用于计算 shadow （shadow splatting）
    # 计算 世界坐标系 到 光源坐标系 的变换矩阵
    # 目前每个高斯点的坐标采取 行向量 来表示
    object_center=gau.get_xyz.mean(dim=0).detach().clone()
    if use_customized_light_direction:
        orginal_light_vector = light_position - object_center
        distance_to_object = torch.linalg.norm(orginal_light_vector)       
        light_position = object_center + distance_to_object * light_dir
    
    
    
    world_view_transform_light=look_at_cu(light_position,
                                       object_center,
                                       up_dir=torch.tensor([0, 0, 1], dtype=torch.float32, device="cuda"),
                                       camera_direction=light_dir)
    world_view_transform_light = world_view_transform_light.to(
                                        dtype=viewpoint_camera.world_view_transform.dtype,
                                        device=viewpoint_camera.world_view_transform.device)
    # 为了对齐 点 为行向量: P' = Pro Tw2c P  ->  P'^T = P^T (Tw2c)^T (Pro)^T
    # 这里对 投影矩阵进行转置, torch.transpose(0, 1)
    
    # Keep the default light camera consistent with the original training path:
    # a perspective light camera derived from the view camera/light distance.
    # The orthographic branch remains available for explicit experiments only.
    ortho = False
    object_id = None
    light_box = False
    xyz = gau.get_xyz
    shadow_map = shadowmap_render
    depth_image = None
    
    light_stream_ctx = torch.cuda.stream(light_stream) if light_stream is not None else nullcontext()
    calc_stream_ctx = torch.cuda.stream(calc_stream) if calc_stream is not None else nullcontext()
    current_stream = torch.cuda.current_stream()
    
    if object_id is not None:
        """
        根据 id 设置 mask
        """
        # mask_sky = object_id[:-2, ] == 0
        # mask_scene = (object_id[:-2, ] == 1) | mask_sky
        # mask_blanket = object_id[:-2, ] >= 100
    
    
    if ortho:
            xyz_4 = torch.cat([xyz, torch.ones_like(xyz[:, 0:1])], dim=1)
            xyz_ortho_scene = xyz_4 @ world_view_transform_light
            x_min, y_min, z_min = xyz_ortho_scene[:, 0].min(), xyz_ortho_scene[:, 1].min(), xyz_ortho_scene[:, 2].min()
            x_max, y_max, z_max = xyz_ortho_scene[:, 0].max(), xyz_ortho_scene[:, 1].max(), xyz_ortho_scene[:, 2].max()
            extent_scene = torch.stack([
                x_max - x_min,
                y_max - y_min,
                z_max - z_min
            ])
            # print("x_max, y_max, z_max: ", x_max, y_max, z_max)
            # print("x_min, y_min, z_min: ", x_min, y_min, z_min)
            
            if light_box:
                """
                根据 mask 设置 mask
                """
                xyz_ortho_object = None
                # mask_sky_scene = mask_scene[~mask_sky]
                # xyz_ortho_object = xyz_ortho_scene[~mask_sky_scene]
                try:
                    x_min, y_min, z_min = xyz_ortho_object[:, 0].min(), xyz_ortho_object[:, 1].min(), xyz_ortho_object[:, 2].min()
                    x_max, y_max, z_max = xyz_ortho_object[:, 0].max(), xyz_ortho_object[:, 1].max(), xyz_ortho_object[:, 2].max()
                    extent_object = torch.stack([
                            x_max - x_min,
                            y_max - y_min,
                            z_max - z_min 
                            ])
                    # ratio = torch.sqrt( (extent_object / extent_scene).abs().max())
                    # print("ratio: ", ratio)
                    # print("x_max, y_max, z_max: ", x_max, y_max, z_max)
                    # print("x_min, y_min, z_min: ", x_min, y_min, z_min)
                except:
                    print("xyz_ortho_object is None, please set object_id and corrsponding mask")

            else:
                ratio = 1.0

                
            # padding_factor = 0.1 # 10% 扩展
            # x_min -= extent_scene[0] * padding_factor
            # y_min -= extent_scene[1] * padding_factor
            # z_min -= extent_scene[2] * padding_factor
            # x_max += extent_scene[0] * padding_factor
            # y_max += extent_scene[1] * padding_factor
            # z_max += extent_scene[2] * padding_factor
            light_orthoproj_matrix = getOrthographicMatrixFromBounds(
                xmin=x_min,
                xmax=x_max,
                ymin=y_min,
                ymax=y_max,
                zmin=z_min,
                zmax=z_max
            ).transpose(0, 1).cuda()
    else:
        ratio = 1.0
        
    
    if not ortho:
        light_persp_proj_matrix = getProjectionMatrix(znear=viewpoint_camera.znear, zfar=viewpoint_camera.zfar, fovX=fovx_far, fovY=fovy_far).transpose(0,1).cuda()
        full_persp_proj_transform_light = (world_view_transform_light.unsqueeze(0).bmm(light_persp_proj_matrix.unsqueeze(0))).squeeze(0)
        full_ortho_proj_transform_light = None
    else:
        full_ortho_proj_transform_light = (world_view_transform_light.unsqueeze(0).bmm(light_orthoproj_matrix.unsqueeze(0))).squeeze(0)
        full_persp_proj_transform_light = None



    d3_ortho_cov = None
    
    # python 实现正交投影计算协方差
    if False:    
        d3_ortho_cov = computeCov2D_ortho_python(xyz, viewpoint_camera.image_width, viewpoint_camera.image_height, \
        light_orthoproj_matrix, fx_far, fy_far, gau.get_scaling,gau.get_rotation, world_view_transform_light, scaling_modifier,\
        full_ortho_proj_transform_light)


        
        
        
    # 设置光源的高斯泼溅参数


    with light_stream_ctx:
        # 3）光源方向的高斯泼溅 ———— 高斯场景准备工作: 
        # 1）视角方向的高斯泼溅 ———— 高斯场景准备工作:

        # 获得高斯点的 3D坐标、透明度、并初始化 2D坐标
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        means3D = gau.get_xyz
        screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda")
        opacity = gau.get_opacity
        if gau.use_hgs:
            hgs_normals = gau.get_hgs_normals
            hgs_opacities = gau.get_hgs_opacities
        else:
            hgs_normals = None
            hgs_opacities = None



        # 计算高斯点的方差:
        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3Ds_precomp = None
        if pipe.compute_cov3D_python:
            cov3Ds_precomp = gau.get_covariance(scaling_modifier)
        else:
            scales = gau.get_scaling
            rotations = gau.get_rotation
        shadow_stage = get_shadow_backward_stage(modelset, iteration)
        shadow_means3D = _shadow_stage_tensor(means3D, shadow_stage["xyz"])
        shadow_opacity = _shadow_stage_tensor(opacity, shadow_stage["opacity"])
        shadow_scales = _shadow_stage_tensor(scales, shadow_stage["scaling"])
        shadow_rotations = _shadow_stage_tensor(rotations, shadow_stage["rotation"])
        if pipe.compute_cov3D_python:
            shadow_cov3Ds_precomp = gau.covariance_activation(
                shadow_scales if shadow_scales is not None else gau.get_scaling,
                scaling_modifier,
                shadow_rotations if shadow_rotations is not None else gau.get_rotation,
            )
        else:
            shadow_cov3Ds_precomp = cov3Ds_precomp
        if d3_ortho_cov is not None:
            shadow_d3_ortho_cov = d3_ortho_cov
            if not (shadow_stage["xyz"] and shadow_stage["scaling"] and shadow_stage["rotation"]):
                shadow_d3_ortho_cov = d3_ortho_cov.detach()
        else:
            shadow_d3_ortho_cov = None

        # 标准来看会造成死锁，但是由于每次循环三个流基本同步，light_stream 在 default 等待light前抵达，因此不会造成死锁
        # 不建议
        if light_stream is not None:
            light_stream.wait_stream(torch.cuda.default_stream())


        # 3）光源方向的高斯泼溅 ———— 光源方向高斯泼溅信息计算: 

        # shadow splatting
           # 等待计算完成，因为下面要使用新的流，且有依赖   
        if gau.use_MBRDF:
        
            opacity_light = torch.zeros([scales.shape[0],1], dtype=torch.float32, device='cuda')

            assert viewpoint_camera.pl_pos.shape[0] == 1 # (1, 3)
            # calculate view and light dirs
            pl_pos_expand = viewpoint_camera.pl_pos.expand(gau.get_xyz.shape[0], -1) # (N, 3)
            wi_ray = pl_pos_expand - gau.get_xyz # (N, 3)
            # Keep light-direction normalization finite even if a Gaussian gets
            # extremely close to the point light during training.
            wi_dist2 = torch.sum(wi_ray**2, dim=-1, keepdim=True).clamp_min(1e-12)
            dist_2_inv = 1.0 / wi_dist2
            wi = wi_ray * torch.sqrt(dist_2_inv) # (N, 3)
            # 归一化视角方向
            camera_center_for_brdf = viewpoint_camera.camera_center
            if (
                str(getattr(gau, "rasterizer", "")).startswith("2dgs")
                and os.getenv("GS3_2DGS_DETACH_VIEWDIR", "0") == "1"
            ):
                camera_center_for_brdf = camera_center_for_brdf.detach()
            wo = _safe_normalize(camera_center_for_brdf - gau.get_xyz) # (N, 3) 

            encoding_pos = None

            hgs_opacities_shadow = torch.zeros_like(hgs_opacities, dtype=torch.float32, device='cuda') if gau.use_hgs else None
            hgs_opacities_light = torch.zeros_like(hgs_opacities, dtype=torch.float32, device='cuda') if gau.use_hgs else None

            W_light = int(ratio * viewpoint_camera.image_width)
            H_light = int(ratio * viewpoint_camera.image_height)
            W = int(viewpoint_camera.image_width)
            H = int(viewpoint_camera.image_height)

            light_object_ids = torch.full(
                (means3D.shape[0], 3),
                2.0,
                dtype=torch.float32,
                device="cuda",
            )
            if object_id is not None:
                object_id_flat = object_id.reshape(-1).to(dtype=torch.float32, device="cuda")
                if object_id_flat.shape[0] < means3D.shape[0]:
                    raise ValueError(
                        f"object_id has {object_id_flat.shape[0]} entries, "
                        f"but the light rasterizer expects {means3D.shape[0]}"
                    )
                light_object_ids[:, 0] = object_id_flat[: means3D.shape[0]]

            if shadow_map:
                
                raster_settings = v_3dgs_settings(
                    image_height = H_light,
                    image_width = W_light,
                    tanfovx = tanfovx_far,
                    tanfovy = tanfovy_far,
                    bg = bg_color[:3],
                    scale_modifier = scaling_modifier,
                    viewmatrix = world_view_transform_light,
                    projmatrix = full_ortho_proj_transform_light if ortho else full_persp_proj_transform_light,
                    sh_degree = gau.active_sh_degree,
                    campos = viewpoint_camera.pl_pos[0],
                    prefiltered = False,
                    debug = pipe.debug,
                    antialiasing = False,
                    znear = viewpoint_camera.znear,
                    zfar = viewpoint_camera.zfar,
                    )
                
                rasterizer_inputs = {
                    # 高斯点相关
                    "means3D": shadow_means3D,
                    "means2D": screenspace_points,
                    "shs": None,
                    "colors_precomp": light_object_ids,
                    "opacities": shadow_opacity,
                    "scales": None if shadow_d3_ortho_cov is not None else shadow_scales,
                    "rotations": None if shadow_d3_ortho_cov is not None else shadow_rotations,
                    "cov3D_precomp": shadow_d3_ortho_cov if ortho else shadow_cov3Ds_precomp,
                    "shadow": None,
                }
                
                if ortho:
                    shadow_map_rasterizer = ortho_rasterizer(raster_settings=raster_settings )
                else:
                     shadow_map_rasterizer = persp_rasterizer(raster_settings=raster_settings )
                _, __, depth_image = shadow_map_rasterizer(**rasterizer_inputs)
                shadow = torch.ones_like(opacity, dtype=torch.float32, device='cuda')
                _maybe_cuda_sync()

                
            else:
                
                projmatrix = full_ortho_proj_transform_light if ortho else full_persp_proj_transform_light
                raster_settings_light = light_settings(
                    image_height = H_light,
                    image_width = W_light,
                    tanfovx = tanfovx_far,
                    tanfovy = tanfovy_far,
                    bg = bg_color[:3],
                    scale_modifier = scaling_modifier,
                    viewmatrix = world_view_transform_light,
                    projmatrix = projmatrix,
                    sh_degree = gau.active_sh_degree,
                    campos = viewpoint_camera.pl_pos[0],
                    prefiltered = False,
                    debug = pipe.debug,
                    low_pass_filter_radius = 0.3,
                    ortho = ortho
                    )
                
                rasterizer_light = light_rasterizer(raster_settings=raster_settings_light)
                light_inputs = {
                    # 高斯点相关
                    "means3D": shadow_means3D,
                    "means2D": screenspace_points,
                    "shs": None,
                    # colors_precomp or object_id
                    "colors_precomp": light_object_ids,
                    "opacities": shadow_opacity,
                    "scales": None if shadow_d3_ortho_cov is not None else shadow_scales,
                    "rotations": None if shadow_d3_ortho_cov is not None else shadow_rotations,
                    "cov3Ds_precomp": shadow_d3_ortho_cov if ortho else shadow_cov3Ds_precomp,
                    # 传递梯度用
                    "projmatrix": projmatrix,
                    "viewmatrix": world_view_transform_light,

                    # 阴影相关
                    "non_trans": opacity_light,
                    "offset": modelset.offset,
                    "thres": 4,

                    # prune 相关
                    "is_train": is_train,
                    
                    # hgs 相关
                    "hgs": gau.use_hgs,
                    "hgs_normals": hgs_normals,
                    "hgs_opacities": hgs_opacities,
                    "hgs_opacities_shadow": hgs_opacities_shadow, 
                    "hgs_opacities_light": hgs_opacities_light, 

                    # 流
                    "streams": None # 暂时没用，（用于内部多个流）

                }

                _, out_weight, _, shadow, non_trans, invdepths = rasterizer_light(**light_inputs)
                if torch.isnan(shadow).any():
                    print("shadow is nan")
                    assert False

    # 4）光源方向的高斯泼溅 ———— ① 计算最终阴影 ② 计算其他效果 ③ 计算高斯点的最终颜色:

    # MBDRF 和 SH 的选择
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if gau.use_MBRDF:
            with calc_stream_ctx:
                """
                使用计算颜色的流，和光源流同步，两者间不存在数据竞争
                """
                # local_axises 由 4元数 构建，因此高斯法线是归一化的
                local_z = local_axises[:, :, 2] # (N, 3)

                # transfer to local axis
                # 利用从 mbrdf 参数组中的得到的 local_axises 将 wi 和 wo 转移到对应的 高斯点-asg 坐标系 中
                """
                torch.einsum(): torch 计算的升级版，能够自定义很多种计算方式，例如：
                    - "Ki,Kij->Kj" 相当于 bmm
                    - "ik,kj->ij" 相当于 matmul
                    - "i,i->" 相当于 dot
                    - "ij->j" 相当于 torch.sum(A, dim=0)
                    等等 
                """
                # local矩阵是 列向量作为基，因此 wi 和 wo 需要左乘 local_axises 的逆矩阵
                # 而这里 wi 和 wo 是行向量，因此需要右乘 local_axises 的逆矩阵的转置，也就是 local_axises 的本身
                wi_local = torch.einsum('Ki,Kij->Kj', wi, local_axises) # (N, 3)
                wo_local = torch.einsum('Ki,Kij->Kj', wo, local_axises) # (N, 3)

                # shading functions:
                """
                该部分计算每个高斯点的直接光照贡献，不考虑阴影（遮挡）或全局光照（间接光照）的影响。
                计算基于入射光的漫反射（Lambertian 反射）和镜面反射（基于 ASG 近似的高光反射）。

                · asg: asg 的旋转矩阵用于调整 asg 在高斯点局部坐标系中的方向
                        - asg 的 z 轴为 高斯球面的中心，也就是高光最密集的方向
                · mbdrf: 不直接使用表面法线，而是用 半角向量 h 作为“微表面法线” 进行计算
                        - 通过计算 h 和 asg 的夹角，来控制高光强度和形状，最终得出 wo 方位的反射强度

                · colors_precomp: 它代表的是 光照强度(irradiance)，由漫反射和镜面反射的贡献相加计算得到
                · local_axises: 是 高斯点的局部坐标系，由每个高斯点的局部旋转矩阵 local_q 得到
                · local_z: local_axises 中的 z 轴，相当于每一个高斯点的法线
                · cosTheta: Lambertian 余弦项，考虑入射角度对光照强度的影响
                · dist_2_inv: 光源的距离的倒数，考虑光强的距离衰减（通常 光照强度 与距离平方 成反比）
                
                -> 因此使用 colors_precomp * cosTheta * dist_2_inv 来修正颜色
                """
                """
                类似于 relu 函数，但是当 x < 0 时，输出为 0.01 * (exp(x) - 1)，而不是 0
                这里使 cosTheta dot(nrm, wi) 可能为负，意味着角度大于 90°，但是我们仍然希望计算得到一个非负的值，因此使用 elu，并在后面使用 tmp 来修正
                ！！！为什么不用 softplus 函数？
                    - 因为 点乘 的范围最小值为 -1，因此 elu 的输出最小值为 0.01 * (exp(-1) - 1) = -0.069，在后面使用正的 tmp 来修正，从而实现值域大于等于 0
                    - 而 -1 在 softplus 中，输出为 0.318，显然太大了，因此使用 elu 更合理，当然也可以使用负的 tmp 来修正
                    - 而且 elu 可以对负数部分 用 alpha 来进一步调整，而 softplus 不能
                """
                cosTheta = _NdotWi(local_z, wi, torch.nn.ELU(alpha=0.01), 0.01)     # local_z, wi 都是朝外的，方向一致
                diffuse = gau.get_kd / math.pi      # (N, 1)
                asg_1 =   gau.asg_func(wi_local, wo_local, gau.get_alpha_asg, asg_scales, asg_axises)
                
            
            # calc_stream.wait_stream(light_stream)


            # 等待所有分流完成
            if light_stream is not None:
                current_stream.wait_stream(light_stream)
            if calc_stream is not None and calc_stream is not light_stream:
                current_stream.wait_stream(calc_stream)

            """
            每个半高斯点，都对应一个阴影值，因此阴影值的形状为 [2N,]
            有两种采取方式：
                1. 半高斯公用一个中心，传入 [N,2] 
                2. 每个半高斯点的求出新的中心，传入 [2N, 1] or [N, 2]
            """
            # shadow 为 0 表示完全被遮挡，1 表示完全不被遮挡
            # opacity_light: atomicAdd exp(power_j）    每个高斯点被其他高斯所遮挡影响的总密度
            # shadow: atomicAdd exp(power_j)*(T）       每个高斯点被其他高斯所遮挡影响的总阴影值
            if shadow_map:
                pass
            else:
                # shaodow splat values
                non_trans = torch.clamp_min(non_trans, 1e-6)    # 防止最小值为0，产生 NaN
                # 归一化阴影值，使其不受不透明度影响
                shadow = shadow / non_trans # [N, 1]  /  [N, 2]   
                # shadow[non_trans<1e-4] = 0
                assert not torch.isnan(shadow).any()
            
            # 神经网络优化 shadow 和 其他效果
            # neural components
            # 前期这里的得到的 shadow 和 other_effects 都是 0，因为此时没有开启神经网络优化
            detach_shadow = getattr(modelset, "detach_shadow", False)
            use_shadow_refine_mlp = getattr(modelset, "use_shadow_refine_mlp", True)
            shadow_for_phase = shadow.detach() if detach_shadow else shadow
            shadow2, other_effects, asg_3, __ = gau.neural_phasefunc(wi, wo, gau.get_xyz, gau.get_neural_material, \
                                                encoding_pos, None, \
                                                shadow_for_phase, asg_1, asg_mlp) # (N, 1), (N, 3)
            
            selected_decay = shadow2 if shadow2 is not None else shadow_for_phase
            if not use_shadow_refine_mlp:
                selected_decay = shadow_for_phase
            # detach_shadow should only stop gradients from the analytic shadow
            # hint back into the light rasterizer / Gaussian geometry. The final
            # rendered decay branch itself should remain trainable so the shadow
            # MLP can still learn from image-space supervision, matching the
            # original "shadow participates in forward but has no shadow backward"
            # behavior.
            decay = selected_decay

            specular = gau.get_ks * asg_3 # (N, 3)

            # 刚开始只考虑 漫反射，不考虑其他反射，优化高斯的基础颜色
            if fix_labert:
                colors_precomp = diffuse
            else:
                colors_precomp = diffuse + specular 
            # intensity decays with distance
            colors_precomp = colors_precomp * cosTheta * dist_2_inv    


            # combine all components，按通道拼接，等待传入视角方向的高斯泼溅
            colors_precomp = torch.concat([colors_precomp * inten_scale, decay, other_effects * dist_2_inv * inten_scale], dim=-1) # (N, 7/8)
            # colors_precomp = torch.cat([colors_precomp[ : , : 4], colors_precomp[ : , 5: 8]], dim=-1)
        
        # true 在python中计算 sh，否则在 cuda 中计算
        elif pipe.convert_SHs_python:
            # 获得 SH 的特征
            """
            gau.get_features:   features_dc = self._features_dc
                                features_rest = self._features_rest
                                return torch.cat((features_dc, features_rest), dim=1) 
                                输入：(N, 1, C) + (N, D-1, C)
                                输出：(N, D, C)
            transpose:          (N, D, C)   ->  (N, C, D)
            view:               (N, C, D)   ->  (N, C, D)   
            """
            shs_view = gau.get_features.transpose(1, 2).contiguous().view(-1, 3, (gau.max_sh_degree+1)**2)
            dir_pp = (gau.get_xyz - viewpoint_camera.camera_center.repeat(gau.get_features.shape[0], 1))
            # 归一化方向
            """
            为什么要加 keepdim = True?
            因为:
                广播只能向前广播，比如 (x) -> (1, x) -> (N, x)
                不能向后广播，比如 (x) -> (N, 1) -> (N, x)
            """
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            # 计算 SH 到 RGB 的转换
            sh2rgb = eval_sh(gau.active_sh_degree, shs_view, dir_pp_normalized)
            # 将 sh2rgb [-0.5, 0.5] -> [0, 1]，并且截断不合理的值
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        
        else:
            # 不使用 SH，pass
            shs = gau.get_features
    
    # 如果 override_color 不为 None，则使用 override_color 作为颜色
    else:
        colors_precomp = override_color



    # 2）视角方向的高斯泼溅 ———— 高斯场景泼溅:

    # 等待所有流完成
    _maybe_cuda_sync()

    # 作用：确保 means2d 在反向传播时保留梯度
    

    # 3dgs 原代码

    transmat_grad_holder = None

    # if gau.rasterizer == "3dgs":
    #     #print("\ndebug: current rasterizer: 3dgs")

    #     means2D = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda")
    #     raster_settings = basic_settings(
    #         image_height=int(viewpoint_camera.image_height),
    #         image_width=int(viewpoint_camera.image_width),
    #         tanfovx=tanfovx,
    #         tanfovy=tanfovy,
    #         bg=bg_color,
    #         scale_modifier=scaling_modifier,
    #         # 这里的转换矩阵，行为基，在 camera.py 中提前转置了
    #         viewmatrix=viewpoint_camera.world_view_transform,
    #         projmatrix=viewpoint_camera.full_proj_transform,
    #         sh_degree=gau.max_sh_degree,
    #         campos=viewpoint_camera.camera_center,
    #         prefiltered=False,
    #         debug=pipe.debug,
    #         low_pass_filter_radius=0.3,
    #     )

    #     rasterizer = basic_rasterizer(raster_settings=raster_settings)


    #     # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    #     # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.

    #     # means3D1, scales1, opacities1, colors_precomp1 = GradientScaler.apply(means3D, scales, opacity, colors_precomp)



    #     rasterizer_inputs = {
    #         # 高斯点相关
    #         "means3D": means3D,
    #         "means2D": screenspace_points,
    #         "shs": None,
    #         "colors_precomp": colors_precomp,
    #         "opacities": opacity,
    #         "scales": scales,
    #         "rotations": rotations,
    #         "cov3Ds_precomp": cov3Ds_precomp,
            
    #         # hgs 相关
    #         "hgs": gau.use_hgs,
    #         "hgs_normals": hgs_normals,
    #         "hgs_opacities": hgs_opacities
    #     }

        
    #     # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    #     rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(**rasterizer_inputs)

    
    if gau.rasterizer in ("2dgs", "2dgs_3ch"):
        means2D = screenspace_points
        scales_2dgs = scales[..., :2] if scales is not None and scales.numel() > 0 else scales
        transmat_grad_holder = None
        if getattr(viewpoint_camera, "cam_pose_adj", None) is not None and viewpoint_camera.cam_pose_adj.requires_grad:
            transmat_grad_holder = torch.zeros(
                (means3D.shape[0], 9),
                dtype=torch.float32,
                device=means3D.device,
                requires_grad=True,
            )

        if gau.rasterizer == "2dgs" and colors_precomp is not None and colors_precomp.shape[-1] >= 7:
            surfel_bg = bg_color
            if surfel_bg.shape[0] == 3:
                surfel_bg = torch.cat(
                    [surfel_bg, torch.zeros(4, dtype=surfel_bg.dtype, device=surfel_bg.device)],
                    dim=0,
                )
            raster_settings_2dgs = _build_2dgs_raster_settings(
                viewpoint_camera, pipe, surfel_bg, scaling_modifier, gau.active_sh_degree
            )
            rasterizer_2dgs = surfel_rasterizer_deferred(raster_settings=raster_settings_2dgs)
            rendered_2dgs, radii, allmap = rasterizer_2dgs(
                means3D=means3D,
                means2D=means2D,
                opacities=opacity,
                shs=None,
                colors_precomp=colors_precomp,
                scales=scales_2dgs,
                rotations=rotations,
                cov3D_precomp=cov3Ds_precomp,
                texture_color=None,
                texture_alpha=None,
                use_textures=False,
                transmat_grad_holder=transmat_grad_holder,
            )
            rendered_image = rendered_2dgs
            if shadow_map:
                render_alpha = allmap[1:2].clamp_min(1e-8)
                depth_image = torch.nan_to_num(allmap[0:1] / render_alpha, 0, 0)
                rendered_image = torch.cat([rendered_2dgs, depth_image], dim=0)
        else:
            if colors_precomp is not None and colors_precomp.shape[-1] >= 7:
                colors_precomp_3ch = colors_precomp[:, 0:3] * colors_precomp[:, 3:4] + colors_precomp[:, 4:7]
            else:
                colors_precomp_3ch = colors_precomp
            raster_settings_2dgs = _build_2dgs_raster_settings(
                viewpoint_camera, pipe, bg_color, scaling_modifier, gau.active_sh_degree
            )
            rasterizer_2dgs = surfel_rasterizer(raster_settings=raster_settings_2dgs)
            rendered_rgb, radii, allmap = rasterizer_2dgs(
                means3D=means3D,
                means2D=means2D,
                opacities=opacity,
                shs=shs if colors_precomp_3ch is None else None,
                colors_precomp=colors_precomp_3ch,
                scales=scales_2dgs,
                rotations=rotations,
                cov3D_precomp=cov3Ds_precomp,
                texture_color=None,
                texture_alpha=None,
                use_textures=False,
                transmat_grad_holder=transmat_grad_holder,
            )
            rendered_image = torch.cat(
                [
                    rendered_rgb,
                    torch.ones(
                        (1, rendered_rgb.shape[1], rendered_rgb.shape[2]),
                        dtype=rendered_rgb.dtype,
                        device=rendered_rgb.device,
                    ),
                    torch.zeros(
                        (3, rendered_rgb.shape[1], rendered_rgb.shape[2]),
                        dtype=rendered_rgb.dtype,
                        device=rendered_rgb.device,
                    ),
                ],
                dim=0,
            )
            if shadow_map:
                render_alpha = allmap[1:2].clamp_min(1e-8)
                depth_image = torch.nan_to_num(allmap[0:1] / render_alpha, 0, 0)
                rendered_image = torch.cat([rendered_image, depth_image], dim=0)

        try:
            means2D.retain_grad()
        except:
            pass

    if gau.rasterizer == "gsplat":
        
        focalx = fov2focal(viewpoint_camera.FoVx, viewpoint_camera.image_width)
        focaly = fov2focal(viewpoint_camera.FoVy, viewpoint_camera.image_height)
        K = torch.tensor([[focalx, 0, viewpoint_camera.cx], [0, focaly, viewpoint_camera.cy], [0., 0., 1.]], device="cuda")

        splat_inputs = {
            "means": means3D, # [N, 3]
            "quats": rotations, # [N, 4]
            "scales": scales, # [N, 3]
            "opacities": opacity.squeeze(-1), # [N]
            "colors": colors_precomp, # [N, 7]
            "viewmats": viewpoint_camera.world_view_transform.transpose(0, 1)[None, ...], # [1, 4, 4]
            "Ks": K[None, ...], # [1, 3, 3]
            "width": int(viewpoint_camera.image_width),
            "height": int(viewpoint_camera.image_height),
            "near_plane": viewpoint_camera.znear,
            "far_plane": viewpoint_camera.zfar,
            "eps2d": 0.3,
            "sh_degree": None,
            "packed": False,
            "backgrounds": bg_color[None, ...]   # [1, 7]
            } 
        
        # 启动 expected_depth
        if shadow_map:
            splat_inputs["render_mode"] = "RGB+ED"
            
        #（这里使用的 gsplat 库，没有用原装 3dgs 的库）
        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        # 传入之前根据光照角度得到的 colors_precomp，进行当前视角的高斯泼溅
        rendered_image, alphas, meta = rasterization(**splat_inputs)
        # The intermediate results from fully_fused_projection
        # (H, W, C) → (C, H, W)，即 (RGB 通道数，高度，宽度)
        rendered_image = rendered_image[0].permute(2, 0, 1)
        radii = meta['radii'].squeeze(0)
        means2D = meta["means2d"]

        try:
            """
            torch.retain_grad() 作用：保留梯度，使得在反向传播时，该变量的梯度不会被释放 
                - PyTorch 默认只存叶子节点 (即参数） 的 grad
                - 因此即便中间变量的 grad 为 True，也不会保留，因为它不会被用于更新参数
                # intermediate_variable.is_leaf = False
                - 但是有时候我们需要中间变量的 grad，比如我们需要计算中间变量的梯度，或者我们需要中间变量的梯度来更新参数
                -> 因此，我们可以使用 retain_grad() 来保留中间变量的梯度
            """
            means2D.retain_grad() # [1, N, 2]
        except:
            pass


    _maybe_cuda_sync()

    #print("time", before.elapsed_time(after))
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.

    
    
    
    
    keep_backward_info = _env_enabled("SSGS_BACKWARD_INFO")
    backward_info = {}
    # 保留所有需要梯度的变量
    for var_name, var in ([
        ("means3D", means3D), ("scales", scales), ("rotations", rotations),
        ("colors_precomp", colors_precomp), ("opacity", opacity), ("diffuse", diffuse),
        ("specular", specular), ("cosTheta", cosTheta), ("dist_2_inv", dist_2_inv),
        ("decay", decay), ("other_effects", other_effects), ("asg_3", asg_3),
        ("asg_1", asg_1), ("shadow", shadow), ("rendered_image", rendered_image),
        ("radii", radii), ("means2D", means2D)
    ] if keep_backward_info else []):
        try:
            backward_info[var_name] = var
            var.retain_grad()
        except:
            pass

    if shadow_map:        
        expected_depth = rendered_image[7, :, :].detach() if shadow_map else None
        x_pix = torch.arange(W, device="cuda")
        y_pix = torch.arange(H, device="cuda")
        fx, fy = focalx, focaly
        cx, cy = viewpoint_camera.cx, viewpoint_camera.cy
        x_dir = ((x_pix + 0.5) - cx) / fx      # (W,)
        y_dir = ((y_pix + 0.5) - cy) / fy      # (H,)
        X = x_dir.unsqueeze(0) * expected_depth     # (1,W) * (H,W) -> (H,W)
        Y = y_dir.unsqueeze(1) * expected_depth     # (H,1) * (H,W) -> (H,W)
        Z = expected_depth 
        points_cam = torch.stack([X, Y, Z], dim=-1)
        # homo 
        points_cam = torch.cat([points_cam, torch.ones_like(Z).unsqueeze(-1)], dim=-1)
        pts_world = points_cam @ viewpoint_camera.world_view_transform.inverse()
        pts_light = pts_world @ full_ortho_proj_transform_light
        pts_light_ndc = pts_light[ : , : , : 2] / pts_light[ : , : , 3:4]
        pts_light_pix_w = ((pts_light_ndc[...,0] + 1.0) * W_light - 1.0) * 0.5
        pts_light_pix_h = ((pts_light_ndc[...,1] + 1.0) * H_light - 1.0) * 0.5
        dpts_light_depth = pts_light[ : , : , 2] 
        smooth = True
        if smooth: 
            u_norm = pts_light_pix_w.div(W_light-1).mul(2).sub(1)  # [-1,1]
            v_norm = pts_light_pix_h.div(H_light-1).mul(2).sub(1)  # [-1,1]
            grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0) 
            shadow_map_4d = depth_image.squeeze().unsqueeze(0).unsqueeze(0)
            sampled = F.grid_sample(
                shadow_map_4d,               # (1,1,H_light,W_light) depth map
                grid,                        # normalized coords
                mode='bilinear',             # bilinear interpolation of the *depth* texels
                padding_mode='border',
                align_corners=True
            )
            stored_depth = sampled.squeeze(0).squeeze(0)
            if debug:
                print("stored_depth", stored_depth)
        else:
            pts_light_pix_h = pts_light_pix_h.round().clamp(0, H_light-1).long()
            pts_light_pix_w = pts_light_pix_w.round().clamp(0, W_light-1).long()
            flat_idx   = (pts_light_pix_h * W_light + pts_light_pix_w).view(-1)     # (H*W,)
            depth_flat = depth_image.contiguous().view(-1)                         # (H_light*W_light,)
            stored_depth = depth_flat[flat_idx]                                     # (H*W,)
            stored_depth = stored_depth.view(pts_light_pix_h.shape)                # (H, W)
        bias = 0.1
        mash_nan = stored_depth.isnan().unsqueeze(0)
        shadow_mask = (dpts_light_depth - bias) <= stored_depth
        shadow_map_shadow = shadow_mask.float().unsqueeze(0)
        shadow_map_shadow[mash_nan] = 0
        # shadow_map_shadow += final_T
        shadow_map_shadow = torch.clamp(shadow_map_shadow, 0, 1)
        
    # precomp 的通道顺序：
    # 1. 高斯颜色 [N, 3]
    # 2. 阴影 [N, 1]
    # 3. 其他效果 [N, 3]
    # 返回渲染结果，根据不同的通道进行切分，获得各个部分的高斯泼溅结果
    """
    !!!  colors_precomp = torch.concat
    ([colors_precomp * inten_scale, decay, other_effects * dist_2_inv * inten_scale], dim=-1) # (N, 7)
    """
    output_package = {"render": rendered_image[0:3, :, :],
            "shadow": shadow_map_shadow if shadow_map else rendered_image[3:4, :, :],
            "other_effects": rendered_image[4:7, :, :],
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,     # [N]
            # 权重值，由不透明度（累计得到的），2d覆盖范围，深度等信息得到的
            # 例如：
            # - 一个完全不透明、离相机很近、覆盖多个像素的点 -> 大权重
            # - 一个半透明、离相机远、只覆盖一个像素的点 -> 小权重
            "out_weight": out_weight if not shadow_map else torch.zeros([means3D.shape[0], 1], dtype=torch.float32, device='cuda'),
            "backward_info": backward_info,       # [N, 1]
            "shadow_stage": shadow_stage,
            "transmat_grad_holder": transmat_grad_holder,
            #"asg3": asg_3                 # [N, asg_channel_num]
            "expected_depth": rendered_image[7:8, :, :] if shadow_map else None,
            "depth_image": depth_image if shadow_map else None
            } 
    
    return output_package

def _dot(x, y):
    return torch.sum(x * y, -1, keepdim=True)

def _safe_normalize(x):
    return torch.nn.functional.normalize(x, dim = -1, eps=1e-8)

def _NdotWi(nrm, wi, elu, a):
    """
    nrm: (N, 3)
    wi: (N, 3)
    _dot(nrm, wi): (N, 1)
    return (N, 1)
    """
    tmp  = a * (1. - 1 / math.e)
    return (elu(_dot(nrm, wi)) + tmp) / (1. + tmp)
