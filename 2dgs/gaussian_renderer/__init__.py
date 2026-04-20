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

import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from gaussian_renderer.textured import rasterize_with_texture_module, TextureRenderInputs

from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_surfel_rasterization_deferred import GaussianRasterizationSettings as GaussianRasterizationSettings_7ch, GaussianRasterizer as GaussianRasterizer_7ch
from scene.gaussian_model import GaussianModel
from utils.point_utils import depth_to_normal

_SHADOW_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "submodules", "diff-surfel-rasterization-shadow")
)
if os.path.isdir(_SHADOW_ROOT) and _SHADOW_ROOT not in sys.path:
    sys.path.insert(0, _SHADOW_ROOT)

from diff_surfel_rasterization_shadow import (  # noqa: E402
    GaussianRasterizationSettings as ShadowRasterSettings,
    GaussianRasterizer as ShadowRasterizer,
)
from utils.graphics_utils import getProjectionMatrix


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


def _build_light_transform(viewpoint_camera, means3d, pipe):
    """Build the light-camera transform used by the shadow pass.

    Default behavior mirrors the working gs3 path:
    - perspective light camera
    - focal length scaled from the view/light distance ratio

    An orthographic branch is still supported for explicit experiments only.
    """
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    fx_origin = viewpoint_camera.image_width / (2.0 * tanfovx)
    fy_origin = viewpoint_camera.image_height / (2.0 * tanfovy)

    object_center = means3d.mean(dim=0).detach().cpu().numpy()
    light_position = viewpoint_camera.pl_pos.detach().cpu().numpy().reshape(-1)
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

    ortho = bool(getattr(pipe, "shadow_ortho", False))
    if ortho:
        xyz_h = torch.cat([means3d, torch.ones_like(means3d[:, :1])], dim=1)
        xyz_ortho = xyz_h @ world_view_transform_light
        x_min, y_min, z_min = xyz_ortho[:, 0].min(), xyz_ortho[:, 1].min(), xyz_ortho[:, 2].min()
        x_max, y_max, z_max = xyz_ortho[:, 0].max(), xyz_ortho[:, 1].max(), xyz_ortho[:, 2].max()

        light_ortho_proj_matrix = _get_orthographic_matrix_from_bounds(
            xmin=x_min, xmax=x_max, ymin=y_min, ymax=y_max, zmin=z_min, zmax=z_max,
        ).transpose(0, 1).cuda()
        light_projmatrix = (
            world_view_transform_light.unsqueeze(0).bmm(light_ortho_proj_matrix.unsqueeze(0))
        ).squeeze(0)
    else:
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
        ortho=ortho,
    )


def _compute_shadow_pass(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0):
    """Compute shadow map using the 2DGS surfel shadow rasterizer.

    With texture:    per_uv_shadow [N, R, R]  (per UV texel)
    Without texture: per_point_shadow [N, 1]  (per Gaussian)

    Returns a dict with:
        per_point_shadow  – [N, 1]  (mean over UV map, used as mBRDF hint)
        per_uv_shadow     – [N, R, R] or None
        light_viewmatrix / light_projmatrix
    """
    if viewpoint_camera.pl_pos is None or pc.get_xyz.numel() == 0:
        return None

    means3d = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    lt = _build_light_transform(viewpoint_camera, means3d, pipe)

    # With texture: use real per-UV texture_alpha → per_uv_shadow [N, R, R]
    # Without texture: use 1×1 uniform opacity as texture_alpha → per_uv_shadow [N, 1, 1]
    # Both paths use ShadowRasterizer (2DGS surfel rasterizer, not 3DGS)
    use_textures = bool(getattr(pc, "use_textures", False))

    if use_textures:
        texture_alpha = pc.get_texture_alpha  # [N, 1, R, R] → kernel uses per-UV path
    else:
        texture_alpha = torch.empty(0, device="cuda")  # texture_resolution=0 → kernel uses per-Gaussian path

    shadow_settings = ShadowRasterSettings(
        image_height=lt["h_light"],
        image_width=lt["w_light"],
        tanfovx=lt["tanfovx_far"],
        tanfovy=lt["tanfovy_far"],
        bg=bg_color[:3],
        scale_modifier=scaling_modifier,
        viewmatrix=lt["world_view_transform_light"],
        projmatrix=lt["light_projmatrix"],
        sh_degree=pc.active_sh_degree,
        campos=torch.tensor(lt["light_position"], dtype=torch.float32, device="cuda"),
        prefiltered=False,
        debug=getattr(pipe, "debug", False),
        low_pass_filter_radius=0.3,
        ortho=lt["ortho"],
        use_textures=use_textures,
    )
    shadow_rasterizer = ShadowRasterizer(raster_settings=shadow_settings)
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
        texture_alpha=texture_alpha,
        texture_sigma_factor=getattr(pc, "texture_sigma_factor", 3.0),
        non_trans=None,
        offset=getattr(pipe, "shadow_offset", 0.015),
        thres=-1.0,
        is_train=False,
    )
    non_trans_safe = torch.clamp_min(non_trans, 1e-6)
    N = means3d.shape[0]

    if use_textures:
        # out_trans / non_trans: [N*R*R] flat → reshape to [N, R, R]
        R = int(round((out_trans.numel() / max(N, 1)) ** 0.5))
        per_uv_shadow = out_trans.view(N, R, R) / non_trans_safe.view(N, R, R)  # [N, R, R]
        per_point_shadow = per_uv_shadow.mean(dim=(-2, -1), keepdim=False).unsqueeze(-1)  # [N, 1]
    else:
        # out_trans / non_trans: [N] flat → per-Gaussian shadow
        per_uv_shadow = None
        per_point_shadow = (out_trans / non_trans_safe).unsqueeze(-1)  # [N, 1]

    return {
        "per_point_shadow": per_point_shadow,
        "per_uv_shadow": per_uv_shadow,
        "light_viewmatrix": lt["world_view_transform_light"],
        "light_projmatrix": lt["light_projmatrix"],
    }


def _safe_normalize(vec):
    return F.normalize(vec, p=2, dim=-1, eps=1e-6)


def _ndotwi(normals, wi):
    import math
    a = 0.01
    tmp = a * (1.0 - 1.0 / math.e)
    return (F.elu(torch.sum(normals * wi, dim=-1, keepdim=True), alpha=a) + tmp) / (1.0 + tmp)


def _compute_mbrdf_colors(viewpoint_camera, pc: GaussianModel, shadow_pkg, fix_lambert: bool = False):
    """Compute mBRDF lighting components.

    Returns {"basecolor", "shadow", "other_effects"} where:
        basecolor    = (diffuse + specular_asg) * ndotwi * inv_d2   (NO shadow baked in)
        shadow       = decay from neural_phasefunc                  (optimised shadow [N,1] or [N,R,R])
        other_effects = residual from neural_phasefunc              [N,3]

    No tex : basecolor [N,3],     shadow [N,1]
    Tex    : basecolor [N,3,R,R], shadow [N,R,R]

    Direct path:  final = basecolor * shadow + other_effects  (computed in render())
    Deferred path: 7-ch concat → rasterizer → split & compose  (TODO)
    """
    if viewpoint_camera.pl_pos is None:
        return None

    dev = pc.get_xyz.device
    N = pc.get_xyz.shape[0]

    pl_pos = viewpoint_camera.pl_pos
    if not isinstance(pl_pos, torch.Tensor):
        pl_pos = torch.tensor(pl_pos, dtype=torch.float32)
    pl_pos = pl_pos.to(dev)
    if pl_pos.ndim == 1:
        pl_pos = pl_pos.unsqueeze(0)

    cam_center = viewpoint_camera.camera_center
    if not isinstance(cam_center, torch.Tensor):
        cam_center = torch.tensor(cam_center, dtype=torch.float32)
    cam_center = cam_center.to(dev)

    # ── per-Gaussian geometry ──────────────────────────────────────────────
    pl_pos3 = pl_pos[0].unsqueeze(0).expand(N, -1)       # [N, 3]
    wi_ray = pl_pos3 - pc.get_xyz                         # [N, 3]
    wi_dist2 = wi_ray.pow(2).sum(-1, keepdim=True).clamp_min(1e-12)  # [N, 1]
    wi = wi_ray * wi_dist2.rsqrt()                        # [N, 3]
    if getattr(viewpoint_camera, "cam_pose_adj", None) is not None and viewpoint_camera.cam_pose_adj.requires_grad:
        cam_center = cam_center.detach()
    wo = _safe_normalize(cam_center - pc.get_xyz)         # [N, 3]

    local_axises = pc.get_local_axis                      # [N, 3, 3]
    local_z = local_axises[:, :, 2]                       # [N, 3]
    wi_local = torch.einsum("ki,kij->kj", wi, local_axises)
    wo_local = torch.einsum("ki,kij->kj", wo, local_axises)

    diffuse = pc.get_kd / math.pi                         # [N, 3]
    asg_scales = pc.asg_func.get_asg_lam_miu
    asg_axises = pc.asg_func.get_asg_axis
    asg_1 = pc.asg_func(wi_local, wo_local, pc.get_alpha_asg, asg_scales, asg_axises)

    ndotwi = _ndotwi(local_z, wi)                         # [N, 1]
    inv_d2 = 1.0 / wi_dist2                               # [N, 1]

    # ── tex path: phasefunc per-UV ─────────────────────────────────────────
    per_uv_shadow = None if shadow_pkg is None else shadow_pkg.get("per_uv_shadow")
    has_tex = per_uv_shadow is not None  # [N, R, R]

    if has_tex:
        R = per_uv_shadow.shape[-1]
        NRR = N * R * R

        hint_flat = per_uv_shadow.reshape(NRR, 1)         # [N*R*R, 1]

        def _expand(t):
            D = t.shape[-1]
            return t[:, None, None, :].expand(N, R, R, D).reshape(NRR, D)

        decay_flat, oe_flat, _, _ = pc.neural_phasefunc(
            _expand(wi), _expand(wo), _expand(pc.get_xyz), _expand(pc.get_neural_material),
            hint=hint_flat, asg_1=_expand(asg_1), asg_mlp=pc.asg_mlp,
        )
        if decay_flat is None:
            decay_flat = torch.ones((NRR, 1), dtype=torch.float32, device=dev)

        shadow = decay_flat.reshape(N, R, R)              # [N, R, R]

        # basecolor per-UV: texture_color * ndotwi * inv_d2  (ASG high-freq specular still per-Gaussian here)
        texture_color = pc.get_texture_color              # [N, 3, R, R]
        basecolor = texture_color * ndotwi[:, :, None, None] * inv_d2[:, :, None, None]  # [N, 3, R, R]

        if oe_flat is not None:
            wi_dist2_flat = _expand(wi_dist2)
            other_effects = (oe_flat * (1.0 / wi_dist2_flat)).reshape(N, R, R, 3).mean(dim=(1, 2))  # [N,3]
        else:
            other_effects = None

        return {"basecolor": basecolor, "shadow": shadow, "other_effects": other_effects}

    # ── no-tex path: phasefunc per-Gaussian ───────────────────────────────
    shadow_hint = None if shadow_pkg is None else shadow_pkg["per_point_shadow"]

    decay, other_effects, asg_3, _ = pc.neural_phasefunc(
        wi, wo, pc.get_xyz, pc.get_neural_material,
        hint=shadow_hint, asg_1=asg_1, asg_mlp=pc.asg_mlp,
    )
    if decay is None:
        decay = torch.ones((N, 1), dtype=torch.float32, device=dev)

    if fix_lambert:
        basecolor = diffuse * ndotwi * inv_d2              # [N, 3]
    else:
        specular = pc.get_ks * asg_3
        basecolor = (diffuse + specular) * ndotwi * inv_d2  # [N, 3]

    if other_effects is not None:
        other_effects = other_effects * inv_d2

    return {"basecolor": basecolor, "shadow": decay, "other_effects": other_effects}


def _build_output_dict(means2D, radii, rendered_image, allmap, viewpoint_camera, pc, pipe, shadow_pkg, transmat_grad_holder, rendered_split=None):
    """Pack rasterizer outputs into the standard return dict."""
    render_alpha = allmap[1:2]
    render_normal = (
        allmap[2:5].permute(1, 2, 0) @ viewpoint_camera.world_view_transform[:3, :3].T
    ).permute(2, 0, 1)
    render_depth_median  = torch.nan_to_num(allmap[5:6], 0, 0)
    render_depth_expected = torch.nan_to_num(allmap[0:1] / render_alpha, 0, 0)
    surf_depth  = render_depth_expected * (1 - pipe.depth_ratio) + pipe.depth_ratio * render_depth_median
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth).permute(2, 0, 1) * render_alpha.detach()
    out = {
        "render":            rendered_image,
        "viewspace_points":  means2D,
        "visibility_filter": radii > 0,
        "radii":             radii,
        "rend_alpha":        render_alpha,
        "rend_normal":       render_normal,
        "rend_dist":         allmap[6:7],
        "surf_depth":        surf_depth,
        "surf_normal":       surf_normal,
        "pre_shadow":        shadow_pkg["per_point_shadow"] if shadow_pkg else None,
        "transmat_grad_holder": transmat_grad_holder,
    }
    if rendered_split is not None:
        out["render_base"] = rendered_split[0:3]
        out["render_shadow"] = rendered_split[3:4]
        out["render_other_effects"] = rendered_split[4:7]
    return out


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
           scaling_modifier=1.0, override_color=None, fix_lambert=False,
           apply_shadow=True, deferred=False, return_deferred_split=False):
    """Render the scene.

    Two rendering modes (controlled by ``deferred``):

    Direct  (deferred=False):
        Shadow is baked into per-Gaussian / per-UV colors before splatting.
        Single rasterizer pass → final image.

    Deferred (deferred=True):
        Pass 1 – splat basecolor (no shadow)  → basecolor_img [3,H,W]
        Pass 2 – splat other_effects          → effects_img   [3,H,W]
        Screen space composition: basecolor_img * shadow_img + effects_img

    Both modes support tex (diff-surfel-rasterization-texture) and
    no-tex (diff-surfel-rasterization) paths.

    fix_lambert:  True → diffuse-only BRDF (Phase 0/1/2)
    apply_shadow: False → skip shadow pass entirely (Phase 0 warmup)
    """

    # ══ Step 1: Shadow pass (light-space surfel splatting) ════════════════
    shadow_pkg = None
    if getattr(pipe, "shadow_pass", False) and apply_shadow:
        shadow_pkg = _compute_shadow_pass(viewpoint_camera, pc, pipe, bg_color, scaling_modifier)

    # ══ Step 2: Appearance parameterization (mBRDF vs SH) ════════════════════
    # mBRDF is the semantic model; SH is an alternative parameterization.
    # If pc.use_mbrdf=True, we always construct an mbrdf dict (Phase 0 uses decay=1, other=0).
    mbrdf = None
    use_mbrdf = getattr(pc, "use_mbrdf", False) and override_color is None
    if use_mbrdf:
        if not apply_shadow:
            # Phase 0: no phasefunc, no shadow. Keep mBRDF representation explicit.
            N = pc.get_xyz.shape[0]
            pl_pos = viewpoint_camera.pl_pos
            if not isinstance(pl_pos, torch.Tensor):
                pl_pos = torch.tensor(pl_pos, dtype=torch.float32)
            pl_pos = pl_pos.to(pc.get_xyz.device)
            if pl_pos.ndim == 1:
                pl_pos = pl_pos.unsqueeze(0)

            pl_pos3 = pl_pos[0].unsqueeze(0).expand(N, -1)
            wi_ray = pl_pos3 - pc.get_xyz
            wi_dist2 = wi_ray.pow(2).sum(-1, keepdim=True).clamp_min(1e-12)
            wi = wi_ray * wi_dist2.rsqrt()
            local_z = pc.get_local_axis[:, :, 2]
            ndotwi = _ndotwi(local_z, wi)
            inv_d2 = 1.0 / wi_dist2
            if getattr(pc, "use_textures", False):
                # basecolor comes from texture in render_textured; only provide scalar decay + zero other.
                mbrdf = {
                    "basecolor": None,
                    "shadow": torch.ones((N, 1), dtype=torch.float32, device=pc.get_xyz.device),
                    "other_effects": torch.zeros((N, 3), dtype=torch.float32, device=pc.get_xyz.device),
                }
            else:
                mbrdf = {
                    # Match gs3 Phase 0: even before shadow/phase learning starts,
                    # the diffuse warmup still uses the point-light geometry term.
                    "basecolor": (pc.get_kd / math.pi) * ndotwi * inv_d2,  # [N,3]
                    "shadow": torch.ones((N, 1), dtype=torch.float32, device=pc.get_xyz.device),
                    "other_effects": torch.zeros((N, 3), dtype=torch.float32, device=pc.get_xyz.device),
                }
        else:
            mbrdf = _compute_mbrdf_colors(viewpoint_camera, pc, shadow_pkg, fix_lambert=fix_lambert)

    # ══ Step 3: Rasterization ═══════════════════════════════════════════════
    colors_precomp = None
    rendered_image, radii, allmap, means2D, transmat_grad_holder, rendered_split = rasterize_with_texture_module(
        viewpoint_camera=viewpoint_camera,
        pc=pc,
        pipe=pipe,
        bg_color=bg_color,
        scaling_modifier=scaling_modifier,
        inputs=TextureRenderInputs(
            deferred=deferred,
            mbrdf = mbrdf,
            colors_precomp=colors_precomp,
            return_split=return_deferred_split,
        ),
    )
    return _build_output_dict(
        means2D, radii, rendered_image, allmap, viewpoint_camera, pc, pipe, shadow_pkg, transmat_grad_holder, rendered_split
    )
