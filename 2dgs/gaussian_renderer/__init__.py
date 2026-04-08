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

from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.point_utils import depth_to_normal
from utils.sh_utils import eval_sh

_SHADOW_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "submodules", "diff-surfel-rasterization-shadow")
)
if os.path.isdir(_SHADOW_ROOT) and _SHADOW_ROOT not in sys.path:
    sys.path.insert(0, _SHADOW_ROOT)

from diff_surfel_rasterization_shadow import (  # noqa: E402
    GaussianRasterizationSettings as ShadowRasterSettings,
    GaussianRasterizer as ShadowRasterizer,
)


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
    """Build world-to-light orthographic projection, returning matrices and tanfov values."""
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

    camera_position = viewpoint_camera.camera_center.detach().cpu().numpy() * getattr(pipe, "shadow_light_scale", 1.0)
    light_norm = max(float(np.sum(light_position * light_position)), 1e-8)
    camera_norm = max(float(np.sum(camera_position * camera_position)), 1e-8)
    f_scale_ratio = math.sqrt(light_norm / camera_norm) * 0.2
    fx_far = fx_origin * f_scale_ratio
    fy_far = fy_origin * f_scale_ratio

    xyz_h = torch.cat([means3d, torch.ones_like(means3d[:, :1])], dim=1)
    xyz_ortho = xyz_h @ world_view_transform_light
    x_min, y_min, z_min = xyz_ortho[:, 0].min(), xyz_ortho[:, 1].min(), xyz_ortho[:, 2].min()
    x_max, y_max, z_max = xyz_ortho[:, 0].max(), xyz_ortho[:, 1].max(), xyz_ortho[:, 2].max()

    light_ortho_proj_matrix = _get_orthographic_matrix_from_bounds(
        xmin=x_min, xmax=x_max, ymin=y_min, ymax=y_max, zmin=z_min, zmax=z_max,
    ).transpose(0, 1).cuda()
    full_ortho_proj_transform_light = (
        world_view_transform_light.unsqueeze(0).bmm(light_ortho_proj_matrix.unsqueeze(0))
    ).squeeze(0)

    tanfovx_far = 0.5 * viewpoint_camera.image_width / fx_far
    tanfovy_far = 0.5 * viewpoint_camera.image_height / fy_far
    resolution_scale = max(float(getattr(pipe, "shadow_resolution_scale", 1.0)), 0.25)
    h_light = min(max(32, int(resolution_scale * viewpoint_camera.image_height)), 2048)
    w_light = min(max(32, int(resolution_scale * viewpoint_camera.image_width)), 2048)

    return dict(
        world_view_transform_light=world_view_transform_light,
        full_ortho_proj_transform_light=full_ortho_proj_transform_light,
        tanfovx_far=tanfovx_far,
        tanfovy_far=tanfovy_far,
        h_light=h_light,
        w_light=w_light,
        light_position=light_position,
    )


def _compute_shadow_pass(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0):
    """Compute shadow map using the 2DGS surfel shadow rasterizer.

    With texture:    per_uv_shadow [N, R, R]  (per UV texel)
    Without texture: per_point_shadow [N, 1]  (per Gaussian)

    Returns a dict with:
        per_point_shadow  – [N, 1]  (mean over UV map, used as mBRDF hint)
        per_uv_shadow     – [N, R, R] or None
        shadow_image      – [1, H, W] rendered shadow channel
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
    has_tex_alpha = (
        getattr(pc, "use_textures", False)
        and hasattr(pc, "get_texture_alpha")
        and pc.get_texture_alpha is not None
        and pc.get_texture_alpha.numel() > 0
    )

    if has_tex_alpha:
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
        projmatrix=lt["full_ortho_proj_transform_light"],
        sh_degree=pc.active_sh_degree,
        campos=torch.tensor(lt["light_position"], dtype=torch.float32, device="cuda"),
        prefiltered=False,
        debug=getattr(pipe, "debug", False),
        low_pass_filter_radius=0.3,
        ortho=True,
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

    if has_tex_alpha:
        # out_trans / non_trans: [N*R*R] flat → reshape to [N, R, R]
        R = int(round((out_trans.numel() / max(N, 1)) ** 0.5))
        per_uv_shadow = torch.clamp(
            out_trans.view(N, R, R) / non_trans_safe.view(N, R, R), 0.0, 1.0
        )  # [N, R, R]
        per_point_shadow = per_uv_shadow.mean(dim=(-2, -1), keepdim=False).unsqueeze(-1)  # [N, 1]
    else:
        # out_trans / non_trans: [N] flat → per-Gaussian shadow
        per_uv_shadow = None
        per_point_shadow = torch.clamp(out_trans / non_trans_safe, 0.0, 1.0).unsqueeze(-1)  # [N, 1]

    shadow_img = _render_shadow_image(
        viewpoint_camera, pc, pipe, bg_color, scaling_modifier,
        per_point_shadow, tanfovx, tanfovy
    )
    return {
        "per_point_shadow": per_point_shadow.detach(),
        "per_uv_shadow": per_uv_shadow.detach() if per_uv_shadow is not None else None,
        "shadow_image": shadow_img,
        "light_viewmatrix": lt["world_view_transform_light"],
        "light_projmatrix": lt["full_ortho_proj_transform_light"],
    }


def _render_shadow_image(viewpoint_camera, pc, pipe, bg_color, scaling_modifier,
                         per_point_shadow, tanfovx, tanfovy):
    """Splat per_point_shadow [N,1] into a [1,H,W] image using the 2DGS rasterizer."""
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color[:3],
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=getattr(pipe, "debug", False),
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    shadow_color = per_point_shadow.expand(-1, 3).contiguous()  # [N,3] grey
    rendered, _radii, _allmap = rasterizer(
        means3D=pc.get_xyz,
        means2D=torch.zeros_like(pc.get_xyz, requires_grad=False),
        shs=None,
        colors_precomp=shadow_color,
        opacities=pc.get_opacity,
        scales=pc.get_scaling,
        rotations=pc.get_rotation,
        cov3D_precomp=None,
    )
    return rendered[:1]  # [1, H, W]


def _safe_normalize(vec):
    return F.normalize(vec, p=2, dim=-1, eps=1e-6)


def _ndotwi(normals, wi):
    return F.elu(torch.sum(normals * wi, dim=-1, keepdim=True), alpha=0.01) + 0.01


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


def _build_raster_settings(viewpoint_camera, pc, pipe, bg_color, scaling_modifier):
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


def _build_output_dict(means2D, radii, rendered_image, allmap, viewpoint_camera, pc, pipe, shadow_pkg):
    """Pack rasterizer outputs into the standard return dict."""
    render_alpha = allmap[1:2]
    render_normal = (
        allmap[2:5].permute(1, 2, 0) @ viewpoint_camera.world_view_transform[:3, :3].T
    ).permute(2, 0, 1)
    render_depth_median  = torch.nan_to_num(allmap[5:6], 0, 0)
    render_depth_expected = torch.nan_to_num(allmap[0:1] / render_alpha, 0, 0)
    surf_depth  = render_depth_expected * (1 - pipe.depth_ratio) + pipe.depth_ratio * render_depth_median
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth).permute(2, 0, 1) * render_alpha.detach()
    return {
        "render":            rendered_image,
        "viewspace_points":  means2D,
        "visibility_filter": radii > 0,
        "radii":             radii,
        "rend_alpha":        render_alpha,
        "rend_normal":       render_normal,
        "rend_dist":         allmap[6:7],
        "surf_depth":        surf_depth,
        "surf_normal":       surf_normal,
        "shadow":            shadow_pkg["shadow_image"]    if shadow_pkg else None,
        "pre_shadow":        shadow_pkg["per_point_shadow"] if shadow_pkg else None,
    }


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
           scaling_modifier=1.0, override_color=None, fix_lambert=False,
           apply_shadow=True, deferred=False):
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

    # ══ Step 2: mBRDF — compute decay + other_effects ════════════════════
    # Phase 0 (no shadow): flat kd, skip phasefunc
    # Phase 1+ (shadow on): phasefunc → decay (optimised shadow) + other_effects
    mbrdf = None
    if getattr(pc, "use_mbrdf", False) and override_color is None:
        if not apply_shadow:
            # Phase 0 warmup: no phasefunc, no shadow
            # tex path: mbrdf=None → render raw texture_color directly
            # no-tex path: flat kd/π as per-Gaussian color
            if not getattr(pc, "use_textures", False):
                N = pc.get_xyz.shape[0]
                mbrdf = {
                    "basecolor": (pc.get_kd / math.pi).expand(-1, 3).contiguous(),  # [N,3]
                    "shadow": torch.ones((N, 1), dtype=torch.float32, device=pc.get_xyz.device),
                    "other_effects": None,
                }
            # else: mbrdf stays None → render_textured uses raw texture_color
        else:
            mbrdf = _compute_mbrdf_colors(viewpoint_camera, pc, shadow_pkg, fix_lambert=fix_lambert)

    # ══ Step 3: Route to textured / plain renderer ════════════════════════
    if getattr(pc, "use_textures", False):
        from gaussian_renderer.textured import render_textured
        return render_textured(
            viewpoint_camera, pc, pipe, bg_color,
            scaling_modifier=scaling_modifier,
            override_color=override_color,
            shadow_pkg=shadow_pkg,
            mbrdf=mbrdf,
            deferred=deferred,
        )

    # ══ No-tex path ═══════════════════════════════════════════════════════
    means2D = torch.zeros_like(pc.get_xyz, requires_grad=True, device="cuda") + 0
    try:
        means2D.retain_grad()
    except Exception:
        pass

    raster_settings, scales, rotations, cov3D_precomp = _build_raster_settings(
        viewpoint_camera, pc, pipe, bg_color, scaling_modifier
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    def _splat(colors_precomp, shs=None):
        img, radii, allmap = rasterizer(
            means3D=pc.get_xyz, means2D=means2D,
            shs=shs, colors_precomp=colors_precomp,
            opacities=pc.get_opacity,
            scales=scales, rotations=rotations, cov3D_precomp=cov3D_precomp,
        )
        return img, radii, allmap
    
    def _splat_7ch(colors_precomp, shs=None):
        img, radii, allmap = rasterizer_7ch(
            means3D=pc.get_xyz, means2D=means2D,
            shs=shs, colors_precomp=colors_precomp,
            opacities=pc.get_opacity,
            scales=scales, rotations=rotations, cov3D_precomp=cov3D_precomp,
        )
        return img, radii, allmap

    if deferred:
        # ── Deferred: requires 7-channel CUDA kernel (TODO) ───────────────
        # When ready, replace with:
        #   raw, radii, allmap = rasterizer_7ch(...)
        #   rendered_image = raw[:3] * raw[3:4] + raw[4:7]
        raise NotImplementedError("Deferred 7-channel kernel not yet implemented.")

    else:
        # ── Direct: final = basecolor * shadow + other_effects → 3-ch rasterizer ──
        if override_color is not None:
            colors_precomp, shs = override_color, None
        elif mbrdf is not None:
            c = mbrdf["basecolor"] * mbrdf["shadow"]       # [N,3]
            if mbrdf["other_effects"] is not None:
                c = c + mbrdf["other_effects"]
            colors_precomp, shs = torch.clamp_min(c, 0.0), None
        else:
            colors_precomp, shs = None, pc.get_features

        rendered_image, radii, allmap = _splat(colors_precomp, shs=shs)

    return _build_output_dict(means2D, radii, rendered_image, allmap,
                              viewpoint_camera, pc, pipe, shadow_pkg)
