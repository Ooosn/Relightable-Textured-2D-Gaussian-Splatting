import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SUBMODULE_PATHS = [
    os.path.join(_REPO_ROOT, "..", "gs2dgs", "submodules", "diff-surfel-rasterization-shadow"),
]
for _path in reversed(_SUBMODULE_PATHS):
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

from diff_surfel_rasterization_shadow import GaussianRasterizationSettings as ShadowRasterSettings
from diff_surfel_rasterization_shadow import GaussianRasterizer as ShadowRasterizer
from gaussian_renderer.textured import TextureRenderInputs, rasterize_with_texture_module
from utils.general_utils import build_rotation
from utils.graphics_utils import getProjectionMatrix


def _dot(x, y):
    return torch.sum(x * y, -1, keepdim=True)


def _safe_normalize(x):
    return F.normalize(x, dim=-1, eps=1e-8)


def _NdotWi(nrm, wi, elu, a):
    tmp = a * (1.0 - 1.0 / math.e)
    return (elu(_dot(nrm, wi)) + tmp) / (1.0 + tmp)


def _project_to_local(vec, local_axises):
    return torch.einsum("ki,kij->kj", vec, local_axises)


def _canonicalize_quaternion(q):
    q = q.to(dtype=torch.float32)
    finite = torch.isfinite(q).all(dim=1, keepdim=True)
    norm = q.norm(dim=1, keepdim=True)
    identity = torch.zeros_like(q)
    identity[:, 0] = 1.0
    q = torch.where(finite & (norm > 1e-8), q, identity)
    q = torch.where(q[:, 0:1] < 0.0, -q, q)
    return q


def _frame_from_texture_local_q(fallback_axises, texture_local_q):
    if texture_local_q is None or texture_local_q.numel() == 0:
        return fallback_axises
    return build_rotation(_canonicalize_quaternion(texture_local_q)).to(
        device=fallback_axises.device,
        dtype=fallback_axises.dtype,
    )


def _mean_dynamic_texture_values(flat_values, texture_dims):
    if texture_dims.numel() == 0:
        return flat_values
    counts = (texture_dims[:, 0].to(torch.long) * texture_dims[:, 1].to(torch.long)).clamp_min(1)
    offsets = texture_dims[:, 2].to(torch.long)
    values = flat_values.reshape(flat_values.shape[0], -1)
    out = torch.empty((texture_dims.shape[0], values.shape[1]), dtype=values.dtype, device=values.device)
    point_ids = torch.arange(texture_dims.shape[0], dtype=torch.long, device=values.device)
    texel_point_ids = torch.repeat_interleave(point_ids, counts)
    total_texels = int(counts.sum().item())
    if total_texels > 0:
        starts = torch.cumsum(counts, dim=0) - counts
        local_offsets = torch.arange(total_texels, dtype=torch.long, device=values.device) - torch.repeat_interleave(starts, counts)
        flat_ids = torch.repeat_interleave(offsets, counts) + local_offsets
        accum = torch.zeros_like(out)
        accum.index_add_(0, texel_point_ids, values[flat_ids])
        out = accum / counts.to(values.dtype).unsqueeze(-1)
    return out.reshape(texture_dims.shape[0], *flat_values.shape[1:])


def _dynamic_texture_flat_ids(texture_dims, device):
    counts = (texture_dims[:, 0].to(torch.long) * texture_dims[:, 1].to(torch.long)).clamp_min(1)
    offsets = texture_dims[:, 2].to(torch.long)
    point_ids = torch.arange(texture_dims.shape[0], dtype=torch.long, device=device)
    texel_ids = torch.repeat_interleave(point_ids, counts.to(device))
    total_texels = int(counts.sum().item())
    starts = torch.cumsum(counts, dim=0) - counts
    local_offsets = torch.arange(total_texels, dtype=torch.long, device=device) - torch.repeat_interleave(starts.to(device), counts.to(device))
    flat_ids = torch.repeat_interleave(offsets.to(device), counts.to(device)) + local_offsets
    return counts.to(device), texel_ids, flat_ids, total_texels


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


def _compute_texture_shadow_pass(viewpoint_camera, gau, pipe, bg_color, scaling_modifier=1.0, per_uv=True):
    if viewpoint_camera.pl_pos is None or gau.get_xyz.numel() == 0:
        return None

    means3d = gau.get_xyz
    lt = _build_light_transform_2dgs(viewpoint_camera, means3d, pipe)
    dynamic_textures = bool(getattr(gau, "has_dynamic_textures", False))
    # The current shadow CUDA path indexes texture_alpha as [P, 1, R, R].
    # Dynamic flat-atlas textures do not match that interface yet, so keep the
    # shadow pass per-Gaussian for dynamic atlases instead of passing invalid
    # texture_dims into the rasterizer.
    use_per_uv_shadow = bool(per_uv) and not dynamic_textures
    texture_dims = (
        gau.get_texture_dims
        if dynamic_textures
        else torch.empty(0, device="cuda", dtype=torch.int32)
    )

    shadow_settings = ShadowRasterSettings(
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
        use_textures=use_per_uv_shadow,
    )
    shadow_rasterizer = ShadowRasterizer(raster_settings=shadow_settings)
    light_colors = torch.ones((means3d.shape[0], 3), dtype=torch.float32, device="cuda")

    _, _, _, out_trans, non_trans, _ = shadow_rasterizer(
        means3D=means3d,
        means2D=torch.zeros_like(means3d, requires_grad=True),
        shs=None,
        colors_precomp=light_colors,
        opacities=gau.get_opacity,
        scales=gau.get_scaling,
        rotations=gau.get_rotation,
        cov3Ds_precomp=None,
        texture_alpha=gau.get_texture_alpha if use_per_uv_shadow else torch.empty(0, device="cuda"),
        texture_sigma_factor=float(getattr(gau, "texture_sigma_factor", 3.0)),
        non_trans=None,
        offset=getattr(pipe, "shadow_offset", 0.015),
        thres=-1.0,
        is_train=False,
    )

    non_trans_safe = torch.clamp_min(non_trans, 1e-6)
    if not use_per_uv_shadow:
        per_point_shadow = (out_trans / non_trans_safe).unsqueeze(-1)
        if bool(getattr(gau, "has_dynamic_textures", False)):
            dims = gau.get_texture_dims
            counts = (dims[:, 0].to(torch.long) * dims[:, 1].to(torch.long)).clamp_min(1)
            per_uv_shadow = torch.repeat_interleave(per_point_shadow.reshape(-1), counts)
        else:
            tex_res = int(getattr(gau, "texture_resolution", 1))
            per_uv_shadow = per_point_shadow.reshape(means3d.shape[0], 1, 1).expand(means3d.shape[0], tex_res, tex_res)
    elif texture_dims.numel() > 0:
        per_uv_shadow = (out_trans / non_trans_safe).reshape(-1)
        per_point_shadow = _mean_dynamic_texture_values(
            per_uv_shadow,
            texture_dims,
        ).unsqueeze(-1)
    else:
        tex_res = int(getattr(gau, "texture_resolution", 1))
        per_uv_shadow = (out_trans / non_trans_safe).view(means3d.shape[0], tex_res, tex_res)
        per_point_shadow = per_uv_shadow.mean(dim=(-2, -1), keepdim=False).unsqueeze(-1)

    return {
        "per_point_shadow": per_point_shadow,
        "per_uv_shadow": per_uv_shadow,
        "light_viewmatrix": lt["world_view_transform_light"],
        "light_projmatrix": lt["light_projmatrix"],
    }


def _compute_texture_mbrdf(viewpoint_camera, gau, shadow_pkg, fix_labert=False):
    if viewpoint_camera.pl_pos is None:
        return None

    dev = gau.get_xyz.device
    num_points = gau.get_xyz.shape[0]
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
    if os.getenv("GS3_2DGS_DETACH_VIEWDIR", "0") == "1":
        cam_center = cam_center.detach()

    pl_pos3 = pl_pos[0].unsqueeze(0).expand(num_points, -1)
    wi_ray = pl_pos3 - gau.get_xyz
    wi_dist2 = wi_ray.pow(2).sum(-1, keepdim=True).clamp_min(1e-12)
    dist_2_inv = 1.0 / wi_dist2
    wi = wi_ray * dist_2_inv.sqrt()
    wo = _safe_normalize(cam_center - gau.get_xyz)

    local_axises = gau.get_local_axis
    local_z = local_axises[:, :, 2]
    wi_local = _project_to_local(wi, local_axises)
    wo_local = _project_to_local(wo, local_axises)
    cos_theta = _NdotWi(local_z, wi, torch.nn.ELU(alpha=0.01), 0.01)
    asg_scales = gau.asg_func.get_asg_lam_miu
    asg_axises = gau.asg_func.get_asg_axis
    asg_1 = gau.asg_func(wi_local, wo_local, gau.get_alpha_asg, asg_scales, asg_axises)

    per_uv_shadow = None if shadow_pkg is None else shadow_pkg.get("per_uv_shadow")
    if per_uv_shadow is None:
        raise RuntimeError("Texture rendering requires the texture-aware shadow pass.")

    if bool(getattr(gau, "has_dynamic_textures", False)):
        texture_effect_mode = str(getattr(gau, "texture_effect_mode", "per_uv_micro_normal"))

        texture_dims = gau.get_texture_dims
        counts, texel_ids, flat_ids, total_texels = _dynamic_texture_flat_ids(texture_dims, dev)
        per_uv_shadow = per_uv_shadow.reshape(total_texels, 1)

        if texture_effect_mode == "per_uv_micro_normal":
            basecolor = torch.empty((total_texels, 3), dtype=torch.float32, device=dev)
            shadow = torch.empty((total_texels, 1), dtype=torch.float32, device=dev)
            other_effects = torch.empty((total_texels, 3), dtype=torch.float32, device=dev)
            texture_multiplier = 2.0 * gau.get_texture_color
            texture_local_q = gau.get_texture_local_q
            if texture_local_q.numel() == 0:
                texture_local_q = None

            local_axes_uv = _frame_from_texture_local_q(
                local_axises[texel_ids],
                texture_local_q[flat_ids] if texture_local_q is not None else None,
            )
            wi_uv = wi[texel_ids]
            wo_uv = wo[texel_ids]
            wi_local_uv = _project_to_local(wi_uv, local_axes_uv)
            wo_local_uv = _project_to_local(wo_uv, local_axes_uv)
            cos_theta_uv = _NdotWi(local_axes_uv[:, :, 2], wi_uv, torch.nn.ELU(alpha=0.01), 0.01)
            asg_uv = gau.asg_func(
                wi_local_uv,
                wo_local_uv,
                gau.get_alpha_asg[texel_ids],
                asg_scales,
                asg_axises,
            )

            decay_flat, other_effects_flat, asg_3_flat, _ = gau.neural_phasefunc(
                wi_uv,
                wo_uv,
                gau.get_xyz[texel_ids],
                gau.get_neural_material[texel_ids],
                hint=per_uv_shadow[flat_ids],
                asg_1=asg_uv,
                asg_mlp=gau.asg_mlp,
            )
            if decay_flat is None:
                decay_flat = torch.ones((total_texels, 1), dtype=torch.float32, device=dev)
            if other_effects_flat is None:
                other_effects[:] = 0.0
            else:
                other_effects[flat_ids] = other_effects_flat * dist_2_inv[texel_ids]

            diffuse_flat = gau.get_kd[texel_ids] * texture_multiplier[flat_ids] / math.pi
            if fix_labert:
                specular_flat = 0.0
            else:
                if asg_3_flat is None:
                    asg_3_flat = asg_uv
                specular_flat = gau.get_ks[texel_ids] * asg_3_flat
            basecolor[flat_ids] = (
                (diffuse_flat + specular_flat)
                * cos_theta_uv
                * dist_2_inv[texel_ids]
            )
            shadow[flat_ids] = decay_flat

            return {"basecolor": basecolor, "shadow": shadow, "other_effects": other_effects}

        if texture_effect_mode == "uv_specular_gain":
            point_shadow = shadow_pkg.get("per_point_shadow") if shadow_pkg is not None else None
            if point_shadow is None:
                point_shadow = _mean_dynamic_texture_values(
                    per_uv_shadow,
                    texture_dims,
                )
            decay_g, other_effects_g, _, _ = gau.neural_phasefunc(
                wi,
                wo,
                gau.get_xyz,
                gau.get_neural_material,
                hint=point_shadow,
                asg_1=asg_1,
                asg_mlp=False,
            )
            if decay_g is None:
                decay_g = torch.ones((num_points, 1), dtype=torch.float32, device=dev)
            if other_effects_g is None:
                other_effects_g = torch.zeros((num_points, 3), dtype=torch.float32, device=dev)

            basecolor = torch.empty((total_texels, 3), dtype=torch.float32, device=dev)
            shadow = torch.empty((total_texels, 1), dtype=torch.float32, device=dev)
            other_effects = torch.empty((total_texels, 3), dtype=torch.float32, device=dev)
            texture_multiplier = 2.0 * gau.get_texture_color
            specular_gain = gau.get_texture_specular_gain
            if specular_gain.numel() == 0:
                specular_gain = torch.ones((total_texels, 1), dtype=torch.float32, device=dev)

            diffuse_flat = gau.get_kd[texel_ids] * texture_multiplier[flat_ids] / math.pi
            if fix_labert:
                specular_flat = 0.0
            else:
                specular_flat = gau.get_ks[texel_ids] * asg_1[texel_ids] * specular_gain[flat_ids]
            basecolor[flat_ids] = (
                (diffuse_flat + specular_flat)
                * cos_theta[texel_ids]
                * dist_2_inv[texel_ids]
            )
            shadow_delta = per_uv_shadow[flat_ids] - point_shadow[texel_ids]
            shadow[flat_ids] = torch.clamp(decay_g[texel_ids] + shadow_delta, 0.0, 1.0)
            other_effects[flat_ids] = other_effects_g[texel_ids] * dist_2_inv[texel_ids]

            return {"basecolor": basecolor, "shadow": shadow, "other_effects": other_effects}

        if texture_effect_mode != "per_uv":
            raise ValueError("Dynamic texture resolution currently requires texture_effect_mode='per_uv', 'per_uv_micro_normal', or 'uv_specular_gain'.")

        basecolor = torch.empty((total_texels, 3), dtype=torch.float32, device=dev)
        shadow = torch.empty((total_texels, 1), dtype=torch.float32, device=dev)
        other_effects = torch.empty((total_texels, 3), dtype=torch.float32, device=dev)
        texture_multiplier = 2.0 * gau.get_texture_color

        decay_flat, other_effects_flat, asg_3_flat, _ = gau.neural_phasefunc(
            wi[texel_ids],
            wo[texel_ids],
            gau.get_xyz[texel_ids],
            gau.get_neural_material[texel_ids],
            hint=per_uv_shadow[flat_ids],
            asg_1=asg_1[texel_ids],
            asg_mlp=gau.asg_mlp,
        )
        if decay_flat is None:
            decay_flat = torch.ones((total_texels, 1), dtype=torch.float32, device=dev)
        if other_effects_flat is None:
            other_effects[:] = 0.0
        else:
            other_effects[flat_ids] = other_effects_flat * (1.0 / wi_dist2[texel_ids])

        if fix_labert:
            specular_flat = 0.0
        else:
            if asg_3_flat is None:
                asg_3_flat = asg_1[texel_ids]
            specular_flat = gau.get_ks[texel_ids] * asg_3_flat
        diffuse_flat = gau.get_kd[texel_ids] * texture_multiplier[flat_ids] / math.pi
        shadow[flat_ids] = decay_flat
        basecolor[flat_ids] = (
            (diffuse_flat + specular_flat)
            * cos_theta[texel_ids]
            * dist_2_inv[texel_ids]
        )

        return {"basecolor": basecolor, "shadow": shadow, "other_effects": other_effects}

    tex_res = per_uv_shadow.shape[-1]
    texture_effect_mode = str(getattr(gau, "texture_effect_mode", "per_uv_micro_normal"))

    if texture_effect_mode == "per_uv_micro_normal":
        basecolor = torch.empty((num_points, 3, tex_res, tex_res), dtype=torch.float32, device=dev)
        shadow = torch.empty((num_points, 1, tex_res, tex_res), dtype=torch.float32, device=dev)
        other_effects = torch.empty((num_points, 3, tex_res, tex_res), dtype=torch.float32, device=dev)
        texture_multiplier = 2.0 * gau.get_texture_color
        texture_local_q = gau.get_texture_local_q
        if texture_local_q.numel() == 0:
            texture_local_q = None

        num_texels = num_points * tex_res * tex_res
        texel_ids = torch.arange(num_points, dtype=torch.long, device=dev).repeat_interleave(tex_res * tex_res)
        q = (
            texture_local_q.permute(0, 2, 3, 1).reshape(num_texels, 4)
            if texture_local_q is not None
            else None
        )
        local_axes_uv = _frame_from_texture_local_q(local_axises[texel_ids], q)
        wi_uv = wi[texel_ids]
        wo_uv = wo[texel_ids]
        wi_local_uv = _project_to_local(wi_uv, local_axes_uv)
        wo_local_uv = _project_to_local(wo_uv, local_axes_uv)
        cos_theta_uv = _NdotWi(local_axes_uv[:, :, 2], wi_uv, torch.nn.ELU(alpha=0.01), 0.01)
        asg_uv = gau.asg_func(
            wi_local_uv,
            wo_local_uv,
            gau.get_alpha_asg[texel_ids],
            asg_scales,
            asg_axises,
        )

        decay_flat, other_effects_flat, asg_3_flat, _ = gau.neural_phasefunc(
            wi_uv,
            wo_uv,
            gau.get_xyz[texel_ids],
            gau.get_neural_material[texel_ids],
            hint=per_uv_shadow.reshape(num_texels, 1),
            asg_1=asg_uv,
            asg_mlp=gau.asg_mlp,
        )
        if decay_flat is None:
            decay_flat = torch.ones((num_texels, 1), dtype=torch.float32, device=dev)
        shadow[:] = decay_flat.reshape(num_points, tex_res, tex_res, 1).permute(0, 3, 1, 2).contiguous()
        if other_effects_flat is None:
            other_effects[:] = 0.0
        else:
            other_effects[:] = (
                other_effects_flat * dist_2_inv[texel_ids]
            ).reshape(num_points, tex_res, tex_res, 3).permute(0, 3, 1, 2).contiguous()

        diffuse_flat = (
            gau.get_kd[texel_ids]
            * texture_multiplier.permute(0, 2, 3, 1).reshape(num_texels, 3)
            / math.pi
        )
        if fix_labert:
            specular_flat = 0.0
        else:
            if asg_3_flat is None:
                asg_3_flat = asg_uv
            specular_flat = gau.get_ks[texel_ids] * asg_3_flat
        basecolor[:] = (
            (diffuse_flat + specular_flat)
            * cos_theta_uv
            * dist_2_inv[texel_ids]
        ).reshape(num_points, tex_res, tex_res, 3).permute(0, 3, 1, 2).contiguous()

        return {"basecolor": basecolor, "shadow": shadow, "other_effects": other_effects}

    if texture_effect_mode == "uv_specular_gain":
        point_shadow = shadow_pkg.get("per_point_shadow") if shadow_pkg is not None else None
        if point_shadow is None:
            point_shadow = per_uv_shadow.mean(dim=(-2, -1), keepdim=False).unsqueeze(-1)
        decay_g, other_effects_g, _, _ = gau.neural_phasefunc(
            wi,
            wo,
            gau.get_xyz,
            gau.get_neural_material,
            hint=point_shadow,
            asg_1=asg_1,
            asg_mlp=False,
        )
        if decay_g is None:
            decay_g = torch.ones((num_points, 1), dtype=torch.float32, device=dev)
        if other_effects_g is None:
            other_effects_g = torch.zeros((num_points, 3), dtype=torch.float32, device=dev)

        texture_diffuse = gau.get_kd[:, :, None, None] * (2.0 * gau.get_texture_color) / math.pi
        if fix_labert:
            specular_uv = 0.0
        else:
            specular_gain = gau.get_texture_specular_gain
            if specular_gain.numel() == 0:
                specular_gain = torch.ones((num_points, 1, tex_res, tex_res), dtype=torch.float32, device=dev)
            specular_uv = (gau.get_ks * asg_1)[:, :, None, None] * specular_gain
        raw_shadow = per_uv_shadow[:, None, :, :]
        shadow = torch.clamp(decay_g[:, :, None, None] + raw_shadow - point_shadow[:, :, None, None], 0.0, 1.0)
        other_effects = (other_effects_g * dist_2_inv)[:, :, None, None].expand(-1, -1, tex_res, tex_res)
        basecolor = (texture_diffuse + specular_uv) * cos_theta[:, :, None, None] * dist_2_inv[:, :, None, None]
        return {"basecolor": basecolor, "shadow": shadow, "other_effects": other_effects}

    shadow = torch.empty((num_points, tex_res, tex_res), dtype=torch.float32, device=dev)
    specular_uv = torch.empty((num_points, 3, tex_res, tex_res), dtype=torch.float32, device=dev)
    if texture_effect_mode == "per_uv":
        other_effects = torch.empty((num_points, 3, tex_res, tex_res), dtype=torch.float32, device=dev)
    elif texture_effect_mode == "legacy_mean_other":
        other_effects = torch.empty((num_points, 3), dtype=torch.float32, device=dev)
    else:
        raise ValueError(f"Unknown texture_effect_mode: {texture_effect_mode}")

    num_texels = num_points * tex_res * tex_res
    texel_ids = torch.arange(num_points, dtype=torch.long, device=dev).repeat_interleave(tex_res * tex_res)

    def _expand_all(tensor):
        dim = tensor.shape[-1]
        return tensor[:, None, None, :].expand(num_points, tex_res, tex_res, dim).reshape(num_texels, dim)

    decay_flat, other_effects_flat, asg_3_flat, _ = gau.neural_phasefunc(
        _expand_all(wi),
        _expand_all(wo),
        _expand_all(gau.get_xyz),
        _expand_all(gau.get_neural_material),
        hint=per_uv_shadow.reshape(num_texels, 1),
        asg_1=_expand_all(asg_1),
        asg_mlp=gau.asg_mlp,
    )
    if decay_flat is None:
        decay_flat = torch.ones((num_texels, 1), dtype=torch.float32, device=dev)

    shadow[:] = decay_flat.reshape(num_points, tex_res, tex_res)
    if fix_labert:
        specular_uv[:] = 0.0
    else:
        if asg_3_flat is None:
            asg_3_flat = _expand_all(asg_1)
        specular_uv[:] = (
            _expand_all(gau.get_ks) * asg_3_flat
        ).reshape(num_points, tex_res, tex_res, 3).permute(0, 3, 1, 2).contiguous()

    if other_effects_flat is None:
        other_effects[:] = 0.0
    else:
        other_effects_uv = (
            other_effects_flat * (1.0 / _expand_all(wi_dist2))
        ).reshape(num_points, tex_res, tex_res, 3).permute(0, 3, 1, 2).contiguous()
        if texture_effect_mode == "per_uv":
            other_effects[:] = other_effects_uv
        else:
            other_effects[:] = other_effects_uv.mean(dim=(2, 3))

    texture_diffuse = gau.get_kd[:, :, None, None] * (2.0 * gau.get_texture_color) / math.pi
    basecolor = (texture_diffuse + specular_uv) * cos_theta[:, :, None, None] * dist_2_inv[:, :, None, None]
    return {"basecolor": basecolor, "shadow": shadow, "other_effects": other_effects}


def _get_shadow_backward_stage(modelset, iteration: int):
    if getattr(modelset, "detach_shadow", False):
        return {"xyz": False, "opacity": False, "scaling": False, "rotation": False}
    if not getattr(modelset, "shadow_backward_stage_enabled", False):
        return {"xyz": True, "opacity": True, "scaling": True, "rotation": True}
    return {
        "xyz": iteration >= int(getattr(modelset, "shadow_backward_xyz_from_iter", 0)),
        "opacity": iteration >= int(getattr(modelset, "shadow_backward_opacity_from_iter", 0)),
        "scaling": iteration >= int(getattr(modelset, "shadow_backward_scaling_from_iter", 0)),
        "rotation": iteration >= int(getattr(modelset, "shadow_backward_rotation_from_iter", 0)),
    }


def _build_render_pkg(render, shadow, other_effects, means2D, radii, allmap, transmat_grad_holder, modelset, iteration, shadow_pkg):
    render_alpha = allmap[1:2].clamp_min(1e-8)
    expected_depth = torch.nan_to_num(allmap[0:1] / render_alpha, 0, 0)
    return {
        "render": render,
        "shadow": shadow,
        "other_effects": other_effects,
        "viewspace_points": means2D,
        "visibility_filter": radii > 0,
        "radii": radii,
        "out_weight": torch.zeros((means2D.shape[0], 1), dtype=torch.float32, device=means2D.device),
        "backward_info": {},
        "shadow_stage": _get_shadow_backward_stage(modelset, iteration),
        "transmat_grad_holder": transmat_grad_holder,
        "expected_depth": expected_depth,
        "depth_image": expected_depth,
        "pre_shadow": None if shadow_pkg is None else shadow_pkg["per_point_shadow"],
        "render_base": render,
        "render_shadow": shadow,
        "render_other_effects": other_effects,
    }


def render_2dgs_texture_deferred(
    viewpoint_camera,
    gau,
    pipe,
    bg_color,
    modelset,
    scaling_modifier=1.0,
    fix_labert=False,
    iteration=0,
):
    if not getattr(gau, "use_MBRDF", False):
        raise RuntimeError("The gs3 texture branch is currently wired for deferred mBRDF training.")

    shadow_pkg = _compute_texture_shadow_pass(viewpoint_camera, gau, pipe, bg_color, scaling_modifier)
    mbrdf = _compute_texture_mbrdf(viewpoint_camera, gau, shadow_pkg, fix_labert=fix_labert)
    rendered_image, radii, allmap, means2D, transmat_grad_holder, rendered_split = rasterize_with_texture_module(
        viewpoint_camera=viewpoint_camera,
        pc=gau,
        pipe=pipe,
        bg_color=bg_color,
        scaling_modifier=scaling_modifier,
        inputs=TextureRenderInputs(
            deferred=True,
            mbrdf=mbrdf,
            colors_precomp=None,
            return_split=True,
        ),
    )
    render_base = rendered_split[0:3]
    shadow = rendered_split[3:4]
    other_effects = rendered_split[4:7]
    pkg = _build_render_pkg(render_base, shadow, other_effects, means2D, radii, allmap, transmat_grad_holder, modelset, iteration, shadow_pkg)
    pkg["render_composed"] = rendered_image
    return pkg
