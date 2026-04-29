import importlib.util
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from gaussian_renderer.textured import TextureRenderInputs, rasterize_with_texture_module

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_GS3_RENDER_PATH = os.path.abspath(
    os.path.join(_REPO_ROOT, "..", "gs3", "gaussian_renderer", "__init__.py")
)
_SUBMODULE_PATHS = [
    os.path.join(_REPO_ROOT, "submodules", "surfel-texture"),
    os.path.join(_REPO_ROOT, "submodules", "surfel-texture-deferred"),
    os.path.join(_REPO_ROOT, "submodules", "diff-surfel-rasterization-shadow"),
]
for _path in reversed(_SUBMODULE_PATHS):
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

from diff_surfel_rasterization_shadow import GaussianRasterizationSettings as ShadowRasterSettings
from diff_surfel_rasterization_shadow import GaussianRasterizer as ShadowRasterizer
from surfel_texture import GaussianRasterizationSettings as SurfelRasterSettings

from scene.gaussian_model_2dgs_adapter import GaussianModel2DGSAdapter as GaussianModel
from utils.graphics_utils import getProjectionMatrix

_GS3_LEGACY_RENDER_MOD = None


def _load_gs3_legacy_render_module():
    global _GS3_LEGACY_RENDER_MOD
    if _GS3_LEGACY_RENDER_MOD is None:
        spec = importlib.util.spec_from_file_location("gs3_legacy_gaussian_renderer", _GS3_RENDER_PATH)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load gs3 renderer from {_GS3_RENDER_PATH}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _GS3_LEGACY_RENDER_MOD = module
    return _GS3_LEGACY_RENDER_MOD


def _should_delegate_to_gs3_renderer(gau):
    return getattr(gau, "gs2dgs_backend", None) == "legacy" and not bool(getattr(gau, "use_textures", False))


def _safe_normalize(x):
    return F.normalize(x, dim=-1, eps=1e-6)


def _iter_texture_point_chunks(counts, max_points, max_texels):
    num_points = int(counts.shape[0])
    max_points = max(1, int(max_points))
    max_texels = max(1, int(max_texels))
    start = 0
    while start < num_points:
        point_end = min(start + max_points, num_points)
        chunk_counts = counts[start:point_end]
        cumulative = torch.cumsum(chunk_counts, dim=0)
        valid = torch.nonzero(cumulative <= max_texels, as_tuple=False).flatten()
        if valid.numel() == 0:
            end = start + 1
        else:
            end = start + int(valid[-1].item()) + 1
        yield start, end
        start = end


def _mean_dynamic_texture_values(flat_values, texture_dims, max_points, max_texels):
    counts = (texture_dims[:, 0].to(torch.long) * texture_dims[:, 1].to(torch.long)).clamp_min(1)
    offsets = texture_dims[:, 2].to(torch.long)
    num_points = int(texture_dims.shape[0])
    values_2d = flat_values.reshape(flat_values.shape[0], -1)
    out = torch.empty((num_points, values_2d.shape[1]), dtype=flat_values.dtype, device=flat_values.device)

    for start, end in _iter_texture_point_chunks(counts, max_points, max_texels):
        chunk_counts = counts[start:end]
        for count_value in torch.unique(chunk_counts).tolist():
            local_idx = torch.nonzero(chunk_counts == count_value, as_tuple=False).flatten()
            if local_idx.numel() == 0:
                continue
            point_idx = local_idx + start
            local = torch.arange(count_value, device=flat_values.device, dtype=torch.long)
            src = offsets[point_idx, None] + local[None, :]
            out[point_idx] = values_2d[src.reshape(-1)].view(point_idx.numel(), -1, values_2d.shape[1]).mean(dim=1)

    return out.reshape((num_points,) + flat_values.shape[1:])


def _NdotWi(nrm, wi, elu, a):
    tmp = a * (1.0 - 1.0 / math.e)
    return (elu((nrm * wi).sum(dim=-1, keepdim=True)) + tmp) / (1.0 + tmp)


def _build_2dgs_raster_settings(viewpoint_camera, pipe, bg_color, scaling_modifier, sh_degree):
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    return SurfelRasterSettings(
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
    lt = _build_light_transform_2dgs(viewpoint_camera, means3d, pipe)
    use_textures = bool(getattr(gau, "use_textures", False)) and bool(getattr(pipe, "enable_texture", True))
    texture_alpha = gau.get_texture_alpha if use_textures else torch.empty(0, device="cuda")
    texture_dims = gau.get_texture_dims if use_textures and bool(getattr(gau, "has_dynamic_textures", False)) else torch.empty(0, device="cuda", dtype=torch.int32)

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
        use_textures=use_textures,
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
        texture_alpha=texture_alpha,
        texture_dims=texture_dims,
        texture_sigma_factor=float(getattr(gau, "texture_sigma_factor", 3.0)),
        non_trans=None,
        offset=getattr(pipe, "shadow_offset", 0.015),
        thres=-1.0,
        is_train=False,
    )
    non_trans_safe = torch.clamp_min(non_trans, 1e-6)
    if use_textures:
        if texture_dims.numel() > 0:
            per_uv_shadow = (out_trans / non_trans_safe).reshape(-1)
            per_point_shadow = _mean_dynamic_texture_values(
                per_uv_shadow,
                texture_dims,
                max_points=int(getattr(gau, "texture_phase_chunk_points", 4096)),
                max_texels=int(getattr(gau, "texture_phase_chunk_texels", 262_144)),
            ).unsqueeze(-1)
        else:
            tex_res = int(getattr(gau, "texture_resolution", 1))
            per_uv_shadow = (out_trans / non_trans_safe).view(means3d.shape[0], tex_res, tex_res)
            per_point_shadow = per_uv_shadow.mean(dim=(-2, -1), keepdim=False).unsqueeze(-1)
    else:
        per_uv_shadow = None
        per_point_shadow = (out_trans / non_trans_safe).unsqueeze(-1)

    return {
        "per_point_shadow": per_point_shadow,
        "per_uv_shadow": per_uv_shadow,
        "light_viewmatrix": lt["world_view_transform_light"],
        "light_projmatrix": lt["light_projmatrix"],
    }


def _compute_mbrdf_colors_2dgs(viewpoint_camera, gau, shadow_pkg, fix_labert=False):
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
    wi_local = torch.einsum("ki,kij->kj", wi, local_axises)
    wo_local = torch.einsum("ki,kij->kj", wo, local_axises)
    cos_theta = _NdotWi(local_z, wi, torch.nn.ELU(alpha=0.01), 0.01)
    diffuse = gau.get_kd / math.pi
    asg_scales = gau.asg_func.get_asg_lam_miu
    asg_axises = gau.asg_func.get_asg_axis
    asg_1 = gau.asg_func(wi_local, wo_local, gau.get_alpha_asg, asg_scales, asg_axises)

    per_uv_shadow = None if shadow_pkg is None else shadow_pkg.get("per_uv_shadow")
    if bool(getattr(gau, "use_textures", False)) and per_uv_shadow is not None and bool(getattr(gau, "has_dynamic_textures", False)):
        texture_effect_mode = str(getattr(gau, "texture_effect_mode", "per_uv"))
        if texture_effect_mode != "per_uv":
            raise ValueError("Dynamic texture resolution currently requires texture_effect_mode='per_uv'.")

        texture_dims = gau.get_texture_dims
        counts = (texture_dims[:, 0].to(torch.long) * texture_dims[:, 1].to(torch.long)).clamp_min(1)
        offsets = texture_dims[:, 2].to(torch.long)
        total_texels = int(counts.sum().item())
        chunk_points = max(1, int(getattr(gau, "texture_phase_chunk_points", 4096)))
        chunk_texels = max(1, int(getattr(gau, "texture_phase_chunk_texels", 262_144)))
        per_uv_shadow = per_uv_shadow.reshape(total_texels, 1)

        basecolor = torch.empty((total_texels, 3), dtype=torch.float32, device=dev)
        shadow = torch.empty((total_texels, 1), dtype=torch.float32, device=dev)
        other_effects = torch.empty((total_texels, 3), dtype=torch.float32, device=dev)
        texture_color = gau.get_texture_color

        for start, end in _iter_texture_point_chunks(counts, chunk_points, chunk_texels):
            point_ids = torch.arange(start, end, dtype=torch.long, device=dev)
            texel_ids = torch.repeat_interleave(point_ids, counts[start:end])
            flat_start = int(offsets[start].item())
            flat_end = int((offsets[end - 1] + counts[end - 1]).item())
            num_texels = flat_end - flat_start

            def _expand_chunk(tensor):
                return tensor[texel_ids]

            decay_flat, other_effects_flat, asg_3_flat, _ = gau.neural_phasefunc(
                _expand_chunk(wi),
                _expand_chunk(wo),
                _expand_chunk(gau.get_xyz),
                _expand_chunk(gau.get_neural_material),
                hint=per_uv_shadow[flat_start:flat_end],
                asg_1=_expand_chunk(asg_1),
                asg_mlp=gau.asg_mlp,
            )
            if decay_flat is None:
                decay_flat = torch.ones((num_texels, 1), dtype=torch.float32, device=dev)
            if asg_3_flat is None:
                asg_3_flat = _expand_chunk(asg_1)
            if other_effects_flat is None:
                other_effects[flat_start:flat_end] = 0.0
            else:
                other_effects[flat_start:flat_end] = other_effects_flat * (1.0 / _expand_chunk(wi_dist2))

            shadow[flat_start:flat_end] = decay_flat
            specular_flat = _expand_chunk(gau.get_ks) * asg_3_flat
            basecolor[flat_start:flat_end] = (
                (texture_color[flat_start:flat_end] + specular_flat)
                * _expand_chunk(cos_theta)
                * _expand_chunk(dist_2_inv)
            )

        return {"basecolor": basecolor, "shadow": shadow, "other_effects": other_effects}

    if bool(getattr(gau, "use_textures", False)) and per_uv_shadow is not None:
        tex_res = per_uv_shadow.shape[-1]
        texture_effect_mode = str(getattr(gau, "texture_effect_mode", "per_uv"))

        chunk_points = max(1, int(getattr(gau, "texture_phase_chunk_points", 4096)))
        shadow = torch.empty((num_points, tex_res, tex_res), dtype=torch.float32, device=dev)
        specular_uv = torch.empty((num_points, 3, tex_res, tex_res), dtype=torch.float32, device=dev)
        if texture_effect_mode == "per_uv":
            other_effects = torch.empty((num_points, 3, tex_res, tex_res), dtype=torch.float32, device=dev)
        elif texture_effect_mode == "legacy_mean_other":
            other_effects = torch.empty((num_points, 3), dtype=torch.float32, device=dev)
        else:
            raise ValueError(f"Unknown texture_effect_mode: {texture_effect_mode}")

        for start in range(0, num_points, chunk_points):
            end = min(start + chunk_points, num_points)
            count = end - start
            num_texels = count * tex_res * tex_res

            def _expand_chunk(tensor):
                tensor = tensor[start:end]
                dim = tensor.shape[-1]
                return tensor[:, None, None, :].expand(count, tex_res, tex_res, dim).reshape(num_texels, dim)

            decay_flat, other_effects_flat, asg_3_flat, _ = gau.neural_phasefunc(
                _expand_chunk(wi),
                _expand_chunk(wo),
                _expand_chunk(gau.get_xyz),
                _expand_chunk(gau.get_neural_material),
                hint=per_uv_shadow[start:end].reshape(num_texels, 1),
                asg_1=_expand_chunk(asg_1),
                asg_mlp=gau.asg_mlp,
            )
            if decay_flat is None:
                decay_flat = torch.ones((num_texels, 1), dtype=torch.float32, device=dev)
            if asg_3_flat is None:
                asg_3_flat = _expand_chunk(asg_1)

            shadow[start:end] = decay_flat.reshape(count, tex_res, tex_res)
            specular_chunk = (
                _expand_chunk(gau.get_ks) * asg_3_flat
            ).reshape(count, tex_res, tex_res, 3).permute(0, 3, 1, 2).contiguous()
            specular_uv[start:end] = specular_chunk

            if other_effects_flat is None:
                if texture_effect_mode == "per_uv":
                    other_effects[start:end] = 0.0
                else:
                    other_effects[start:end] = 0.0
            else:
                other_effects_uv = (
                    other_effects_flat * (1.0 / _expand_chunk(wi_dist2))
                ).reshape(count, tex_res, tex_res, 3).permute(0, 3, 1, 2).contiguous()
                if texture_effect_mode == "per_uv":
                    other_effects[start:end] = other_effects_uv
                else:
                    other_effects[start:end] = other_effects_uv.mean(dim=(2, 3))

        basecolor = (gau.get_texture_color + specular_uv) * cos_theta[:, :, None, None] * dist_2_inv[:, :, None, None]
        return {"basecolor": basecolor, "shadow": shadow, "other_effects": other_effects}

    shadow_hint = None if shadow_pkg is None else shadow_pkg["per_point_shadow"]
    decay, other_effects, asg_3, _ = gau.neural_phasefunc(
        wi, wo, gau.get_xyz, gau.get_neural_material,
        hint=shadow_hint, asg_1=asg_1, asg_mlp=gau.asg_mlp,
    )
    if decay is None:
        decay = torch.ones((num_points, 1), dtype=torch.float32, device=dev)

    if fix_labert:
        basecolor = diffuse * cos_theta * dist_2_inv
    else:
        specular = gau.get_ks * asg_3
        basecolor = (diffuse + specular) * cos_theta * dist_2_inv

    if other_effects is None:
        other_effects = torch.zeros_like(basecolor)
    else:
        other_effects = other_effects * dist_2_inv
    return {"basecolor": basecolor, "shadow": decay, "other_effects": other_effects}


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
        "shadow_stage": get_shadow_backward_stage(modelset, iteration),
        "transmat_grad_holder": transmat_grad_holder,
        "expected_depth": expected_depth,
        "depth_image": expected_depth,
        "pre_shadow": None if shadow_pkg is None else shadow_pkg["per_point_shadow"],
        "render_base": render,
        "render_shadow": shadow,
        "render_other_effects": other_effects,
    }


def get_shadow_backward_stage(modelset, iteration: int):
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


def render(
    viewpoint_camera,
    gau: GaussianModel,
    light_stream,
    calc_stream,
    local_axises,
    asg_scales,
    asg_axises,
    pipe,
    bg_color: torch.Tensor,
    modelset,
    shadowmap_render=False,
    scaling_modifier=1.0,
    override_color=None,
    fix_labert=False,
    inten_scale=1.0,
    is_train=False,
    asg_mlp=False,
    iteration=0,
):
    if _should_delegate_to_gs3_renderer(gau):
        gs3_render = _load_gs3_legacy_render_module().render
        return gs3_render(
            viewpoint_camera,
            gau,
            light_stream,
            calc_stream,
            local_axises,
            asg_scales,
            asg_axises,
            pipe,
            bg_color,
            modelset,
            shadowmap_render=shadowmap_render,
            scaling_modifier=scaling_modifier,
            override_color=override_color,
            fix_labert=fix_labert,
            inten_scale=inten_scale,
            is_train=is_train,
            asg_mlp=asg_mlp,
            iteration=iteration,
        )

    rasterizer = str(getattr(gau, "rasterizer", ""))
    if rasterizer not in {"2dgs", "2dgs_3ch"}:
        raise NotImplementedError(f"gs2dgs only supports 2dgs rasterizers, got {rasterizer!r}")
    if gau.get_scaling.shape[-1] != 2:
        raise RuntimeError("gs2dgs expects 2DGS scaling with shape [..., 2].")

    shadow_pkg = _compute_shadow_pass_2dgs_native(viewpoint_camera, gau, pipe, bg_color, scaling_modifier) if gau.use_MBRDF else None
    mbrdf = _compute_mbrdf_colors_2dgs(viewpoint_camera, gau, shadow_pkg, fix_labert=fix_labert) if gau.use_MBRDF else None

    if rasterizer == "2dgs_3ch":
        rendered_rgb, radii, allmap, means2D, transmat_grad_holder, _ = rasterize_with_texture_module(
            viewpoint_camera=viewpoint_camera,
            pc=gau,
            pipe=pipe,
            bg_color=bg_color[:3],
            scaling_modifier=scaling_modifier,
            inputs=TextureRenderInputs(
                deferred=False,
                mbrdf=mbrdf,
                colors_precomp=override_color,
                return_split=False,
            ),
        )
        shadow = torch.ones((1, rendered_rgb.shape[1], rendered_rgb.shape[2]), dtype=rendered_rgb.dtype, device=rendered_rgb.device)
        other_effects = torch.zeros((3, rendered_rgb.shape[1], rendered_rgb.shape[2]), dtype=rendered_rgb.dtype, device=rendered_rgb.device)
        pkg = _build_render_pkg(rendered_rgb, shadow, other_effects, means2D, radii, allmap, transmat_grad_holder, modelset, iteration, shadow_pkg)
        pkg["render_base"] = rendered_rgb
        return pkg

    rendered_image, radii, allmap, means2D, transmat_grad_holder, rendered_split = rasterize_with_texture_module(
        viewpoint_camera=viewpoint_camera,
        pc=gau,
        pipe=pipe,
        bg_color=bg_color,
        scaling_modifier=scaling_modifier,
        inputs=TextureRenderInputs(
            deferred=True,
            mbrdf=mbrdf,
            colors_precomp=override_color,
            return_split=True,
        ),
    )
    render_base = rendered_split[0:3]
    shadow = rendered_split[3:4]
    other_effects = rendered_split[4:7]
    pkg = _build_render_pkg(render_base, shadow, other_effects, means2D, radii, allmap, transmat_grad_holder, modelset, iteration, shadow_pkg)
    pkg["render_base"] = render_base
    pkg["render_shadow"] = shadow
    pkg["render_other_effects"] = other_effects
    pkg["render_composed"] = rendered_image
    return pkg
