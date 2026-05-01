#!/usr/bin/env python3
"""Profile the 2DGS texture deferred render path on one camera view."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import statistics
import sys
from typing import Callable

import torch
from argparse import ArgumentParser


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
FRESH_SHADOW_PATH = os.path.join(ROOT, "submodules", "diff-surfel-rasterization-shadow")
FRESH_TEXTURE_PATH = os.path.join(ROOT, "submodules", "surfel-texture")
FRESH_TEXTURE_DEFERRED_PATH = os.path.join(ROOT, "submodules", "surfel-texture-deferred")
for _fresh_path in (FRESH_TEXTURE_DEFERRED_PATH, FRESH_TEXTURE_PATH, FRESH_SHADOW_PATH):
    if os.path.isdir(_fresh_path):
        while _fresh_path in sys.path:
            sys.path.remove(_fresh_path)
        sys.path.insert(0, _fresh_path)
import diff_surfel_rasterization_shadow as _fresh_shadow_module  # noqa: E402,F401
import surfel_texture as _fresh_texture_module  # noqa: E402,F401
import surfel_texture_deferred as _fresh_texture_deferred_module  # noqa: E402,F401
print(f"[profile] preloaded shadow module: {_fresh_shadow_module.__file__}", flush=True)
print(f"[profile] preloaded texture module: {_fresh_texture_module.__file__}", flush=True)
print(f"[profile] preloaded texture deferred module: {_fresh_texture_deferred_module.__file__}", flush=True)

from arguments import ModelParams, OptimizationParams, PipelineParams  # noqa: E402
from gaussian_renderer import (  # noqa: E402
    _NdotWi,
    _build_2dgs_raster_settings,
    _compute_shadow_pass_2dgs_native,
    _safe_normalize,
    render,
    surfel_rasterizer_deferred,
)
from gaussian_renderer.texture_branch import (  # noqa: E402
    _compute_texture_mbrdf,
    _compute_texture_shadow_pass,
)
from gaussian_renderer.textured import TextureRenderInputs, rasterize_with_texture_module  # noqa: E402
from scene import Scene  # noqa: E402
from scene.gaussian_model_2dgs_adapter import GaussianModel2DGSAdapter  # noqa: E402


def _unpack_training_checkpoint(checkpoint):
    if not isinstance(checkpoint, (tuple, list)):
        return checkpoint, None, None
    if len(checkpoint) == 2:
        model_params, first_iter = checkpoint
        return model_params, first_iter, None
    if len(checkpoint) == 3:
        model_params, first_iter, scene_state = checkpoint
        return model_params, first_iter, scene_state
    raise ValueError(f"Unsupported checkpoint format with {len(checkpoint)} entries")


def _build_gs_args(profile_args: argparse.Namespace):
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    argv = [
        "-s",
        profile_args.source_path,
        "-m",
        profile_args.model_path,
        "--data_device",
        "cpu",
        "--view_num",
        str(profile_args.view_num),
        "--iterations",
        str(profile_args.iterations),
        "--rasterizer",
        "2dgs",
        "--sh_degree",
        "0",
        "--resolution",
        "1",
        "--use_nerual_phasefunc",
        "--texture_effect_mode",
        profile_args.texture_effect_mode,
        "--mbrdf_normal_source",
        profile_args.mbrdf_normal_source,
        "--cam_opt",
        "--pl_opt",
        "--eval",
    ]
    if not profile_args.no_textures:
        argv.append("--use_textures")
    if profile_args.texture_dynamic_resolution:
        argv.extend(
            [
                "--texture_dynamic_resolution",
                "--texture_min_resolution",
                str(profile_args.texture_min_resolution),
                "--texture_max_resolution",
                str(profile_args.texture_max_resolution),
            ]
        )
    args = parser.parse_args(argv)
    return lp.extract(args), op.extract(args), pp.extract(args)


def _cuda_ms(fn: Callable[[], object], warmup: int, repeat: int):
    for _ in range(warmup):
        out = fn()
        if isinstance(out, tuple):
            del out
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
        if isinstance(out, tuple):
            del out
    return {
        "mean_ms": statistics.fmean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "samples_ms": times,
    }


def _texture_stats(gau):
    num_points = int(gau.get_xyz.shape[0])
    stats = {"num_points": num_points}
    if not bool(getattr(gau, "use_textures", False)):
        stats["dynamic"] = False
        stats["use_textures"] = False
        stats["total_texels"] = 0
        stats["avg_texels_per_point"] = 0.0
        return stats
    stats["use_textures"] = True
    if bool(getattr(gau, "has_dynamic_textures", False)):
        dims = gau.get_texture_dims
        counts = (dims[:, 0].long() * dims[:, 1].long()).clamp_min(1)
        stats["dynamic"] = True
        stats["total_texels"] = int(counts.sum().item())
        stats["avg_texels_per_point"] = float(counts.float().mean().item()) if counts.numel() else 0.0
        hist = {}
        for h, w in dims[:, :2].detach().cpu().tolist():
            key = f"{int(h)}x{int(w)}"
            hist[key] = hist.get(key, 0) + 1
        stats["resolution_hist"] = hist
    else:
        tex_res = int(getattr(gau, "texture_resolution", 1))
        stats["dynamic"] = False
        stats["texture_resolution"] = tex_res
        stats["total_texels"] = num_points * tex_res * tex_res
        stats["avg_texels_per_point"] = float(tex_res * tex_res)
    return stats


def _compute_native_mbrdf(viewpoint_camera, gau, shadow_pkg, fix_labert=False):
    means3d = gau.get_xyz
    pl_pos_expand = viewpoint_camera.pl_pos.expand(means3d.shape[0], -1)
    wi_ray = pl_pos_expand - means3d
    wi_dist2 = torch.sum(wi_ray**2, dim=-1, keepdim=True).clamp_min(1e-12)
    dist_2_inv = 1.0 / wi_dist2
    wi = wi_ray * torch.sqrt(dist_2_inv)
    camera_center_for_brdf = viewpoint_camera.camera_center
    if os.getenv("GS3_2DGS_DETACH_VIEWDIR", "0") == "1":
        camera_center_for_brdf = camera_center_for_brdf.detach()
    wo = _safe_normalize(camera_center_for_brdf - means3d)

    local_axises = gau.get_local_axis
    local_z = local_axises[:, :, 2]
    wi_local = torch.einsum("Ki,Kij->Kj", wi, local_axises)
    wo_local = torch.einsum("Ki,Kij->Kj", wo, local_axises)
    cos_theta = _NdotWi(local_z, wi, torch.nn.ELU(alpha=0.01), 0.01)
    diffuse = gau.get_kd / math.pi
    asg_scales = gau.asg_func.get_asg_lam_miu
    asg_axises = gau.asg_func.get_asg_axis
    asg_1 = gau.asg_func(wi_local, wo_local, gau.get_alpha_asg, asg_scales, asg_axises)
    shadow_hint = None if shadow_pkg is None else shadow_pkg["per_point_shadow"]
    decay, other_effects, _, _ = gau.neural_phasefunc(
        wi,
        wo,
        means3d,
        gau.get_neural_material,
        hint=shadow_hint,
    )
    if decay is None:
        decay = torch.ones((means3d.shape[0], 1), dtype=torch.float32, device=means3d.device)
    if fix_labert:
        basecolor = diffuse * cos_theta * dist_2_inv
    else:
        specular = gau.get_ks * asg_1
        basecolor = (diffuse + specular) * cos_theta * dist_2_inv
    if other_effects is None:
        other_effects = torch.zeros_like(basecolor)
    else:
        other_effects = other_effects * dist_2_inv
    return torch.cat([basecolor, decay, other_effects], dim=1)


def _rasterize_native_deferred(viewpoint_camera, gau, pipe, bg_color, colors_precomp, scaling_modifier=1.0):
    means3d = gau.get_xyz
    means2d = torch.zeros_like(means3d, dtype=torch.float32, requires_grad=True, device=means3d.device)
    transmat_grad_holder = None
    if getattr(viewpoint_camera, "cam_pose_adj", None) is not None and viewpoint_camera.cam_pose_adj.requires_grad:
        transmat_grad_holder = torch.zeros(
            (means3d.shape[0], 9),
            dtype=torch.float32,
            device=means3d.device,
            requires_grad=True,
        )
    surfel_bg = bg_color
    if surfel_bg.shape[0] == 3:
        surfel_bg = torch.cat(
            [surfel_bg, torch.zeros(4, dtype=surfel_bg.dtype, device=surfel_bg.device)],
            dim=0,
        )
    raster_settings = _build_2dgs_raster_settings(
        viewpoint_camera,
        pipe,
        surfel_bg,
        scaling_modifier,
        gau.active_sh_degree,
    )
    rasterizer = surfel_rasterizer_deferred(raster_settings=raster_settings)
    return rasterizer(
        means3D=means3d,
        means2D=means2d,
        opacities=gau.get_opacity,
        shs=None,
        colors_precomp=colors_precomp,
        scales=gau.get_scaling,
        rotations=gau.get_rotation,
        cov3D_precomp=None,
        texture_color=None,
        texture_alpha=None,
        use_textures=False,
        transmat_grad_holder=transmat_grad_holder,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_path", "--source", dest="source_path", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--texture_effect_mode", default="uvshadow_micro_normal_specular_full")
    ap.add_argument("--mbrdf_normal_source", default="local_q")
    ap.add_argument("--view_num", type=int, default=2000)
    ap.add_argument("--iterations", type=int, default=100000)
    ap.add_argument("--view_index", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--texture_dynamic_resolution", action="store_true")
    ap.add_argument("--no_textures", action="store_true")
    ap.add_argument("--texture_min_resolution", type=int, default=4)
    ap.add_argument("--texture_max_resolution", type=int, default=64)
    ap.add_argument("--profile_backward", action="store_true")
    ap.add_argument("--output_json", default="")
    args = ap.parse_args()

    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True

    print("[profile] building args", flush=True)
    modelset, opt, pipe = _build_gs_args(args)
    print(f"[profile] loading checkpoint: {args.checkpoint}", flush=True)
    checkpoint_payload = torch.load(args.checkpoint, weights_only=False)
    model_params, first_iter, scene_state = _unpack_training_checkpoint(checkpoint_payload)

    print("[profile] creating model and scene", flush=True)
    gaussians = GaussianModel2DGSAdapter(modelset, opt)
    scene = Scene(
        modelset,
        gaussians,
        opt=opt,
        shuffle=False,
        skip_train=False,
        skip_test=False,
        checkpoint=args.checkpoint,
        load_optimized_cameras=scene_state is None,
    )
    print("[profile] restoring gaussian state", flush=True)
    gaussians.restore(model_params, opt)
    if scene_state is not None:
        print("[profile] restoring camera/light state", flush=True)
        scene.restore(scene_state)
    if getattr(gaussians, "neural_phasefunc", None) is not None:
        gaussians.neural_phasefunc.eval()

    views = scene.getTestCameras()
    view = views[args.view_index % len(views)]
    bg_color = [1, 1, 1] if modelset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    stats = _texture_stats(gaussians)
    print(f"[profile] stats: {json.dumps(stats, sort_keys=True)}", flush=True)
    import diff_surfel_rasterization_shadow as shadow_mod
    print(f"[profile] shadow module: {shadow_mod.__file__}", flush=True)
    print(f"[profile] shadow forward: {inspect.signature(shadow_mod.GaussianRasterizer.forward)}", flush=True)
    stats.update(
        {
            "checkpoint": args.checkpoint,
            "first_iter": first_iter,
            "texture_effect_mode": getattr(gaussians, "texture_effect_mode", args.texture_effect_mode),
            "mbrdf_normal_source": getattr(gaussians, "mbrdf_normal_source", args.mbrdf_normal_source),
            "view_image": getattr(view, "image_name", str(args.view_index)),
            "image_size": [int(view.image_width), int(view.image_height)],
        }
    )

    def full_render_fn():
        return render(
            view,
            gaussians,
            None,
            None,
            None,
            None,
            None,
            pipe,
            background,
            modelset,
            scaling_modifier=1.0,
            fix_labert=False,
            iteration=int(first_iter or 0),
        )

    with torch.no_grad():
        if args.no_textures:
            print("[profile] priming native no-texture stages", flush=True)
            native_shadow_pkg = _compute_shadow_pass_2dgs_native(view, gaussians, pipe, background, 1.0)
            native_colors = _compute_native_mbrdf(view, gaussians, native_shadow_pkg, fix_labert=False)
            print("[profile] timing native no-texture stages", flush=True)
            timings = {
                "native_shadow_pass": _cuda_ms(
                    lambda: _compute_shadow_pass_2dgs_native(view, gaussians, pipe, background, 1.0),
                    args.warmup,
                    args.repeat,
                ),
                "native_mbrdf_precompute": _cuda_ms(
                    lambda: _compute_native_mbrdf(view, gaussians, native_shadow_pkg, fix_labert=False),
                    args.warmup,
                    args.repeat,
                ),
                "native_raster": _cuda_ms(
                    lambda: _rasterize_native_deferred(view, gaussians, pipe, background, native_colors, 1.0),
                    args.warmup,
                    args.repeat,
                ),
                "full_render": _cuda_ms(
                    full_render_fn,
                    args.warmup,
                    args.repeat,
                )
            }
            result = {"stats": stats, "timings": timings}
            print(json.dumps(result, indent=2, sort_keys=True))
            if args.output_json:
                with open(args.output_json, "w", encoding="utf8") as f:
                    json.dump(result, f, indent=2, sort_keys=True)
            return

        print("[profile] priming shadow and mbrdf", flush=True)
        shadow_pkg = _compute_texture_shadow_pass(view, gaussians, pipe, background, 1.0)
        mbrdf = _compute_texture_mbrdf(view, gaussians, shadow_pkg, fix_labert=False)

        timings = {}
        print("[profile] timing shadow_pass", flush=True)
        timings["shadow_pass"] = _cuda_ms(
            lambda: _compute_texture_shadow_pass(view, gaussians, pipe, background, 1.0),
            args.warmup,
            args.repeat,
        )
        print("[profile] timing shadow_pass_point", flush=True)
        timings["shadow_pass_point"] = _cuda_ms(
            lambda: _compute_texture_shadow_pass(view, gaussians, pipe, background, 1.0, per_uv=False),
            args.warmup,
            args.repeat,
        )
        print("[profile] timing mbrdf_precompute", flush=True)
        timings["mbrdf_precompute"] = _cuda_ms(
            lambda: _compute_texture_mbrdf(view, gaussians, shadow_pkg, fix_labert=False),
            args.warmup,
            args.repeat,
        )
        print("[profile] timing texture_raster", flush=True)
        timings["texture_raster"] = _cuda_ms(
            lambda: rasterize_with_texture_module(
                viewpoint_camera=view,
                pc=gaussians,
                pipe=pipe,
                bg_color=background,
                scaling_modifier=1.0,
                inputs=TextureRenderInputs(
                    deferred=True,
                    mbrdf=mbrdf,
                    colors_precomp=None,
                    return_split=True,
                ),
            ),
            args.warmup,
            args.repeat,
        )
        print("[profile] timing full_render", flush=True)
        timings["full_render"] = _cuda_ms(
            full_render_fn,
            args.warmup,
            args.repeat,
        )

    if args.profile_backward:
        print("[profile] timing full_render_forward_backward", flush=True)

        def full_forward_backward():
            if getattr(gaussians, "optimizer", None) is not None:
                gaussians.optimizer.zero_grad(set_to_none=True)
            if getattr(scene, "optimizer", None) is not None:
                scene.optimizer.zero_grad(set_to_none=True)
            pkg = render(
                view,
                gaussians,
                None,
                None,
                None,
                None,
                None,
                pipe,
                background,
                modelset,
                scaling_modifier=1.0,
                fix_labert=False,
                iteration=int(first_iter or 0),
            )
            image = pkg.get("render_composed", pkg.get("render"))
            loss = image.mean()
            loss.backward()
            return loss

        torch.cuda.reset_peak_memory_stats()
        timings["full_render_forward_backward"] = _cuda_ms(
            full_forward_backward,
            max(0, min(args.warmup, 1)),
            max(1, min(args.repeat, 3)),
        )
        stats["peak_memory_mb_after_backward_profile"] = int(torch.cuda.max_memory_allocated() // (1024 * 1024))

    result = {"stats": stats, "timings": timings}
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf8") as f:
            json.dump(result, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
