import json
import os
from argparse import ArgumentParser
from pathlib import Path

import torch
import torchvision
from tqdm import tqdm
import lpips

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import Scene
from scene.gaussian_model_2dgs_adapter import GaussianModel2DGSAdapter as GaussianModel
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
from utils.system_utils import searchForMaxIteration


loss_fn_alex = None


def get_lpips_model():
    global loss_fn_alex
    if loss_fn_alex is None:
        loss_fn_alex = lpips.LPIPS(net="vgg").cuda().eval()
    return loss_fn_alex


def create_render_streams():
    serial_stream = os.environ.get("SSGS_SERIAL_STREAM", "0") == "1"
    if serial_stream:
        return None, None
    return torch.cuda.Stream(), torch.cuda.Stream()


def _save_image(image: torch.Tensor, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(torch.clamp(image, 0.0, 1.0), path.as_posix())


def _render_image(modelset, render_pkg):
    image = render_pkg["render"] * render_pkg["shadow"] + render_pkg["other_effects"]
    if getattr(modelset, "gamma_change", False):
        image = image / (1.0 + image)
    return torch.clamp(image, 0.0, 1.0)


def _evaluate_view(modelset, view, render_pkg):
    image = _render_image(modelset, render_pkg)
    gt = torch.clamp(view.original_image.to("cuda"), 0.0, 1.0)
    return {
        "image": image,
        "gt": gt,
        "l1": float(l1_loss(image, gt).mean().item()),
        "psnr": float(psnr(image, gt).mean().item()),
        "ssim": float(ssim(image, gt).mean().item()),
        "lpips": float(get_lpips_model().forward(image, gt).squeeze().item()),
    }


def _restore_checkpoint_if_available(model_path: Path, iteration: int, gaussians, opt_args):
    checkpoint_path = model_path / f"chkpnt{iteration}.pth"
    if not checkpoint_path.exists():
        return None
    payload = torch.load(checkpoint_path.as_posix(), weights_only=False)
    model_params = payload["gaussians"] if isinstance(payload, dict) and "gaussians" in payload else payload[0]
    gaussians.restore(model_params, opt_args)
    return checkpoint_path


def _checkpoint_meta(payload):
    if not isinstance(payload, dict):
        return {}
    meta = payload.get("meta")
    return meta if isinstance(meta, dict) else {}


def _checkpoint_backend(payload):
    meta = _checkpoint_meta(payload)
    backend = meta.get("gaussian_backend")
    if backend in {"legacy", "native"}:
        return backend
    if isinstance(payload, dict) and "gaussians" in payload:
        return "native"
    return None


def _apply_checkpoint_model_hints(modelset, payload):
    if payload is None:
        return
    meta = _checkpoint_meta(payload)
    model_flags = meta.get("model_flags", {})
    if isinstance(model_flags, dict):
        for key, value in model_flags.items():
            setattr(modelset, key, value)
    backend = _checkpoint_backend(payload)
    if backend is not None:
        setattr(modelset, "force_native_2dgs_core", backend == "native")


def _resolve_iteration(model_path: Path, load_iteration: int):
    if load_iteration != -1:
        return load_iteration
    point_cloud_root = model_path / "point_cloud"
    if not point_cloud_root.exists():
        return None
    return searchForMaxIteration(point_cloud_root.as_posix())


def _resolve_checkpoint_path(model_path: Path, load_iteration: int):
    iteration = _resolve_iteration(model_path, load_iteration)
    if iteration is None:
        return None
    checkpoint_path = model_path / f"chkpnt{iteration}.pth"
    if not checkpoint_path.exists():
        return None
    return checkpoint_path


def _peek_checkpoint_payload(model_path: Path, load_iteration: int):
    checkpoint_path = _resolve_checkpoint_path(model_path, load_iteration)
    if checkpoint_path is None:
        return None
    return torch.load(checkpoint_path.as_posix(), weights_only=False)


def render_split(modelset, scene, gaussians, pipe, split_name, cameras, output_root: Path):
    if not cameras:
        return None

    split_root = output_root / split_name
    light_stream, calc_stream = create_render_streams()
    local_axises = gaussians.get_local_axis
    asg_scales = gaussians.asg_func.get_asg_lam_miu
    asg_axises = gaussians.asg_func.get_asg_axis
    bg_color = [1, 1, 1, 1, 0, 0, 0] if modelset.white_background else [0, 0, 0, 0, 0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    render_args = {"modelset": modelset, "pipe": pipe, "bg_color": background}

    per_view = []
    running = {"l1": 0.0, "psnr": 0.0, "ssim": 0.0, "lpips": 0.0}

    for idx, view in enumerate(tqdm(cameras, desc=f"Rendering {split_name}")):
        render_pkg = render(view, gaussians, light_stream, calc_stream, local_axises, asg_scales, asg_axises, **render_args)
        metrics = _evaluate_view(modelset, view, render_pkg)

        stem = f"{idx:05d}_{view.image_name}"
        _save_image(metrics["image"], split_root / "render" / f"{stem}.png")
        _save_image(render_pkg["render"], split_root / "base" / f"{stem}.png")
        _save_image(render_pkg["shadow"], split_root / "shadow" / f"{stem}.png")
        _save_image(render_pkg["other_effects"], split_root / "other" / f"{stem}.png")
        _save_image(metrics["gt"], split_root / "gt" / f"{stem}.png")

        row = {
            "index": idx,
            "view": view.image_name,
            "l1": metrics["l1"],
            "psnr": metrics["psnr"],
            "ssim": metrics["ssim"],
            "lpips": metrics["lpips"],
        }
        per_view.append(row)
        for key in running:
            running[key] += row[key]

    count = len(per_view)
    summary = {
        "split": split_name,
        "num_views": count,
        "l1": running["l1"] / count,
        "psnr": running["psnr"] / count,
        "ssim": running["ssim"] / count,
        "lpips": running["lpips"] / count,
    }
    split_root.mkdir(parents=True, exist_ok=True)
    with (split_root / "results.json").open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_view": per_view}, f, indent=2)
    return summary


def render_sets(modelset, opt_args, pipe, load_iteration, skip_train, skip_test):
    modelset.data_device = "cpu"
    model_path = Path(modelset.model_path)
    checkpoint_path = _resolve_checkpoint_path(model_path, load_iteration)
    checkpoint_payload = None if checkpoint_path is None else torch.load(checkpoint_path.as_posix(), weights_only=False)
    _apply_checkpoint_model_hints(modelset, checkpoint_payload)
    gaussians = GaussianModel(modelset)
    checkpoint_scene_state = checkpoint_payload if isinstance(checkpoint_payload, dict) and "gaussians" in checkpoint_payload else None
    resolved_iteration = _resolve_iteration(model_path, load_iteration)
    point_cloud_exists = (
        resolved_iteration is not None
        and (model_path / "point_cloud" / f"iteration_{resolved_iteration}" / "point_cloud.ply").exists()
    )
    if checkpoint_path is not None and not point_cloud_exists:
        scene = Scene(
            modelset,
            gaussians,
            shuffle=False,
            skip_train=skip_train,
            skip_test=skip_test,
            checkpoint=checkpoint_path.as_posix(),
            checkpoint_state=checkpoint_scene_state,
        )
    else:
        scene = Scene(modelset, gaussians, load_iteration=load_iteration, shuffle=False, skip_train=skip_train, skip_test=skip_test)
    restored_checkpoint_path = _restore_checkpoint_if_available(Path(scene.model_path), scene.loaded_iter, gaussians, opt_args)

    output_root = Path(scene.model_path) / "render_eval" / f"iteration_{scene.loaded_iter:05d}"
    output_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "iteration": int(scene.loaded_iter),
        "checkpoint_path": None if restored_checkpoint_path is None else restored_checkpoint_path.as_posix(),
        "rasterizer": str(getattr(modelset, "rasterizer", "2dgs")),
        "use_textures": bool(getattr(modelset, "use_textures", False)),
        "texture_effect_mode": str(getattr(modelset, "texture_effect_mode", "per_uv")),
        "gaussian_backend": _checkpoint_backend(checkpoint_payload),
    }

    results = {"meta": metadata, "splits": {}}
    if not skip_train:
        train_summary = render_split(modelset, scene, gaussians, pipe, "train", scene.getTrainCameras(), output_root)
        if train_summary is not None:
            results["splits"]["train"] = train_summary
    if not skip_test:
        test_summary = render_split(modelset, scene, gaussians, pipe, "test", scene.getTestCameras(), output_root)
        if test_summary is not None:
            results["splits"]["test"] = test_summary

    with (output_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser(description="Render gs2dgs outputs for a saved iteration.")
    mp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    parser.add_argument("--iteration", "--load_iteration", dest="load_iteration", type=int, default=-1)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    try:
        args = get_combined_args(parser)
    except Exception:
        args = parser.parse_args()

    opt_parser = ArgumentParser(add_help=False)
    op = OptimizationParams(opt_parser)
    opt_defaults = opt_parser.parse_args([])
    for key, value in vars(opt_defaults).items():
        if not hasattr(args, key):
            setattr(args, key, value)

    safe_state(args.quiet)
    render_sets(
        mp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.load_iteration,
        args.skip_train,
        args.skip_test,
    )
