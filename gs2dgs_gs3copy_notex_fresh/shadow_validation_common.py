import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from PIL import Image, ImageDraw

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import GaussianModel, render
from make_mesh_shadow_dataset import build_camera_dirs, build_scene, normalize
from scene import Scene


_GT_SCENE_CACHE: Dict[str, dict] = {}


@dataclass
class ShadowRuntime:
    source_path: Path
    runtime_dir: Path
    modelset: object
    opt: object
    pipe: object
    gaussians: GaussianModel
    scene: Scene
    background: torch.Tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def ensure_blocking_cuda() -> None:
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")


def default_background_tensor(white_background: bool = True) -> torch.Tensor:
    values = [1, 1, 1, 1, 0, 0, 0] if white_background else [0, 0, 0, 0, 0, 0, 0]
    return torch.tensor(values, dtype=torch.float32, device="cuda")


def build_default_param_triplet(source_path: Path, model_path: Path):
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Shadow validation defaults")
    mp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    defaults = parser.parse_args([])

    modelset = mp.extract(defaults)
    opt = op.extract(defaults)
    pipe = pp.extract(defaults)

    modelset.source_path = os.path.abspath(str(source_path))
    modelset.model_path = str(model_path)
    modelset.white_background = True
    modelset.eval = True
    modelset.use_nerual_phasefunc = True
    modelset.cam_opt = False
    modelset.pl_opt = False
    modelset.view_num = 12
    modelset.load_num = 12
    modelset.offset = 0.1
    modelset.data_device = "cpu"

    return modelset, opt, pipe


def build_shadow_runtime(source_path: Path, runtime_dir: Path, seed: int = 0) -> ShadowRuntime:
    ensure_blocking_cuda()
    set_seed(seed)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    modelset, opt, pipe = build_default_param_triplet(source_path, runtime_dir)
    gaussians = GaussianModel(modelset, opt)
    scene = Scene(
        modelset,
        gaussians,
        opt=None,
        shuffle=False,
        valid=False,
        skip_train=False,
        skip_test=False,
    )

    if getattr(modelset, "use_nerual_phasefunc", False):
        try:
            gaussians.neural_phasefunc.eval()
        except Exception:
            pass

    return ShadowRuntime(
        source_path=source_path,
        runtime_dir=runtime_dir,
        modelset=modelset,
        opt=opt,
        pipe=pipe,
        gaussians=gaussians,
        scene=scene,
        background=default_background_tensor(True),
    )


def render_shadow(runtime: ShadowRuntime, viewpoint_camera, iteration: int = 0):
    local_axises = runtime.gaussians.get_local_axis
    asg_scales = runtime.gaussians.asg_func.get_asg_lam_miu
    asg_axises = runtime.gaussians.asg_func.get_asg_axis
    render_args = {
        "modelset": runtime.modelset,
        "pipe": runtime.pipe,
        "bg_color": runtime.background,
        "is_train": False,
        "asg_mlp": False,
        "fix_labert": True,
        "iteration": iteration,
    }
    return render(
        viewpoint_camera,
        runtime.gaussians,
        None,
        None,
        local_axises,
        asg_scales,
        asg_axises,
        **render_args,
    )


def clear_all_grads(parameters: Iterable[torch.nn.Parameter]) -> None:
    for param in parameters:
        if param.grad is not None:
            param.grad = None


def tensor_to_uint8_hwc(image: torch.Tensor) -> np.ndarray:
    image = torch.clamp(image.detach().cpu(), 0.0, 1.0)
    if image.ndim == 2:
        image = image.unsqueeze(0)
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    return (image.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)


def error_heatmap(prediction: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    error = torch.abs(prediction - target).detach().cpu()
    if error.ndim == 3:
        error = error.mean(dim=0)
    error = error.numpy()
    scale = max(float(error.max()), 1e-6)
    error = np.clip(error / scale, 0.0, 1.0)
    red = (error * 255.0 + 0.5).astype(np.uint8)
    green = ((1.0 - error) * 220.0 + 0.5).astype(np.uint8)
    blue = ((1.0 - error) * 96.0 + 0.5).astype(np.uint8)
    return np.stack([red, green, blue], axis=-1)


def save_shadow_images(
    output_dir: Path,
    gt_shadow: torch.Tensor,
    pred_shadow: torch.Tensor,
    prefix: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_img = Image.fromarray(tensor_to_uint8_hwc(gt_shadow))
    pred_img = Image.fromarray(tensor_to_uint8_hwc(pred_shadow))
    err_img = Image.fromarray(error_heatmap(pred_shadow, gt_shadow))
    gt_img.save(output_dir / f"{prefix}_gt_shadow.png")
    pred_img.save(output_dir / f"{prefix}_pred_shadow.png")
    err_img.save(output_dir / f"{prefix}_abs_error.png")


def save_before_after_panel(
    output_dir: Path,
    gt_shadow: torch.Tensor,
    before_shadow: torch.Tensor,
    after_shadow: torch.Tensor,
    before_label: str,
    after_label: str,
    footer_lines: Iterable[str],
    filename: str = "panel.png",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    footer = list(footer_lines)
    images = [
        Image.fromarray(tensor_to_uint8_hwc(gt_shadow)),
        Image.fromarray(tensor_to_uint8_hwc(before_shadow)),
        Image.fromarray(error_heatmap(before_shadow, gt_shadow)),
        Image.fromarray(tensor_to_uint8_hwc(after_shadow)),
        Image.fromarray(error_heatmap(after_shadow, gt_shadow)),
    ]
    labels = ["GT Shadow", before_label, "Error Before", after_label, "Error After"]
    w, h = images[0].size
    title_h = 24
    footer_h = 16 * max(len(footer), 1) + 12
    canvas = Image.new("RGB", (w * len(images), h + title_h + footer_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for idx, (label, img) in enumerate(zip(labels, images)):
        x = idx * w
        canvas.paste(img, (x, title_h))
        draw.text((x + 6, 4), label, fill=(0, 0, 0))
    for idx, line in enumerate(footer):
        draw.text((6, h + title_h + 8 + idx * 16), line, fill=(0, 0, 0))
    canvas.save(output_dir / filename)


def plot_scalar_curves(curves: Dict[str, Iterable[float]], out_path: Path, ylabel: str, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    for name, values in curves.items():
        plt.plot(list(values), label=name)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if len(curves) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _frame_name_variants(file_path: str):
    path = Path(file_path)
    return {path.stem, path.name}


def _build_shadow_reference_bundle(dataset_root: Path):
    cache_key = str(dataset_root.resolve())
    if cache_key in _GT_SCENE_CACHE:
        return _GT_SCENE_CACHE[cache_key]

    with (dataset_root / "scene_meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    with (dataset_root / "transforms_train.json").open("r", encoding="utf-8") as f:
        train = json.load(f)
    with (dataset_root / "transforms_test.json").open("r", encoding="utf-8") as f:
        test = json.load(f)

    objects, _, _ = build_scene(meta["preset"])
    ray_scene = o3d.t.geometry.RaycastingScene()
    for obj in objects:
        ray_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(obj.mesh))

    frame_lookup = {}
    for split_name, payload in [("train", train), ("test", test)]:
        for frame in payload["frames"]:
            for key in _frame_name_variants(frame["file_path"]):
                frame_lookup[(split_name, key)] = frame

    bundle = {
        "meta": meta,
        "ray_scene": ray_scene,
        "frame_lookup": frame_lookup,
        "camera_angle_x": float(train["camera_angle_x"]),
    }
    _GT_SCENE_CACHE[cache_key] = bundle
    return bundle


def render_shadow_reference(dataset_root: Path, split: str, image_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    bundle = _build_shadow_reference_bundle(dataset_root)
    frame = bundle["frame_lookup"].get((split, image_name))
    if frame is None:
        raise KeyError(f"Could not find frame for split={split}, image_name={image_name}")

    meta = bundle["meta"]
    c2w = np.asarray(frame["transform_matrix"], dtype=np.float32)
    light_pos = np.asarray(frame["pl_pos"], dtype=np.float32)
    shadow_np, hit_mask_np = render_shadow_reference_np(
        ray_scene=bundle["ray_scene"],
        c2w=c2w,
        width=int(meta["width"]),
        height=int(meta["height"]),
        fov_x=float(bundle["camera_angle_x"]),
        light_pos=light_pos,
    )
    shadow = torch.from_numpy(shadow_np).unsqueeze(0).cuda()
    hit_mask = torch.from_numpy(hit_mask_np).unsqueeze(0).cuda()
    return shadow, hit_mask


def render_shadow_reference_np(
    ray_scene,
    c2w: np.ndarray,
    width: int,
    height: int,
    fov_x: float,
    light_pos: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    cam_dirs = build_camera_dirs(width, height, fov_x)
    rotation = c2w[:3, :3]
    camera_pos = c2w[:3, 3]
    world_dirs = normalize(cam_dirs @ rotation.T).astype(np.float32)
    origins = np.repeat(camera_pos[None, :], width * height, axis=0).astype(np.float32)
    rays = np.concatenate([origins, world_dirs], axis=1)
    cast = ray_scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))

    t_hit = cast["t_hit"].numpy()
    normals = normalize(cast["primitive_normals"].numpy()).astype(np.float32)
    shadow = np.ones((width * height, 1), dtype=np.float32)
    hit_mask = np.zeros((width * height, 1), dtype=np.float32)

    visible = np.isfinite(t_hit)
    if np.any(visible):
        hit_points = origins[visible] + world_dirs[visible] * t_hit[visible, None]
        hit_normals = normals[visible]
        light_vec = light_pos[None, :] - hit_points
        light_dist = np.linalg.norm(light_vec, axis=1)
        light_dir = normalize(light_vec).astype(np.float32)
        shadow_origins = hit_points + hit_normals * 2e-3 + light_dir * 1e-3
        shadow_rays = np.concatenate([shadow_origins, light_dir], axis=1)
        shadow_cast = ray_scene.cast_rays(o3d.core.Tensor(shadow_rays, dtype=o3d.core.Dtype.Float32))
        shadow_t = shadow_cast["t_hit"].numpy()
        shadow_hit = np.isfinite(shadow_t) & (shadow_t < (light_dist - 5e-3))
        shadow[visible, 0] = np.where(shadow_hit, 0.0, 1.0)
        hit_mask[visible, 0] = 1.0

    return shadow.reshape(height, width), hit_mask.reshape(height, width)


def masked_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if prediction.ndim == 2:
        prediction = prediction.unsqueeze(0)
    if target.ndim == 2:
        target = target.unsqueeze(0)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    masked_error = torch.abs(prediction - target) * mask
    denom = mask.sum().clamp_min(1.0)
    return masked_error.sum() / denom


def gradient_summary(parameter: torch.nn.Parameter, row_mask: Optional[torch.Tensor] = None) -> dict:
    grad = parameter.grad
    if grad is None:
        return {
            "present": False,
            "finite": True,
            "absmax": 0.0,
            "meanabs": 0.0,
            "nonzero": 0,
        }
    grad_view = grad.detach()
    if row_mask is not None:
        grad_view = grad_view[row_mask]
    finite = bool(torch.isfinite(grad_view).all().item())
    abs_view = grad_view.abs()
    return {
        "present": True,
        "finite": finite,
        "absmax": float(abs_view.max().item()) if abs_view.numel() > 0 else 0.0,
        "meanabs": float(abs_view.mean().item()) if abs_view.numel() > 0 else 0.0,
        "nonzero": int((abs_view > 0).sum().item()),
    }


def collect_gaussian_gradient_summary(
    gaussians: GaussianModel,
    row_mask: Optional[torch.Tensor] = None,
) -> dict:
    return {
        "xyz": gradient_summary(gaussians._xyz, row_mask),
        "scale": gradient_summary(gaussians._scaling, row_mask),
        "opacity": gradient_summary(gaussians._opacity, row_mask),
        "rotation": gradient_summary(gaussians._rotation, row_mask),
    }


def optimizer_lr_for_group(runtime: ShadowRuntime, group_name: str) -> float:
    if group_name == "xyz":
        return float(runtime.opt.position_lr_init * runtime.gaussians.spatial_lr_scale)
    if group_name == "scale":
        return float(runtime.opt.scaling_lr)
    if group_name == "opacity":
        return float(runtime.opt.opacity_lr)
    raise ValueError(f"Unsupported group name: {group_name}")


def inverse_sigmoid_tensor(probability: torch.Tensor) -> torch.Tensor:
    probability = torch.clamp(probability, 1e-6, 1.0 - 1e-6)
    return torch.log(probability / (1.0 - probability))
