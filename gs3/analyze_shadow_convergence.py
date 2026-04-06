import csv
import json
import math
import os
from argparse import ArgumentParser
from pathlib import Path

import lpips
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim


def tensor_to_uint8_chw(image: torch.Tensor) -> np.ndarray:
    image = torch.clamp(image.detach().cpu(), 0.0, 1.0)
    return (image.numpy() * 255.0 + 0.5).astype(np.uint8)


def tensor_to_uint8_hwc(image: torch.Tensor) -> np.ndarray:
    return np.transpose(tensor_to_uint8_chw(image), (1, 2, 0))


def error_heatmap(rendered: torch.Tensor, gt: torch.Tensor) -> np.ndarray:
    err = torch.mean(torch.abs(rendered - gt), dim=0).detach().cpu().numpy()
    err = np.clip(err / max(float(err.max()), 1e-6), 0.0, 1.0)
    cmap = plt.get_cmap("inferno")
    return (cmap(err)[..., :3] * 255.0 + 0.5).astype(np.uint8)


def shadow_to_rgb(shadow: torch.Tensor) -> np.ndarray:
    shadow = torch.clamp(shadow.detach().cpu(), 0.0, 1.0)
    if shadow.ndim == 2:
        shadow = shadow.unsqueeze(0)
    if shadow.shape[0] == 1:
        shadow = shadow.repeat(3, 1, 1)
    return tensor_to_uint8_hwc(shadow)


def make_panel(gt: torch.Tensor, rendered: torch.Tensor, shadow: torch.Tensor, metrics: dict, out_path: Path):
    gt_img = Image.fromarray(tensor_to_uint8_hwc(gt))
    render_img = Image.fromarray(tensor_to_uint8_hwc(rendered))
    err_img = Image.fromarray(error_heatmap(rendered, gt))
    shadow_img = Image.fromarray(shadow_to_rgb(shadow))

    w, h = gt_img.size
    title_h = 24
    footer_h = 44
    canvas = Image.new("RGB", (w * 4, h + title_h + footer_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    labels = ["GT", "Render", "Abs Error", "Shadow"]
    images = [gt_img, render_img, err_img, shadow_img]
    for idx, (label, img) in enumerate(zip(labels, images)):
        x = idx * w
        canvas.paste(img, (x, title_h))
        draw.text((x + 6, 4), label, fill=(0, 0, 0))

    footer = (
        f"L1 {metrics['l1']:.4f}  "
        f"PSNR {metrics['psnr']:.2f}  "
        f"SSIM {metrics['ssim']:.4f}  "
        f"LPIPS {metrics['lpips']:.4f}"
    )
    draw.text((6, h + title_h + 10), footer, fill=(0, 0, 0))
    canvas.save(out_path)


def build_model_and_scene(model_params, optim_params, iteration: int):
    gaussians = GaussianModel(model_params, optim_params)
    checkpoint_path = Path(model_params.model_path) / f"chkpnt{iteration}.pth"
    if checkpoint_path.exists():
        scene = Scene(
            model_params,
            gaussians,
            opt=optim_params,
            checkpoint=str(checkpoint_path),
            shuffle=False,
            valid=False,
            skip_train=False,
            skip_test=False,
        )
        model_args, _ = torch.load(checkpoint_path, weights_only=False)
        gaussians.restore(model_args, optim_params)
    else:
        scene = Scene(model_params, gaussians, load_iteration=iteration, shuffle=False, valid=False, skip_train=False, skip_test=False)
        print(f"Warning: checkpoint not found, evaluating PLY only: {checkpoint_path}")

    if model_params.use_nerual_phasefunc and hasattr(gaussians, "neural_phasefunc") and gaussians.neural_phasefunc is not None:
        try:
            gaussians.neural_phasefunc.eval()
        except Exception:
            pass
    return gaussians, scene


def evaluate_iteration(model_params, optim_params, pipeline_params, iteration: int, panels_per_split: int, output_root: Path):
    gaussians, scene = build_model_and_scene(model_params, optim_params, iteration)
    background = torch.tensor(
        [1, 1, 1, 1, 0, 0, 0] if model_params.white_background else [0, 0, 0, 0, 0, 0, 0],
        dtype=torch.float32,
        device="cuda",
    )
    render_args = {
        "modelset": model_params,
        "pipe": pipeline_params,
        "bg_color": background,
        "is_train": False,
        "asg_mlp": False,
        "iteration": iteration,
    }
    if iteration < optim_params.spcular_freeze_step + optim_params.fit_linear_step:
        render_args["fix_labert"] = True
    light_stream = torch.cuda.Stream()
    calc_stream = torch.cuda.Stream()

    summary = {}
    for split_name, cameras in [("test", scene.getTestCameras()), ("train", scene.getTrainCameras())]:
        split_dir = output_root / f"iter_{iteration}" / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        local_axises = scene.gaussians.get_local_axis
        asg_scales = scene.gaussians.asg_func.get_asg_lam_miu
        asg_axises = scene.gaussians.asg_func.get_asg_axis

        for idx, viewpoint_cam in enumerate(cameras):
            render_pkg = render(viewpoint_cam, scene.gaussians, light_stream, calc_stream, local_axises, asg_scales, asg_axises, **render_args)
            mimage = render_pkg["render"]
            shadow = render_pkg["shadow"]
            other_effects = render_pkg["other_effects"]
            image = mimage * shadow + other_effects

            if model_params.gamma_change:
                image = torch.clamp(image, 0.0, 1.0)
                image = image / (1.0 + image)
                image = torch.clamp(image, 0.0, 1.0)
            else:
                image = torch.clamp(image, 0.0, 1.0)

            gt_image = torch.clamp(viewpoint_cam.original_image.to("cuda"), 0.0, 1.0)
            metrics = {
                "view": viewpoint_cam.image_name,
                "l1": float(l1_loss(image, gt_image).mean().double().item()),
                "psnr": float(psnr(image, gt_image).mean().double().item()),
                "ssim": float(ssim(image, gt_image).mean().double().item()),
                "lpips": float(loss_fn_vgg.forward(image, gt_image).squeeze().item()),
            }
            rows.append(metrics)

            if idx < panels_per_split:
                panel_path = split_dir / f"{idx:03d}_{viewpoint_cam.image_name}_panel.png"
                make_panel(gt_image, image, shadow, metrics, panel_path)

        with (split_dir / "per_view_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["view", "l1", "psnr", "ssim", "lpips"])
            writer.writeheader()
            writer.writerows(rows)

        mean_metrics = {
            "l1": float(np.mean([r["l1"] for r in rows])),
            "psnr": float(np.mean([r["psnr"] for r in rows])),
            "ssim": float(np.mean([r["ssim"] for r in rows])),
            "lpips": float(np.mean([r["lpips"] for r in rows])),
            "num_views": len(rows),
        }
        summary[split_name] = mean_metrics

    del scene
    del gaussians
    torch.cuda.empty_cache()
    return summary


def plot_metric_trends(summaries, output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)
    iterations = sorted(summaries.keys())
    for split in ["test", "train"]:
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        metric_names = ["psnr", "ssim", "lpips", "l1"]
        for ax, metric in zip(axes.flatten(), metric_names):
            values = [summaries[it][split][metric] for it in iterations]
            ax.plot(iterations, values, marker="o")
            ax.set_title(f"{split} {metric}")
            ax.set_xlabel("iteration")
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_root / f"{split}_metric_trends.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    parser = ArgumentParser(description="Analyze convergence with GT comparisons and visual panels.")
    mp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--load_iterations", nargs="+", type=int, required=True)
    parser.add_argument("--panels_per_split", type=int, default=3)
    parser.add_argument("--analysis_dir", type=str, default=None)

    args = get_combined_args(parser)
    model_params = mp.extract(args)
    optim_params = op.extract(args)
    pipeline_params = pp.extract(args)
    model_params.data_device = "cpu"

    analysis_dir_arg = getattr(args, "analysis_dir", None)
    analysis_dir = Path(analysis_dir_arg) if analysis_dir_arg else Path(model_params.model_path) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    loss_fn_vgg = lpips.LPIPS(net="vgg").cuda()
    loss_fn_vgg.eval()

    all_summaries = {}
    for iteration in args.load_iterations:
        print(f"Evaluating iteration {iteration}")
        all_summaries[iteration] = evaluate_iteration(
            model_params=model_params,
            optim_params=optim_params,
            pipeline_params=pipeline_params,
            iteration=iteration,
            panels_per_split=args.panels_per_split,
            output_root=analysis_dir,
        )

    with (analysis_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    with (analysis_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "split", "l1", "psnr", "ssim", "lpips", "num_views"])
        for iteration in sorted(all_summaries.keys()):
            for split in ["test", "train"]:
                row = all_summaries[iteration][split]
                writer.writerow([iteration, split, row["l1"], row["psnr"], row["ssim"], row["lpips"], row["num_views"]])

    plot_metric_trends(all_summaries, analysis_dir)
    print(json.dumps(all_summaries, indent=2))
