import json
from argparse import ArgumentParser
from pathlib import Path

import torch

from shadow_validation_common import (
    build_shadow_runtime,
    clear_all_grads,
    collect_gaussian_gradient_summary,
    masked_l1,
    render_shadow,
    render_shadow_reference,
    save_shadow_images,
)


def run_probe_for_dataset(dataset_root: Path, output_root: Path, seed: int) -> dict:
    runtime_dir = output_root / "_runtime" / dataset_root.name
    runtime = build_shadow_runtime(dataset_root, runtime_dir, seed=seed)
    dataset_output = output_root / dataset_root.name
    dataset_output.mkdir(parents=True, exist_ok=True)

    cameras_by_split = {
        "train": sorted(runtime.scene.getTrainCameras(), key=lambda cam: cam.image_name)[:1],
        "test": sorted(runtime.scene.getTestCameras(), key=lambda cam: cam.image_name)[:1],
    }

    summaries = {}
    params = [
        runtime.gaussians._xyz,
        runtime.gaussians._scaling,
        runtime.gaussians._opacity,
        runtime.gaussians._rotation,
    ]
    for split, cameras in cameras_by_split.items():
        for camera in cameras:
            clear_all_grads(params)
            render_pkg = render_shadow(runtime, camera, iteration=0)
            pred_shadow = render_pkg["shadow"]
            gt_shadow, hit_mask = render_shadow_reference(dataset_root, split, camera.image_name)
            loss = masked_l1(pred_shadow, gt_shadow, hit_mask)
            loss.backward()
            summary = {
                "dataset": dataset_root.name,
                "split": split,
                "view": camera.image_name,
                "loss": float(loss.item()),
                "pred_shadow_finite": bool(torch.isfinite(pred_shadow).all().item()),
                "gt_shadow_finite": bool(torch.isfinite(gt_shadow).all().item()),
                "hit_mask_pixels": int(hit_mask.sum().item()),
                "gradients": collect_gaussian_gradient_summary(runtime.gaussians),
            }
            summaries[f"{split}/{camera.image_name}"] = summary

            view_dir = dataset_output / split / camera.image_name
            save_shadow_images(view_dir, gt_shadow, pred_shadow, prefix="probe")
            with (view_dir / "gradient_summary.json").open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

    with (dataset_output / "probe_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    return summaries


def main():
    parser = ArgumentParser(description="Run shadow-gradient connectivity probes on synthetic datasets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["synthetic_shadow_single_object", "synthetic_shadow_mesh"],
    )
    parser.add_argument("--output_root", type=str, default="output/shadow_gradient_validation/probe")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    aggregate = {}
    for dataset in args.datasets:
        dataset_root = Path(dataset).resolve()
        aggregate[dataset_root.name] = run_probe_for_dataset(dataset_root, output_root, args.seed)

    with (output_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
