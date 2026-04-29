import csv
import json
from argparse import ArgumentParser
from pathlib import Path

import torch

from shadow_validation_common import (
    build_shadow_runtime,
    clear_all_grads,
    collect_gaussian_gradient_summary,
    inverse_sigmoid_tensor,
    masked_l1,
    optimizer_lr_for_group,
    plot_scalar_curves,
    render_shadow,
    render_shadow_reference,
    save_before_after_panel,
)


def parameter_error(gaussians, originals, mask: torch.Tensor, group: str) -> float:
    if group == "xyz":
        current = gaussians._xyz[mask]
        reference = originals["xyz"][mask]
        return float(torch.linalg.norm(current - reference, dim=-1).mean().item())
    if group == "scale":
        current = gaussians.get_scaling[mask]
        reference = originals["scale"][mask]
        return float(torch.abs(current - reference).mean().item())
    if group == "opacity":
        current = gaussians.get_opacity[mask]
        reference = originals["opacity"][mask]
        return float(torch.abs(current - reference).mean().item())
    raise ValueError(group)


def set_trainable_group(gaussians, group: str):
    tracked = [
        gaussians._xyz,
        gaussians._scaling,
        gaussians._opacity,
        gaussians._rotation,
        gaussians.kd,
        gaussians.ks,
        gaussians.alpha_asg,
        gaussians.local_q,
        gaussians.neural_material,
    ]
    flags = [(param, bool(param.requires_grad)) for param in tracked]
    for param, _ in flags:
        param.requires_grad_(False)
    module_flags = []
    for module in [gaussians.asg_func, gaussians.neural_phasefunc]:
        for param in module.parameters():
            module_flags.append((param, bool(param.requires_grad)))
            param.requires_grad_(False)

    if group == "xyz":
        gaussians._xyz.requires_grad_(True)
        return [gaussians._xyz], flags + module_flags
    if group == "scale":
        gaussians._scaling.requires_grad_(True)
        return [gaussians._scaling], flags + module_flags
    if group == "opacity":
        gaussians._opacity.requires_grad_(True)
        return [gaussians._opacity], flags + module_flags
    raise ValueError(group)


def restore_trainable_flags(flags):
    for param, value in flags:
        param.requires_grad_(value)


def log_scalar(value: float) -> float:
    return float(torch.log(torch.tensor(value, dtype=torch.float32)).item())


def apply_perturbation(runtime, mask: torch.Tensor, group: str, light_pos: torch.Tensor):
    xyz = runtime.gaussians.get_xyz.detach()
    selected_center = xyz[mask].mean(dim=0)
    light_dir = selected_center - light_pos
    light_dir = light_dir / light_dir.norm().clamp_min(1e-8)

    with torch.no_grad():
        if group == "xyz":
            runtime.gaussians._xyz[mask] += light_dir * 0.12
        elif group == "scale":
            runtime.gaussians._scaling[mask] += log_scalar(1.20)
        elif group == "opacity":
            actual = runtime.gaussians.get_opacity.detach().clone()
            actual[mask] = (actual[mask] * 0.70).clamp(1e-4, 1.0 - 1e-4)
            runtime.gaussians._opacity.copy_(inverse_sigmoid_tensor(actual))
        else:
            raise ValueError(group)


def save_curve_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_single_experiment(
    dataset_root: Path,
    output_root: Path,
    group: str,
    steps: int,
    seed: int,
) -> dict:
    runtime_dir = output_root / "_runtime" / f"{dataset_root.name}_{group}"
    runtime = build_shadow_runtime(dataset_root, runtime_dir, seed=seed)
    fixed_camera = sorted(runtime.scene.getTestCameras(), key=lambda cam: cam.image_name)[0]
    opt_cameras = sorted(runtime.scene.getTrainCameras(), key=lambda cam: cam.image_name)
    fixed_gt_shadow, fixed_hit_mask = render_shadow_reference(dataset_root, "test", fixed_camera.image_name)
    opt_targets = []
    for camera in opt_cameras:
        gt_shadow, hit_mask = render_shadow_reference(dataset_root, "train", camera.image_name)
        opt_targets.append((camera, gt_shadow, hit_mask))

    with (dataset_root / "scene_meta.json").open("r", encoding="utf-8") as f:
        scene_meta = json.load(f)
    light_pos = torch.tensor(scene_meta["light_pos"], dtype=torch.float32, device="cuda")

    originals = {
        "xyz": runtime.gaussians._xyz.detach().clone(),
        "scale": runtime.gaussians.get_scaling.detach().clone(),
        "opacity": runtime.gaussians.get_opacity.detach().clone(),
    }

    xyz = originals["xyz"]
    mask = ((xyz[:, 2] > 0.35) & ((xyz[:, 0] ** 2 + xyz[:, 1] ** 2) < (0.9 ** 2))).detach()
    if int(mask.sum().item()) == 0:
        raise RuntimeError(f"Selection mask is empty for dataset {dataset_root.name}")

    apply_perturbation(runtime, mask, group, light_pos)
    optim_params, flags = set_trainable_group(runtime.gaussians, group)
    optimizer = torch.optim.Adam(optim_params, lr=optimizer_lr_for_group(runtime, group))

    before_pkg = render_shadow(runtime, fixed_camera, iteration=0)
    before_shadow = before_pkg["shadow"].detach()
    before_loss = float(masked_l1(before_shadow, fixed_gt_shadow, fixed_hit_mask).item())
    before_error = parameter_error(runtime.gaussians, originals, mask, group)

    curves = []
    first_grad_summary = None
    params = [
        runtime.gaussians._xyz,
        runtime.gaussians._scaling,
        runtime.gaussians._opacity,
        runtime.gaussians._rotation,
    ]
    for step in range(steps):
        clear_all_grads(params)
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for camera, gt_shadow, hit_mask in opt_targets:
            render_pkg = render_shadow(runtime, camera, iteration=0)
            pred_shadow = render_pkg["shadow"]
            losses.append(masked_l1(pred_shadow, gt_shadow, hit_mask))
        loss = torch.stack(losses).mean()
        loss.backward()
        if first_grad_summary is None:
            first_grad_summary = collect_gaussian_gradient_summary(runtime.gaussians, mask)
        optimizer.step()
        curves.append(
            {
                "step": step,
                "shadow_loss": float(loss.item()),
                "parameter_error": parameter_error(runtime.gaussians, originals, mask, group),
            }
        )

    after_pkg = render_shadow(runtime, fixed_camera, iteration=0)
    after_shadow = after_pkg["shadow"].detach()
    after_loss = float(masked_l1(after_shadow, fixed_gt_shadow, fixed_hit_mask).item())
    after_error = parameter_error(runtime.gaussians, originals, mask, group)
    best_shadow_idx = min(range(len(curves)), key=lambda idx: curves[idx]["shadow_loss"])
    best_param_idx = min(range(len(curves)), key=lambda idx: curves[idx]["parameter_error"])

    exp_dir = output_root / dataset_root.name / group
    footer = [
        f"view={fixed_camera.image_name}  selected={int(mask.sum().item())}",
        f"loss before={before_loss:.6f}  after={after_loss:.6f}",
        f"param error before={before_error:.6f}  after={after_error:.6f}",
    ]
    save_before_after_panel(
        exp_dir,
        fixed_gt_shadow,
        before_shadow,
        after_shadow,
        before_label="Perturbed",
        after_label="Shadow-Optimized",
        footer_lines=footer,
    )
    plot_scalar_curves(
        {"shadow_loss": [row["shadow_loss"] for row in curves]},
        exp_dir / "shadow_loss_curve.png",
        "shadow L1",
        f"{dataset_root.name} {group}",
    )
    plot_scalar_curves(
        {"parameter_error": [row["parameter_error"] for row in curves]},
        exp_dir / "parameter_error_curve.png",
        "parameter error",
        f"{dataset_root.name} {group}",
    )
    save_curve_csv(exp_dir / "curve.csv", curves)

    summary = {
        "dataset": dataset_root.name,
        "group": group,
        "view": fixed_camera.image_name,
        "selected_count": int(mask.sum().item()),
        "steps": steps,
        "optimization_views": len(opt_targets),
        "learning_rate": optimizer_lr_for_group(runtime, group),
        "shadow_loss_before": before_loss,
        "shadow_loss_after": after_loss,
        "parameter_error_before": before_error,
        "parameter_error_after": after_error,
        "shadow_loss_improved": after_loss < before_loss,
        "parameter_error_improved": after_error < before_error,
        "best_shadow_step": int(curves[best_shadow_idx]["step"]),
        "best_shadow_loss": float(curves[best_shadow_idx]["shadow_loss"]),
        "best_parameter_error_step": int(curves[best_param_idx]["step"]),
        "best_parameter_error": float(curves[best_param_idx]["parameter_error"]),
        "best_shadow_loss_improved": float(curves[best_shadow_idx]["shadow_loss"]) < before_loss,
        "best_parameter_error_improved": float(curves[best_param_idx]["parameter_error"]) < before_error,
        "initial_gradient_summary": first_grad_summary,
    }
    with (exp_dir / "gradient_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    restore_trainable_flags(flags)
    return summary


def main():
    parser = ArgumentParser(description="Validate shadow-only recovery on synthetic datasets.")
    parser.add_argument("--single_dataset", type=str, default="synthetic_shadow_single_object")
    parser.add_argument("--mesh_dataset", type=str, default="synthetic_shadow_mesh")
    parser.add_argument("--output_root", type=str, default="output/shadow_gradient_validation/synthetic_recovery")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    single_root = Path(args.single_dataset).resolve()
    mesh_root = Path(args.mesh_dataset).resolve()

    single_results = {}
    for group in ["xyz", "scale", "opacity"]:
        single_results[group] = run_single_experiment(
            dataset_root=single_root,
            output_root=output_root,
            group=group,
            steps=args.steps,
            seed=args.seed,
        )

    groups_with_param_improve = [
        name for name, result in single_results.items() if result["best_parameter_error_improved"]
    ]
    if groups_with_param_improve:
        best_group = min(
            groups_with_param_improve,
            key=lambda name: single_results[name]["best_parameter_error"] / max(single_results[name]["parameter_error_before"], 1e-8),
        )
    else:
        best_group = min(
            single_results.keys(),
            key=lambda name: single_results[name]["best_shadow_loss"] / max(single_results[name]["shadow_loss_before"], 1e-8),
        )
    mesh_result = run_single_experiment(
        dataset_root=mesh_root,
        output_root=output_root,
        group=best_group,
        steps=args.steps,
        seed=args.seed,
    )

    summary = {
        "single_object": single_results,
        "mesh_followup_group": best_group,
        "mesh_followup": mesh_result,
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
