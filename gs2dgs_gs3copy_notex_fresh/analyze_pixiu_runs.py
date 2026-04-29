import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def read_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_image(path: Path):
    image = Image.open(path)
    arr = np.asarray(image).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr


def luminance(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def foreground_mask(gt):
    if gt.shape[-1] == 4:
        return gt[..., 3] > 0.0

    rgb = gt[..., :3]
    lum = luminance(rgb)
    near_white = np.all(rgb > 0.98, axis=-1)
    near_black = np.all(rgb < 0.02, axis=-1)

    if np.mean(near_white) > 0.25:
        mask = ~near_white
    elif np.mean(near_black) > 0.25:
        mask = ~near_black
    else:
        mask = np.ones(lum.shape, dtype=bool)
    return mask


def safe_mean(values):
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def image_metrics_for_view(view_dir: Path):
    gt = load_image(view_dir / "gt.png")
    render = load_image(view_dir / "render.png")
    base = load_image(view_dir / "base.png")

    gt_rgb = gt[..., :3]
    render_rgb = render[..., :3]
    base_rgb = base[..., :3]

    fg = foreground_mask(gt)
    gt_luma = luminance(gt_rgb)
    render_luma = luminance(render_rgb)
    base_luma = luminance(base_rgb)

    fg_values = gt_luma[fg]
    if fg_values.size == 0:
        fg = np.ones(gt_luma.shape, dtype=bool)
        fg_values = gt_luma[fg]

    q25 = np.quantile(fg_values, 0.25)
    q75 = np.quantile(fg_values, 0.75)
    dark_fg = fg & (gt_luma <= q25)
    lit_fg = fg & (gt_luma >= q75)

    rgb_err = np.abs(render_rgb - gt_rgb).mean(axis=-1)
    luma_err = np.abs(render_luma - gt_luma)

    base_dark = base_luma[dark_fg]
    base_lit = base_luma[lit_fg]
    render_dark = render_luma[dark_fg]
    render_lit = render_luma[lit_fg]

    return {
        "dark_fg_rgb_l1": safe_mean(rgb_err[dark_fg]),
        "dark_fg_luma_l1": safe_mean(luma_err[dark_fg]),
        "base_dark_to_lit_ratio": float(np.mean(base_dark) / max(float(np.mean(base_lit)), 1e-8))
        if base_dark.size and base_lit.size
        else float("nan"),
        "render_dark_to_lit_ratio": float(np.mean(render_dark) / max(float(np.mean(render_lit)), 1e-8))
        if render_dark.size and render_lit.size
        else float("nan"),
    }


def aggregate_fixed_view_metrics(run_dir: Path, iteration: int, split: str):
    root = run_dir / "convergence" / "fixed_views" / f"iter_{iteration:05d}" / split
    if not root.exists():
        return {}
    rows = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "gt.png").exists():
            rows.append(image_metrics_for_view(child))
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: safe_mean(np.array([row[k] for row in rows], dtype=np.float32)) for k in keys}


def parse_eval_metrics(run_dir: Path):
    rows = read_jsonl(run_dir / "convergence" / "eval_metrics.jsonl")
    by_split = {"test": {}, "train": {}}
    for row in rows:
        split = row.get("split")
        iteration = int(row.get("iteration"))
        by_split.setdefault(split, {})[iteration] = row
    return by_split


def checkpoint_geometry_metrics(ckpt_path: Path):
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model_params, iteration = ckpt
    scaling = model_params["_scaling"]
    if not torch.is_tensor(scaling):
        scaling = torch.tensor(scaling)
    scales = torch.exp(scaling.float()).detach().cpu().numpy()
    min_axis = scales.min(axis=1)
    max_axis = scales.max(axis=1)
    anisotropy = max_axis / np.maximum(min_axis, 1e-12)

    return {
        "iteration": int(iteration),
        "num_points": int(scales.shape[0]),
        "min_axis_median": float(np.median(min_axis)),
        "anisotropy_median": float(np.median(anisotropy)),
        "line_like_ratio": float(np.mean(anisotropy > 20.0)),
        "tiny_axis_ratio": float(np.mean(min_axis < 1e-6)),
    }


def all_checkpoint_metrics(run_dir: Path):
    rows = []
    for ckpt_path in sorted(run_dir.glob("chkpnt*.pth")):
        rows.append(checkpoint_geometry_metrics(ckpt_path))
    return rows


def densify_summary(run_dir: Path):
    rows = read_jsonl(run_dir / "convergence" / "densify_events.jsonl")
    if not rows:
        return {}
    clones = np.array([r.get("num_clone", 0) for r in rows], dtype=np.float32)
    splits = np.array([r.get("num_split", 0) for r in rows], dtype=np.float32)
    pruned = np.array([r.get("num_pruned", 0) for r in rows], dtype=np.float32)
    grad_p95 = np.array([r.get("visible_grad_p95", float("nan")) for r in rows], dtype=np.float32)
    return {
        "densify_events": int(len(rows)),
        "clone_total": float(np.nansum(clones)),
        "split_total": float(np.nansum(splits)),
        "pruned_total": float(np.nansum(pruned)),
        "visible_grad_p95_max": float(np.nanmax(grad_p95)),
    }


def run_label(run_dir: Path):
    return run_dir.name


def final_iteration_from_eval(eval_by_split):
    test_rows = eval_by_split.get("test", {})
    if not test_rows:
        return None
    return max(test_rows.keys())


def extract_run_summary(run_dir: Path):
    eval_by_split = parse_eval_metrics(run_dir)
    final_iter = final_iteration_from_eval(eval_by_split)
    if final_iter is None:
        return None

    test_row = eval_by_split["test"][final_iter]
    train_row = eval_by_split.get("train", {}).get(final_iter, {})
    fixed_test = aggregate_fixed_view_metrics(run_dir, final_iter, "test")
    geom_rows = all_checkpoint_metrics(run_dir)
    geom_final = max(geom_rows, key=lambda x: x["iteration"]) if geom_rows else {}
    densify = densify_summary(run_dir)

    summary = {
        "run": run_label(run_dir),
        "final_iteration": final_iter,
        "test_l1": test_row.get("l1"),
        "test_psnr": test_row.get("psnr"),
        "test_ssim": test_row.get("ssim"),
        "test_lpips": test_row.get("lpips"),
        "train_l1": train_row.get("l1"),
        "train_psnr": train_row.get("psnr"),
        "train_ssim": train_row.get("ssim"),
        "train_lpips": train_row.get("lpips"),
        "dark_fg_rgb_l1": fixed_test.get("dark_fg_rgb_l1"),
        "dark_fg_luma_l1": fixed_test.get("dark_fg_luma_l1"),
        "base_dark_to_lit_ratio": fixed_test.get("base_dark_to_lit_ratio"),
        "render_dark_to_lit_ratio": fixed_test.get("render_dark_to_lit_ratio"),
        "num_points": geom_final.get("num_points"),
        "min_axis_median": geom_final.get("min_axis_median"),
        "anisotropy_median": geom_final.get("anisotropy_median"),
        "line_like_ratio": geom_final.get("line_like_ratio"),
        "tiny_axis_ratio": geom_final.get("tiny_axis_ratio"),
        "densify_events": densify.get("densify_events"),
        "clone_total": densify.get("clone_total"),
        "split_total": densify.get("split_total"),
        "pruned_total": densify.get("pruned_total"),
        "visible_grad_p95_max": densify.get("visible_grad_p95_max"),
        "run_dir": str(run_dir),
    }
    return summary, eval_by_split, geom_rows


def make_metric_plot(path: Path, run_data, split: str, metric: str):
    if plt is None:
        return
    plt.figure(figsize=(8, 5))
    for run_name, eval_by_split in run_data.items():
        points = sorted(eval_by_split.get(split, {}).items())
        if not points:
            continue
        xs = [it for it, _ in points]
        ys = [row.get(metric, float("nan")) for _, row in points]
        plt.plot(xs, ys, marker="o", label=run_name)
    plt.xlabel("Iteration")
    plt.ylabel(metric)
    plt.title(f"{split} {metric}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def make_geometry_plot(path: Path, geom_data, metric: str):
    if plt is None:
        return
    plt.figure(figsize=(8, 5))
    for run_name, rows in geom_data.items():
        if not rows:
            continue
        xs = [row["iteration"] for row in rows]
        ys = [row.get(metric, float("nan")) for row in rows]
        plt.plot(xs, ys, marker="o", label=run_name)
    plt.xlabel("Iteration")
    plt.ylabel(metric)
    plt.title(metric)
    plt.grid(True, alpha=0.3)
    plt.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def make_panel_grid(path: Path, run_dirs, iteration: int, split: str):
    panels = []
    labels = []
    for run_dir in run_dirs:
        panel = run_dir / "convergence" / "fixed_views" / f"iter_{iteration:05d}" / split / "panel.png"
        if panel.exists():
            panels.append(Image.open(panel).convert("RGB"))
            labels.append(run_dir.name)
    if not panels:
        return

    width = max(p.width for p in panels)
    title_h = 32
    total_h = sum(p.height + title_h for p in panels)
    canvas = Image.new("RGB", (width, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    y = 0
    for label, panel in zip(labels, panels):
        draw.rectangle([0, y, width, y + title_h], fill=(240, 240, 240))
        draw.text((10, y + 8), label, fill=(0, 0, 0))
        canvas.paste(panel, (0, y + title_h))
        y += title_h + panel.height
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def write_markdown_report(path: Path, summaries):
    lines = [
        "# Pixiu Run Analysis",
        "",
        "| run | iter | test_psnr | test_lpips | dark_fg_rgb_l1 | dark_fg_luma_l1 | base_dark_to_lit_ratio | num_points | line_like_ratio | tiny_axis_ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {run} | {final_iteration} | {test_psnr:.4f} | {test_lpips:.4f} | {dark_fg_rgb_l1:.4f} | "
            "{dark_fg_luma_l1:.4f} | {base_dark_to_lit_ratio:.4f} | {num_points} | {line_like_ratio:.4f} | {tiny_axis_ratio:.4f} |".format(
                **{k: (v if v is not None and not (isinstance(v, float) and math.isnan(v)) else float("nan")) for k, v in row.items()}
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def discover_runs(output_root: Path, requested_runs):
    if requested_runs:
        return [output_root / run for run in requested_runs]
    return sorted([child for child in output_root.iterdir() if child.is_dir() and (child / "convergence" / "eval_metrics.jsonl").exists()])


def main():
    parser = argparse.ArgumentParser(description="Single-file analyzer for Pixiu training runs.")
    parser.add_argument("--output_root", required=True, type=str)
    parser.add_argument("--runs", nargs="*", default=None)
    parser.add_argument("--report_dir", type=str, default=None)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    report_dir = Path(args.report_dir) if args.report_dir else output_root / "analysis"
    report_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = discover_runs(output_root, args.runs)
    summaries = []
    run_eval_data = {}
    run_geom_data = {}

    for run_dir in run_dirs:
        result = extract_run_summary(run_dir)
        if result is None:
            continue
        summary, eval_by_split, geom_rows = result
        summaries.append(summary)
        run_eval_data[run_dir.name] = eval_by_split
        run_geom_data[run_dir.name] = geom_rows

    if not summaries:
        raise SystemExit("No completed runs with eval_metrics.jsonl were found.")

    summaries.sort(key=lambda row: row["run"])
    summary_csv = report_dir / "summary.csv"
    write_csv(summary_csv, summaries, list(summaries[0].keys()))

    write_markdown_report(report_dir / "summary.md", summaries)

    for metric in ["psnr", "lpips", "ssim", "l1"]:
        make_metric_plot(report_dir / f"test_{metric}.png", run_eval_data, "test", metric)
    for metric in ["min_axis_median", "anisotropy_median", "line_like_ratio", "tiny_axis_ratio"]:
        make_geometry_plot(report_dir / f"{metric}.png", run_geom_data, metric)

    # Use the minimum final iteration that all runs have for the comparison panels.
    final_iters = [row["final_iteration"] for row in summaries]
    common_iter = min(final_iters)
    make_panel_grid(report_dir / f"final_test_panels_iter_{common_iter:05d}.png", run_dirs, common_iter, "test")
    make_panel_grid(report_dir / f"final_train_panels_iter_{common_iter:05d}.png", run_dirs, common_iter, "train")

    print(f"Wrote analysis to {report_dir}")
    print(f"Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
