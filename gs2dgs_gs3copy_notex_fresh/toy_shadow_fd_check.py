import json
import math
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "submodules" / "diff-gaussian-rasterization_light"))

from diff_gaussian_rasterization_light import GaussianRasterizationSettings, GaussianRasterizer  # noqa: E402


DEVICE = "cuda"


def make_identity_settings(width=32, height=32):
    return GaussianRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=1.0,
        tanfovy=1.0,
        bg=torch.zeros(3, dtype=torch.float32, device=DEVICE),
        scale_modifier=1.0,
        viewmatrix=torch.eye(4, dtype=torch.float32, device=DEVICE),
        projmatrix=torch.eye(4, dtype=torch.float32, device=DEVICE),
        sh_degree=0,
        campos=torch.zeros(3, dtype=torch.float32, device=DEVICE),
        prefiltered=False,
        debug=False,
        low_pass_filter_radius=0.3,
        ortho=False,
    )


def base_case_tensors(occluder_z, occluder_opacity, occluder_scale):
    zero = torch.zeros((), device=DEVICE, dtype=torch.float32)
    near = torch.stack([zero, zero, occluder_z])
    near_2 = torch.tensor([0.02, 0.0, 2.02], device=DEVICE, dtype=torch.float32)
    near_3 = torch.tensor([-0.01, 0.01, 2.03], device=DEVICE, dtype=torch.float32)
    target = torch.tensor([0.03, -0.02, 2.06], device=DEVICE, dtype=torch.float32)
    means3d = torch.stack([near, near_2, near_3, target], dim=0)
    means2d = torch.zeros((4, 3), device=DEVICE, dtype=torch.float32)
    colors = torch.tensor(
        [
            [2.0, 0.7, 0.7],
            [2.0, 0.5, 0.5],
            [2.0, 0.6, 0.6],
            [2.0, 0.8, 0.8],
        ],
        device=DEVICE,
        dtype=torch.float32,
    )
    opacities = torch.stack(
        [
            occluder_opacity.reshape(1),
            torch.tensor([0.35], device=DEVICE, dtype=torch.float32),
            torch.tensor([0.45], device=DEVICE, dtype=torch.float32),
            torch.tensor([0.55], device=DEVICE, dtype=torch.float32),
        ],
        dim=0,
    )
    scales = torch.stack(
        [
            occluder_scale.repeat(3),
            torch.tensor([0.18, 0.18, 0.18], device=DEVICE, dtype=torch.float32),
            torch.tensor([0.17, 0.17, 0.17], device=DEVICE, dtype=torch.float32),
            torch.tensor([0.18, 0.18, 0.18], device=DEVICE, dtype=torch.float32),
        ],
        dim=0,
    )
    rotations = torch.zeros((4, 4), device=DEVICE, dtype=torch.float32)
    rotations[:, 0] = 1.0
    return means3d, means2d, colors, opacities, scales, rotations


def shadow_loss_for_case(rasterizer, occluder_z, occluder_opacity, occluder_scale):
    means3d, means2d, colors, opacities, scales, rotations = base_case_tensors(
        occluder_z=occluder_z,
        occluder_opacity=occluder_opacity,
        occluder_scale=occluder_scale,
    )
    non_trans = torch.zeros((means3d.shape[0], 1), device=DEVICE, dtype=torch.float32)
    _, _, _, out_trans, non_trans, _ = rasterizer(
        means3D=means3d,
        means2D=means2d,
        shs=None,
        colors_precomp=colors,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3Ds_precomp=None,
        non_trans=non_trans,
        offset=0.05,
        thres=4.0,
        is_train=False,
        hgs=False,
        hgs_normals=None,
        hgs_opacities=None,
        hgs_opacities_shadow=None,
        hgs_opacities_light=None,
        streams=None,
    )
    shadow = out_trans / non_trans.clamp_min(1e-6)
    target_shadow = shadow[3, 0]
    return 1.0 - target_shadow, float(target_shadow.item())


def analytic_and_numeric_gradient(rasterizer, variable: str, eps: float):
    z = torch.tensor(2.0, device=DEVICE, dtype=torch.float32, requires_grad=(variable == "xyz"))
    opacity = torch.tensor(0.25, device=DEVICE, dtype=torch.float32, requires_grad=(variable == "opacity"))
    scale = torch.tensor(0.18, device=DEVICE, dtype=torch.float32, requires_grad=(variable == "scale"))
    loss, base_shadow = shadow_loss_for_case(rasterizer, z, opacity, scale)
    loss.backward()

    if variable == "xyz":
        analytic = float(z.grad.item())
        plus = shadow_loss_for_case(
            rasterizer,
            torch.tensor(2.0 + eps, device=DEVICE),
            torch.tensor(0.25, device=DEVICE),
            torch.tensor(0.18, device=DEVICE),
        )[0]
        minus = shadow_loss_for_case(
            rasterizer,
            torch.tensor(2.0 - eps, device=DEVICE),
            torch.tensor(0.25, device=DEVICE),
            torch.tensor(0.18, device=DEVICE),
        )[0]
        expected_sign = "negative"
    elif variable == "opacity":
        analytic = float(opacity.grad.item())
        plus = shadow_loss_for_case(
            rasterizer,
            torch.tensor(2.0, device=DEVICE),
            torch.tensor(0.25 + eps, device=DEVICE),
            torch.tensor(0.18, device=DEVICE),
        )[0]
        minus = shadow_loss_for_case(
            rasterizer,
            torch.tensor(2.0, device=DEVICE),
            torch.tensor(0.25 - eps, device=DEVICE),
            torch.tensor(0.18, device=DEVICE),
        )[0]
        expected_sign = "positive"
    elif variable == "scale":
        analytic = float(scale.grad.item())
        plus = shadow_loss_for_case(
            rasterizer,
            torch.tensor(2.0, device=DEVICE),
            torch.tensor(0.25, device=DEVICE),
            torch.tensor(0.18 + eps, device=DEVICE),
        )[0]
        minus = shadow_loss_for_case(
            rasterizer,
            torch.tensor(2.0, device=DEVICE),
            torch.tensor(0.25, device=DEVICE),
            torch.tensor(0.18 - eps, device=DEVICE),
        )[0]
        expected_sign = "positive"
    else:
        raise ValueError(variable)

    numeric = float(((plus - minus) / (2.0 * eps)).item())
    abs_err = abs(analytic - numeric)
    rel_err = abs_err / max(1e-6, abs(numeric))
    return {
        "variable": variable,
        "base_shadow": base_shadow,
        "analytic_grad": analytic,
        "numeric_grad": numeric,
        "abs_err": abs_err,
        "rel_err": rel_err,
        "sign_match": math.copysign(1.0, analytic) == math.copysign(1.0, numeric),
        "expected_sign": expected_sign,
    }


def main():
    parser = ArgumentParser(description="Finite-difference checks for toy shadow gradients.")
    parser.add_argument("--output_root", type=str, default="output/shadow_gradient_validation/toy_fd")
    parser.add_argument("--eps", type=float, default=1e-3)
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    rasterizer = GaussianRasterizer(make_identity_settings())
    results = {
        variable: analytic_and_numeric_gradient(rasterizer, variable, eps=float(args.eps))
        for variable in ["xyz", "opacity", "scale"]
    }

    with (output_root / "toy_fd_summary.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
