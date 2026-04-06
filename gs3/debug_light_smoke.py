import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "submodules" / "diff-gaussian-rasterization_light"))

from diff_gaussian_rasterization_light import (  # noqa: E402
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from utils.graphics_utils import getProjectionMatrix  # noqa: E402


DEVICE = "cuda"


def make_settings(width=64, height=64, fovx_deg=60.0, fovy_deg=60.0):
    fovx = math.radians(fovx_deg)
    fovy = math.radians(fovy_deg)
    tanfovx = math.tan(fovx * 0.5)
    tanfovy = math.tan(fovy * 0.5)

    view = torch.eye(4, dtype=torch.float32, device=DEVICE)
    proj = getProjectionMatrix(0.01, 100.0, fovx, fovy).transpose(0, 1).cuda()
    full_proj = (view.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)

    return GaussianRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=torch.zeros(3, dtype=torch.float32, device=DEVICE),
        scale_modifier=1.0,
        viewmatrix=view,
        projmatrix=full_proj,
        sh_degree=0,
        campos=torch.zeros(3, dtype=torch.float32, device=DEVICE),
        prefiltered=False,
        debug=False,
        low_pass_filter_radius=0.3,
        ortho=False,
    )


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


def make_case(num_points=64, seed=0):
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(seed)

    means3d = torch.empty((num_points, 3), device=DEVICE, dtype=torch.float32)
    means3d[:, 0] = torch.empty(num_points, device=DEVICE).uniform_(-0.6, 0.6, generator=gen)
    means3d[:, 1] = torch.empty(num_points, device=DEVICE).uniform_(-0.6, 0.6, generator=gen)
    means3d[:, 2] = torch.empty(num_points, device=DEVICE).uniform_(2.0, 6.0, generator=gen)
    means3d.requires_grad_(True)

    means2d = torch.zeros_like(means3d, requires_grad=True)
    colors = torch.empty((num_points, 3), device=DEVICE, dtype=torch.float32)
    colors[:, 0] = 2.0
    colors[:, 1:] = torch.rand((num_points, 2), device=DEVICE, dtype=torch.float32, generator=gen)
    colors.requires_grad_(True)

    opacities = torch.empty((num_points, 1), device=DEVICE, dtype=torch.float32).uniform_(0.05, 0.5, generator=gen)
    opacities.requires_grad_(True)

    cov3d = torch.zeros((num_points, 6), device=DEVICE, dtype=torch.float32)
    cov3d[:, 0] = torch.empty(num_points, device=DEVICE).uniform_(0.01, 0.08, generator=gen)
    cov3d[:, 3] = torch.empty(num_points, device=DEVICE).uniform_(0.01, 0.08, generator=gen)
    cov3d[:, 5] = torch.empty(num_points, device=DEVICE).uniform_(0.01, 0.08, generator=gen)
    cov3d.requires_grad_(True)

    return means3d, means2d, colors, opacities, cov3d


def make_visible_pair_case(z_far, near_opacity=0.25, far_opacity=0.55):
    means3d = torch.tensor(
        [
            [0.0, 0.0, 2.0],
            [0.02, 0.0, 2.1],
            [-0.01, 0.01, 2.3],
            [0.03, -0.02, z_far],
        ],
        device=DEVICE,
        dtype=torch.float32,
        requires_grad=True,
    )
    means2d = torch.zeros((4, 3), device=DEVICE, dtype=torch.float32, requires_grad=True)
    colors = torch.tensor(
        [
            [2.0, 0.7, 0.7],
            [2.0, 0.5, 0.5],
            [2.0, 0.6, 0.6],
            [2.0, 0.8, 0.8],
        ],
        device=DEVICE,
        dtype=torch.float32,
        requires_grad=True,
    )
    opacities = torch.tensor(
        [[near_opacity], [0.35], [0.45], [far_opacity]],
        device=DEVICE,
        dtype=torch.float32,
        requires_grad=True,
    )
    cov3d = torch.tensor(
        [
            [0.03, 0.0, 0.0, 0.03, 0.0, 0.03],
            [0.04, 0.0, 0.0, 0.03, 0.0, 0.04],
            [0.03, 0.0, 0.0, 0.04, 0.0, 0.03],
            [0.04, 0.0, 0.0, 0.04, 0.0, 0.04],
        ],
        device=DEVICE,
        dtype=torch.float32,
        requires_grad=True,
    )
    return means3d, means2d, colors, opacities, cov3d


def make_single_case():
    means3d = torch.tensor([[0.0, 0.0, 2.0]], device=DEVICE, dtype=torch.float32, requires_grad=True)
    means2d = torch.zeros((1, 3), device=DEVICE, dtype=torch.float32, requires_grad=True)
    colors = torch.tensor([[2.0, 0.7, 0.7]], device=DEVICE, dtype=torch.float32, requires_grad=True)
    opacities = torch.tensor([[0.5]], device=DEVICE, dtype=torch.float32, requires_grad=True)
    cov3d = torch.tensor([[0.03, 0.0, 0.0, 0.03, 0.0, 0.03]], device=DEVICE, dtype=torch.float32, requires_grad=True)
    return means3d, means2d, colors, opacities, cov3d


def make_stack_case(num_points, z_step, opacity, cov):
    means3d = torch.zeros((num_points, 3), device=DEVICE, dtype=torch.float32)
    means3d[:, 0] = torch.linspace(-0.01, 0.01, num_points, device=DEVICE)
    means3d[:, 1] = torch.linspace(0.01, -0.01, num_points, device=DEVICE)
    means3d[:, 2] = 2.0 + z_step * torch.arange(num_points, device=DEVICE, dtype=torch.float32)
    means3d.requires_grad_(True)

    means2d = torch.zeros_like(means3d, requires_grad=True)
    colors = torch.zeros((num_points, 3), device=DEVICE, dtype=torch.float32)
    colors[:, 0] = 2.0
    colors.requires_grad_(True)

    opacities = torch.full((num_points, 1), opacity, device=DEVICE, dtype=torch.float32, requires_grad=True)
    cov3d = torch.zeros((num_points, 6), device=DEVICE, dtype=torch.float32)
    cov3d[:, 0] = cov
    cov3d[:, 3] = cov
    cov3d[:, 5] = cov
    cov3d.requires_grad_(True)
    return means3d, means2d, colors, opacities, cov3d


def rasterize_case(
    rasterizer,
    means3d,
    means2d,
    colors,
    opacities,
    cov3d,
    offset,
    thres,
    loss_builder,
):
    non_trans = torch.zeros((means3d.shape[0], 1), device=DEVICE, dtype=torch.float32)
    visible = rasterizer.markVisible(means3d.detach())

    _, _, radii, out_trans, non_trans, _ = rasterizer(
        means3D=means3d,
        means2D=means2d,
        shs=None,
        colors_precomp=colors,
        opacities=opacities,
        scales=None,
        rotations=None,
        cov3Ds_precomp=cov3d,
        non_trans=non_trans,
        offset=float(offset),
        thres=float(thres),
        is_train=False,
        hgs=False,
        hgs_normals=None,
        hgs_opacities=None,
        hgs_opacities_shadow=None,
        hgs_opacities_light=None,
        streams=None,
    )

    shadow = out_trans / non_trans.clamp_min(1e-6)
    loss = loss_builder(shadow, out_trans, non_trans)
    loss.backward()
    return visible, radii, out_trans, non_trans, shadow, loss


def summarize_result(means3d, means2d, colors, opacities, cov3d, radii, out_trans, non_trans, shadow, extra):
    grads = {
        "means3d": means3d.grad,
        "means2d": means2d.grad,
        "colors": colors.grad,
        "opacities": opacities.grad,
        "cov3d": cov3d.grad,
    }

    result = dict(extra)
    result.update(
        {
            "radii_positive": int((radii > 0).sum().item()),
            "out_trans_finite": bool(torch.isfinite(out_trans).all().item()),
            "non_trans_finite": bool(torch.isfinite(non_trans).all().item()),
            "shadow_finite": bool(torch.isfinite(shadow).all().item()),
            "shadow_count": int(shadow.numel()),
            "shadow_values": [float(x) for x in shadow.detach().flatten().cpu().tolist()[:8]],
        }
    )
    for name, grad in grads.items():
        result[f"{name}_finite"] = grad is not None and bool(torch.isfinite(grad).all().item())
        result[f"{name}_nan_count"] = 0 if grad is None else int(torch.isnan(grad).sum().item())
        result[f"{name}_absmax"] = 0.0 if grad is None else float(grad.abs().max().item())

    return result


def run_random_once(rasterizer, seed=0, offset=0.2, thres=-1.0):
    means3d, means2d, colors, opacities, cov3d = make_case(seed=seed)
    visible, radii, out_trans, non_trans, shadow, loss = rasterize_case(
        rasterizer,
        means3d,
        means2d,
        colors,
        opacities,
        cov3d,
        offset=offset,
        thres=thres,
        loss_builder=lambda shadow, out_trans, non_trans: shadow.sum(),
    )

    return summarize_result(
        means3d,
        means2d,
        colors,
        opacities,
        cov3d,
        radii,
        out_trans,
        non_trans,
        shadow,
        {
            "suite": "random",
            "seed": seed,
            "visible_count": int(visible.sum().item()),
            "loss": float(loss.item()),
        },
    )


def run_stress_once(rasterizer, name, num_points, z_step, opacity, cov, offset):
    means3d, means2d, colors, opacities, cov3d = make_stack_case(num_points, z_step, opacity, cov)
    visible, radii, out_trans, non_trans, shadow, loss = rasterize_case(
        rasterizer,
        means3d,
        means2d,
        colors,
        opacities,
        cov3d,
        offset=offset,
        thres=-1.0,
        loss_builder=lambda shadow, out_trans, non_trans: shadow.square().sum() + out_trans.square().sum(),
    )

    return summarize_result(
        means3d,
        means2d,
        colors,
        opacities,
        cov3d,
        radii,
        out_trans,
        non_trans,
        shadow,
        {
            "suite": name,
            "num_points": num_points,
            "visible_count": int(visible.sum().item()),
            "z_step": z_step,
            "opacity": opacity,
            "cov": cov,
            "offset": offset,
            "loss": float(loss.item()),
        },
    )


def run_contract_case(name, rasterizer, builder, offset, thres=4.0):
    means3d, means2d, colors, opacities, cov3d = builder()
    visible, radii, out_trans, non_trans, shadow, loss = rasterize_case(
        rasterizer,
        means3d,
        means2d,
        colors,
        opacities,
        cov3d,
        offset=offset,
        thres=thres,
        loss_builder=lambda shadow, out_trans, non_trans: shadow.square().sum(),
    )
    return summarize_result(
        means3d,
        means2d,
        colors,
        opacities,
        cov3d,
        radii,
        out_trans,
        non_trans,
        shadow,
        {
            "suite": "contract",
            "name": name,
            "visible_count": int(visible.sum().item()),
            "offset": offset,
            "loss": float(loss.item()),
        },
    )


def eval_pair_shadow_component(rasterizer, near_opacity, z_far=2.06, offset=0.05):
    means3d, means2d, colors, opacities, cov3d = make_visible_pair_case(z_far=z_far, near_opacity=near_opacity)
    for tensor in (means3d, means2d, colors, opacities, cov3d):
        tensor.requires_grad_(False)
    non_trans = torch.zeros((means3d.shape[0], 1), device=DEVICE, dtype=torch.float32)
    _, _, _, out_trans, non_trans, _ = rasterizer(
        means3D=means3d,
        means2D=means2d,
        shs=None,
        colors_precomp=colors,
        opacities=opacities,
        scales=None,
        rotations=None,
        cov3Ds_precomp=cov3d,
        non_trans=non_trans,
        offset=float(offset),
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
    return float(shadow[3, 0].item())


def run_opacity_finite_difference(rasterizer, eps=1e-3):
    means3d, means2d, colors, opacities, cov3d = make_visible_pair_case(z_far=2.06, near_opacity=0.25)
    _, _, _, _, shadow, _ = rasterize_case(
        rasterizer,
        means3d,
        means2d,
        colors,
        opacities,
        cov3d,
        offset=0.05,
        thres=4.0,
        loss_builder=lambda shadow, out_trans, non_trans: shadow[3, 0],
    )
    analytic = float(opacities.grad[0, 0].item())
    numeric = (
        eval_pair_shadow_component(rasterizer, 0.25 + eps)
        - eval_pair_shadow_component(rasterizer, 0.25 - eps)
    ) / (2.0 * eps)
    abs_err = abs(analytic - numeric)
    rel_err = abs_err / max(1e-6, abs(numeric))
    return {
        "suite": "finite_difference",
        "eps": eps,
        "analytic_grad": analytic,
        "numeric_grad": numeric,
        "abs_err": abs_err,
        "rel_err": rel_err,
    }


def finite_failures(result):
    return [k for k, v in result.items() if k.endswith("_finite") and not v]


def contract_failures(name, result):
    failures = finite_failures(result)
    shadow = result["shadow_values"]

    if name == "single_visible":
        if result["radii_positive"] < 1 or abs(shadow[0] - 1.0) > 1e-4:
            failures.append("single_shadow_contract")
    elif name == "double_within_offset":
        if result["radii_positive"] < 2 or shadow[3] < 0.99:
            failures.append("within_offset_contract")
    elif name == "double_beyond_offset":
        if result["radii_positive"] < 2 or shadow[3] > 0.95:
            failures.append("beyond_offset_contract")
    elif name == "early_stop_stack":
        if result["radii_positive"] < 16:
            failures.append("early_stop_visibility_contract")
    elif name == "overlap_pressure":
        if result["radii_positive"] < 16:
            failures.append("overlap_visibility_contract")

    return failures


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke test.")

    torch.manual_seed(0)
    rasterizer = GaussianRasterizer(make_settings())
    identity_rasterizer = GaussianRasterizer(make_identity_settings())

    failures = []

    print("== Deterministic Contract Cases ==")
    contract_cases = [
        ("single_visible", lambda: make_single_case(), identity_rasterizer, 0.05),
        ("double_within_offset", lambda: make_visible_pair_case(z_far=2.05), identity_rasterizer, 0.05),
        ("double_beyond_offset", lambda: make_visible_pair_case(z_far=2.06), identity_rasterizer, 0.05),
        (
            "early_stop_stack",
            lambda: make_stack_case(num_points=512, z_step=0.001, opacity=0.99, cov=0.12),
            identity_rasterizer,
            0.05,
        ),
        (
            "overlap_pressure",
            lambda: make_stack_case(num_points=1024, z_step=0.001, opacity=0.99, cov=0.12),
            identity_rasterizer,
            0.10,
        ),
    ]
    for name, builder, case_rasterizer, offset in contract_cases:
        result = run_contract_case(name, case_rasterizer, builder, offset=offset)
        bad = contract_failures(name, result)
        print(result)
        if bad:
            failures.append((name, bad, result))

    print("\n== Random Synthetic Cases ==")
    for offset in [0.05, 0.1, 0.2, 0.5]:
        print(f"\n[offset={offset}]")
        for seed in [0, 1, 2, 3, 4]:
            result = run_random_once(rasterizer, seed=seed, offset=offset)
            bad = finite_failures(result)
            print(result)
            if bad:
                failures.append((f"random_offset_{offset}_seed_{seed}", bad, result))

    print("\n== Stress Cases ==")
    stress_cases = [
        ("stress", 1024, 0.002, 0.99, 0.10, 0.05),
        ("stress", 1024, 0.001, 0.99, 0.12, 0.05),
        ("stress", 1024, 0.001, 0.99, 0.12, 0.10),
        ("stress", 1536, 0.001, 0.99, 0.12, 0.10),
        ("stress", 2048, 0.001, 0.99, 0.12, 0.10),
    ]
    for case in stress_cases:
        result = run_stress_once(rasterizer, *case)
        bad = finite_failures(result)
        print(result)
        if bad:
            failures.append((case, bad, result))

    print("\n== Opacity Finite Difference ==")
    fd_result = run_opacity_finite_difference(identity_rasterizer, eps=1e-3)
    print(fd_result)
    if fd_result["rel_err"] > 0.08 or not math.isfinite(fd_result["analytic_grad"]) or not math.isfinite(fd_result["numeric_grad"]):
        failures.append(("finite_difference", ["opacity_grad_contract"], fd_result))

    if failures:
        print("\nFAILURES")
        for item in failures:
            print(item)
        raise SystemExit(1)

    print("\nAll deterministic, random, stress, and finite-difference checks passed.")


if __name__ == "__main__":
    main()
