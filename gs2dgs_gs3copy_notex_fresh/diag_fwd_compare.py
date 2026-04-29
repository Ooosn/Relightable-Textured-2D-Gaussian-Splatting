"""
Compare forward rendering output pixel-by-pixel between gsplat and 3dgs
using the same Gaussian parameters.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "submodules", "diff-gaussian-rasterization"))

import torch
import numpy as np
from argparse import Namespace

# Minimal setup - directly create tensors
torch.manual_seed(0)

# Load a checkpoint
ckpt_path = r"D:\RTS\output\gs3_NRHints_Pixiu\chkpnt7000.pth"
ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
state = ckpt[0]

means3D = state["_xyz"].cuda().detach()
scales = state["_scaling"].cuda().detach()
rotations = state["_rotation"].cuda().detach()
opacity = state["_opacity"].cuda().detach()

N = means3D.shape[0]
print(f"N = {N} gaussians")

# Create a simple 7-channel color (random but fixed)
torch.manual_seed(42)
colors = torch.rand(N, 7, device="cuda")

# Camera params from the dataset
W, H = 512, 512
fx, fy = 3095.45, 3095.45
cx, cy = 222.98, 243.01
FoVx = 2 * np.arctan(W / (2*fx))
FoVy = 2 * np.arctan(H / (2*fy))
tanfovx = np.tan(FoVx / 2)
tanfovy = np.tan(FoVy / 2)

# Simple camera - identity rotation, looking at origin
from utils.graphics_utils import getWorld2View2, getProjectionMatrixWithPrincipalPoint, fov2focal
R = np.eye(3)
T = np.array([0, 0, 3.0])
world_view = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).float().cuda()
proj_matrix = getProjectionMatrixWithPrincipalPoint(
    znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, width=W, height=H
).transpose(0, 1).float().cuda()
full_proj = (world_view.unsqueeze(0).bmm(proj_matrix.unsqueeze(0))).squeeze(0)
campos = world_view.inverse()[3, :3]

bg = torch.zeros(7, device="cuda")

# --- 3dgs forward ---
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
raster_settings = GaussianRasterizationSettings(
    image_height=H, image_width=W,
    tanfovx=tanfovx, tanfovy=tanfovy,
    bg=bg,
    scale_modifier=1.0,
    viewmatrix=world_view,
    projmatrix=full_proj,
    sh_degree=0,
    campos=campos,
    prefiltered=False,
    debug=False,
    low_pass_filter_radius=0.3,
)
rasterizer_3dgs = GaussianRasterizer(raster_settings=raster_settings)

screenspace = torch.zeros_like(means3D, requires_grad=True, device="cuda")
img_3dgs, radii_3dgs, depth_3dgs, alpha_3dgs = rasterizer_3dgs(
    means3D=means3D, means2D=screenspace, shs=None,
    colors_precomp=colors, opacities=opacity,
    scales=scales, rotations=rotations, cov3Ds_precomp=None,
    hgs=False, hgs_normals=torch.Tensor([]), hgs_opacities=torch.Tensor([])
)

# --- gsplat forward ---
from gsplat import rasterization
K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0., 0., 1.]], device="cuda")
gsplat_out, gsplat_alphas, gsplat_meta = rasterization(
    means=means3D, quats=rotations, scales=scales,
    opacities=opacity.squeeze(-1), colors=colors,
    viewmats=world_view.transpose(0, 1)[None, ...],
    Ks=K[None, ...],
    width=W, height=H,
    near_plane=0.01, far_plane=100.0,
    eps2d=0.3, sh_degree=None, packed=False,
    backgrounds=bg[None, ...],
)
img_gsplat = gsplat_out[0].permute(2, 0, 1)  # (H,W,C) -> (C,H,W)

# --- Compare ---
print(f"\n3dgs  image shape: {img_3dgs.shape}, range: [{img_3dgs.min():.6f}, {img_3dgs.max():.6f}]")
print(f"gsplat image shape: {img_gsplat.shape}, range: [{img_gsplat.min():.6f}, {img_gsplat.max():.6f}]")

diff = (img_3dgs - img_gsplat).abs()
print(f"\nPixel difference:")
print(f"  mean: {diff.mean():.8e}")
print(f"  max:  {diff.max():.8e}")
print(f"  std:  {diff.std():.8e}")

# Per-channel
for ch in range(7):
    d = diff[ch]
    print(f"  ch{ch}: mean={d.mean():.6e}, max={d.max():.6e}, "
          f"pct>1e-4={100*(d>1e-4).float().mean():.2f}%, "
          f"pct>1e-3={100*(d>1e-3).float().mean():.2f}%")

# Also check radii
radii_gsplat = gsplat_meta['radii'].squeeze(0)
r3 = (radii_3dgs > 0).sum().item()
rg = (radii_gsplat > 0).sum().item()
print(f"\nVisible gaussians: 3dgs={r3}, gsplat={rg}, diff={r3-rg}")

# Check if any gaussians are visible in one but not the other
mask_3dgs = radii_3dgs > 0
mask_gsplat = radii_gsplat > 0
only_3dgs = (mask_3dgs & ~mask_gsplat).sum().item()
only_gsplat = (~mask_3dgs & mask_gsplat).sum().item()
print(f"Only in 3dgs: {only_3dgs}, Only in gsplat: {only_gsplat}")

# --- Backward comparison ---
print("\n=== BACKWARD COMPARISON ===")

# Need colors with grad
colors_3dgs = colors.clone().detach().requires_grad_(True)
colors_gsplat = colors.clone().detach().requires_grad_(True)
means3D_3dgs = means3D.clone().detach().requires_grad_(True)
means3D_gsplat = means3D.clone().detach().requires_grad_(True)
opacity_3dgs = opacity.clone().detach().requires_grad_(True)
opacity_gsplat = opacity.clone().detach().requires_grad_(True)
scales_3dgs = scales.clone().detach().requires_grad_(True)
scales_gsplat = scales.clone().detach().requires_grad_(True)
rotations_3dgs = rotations.clone().detach().requires_grad_(True)
rotations_gsplat = rotations.clone().detach().requires_grad_(True)

# 3dgs forward+backward
sp_3dgs = torch.zeros_like(means3D, requires_grad=True, device="cuda")
img_3dgs2, _, _, _ = rasterizer_3dgs(
    means3D=means3D_3dgs, means2D=sp_3dgs, shs=None,
    colors_precomp=colors_3dgs, opacities=opacity_3dgs,
    scales=scales_3dgs, rotations=rotations_3dgs, cov3Ds_precomp=None,
    hgs=False, hgs_normals=torch.Tensor([]), hgs_opacities=torch.Tensor([])
)
loss_3dgs = img_3dgs2.sum()
loss_3dgs.backward()

# gsplat forward+backward
gsplat_out2, _, _ = rasterization(
    means=means3D_gsplat, quats=rotations_gsplat, scales=scales_gsplat,
    opacities=opacity_gsplat.squeeze(-1), colors=colors_gsplat,
    viewmats=world_view.transpose(0, 1)[None, ...],
    Ks=K[None, ...],
    width=W, height=H,
    near_plane=0.01, far_plane=100.0,
    eps2d=0.3, sh_degree=None, packed=False,
    backgrounds=bg[None, ...],
)
img_gsplat2 = gsplat_out2[0].permute(2, 0, 1)
loss_gsplat = img_gsplat2.sum()
loss_gsplat.backward()

print(f"loss_3dgs={loss_3dgs.item():.10f}, loss_gsplat={loss_gsplat.item():.10f}")

for name, g3, gg in [
    ("colors", colors_3dgs.grad, colors_gsplat.grad),
    ("means3D", means3D_3dgs.grad, means3D_gsplat.grad),
    ("opacity", opacity_3dgs.grad, opacity_gsplat.grad),
    ("scales", scales_3dgs.grad, scales_gsplat.grad),
    ("rotations", rotations_3dgs.grad, rotations_gsplat.grad),
]:
    if g3 is None or gg is None:
        print(f"  {name}: one has no grad")
        continue
    d = (g3 - gg).abs()
    rel = d / (gg.abs() + 1e-20)
    print(f"  {name:12s}: abs_diff mean={d.mean():.6e} max={d.max():.6e} | "
          f"rel_diff mean={rel.mean():.6e} max={rel.max():.6e} | "
          f"3dgs_absmean={g3.abs().mean():.6e} gsplat_absmean={gg.abs().mean():.6e}")
