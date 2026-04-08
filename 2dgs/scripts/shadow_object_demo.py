import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TWO_DGS_ROOT = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(TWO_DGS_ROOT)
if TWO_DGS_ROOT not in sys.path:
    sys.path.insert(0, TWO_DGS_ROOT)
SHADOW_SUBMODULE = os.path.join(TWO_DGS_ROOT, "submodules", "diff-surfel-rasterization-shadow")
if SHADOW_SUBMODULE not in sys.path:
    sys.path.insert(0, SHADOW_SUBMODULE)

from scene import Scene  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from utils.sh_utils import eval_sh  # noqa: E402
from utils.graphics_utils import getProjectionMatrix  # noqa: E402
from diff_surfel_rasterization_shadow import (  # noqa: E402
    GaussianRasterizationSettings as ShadowRasterSettings,
)
from diff_surfel_rasterization_shadow import GaussianRasterizer as ShadowRasterizer  # noqa: E402


MODEL_PATH = r"d:\RTS\output\vanilla2dgs_synth_single_object_shadowdemo_15000"
SOURCE_PATH = r"d:\RTS\data\synthetic_shadow_single_object"
ITERATION = 7000
OUT_DIR = os.path.join(MODEL_PATH, "shadow_diag_iter7000")
LIGHT_PRESETS = {
    "low_left": [1.20, -0.90, 0.35],
    "low_right": [-1.20, -0.80, 0.35],
    "front_low": [0.25, -1.40, 0.40],
    "back_low": [0.35, 1.20, 0.35],
}


def look_at(eye, target, up):
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    camera_direction = eye - target
    camera_direction = camera_direction / np.linalg.norm(camera_direction)
    if abs(np.dot(up, camera_direction)) > 0.9:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    camera_right = np.cross(up, camera_direction)
    camera_right = camera_right / np.linalg.norm(camera_right)
    camera_up = np.cross(camera_direction, camera_right)
    camera_up = camera_up / np.linalg.norm(camera_up)

    rotation = np.zeros((4, 4), dtype=np.float32)
    rotation[0, :3] = camera_right
    rotation[1, :3] = camera_up
    rotation[2, :3] = camera_direction
    rotation[3, 3] = 1.0

    translation = np.eye(4, dtype=np.float32)
    translation[:3, -1] = -eye

    view = rotation @ translation
    view[1:3, :] *= -1
    return torch.tensor(view.T, dtype=torch.float32, device="cuda")


def build_orthographic_projection(xmin, xmax, ymin, ymax, zmin, zmax):
    proj = torch.tensor(
        [
            [2.0 / (xmax - xmin), 0.0, 0.0, -(xmax + xmin) / (xmax - xmin)],
            [0.0, 2.0 / (ymax - ymin), 0.0, -(ymax + ymin) / (ymax - ymin)],
            [0.0, 0.0, 1.0 / (zmax - zmin), -zmin / (zmax - zmin)],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    return proj.transpose(0, 1)


def compute_base_rgb(camera, gaussians):
    shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree + 1) ** 2)
    dir_pp = gaussians.get_xyz - camera.camera_center.repeat(gaussians.get_features.shape[0], 1)
    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return torch.clamp(eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized) + 0.5, 0.0, 1.0)


def save_labeled_contact_sheet(image_paths, labels, out_path, tile_size):
    cols = 2
    rows = math.ceil(len(image_paths) / cols)
    label_h = 18
    canvas = Image.new("RGB", (cols * tile_size[0], rows * (tile_size[1] + label_h)), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for idx, (path, label) in enumerate(zip(image_paths, labels)):
        img = Image.open(path).convert("RGB")
        x = (idx % cols) * tile_size[0]
        y = (idx // cols) * (tile_size[1] + label_h)
        canvas.paste(img, (x, y + label_h))
        draw.text((x + 4, y + 2), label, fill=(0, 0, 0))
    canvas.save(out_path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    dataset = SimpleNamespace(
        sh_degree=3,
        use_textures=False,
        texture_resolution=4,
        use_mbrdf=False,
        basis_asg_num=8,
        phasefunc_hidden_size=32,
        phasefunc_hidden_layers=3,
        phasefunc_frequency=4,
        neural_material_size=6,
        asg_channel_num=1,
        asg_mlp=False,
        asg_alpha_num=1,
        source_path=SOURCE_PATH,
        model_path=MODEL_PATH,
        images="images",
        resolution=-1,
        white_background=True,
        data_device="cuda",
        eval=True,
    )
    pipe = SimpleNamespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        depth_ratio=0.0,
        debug=False,
        shadow_pass=False,
    )

    gaussians = GaussianModel(
        dataset.sh_degree,
        dataset.use_textures,
        dataset.texture_resolution,
        dataset.use_mbrdf,
        dataset.basis_asg_num,
        dataset.phasefunc_hidden_size,
        dataset.phasefunc_hidden_layers,
        dataset.phasefunc_frequency,
        dataset.neural_material_size,
        dataset.asg_channel_num,
        dataset.asg_mlp,
        dataset.asg_alpha_num,
    )
    scene = Scene(dataset, gaussians, load_iteration=ITERATION, shuffle=False)
    cameras = scene.getTestCameras() if len(scene.getTestCameras()) > 0 else scene.getTrainCameras()
    camera = cameras[0]

    xyz = gaussians.get_xyz.detach()
    with torch.no_grad():
        base_rgb = compute_base_rgb(camera, gaussians)
        bg_color = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
        plain_pkg = render(camera, gaussians, pipe, bg_color)
        torchvision.utils.save_image(plain_pkg["render"], os.path.join(OUT_DIR, "plain.png"))
        torchvision.utils.save_image(camera.original_image, os.path.join(OUT_DIR, "gt.png"))
        center = xyz.mean(dim=0)
        radius = torch.linalg.norm(xyz - center, dim=1).amax().item()
        xyz_h = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1)
        shadowed_paths = []
        factor_paths = []
        labels = []

        for light_name, light_dir_list in LIGHT_PRESETS.items():
            light_dir = torch.tensor(light_dir_list, dtype=torch.float32, device="cuda")
            light_dir = light_dir / light_dir.norm()
            light_position = center + light_dir * (radius * 3.8)

            light_view = look_at(
                light_position.detach().cpu().numpy(),
                center.detach().cpu().numpy(),
                np.array([0.0, 0.0, 1.0], dtype=np.float32),
            )
            xyz_light = xyz_h @ light_view
            margin_xy = max(radius * 0.25, 0.1)
            margin_z = max(radius * 0.50, 0.2)
            xmin = xyz_light[:, 0].min() - margin_xy
            xmax = xyz_light[:, 0].max() + margin_xy
            ymin = xyz_light[:, 1].min() - margin_xy
            ymax = xyz_light[:, 1].max() + margin_xy
            zmin = torch.clamp(xyz_light[:, 2].min() - margin_z, min=0.05)
            zmax = xyz_light[:, 2].max() + margin_z

            light_proj = build_orthographic_projection(xmin, xmax, ymin, ymax, zmin, zmax)
            light_full_proj = light_view.unsqueeze(0).bmm(light_proj.unsqueeze(0)).squeeze(0)
            shadow_settings = ShadowRasterSettings(
                image_height=512,
                image_width=512,
                tanfovx=1.0,
                tanfovy=1.0,
                bg=torch.zeros(3, dtype=torch.float32, device="cuda"),
                scale_modifier=1.0,
                viewmatrix=light_view,
                projmatrix=light_full_proj,
                sh_degree=gaussians.active_sh_degree,
                campos=light_position,
                prefiltered=False,
                debug=False,
                low_pass_filter_radius=0.3,
                ortho=True,
            )

            shadow_rasterizer = ShadowRasterizer(raster_settings=shadow_settings)
            dummy_means2d = torch.zeros_like(gaussians.get_xyz, requires_grad=False)
            ones = torch.ones((gaussians.get_xyz.shape[0], 3), dtype=torch.float32, device="cuda")
            non_trans = torch.zeros((gaussians.get_xyz.shape[0], 1), dtype=torch.float32, device="cuda")
            _, _, radii, out_trans, non_trans, _ = shadow_rasterizer(
                means3D=gaussians.get_xyz,
                means2D=dummy_means2d,
                opacities=gaussians.get_opacity,
                shs=None,
                colors_precomp=ones,
                scales=gaussians.get_scaling,
                rotations=gaussians.get_rotation,
                cov3Ds_precomp=None,
                texture_alpha=None,
                texture_sigma_factor=3.0,
                non_trans=non_trans,
                offset=0.15,
                thres=-1.0,
                is_train=False,
            )

            shadow_ratio = torch.ones_like(out_trans)
            valid = non_trans.squeeze(-1) > 1e-6
            shadow_ratio[valid] = (out_trans[valid] / non_trans[valid]).clamp(0.0, 1.0)
            shadow_rgb = torch.clamp(base_rgb * shadow_ratio, 0.0, 1.0)
            shadow_factor_rgb = shadow_ratio.repeat(1, 3)

            shadow_pkg = render(camera, gaussians, pipe, bg_color, override_color=shadow_rgb)
            factor_pkg = render(camera, gaussians, pipe, bg_color, override_color=shadow_factor_rgb)

            light_dir_out = os.path.join(OUT_DIR, light_name)
            os.makedirs(light_dir_out, exist_ok=True)
            shadow_path = os.path.join(light_dir_out, "shadowed.png")
            factor_path = os.path.join(light_dir_out, "shadow_factor.png")
            stats_path = os.path.join(light_dir_out, "stats.txt")
            torchvision.utils.save_image(shadow_pkg["render"], shadow_path)
            torchvision.utils.save_image(factor_pkg["render"], factor_path)
            with open(stats_path, "w", encoding="utf-8") as f:
                f.write(f"iteration={ITERATION}\n")
                f.write(f"num_points={gaussians.get_xyz.shape[0]}\n")
                f.write(f"camera={camera.image_name}\n")
                f.write(f"light_name={light_name}\n")
                f.write(f"light_position={light_position.detach().cpu().tolist()}\n")
                f.write(f"shadow_ratio_mean={shadow_ratio.mean().item():.6f}\n")
                f.write(f"shadow_ratio_min={shadow_ratio.min().item():.6f}\n")
                f.write(f"shadow_ratio_max={shadow_ratio.max().item():.6f}\n")
                f.write(f"shadow_valid_count={(non_trans.squeeze(-1) > 1e-6).sum().item()}\n")
                f.write(f"visible_radii_nonzero={(radii > 0).sum().item()}\n")

            shadowed_paths.append(shadow_path)
            factor_paths.append(factor_path)
            labels.append(light_name)
            print(
                f"{light_name}: mean={shadow_ratio.mean().item():.6f} "
                f"min={shadow_ratio.min().item():.6f} max={shadow_ratio.max().item():.6f}"
            )

        save_labeled_contact_sheet(
            shadowed_paths,
            labels,
            os.path.join(OUT_DIR, "shadowed_contact.png"),
            (camera.image_width, camera.image_height),
        )
        save_labeled_contact_sheet(
            factor_paths,
            labels,
            os.path.join(OUT_DIR, "shadow_factor_contact.png"),
            (camera.image_width, camera.image_height),
        )
        print(f"Saved shadow diagnostics to {OUT_DIR}")
        print(f"camera={camera.image_name}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
