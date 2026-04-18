#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import json
import os
from re import A
from pathlib import Path
import torch
import numpy as np
import math
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, get_shadow_backward_stage
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, build_scaling_rotation
from utils.graphics_utils import getWorld2View2_cu
from utils.lie_groups import exp_map_SO3xR3
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from PIL import Image, ImageDraw
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import lpips
loss_fn_alex = None


def get_lpips_model():
    global loss_fn_alex
    if loss_fn_alex is None:
        loss_fn_alex = lpips.LPIPS(net='vgg').cuda().eval()
    return loss_fn_alex


def create_render_streams():
    serial_stream = os.environ.get("SSGS_SERIAL_STREAM", "0") == "1"
    if serial_stream:
        return None, None
    return torch.cuda.Stream(), torch.cuda.Stream()


def _get_env_int(name: str, default: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def append_jsonl(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
        f.write("\n")


def _check_vram_limit(pipe):
    if not torch.cuda.is_available():
        return
    limit_gb = float(getattr(pipe, "max_vram_gb", 0.0) or 0.0)
    if limit_gb <= 0.0:
        return
    peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
    if peak_reserved > limit_gb:
        raise RuntimeError(
            f"VRAM limit exceeded: peak_reserved={peak_reserved:.2f} GB > limit={limit_gb:.2f} GB"
        )


def _tensor_finite(tensor):
    if tensor is None:
        return True
    return bool(torch.isfinite(tensor.detach()).all().item())


def _tensor_absmax(tensor):
    if tensor is None:
        return None
    tensor = tensor.detach()
    if tensor.numel() == 0:
        return 0.0
    return float(tensor.abs().max().item())


def _grad_health(parameter):
    grad = None if parameter is None else parameter.grad
    return {
        "present": grad is not None,
        "finite": _tensor_finite(grad) if grad is not None else True,
        "absmax": _tensor_absmax(grad),
    }


def _camera_projection_bridge_2dgs(viewpoint_cam, means3D, viewspace_point_tensor, transmat_grad_holder=None, splat2world=None):
    if viewspace_point_tensor is None or viewspace_point_tensor.grad is None:
        return False
    if not getattr(viewpoint_cam.cam_pose_adj, "requires_grad", False):
        return False

    dL_dmean2D = viewspace_point_tensor.grad.detach()
    if dL_dmean2D.ndim == 3:
        dL_dmean2D = dL_dmean2D.squeeze(0)
    if dL_dmean2D.shape[-1] < 2:
        return False
    dL_dmean2D = dL_dmean2D[:, :2]
    if not torch.isfinite(dL_dmean2D).all():
        return False
    if float(dL_dmean2D.abs().max().item()) < 1e-12:
        return False

    adj = exp_map_SO3xR3(viewpoint_cam.cam_pose_adj)
    dR = adj[0, :3, :3]
    dt = adj[0, :3, 3]
    R = viewpoint_cam.R_cu.matmul(dR.T)
    T = dt + dR.matmul(viewpoint_cam.T_cu)
    world_view_transform = getWorld2View2_cu(
        R, T, viewpoint_cam.trans_cu, viewpoint_cam.scale_cu
    ).transpose(0, 1)
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(viewpoint_cam.projection_matrix.unsqueeze(0))
    ).squeeze(0)

    means_h = torch.cat([means3D.detach(), torch.ones_like(means3D[:, :1])], dim=-1)
    clip = means_h @ full_proj_transform
    w = clip[:, 3:4].clamp_min(1e-7)
    ndc_xy = clip[:, :2] / w

    pixel_xy = torch.empty_like(ndc_xy)
    pixel_xy[:, 0] = ((ndc_xy[:, 0] + 1.0) * viewpoint_cam.image_width - 1.0) * 0.5
    pixel_xy[:, 1] = ((ndc_xy[:, 1] + 1.0) * viewpoint_cam.image_height - 1.0) * 0.5

    proxy = (pixel_xy * dL_dmean2D).sum()

    if transmat_grad_holder is not None and transmat_grad_holder.grad is not None and splat2world is not None:
        dL_dtransMat = transmat_grad_holder.grad.detach()
        if dL_dtransMat.ndim == 3:
            dL_dtransMat = dL_dtransMat.reshape(dL_dtransMat.shape[0], -1)
        if (
            dL_dtransMat.shape[-1] == 9
            and torch.isfinite(dL_dtransMat).all()
            and float(dL_dtransMat.abs().max().item()) >= 1e-12
        ):
            W, H = viewpoint_cam.image_width, viewpoint_cam.image_height
            near, far = viewpoint_cam.znear, viewpoint_cam.zfar
            ndc2pix = torch.tensor(
                [
                    [W / 2, 0, 0, (W - 1) / 2],
                    [0, H / 2, 0, (H - 1) / 2],
                    [0, 0, far - near, near],
                    [0, 0, 0, 1],
                ],
                dtype=full_proj_transform.dtype,
                device=full_proj_transform.device,
            ).T
            trans_proxy = (
                splat2world[:, [0, 1, 3]] @ (full_proj_transform @ ndc2pix)[:, [0, 1, 3]]
            ).permute(0, 2, 1).reshape(-1, 9)
            proxy = proxy + (trans_proxy * dL_dtransMat).sum()

    if not torch.isfinite(proxy):
        return False
    proxy.backward()
    return True


def _build_splat2world_2dgs(means3D, scaling_2d, rotation, scaling_modifier=1.0):
    rs = build_scaling_rotation(
        torch.cat([scaling_2d * scaling_modifier, torch.ones_like(scaling_2d)], dim=-1),
        rotation,
    ).permute(0, 2, 1)
    trans = torch.zeros((means3D.shape[0], 4, 4), dtype=means3D.dtype, device=means3D.device)
    trans[:, :3, :3] = rs
    trans[:, 3, :3] = means3D
    trans[:, 3, 3] = 1
    return trans


def _masked_vector_stats(values: torch.Tensor, mask: torch.Tensor | None = None):
    if values is None:
        return {"count": 0, "median": None, "p95": None, "max": None}
    flat = values.detach().reshape(-1)
    if mask is not None:
        flat_mask = mask.detach().reshape(-1)
        if flat_mask.shape[0] == flat.shape[0]:
            flat = flat[flat_mask]
    if flat.numel() == 0:
        return {"count": 0, "median": None, "p95": None, "max": None}
    flat = flat.float()
    return {
        "count": int(flat.numel()),
        "median": float(flat.median().item()),
        "p95": float(torch.quantile(flat, 0.95).item()),
        "max": float(flat.max().item()),
    }


def collect_densify_diagnostics(iteration, viewspace_point_tensor, visibility_filter, out_weights, num_points_before):
    grad = None
    if viewspace_point_tensor is not None and viewspace_point_tensor.grad is not None:
        grad = viewspace_point_tensor.grad.detach().squeeze(0)
        if grad.ndim == 2 and grad.shape[-1] >= 2:
            grad = torch.norm(grad[:, :2], dim=-1)
    grad_stats = _masked_vector_stats(grad, visibility_filter)
    out_weight_values = None
    if out_weights is not None:
        out_weight_values = out_weights.detach().reshape(-1)
    out_weight_stats = _masked_vector_stats(out_weight_values)
    return {
        "iteration": int(iteration),
        "num_points_before": int(num_points_before),
        "visible_grad_count": grad_stats["count"],
        "visible_grad_median": grad_stats["median"],
        "visible_grad_p95": grad_stats["p95"],
        "visible_grad_max": grad_stats["max"],
        "out_weight_count": out_weight_stats["count"],
        "out_weight_median": out_weight_stats["median"],
        "out_weight_p95": out_weight_stats["p95"],
        "out_weight_max": out_weight_stats["max"],
    }


def compute_shadow_scale_regularization(gaussians, modelset, iteration):
    if not getattr(modelset, "shadow_scale_reg_enabled", False):
        return None
    if iteration < int(getattr(modelset, "shadow_scale_reg_from_iter", 0)):
        return None

    scales = gaussians.get_scaling
    min_axis = scales.min(dim=1).values.clamp_min(1e-12)
    max_axis = scales.max(dim=1).values.clamp_min(1e-12)
    anisotropy = max_axis / min_axis

    floor = float(getattr(modelset, "shadow_scale_min_axis_floor", 1e-6))
    floor_weight = float(getattr(modelset, "shadow_scale_floor_weight", 0.0))
    anisotropy_thresh = float(getattr(modelset, "shadow_scale_anisotropy_thresh", 20.0))
    anisotropy_weight = float(getattr(modelset, "shadow_scale_anisotropy_weight", 0.0))

    floor_penalty = torch.zeros([], device=scales.device, dtype=scales.dtype)
    anisotropy_penalty = torch.zeros([], device=scales.device, dtype=scales.dtype)

    if floor_weight > 0.0:
        floor_gap = torch.relu(torch.log(torch.tensor(floor, device=scales.device, dtype=scales.dtype)) - torch.log(min_axis))
        floor_penalty = torch.mean(floor_gap * floor_gap)
    if anisotropy_weight > 0.0:
        aniso_gap = torch.relu(torch.log(anisotropy) - math.log(anisotropy_thresh))
        anisotropy_penalty = torch.mean(aniso_gap * aniso_gap)

    total = floor_weight * floor_penalty + anisotropy_weight * anisotropy_penalty
    return {
        "total": total,
        "floor_penalty": floor_penalty,
        "anisotropy_penalty": anisotropy_penalty,
        "min_axis_median": float(min_axis.median().item()),
        "anisotropy_median": float(anisotropy.median().item()),
    }


def collect_training_health(iteration, ll1, loss, image, shadow, other_effects, gaussians, num_points, densify_due, opacity_reset_due, shadow_stage, modelset):
    xyz_health = _grad_health(getattr(gaussians, "_xyz", None))
    scaling_health = _grad_health(getattr(gaussians, "_scaling", None))
    opacity_health = _grad_health(getattr(gaussians, "_opacity", None))
    rotation_health = _grad_health(getattr(gaussians, "_rotation", None))
    loss_finite = _tensor_finite(loss)
    render_finite = _tensor_finite(image)
    shadow_finite = _tensor_finite(shadow)
    other_effects_finite = _tensor_finite(other_effects)
    all_finite = all(
        [
            loss_finite,
            render_finite,
            shadow_finite,
            other_effects_finite,
            xyz_health["finite"],
            scaling_health["finite"],
            opacity_health["finite"],
            rotation_health["finite"],
        ]
    )
    return {
        "iteration": int(iteration),
        "l1": float(ll1.item()),
        "loss": float(loss.item()),
        "loss_finite": loss_finite,
        "render_finite": render_finite,
        "shadow_finite": shadow_finite,
        "other_effects_finite": other_effects_finite,
        "xyz_grad_present": xyz_health["present"],
        "xyz_grad_finite": xyz_health["finite"],
        "xyz_grad_absmax": xyz_health["absmax"],
        "scaling_grad_present": scaling_health["present"],
        "scaling_grad_finite": scaling_health["finite"],
        "scaling_grad_absmax": scaling_health["absmax"],
        "opacity_grad_present": opacity_health["present"],
        "opacity_grad_finite": opacity_health["finite"],
        "opacity_grad_absmax": opacity_health["absmax"],
        "rotation_grad_present": rotation_health["present"],
        "rotation_grad_finite": rotation_health["finite"],
        "rotation_grad_absmax": rotation_health["absmax"],
        "num_points": int(num_points),
        "densify_due": bool(densify_due),
        "opacity_reset_due": bool(opacity_reset_due),
        "detach_shadow": bool(getattr(modelset, "detach_shadow", False)),
        "use_shadow_refine_mlp": bool(getattr(modelset, "use_shadow_refine_mlp", True)),
        "shadow_backward_stage_enabled": bool(getattr(modelset, "shadow_backward_stage_enabled", False)),
        "shadow_scale_reg_enabled": bool(getattr(modelset, "shadow_scale_reg_enabled", False)),
        "shadow_backward_xyz_enabled": bool(shadow_stage["xyz"]),
        "shadow_backward_opacity_enabled": bool(shadow_stage["opacity"]),
        "shadow_backward_scaling_enabled": bool(shadow_stage["scaling"]),
        "shadow_backward_rotation_enabled": bool(shadow_stage["rotation"]),
        "all_finite": all_finite,
    }


def _tensor_to_uint8_hwc(image: torch.Tensor) -> np.ndarray:
    image = torch.clamp(image.detach().cpu(), 0.0, 1.0)
    if image.ndim == 2:
        image = image.unsqueeze(0)
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    return (image.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)


def _error_heatmap(rendered: torch.Tensor, gt: torch.Tensor) -> np.ndarray:
    err = torch.mean(torch.abs(rendered - gt), dim=0).detach().cpu().numpy()
    max_err = max(float(err.max()), 1e-6)
    err = np.clip(err / max_err, 0.0, 1.0)
    red = (err * 255.0 + 0.5).astype(np.uint8)
    green = ((1.0 - err) * 200.0 + 0.5).astype(np.uint8)
    blue = ((1.0 - err) * 80.0 + 0.5).astype(np.uint8)
    return np.stack([red, green, blue], axis=-1)


def save_fixed_view_artifacts(output_dir: Path, gt: torch.Tensor, rendered: torch.Tensor, base_render: torch.Tensor, shadow: torch.Tensor, metrics: dict):
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_img = Image.fromarray(_tensor_to_uint8_hwc(gt))
    render_img = Image.fromarray(_tensor_to_uint8_hwc(rendered))
    base_img = Image.fromarray(_tensor_to_uint8_hwc(base_render))
    shadow_img = Image.fromarray(_tensor_to_uint8_hwc(shadow))
    error_img = Image.fromarray(_error_heatmap(rendered, gt))

    gt_img.save(output_dir / "gt.png")
    render_img.save(output_dir / "render.png")
    base_img.save(output_dir / "base.png")
    shadow_img.save(output_dir / "shadow.png")
    error_img.save(output_dir / "abs_error.png")

    w, h = gt_img.size
    title_h = 24
    footer_h = 44
    panel = Image.new("RGB", (w * 5, h + title_h + footer_h), (255, 255, 255))
    draw = ImageDraw.Draw(panel)

    labels = ["GT", "Render", "Base", "Abs Error", "Shadow"]
    images = [gt_img, render_img, base_img, error_img, shadow_img]
    for idx, (label, image) in enumerate(zip(labels, images)):
        x = idx * w
        panel.paste(image, (x, title_h))
        draw.text((x + 6, 4), label, fill=(0, 0, 0))

    footer = (
        f"L1 {metrics['l1']:.4f}  "
        f"PSNR {metrics['psnr']:.2f}  "
        f"SSIM {metrics['ssim']:.4f}  "
        f"LPIPS {metrics['lpips']:.4f}"
    )
    draw.text((6, h + title_h + 10), footer, fill=(0, 0, 0))
    panel.save(output_dir / "panel.png")

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def save_fixed_view_collection(split_dir: Path, view_panels):
    if not view_panels:
        return

    rows = []
    max_width = 0
    total_height = 0
    title_h = 26

    for view_name, panel_path in view_panels:
        image = Image.open(panel_path).convert("RGB")
        rows.append((view_name, image))
        max_width = max(max_width, image.width)
        total_height += image.height + title_h

    canvas = Image.new("RGB", (max_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    y = 0
    for view_name, image in rows:
        draw.text((8, y + 5), view_name, fill=(0, 0, 0))
        canvas.paste(image, (0, y + title_h))
        y += image.height + title_h

    canvas.save(split_dir / "panel.png")

#training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.unfreeze_iterations, args.debug_from)
def training(modelset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, unfreeze_iterations, debug_from):
    first_iter = 0

    # 准备输出和日志记录器，更新模型路径
    tb_writer = prepare_output_and_logger(modelset)

    # 初始化高斯实例
    gaussians = GaussianModel(modelset, opt)

    # 根据 训练args，优化args 以及 初始高斯 建立场景实例，最终的高斯实例
    """
        1. 加载场景数据集，无论是否加载旧模型都需要
            对于非加载旧模型，则为重新训练，需要创建新的模型文件夹，并将点云文件拷贝到新建的模型文件夹下，准备训练集的相机信息，将相机信息写入到 cameras.json 文件中
            而对于加载旧模型，该相机信息已经存在，所以不需要再次创建
        2. 加载高斯模型，如果加载旧模型则直接从点云中加载高斯模型，否则从场景信息的点云信息中创建高斯模型
        3. 根据命令行参数判断是否需要添加优化相机参数和光源参数
    """
    scene = Scene(modelset, gaussians, opt=opt, shuffle=True, checkpoint=checkpoint)
    
    # 按照 opt对象 初始化参数设置


    # 恢复模型模型参数以及优化器状态，覆盖之前初始化的模型
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, opt)
    else:
        gaussians.training_setup(opt)
    
    # 设置背景颜色，1,1,1,1,0,0,0表示白色背景，0,0,0,0,0,0,0表示黑色背景
    bg_color = [1, 1, 1, 1, 0, 0, 0] if modelset.white_background else [0, 0, 0, 0, 0, 0, 0]
    # 将 bg_color 转换为 PyTorch 的张量，并将其分配到 GPU（device="cuda"）以加速后续的计算。
    # 
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 新建事件，用于记录迭代开始和结束的时间
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # 初始化一些训练过程中需要使用的变量
    prune_visibility = False    # 可见性修剪，是否剔除不可见的点，可以释放内存，提高内存使用效率
    viewpoint_stack = None    # 存储相机视点（viewpoints）的堆栈，在训练过程中，会从堆栈中弹出一个相机视点，用于渲染图像，从而完成对所有视角的遍历。
    opt_test = False    # 当前是否处于优化测试模式
    opt_test_ready = False    # 是否准备好进行优化测试
    """
    指数移动平均（EMA）损失，用于记录训练过程中的损失值，只是为了平滑损失曲线，更好地可视化训练过程，不参与模型的训练。
    ema_loss_for_log = α * 当前损失 + (1 - α) * ema_loss_for_log 类似于梯度更新中的一阶动量。
    """
    ema_loss_for_log = 0.0    # 初始值为 0.0，表示尚未计算损失值
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")    #使用 tqdm 创建一个可视化的进度条，用于显示训练的进度
    first_iter += 1    # 迭代次数加一，开始下一次迭代
    print("first_iter", first_iter)
    convergence_dir = Path(scene.model_path) / "convergence"
    health_log_path = convergence_dir / "train_health.jsonl"
    densify_log_path = convergence_dir / "densify_events.jsonl"
    train_health_interval = max(_get_env_int("SSGS_TRAIN_HEALTH", 0), 0)
    collect_backward_stats = os.environ.get("SSGS_BACKWARD_INFO", "0") == "1"
    """
    相位函数（Phase Function） 是一个来自图形学和物理学的概念，主要用于描述光与介质相互作用时，光的散射方向分布。
    本文中，即为神经相位函数（Neural Phase Function），是一个用于描述材质表面散射特性的神经网络模型。
    """
    """
    ASG（Anisotropic Spherical Gaussians, 各向异性球面高斯）
    本文中，ASG 作为镜面反射的模型，用于描述材质表面的镜面反射特性。
    """
    # 用于记录是否冻结了相位函数
    # 根据当前迭代次数，决定是否冻结相位函数
    phase_func_freezed = False
    asg_freezed = True
    asg_mlp = False

    info = None
    crop_extent = None
    record_info = None

    if collect_backward_stats:
        record_info = {
            "means3D_grad": 0, "scales_grad": 0, "rotations_grad": 0, "opacity_grad": 0,
            "radii": 0, "means2D_grad": 0, "num_clone_ratio": 0, "num_split_ratio": 0,
            "colors_precomp_grad": 0, "ks": 0, "kd": 0, "diffuse_grad": 0, "specular_grad": 0,
            "cosTheta_grad": 0, "dist_2_inv_grad": 0, 
            "decay_grad": 0, "other_effects_grad": 0, "shadow_grad": 0,
            "asg_3_grad": 0, "asg_1_grad": 0, 
            "base_image_grad": 0, "shadow_image_grad": 0, "other_effects_image_grad": 0,
            "base_image": 0, "shadow_image": 0, "other_effects_image": 0,
            }

    metrics_by_category = None
    if collect_backward_stats:
        metrics_by_category = {
            "3_gaussians": ["means3D_grad", "scales_grad", "rotations_grad", "opacity_grad"],
            "4_densification": ["radii", "means2D_grad", "num_clone_ratio", "num_split_ratio"],
            "5_before_rasterization": ["colors_precomp_grad", "ks", "kd", "diffuse_grad", "specular_grad", 
                                        "cosTheta_grad", "dist_2_inv_grad",
                                        "decay_grad", "other_effects_grad", "shadow_grad", 
                                        "asg_3_grad", "asg_1_grad"],
            "6_image_grad": ["base_image_grad", "shadow_image_grad", "other_effects_image_grad"],
            "7_image": ["base_image", "shadow_image", "other_effects_image"],
            }


    loss_fn_alex.to(gaussians.get_features.device)

    if first_iter < unfreeze_iterations:
        gaussians.neural_phasefunc.freeze()
        phase_func_freezed = True
    elif first_iter >= unfreeze_iterations and first_iter < opt.spcular_freeze_step:
        gaussians.neural_phasefunc.unfreeze()
        phase_func_freezed = False
    elif first_iter >= opt.spcular_freeze_step and first_iter < opt.spcular_freeze_step + opt.fit_linear_step:
        gaussians.neural_phasefunc.freeze()
        gaussians.neural_material.requires_grad_(False)
    elif first_iter >= opt.spcular_freeze_step + opt.fit_linear_step:
        gaussians.neural_phasefunc.unfreeze()
        gaussians.neural_material.requires_grad_(True)
        
    # initialize parallel GPU stream 多流并行
    # 有时会出现错误，可以尝试关闭，改为串行，torch.cuda.current_stream().synchronize()
    light_stream, calc_stream = create_render_streams()




    """开始训练"""
    # 每次迭代，都会从视点堆栈中选择一个视点，然后渲染图像，计算损失，更新模型参数，并不是每次计算全部视点的损失
    for iteration in range(first_iter, opt.iterations + 1):    #左闭右开区间，因此加1
        iter_start.record()    # 记录迭代开始的时间
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # update lr of asg
        gaussians.update_learning_rate(iteration, \
                                        asg_freeze_step=opt.asg_lr_freeze_step, \
                                        local_q_freeze_step=opt.local_q_lr_freeze_step, \
                                        freeze_phasefunc_steps=opt.freeze_phasefunc_steps)
        # opt camera or point light
        if scene.optimizing:
            scene.update_lr(iteration, \
                            freez_train_cam=opt.train_cam_freeze_step, \
                            freez_train_pl=opt.train_pl_freeze_step, \
                            cam_opt=modelset.cam_opt, \
                            pl_opt=modelset.pl_opt)
            
            
        # Every 1000 its we increase the levels of SH up to a maximum degree
        # 采取神经网络后，不再使用球谐函数，因此不再需要逐级增加球谐函数的阶数
        if iteration % 1000 == 0:
            if not modelset.use_nerual_phasefunc:
                gaussians.oneupSHdegree()
        
        # 在早期训练时，冻结 asg 参数，快速收敛
        # 但是并没有冻结 asg_sigma，为了先获得一个基础的 高光形状，并未考虑拟合复杂的高光形状
        if iteration <= opt.asg_freeze_step:
            gaussians.asg_func.asg_scales.requires_grad_(False)
            gaussians.asg_func.asg_rotation.requires_grad_(False)
        # else if iteration > opt.asg_freeze_step and asg_freezed:
        # 后续的迭代中，asg_freezed 可能已为 False，所以不需要重新设置
        elif asg_freezed:
            asg_freezed = False
            gaussians.asg_func.asg_scales.requires_grad_(True)
            gaussians.asg_func.asg_rotation.requires_grad_(True)
            print("set ansio param requires_grad: ", gaussians.asg_func.asg_scales.requires_grad)
        
        # Pick a random Camera
        # 如果视点堆栈为空，则根据 opt_test_ready 和 opt_test 的值，进行轮番选择训练视点和测试视点
        if not viewpoint_stack:
            # only do pose opt for test sets
            if opt_test_ready and scene.optimizing:
                opt_test = True
                # 重新填装测试视点堆栈
                viewpoint_stack = scene.getTestCameras().copy()
                opt_test_ready = False
            else:
                opt_test = False
                # 重新填装训练视点堆栈
                viewpoint_stack = scene.getTrainCameras().copy()
                opt_test_ready = True

        # 为当前迭代选择一个视点
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))



        """！开始渲染"""
        # debug用，一般不需要
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # 选择背景颜色，如果opt输出为随机背景，则随机选择一个背景颜色，否则使用固定背景颜色
        bg = torch.rand((7), device="cuda") if opt.random_background else background
        
        # precompute shading frames and ASG frames
        local_axises = gaussians.get_local_axis # (K, 3, 3)
        asg_scales = gaussians.asg_func.get_asg_lam_miu # (basis_asg_num, 2)
        asg_axises = gaussians.asg_func.get_asg_axis    # (basis_asg_num, 3, 3)

        # only opt with diffuse term at the beginning for a stable training process
        if iteration < opt.spcular_freeze_step + opt.fit_linear_step:   # 只考虑漫反射，不考虑镜面反射，刚开始训练时，先优化漫反射，赋予每个高斯点一个基础颜色
            renderArgs = {"modelset": modelset, "pipe": pipe, "bg_color": bg, "fix_labert": True, "is_train": prune_visibility, "asg_mlp": asg_mlp, "iteration": iteration}
            render_pkg = render(viewpoint_cam, gaussians, light_stream, calc_stream, local_axises, asg_scales, asg_axises, **renderArgs) 
        else:    # 开始考虑镜面反射             
            renderArgs = {"modelset": modelset, "pipe": pipe, "bg_color": bg, "is_train": prune_visibility, "asg_mlp": asg_mlp, "iteration": iteration}
            render_pkg = render(viewpoint_cam, gaussians, light_stream, calc_stream, local_axises, asg_scales, asg_axises, **renderArgs)
        _check_vram_limit(pipe)

        # 此外，取出各个高斯点云坐标、可见性、半径，用于后续修剪
        viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        image, shadow, other_effects, backward_info, expected_depth = render_pkg["render"], render_pkg["shadow"], render_pkg["other_effects"], render_pkg["backward_info"], render_pkg["expected_depth"]
        transmat_grad_holder = render_pkg.get("transmat_grad_holder")


        # 如果迭代次数小于 unfreeze_iterations，则不考虑阴影和次要效果，此时 shadow 和 other_effects 都是 0
        if iteration <= unfreeze_iterations:    # 5000
            image = image
        else:
            image = image * shadow + other_effects
        _check_vram_limit(pipe)



        """！！Loss部分"""
        # 获取真实图片数据
        gt_image = viewpoint_cam.original_image.cuda()
        
        """
        · HDR 线性空间的数值直接表示物理光照强度，例如：
            - 一个非常亮的像素可能是 100.0。
            - 一个较暗的像素可能是 0.1。
        · 在计算机存储 HDR 数据时，它们的数值比例是线性的：
            - 亮部（100.0）比暗部（0.1）大 1000 倍。
        · 这种极大的动态范围，使得暗部在数据中几乎看不见，而亮部占据主导。
        · 人眼感知不是线性的，对暗部更敏感，对亮部的变化不敏感。
        · 如果直接显示 HDR 线性数据，亮部会过曝，而暗部基本不可见。
        ~ 因此，需要进行 Gamma 校正，将 HDR 线性数据转换为 非线性空间，使得人眼更容易区分 100 和 1000 之间的亮度差异。
        """
        # 初始阶段，固定 Gamma，使数据更接近 sRGB 颜色分布，避免大范围的数值差异带来的过曝，暗部细节丢失
        # 第二阶段逐步调整 Gamma 使其趋近于 1，让网络逐步适应 HDR 线性数据，而不是突然切换，避免网络过早收敛到错误的分布
        if modelset.hdr:
            if iteration <= opt.spcular_freeze_step:
                gt_image = torch.pow(gt_image, 1./2.2)
            elif iteration < opt.spcular_freeze_step + opt.fit_linear_step//2:
                # 慢慢从 2.2 转换到 1.1 ，why not 1.0?
                # 当 iteration 大于  opt.spcular_freeze_step + opt.fit_linear_step//2 后，则直接使用原 gt_image，即 gamma 为 1
                gamma = 1.1 * float(opt.spcular_freeze_step + opt.fit_linear_step - iteration + 1) / float(opt.fit_linear_step // 2 + 1)
                gt_image = torch.pow(gt_image, 1./gamma)
        else:
            # image = torch.clip(image, 0.0, 1.0)
            # gamma 校正
            if modelset.gamma_change:
                if iteration < 4 *(opt.spcular_freeze_step + opt.fit_linear_step//2): 
                    image = image / (1.0 + image)
                    image = torch.clamp(image, min=1e-4) 
                    gamma = 1.0 + 1.2 * (1-(4 *(opt.spcular_freeze_step + opt.fit_linear_step//2) - iteration) / (4 *(opt.spcular_freeze_step + opt.fit_linear_step//2)))
                    image = torch.clip(image, 0.0, 1.0)
                    # image = torch.pow(image, 1./gamma)
                    
                else:
                    image = image / (1.0 + image)
                    image = torch.clamp(image, min=1e-4) 
                    image = torch.clip(image, 0.0, 1.0)
                    # image = torch.pow(image, 1./2.2)
            else:
                # start_clamp = 0.0 - 0.5 * (50000 - iteration) / 50000  if iteration < 50000 else 0.0
                # end_clamp = 1.0 + 0.5 * (50000 - iteration) / 50000 if iteration < 50000 else 1.0
                # image = torch.clip(image, start_clamp, end_clamp)
                image = torch.clip(image, 0.0, 1.0)

        Ll1 = l1_loss(image, gt_image)      # lamda_dssim 默认 0.2
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        shadow_scale_reg = compute_shadow_scale_regularization(gaussians, modelset, iteration)
        if shadow_scale_reg is not None:
            loss = loss + shadow_scale_reg["total"]
        
        # 反向传播，计算各个参数的梯度
        # 尚未更新参数，等待后续挑选更新
        loss.backward()
        _check_vram_limit(pipe)
        if (
            scene.optimizing
            and str(getattr(gaussians, "rasterizer", ""))[:5] == "2dgs"
            and os.getenv("GS3_DISABLE_2DGS_CAM_BRIDGE", "0") != "1"
        ):
            splat2world_2dgs = _build_splat2world_2dgs(
                gaussians.get_xyz.detach(),
                gaussians.get_scaling[:, :2].detach(),
                gaussians.get_rotation.detach(),
                scaling_modifier=1.0,
            )
            _camera_projection_bridge_2dgs(
                viewpoint_cam,
                gaussians.get_xyz,
                viewspace_point_tensor,
                transmat_grad_holder=transmat_grad_holder,
                splat2world=splat2world_2dgs,
            )
            _check_vram_limit(pipe)
        iter_end.record() # 记录迭代结束的时间



        """！！参数更新部分"""
        # torch.no_grad() 防止污染计算图，加快计算速度
        with torch.no_grad():
            densify_due = iteration < opt.densify_until_iter and iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0
            opacity_reset_due = iteration % opt.opacity_reset_interval == 0 or (modelset.white_background and iteration == opt.densify_from_iter)
            health_due = (
                iteration in testing_iterations
                or (
                    train_health_interval > 0
                    and (
                        iteration == 1
                        or iteration == opt.iterations
                        or densify_due
                        or opacity_reset_due
                        or iteration % train_health_interval == 0
                    )
                )
            )
            if health_due:
                shadow_stage = render_pkg.get("shadow_stage", get_shadow_backward_stage(modelset, iteration))
                health = collect_training_health(
                    iteration=iteration,
                    ll1=Ll1,
                    loss=loss,
                    image=image,
                    shadow=shadow,
                    other_effects=other_effects,
                    gaussians=gaussians,
                    num_points=gaussians.get_xyz.shape[0],
                    densify_due=densify_due,
                    opacity_reset_due=opacity_reset_due,
                    shadow_stage=shadow_stage,
                    modelset=modelset,
                )
                append_jsonl(health_log_path, health)
                if not health["all_finite"]:
                    raise FloatingPointError(
                        f"Non-finite training state at iter {iteration}: "
                        f"loss={health['loss_finite']} render={health['render_finite']} "
                        f"shadow={health['shadow_finite']} xyz_grad={health['xyz_grad_finite']} "
                        f"scaling_grad={health['scaling_grad_finite']} opacity_grad={health['opacity_grad_finite']}"
                    )

            # Progress bar，平滑损失曲线
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                num_gaussians = int(gaussians.get_xyz.shape[0])
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.{7}f}",
                    "Gaussians": f"{num_gaussians:,}",
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            log_train_scalars = tb_writer is not None and (iteration % 10 == 0 or iteration in testing_iterations)
            elapsed_ms = iter_start.elapsed_time(iter_end) if log_train_scalars else 0.0
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed_ms, \
                testing_iterations, scene, render, renderArgs, gamma=2.2 if modelset.hdr else 1.0, \
                metrics_by_category = metrics_by_category, 
                info = record_info, 
                modelset = modelset)
            
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            # 对于测试集：：：：：  
            # 只用于优化场景光源和相机信息，因为这些信息可能是未知的，需要通过优化来估计    
            if opt_test and scene.optimizing:  
                """这里是否有一个漏洞，当 scene.optimizing 为 false 时，测试集也会进入后续的 else 分支，但是本文中可能默认 optimizing 为 true ，阴差阳错"""
                # 不会用于直接优化高斯模型
                if iteration < opt.iterations:
                    """"
                    在相机初始化时，cam_params 和 pl_params 都是 torch.nn.Parameter，默认开启了梯度计算（requires_grad=True）。
                    因此，当 viewpoint_cam 中的每个视角（即相机对象）参与计算时，这些参数会被自动追踪并记录到计算图中。
                    因为根据 torch 以及之前我们自己定义的函数，传递的都是地址，所以可以实现有效的一一对应的更新。

                    由于 PyTorch 的机制，我们在初始化时传递的是参数的引用（内存地址），因此这些参数在计算图中的位置与原始定义的位置一一对应。
                    换句话说，无论是在计算还是更新过程中，操作的始终是这个对象所包含的实际参数，确保了一一对应的关系。

                    优化器通过初始化时绑定的参数引用（内存地址），能够直接访问这些参数的 .grad 属性。
                    在调用 optimizer.step() 时，优化器根据计算出的梯度和设置的学习率更新参数值，实现了有效的一一对应更新。
                    """
                    # 更新相机参数和光源参数
                    scene.optimizer.step()
                    # 梯度清零
                    scene.optimizer.zero_grad(set_to_none = True)
                    # 梯度清零
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    """
                    1. set_to_none=True
                        .grad 直接变成 None。
                        更省显存，因为 PyTorch 不会存 0 矩阵，而是完全释放梯度张量。
                        计算时 会跳过 None 梯度的参数，不执行 += 操作
                    2. set_to_none=False (default)
                        梯度 .grad 变成全 0 矩阵。
                        依然占用显存，但 PyTorch 计算梯度时 不会跳过这些参数，仍然会进行 += 操作（梯度累积时有用）。
                    """
    

            # 进入训练集：：：：：
            # 1) 对于非测试集，即训练集进入正常阶段，先进行高斯密集化，高斯修剪以及场景参数优化:
            else:

                # 第一步：：Densification, 高斯复制或分裂，when opt.density_from_iter < iteration < opt.densify_until_iter
                # 如果迭代次数小于高斯密集化迭代次数，则进行高斯密集化
                if iteration < opt.densify_until_iter:
                    # 更新每个可见高斯点在2D投影上的最大半径
                    # visibility_filter: 标记哪些点是可见的
                    # max_radii2D: 记录每个点在屏幕空间的最大半径
                    """
                    Bonus:
                        布尔索引：
                            布尔索引是一种用于选择数组中满足特定条件的元素的索引方式。
                            布尔索引通过一个布尔数组来选择数组中的元素，该布尔数组与原数组形状相同，其中的元素为True或False，表示是否选择该元素。
                            布尔索引通常用于根据某些条件过滤数组中的元素，或者根据条件对数组进行操作。
                    """
                    # 通过布尔索引更新当前视角下，可见高斯点在2D投影上的最大半径，其布尔索引由 render_pkg 中的 visibility_filter 提供
                    # radii：当前视角下，可见高斯点的半径（3*最长轴）
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    """
                    # 获得各个高斯点云坐标、可见性、半径
                    viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                    # 获得渲染结果
                    image, shadow, other_effects = render_pkg["render"], render_pkg["shadow"], render_pkg["other_effects"]
                    """
                    # 记录高斯密集化统计信息
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, image.shape[2], image.shape[1], render_pkg["out_weight"])

                    # 如果迭代次数大于高斯密集化开始迭代次数，并且迭代次数是高斯密集化迭代间隔的倍数，则进行高斯密集化
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        # size_threshold 是高斯密集化过程中，高斯点尺寸阈值
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        # 进行高斯密集化(复制和分裂)，传入的最大梯度，最小透明度，场景范围（相机视锥），尺寸阈值
                        # 最小透明度和尺寸阈值在这里直接调节
                        # 共涉及 高斯密集化，透明度修剪，尺寸修剪，权重修剪，四种对高斯点的修改
                        if False:
                            if iteration % 1000 == 0:
                                print("################################")
                                print(gaussians.xyz_gradient_accum)
                                print(gaussians.denom)
                            print("################################")
                        # 高斯密集化，并进行高斯修剪，# 梯度阈值，最小透明度，场景范围，尺寸阈值

                        if record_info is not None and backward_info:
                            smooth = False
                            for key, value in record_info.items():
                                try:
                                    if key[-5:] == "_grad":
                                        temp = backward_info[key.replace("_grad", "")].grad.detach()
                                    else:
                                        temp = backward_info[key].detach()
                                    temp = temp if temp is not None else torch.zeros((1))
                                    if smooth:
                                        record_info[key] = value*0.7 + temp.float().abs().mean()*0.3
                                    else:
                                        record_info[key] = temp.float().abs().mean()
                                except:
                                    pass

                            rendered_image_grad = backward_info["colors_precomp"].grad.detach().float()
                            base_image_grad = rendered_image_grad[:,0:3].abs().mean()
                            shadow_image_grad = rendered_image_grad[:,3:4].abs().mean()
                            other_effects_image_grad = rendered_image_grad[:,4:7].abs().mean()
                            record_info["base_image_grad"] = record_info["base_image_grad"]*0.7 + base_image_grad*0.3
                            record_info["shadow_image_grad"] = record_info["shadow_image_grad"]*0.7 + shadow_image_grad*0.3
                            record_info["other_effects_image_grad"] = record_info["other_effects_image_grad"]*0.7 + other_effects_image_grad*0.3

                            record_info["base_image"] = backward_info["rendered_image"][0:3, :, :].detach().float().mean()
                            record_info["shadow_image"] = backward_info["rendered_image"][3:4, :, :].detach().float().mean()
                            record_info["other_effects_image"] = backward_info["rendered_image"][4:7, :, :].detach().float().mean()

                        # if iteration % opt.opacity_reset_interval == 0: crop_extent = 1e-6
                        densify_diag = collect_densify_diagnostics(
                            iteration=iteration,
                            viewspace_point_tensor=viewspace_point_tensor,
                            visibility_filter=visibility_filter,
                            out_weights=render_pkg["out_weight"],
                            num_points_before=gaussians.get_xyz.shape[0],
                        )
                        num_points_before = gaussians.get_xyz.shape[0]
                        num_clone, num_split = gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, opt.bigsize_threshold, crop_extent)
                        num_points_after = gaussians.get_xyz.shape[0]
                        densify_diag.update(
                            {
                                "num_clone": int(num_clone),
                                "num_split": int(num_split),
                                "num_points_after": int(num_points_after),
                                "num_pruned": int(max(num_points_before + int(num_clone) + int(num_split) - num_points_after, 0)),
                                "detach_shadow": bool(getattr(modelset, "detach_shadow", False)),
                                "use_shadow_refine_mlp": bool(getattr(modelset, "use_shadow_refine_mlp", True)),
                                "shadow_backward_stage_enabled": bool(getattr(modelset, "shadow_backward_stage_enabled", False)),
                                "shadow_scale_reg_enabled": bool(getattr(modelset, "shadow_scale_reg_enabled", False)),
                            }
                        )
                        append_jsonl(densify_log_path, densify_diag)
                        if record_info is not None:
                            record_info["num_clone_ratio"] = record_info["num_clone_ratio"]*0.7 + num_clone / gaussians.get_xyz.shape[0]*0.3
                            record_info["num_split_ratio"] = record_info["num_split_ratio"]*0.7 + num_split / gaussians.get_xyz.shape[0]*0.3

                            ks = gaussians.get_ks.detach()
                            ks = ks.abs().float().mean()
                            kd = gaussians.get_kd.detach()
                            kd = kd.abs().float().mean()
                            record_info["ks"] = record_info["ks"]*0.7 + ks*0.3
                            record_info["kd"] = record_info["kd"]*0.7 + kd*0.3
                        


                        # 判断高斯点数量是否超过最大高斯点数量的95%，如果超过，则进行高斯修剪，剔除掉不可见的点
                        # 不超过，则 prune_visibility 始终为 False，该分支和后续的 out_weight 相关
                        # 当 prune_visibility 为 False 时，out_weight 不参与计算，因此输出权重为 0
                        if gaussians.get_xyz.shape[0] > gaussians.maximum_gs * 0.95:
                            prune_visibility = True
                    
                    # 如果迭代次数是透明度重置迭代次数的倍数，或者在白色背景且迭代次数等于高斯密集化开始迭代次数，则进行透明度重置
                    if iteration % opt.opacity_reset_interval == 0 or (modelset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

                # 已完成高斯密集化阶段，后续不再进行高斯密集化，高斯点不会发生变多的情况，因此 prune_visibility 设置为 False
                else:
                    prune_visibility = False

                # 2) 完成高斯修剪阶段，再开始更新参数:
                # Optimizer step
                if iteration < opt.iterations:  # 判断是否处于优化阶段
                    gaussians.optimizer.step()
                    # opt the camera pose
                    if scene.optimizing:    # 判断是否开启相机优化
                        scene.optimizer.step()
                        scene.optimizer.zero_grad(set_to_none = True)
                    gaussians.optimizer.zero_grad(set_to_none = True)   # 梯度清零

            """
            1. 默认冻结 高斯点相位函数，初期训练高斯点场景空间为主，并只简单优化漫反射 kd (阴影和次要效果为 0)
            2. 抵达 unfreeze_iterations 时，解冻 高斯点相位函数，开始优化 阴影，材质，次要效果等
            3. 抵达 spcular_freeze_step 时，冻结 高斯点相位函数，再次只优化漫反射 kd
            4. 抵达spcular_freeze_step + fit_linear_step 之后，解冻 高斯点相位函数，并同时开始同时优化镜面反射 ks
            
            总结：
            初期冻结相位函数：训练场景空间为主，
            然后解冻相位函数：获得一个基础的阴影和次要效果，
            然后冻结相位函数，专注于优化漫反射: 在此期间，如果图片为 hdr 格式，则 逐渐调整 gamma 值，逐步转换到 线性空间，防止过曝，暗部细节丢失
            当抵达 opt.spcular_freeze_step + opt.fit_linear_step 时：开始全面优化，包括 相变函数，包括镜面反射，
                - 但此时 asg 部分参数（scale，rotation）还未解冻，取决于 asg_freeze_step (22000)
                - asg 参数只优化 sigma，不优化各向异性参数
            当抵达 asg_freeze_step 时：解冻 asg 各向异性参数，开始优化 asg 各向异性参数
            """
            # 高斯点相位函数（负责处理光照相关的相位特性（如材质、阴影、次要效果等））
            # 判断是否解冻 高斯点相位函数，解冻后该函数失效，由下列代码接管
            if phase_func_freezed and iteration >= unfreeze_iterations: # 5000
                gaussians.neural_phasefunc.unfreeze()
                phase_func_freezed = False

            # 抵达 spcular_freeze_step 时：冻结相变函数，但依然只考虑 漫反射
            """
            如果 图片 为 hdr 格式，则 逐渐调整 gamma 值，逐步转换到 线性空间，防止过曝，暗部细节丢失
            """
            if iteration == opt.spcular_freeze_step: # 9000
                gaussians.neural_phasefunc.freeze()
                gaussians.neural_material.requires_grad_(False)

            # spcular_freeze_step + fit_linear_step 之后：解冻相变函数，并开始同时优化 镜面反射 和 漫反射
            if iteration == opt.spcular_freeze_step + opt.fit_linear_step: # 15000
                gaussians.neural_phasefunc.unfreeze()
                gaussians.neural_material.requires_grad_(True)
            

            if False:
                if iteration > 10000:
                    temp = gaussians.local_q
                    gaussians.local_q.requires_grad_(False)

                if iteration > 18000 and iteration < 40000:
                    gaussians.local_q.requires_grad_(True)
                    if iteration % 5000 == 0:
                        gaussians.reset_local_q(temp)
            


            # 判断是否保存模型
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            
            if modelset.asg_mlp and iteration == opt.asg_mlp_freeze: # 40000
                asg_mlp = True
            
            elif modelset.alpha_change and iteration == opt.asg_change_freeze:
                gaussians.change_alpha_asg(gaussians.alpha_asg)
                print("alpha changed")
        # 更新相机和光源参数：（之前只是更新了 adj 参数，尚未对相机和光源进行直接更新)，注意这里没有设置 torch.no_grad()
        # update cam and light
        if scene.optimizing:
            viewpoint_cam.update("SO3xR3")
            if False:
                print("################################")
                print(viewpoint_cam.cam_pose_adj)
                print(viewpoint_cam.cam_pose_adj.grad)
                print(viewpoint_cam.pl_adj)
                print(viewpoint_cam.pl_adj.grad)
                print("################################")
                

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    convergence_dir = Path(args.model_path) / "convergence"
    (convergence_dir / "fixed_views").mkdir(parents=True, exist_ok=True)
    # 储存模型所使用的参数，用于后续查看以及渲染使用
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, gamma=1.0, metrics_by_category=None, info=None, modelset=None):
    if tb_writer and (iteration % 10 == 0 or iteration in testing_iterations):
        tb_writer.add_scalar('1_loss/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('1_loss/train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('2_iter_time', elapsed, iteration)     
        if info is not None and iteration % 100 == 0:
            for category, metrics in metrics_by_category.items():
                for metric in metrics:
                    try:
                        if metric in info:
                            tb_writer.add_scalar(f'{category}/{metric}', info[metric], iteration)
                    except: 
                        pass
            scale = scene.gaussians.get_scaling.detach().float().abs().mean()
            rotation = scene.gaussians.get_rotation.detach().float().abs().mean()
            opacity = scene.gaussians.get_opacity.detach().float().abs().mean()
            means3D = scene.gaussians.get_xyz.detach().float().abs().mean()
            alpha_asg = scene.gaussians.get_alpha_asg.detach().float().abs().mean()
            local_axis = scene.gaussians.get_local_axis.detach().float().abs().mean()
            tb_writer.add_scalar('10_scene/scale', scale, iteration)
            tb_writer.add_scalar('10_scene/rotation', rotation, iteration)
            tb_writer.add_scalar('10_scene/opacity', opacity, iteration)
            tb_writer.add_scalar('10_scene/means3D', means3D, iteration)
            tb_writer.add_scalar('10_scene/alpha_asg', alpha_asg, iteration)
            tb_writer.add_scalar('10_scene/local_axis', local_axis, iteration)
          
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        eval_log_path = Path(scene.model_path) / "convergence" / "eval_metrics.jsonl"
        sorted_test_cameras = sorted(scene.getTestCameras(), key=lambda cam: cam.image_name)
        sorted_train_cameras = sorted(scene.getTrainCameras(), key=lambda cam: cam.image_name)
        validation_configs = (
            {'name': 'test', 'cameras' : sorted_test_cameras},
            {'name': 'train', 'cameras' : sorted_train_cameras[:min(5, len(sorted_train_cameras))]},
        )

        light_stream, calc_stream = create_render_streams()
        local_axises = scene.gaussians.get_local_axis # (K, 3, 3)
        asg_scales = scene.gaussians.asg_func.get_asg_lam_miu # (basis_asg_num, sg_num, 2)
        asg_axises = scene.gaussians.asg_func.get_asg_axis    # (basis_asg_num, sg_num, 3, 3)
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                temp_name = "z" + config['name']
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                fixed_dir = Path(scene.model_path) / "convergence" / "fixed_views" / f"iter_{iteration:05d}" / config["name"]
                fixed_view_panels = []
                for idx, viewpoint_cam in enumerate(config['cameras']):
                    render_pkg = render(viewpoint_cam, scene.gaussians, light_stream, calc_stream, local_axises, asg_scales, asg_axises, **renderArgs)
                    mimage, shadow, other_effects = render_pkg["render"], render_pkg["shadow"], render_pkg["other_effects"]
                    image = mimage * shadow + other_effects
                    if modelset.gamma_change:
                        image = image.clamp(0.0, 1.0)
                        image = image / (1.0 + image)
                        image = torch.clip(image, 0.0, 1.0)
                    else:
                        image = image.clamp(0.0, 1.0)
                    # image = torch.pow(image, 1./2.2)
                    gt_image = torch.clamp(viewpoint_cam.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(temp_name + "_view_{}/render".format(viewpoint_cam.image_name), image[None].pow(1./gamma), global_step=iteration)
                        tb_writer.add_images(temp_name + "_view_{}/shadow".format(viewpoint_cam.image_name), shadow[None].pow(1./gamma), global_step=iteration)
                        tb_writer.add_images(temp_name + "_view_{}/other_effects".format(viewpoint_cam.image_name), other_effects[None].pow(1./gamma), global_step=iteration)
                        tb_writer.add_images(temp_name + "_view_{}/mimage".format(viewpoint_cam.image_name), mimage[None].pow(1./gamma), global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(temp_name + "_view_{}/ground_truth".format(viewpoint_cam.image_name), gt_image[None].pow(1./gamma), global_step=iteration)
                    view_l1 = l1_loss(image, gt_image).mean().double()
                    view_psnr = psnr(image, gt_image).mean().double()
                    view_ssim = ssim(image, gt_image).mean().double()
                    view_lpips = get_lpips_model().forward(image, gt_image).squeeze()
                    if idx < 3:
                        fixed_metrics = {
                            "iteration": int(iteration),
                            "split": config["name"],
                            "view": viewpoint_cam.image_name,
                            "l1": float(view_l1.item()),
                            "psnr": float(view_psnr.item()),
                            "ssim": float(view_ssim.item()),
                            "lpips": float(view_lpips.item()),
                            "texture_effect_mode": str(getattr(modelset, "texture_effect_mode", "per_uv")),
                        }
                        view_dir = fixed_dir / f"{idx:03d}_{viewpoint_cam.image_name}"
                        save_fixed_view_artifacts(
                            view_dir,
                            gt_image,
                            image,
                            mimage,
                            shadow,
                            fixed_metrics,
                        )
                        fixed_view_panels.append((viewpoint_cam.image_name, view_dir / "panel.png"))

                    l1_test += view_l1
                    psnr_test += view_psnr
                    ssim_test += view_ssim
                    lpips_test += view_lpips
                save_fixed_view_collection(fixed_dir, fixed_view_panels)
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                eval_row = {
                    "iteration": int(iteration),
                    "split": config["name"],
                    "num_views": int(len(config["cameras"])),
                    "l1": float(l1_test.item()),
                    "psnr": float(psnr_test.item()),
                    "ssim": float(ssim_test.item()),
                    "lpips": float(lpips_test.item()),
                    "fixed_view": config["cameras"][0].image_name,
                    "texture_effect_mode": str(getattr(modelset, "texture_effect_mode", "per_uv")),
                }
                shadow_stage = get_shadow_backward_stage(modelset, iteration)
                eval_row.update(
                    {
                        "detach_shadow": bool(getattr(modelset, "detach_shadow", False)),
                        "use_shadow_refine_mlp": bool(getattr(modelset, "use_shadow_refine_mlp", True)),
                        "shadow_backward_stage_enabled": bool(getattr(modelset, "shadow_backward_stage_enabled", False)),
                        "shadow_scale_reg_enabled": bool(getattr(modelset, "shadow_scale_reg_enabled", False)),
                        "shadow_backward_xyz_enabled": bool(shadow_stage["xyz"]),
                        "shadow_backward_opacity_enabled": bool(shadow_stage["opacity"]),
                        "shadow_backward_scaling_enabled": bool(shadow_stage["scaling"]),
                        "shadow_backward_rotation_enabled": bool(shadow_stage["rotation"]),
                    }
                )
                append_jsonl(eval_log_path, eval_row)
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], l1_test, psnr_test,ssim_test,lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(temp_name + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(temp_name + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(temp_name + '/loss_viewpoint - ssim', ssim_test, iteration)  
                    tb_writer.add_scalar(temp_name + '/loss_viewpoint - lpips', lpips_test, iteration)  


        if tb_writer:
            tb_writer.add_histogram("2_scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('2_scene/total_points', scene.gaussians.get_xyz.shape[0], iteration)



        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--unfreeze_iterations", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])     # 存储的迭代次数列表
    parser.add_argument("--start_checkpoint", type=str, default = None)

    # 令 args 包含 parser 中定义的所有参数的键和值
    args = parser.parse_args(sys.argv[1:])  # argument vector， 第一位是文件名，所以从第二位开始解析
    if args.iterations not in args.save_iterations:
        args.save_iterations.append(args.iterations)
    if args.iterations not in args.test_iterations:
        args.test_iterations.append(args.iterations)
    if args.iterations not in args.checkpoint_iterations:
        args.checkpoint_iterations.append(args.iterations)
    
    # Prepare training
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)  #设置输出流是否静默

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)  #如果命令行参数中包含 --detect_anomaly，则根据 stong_true 设置为True，将进行异常检测


    # Start training
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.unfreeze_iterations, args.debug_from)

    # All done
    print("\nTraining complete.")
