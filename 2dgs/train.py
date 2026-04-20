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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from scene.cameras import _getWorld2View2_cu
from utils.general_utils import safe_state
from utils.lie_groups import exp_map_SO3xR3
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def _camera_projection_bridge(viewpoint_cam, means3D, viewspace_point_tensor, transmat_grad_holder=None, splat2world=None):
    """Bridge dL/d(pixel mean2D) back into cam_pose_adj through full_proj_transform.

    The surfel rasterizer treats view/proj matrices as constants inside CUDA, so
    pose optimization otherwise only receives the regularization term. We rebuild
    a lightweight differentiable projection here and inject the already-computed
    dL/dmean2D from the rasterizer back into the camera parameters.
    """
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
    world_view_transform = _getWorld2View2_cu(
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

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
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
    scene = Scene(dataset, gaussians, opt=opt)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    opt_test = False
    opt_test_ready = False
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    # ── Multi-phase training schedule (mirrors gs3) ──────────────────────────
    # Retrieve phase thresholds (with safe defaults for models that don't use mBRDF)
    unfreeze_iter   = getattr(opt, "unfreeze_iterations", 5000)
    spcular_freeze  = getattr(opt, "spcular_freeze_step", 9000)
    fit_linear      = getattr(opt, "fit_linear_step", 7000)
    asg_freeze      = getattr(opt, "asg_freeze_step", 22000)
    use_mbrdf       = getattr(dataset, "use_mbrdf", False)

    if use_mbrdf and gaussians.neural_phasefunc is not None:
        if first_iter < unfreeze_iter:
            gaussians.neural_phasefunc.freeze()
        elif first_iter < spcular_freeze + fit_linear:
            gaussians.neural_phasefunc.freeze()
        else:
            gaussians.neural_phasefunc.unfreeze()
    asg_freezed = True

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(
            iteration,
            asg_freeze_step=getattr(opt, "asg_lr_freeze_step", 0),
            local_q_freeze_step=getattr(opt, "local_q_lr_freeze_step", 0),
            freeze_phasefunc_steps=getattr(opt, "freeze_phasefunc_steps", 0),
        )

        # ── Camera / light pose optimization ──────────────────────────────────
        scene.update_lr(iteration, opt)

        # ── Phase control ─────────────────────────────────────────────────────
        if use_mbrdf and gaussians.neural_phasefunc is not None:
            if iteration == unfreeze_iter:
                gaussians.neural_phasefunc.unfreeze()
                print(f"\n[ITER {iteration}] Phase 1: unfreeze neural_phasefunc, shadow ON")
            elif iteration == spcular_freeze:
                gaussians.neural_phasefunc.freeze()
                if hasattr(gaussians, "neural_material"):
                    gaussians.neural_material.requires_grad_(False)
                print(f"\n[ITER {iteration}] Phase 2: re-freeze phasefunc, diffuse-only fit")
            elif iteration == spcular_freeze + fit_linear:
                gaussians.neural_phasefunc.unfreeze()
                if hasattr(gaussians, "neural_material"):
                    gaussians.neural_material.requires_grad_(True)
                print(f"\n[ITER {iteration}] Phase 3: unfreeze phasefunc + specular ks")

        if use_mbrdf and gaussians.asg_func is not None:
            if iteration <= asg_freeze:
                gaussians.asg_func.asg_scales.requires_grad_(False)
                gaussians.asg_func.asg_rotation.requires_grad_(False)
            elif asg_freezed:
                asg_freezed = False
                gaussians.asg_func.asg_scales.requires_grad_(True)
                gaussians.asg_func.asg_rotation.requires_grad_(True)
                print(f"\n[ITER {iteration}] Phase 4: unfreeze ASG anisotropic params")

        # Determine render flags for this phase, mirroring gs3 exactly:
        #  Phase 0 (0 → unfreeze):                   kd flat,  no shadow  (geometry warmup)
        #  Phase 1 (unfreeze → spcular_freeze):       kd*decay, shadow ON  (phasefunc learns shadow)
        #  Phase 2 (spcular_freeze → +fit_linear):    kd*decay, shadow ON  (phasefunc frozen, kd fit)
        #  Phase 3 (spcular_freeze+fit_linear → asg): full BRDF*decay, shadow ON
        #  Phase 4 (asg_freeze → ):                   full BRDF*decay + ASG aniso
        if not use_mbrdf:
            fix_lambert  = False
            apply_shadow = False
        elif iteration <= unfreeze_iter:
            # Phase 0: geometry warmup, pure kd, no phasefunc cost
            fix_lambert  = True
            apply_shadow = False
        elif iteration <= spcular_freeze + fit_linear:
            # Phase 1 & 2: kd * phasefunc_decay (specular OFF, shadow ON)
            fix_lambert  = True
            apply_shadow = True
        else:
            # Phase 3 & 4: full BRDF * decay
            fix_lambert  = False
            apply_shadow = True

        # Match gs3: SH degree growth is only meaningful for the non-mBRDF path.
        if iteration % 1000 == 0 and not use_mbrdf:
            gaussians.oneupSHdegree()

        # Pick a random Camera. When pose/light optimization is active, mirror gs3:
        # alternate between train cameras and test cameras so test poses/lights are
        # also optimized instead of being left as dead optimizer params.
        if not viewpoint_stack:
            if scene.pose_optimizer is not None and opt_test_ready:
                opt_test = True
                viewpoint_stack = scene.getTestCameras().copy()
                opt_test_ready = False
            else:
                opt_test = False
                viewpoint_stack = scene.getTrainCameras().copy()
                if scene.pose_optimizer is not None:
                    opt_test_ready = True
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Apply learned pose / light adjustment before rendering
        if scene.pose_optimizer is not None:
            viewpoint_cam.update()

        render_pkg = render(viewpoint_cam, gaussians, pipe, background,
                            fix_lambert=fix_lambert, apply_shadow=apply_shadow,
                            deferred=True)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        transmat_grad_holder = render_pkg.get("transmat_grad_holder")
        
        gt_image = viewpoint_cam.original_image.cuda()
        if getattr(dataset, "use_mbrdf", False):
            # Match the working gs3 relighting loss path: supervise the final
            # image in display space and do not add 2DGS-specific normal/dist
            # regularizers on top of the BRDF training objective.
            image = torch.clamp(image, 0.0, 1.0)
            normal_loss = torch.zeros([], dtype=image.dtype, device=image.device)
            dist_loss = torch.zeros([], dtype=image.dtype, device=image.device)
        else:
            # Vanilla 2DGS path keeps the original geometric regularizers.
            lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
            lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

            rend_dist = render_pkg["rend_dist"]
            rend_normal  = render_pkg['rend_normal']
            surf_normal = render_pkg['surf_normal']
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            normal_loss = lambda_normal * (normal_error).mean()
            dist_loss = lambda_dist * (rend_dist).mean()

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        total_loss = loss + dist_loss + normal_loss

        total_loss.backward()

        # Step pose optimizer (camera + light)
        if scene.pose_optimizer is not None:
            _camera_projection_bridge(
                viewpoint_cam,
                gaussians.get_xyz,
                viewspace_point_tensor,
                transmat_grad_holder=transmat_grad_holder,
                splat2world=gaussians.get_covariance(1.0),
            )
            scene.pose_optimizer.step()
            scene.pose_optimizer.zero_grad(set_to_none=True)
            viewpoint_cam.update()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log


            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            # Only train/test pose-light optimization should run on test cameras.
            if not opt_test:
                # Densification
                if iteration < opt.densify_until_iter:
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                    
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
            else:
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

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
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    if hasattr(viewpoint, "update"):
                        viewpoint.update()
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")
