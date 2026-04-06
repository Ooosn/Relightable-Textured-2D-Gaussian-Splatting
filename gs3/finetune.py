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
import math
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torch import nn
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import lpips
loss_fn_alex = lpips.LPIPS(net='vgg')


asgmlp_debug = True

#training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.unfreeze_iterations, args.debug_from)
def finetune(modelset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, start_checkpoint, unfreeze_iterations, debug_from):
    first_iter = 0

    tb_writer = prepare_output_and_logger(modelset)

    # 初始化高斯点参数
    gaussians = GaussianModel(modelset, opt)

    # 冻结相机和光源参数
    """
    if modelset.use_hgs_finetune:
        self.cam_pose_adj = torch.nn.Parameter(torch.zeros((1, 6), requires_grad=False).cuda())
        self.pl_adj = torch.nn.Parameter(torch.zeros((1, 3), requires_grad=False).cuda())
    """
    scene = Scene(modelset, gaussians, opt=opt, shuffle=True)   # model.finetune = True, 因此 相机和光源参数不参与训练

    # 冻结非微调 高斯点参数
    """
    def training_setup(self, training_args) 处冻结非微调高斯点参数
    """
    gaussians.training_setup(opt)

    scene.optimizing = False

    if start_checkpoint:
        (model_params, first_iter) = torch.load(start_checkpoint)
        gaussians.restore(model_params, opt)
    else:
        assert False, "finetune: start_checkpoint is not provided"
    

    bg_color = [1, 1, 1, 1, 0, 0, 0] if modelset.white_background else [0, 0, 0, 0, 0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    prune_visibility = False   
    viewpoint_stack = None    
    opt_test = False   
    opt_test_ready = False   
    ema_loss_for_log = 0.0    
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")    
    first_iter += 1    

    phase_func_freezed = False
    asg_freezed = True
    asg_mlp = False
    loss_fn_alex.to(gaussians.get_features.device)

    if first_iter < unfreeze_iterations:
        gaussians.neural_phasefunc.freeze()
        phase_func_freezed = True

    light_stream = torch.cuda.Stream()  
    calc_stream = torch.cuda.Stream()   



    """开始训练"""
    for iteration in range(first_iter, opt.iterations + 10):   
        iter_start.record()    

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

        if iteration % 1000 == 0:
            if not modelset.use_nerual_phasefunc:
                gaussians.oneupSHdegree()


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

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))



        """！开始渲染"""
        if (iteration - 1) == debug_from:
            pipe.debug = True
        bg = torch.rand((7), device="cuda") if opt.random_background else background
        
        # precompute shading frames and ASG frames
        local_axises = gaussians.get_local_axis # (K, 3, 3)
        asg_scales = gaussians.asg_func.get_asg_lam_miu # (basis_asg_num, 2)
        asg_axises = gaussians.asg_func.get_asg_axis    # (basis_asg_num, 3, 3)

        # only opt with diffuse term at the beginning for a stable training process
        if iteration < opt.spcular_freeze_step + opt.fit_linear_step:   
            renderArgs = {"pipe": pipe, "bg_color": bg, "fix_labert": True, "is_train": prune_visibility, "asg_mlp": asg_mlp, "iteration": iteration}
            render_pkg = render(viewpoint_cam, gaussians, light_stream, calc_stream, local_axises, asg_scales, asg_axises, **renderArgs) 
        else:     
            renderArgs = {"pipe": pipe, "bg_color": bg, "is_train": prune_visibility, "asg_mlp": asg_mlp, "iteration": iteration}
            render_pkg = render(viewpoint_cam, gaussians, light_stream, calc_stream, local_axises, asg_scales, asg_axises, **renderArgs)
            
        viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        image, shadow, other_effects = render_pkg["render"], render_pkg["shadow"], render_pkg["other_effects"]

        if False:   
            if render_pkg["out_weight"].sum() > 0:
                print("################################")
                print(render_pkg["out_weight"])
                print("################################")
        
        if iteration <= unfreeze_iterations:
            image = image
        else:
            image = image * shadow + other_effects


        watch_image = True    
        if watch_image:
            # 将图像转换为numpy数组并显示
            import matplotlib.pyplot as plt
            import numpy as np
            from matplotlib.widgets import Button

            # 将tensor转换为numpy数组并调整维度顺序
            img_np = image.detach().cpu().numpy()
            img_np = np.transpose(img_np, (1, 2, 0))

            # 创建新的图形窗口和子图
            fig, ax = plt.subplots(figsize=(10, 10))
            plt.subplots_adjust(bottom=0.15) # 为按钮留出空间
            ax.imshow(img_np)
            ax.axis('off')
            ax.set_title(f'Iteration {iteration}')

            # 创建下一步按钮
            ax_button = plt.axes([0.8, 0.05, 0.1, 0.05])
            btn_next = Button(ax_button, 'Next')

            def on_next(event):
                plt.close()

            btn_next.on_clicked(on_next)
            plt.show()




        """！！Loss部分"""
        # 获取真实图片数据
        gt_image = viewpoint_cam.original_image.cuda()

        if modelset.hdr:
            if iteration <= opt.spcular_freeze_step:
                gt_image = torch.pow(gt_image, 1./2.2)
            elif iteration < opt.spcular_freeze_step + opt.fit_linear_step//2:
                gamma = 1.1 * float(opt.spcular_freeze_step + opt.fit_linear_step - iteration + 1) / float(opt.fit_linear_step // 2 + 1)
                gt_image = torch.pow(gt_image, 1./gamma)
        else:
            image = torch.clip(image, 0.0, 1.0)

        Ll1 = l1_loss(image, gt_image)      # lamda_dssim 默认 0.2
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # 反向传播，计算各个参数的梯度
        # 尚未更新参数，等待后续挑选更新
        loss.backward()
        iter_end.record() # 记录迭代结束的时间

        scene.optimizer.zero_grad(set_to_none = True)
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none = True)



        """！！参数更新部分"""
        # torch.no_grad() 防止污染计算图，加快计算速度
        with torch.no_grad():
            # Progress bar，平滑损失曲线
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), \
                testing_iterations, scene, render, renderArgs, gamma=2.2 if modelset.hdr else 1.0)
            
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

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

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, gamma=1.0):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        light_stream = torch.cuda.Stream()
        calc_stream = torch.cuda.Stream()
        local_axises = scene.gaussians.get_local_axis # (K, 3, 3)
        asg_scales = scene.gaussians.asg_func.get_asg_lam_miu # (basis_asg_num, sg_num, 2)
        asg_axises = scene.gaussians.asg_func.get_asg_axis    # (basis_asg_num, sg_num, 3, 3)
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                for idx, viewpoint_cam in enumerate(config['cameras']):
                    render_pkg = render(viewpoint_cam, scene.gaussians, light_stream, calc_stream, local_axises, asg_scales, asg_axises, **renderArgs)
                    mimage, shadow, other_effects = render_pkg["render"], render_pkg["shadow"], render_pkg["other_effects"]
                    image = torch.clamp(mimage * shadow + other_effects, 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint_cam.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint_cam.image_name), image[None].pow(1./gamma), global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint_cam.image_name), gt_image[None].pow(1./gamma), global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    lpips_test += loss_fn_alex.forward(image, gt_image).squeeze()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])        
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], l1_test, psnr_test,ssim_test,lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)  
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)  


        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
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
    args.save_iterations.append(args.iterations)    # 将最后一个迭代次数加入保存迭代次数列表
    
    # Prepare training
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)  #设置输出流是否静默

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)  #如果命令行参数中包含 --detect_anomaly，则根据 stong_true 设置为True，将进行异常检测


    # Start training
    finetune(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.unfreeze_iterations, args.debug_from)

    # All done
    print("\nTraining complete.")
