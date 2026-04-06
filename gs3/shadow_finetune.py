import os
from re import A
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
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import lpips
loss_fn_alex = lpips.LPIPS(net='vgg')


asgmlp_debug = False

#training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.unfreeze_iterations, args.debug_from)
def training(modelset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, unfreeze_iterations, debug_from):

    tb_writer = prepare_output_and_logger(modelset)
    gaussians = GaussianModel(modelset, opt)
    (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
    
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


    ema_loss_for_log = 0.0    
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")   
    first_iter += 1   
    print("Prepare for shadow finetune, First_iter", first_iter)


    phase_func_freezed = False
    asg_freezed = True
    asg_mlp = False

    info = None
    crop_extent = None
    record_info = None

    if True:
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
    
    
    gaussians.neural_phasefunc.freeze()
    for param in gaussians.asg_func.parameters():
        param.requires_grad = False
    
    
        
    # initialize parallel GPU stream 多流并行
    # 有时会出现错误，可以尝试关闭，改为串行，torch.cuda.current_stream().synchronize()
    light_stream = torch.cuda.Stream()  # 创建新的 CUDA 流
    calc_stream = torch.cuda.Stream()   # 创建另一个独立的 CUDA 流




    """开始训练"""
    # 每次迭代，都会从视点堆栈中选择一个视点，然后渲染图像，计算损失，更新模型参数，并不是每次计算全部视点的损失
    for iteration in range(first_iter, opt.iterations + 1):    #左闭右开区间，因此加1
        iter_start.record()    # 记录迭代开始的时间

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
               
        renderArgs = {"pipe": pipe, "bg_color": bg, "is_train": prune_visibility, "asg_mlp": asg_mlp, "iteration": iteration}
        render_pkg = render(viewpoint_cam, gaussians, light_stream, calc_stream, local_axises, asg_scales, asg_axises, **renderArgs)
        
        # 此外，取出各个高斯点云坐标、可见性、半径，用于后续修剪
        viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        image, shadow, other_effects, backward_info = render_pkg["render"], render_pkg["shadow"], render_pkg["other_effects"], render_pkg["backward_info"]




        """！！Loss部分"""
        # 获取真实图片数据
        gt_image = viewpoint_cam.original_image.cuda()
       
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
                pass

        Ll1 = l1_loss(image, gt_image)      # lamda_dssim 默认 0.2
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # 反向传播，计算各个参数的梯度
        # 尚未更新参数，等待后续挑选更新
        loss.backward()
        iter_end.record() # 记录迭代结束的时间



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
                testing_iterations, scene, render, renderArgs, gamma=2.2 if modelset.hdr else 1.0, \
                metrics_by_category = metrics_by_category, info = record_info, 
                modelset = modelset)
            
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            # 对于测试集：：：：：  
            # 只用于优化场景光源和相机信息，因为这些信息可能是未知的，需要通过优化来估计    
            if opt_test and scene.optimizing:  
                if iteration < opt.iterations:
                    scene.optimizer.step()
                    scene.optimizer.zero_grad(set_to_none = True)
                    gaussians.optimizer.zero_grad(set_to_none = True)

            else:
                if iteration < opt.densify_until_iter:
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, image.shape[2], image.shape[1], render_pkg["out_weight"])
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                        smooth = False
                        import time
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
                            except: pass
                            #except Exception as e: print(f"[error] failed to update '{key}': {str(e)}")
                        
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
                        num_clone, num_split = gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, opt.bigsize_threshold, crop_extent)
                        

                        record_info["num_clone_ratio"] = record_info["num_clone_ratio"]*0.7 + num_clone / gaussians.get_xyz.shape[0]*0.3
                        record_info["num_split_ratio"] = record_info["num_split_ratio"]*0.7 + num_split / gaussians.get_xyz.shape[0]*0.3
                        
                        ks = gaussians.get_ks.detach()
                        ks = ks.abs().float().mean()
                        kd = gaussians.get_kd.detach()
                        kd = kd.abs().float().mean()
                        record_info["ks"] = record_info["ks"]*0.7 + ks*0.3
                        record_info["kd"] = record_info["kd"]*0.7 + kd*0.3
                        


                        if gaussians.get_xyz.shape[0] > gaussians.maximum_gs * 0.95:
                            prune_visibility = True
                    

                    if iteration % opt.opacity_reset_interval == 0 or (modelset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

                else:
                    prune_visibility = False

                if iteration < opt.iterations:  
                    gaussians.optimizer.step()
                    # opt the camera pose
                    if scene.optimizing:    
                        scene.optimizer.step()
                        scene.optimizer.zero_grad(set_to_none = True)
                    gaussians.optimizer.zero_grad(set_to_none = True)   # 梯度清零


            # 判断是否保存模型
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


        # 更新相机和光源参数：（之前只是更新了 adj 参数，尚未对相机和光源进行直接更新)，注意这里没有设置 torch.no_grad()
        # update cam and light
        if scene.optimizing:
            viewpoint_cam.update("SO3xR3")
                


    


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

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, gamma=1.0, metrics_by_category=None, info=None, modelset=None):
    if tb_writer:
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
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        light_stream = torch.cuda.Stream()
        calc_stream = torch.cuda.Stream()
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
                write_gt = False
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
                        if iteration == testing_iterations[0] and not write_gt:
                            tb_writer.add_images(temp_name + "_view_{}/ground_truth".format(viewpoint_cam.image_name), gt_image[None].pow(1./gamma), global_step=iteration)
                            write_gt = True
                        
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
    args.save_iterations.append(args.iterations)    # 将最后一个迭代次数加入保存迭代次数列表
    
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
