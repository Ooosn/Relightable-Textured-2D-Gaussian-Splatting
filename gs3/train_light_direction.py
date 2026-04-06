import os
from PIL import Image
from re import A
import torch
import numpy as np
import math
from torchviz import make_dot
from random import randint
import torchvision
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.system_utils import searchForMaxIteration
from scene.neural_phase_function import Neural_phase
from scene.mixture_ASG import Mixture_of_ASG
import uuid
from rich import print
from rich.panel import Panel
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from tqdm import tqdm
from utils.image_utils import psnr
from utils.general_utils import freeze_anything
from argparse import ArgumentParser, Namespace
from utils.general_utils import PILtoTorch, ExrtoTorch
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import lpips
loss_fn_alex = lpips.LPIPS(net='vgg')


asgmlp_debug = False



def training_light_direction(modelset, opt, pipe, testing_iterations, saving_iterations, 
                          checkpoint_iterations, load_iteration, final_iteration, \
                          write_images, valid, skip_train, skip_test, opt_pose):

    first_iter = 0
    gaussians = GaussianModel(modelset)
    tb_writer = prepare_output_and_logger(modelset)
    if load_iteration == -1:
        load_iteration = searchForMaxIteration(os.path.join(modelset.model_path, "point_cloud"))
    
    _model_path = os.path.join(modelset.model_path, f"chkpnt{load_iteration}.pth")
    print(modelset.source_path)
    if opt_pose:
        modelset.source_path = os.path.join(modelset.model_path, f'point_cloud/iteration_{load_iteration}')
    
    
    scene = Scene(modelset, gaussians, load_iteration=load_iteration, shuffle=False, valid=valid, \
                  skip_train=skip_train, skip_test=skip_test)
    
    if os.path.exists(_model_path):
        # initialize ASG and MLP
        (model_params, _) = torch.load(_model_path, weights_only=False)
    else:
        raise Exception(f"Could not find : {_model_path}")
            
    if modelset.use_nerual_phasefunc:
        gaussians.asg_func = Mixture_of_ASG(modelset.basis_asg_num, modelset.asg_channel_num)
        gaussians.neural_phasefunc = Neural_phase(hidden_feature_size=modelset.phasefunc_hidden_size, \
                                    hidden_feature_layers=modelset.phasefunc_hidden_layers, \
                                    frequency=modelset.phasefunc_frequency, \
                                    neural_material_size=modelset.neural_material_size, \
                                    asg_mlp = gaussians.asg_mlp).to(device="cuda")
        if isinstance(model_params, dict):
            # load ASG parameters
            gaussians.asg_func.asg_sigma = model_params["asg_sigma"]
            gaussians.asg_func.asg_rotation = model_params["asg_rotation"]
            gaussians.asg_func.asg_scales = model_params["asg_scales"]
            # load MLP parameters
            gaussians.neural_phasefunc.load_state_dict(model_params["neural_phasefunc"])
            gaussians.neural_phasefunc.eval()

        else:
            # load ASG parameters
            gaussians.asg_func.asg_sigma = model_params[8]
            gaussians.asg_func.asg_rotation = model_params[9]
            gaussians.asg_func.asg_scales = model_params[10]
            # load MLP parameters
            gaussians.neural_phasefunc.load_state_dict(model_params[14])
            gaussians.neural_phasefunc.eval()


    bg_color = [1, 1, 1, 1, 0, 0, 0] if modelset.white_background else [0, 0, 0, 0, 0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    
    prune_visibility = False    # 可见性修剪，是否剔除不可见的点，可以释放内存，提高内存使用效率
    viewpoint_stack = None    # 存储相机视点（viewpoints）的堆栈，在训练过程中，会从堆栈中弹出一个相机视点，用于渲染图像，从而完成对所有视角的遍历。
    opt_test = False    # 当前是否处于优化测试模式
    opt_test_ready = False    # 是否准备好进行优化测试
    
    ema_loss_for_log = 0.0    # 初始值为 0.0，表示尚未计算损失值
    progress_bar = tqdm(range(first_iter, final_iteration), desc="Training progress")    #使用 tqdm 创建一个可视化的进度条，用于显示训练的进度
    first_iter += 1    # 迭代次数加一，开始下一次迭代
    print("first_iter", first_iter)
    
    
    # 添加光源参数，冻结其它参数
    freeze_anything(gaussians)
    light_direction = torch.tensor([-80, -47, 50], dtype=torch.float32, device="cuda")
    gaussians.light_direction = torch.nn.Parameter(light_direction, requires_grad=True)
    l = [
        {'params': [gaussians.light_direction], 'lr': 0.00001, "name": "light_direction"}
    ]
    optimizer = torch.optim.Adam(l, lr=0.001)
    
    
    light_stream = torch.cuda.Stream()  # 创建新的 CUDA 流
    calc_stream = torch.cuda.Stream()   # 创建另一个独立的 CUDA 流
    bg = torch.rand((7), device="cuda") if opt.random_background else background
    viewpoint_cam = scene.getValidCameras()[0]
    image = Image.open(r"E:\gsrelight-data\NRHints\250703_original\Pixiu\valid\ours_100000\renders\volume_shadow\100000.png")
    im_data = np.array(image.convert("RGBA"))
    norm_data = im_data / 255.0
    norm_data = torch.from_numpy(norm_data).to(bg.dtype).to(bg.device)
    arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg[0:3] * (1 - norm_data[:, :, 3:4])
    gt_image = arr.permute(2,0,1).clamp(0.0, 1.0).to("cuda")
    viewpoint_cam.original_image = gt_image
    torchvision.utils.save_image(gt_image, r"D:\log\output_gt.png")
    for iteration in range(first_iter, final_iteration):
        iter_start.record()    # 记录迭代开始的时间
        
        # if not viewpoint_stack:
        # # only do pose opt for test sets
        #     if opt_test_ready and scene.optimizing:
        #         opt_test = True
        #         # 重新填装测试视点堆栈
        #         viewpoint_stack = scene.getTestCameras().copy()
        #         opt_test_ready = False
        #     else:
        #         opt_test = False
        #         # 重新填装训练视点堆栈
        #         viewpoint_stack = scene.getTrainCameras().copy()
        #         opt_test_ready = True
        
        # viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

            
            
        local_axises = gaussians.get_local_axis # (K, 3, 3)
        asg_scales = gaussians.asg_func.get_asg_lam_miu # (basis_asg_num, 2)
        asg_axises = gaussians.asg_func.get_asg_axis    # (basis_asg_num, 3, 3)
        
        renderArgs = {"modelset": modelset, "pipe": pipe, "bg_color": bg, "is_train": prune_visibility, "asg_mlp": False, "iteration": iteration}
        render_pkg = render(viewpoint_cam, gaussians, light_stream, calc_stream, local_axises, asg_scales, asg_axises, **renderArgs)


        image, shadow, other_effects, backward_info, expected_depth = render_pkg["render"], render_pkg["shadow"], render_pkg["other_effects"], render_pkg["backward_info"], render_pkg["expected_depth"]

        
        image = shadow.repeat(3, 1, 1)
        gt_image = viewpoint_cam.original_image.cuda()
        image = torch.clamp(image, min=1e-4) 
        image = torch.clip(image, 0.0, 1.0)
        
        if write_images and iteration % 10 == 0:
            idx = iteration // 10
            print(gaussians.light_direction)
            torchvision.utils.save_image(image, rf"E:\gsrelight-data\NRHints\250703_original\Pixiu\valid\ours_100000\renders\volume_shadow\training_process\image_{idx:05d}.png")
        Ll1 = l1_loss(image, gt_image)      # lamda_dssim 默认 0.2
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        # dot = make_dot(loss, params={"light_direction": gaussians.light_direction})
        # dot.render("graph", format="pdf")  # 生成 graph.pdf
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
                modelset = modelset)
            
            optimizer.step()
            optimizer.zero_grad(set_to_none = True)
            
            
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
    if not os.path.exists(os.path.join(args.model_path, "cfg_args")):
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
    mp = ModelParams(parser, sentinel=True)
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
    
    
    # 训练相关的 args
    parser.add_argument("--skip_train", action="store_true", default=True)
    parser.add_argument("--skip_test", action="store_true", default=True)
    parser.add_argument("--valid", action="store_true", default=True)
    parser.add_argument("--load_iteration", type=int, default=-1)
    parser.add_argument("--final_iteration", type=int, default=10000)
    parser.add_argument("--write_images", action="store_true", default=False)
    parser.add_argument("--gamma", action="store_true", default=False)
    parser.add_argument("--opt_pose", action="store_true", default=False)
    try:
        args = get_combined_args(parser)
    except:
        import sys
        args = parser.parse_args(sys.argv[1:])
        print(11)
    args.wang_debug = False
    
    args_info = f"""
    model_args: {vars(mp.extract(args))}
    load_iteration: {args.load_iteration}
    skip_train: {args.skip_train}
    skip_test: {args.skip_test}
    gamma: {args.gamma}
    hdr: {args.hdr}
    valid: {args.valid}
    """
    
    # Prepare training
    print(Panel(args_info, title="Arguments", expand=False))
    # Initialize system state (RNG)
    safe_state(args.quiet)  #设置输出流是否静默

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)  #如果命令行参数中包含 --detect_anomaly，则根据 stong_true 设置为True，将进行异常检测

    # Start training
    training_light_direction(mp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, \
                             args.save_iterations, args.checkpoint_iterations, \
                             args.load_iteration, args.final_iteration, args.write_images, \
                             args.valid, args.skip_train, args.skip_test, args.opt_pose)

    # All done
    print("\nTraining complete.")