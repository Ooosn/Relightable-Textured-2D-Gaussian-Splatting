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


import torch
import copy
import math
import cv2
from scipy.spatial.transform import Rotation as Rot
import torchvision.transforms.functional as F
from PIL import Image
from scene import Scene
import os
import numpy as np
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
from gaussian_renderer.shadow_render import shadow_render
import torchvision
from utils.general_utils import safe_state
from utils.graphics_utils import focal2fov
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene.gaussian_model_2dgs_adapter import GaussianModel2DGSAdapter
from scene.neural_phase_function import Neural_phase
from scene.mixture_ASG import Mixture_of_ASG
from utils.system_utils import searchForMaxIteration
from rich import print
from rich.panel import Panel
import time
import matplotlib.pyplot as plt
from write_vedio import images_to_video
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=8)

def _is_2dgs_model(modelset):
    return str(getattr(modelset, "rasterizer", "")).startswith("2dgs")

def _make_gaussian_model(modelset, opt=None):
    if _is_2dgs_model(modelset):
        return GaussianModel2DGSAdapter(modelset, opt)
    return GaussianModel(modelset)

def _unpack_training_checkpoint(checkpoint):
    if not isinstance(checkpoint, (tuple, list)):
        return checkpoint, None, None
    if len(checkpoint) == 2:
        model_params, first_iter = checkpoint
        return model_params, first_iter, None
    if len(checkpoint) == 3:
        model_params, first_iter, scene_state = checkpoint
        return model_params, first_iter, scene_state
    raise ValueError(f"Unsupported checkpoint format with {len(checkpoint)} entries")

def render_set(modelset, name, iteration, views, gaussians, pipeline, background, gamma, hdr, write_images=False, calculate_fps=False, force_save=False, 
               synthesize_video = False, shadowmap_render = False):
    model_path = modelset.model_path
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    local_axises = gaussians.get_local_axis         # (K, 3, 3)
    asg_scales = gaussians.asg_func.get_asg_lam_miu # (basis_asg_num, 2, channel_num)
    asg_axises = gaussians.asg_func.get_asg_axis    # (basis_asg_num, 3, 3)
        
    light_stream = torch.cuda.Stream()
    calc_stream = torch.cuda.Stream()
    render_shadow = None
    render_other_effects = None
    render_base = None
    rendering = None
    renderArgs = {"modelset": modelset, "pipe": pipeline, "bg_color": background}
    if calculate_fps:
        epoch_num = 5
        # 记录渲染花费的平均时间
        total_frames = epoch_num * len(views)
        
        # warm-up 热身
        for idx, view in enumerate(tqdm(views, desc="warming up~~~")):
            _ = render(view, gaussians, light_stream, calc_stream, local_axises, asg_scales, asg_axises, **renderArgs)

        torch.cuda.synchronize()
        print("-"*10,"let's go","-"*10)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        views = views * epoch_num

        for idx, view in enumerate(tqdm(views, desc="calculating fps...")):
            render_pkg = render(view, gaussians, light_stream, calc_stream, local_axises, asg_scales, asg_axises, **renderArgs)
        
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)  # 毫秒
        elapsed_seconds = elapsed / (1000.0*total_frames)
        print("(oﾟ▽ﾟ)oヾ(ﾟ∀ﾟゞ)(ﾉ´▽｀)ﾉ♪(ﾉﾟ▽ﾟ)ﾉヾ(✿ﾟ▽ﾟ)ノ٩(๑❛ᴗ❛๑)۶ヾ(◍°∇°◍)ﾉﾞヾ(๑╹◡╹)ﾉ(๑*◡*๑)٩(๑>◡<๑)۶(๑╹◡╹)ﾉ(๑´ㅂ`๑)")
        print(f"Average render time: {elapsed_seconds:.4f} seconds")
        print(f"FPS: {1/elapsed_seconds:.4f}")
    
    else:
        # 正式渲染
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            render_pkg = render(view, gaussians, light_stream, calc_stream, local_axises, asg_scales, asg_axises, **renderArgs, shadowmap_render=shadowmap_render)
            render_shadow = render_pkg["shadow"]
            render_other_effects = render_pkg["other_effects"]
            render_base = render_pkg["render"]
            rendering = render_pkg["render"]* render_pkg["shadow"] + render_pkg["other_effects"]
            expected_depth = render_pkg.get("expected_depth")
            depth_image = render_pkg.get("depth_image")
            image_folder = {}
            if write_images:
                # 如果图片是 hdr 格式，则 gamma 校正
                # png 格式，是直接在 sRGB 空间 下重建的，因此不需要 gamma 校正
                if gamma:
                    if hdr:
                        gt = torch.pow(gt, 1/2.2)
                    rendering = rendering / (1.0 + rendering)
                    rendering = torch.clip(rendering, 0.0, 1.0)
                    rendering = torch.pow(rendering, 1/2.2)
                
                file_prefix = "shadowmap" if shadowmap_render else "volume"
                image_shadow_path = os.path.join(render_path, file_prefix + "_shadow")
                image_other_effects_path = os.path.join(render_path, 'other_effects')
                image_base_path = os.path.join(render_path, 'base')
                image_final_image = os.path.join(render_path, file_prefix + "_final_image")
                image_folder[image_shadow_path] = render_shadow
                image_folder[image_other_effects_path] = render_other_effects
                image_folder[image_base_path] = render_base
                image_folder[image_final_image] = rendering
                
                if expected_depth is not None:
                    image_expected_depth_path = os.path.join(render_path, 'expected_depth')
                    mask = torch.isnan(expected_depth) | (expected_depth <= 0)
                    expected_depth[mask] = torch.nan

                    # 2. 转视差（可选）
                    disp = 1.0 / expected_depth
                    disp[mask] = torch.nan
                    disp_np = disp.cpu().numpy()
                    # 3. 归一化（用 1–99% 分位数裁剪）
                    vmin, vmax = np.nanpercentile(disp_np, (1, 99))
                    disp_clipped = torch.clip(disp, vmin, vmax)
                    normed_1 = (disp_clipped - vmin) / (vmax - vmin)
                    image_folder[image_expected_depth_path] = normed_1
                    
                    print("percentile:", vmin, vmax)
                    
                if depth_image is not None:
                    image_depth_image_path = os.path.join(render_path, 'depth_image')
                    print("depth min/max:", torch.min(depth_image), torch.max(depth_image))
                    mask = torch.isnan(depth_image) | (depth_image <= 1e-3)
                    depth_image[mask] = torch.nan

                    # 2. 转视差（可选）
                    disp = 1.0 / depth_image
                    disp[mask] = torch.nan
                    disp_np = disp.cpu().numpy()
                    # 3. 归一化（用 1–99% 分位数裁剪）  
                    vmin, vmax = np.nanpercentile(disp_np, (1, 99))
                    print("percentile:", vmin, vmax)
                    disp_clipped = torch.clip(disp, vmin, vmax)
                    normed_2 = (disp_clipped - vmin) / (vmax - vmin)
                    normed_2[mask] = 0
                    image_folder[image_depth_image_path] = normed_2
                
                for key, value in image_folder.items():
                    try:
                        os.makedirs(key, exist_ok=True)
                        if force_save:
                            executor.submit(torchvision.utils.save_image, value, os.path.join(key, '{0:05d}'.format(idx) + ".png"))
                        else:
                            if not os.path.exists(os.path.join(key, '{0:05d}'.format(idx) + ".png")):
                                executor.submit(torchvision.utils.save_image, value, os.path.join(key, '{0:05d}'.format(idx) + ".png"))
                    except:
                        print("problem found in", key)
            
        if synthesize_video:
            # 获取文件夹下的图片，按顺序合成视频
            # video_path = os.path.join(render_path, 'video')
            video_path = render_path
            os.makedirs(video_path, exist_ok=True)
            for key, value in image_folder.items():
                out_file_name = key.split("\\")[-1] + "_video" + ".mp4"
                print(out_file_name)
                out_file_path = os.path.join(video_path, out_file_name)
                images_to_video(key, out_file_path, fps=30, size="auto", codec='mp4v')
            

def render_sets(modelset : ModelParams, 
                iteration : int, 
                pipeline : PipelineParams, 
                skip_train : bool, 
                skip_test : bool, 
                opt_pose: bool, 
                gamma: bool,
                hdr: bool,
                valid: bool,
                write_images: bool,
                calculate_fps: bool,
                force_save: bool,
                aaai_render: bool,
                beijing_render: bool,
                synthesize_video: bool,
                shadowmap_render: bool,
                opt=None):
    modelset.data_device = "cpu"

    if opt_pose:
        modelset.source_path = os.path.join(modelset.model_path, f'point_cloud/iteration_{iteration}')


    with torch.no_grad():

        if aaai_render:
            # 新对象
            ply_file = "obj_010_ours.ply"
            filename_target = [ply_file.split(".")[-2]]
            filename_target_append = ["obj_car_ours", "obj_mianbao_ours"]
            filename_target.extend(filename_target_append)
            prefix = []

            world_forward = None
            # 把面板里的度数填进来
            euler_deg = [0, 0, 0]
            r = Rot.from_euler('xyz', euler_deg, degrees=True)
            local_forward = np.array([0, 0, 1])
            world_forward = (r.as_matrix() @ local_forward)


            use_specific_camera = False

            parser = ArgumentParser(description="aaai render parameters")
            modelset = ModelParams(parser)
            modelset.data_device = "cpu"
            modelset.sh_degree = 3
            modelset.model_path = os.path.join(r"C:\Users\namew\Desktop\aaai", filename_target[0])
            modelset.use_nerual_phasefunc = False
            modelset.cam_opt = False
            modelset.pl_opt = False
            modelset.resolution = 1
            try:
                ply_files = [f for f in os.listdir(modelset.model_path) if f.endswith(".ply")]
                for ply_file in ply_files:
                    prefix.append(ply_file.split("_")[0])
                print(prefix)
            except:
                assert print("no prefix")
            background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
            from scene.dataset_readers import sceneLoadTypeCallbacks
            
            from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
            if not os.path.exists(os.path.join(modelset.model_path, "valid")):
                os.makedirs(os.path.join(modelset.model_path, "valid"), exist_ok=True)
            if not os.path.exists(os.path.join(modelset.model_path, "shadow")):
                os.makedirs(os.path.join(modelset.model_path, "shadow"), exist_ok=True)
            if not os.path.exists(os.path.join(modelset.model_path, "specular_render")):
                os.makedirs(os.path.join(modelset.model_path, "specular_render"), exist_ok=True)
            if not os.path.exists(os.path.join(modelset.model_path, "valid_shadow")):
                os.makedirs(os.path.join(modelset.model_path, "valid_shadow"), exist_ok=True)
            if not os.path.exists(os.path.join(modelset.model_path, "only_specular")):
                os.makedirs(os.path.join(modelset.model_path, "only_specular"), exist_ok=True)
            if not os.path.exists(os.path.join(modelset.model_path, "self_shadow")):
                os.makedirs(os.path.join(modelset.model_path, "self_shadow"), exist_ok=True)
            
            


            updated_light_pos = [0, 0, 5000]
            offset = 0.05
            scene_offset_coefficient = 30
            vehicle_offset_coefficient = offset*0.5
            scale_factor = 1.0
            transparent_layer = False
            need_shadow = False
            
            gaus = []
            # 搜寻当前文件下的所有ply文件
            i = 2.0
            num = 0
            for ply_file in ply_files:
                filename = ply_file.split(".")[0]
                print(filename)
                if filename.startswith('bg'):
                    try:
                        modelset.sh_degree = 1
                        scene_gaussians = GaussianModel(modelset)
                        scene_gaussians.load_ply(os.path.join(modelset.model_path, ply_file))
                        scene_gaussians.object_id = torch.ones_like(scene_gaussians._xyz[:, 0])
                        gaus.append(scene_gaussians)
                    except:
                        modelset.sh_degree = 2
                        scene_gaussians = GaussianModel(modelset)
                        scene_gaussians.load_ply(os.path.join(modelset.model_path, ply_file))
                        scene_gaussians.object_id = torch.ones_like(scene_gaussians._xyz[:, 0])
                        gaus.append(scene_gaussians)
                else:
                    if filename.endswith("ours"):
                        if filename in filename_target:
                            transparent_layer = True
                            need_shadow = True
                            modelset.sh_degree = 3
                            object_gaussians = GaussianModel(modelset)
                            object_gaussians.load_ply(os.path.join(modelset.model_path, ply_file))
                            object_id = torch.zeros_like(object_gaussians._xyz[:, 0])
                            object_id[:] = i
                            print(object_id.shape)
                            object_gaussians.object_id = object_id
                            if os.path.exists(os.path.join(modelset.model_path, filename+"_ground.ply")):


                                ground_gaussians = GaussianModel(modelset)
                                ground_gaussians.load_ply(os.path.join(modelset.model_path, filename+"_ground.ply"))
                                object_id_ground = torch.zeros_like(ground_gaussians._xyz[:, 0])
                                object_id_ground[:] = 100 + i
                                print(object_id_ground.shape)
                                ground_gaussians.object_id = object_id_ground
                                object_gaussians.merge_gaussians([ground_gaussians])
                                print("separate ground")
                                print("separate ground")
                                print("separate ground")
                                
                            else:
                                end_index = object_gaussians._xyz.shape[0]
                                object_id = torch.zeros_like(object_gaussians._xyz[:, 0])
                                object_id[:end_index-40000] = i
                                object_id[end_index-40000:] = 100 + i
                                object_gaussians.object_id = object_id
                                print("include ground")
                                print("include ground")
                            i += 1
                            gaus.append(object_gaussians)
                        else:
                            continue
                    


                    else:
                        if filename in filename_target:
                            modelset.sh_degree = 1
                            object_gaussians = GaussianModel(modelset)
                            object_gaussians.load_ply(os.path.join(modelset.model_path, ply_file))
                            object_id = torch.zeros_like(object_gaussians._xyz[:, 0])
                            object_id[:] = 2
                            object_gaussians.object_id = object_id
                            transparent_layer = False
                            need_shadow = False
                            gaus.append(object_gaussians)
                        else:
                            continue
                        
         



                    scale_factor = 1.0

            print(gaus)
            for gaussian in gaus:
                print(gaussian._xyz.shape)
                print(gaussian.object_id.shape)

            specular_dir_path = os.path.join(modelset.model_path, "specular")
            
 
            

            # 读取相机位姿专用，插车用
            if len(gaus) > 1 or "bg" in prefix:
                modelset.load_num = 1 if use_specific_camera else 400
                scene_info = sceneLoadTypeCallbacks["Blender"](modelset.model_path, modelset._white_background, modelset.eval, modelset.view_num, modelset.load_num, \
                                                            valid=valid, skip_train=skip_train, skip_test=skip_test,
                                                            extension=".exr" if hdr else ".png")
                camera_list = cameraList_from_camInfos(scene_info.valid_cameras, 1.0, modelset) 


                # 寻找所有 pth 文件
                if use_specific_camera:
                    camrea_temp = copy.deepcopy(camera_list[0])
                    camera_list = []
                    pth_files = [f for f in os.listdir(modelset.model_path) if f.endswith(".pth")]
                    for pth_file in pth_files:
                        camrea1 = copy.deepcopy(camrea_temp)
                        pkg = torch.load(os.path.join(modelset.model_path, pth_file))
                        image_width = pkg["image_width"]
                        image_height = pkg["image_height"]
                        tanfovx = pkg["tanfovx"]
                        tanfovy = pkg["tanfovy"]
                        bg = pkg["bg"]
                        scale_modifier = pkg["scale_modifier"]
                        viewmatrix = pkg["viewmatrix"]
                        projmatrix = pkg["projmatrix"]
                        campos = pkg["campos"]
                        camrea1.FoVx = math.atan(tanfovx) * 2
                        camrea1.FoVy = math.atan(tanfovy) * 2
                        camrea1.image_width = image_width
                        camrea1.image_height = image_height
                        camrea1.world_view_transform = viewmatrix
                        camrea1.full_proj_transform = projmatrix
                        camrea1.camera_center = campos
                        camera_list.append(camrea1)
                    
                max_sh_degree = 3
                gaussians = gaus[0]
                gaussians.merge_gaussians(gaus[1:], max_sh_degree)
            else:
                scene_info = sceneLoadTypeCallbacks["Blender"](modelset.model_path, modelset._white_background, modelset.eval, modelset.view_num, modelset.load_num, \
                                                            valid=valid, skip_train=skip_train, skip_test=skip_test,
                                                            extension=".exr" if hdr else ".png")
                camera_list = cameraList_from_camInfos(scene_info.valid_cameras, 1.0, modelset) 
                gaussians = gaus[0]


            use_mesh_info = False
            for idx, view in enumerate(tqdm(camera_list, desc="Rendering progress")):

                

                render_pkg = shadow_render(view, gaussians, pipeline, background, updated_light_pos=updated_light_pos, offset=offset, 
                                           scene_offset_coefficient=scene_offset_coefficient, vehicle_offset_coefficient=vehicle_offset_coefficient, 
                                           transparent_layer=transparent_layer, scale_factor=scale_factor, need_shadow=need_shadow,
                                           light_dir=world_forward)

                # shadow = render_pkg["shadow"].cpu().numpy()  
                # kernel_size = 11
                # kernel = np.ones((kernel_size, kernel_size), np.uint8)

                # # 2. 执行“闭合”操作
                # shadow = cv2.morphologyEx(shadow, cv2.MORPH_CLOSE, kernel)
                # shadow = torch.tensor(shadow).cuda()
                shadow = render_pkg["shadow"]
                render_image = render_pkg["render"]

                save_path = os.path.join(modelset.model_path, "valid", f"{idx:05d}_camera.pt")
                torch.save(render_pkg["raster_settings_dict"], save_path)
                if not use_mesh_info:   
                    torchvision.utils.save_image(render_image, os.path.join(modelset.model_path, "valid", '{0:05d}'.format(idx) + ".png"))
                    torchvision.utils.save_image(shadow, os.path.join(modelset.model_path, "shadow", '{0:05d}'.format(idx) + "_shadow.png"))
                    torchvision.utils.save_image(render_image*shadow, os.path.join(modelset.model_path, "valid_shadow", '{0:05d}'.format(idx) + "_shadow.png"))


                
                if use_mesh_info:
                    shadow_img_path = os.path.join(specular_dir_path, f"shadow_{idx:04d}.png")
                    if os.path.exists(shadow_img_path): 
                        shadow_img = Image.open(shadow_img_path).convert("L")
                        shadow_img = np.array(shadow_img)
                        shadow_img_tensor = torch.tensor(shadow_img, dtype=torch.float32).cuda()/255.0
                        shadow = shadow_img_tensor
                        shadow += 0.4
                        shadow = torch.clamp(shadow, 0, 1)

                
                    # 2) B 轻提  +6，防止发脏/偏棕
                    
                    # img_np = (render_image.clamp(0,1)
                    #                     .permute(1,2,0)          # CHW→HWC
                    #                     .cpu().numpy()*255)      # →0-255
                    # img_np = img_np.astype(np.uint8)

                    # # 直接 RGB→HSV（因为 render_image 已确认 RGB）
                    # hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.float32)

                    # # ====== 关键三行：锁定橘黄、提亮、提饱和度 ====== #
                    # hsv[:,:,0] = (hsv[:,:,0] + 4) % 180   # -25 单位

                    # # ② 饱和 Saturation：让颜色更饱满一点
                    # hsv[:,:,1] *= 1.09                    # ×1.20

                    # # ③ 亮度 Value：稍微提亮
                    # hsv[:,:,2] *= 1.08                     # ×1.08
                    # hsv = np.clip(hsv, [0,0,0], [179,255,255])
                    # # ============================================= #

                    # out_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

                    # render_image = torch.from_numpy(out_rgb).permute(2,0,1).cuda().float()/255
                    



                    # self shadow
                    self_shadow_path = os.path.join(specular_dir_path, f"self_shadow_{idx:04d}.png")
                    if os.path.exists(self_shadow_path):
                        self_shadow = Image.open(self_shadow_path).convert("L")
                        self_shadow = np.array(self_shadow)
                        self_shadow_tensor = torch.tensor(self_shadow, dtype=torch.float32).cuda()/255.0
                        self_shadow_tensor += 0.577

                        alpha_img_path = os.path.join(specular_dir_path, f"alpha_{idx:04d}.png")
                        alpha_img = Image.open(alpha_img_path).convert("L")
                        alpha_img = np.array(alpha_img)
                        alpha_img_tensor = torch.tensor(alpha_img, dtype=torch.float32).cuda()/255.0
                        alpha_img_tensor = 1 - alpha_img_tensor
                        shadow_img_tensor = self_shadow_tensor + alpha_img_tensor
                        shadow_img_tensor = torch.clamp(shadow_img_tensor, 0, 1)

                        torchvision.utils.save_image(shadow_img_tensor, os.path.join(modelset.model_path, "self_shadow", '{0:05d}'.format(idx) + "_self_shadow.png"))

                        shadow = shadow_img_tensor * shadow
                
                    # shadow = F.gaussian_blur(shadow , kernel_size=9, sigma=3.0)
                    render_image *= shadow

                    torchvision.utils.save_image(shadow, os.path.join(modelset.model_path, "shadow", '{0:05d}'.format(idx) + "_shadow.png"))
                    torchvision.utils.save_image(render_image, os.path.join(modelset.model_path, "valid_shadow", '{0:05d}'.format(idx) + "_shadow.png"))
                    dir_img_path    = os.path.join(specular_dir_path, f"spec_dir_{idx:04d}.png")
                    ind_img_path    = os.path.join(specular_dir_path, f"spec_ind_{idx:04d}.png")
                    shadow_img_path = os.path.join(specular_dir_path, f"shadow_{idx:04d}.png")
                    alpha_img_path  = os.path.join(specular_dir_path, f"alpha_{idx:04d}.png")
                    if os.path.exists(dir_img_path):
                        dir_img = Image.open(dir_img_path).convert("RGB")
                        dir_img_tensor = torchvision.transforms.ToTensor()(dir_img).cuda()
                        print("11111111111111111111111111111111111111111111111111111111111")
                    else:
                        dir_img_tensor = torch.zeros_like(render_image).cuda()
                    if os.path.exists(ind_img_path):
                        ind_img = Image.open(ind_img_path).convert("RGB")
                        ind_img_tensor = torchvision.transforms.ToTensor()(ind_img).cuda()
                        print("11111111111111111111111111111111111111111111111111111111111")
                    else:
                        ind_img_tensor = torch.zeros_like(render_image).cuda()

                    # 全局高光亮度因子
                    k = 0.8                                            

                    # 高光彩色能量
                    spec_rgb = (dir_img_tensor + ind_img_tensor)   # H×W
                    spec_rgb = F.gaussian_blur(spec_rgb, kernel_size=5, sigma=1.0)
                    intensity_factor = 0.4
                    spec_rgb = spec_rgb * intensity_factor
                    mask = spec_rgb.mean(dim=-1, keepdim=True)
                    torchvision.utils.save_image(spec_rgb, os.path.join(modelset.model_path, "only_specular", '{0:05d}'.format(idx) + "_mask.png"))
                    
                    print(render_image.shape)
                    render_image = render_image + spec_rgb
                    # gamma = 1.2  # gamma > 1 会使图像变暗，可以尝试 1.2, 1.5 等
                    # render_image = torch.pow(render_image, gamma)
                    
                    torchvision.utils.save_image(render_image, os.path.join(modelset.model_path, "specular_render", '{0:05d}'.format(idx) + "_specular.png"))

        elif beijing_render:
            from scene.cameras import Camera
            from utils.graphics_utils import compute_centroid_and_orientation

            use_ground = False
            use_specific_camera = False
            updated_light_pos = None
            offset = 0.2
            scene_offset_coefficient = 10
            vehicle_offset_coefficient = 0.5
            transparent_layer = False
            scale_factor = 1.0
            need_shadow = True
            #path
            filepath = r"C:\Users\namew\Desktop\beijing"
            shishan = True
            

            # 把面板里的度数填进来
            euler_deg = [180, 0, 0]
            r = Rot.from_euler('xyz', euler_deg, degrees=True)
            local_forward = np.array([0, 0, 1])
            world_forward = (r.as_matrix() @ local_forward)

            
            parser = ArgumentParser(description="beijing render parameters")
            modelset = ModelParams(parser)
            modelset.data_device = "cpu"
            modelset.sh_degree = 3
            modelset.use_nerual_phasefunc = False
            modelset.cam_opt = False
            modelset.pl_opt = False
            modelset.resolution = 1



            # 寻找所有npy文件
            npy_files = [f for f in os.listdir(filepath) if f.endswith(".npy")]
            vehicle_trajectorys = {}
            render_path = {}
            for npy_file in npy_files:
                npy_split = npy_file.split("_")
                npy_file_label = npy_split[0]
                # 检查 npy 文件开头标识
                if npy_file_label == "camera":
                    camera_trajectorys = np.load(os.path.join(filepath, npy_file),  allow_pickle=True)
                    camera_types = camera_trajectorys[0].keys()
                    for camera_type in camera_types:
                        render_path[camera_type] = os.path.join(filepath, "combine", camera_type)
                        os.makedirs(render_path[camera_type], exist_ok=True)
                    continue
                
                elif npy_file_label == "car":
                    car_index = npy_split[-1].split(".")[0]
                    vehicle_trajectorys[car_index] = np.load(os.path.join(filepath, npy_file),  allow_pickle=True)
                    continue
            print("vehicle_trajectorys.keys()", vehicle_trajectorys.keys())
            print("render_path", render_path)
            # 寻找所有ply文件
            ply_files = [f for f in os.listdir(filepath) if f.endswith(".ply")]
            gaus = {}
            instance_RT = {}
            instance_xyz = {}

            for ply_file in ply_files:
                ply_split = ply_file.split("_")
                ply_file_label = ply_split[0]

                if ply_file_label == "bg":
                    for sh_deg in [3]:
                        try:
                            modelset.sh_degree = sh_deg
                            scene_gaussians = GaussianModel(modelset)
                            scene_gaussians.load_ply(os.path.join(filepath, ply_file))
                            scene_gaussians.object_id = torch.ones_like(scene_gaussians._xyz[:, 0])
                            if shishan:
                                norm = torch.norm(scene_gaussians._xyz, dim=1)
                                log_norm = torch.log10(norm + 1e-8)
                                # plt.hist(log_norm.cpu().numpy(), bins=100)
                                # plt.xlabel("log10(norm)")
                                # plt.ylabel("Count")
                                # plt.title("Point Distance Distribution")
                                # plt.tight_layout()
                                # plt.savefig("norm_hist.png", dpi=300)
                                # print("Histogram saved to norm_hist.png")
                                skybox_mask = log_norm > 2.8
                                temp_sum = skybox_mask.sum()
                                if temp_sum == 0:
                                    print("No skybox particles found.")
                                    print("No skybox particles found.")
                                    print("No skybox particles found.")
                                else:
                                    print("!!!Sky box exists!!!", "skybox particale num", temp_sum.sum())
                                    print("!!!Sky box exists!!!", "skybox particale num", temp_sum.sum())
                                    print("!!!Sky box exists!!!", "skybox particale num", temp_sum.sum())
                                    scene_gaussians.object_id[skybox_mask] = 0
                            else:
                                scene_gaussians.object_id[0:100000] = 0
                            gaus["bg"] = scene_gaussians
                            break
                        except:
                            continue
                    else:
                        raise RuntimeError("你有毒")
                
                elif ply_file.startswith("car"):
                    instance_index = ply_split[-1].split(".")[0]
                    float_instance_index = float(instance_index)
                    for sh_deg in [0, 1, 2, 3]:
                        try:
                            modelset.sh_degree = sh_deg
                            instance_gaussians = GaussianModel(modelset)
                            instance_gaussians.load_ply(os.path.join(filepath, ply_file))
                            object_id = torch.zeros_like(instance_gaussians._xyz[:, 0])
                            object_id[:] = float_instance_index
                            print(object_id)
                            instance_gaussians.object_id = object_id
                            if use_ground:
                                if os.path.exists(os.path.join(filepath, ply_file.replace("car", "car_ground"))):
                                    ground_gaussians = GaussianModel(modelset)
                                    ground_gaussians.load_ply(os.path.join(filepath, ply_file.replace("car", "car_ground")))
                                    object_id_ground = torch.zeros_like(ground_gaussians._xyz[:, 0])
                                    object_id_ground[:] = 100 + float_instance_index
                                    ground_gaussians.object_id = object_id_ground
                                    instance_gaussians.merge_gaussians([ground_gaussians])               
                                else:
                                    end_index = instance_gaussians._xyz.shape[0]
                                    object_id = torch.zeros_like(instance_gaussians._xyz[:, 0])
                                    object_id[:end_index-40000] = float_instance_index
                                    object_id[end_index-40000:] = 100 + float_instance_index
                                    instance_gaussians.object_id = object_id
                                    print("include ground")
                                    print("include ground")
                                instance_xyz[instance_index] = instance_gaussians._xyz.cpu().numpy()
                                centroid, RT = compute_centroid_and_orientation(instance_xyz[instance_index])
                                instance_RT[instance_index] = RT
                            else:
                                instance_xyz[instance_index] = instance_gaussians._xyz.cpu().numpy()
                                centroid, RT = compute_centroid_and_orientation(instance_xyz[instance_index])
                                instance_RT[instance_index] = RT
                            gaus[instance_index] = instance_gaussians
                            break
                        except:
                            continue
                    else:
                        raise RuntimeError("你有毒")
            print(gaus)

            bg_color = [1,1,1]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

            # render
            idx = 0
            for camera_trajectory in camera_trajectorys:
                if idx > -1:
                    for vehicle_index, vehicle_trajectory in vehicle_trajectorys.items():
                        # each vehicle
                        extrinsic_matrix = vehicle_trajectory[idx]['extrinsic_matrix']
                        RT_targrt = np.eye(4)
                        RT_targrt[:3,:3] = extrinsic_matrix[:3,:3]
                        RT_targrt[:3, 3] = -extrinsic_matrix[:3,:3] @ extrinsic_matrix[:3,3]

                        relative_transform = np.dot(RT_targrt, torch.linalg.inv(instance_RT[vehicle_index]))
                        ones = np.ones((instance_xyz[vehicle_index].shape[0], 1))
                        point_homogeneous = np.concatenate([instance_xyz[vehicle_index], ones], axis=1)
                        
                        point_target = (relative_transform @ point_homogeneous.T).T
                        transformed_point = point_target[:, :3] / point_target[:, 3][:, np.newaxis]
                        gaus[vehicle_index]._xyz = torch.tensor(transformed_point).to(gaus[vehicle_index]._xyz)

                    max_sh_degree = 3
                    # 取 value
                    gaus_list = list(gaus.values())  
                    merged_gaussian = gaus_list[0]
                    merged_gaussian.merge_gaussians(gaus_list[1:], max_sh_degree)

                    for camera_type in camera_types:
                        # import pdb;pdb.set_trace()
                        # if idx > 10:
                        extrinsic_matrix = camera_trajectory[camera_type]['extrinsic_matrix']
                        intrinsic_matrix = camera_trajectory[camera_type]['intrinsic_matrix']
                        fx, fy = intrinsic_matrix[0,0] , intrinsic_matrix[1,1]
                        cx, cy = intrinsic_matrix[0,2], intrinsic_matrix[1,2]
                        FovY = focal2fov(fy, int(cy))
                        FovX = focal2fov(fx, int(cx))
                        R = extrinsic_matrix[:3,:3]
                        T = extrinsic_matrix[:3,3]
                        # import pdb;pdb.set_trace()

                        # 假设 tamplete_cam.original_image 是一个形状为 [3, 534, 948] 的 PyTorch 张量
                        cam = Camera(colmap_id=None, R=R, T=T, 
                            FoVx=FovX, FoVy=FovY, cx=None, cy=None,
                            image=None, gt_alpha_mask=None,
                            image_name=None, uid=None,
                            image_width=1920, image_height=1080)
                        

                        
                        render_pkg = shadow_render(cam, merged_gaussian, pipeline, background, updated_light_pos=updated_light_pos, offset=offset, 
                                                scene_offset_coefficient=scene_offset_coefficient, vehicle_offset_coefficient=vehicle_offset_coefficient, 
                                                transparent_layer=transparent_layer, scale_factor=scale_factor, need_shadow=need_shadow,
                                                light_dir=world_forward)

                        # shadow = render_pkg["shadow"].cpu().numpy()  
                        # kernel_size = 11
                        # kernel = np.ones((kernel_size, kernel_size), np.uint8)

                        # # 2. 执行“闭合”操作
                        # shadow = cv2.morphologyEx(shadow, cv2.MORPH_CLOSE, kernel)
                        # shadow = torch.tensor(shadow).cuda()
                        shadow = render_pkg["shadow"]
                        render_image = render_pkg["render"]
                        save_path = os.path.join(filepath, "valid")
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        torchvision.utils.save_image(render_image, os.path.join(save_path, '{0:05d}'.format(idx) + ".png"))
                        torchvision.utils.save_image(shadow, os.path.join(save_path, '{0:05d}'.format(idx) + "_shadow.png"))
                        torchvision.utils.save_image(render_image*shadow, os.path.join(save_path, '{0:05d}'.format(idx) + "_final.png"))
                        torchvision.utils.save_image(render_image, os.path.join(render_path[camera_type], '{0:05d}'.format(idx) + ".png"))
                        torchvision.utils.save_image(shadow, os.path.join(render_path[camera_type], '{0:05d}'.format(idx) + "_shadow.png"))
                        torchvision.utils.save_image(render_image*shadow, os.path.join(render_path[camera_type], '{0:05d}'.format(idx) + "_final.png"))
                        # 清理中间变量cuda缓存
                        torch.cuda.empty_cache()

                    idx += 1


        else:
            # load Gaussians attributes, establish scene
            gaussians = _make_gaussian_model(modelset, opt)
            
            light_direction = torch.tensor([-80, -47, 50], dtype=torch.float32, device="cuda")
            gaussians.light_direction = torch.nn.Parameter(light_direction, requires_grad=True)
            l = [
                {'params': [gaussians.light_direction], 'lr': 0.001, "name": "light_direction"}
            ]
            optimizer = torch.optim.Adam(l, lr=0.001)
            
            # 创建场景实例
            # iteration 为 -1 则加载最新的模型，否则加载指定迭代次数的模型，如果不存在则创建新的模型文件夹，但这里作为渲染，iteration 肯定要的哇
            # valid 渲染验证集，skip_train 不渲染训练集，skip_test 不渲染测试集
            scene = Scene(modelset, gaussians, load_iteration=iteration, shuffle=False, valid=valid, skip_train=skip_train, skip_test=skip_test)
            
            """
            本项目中: scene 中储存的 model 也就是 ply 文件，只存储了每个高斯的属性，！！！但不包括每个高斯的神经网络参数！！！
                因此需要先从 ply 文件中加载高斯模型参数，然后从 _model_path = os.path.join(modelset.model_path, f"chkpnt{iteration}.pth")
                    中提取神经网络参数和一些共用的参数（比如 asg 基函数参数）
            这里的 model_path 是 模型文件夹，包括两个子模型，一个是 point_cloud; 一个是 chkpnt (torch.save(gaussians.capture()) 得到的文件)
            point_cloud 中存储了高斯模型参数，chkpnt 中存储了优化器状态以及神经网络参数和一些共用的参数
            """
            if modelset.use_nerual_phasefunc:
                if iteration == -1:
                    iteration = searchForMaxIteration(os.path.join(modelset.model_path, "point_cloud"))
                _model_path = os.path.join(modelset.model_path, f"chkpnt{iteration}.pth")
                if os.path.exists(_model_path):
                    # initialize ASG and MLP
                    model_params, first_iter, scene_state = _unpack_training_checkpoint(
                        torch.load(_model_path, weights_only=False)
                    )
                    if _is_2dgs_model(modelset):
                        if opt is None:
                            raise RuntimeError("2DGS checkpoint rendering requires OptimizationParams for restore().")
                        gaussians.restore(model_params, opt)
                        if scene_state is not None:
                            scene.restore(scene_state)
                        if getattr(gaussians, "neural_phasefunc", None) is not None:
                            gaussians.neural_phasefunc.eval()
                    else:
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
                else:
                    raise Exception(f"Could not find : {_model_path}")

            bg_color = [1, 1, 1, 1, 0, 0, 0] if modelset.white_background else [0, 0, 0, 0, 0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            
            if valid:
                print("current dataset includes valid")
                render_set(modelset, "valid", scene.loaded_iter, scene.getValidCameras(), 
                        gaussians, pipeline, background, gamma, hdr, write_images=write_images, calculate_fps=calculate_fps, force_save=force_save, 
                        synthesize_video=synthesize_video, shadowmap_render=shadowmap_render)
            
            if not skip_train:
                print("current dataset includes train")
                render_set(modelset, "train", scene.loaded_iter, scene.getTrainCameras(), 
                        gaussians, pipeline, background, gamma, hdr, write_images=write_images, calculate_fps=calculate_fps, force_save=force_save, 
                        synthesize_video=synthesize_video, shadowmap_render=shadowmap_render)

            if not skip_test:
                print("current dataset includes test")
                render_set(modelset, "test", scene.loaded_iter, scene.getTestCameras(), 
                        gaussians, pipeline, background, gamma, hdr, write_images=write_images, calculate_fps=calculate_fps, force_save=force_save, 
                        synthesize_video=synthesize_video, shadowmap_render=shadowmap_render)
            
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    mp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--load_iteration", default=-1, type=int)   # -1 代表加载最新的模型
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--gamma", action="store_true", default=False)
    parser.add_argument("--opt_pose", action="store_true", default=False)
    parser.add_argument("--valid", action="store_true", default=False)
    parser.add_argument("--write_images", action="store_true", default=False)
    parser.add_argument("--calculate_fps", action="store_true", default=False)
    parser.add_argument("--force_save", action="store_true", default=False)
    parser.add_argument("--aaai_render", action="store_true", default=False)
    parser.add_argument("--beijing_render", action="store_true", default=False)
    parser.add_argument("--synthesize_video", action="store_true", default=False)
    parser.add_argument("--shadowmap_render", action="store_true", default=False)
    # 加载训练模型所使用的参数,如果失败则使用默认参数
    try:
        args = get_combined_args(parser)
    except:
        import sys
        args = parser.parse_args(sys.argv[1:])
    args.wang_debug = False

    args_info = f"""
    model_args: {vars(mp.extract(args))}
    load_iteration: {args.load_iteration}
    skip_train: {args.skip_train}
    skip_test: {args.skip_test}
    opt_pose: {args.opt_pose}
    gamma: {args.gamma}
    hdr: {args.hdr}
    valid: {args.valid}
    """
    
    print(Panel(args_info, title="Arguments", expand=False))
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(mp.extract(args), args.load_iteration, pp.extract(args), \
                args.skip_train, args.skip_test, args.opt_pose, args.gamma, args.hdr, args.valid, args.write_images, args.calculate_fps, 
                args.force_save, args.aaai_render, args.beijing_render, args.synthesize_video, args.shadowmap_render, op.extract(args))
