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

from scene.cameras import Camera
import numpy as np
import torch
from utils.general_utils import PILtoTorch, ExrtoTorch
from utils.graphics_utils import fov2focal

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    """
    Load and process camera information and image data.
    Args:
        args: An object containing various arguments and settings, including:
            - hdr (bool): Flag indicating if HDR images are used.
            - resolution (int): Resolution scaling factor.
            - data_device (str): Device to load data onto.
        id (int): Unique identifier for the camera.
        cam_info: An object containing camera information, including:
            - height (int): Original height of the image.
            - width (int): Original width of the image.
            - image (PIL.Image or similar): The image data.
            - cx (float): Camera principal point x-coordinate.
            - cy (float): Camera principal point y-coordinate.
            - uid (int): Unique identifier for the camera.
            - R (np.ndarray): Rotation matrix, R_W2C
            - T (np.ndarray): Translation vector, T_W2C = -R * T_world
            - FovX (float): Field of view in the x direction.
            - FovY (float): Field of view in the y direction.
            - pl_pos (np.ndarray): Position of the point light.
            - pl_intensity (np.ndarray): Intensity of the point light.
            - image_name (str): Name of the image file.
            - image_path (str): Path to the image file.
        resolution_scale (float): Scaling factor for the resolution.
    Returns:
        Camera: A Camera object with the following attributes:
            - colmap_id (int): Unique identifier for the camera.
            - R (np.ndarray): Rotation matrix.
            - T (np.ndarray): Translation vector.
            - FoVx (float): Field of view in the x direction.
            - FoVy (float): Field of view in the y direction.
            - cx (float): Camera principal point x-coordinate.
            - cy (float): Camera principal point y-coordinate.
            - image (torch.Tensor): Processed image tensor.
            - gt_alpha_mask (torch.Tensor or None): Ground truth alpha mask if available.
            - pl_pos (np.ndarray): Position of the point light.
            - pl_intensity (np.ndarray): Intensity of the point light.
            - image_name (str): Name of the image file.
            - uid (int): Unique identifier for the camera.
            - data_device (str): Device to load data onto.
            - is_hdr (bool): Flag indicating if HDR images are used.
            - image_path (str): Path to the image file.
    """
    if args.hdr:
        orig_h = cam_info.height
        orig_w = cam_info.width
    else:
        orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
        cam_cx = cam_info.cx / (resolution_scale * args.resolution)
        cam_cy = cam_info.cy / (resolution_scale * args.resolution)
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))
        cam_cx = cam_info.cx / scale
        cam_cy = cam_info.cy / scale

    # cam_info.image 中的 图片 是 pil 或 cv2 读取的，并且只进行了初步处理，因此需要转换成 torch 处理文件，并进行通道转换
    if args.hdr:
        resized_image_rgb = ExrtoTorch(cam_info.image, resolution)
    else:
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    use_hgs_finetune = getattr(args, "use_hgs_finetune", False)
    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, cx=cam_cx, cy=cam_cy,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  pl_pos=cam_info.pl_pos, pl_intensity=cam_info.pl_intensity,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device, 
                  is_hdr=args.hdr, image_path=cam_info.image_path, use_hgs_finetune = use_hgs_finetune)

# Load cameras from a list of CameraInfo objects and return a list of Camera objects
# 根据 resolution_scale 和 args 从 CameraInfo 对象列表中加载相机和对应的图片进行修改，返回 Camera 对象列表
def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, cam_info in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, cam_info, resolution_scale))
    
    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
