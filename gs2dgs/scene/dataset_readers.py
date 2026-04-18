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
os.environ['OPENCV_IO_ENABLE_OPENEXR']="1"
import sys
from PIL import Image
from typing import NamedTuple
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from utils.graphics_utils import BasicPointCloud
import cv2 as cv
from tqdm import tqdm


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    cx: float
    cy: float
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    pl_intensity: np.array
    pl_pos: np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    valid_cameras: list
    nerf_normalization: dict
    ply_path: str

"""
函数: 接受一组相机信息，进行归一化处理: 1. 处理场景的包围球（bounding sphere）的中心以及半径
                                     2. 处理相机坐标系标准化（使相机坐标系相机放置在原点，朝向 Z 轴正方向）的平移矩阵
返回: 场景的包围球（bounding sphere）的半径 以及 坐标系标准化平移矩阵
"""
def getNerfppNorm(cam_info):

    """
    函数：通过相机信息，计算场景的几何中心以及包围球半径
    返回：返回场景的几何中心以及包围球半径
    """
    def get_center_and_diag(cam_centers):
        # 将数组中的每个元素，都转成 np.array，然后按 horizontal stack 拼接在一起，也就是按列拼接，每一列代表一个相机中心
        cam_centers = np.hstack(cam_centers)
        # 按列计算均值，也就是场景中心，keepdims=True 保留维度，保留之前每个数据拥有的的维度，这里是一列，其他地方也可能是一个矩阵代表一列，因为这里一列就是一个数据嘛
        # 因为后面还要计算 欧几里得距离，需要用到一维向量的广播性质，因此保持维度
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        # 是计算 每个相机中心点到均值中心 center 的欧几里得距离 dist：distance
        """
        np.linalg.norm: 计算的是 欧几里得距离（L2 范数）。
        """
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        # 取最大的 distance 作为半径
        diagonal = np.max(dist)
        # 返回 center.flatten()，因为坐标向量的格式一般用一维数组表示，因此需要 flatten()
        return center.flatten(), diagonal
    
    cam_centers = []
    for cam in cam_info:
        # 遍历所有相机信息，获取 世界坐标系到相机坐标系的变换矩阵
        # blender 数据可以直接获得相机的外参，也就是旋转矩阵和平移向量，现实生活的一般相机往往是没这些信息的，需要 SfM 、 SLAM 等等来标定
        # 相机外参一般是 3*4，[R | T]，R 是旋转矩阵，T 是平移矩阵，这里就是直接获取相机的外参，将其从 3*4 扩展为 4*4 用于后续的齐次坐标计算
        W2C = getWorld2View2(cam.R, cam.T)
        # 对世界坐标系到相机坐标系的变换矩阵求逆，即为相机坐标系到世界坐标系的变换矩阵
        C2W = np.linalg.inv(W2C)
        # 只提取平移部分，不提取旋转部分，这里是齐次坐标系，0：4*0：4，旋转矩阵为 0：3*0：3，平移矩阵为 0：3*3：4，即最后一列前三个元素，右下角元素（最后一列第四个元素）为 1
        cam_centers.append(C2W[:3, 3:4])

    # 通过每个相机的平移矩阵计算场景的几何中心以及包围球半径
    center, diagonal = get_center_and_diag(cam_centers)
    # 为了保证包围球半径大于场景的几何中心到相机中心的最大距离，这里取 1.1 倍
    radius = diagonal * 1.1

    # new_point = old_point + translate，因此translate 这里就是场景坐标系归一化的平移变换，将场景中心移动至原点
    translate = -center

    return {"translate": translate, "radius": radius}

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readCamerasFromTransforms(path, transformsfile, white_background, view_num, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        cx, cy = None, None
        if "camera_angle_x" in contents.keys():
            fovx = contents["camera_angle_x"]
        else:
            # intrinsics of real capture
            intrinsics = contents['camera_intrinsics']
            cx = intrinsics[0]
            cy = intrinsics[1]
            fx = intrinsics[2]
            fy = intrinsics[3]


        frames = contents["frames"]
        for idx, frame in tqdm(list(enumerate(frames))):

            if view_num > 0 and idx >= view_num:
                break

            if "img_path" in frame.keys():
                cam_name = frame["img_path"]
            else:
                if frame["file_path"][-3:] != "png":
                    cam_name = os.path.join(path, frame["file_path"] + extension)
                else:
                    cam_name = os.path.join(path, frame["file_path"])
            # load per cam intrinsic (for lightstage)
            if "camera_intrinsics" in frame.keys():
                intrinsics = frame['camera_intrinsics']
                cx = intrinsics[0]
                cy = intrinsics[1]
                fx = intrinsics[2]
                fy = intrinsics[3]

            if "R_opt" in frame.keys():
                # load opt pose
                R = np.asarray(frame["R_opt"])
                T = np.asarray(frame["T_opt"])
            else:
                # NeRF 'transform_matrix' is a camera-to-world transform
                # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
                """
                OpenGL/Blender 中，c2w 是一个 camera-to-world 变换矩阵，它的列向量表示新坐标轴在世界坐标系中的方向:
                	- 第一列 c2w[:,0] 代表 相机 x 轴在世界坐标系中的方向。
                	- 第二列 c2w[:,1] 代表 相机 y 轴在世界坐标系中的方向。
                	- 第三列 c2w[:,2] 代表 相机 z 轴在世界坐标系中的方向。
                	- 第四列 c2w[:,3] 代表 相机在世界坐标系中的位置 T = C_W。
                因此 T 在 c2w初 中的取值和坐标轴的旋转无关
                """
                c2w = np.array(frame["transform_matrix"])
                c2w[:3, 1:3] *= -1

                # get the world-to-camera transform and set R, T
                """
                w2c = c2w^-1:
                    - [:3,:3] 中行向量作为坐标轴
                    - T = -R^T C_W
                """
                w2c = np.linalg.inv(c2w)

                # 为了和后面的计算对齐，因此取转置
                """
                !!! 高斯点的坐标一般为 (N,3)，因此坐标以行向量的形式存在，为了和行向量匹配: 
                转置后 列向量作为基向量，坐标采用行向量，因此右乘 R，可将世界坐标系下的点变换到相机坐标系下，得到的也是 行向量坐标
                即: P' = R P    ->    (P')^T = P^T R^T
                """
                R = np.transpose(w2c[:3,:3])  
                # 取出 1D数组，# 如果需要向量结构，自行 unsqueeze()
                T = w2c[:3, 3]

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
            
            # 这里只是读取图片，并进行初步处理
            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            if extension == ".png":
                image = Image.open(image_path)
                im_data = np.array(image.convert("RGBA"))
                norm_data = im_data / 255.0
            elif extension == ".exr":
                image = cv.imread(image_path, cv.IMREAD_UNCHANGED | cv.IMREAD_ANYDEPTH | cv.IMREAD_ANYCOLOR)
                norm_data = cv.cvtColor(image, cv.COLOR_BGRA2RGBA)
            else:
                raise Exception(f"Could not support : {extension}")
            
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            
            if extension == ".png":
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
                if "camera_angle_x" in contents.keys():
                    fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
                    FovY = fovy 
                    FovX = fovx
                else:
                    FovY = focal2fov(fy, image.size[1])
                    FovX = focal2fov(fx, image.size[0])

                if cx == None:
                    cx = image.size[0] / 2.
                if cy == None:
                    cy = image.size[1] / 2.

                cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, cx=cx, cy=cy, image=image,
                                image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1],
                                pl_intensity=None if "pl_intensity" not in frame.keys() else frame["pl_intensity"],
                                pl_pos=None if frame["pl_pos"] is None else frame["pl_pos"]))
            else:
                image = arr.astype(np.float32)
                if "camera_angle_x" in contents.keys():
                    fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[0])
                    FovY = fovy 
                    FovX = fovx
                else:
                    FovY = focal2fov(fy, image.shape[0])
                    FovX = focal2fov(fx, image.shape[1])

                if cx == None:
                    cx = image.shape[1] / 2.
                if cy == None:
                    cy = image.shape[0] / 2.

                cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, cx=cx, cy=cy, image=image,
                                image_path=image_path, image_name=image_name, width=image.shape[1], height=image.shape[0],
                                pl_intensity=None if "pl_intensity" not in frame.keys() else frame["pl_intensity"],
                                pl_pos=None if frame["pl_pos"] is None else frame["pl_pos"]))

            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, view_num, load_num, valid=False, skip_train=False, skip_test=False, extension=".png"):
    print("--------------------------------")
    print("|      json path: ", path)
    print("--------------------------------")
    if valid:
        # Only used for visualization, we use 400 frames for visualization
        # Here actually don't provide the 400 real photos from different but continuous perspective
        print("Reading Valid Transforms")
        load_num = 400 if load_num is None else load_num
        print("load_num: ", load_num)
        valid_cam_infos = readCamerasFromTransforms(path, "transforms_valid.json", white_background, load_num, extension)

        train_cam_infos = []
        test_cam_infos = []

        if not skip_train:
            print("Reading Training Transforms")
            train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, view_num, extension)
        if not skip_test:
            print("Reading Test Transforms")
            test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, view_num, extension)
    else:
        print("Reading Training Transforms")
        train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, view_num, extension)
        if not skip_test:
            print("Reading Test Transforms")
            test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, view_num, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    # 如果处于渲染可视化阶段，则不需要计算包围球
    if valid:
        nerf_normalization = []
    else:
        nerf_normalization = getNerfppNorm(train_cam_infos)
        valid_cam_infos = []

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 1.0 - 0.5
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           valid_cameras=valid_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Blender" : readNerfSyntheticInfo
}
