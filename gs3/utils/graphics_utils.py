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
import math
import numpy as np
from typing import NamedTuple
from sklearn.decomposition import PCA
from utils.general_utils import build_scaling_rotation
from utils.general_utils import strip_symmetric


class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array


"""
函数：几何变换点云，将点云从3维坐标系转换到4维齐次坐标系，然后进行变换，再转换回3维坐标系
返回：变换后的点云
"""
def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)


"""
函数：非 C2W 转换到 W2C 的矩阵，用于组合 W2C 的旋转矩阵和平移向量
返回：齐次坐标系下的 W2C 矩阵
"""
def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)


"""
函数：接受相机外参矩阵，[R | T]， 这里的R，T 虽然是基于 W2C 的，但是由于 cuda 中的 glm 库采用列主序 ，旋转矩阵取了转置，这里需要转置恢复
     - 此外，此函数接受 利用 translate 和 scale 对 R，T 进行归一化，先获得 C2W，然后提取平移矩阵进行归一化计算，再转回 W2C
返回：齐次坐标系下的 W2C 矩阵
注：  使用 numpy 实现，不支持可微分
"""
def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

"""
函数：同理世界坐标到相机坐标的变换矩阵,用 torch 实现，可微分
返回：齐次坐标系下的 W2C 矩阵
注：  使用 torch 实现，支持可微分
"""
def getWorld2View2_cu(R, t, translate=torch.tensor([.0, .0, .0]), scale=1.0):
    Rt = torch.zeros((4, 4), dtype=torch.float32).to(device=R.device)
    Rt[:3, :3] = R.T
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    
    C2W = torch.linalg.inv(Rt)
    C2W_translated = C2W.clone()
    C2W_translated[:3, 3] = (C2W[:3, 3] + translate.to(device=R.device)) * scale
    Rt = torch.linalg.inv(C2W_translated)
    return Rt


"""
函数：投影变化的第一步，准备透视投影 ———— 把 3D 坐标转换成 4D 齐次坐标 
返回：投影矩阵（注意和数学中 投影矩阵 概念区分）    
"""
def getProjectionMatrixWithPrincipalPoint(znear, zfar, fx, fy, cx, cy, width, height):
    """
    用相机内参 fx, fy, cx, cy 构造投影矩阵（兼容 OpenGL）
    width, height 是图像大小（像素），用来归一化 cx, cy。
    """
    P = torch.zeros(4, 4)

    # NDC 坐标范围是 [-1, 1]，需要归一化主点
    P[0, 0] = 2.0 * fx / width
    P[1, 1] = 2.0 * fy / height
    P[0, 2] = 1.0 - 2.0 * cx / width   # 注意：OpenGL 中图像中心默认是 (width/2, height/2)
    P[1, 2] = 2.0 * cy / height - 1.0

    # Z buffer 设定
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -zfar * znear / (zfar - znear)
    P[3, 2] = 1.0

    return P


"""
函数：计算正交投影的投影矩阵
返回：投影矩阵
"""
def getOrthographicMatrixFromBounds(xmin, xmax, ymin, ymax, zmin, zmax):
    assert xmax > xmin and ymax > ymin and zmax > zmin, "防止 min/max 写反了"
    return torch.tensor([
        [ 2/(xmax-xmin),       0,                 0, -(xmax+xmin)/(xmax-xmin)],
        [ 0,              2/(ymax-ymin),          0, -(ymax+ymin)/(ymax-ymin)],
        [ 0,                   0,       1/(zmax-zmin), -zmin/(zmax-zmin)],
        [ 0,                   0,                 0,  1]
    ], dtype=xmin.dtype, device=xmin.device)
    
    
"""
函数：计算透视投影的投影矩阵
返回：投影矩阵
"""
def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


"""
函数:焦距和视场角之间的转换
"""
def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))
def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))



"""
函数：给定相机位置，目标位置，上方向，计算相机坐标系到世界坐标系的齐次变换矩阵
返回：齐次变换矩阵
注：  使用 numpy 实现，不支持可微分
"""
def look_at(camera_position, target_position, up_dir, camera_direction = None):

    # 计算相机坐标:
    """
    在 3D 计算机图形学中，相机有自己的局部坐标系，通常规定：
		- X 轴 (Right): 相机的右方向，通常希望与“世界上方向 (up_dir)”尽可能保持正交
		- Y 轴 (Up): 相机的上方向，必须垂直于 X 轴和 Z 轴，确保是右手坐标系
		- Z 轴 (Backward): 相机的朝向方向（但在 OpenGL 里是 -Z）
    """
    # 计算相机方向（从目标指向相机，即相机的 -Z 轴方向）
    camera_direction = camera_direction if camera_direction is not None else camera_position - target_position
    camera_direction = camera_direction / np.linalg.norm(camera_direction)
    print("camera_direction: ", camera_direction)
    print("original camera_direction: ", (camera_position - target_position)/np.linalg.norm(camera_position - target_position))

    # 计算相机的“右方向” (X 轴方向)
    """
    1. up_dir: 通常是 [0,0,1]，代表世界 z 轴的“上”方向
    2. 得到右方向 camera_right: 这个方向垂直于 up_dir 和 camera_direction 组成的平面
    """
    if np.abs(np.dot(up_dir, camera_direction)) > 0.9:
        up_dir = np.array([0, 1, 0])
    camera_right = np.cross(up_dir, camera_direction)
    camera_right = camera_right / np.linalg.norm(camera_right)

    # 计算相机的“上方向” (Y 轴方向)  
    """
    利用 camera_right 和 camera_direction 计算出 camera_up
    """
    camera_up = np.cross(camera_direction, camera_right)
    camera_up = camera_up / np.linalg.norm(camera_up)

    # 组装 Tc2w（相机坐标系到世界坐标系的齐次变换矩阵）:
    # 这个变换矩阵将 世界坐标系中的点变换到相机坐标系。
    """
    1. `camera_right` 作为第一行，表示相机的 X 轴
    2. `camera_up` 作为第二行，表示相机的 Y 轴
    3. `camera_direction` 作为第三行，表示相机的 -Z 轴
    !!! 标准的 变换矩阵 应该是列向量作为坐标系的基向量
    !!! 可能是因为计算得到的是 数组，而数组对行向量赋值更方便，因此这里使用了行向量作为坐标系的基向量
    """
    rotation_transform = np.zeros((4, 4))
    rotation_transform[0, :3] = camera_right
    rotation_transform[1, :3] = camera_up
    rotation_transform[2, :3] = camera_direction
    rotation_transform[-1, -1] = 1.0    # 

    # 计算 Tw2c (世界坐标系到相机坐标系的齐次变换矩阵):
    # 创建一个 4×4 单位矩阵，并存储平移变换
    """
    np.eye(num):
    torch
    """
    translation_transform = np.eye(4)
    """
    假设相机位置是 camera_position = (X, Y, Z)，我们希望：
		- 把整个世界坐标系 向相机的反方向移动  ( -X, -Y, -Z ) 。
		- 这样，相机在新坐标系下位于原点 (0,0,0)，方便计算。
    ！！！但是，切记，别忘了旋转，所以 平移矩阵应该是 T = -R^T C_W, (R 中列向量作为基向量)
    ！！！这里 R 是 行向量作为坐标轴，因此不需要转置
    """
    translation_transform[:3, -1] = -np.array(camera_position)

    # 组装 平移矩阵和旋转矩阵
    # [:3, :3] 为单位矩阵，[:3, 3] 为 相机在世界坐标系下的坐标
    # 正常来说，旋转矩阵应该
    look_at_transform = np.matmul(rotation_transform, translation_transform)

    # 翻转 Y 轴和 Z 轴，使得矩阵符合期望的坐标系统。
    """
    T = -R^T C_W    ->    T = -(R'R)^T C_W    
        # R' 是相对于 R ，再次变换坐标轴的 变换矩阵，即最终的 变换矩阵（最终坐标系的基向量作为矩阵的列向量）
    !!! 有些是先翻转旋转矩阵，因此在计算平移矩阵时，已经考虑了翻转，这里是在最后统一翻转
    """
    look_at_transform[1:3, :] *= -1
    
    # 为了和后面的计算对齐，因此取转置
    """
    !!! 高斯点的坐标一般为 (N,3)，因此坐标以行向量的形式存在，为了和行向量匹配: 
    设列向量 a ，采用 a^T * (Tc2w)^T 而不是最初的 (Tc2w) * a
    """
    look_at_transform=look_at_transform.T
    
    return look_at_transform

"""
函数：给定相机位置，目标位置，上方向，计算相机坐标系到世界坐标系的齐次变换矩阵
返回：齐次变换矩阵
注：  使用 torch 实现，支持可微分
"""
def look_at_cu(camera_position, target_position, up_dir, camera_direction=None):
    
    camera_position = camera_position.float()
    target_position = target_position.float()
    up_dir = up_dir.float()
    
    # Z 轴方向 (相机朝向，OpenGL 用 -Z)
    camera_direction = (camera_direction if camera_direction is not None 
                        else camera_position - target_position)
    camera_direction = camera_direction / torch.norm(camera_direction)

    # X 轴方向 (右方向)
    parallel = torch.abs(torch.dot(up_dir, camera_direction)) > 0.9
    if parallel:
        up_dir = torch.tensor([0., 1., 0.], device=up_dir.device, dtype=up_dir.dtype)
    camera_right = torch.cross(up_dir, camera_direction, dim=0)
    camera_right = camera_right / torch.norm(camera_right)

    # Y 轴方向 (上方向)
    camera_up = torch.cross(camera_direction, camera_right, dim=0)
    camera_up = camera_up / torch.norm(camera_up)

    # 旋转矩阵
    R = torch.stack([camera_right, camera_up, camera_direction], dim=0)  # [3,3]

    # 平移部分
    t = -(R @ camera_position.view(3, 1))

    # 拼成 4x4 齐次矩阵（不用切片赋值）
    one  = camera_position.new_tensor(1.0)
    zero = camera_position.new_tensor(0.0)
    top    = torch.cat([R, t], dim=1)                   # [3,4]
    bottom = torch.stack([zero, zero, zero, one]).view(1, 4)  # [1,4]
    look_at_transform = torch.cat([top, bottom], dim=0) # [4,4], 具有 grad_fn

    # 翻转 Y/Z 轴：左乘一个对角矩阵 diag(1, -1, -1, 1)
    flip = torch.diag(torch.stack([one, -one, -one, one]))
    look_at_transform = flip @ look_at_transform

    # 行向量约定下返回转置
    return look_at_transform.T


"""
函数：计算点云的质心和方向（基于PCA的旋转矩阵绕y轴逆时针旋转）
返回：质心和旋转-平移矩阵
"""
def compute_centroid_and_orientation(point_cloud, rotation_angle_y=0):
    """
    计算点云的质心和方向（基于PCA的旋转矩阵绕y轴逆时针旋转）

    参数:
        point_cloud (np.ndarray): 点云数据 (N, 3)
        rotation_angle_y (float): 绕y轴逆时针旋转的角度（单位：弧度）

    返回:
        centroid (np.ndarray): 质心 (3,)
        RT (np.ndarray): 旋转-平移矩阵 (4, 4)
    """
    # 计算质心
    centroid = np.mean(point_cloud, axis=0)
    
    # 使用PCA计算主方向
    pca = PCA(n_components=3)
    pca.fit(point_cloud - centroid)  # 去中心化
    
    # 获取PCA的旋转矩阵
    rotation = pca.components_.T  # 主成分矩阵转置
    
    # 确保旋转矩阵的行列式为1（右手法则）
    if np.linalg.det(rotation) < 0:
        rotation[:, 2] = -rotation[:, 2]
    
    # 构造绕y轴逆时针旋转的旋转矩阵
    R_y = np.array([
        [np.cos(rotation_angle_y), 0, np.sin(rotation_angle_y)],
        [0, 1, 0],
        [-np.sin(rotation_angle_y), 0, np.cos(rotation_angle_y)]
    ])
    
    # 将PCA旋转矩阵绕y轴逆时针旋转
    combined_rotation = rotation @ R_y  # 注意这里是先应用PCA旋转，再绕y轴旋转
    
    # 构造RT矩阵
    RT = np.eye(4)
    RT[:3, :3] = combined_rotation
    RT[:3, 3] = centroid
    
    return torch.tensor(centroid), torch.tensor(RT)


    
"""
函数：计算正交投影的协方差
返回：协方差
"""
def computeCov2D_ortho_python(xyz, image_width, image_height, light_orthoproj_matrix, fx_far, fy_far, 
                              scaling, rotation, world_view_transform_light, scaling_modifier,
                              full_ortho_proj_transform_light):
        
        
        xyz_4 = torch.cat([xyz, torch.ones_like(xyz[:, 0:1])], dim=1)
        jacobian = torch.zeros((xyz_4.shape[0], 3, 3), dtype=xyz_4.dtype, device=xyz_4.device)
        jacobian[:, 0, 0] = fx_far / xyz_4[:, 2]
        jacobian[:, 0, 1] = 0
        jacobian[:, 0, 2] = 0
        jacobian[:, 1, 0] = 0
        jacobian[:, 1, 1] =  fy_far / xyz_4[:, 2]
        jacobian[:, 1, 2] = 0
        jacobian[:, 2, 0] = -fx_far * xyz_4[:, 0] / (xyz_4[:, 2] * xyz_4[:, 2])
        jacobian[:, 2, 1] = -fy_far * xyz_4[:, 1] / (xyz_4[:, 2] * xyz_4[:, 2])
        jacobian[:, 2, 2] = fy_far / xyz_4[:, 2]

        sx = image_width 
        sy = image_height

        xyz_proj = xyz_4 @ light_orthoproj_matrix
        xyz_ndc = xyz_proj[:, :3] / xyz_proj[:, 3:4]
        
        
        S = torch.tensor([
            [sx, 0, 0],
            [0, sy, 0],
            [0, 0, 1]], dtype=torch.float32, device='cuda')
        
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        perspective_cov3D_matrix = world_view_transform_light[:3, :3] @ jacobian
        print("perspective_cov3D_matrix.shape: ", perspective_cov3D_matrix.shape)
        # world_view_transform_light 转置后的矩阵
        full_ortho_proj_transform_light33 = full_ortho_proj_transform_light[:3, :3] @ S
        # persp_cov3D = perspective_cov3D_matrix.transpose(-1, -2) @ actual_covariance @ perspective_cov3D_matrix
        # print("persp_covariance: ", persp_cov3D )
        ortho_cov3D = full_ortho_proj_transform_light33.transpose(-1, -2) @ actual_covariance @ full_ortho_proj_transform_light33
        ortho_cov3D = strip_symmetric(ortho_cov3D)
        print("ortho_covariance: ", ortho_cov3D )
        print("ortho_cov3D: ", ortho_cov3D)
        
        d3_ortho_cov = torch.zeros((xyz.shape[0], 3), dtype=torch.float32, device='cuda')
        d3_ortho_cov[:, 0] = ortho_cov3D[:, 0] + 0.3
        d3_ortho_cov[:, 1] = ortho_cov3D[:, 1]
        d3_ortho_cov[:, 2] = ortho_cov3D[:, 3] + 0.3

        return d3_ortho_cov
    
    
"""
函数：将 euler 角度转换为旋转矩阵
返回：旋转矩阵
"""
def euler_to_matrix(euler_deg):
    """
    euler_deg: tensor shape (3,), degrees, requires_grad can be True
    returns: 3x3 rotation matrix (Rz @ Ry @ Rx), keeps autograd graph
    """
    # convert to radians
    euler_rad = torch.deg2rad(euler_deg)

    # unpack into scalar tensors (each is a 0-dim tensor, keeps grad)
    rx, ry, rz = euler_rad
    cx, cy, cz = torch.cos(rx), torch.cos(ry), torch.cos(rz)
    sx, sy, sz = torch.sin(rx), torch.sin(ry), torch.sin(rz)

    # create device/dtype-matching constants quickly
    one  = euler_deg.new_tensor(1.0)
    zero = euler_deg.new_tensor(0.0)

    # build matrices using stack (do NOT use torch.tensor([...cx...]))
    Rx = torch.stack([
        torch.stack([ one,  zero,  zero]),
        torch.stack([ zero,  cx,   -sx]),
        torch.stack([ zero,  sx,    cx])
    ], dim=0)

    Ry = torch.stack([
        torch.stack([  cy,   zero,  sy]),
        torch.stack([ zero,  one,   zero]),
        torch.stack([ -sy,   zero,  cy])
    ], dim=0)

    Rz = torch.stack([
        torch.stack([  cz,  -sz,   zero]),
        torch.stack([  sz,   cz,   zero]),
        torch.stack([ zero, zero,  one])
    ], dim=0)

    R = Rz @ Ry @ Rx
    return R


