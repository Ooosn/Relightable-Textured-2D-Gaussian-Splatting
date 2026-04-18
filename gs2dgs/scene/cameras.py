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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getWorld2View2_cu, getProjectionMatrixWithPrincipalPoint
from utils.lie_groups import exp_map_SO3xR3, exp_map_SE3

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, cx, cy, image, gt_alpha_mask,
                 image_name, uid, pl_pos = None, pl_intensity = None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 is_hdr=False, image_path=None,
                 image_width=None, image_height=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.cx = cx
        self.cy = cy
        self.image_name = image_name
        self.image_path = image_path
        self.cam_pose_adj = torch.nn.Parameter(torch.zeros((1, 6), requires_grad=True).cuda())
        self.pl_adj = torch.nn.Parameter(torch.zeros((1, 3), requires_grad=True).cuda())

        self.R_cu = torch.tensor(R, dtype=torch.float32).cuda()
        self.T_cu = torch.tensor(T, dtype=torch.float32).cuda()

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        if pl_pos is not None:
            self.pl_pos = torch.tensor(pl_pos, dtype=torch.float32).unsqueeze(0).cuda()
            self.pl_pos_init = torch.tensor(pl_pos, dtype=torch.float32).unsqueeze(0).cuda()
        else:
            self.pl_pos = None
            self.pl_pos_init = None
        if image is not None:
            if is_hdr:
                self.original_image = image.to(self.data_device)
            else:
                self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1]
        else:
            assert image_width is not None and image_height is not None, "Because image is None, image_width and image_height must be provided"
            self.image_width = image_width
            self.image_height = image_height
            self.original_image = None

        #if gt_alpha_mask is not None:
            #self.original_image *= gt_alpha_mask.to(self.data_device)
        #else:
            #self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale
        self.trans_cu = torch.tensor(trans, dtype=torch.float32).cuda()
        self.scale_cu = torch.tensor(scale, dtype=torch.float32).cuda()
        # self.cx = self.image_width / 2.0
        # self.cy = self.image_height / 2.0
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        fx = self.image_width / (2.0 * np.tan(self.FoVx / 2.0))
        fy = self.image_height / (2.0 * np.tan(self.FoVy / 2.0))
        self.projection_matrix = getProjectionMatrixWithPrincipalPoint(
            znear=self.znear,
            zfar=self.zfar,
            fx=fx,
            fy=fy,
            cx=self.cx,
            cy=self.cy,
            width=self.image_width,
            height=self.image_height,
        ).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
    
    # update opt cam and light
    # 根据 cam_pose_adj 可训练参数 优化相机外参
    def update(self, mode = "SO3xR3"):
        cam_opt_mat = None
        if mode == "SO3xR3":
            cam_opt_mat = exp_map_SO3xR3(self.cam_pose_adj)
        elif mode == "SE3":
            cam_opt_mat = exp_map_SE3(self.cam_pose_adj)
        
        if cam_opt_mat is not None:
            """
            dR 是 cam_pose_adj 中的旋转矩阵，因此和 R_cu 相乘 产生新的旋转矩阵
            同理，T 是平移矩阵，他要先和 dR 相乘，进行旋转，再与 dT 相加
            """
            dR = cam_opt_mat[0, :3, :3]
            dt = cam_opt_mat[0, :3, 3:]
            R = self.R_cu.matmul(dR.T)
            T = dt.reshape(3) + dR.matmul(self.T_cu)

        self.world_view_transform = (getWorld2View2_cu(R, T, self.trans_cu, self.scale_cu)).transpose(0, 1)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
        self.pl_pos = self.pl_pos_init + self.pl_adj
    
    # get opt cam and light
    def get(self, mode = "SO3xR3"):
        cam_opt_mat = None
        if mode == "SO3xR3":
            cam_opt_mat = exp_map_SO3xR3(self.cam_pose_adj)
        elif mode == "SE3":
            cam_opt_mat = exp_map_SE3(self.cam_pose_adj)
        
        if cam_opt_mat is not None:
            dR = cam_opt_mat[0, :3, :3]
            dt = cam_opt_mat[0, :3, 3:]
            R = self.R_cu.matmul(dR.T)
            T = dt.reshape(3) + dR.matmul(self.T_cu)

        self.pl_pos = self.pl_pos_init + self.pl_adj
        
        return R.detach().cpu().numpy(), T.detach().cpu().numpy(), self.pl_pos.detach().cpu().numpy()

    # camera opt regularization
    def get_loss(self):
        return self.cam_pose_adj[:, :3].norm(dim=-1).mean() * 0.01 + \
        self.cam_pose_adj[:, 3:].norm(dim=-1).mean() * 0.001 + \
        self.pl_adj.norm(dim=-1).mean() * 0.01
            

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

