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
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.lie_groups import exp_map_SO3xR3


def _getWorld2View2_cu(R, t, translate, scale):
    """CUDA-tensor version of getWorld2View2."""
    Rt = torch.zeros((4, 4), dtype=torch.float32, device=R.device)
    Rt[:3, :3] = R.T
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    C2W = torch.linalg.inv(Rt)
    C2W_translated = C2W.clone()
    C2W_translated[:3, 3] = (C2W[:3, 3] + translate.to(R.device)) * scale
    return torch.linalg.inv(C2W_translated)


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid, pl_pos=None, pl_intensity=None,
                 image_path=None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.image_path = image_path
        self.pl_intensity = torch.tensor(pl_intensity, device=data_device).float() if pl_intensity is not None else None

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        self.width = self.image_width
        self.height = self.image_height

        if gt_alpha_mask is not None:
            self.gt_alpha_mask = gt_alpha_mask.to(self.data_device)
        else:
            self.gt_alpha_mask = None
        
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        # Cached CUDA tensors for pose optimization
        self.R_cu = torch.tensor(R, dtype=torch.float32).cuda()
        self.T_cu = torch.tensor(T, dtype=torch.float32).cuda()
        self.trans_cu = torch.tensor(trans, dtype=torch.float32).cuda()
        self.scale_cu = torch.tensor(scale, dtype=torch.float32).cuda()

        # Learnable pose & light adjustments (optimized by scene optimizer)
        self.cam_pose_adj = nn.Parameter(torch.zeros((1, 6), dtype=torch.float32).cuda())
        self.pl_adj       = nn.Parameter(torch.zeros((1, 3), dtype=torch.float32).cuda())

        # pl_pos: both a fixed init copy and a mutable current value
        if pl_pos is not None:
            self.pl_pos_init = torch.tensor(pl_pos, dtype=torch.float32).unsqueeze(0).cuda()
            self.pl_pos      = self.pl_pos_init.clone()
        else:
            self.pl_pos_init = None
            self.pl_pos      = None

        self.projection_matrix = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
        ).transpose(0, 1).cuda()

        # Initialise view transform (no adj yet)
        self.world_view_transform = torch.tensor(
            getWorld2View2(R, T, trans, scale)
        ).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def update(self):
        """Apply cam_pose_adj (SO3xR3) and pl_adj to refresh view transforms."""
        adj = exp_map_SO3xR3(self.cam_pose_adj)   # [1, 3, 4]
        dR = adj[0, :3, :3]
        dt = adj[0, :3, 3]
        R = self.R_cu.matmul(dR.T)
        T = dt + dR.matmul(self.T_cu)

        self.world_view_transform = _getWorld2View2_cu(
            R, T, self.trans_cu, self.scale_cu
        ).transpose(0, 1)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        if self.pl_pos_init is not None:
            self.pl_pos = self.pl_pos_init + self.pl_adj

    def get_loss(self):
        """L2 regularization on pose and light adjustments."""
        return (self.cam_pose_adj[:, :3].norm(dim=-1).mean() * 0.01
                + self.cam_pose_adj[:, 3:].norm(dim=-1).mean() * 0.001
                + self.pl_adj.norm(dim=-1).mean() * 0.01)

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
