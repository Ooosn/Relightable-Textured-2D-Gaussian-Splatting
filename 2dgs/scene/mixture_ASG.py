import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general_utils import build_rotation


def _dot(x, y, keepdim=True):
    return torch.sum(x * y, dim=-1, keepdim=keepdim)


class Mixture_of_ASG(nn.Module):
    def __init__(self, basis_asg_num=8, asg_channel_num=1):
        super().__init__()
        self.basis_asg_num = basis_asg_num
        self.asg_channel_num = asg_channel_num
        self.const = math.sqrt(2.0) * math.pow(math.pi, 2.0 / 3.0)
        self.F0 = 0.04
        self.softplus = nn.Softplus()

        self.register_buffer(
            "asg_sigma_ratio",
            self.softplus(torch.linspace(-2.0, 0.0, steps=basis_asg_num, device="cuda")).unsqueeze(-1),
        )
        self.asg_sigma = nn.Parameter(torch.zeros((basis_asg_num, asg_channel_num), dtype=torch.float32, device="cuda"))
        asg_scales = torch.ones((basis_asg_num, 2, asg_channel_num), dtype=torch.float32, device="cuda") * -2.1972
        asg_scales[:, 0, :] *= 0.5
        self.asg_scales = nn.Parameter(asg_scales)
        asg_rotation = torch.zeros((basis_asg_num, 4), dtype=torch.float32, device="cuda")
        asg_rotation[:, 0] = 1.0
        self.asg_rotation = nn.Parameter(asg_rotation)

    @property
    def get_asg_lam_miu(self):
        return torch.sigmoid(self.asg_scales) * 10.0

    @property
    def get_asg_sigma(self):
        return torch.sigmoid(self.asg_sigma) * self.asg_sigma_ratio

    @property
    def get_asg_axis(self):
        return build_rotation(self.asg_rotation).reshape(-1, 3, 3)

    def forward(self, wi, wo, alpha, asg_scales, asg_axises):
        half = F.normalize(wo + wi, p=2, dim=-1)
        fresnel = self.F0 + (1.0 - self.F0) * torch.clamp(1.0 - _dot(wi, half), 0.0, 1.0).pow(5)
        fresnel = fresnel.unsqueeze(-1)

        half = half.unsqueeze(1).expand(-1, self.basis_asg_num, -1)
        asg_x = asg_axises[:, :, 0].unsqueeze(0)
        asg_y = asg_axises[:, :, 1].unsqueeze(0)
        asg_z = asg_axises[:, :, 2].unsqueeze(0)
        lam = asg_scales[:, 0, :].unsqueeze(0)
        miu = asg_scales[:, 1, :].unsqueeze(0)
        sigma = self.get_asg_sigma.unsqueeze(0)

        s = F.normalize(half - _dot(half, asg_z) * asg_z, p=2, dim=-1)
        aniso_ratio = torch.sqrt(
            (_dot(s, asg_x, keepdim=False).unsqueeze(-1) / lam).pow(2)
            + (_dot(s, asg_y, keepdim=False).unsqueeze(-1) / miu).pow(2)
        )
        cos_theta = _dot(half, asg_z, keepdim=False).unsqueeze(-1).clamp(-1 + 1e-6, 1 - 1e-6)
        asg_res = torch.exp(-0.5 * (torch.arccos(cos_theta) * aniso_ratio / sigma).pow(2))
        asg_res = asg_res / (self.const * sigma)
        mm_asg_res = torch.sum(alpha * asg_res, dim=1, keepdim=True)
        return (mm_asg_res * fresnel).squeeze(1)
