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

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from scene.mixture_ASG import Mixture_of_ASG
from scene.neural_phase_function import Neural_phase
from utils.general_utils import build_rotation, build_scaling_rotation, get_expon_lr_func, inverse_sigmoid, strip_symmetric
from utils.graphics_utils import BasicPointCloud
from utils.sh_utils import RGB2SH, SH2RGB
from utils.system_utils import mkdir_p


class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            rs = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0, 2, 1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:, :3, :3] = rs
            trans[:, 3, :3] = center
            trans[:, 3, 3] = 1
            return trans

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        self.material_activation = torch.nn.Softplus()

    def __init__(
        self,
        sh_degree: int,
        use_textures: bool = False,
        texture_resolution: int = 4,
        use_mbrdf: bool = False,
        basis_asg_num: int = 8,
        hidden_feature_size: int = 32,
        hidden_feature_layers: int = 3,
        phase_frequency: int = 4,
        neural_material_size: int = 6,
        asg_channel_num: int = 1,
        asg_mlp: bool = False,
        asg_alpha_num: int = 1,
    ):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.use_textures = use_textures
        self.texture_resolution = texture_resolution
        self.use_mbrdf = use_mbrdf
        self.basis_asg_num = basis_asg_num
        self.hidden_feature_size = hidden_feature_size
        self.hidden_feature_layers = hidden_feature_layers
        self.phase_frequency = phase_frequency
        self.neural_material_size = neural_material_size
        self.asg_channel_num = asg_channel_num
        self.asg_mlp = asg_mlp
        self.asg_alpha_num = asg_alpha_num

        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._tex_color = torch.empty(0)
        self._tex_alpha = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)

        self.kd = torch.empty(0)
        self.ks = torch.empty(0)
        self.alpha_asg = torch.empty(0)
        self.local_q = torch.empty(0)
        self.neural_material = torch.empty(0)
        self.asg_func = None
        self.neural_phasefunc = None

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._tex_color,
            self._tex_alpha,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.use_textures,
            self.texture_resolution,
            self.use_mbrdf,
            self.basis_asg_num,
            self.hidden_feature_size,
            self.hidden_feature_layers,
            self.phase_frequency,
            self.neural_material_size,
            self.asg_channel_num,
            self.asg_mlp,
            self.asg_alpha_num,
            self.kd,
            self.ks,
            self.alpha_asg,
            self.local_q,
            self.neural_material,
            None if self.asg_func is None else self.asg_func.state_dict(),
            None if self.neural_phasefunc is None else self.neural_phasefunc.state_dict(),
        )

    def restore(self, model_args, training_args):
        if len(model_args) == 12:
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
            ) = model_args
            self._tex_color = torch.empty(0, device="cuda")
            self._tex_alpha = torch.empty(0, device="cuda")
        elif len(model_args) == 16:
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._tex_color,
                self._tex_alpha,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
                self.use_textures,
                self.texture_resolution,
            ) = model_args
        else:
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._tex_color,
                self._tex_alpha,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
                self.use_textures,
                self.texture_resolution,
                self.use_mbrdf,
                self.basis_asg_num,
                self.hidden_feature_size,
                self.hidden_feature_layers,
                self.phase_frequency,
                self.neural_material_size,
                self.asg_channel_num,
                self.asg_mlp,
                self.asg_alpha_num,
                self.kd,
                self.ks,
                self.alpha_asg,
                self.local_q,
                self.neural_material,
                asg_state,
                phase_state,
            ) = model_args
            if self.use_mbrdf:
                self._initialize_mbrdf_modules()
                if asg_state is not None:
                    self.asg_func.load_state_dict(asg_state)
                if phase_state is not None:
                    self.neural_phasefunc.load_state_dict(phase_state)

        self.training_setup(training_args)
        if self.use_textures and self._tex_color.numel() == 0:
            self.initialize_texture_state()
        if self.use_mbrdf and not isinstance(self.kd, nn.Parameter):
            self.initialize_mbrdf_state()
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        try:
            self.optimizer.load_state_dict(opt_dict)
        except ValueError:
            pass

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_texture_color(self):
        return torch.sigmoid(self._tex_color)

    @property
    def get_texture_alpha(self):
        return torch.sigmoid(self._tex_alpha)

    @property
    def get_kd(self):
        return self.material_activation(self.kd)

    @property
    def get_ks(self):
        return self.material_activation(self.ks)

    @property
    def get_alpha_asg(self):
        return self.material_activation(self.alpha_asg)

    @property
    def get_local_axis(self):
        return build_rotation(self.local_q)

    @property
    def get_neural_material(self):
        return self.neural_material

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def _initialize_mbrdf_modules(self):
        self.asg_func = Mixture_of_ASG(self.basis_asg_num, self.asg_channel_num)
        self.neural_phasefunc = Neural_phase(
            hidden_feature_size=self.hidden_feature_size,
            hidden_feature_layers=self.hidden_feature_layers,
            frequency=self.phase_frequency,
            neural_material_size=self.neural_material_size,
            asg_mlp=self.asg_mlp,
        ).to(device="cuda")

    def initialize_mbrdf_state(self):
        if not self.use_mbrdf:
            return
        num_points = self._xyz.shape[0]
        if num_points == 0:
            return
        if self.asg_func is None or self.neural_phasefunc is None:
            self._initialize_mbrdf_modules()
        kd = torch.ones((num_points, 3), dtype=torch.float32, device="cuda") * 0.5
        ks = torch.ones((num_points, 3), dtype=torch.float32, device="cuda") * 0.5
        alpha_asg = torch.zeros((num_points, self.basis_asg_num, self.asg_alpha_num), dtype=torch.float32, device="cuda")
        local_q = torch.zeros((num_points, 4), dtype=torch.float32, device="cuda")
        local_q[:, 0] = 1.0
        neural_material = torch.ones((num_points, self.neural_material_size), dtype=torch.float32, device="cuda")
        self.kd = nn.Parameter(kd.requires_grad_(True))
        self.ks = nn.Parameter(ks.requires_grad_(True))
        self.alpha_asg = nn.Parameter(alpha_asg.requires_grad_(True))
        self.local_q = nn.Parameter(local_q.requires_grad_(True))
        self.neural_material = nn.Parameter(neural_material.requires_grad_(True))

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color_rgb = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        fused_color = RGB2SH(fused_color_rgb)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")
        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float32, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

        if self.use_textures:
            tex_color = fused_color_rgb[:, :, None, None].repeat(1, 1, self.texture_resolution, self.texture_resolution)
            tex_alpha = torch.full(
                (fused_point_cloud.shape[0], 1, self.texture_resolution, self.texture_resolution),
                1.0,
                dtype=torch.float32,
                device="cuda",
            )
            self._tex_color = nn.Parameter(inverse_sigmoid(tex_color).requires_grad_(True))
            self._tex_alpha = nn.Parameter(inverse_sigmoid(tex_alpha).requires_grad_(True))

        if self.use_mbrdf:
            self.initialize_mbrdf_state()

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        groups = [
            {"params": [self._xyz], "lr": training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {"params": [self._features_dc], "lr": training_args.feature_lr, "name": "f_dc"},
            {"params": [self._features_rest], "lr": training_args.feature_lr / 20.0, "name": "f_rest"},
            {"params": [self._opacity], "lr": training_args.opacity_lr, "name": "opacity"},
            {"params": [self._scaling], "lr": training_args.scaling_lr, "name": "scaling"},
            {"params": [self._rotation], "lr": training_args.rotation_lr, "name": "rotation"},
        ]
        if self.use_textures:
            groups.extend(
                [
                    {"params": [self._tex_color], "lr": training_args.texture_lr, "name": "tex_color"},
                    {"params": [self._tex_alpha], "lr": training_args.texture_lr, "name": "tex_alpha"},
                ]
            )
        if self.use_mbrdf:
            if self.asg_func is None or self.neural_phasefunc is None:
                self._initialize_mbrdf_modules()
            groups.extend(
                [
                    {"params": [self.kd], "lr": training_args.kd_lr, "name": "kd"},
                    {"params": [self.ks], "lr": training_args.ks_lr, "name": "ks"},
                    {"params": [self.alpha_asg], "lr": training_args.asg_lr_init, "name": "alpha_asg"},
                    {"params": [self.asg_func.asg_sigma], "lr": training_args.asg_lr_init, "name": "asg_sigma"},
                    {"params": [self.asg_func.asg_scales], "lr": training_args.asg_lr_init, "name": "asg_scales"},
                    {"params": [self.asg_func.asg_rotation], "lr": training_args.asg_lr_init, "name": "asg_rotation"},
                    {"params": [self.local_q], "lr": training_args.local_q_lr_init, "name": "local_q"},
                    {"params": [self.neural_material], "lr": training_args.neural_phasefunc_lr_init, "name": "neural_material"},
                    {"params": list(self.neural_phasefunc.parameters()), "lr": training_args.neural_phasefunc_lr_init, "name": "neural_phasefunc"},
                ]
            )

        self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )
        self.asg_scheduler_args = get_expon_lr_func(
            lr_init=training_args.asg_lr_init,
            lr_final=training_args.asg_lr_final,
            lr_delay_mult=training_args.asg_lr_delay_mult,
            max_steps=training_args.asg_lr_max_steps,
        )
        self.local_q_scheduler_args = get_expon_lr_func(
            lr_init=training_args.local_q_lr_init,
            lr_final=training_args.local_q_lr_final,
            lr_delay_mult=training_args.local_q_lr_delay_mult,
            max_steps=training_args.local_q_lr_max_steps,
        )
        self.neural_phasefunc_scheduler_args = get_expon_lr_func(
            lr_init=training_args.neural_phasefunc_lr_init,
            lr_final=training_args.neural_phasefunc_lr_final,
            lr_delay_mult=training_args.neural_phasefunc_lr_delay_mult,
            max_steps=training_args.neural_phasefunc_lr_max_steps,
        )

    def update_learning_rate(self, iteration, asg_freeze_step=0, local_q_freeze_step=0, freeze_phasefunc_steps=0):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                param_group["lr"] = self.xyz_scheduler_args(iteration)
            elif param_group["name"] in {"alpha_asg", "asg_sigma", "asg_rotation", "asg_scales"}:
                param_group["lr"] = self.asg_scheduler_args(max(0, iteration - asg_freeze_step))
            elif param_group["name"] == "local_q":
                param_group["lr"] = self.local_q_scheduler_args(max(0, iteration - local_q_freeze_step))
            elif param_group["name"] in {"neural_phasefunc", "neural_material"}:
                param_group["lr"] = self.neural_phasefunc_scheduler_args(max(0, iteration - freeze_phasefunc_steps))

    def construct_list_of_attributes(self):
        names = ["x", "y", "z", "nx", "ny", "nz"]
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            names.append(f"f_dc_{i}")
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            names.append(f"f_rest_{i}")
        names.append("opacity")
        for i in range(self._scaling.shape[1]):
            names.append(f"scale_{i}")
        for i in range(self._rotation.shape[1]):
            names.append(f"rot_{i}")
        return names

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        PlyData([PlyElement.describe(elements, "vertex")]).write(path)

    def save_appearance(self, path):
        if not self.use_textures and not self.use_mbrdf:
            return

        mkdir_p(os.path.dirname(path))
        payload = {"texture_resolution": self.texture_resolution}
        if self.use_textures:
            payload["tex_color"] = self._tex_color.detach().cpu()
            payload["tex_alpha"] = self._tex_alpha.detach().cpu()
        if self.use_mbrdf:
            payload.update(
                {
                    "use_mbrdf": True,
                    "basis_asg_num": self.basis_asg_num,
                    "hidden_feature_size": self.hidden_feature_size,
                    "hidden_feature_layers": self.hidden_feature_layers,
                    "phase_frequency": self.phase_frequency,
                    "neural_material_size": self.neural_material_size,
                    "asg_channel_num": self.asg_channel_num,
                    "asg_mlp": self.asg_mlp,
                    "asg_alpha_num": self.asg_alpha_num,
                    "kd": self.kd.detach().cpu(),
                    "ks": self.ks.detach().cpu(),
                    "alpha_asg": self.alpha_asg.detach().cpu(),
                    "local_q": self.local_q.detach().cpu(),
                    "neural_material": self.neural_material.detach().cpu(),
                    "asg_func": self.asg_func.state_dict(),
                    "neural_phasefunc": self.neural_phasefunc.state_dict(),
                }
            )
        torch.save(payload, path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)
        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree
        if self.use_textures:
            self.initialize_texture_state()
        if self.use_mbrdf:
            self.initialize_mbrdf_state()

    def initialize_texture_state(self):
        if not self.use_textures:
            return
        num_points = self._xyz.shape[0]
        tex_res = self.texture_resolution
        if num_points == 0:
            self._tex_color = nn.Parameter(torch.empty(0, device="cuda"))
            self._tex_alpha = nn.Parameter(torch.empty(0, device="cuda"))
            return

        base_color = torch.clamp(SH2RGB(self._features_dc.detach().squeeze(1)), 1e-6, 1 - 1e-6)
        tex_color = base_color[:, :, None, None].repeat(1, 1, tex_res, tex_res)
        tex_alpha = torch.ones((num_points, 1, tex_res, tex_res), dtype=torch.float32, device="cuda")
        self._tex_color = nn.Parameter(inverse_sigmoid(tex_color).requires_grad_(True))
        self._tex_alpha = nn.Parameter(inverse_sigmoid(tex_alpha).requires_grad_(True))

    def load_appearance(self, path):
        if not self.use_textures and not self.use_mbrdf:
            return
        if not os.path.exists(path):
            self.initialize_texture_state()
            self.initialize_mbrdf_state()
            return

        payload = torch.load(path, map_location="cpu")
        self.texture_resolution = int(payload.get("texture_resolution", self.texture_resolution))
        if self.use_textures:
            if "tex_color" in payload:
                self._tex_color = nn.Parameter(payload["tex_color"].to(device="cuda", dtype=torch.float).requires_grad_(True))
                self._tex_alpha = nn.Parameter(payload["tex_alpha"].to(device="cuda", dtype=torch.float).requires_grad_(True))
            else:
                self.initialize_texture_state()
        if self.use_mbrdf:
            if payload.get("use_mbrdf", False):
                self.kd = nn.Parameter(payload["kd"].to(device="cuda", dtype=torch.float).requires_grad_(True))
                self.ks = nn.Parameter(payload["ks"].to(device="cuda", dtype=torch.float).requires_grad_(True))
                self.alpha_asg = nn.Parameter(payload["alpha_asg"].to(device="cuda", dtype=torch.float).requires_grad_(True))
                self.local_q = nn.Parameter(payload["local_q"].to(device="cuda", dtype=torch.float).requires_grad_(True))
                self.neural_material = nn.Parameter(payload["neural_material"].to(device="cuda", dtype=torch.float).requires_grad_(True))
                self._initialize_mbrdf_modules()
                self.asg_func.load_state_dict(payload["asg_func"])
                self.neural_phasefunc.load_state_dict(payload["neural_phasefunc"])
            else:
                self.initialize_mbrdf_state()

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                    del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                if stored_state is not None:
                    self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        n_gaussians = mask.shape[0]
        for group in self.optimizer.param_groups:
            if len(group["params"]) != 1:
                continue
            param = group["params"][0]
            # Skip global (non-per-Gaussian) parameters whose first dim != N
            if param.shape[0] != n_gaussians:
                continue
            stored_state = self.optimizer.state.get(param, None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(param[mask].requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state
            else:
                group["params"][0] = nn.Parameter(param[mask].requires_grad_(True))
            optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.use_textures:
            self._tex_color = optimizable_tensors["tex_color"]
            self._tex_alpha = optimizable_tensors["tex_alpha"]
        if self.use_mbrdf:
            self.kd = optimizable_tensors["kd"]
            self.ks = optimizable_tensors["ks"]
            self.alpha_asg = optimizable_tensors["alpha_asg"]
            self.local_q = optimizable_tensors["local_q"]
            self.neural_material = optimizable_tensors["neural_material"]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) != 1:
                continue
            if group["name"] not in tensors_dict:
                # Global (non-per-Gaussian) parameters like asg_sigma, neural_phasefunc
                # are not replicated during densification.
                continue
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
            optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_tex_color=None,
        new_tex_alpha=None,
        new_kd=None,
        new_ks=None,
        new_alpha_asg=None,
        new_local_q=None,
        new_neural_material=None,
    ):
        tensors = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }
        if self.use_textures:
            tensors["tex_color"] = new_tex_color
            tensors["tex_alpha"] = new_tex_alpha
        if self.use_mbrdf:
            tensors["kd"] = new_kd
            tensors["ks"] = new_ks
            tensors["alpha_asg"] = new_alpha_asg
            tensors["local_q"] = new_local_q
            tensors["neural_material"] = new_neural_material

        optimizable_tensors = self.cat_tensors_to_optimizer(tensors)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.use_textures:
            self._tex_color = optimizable_tensors["tex_color"]
            self._tex_alpha = optimizable_tensors["tex_alpha"]
        if self.use_mbrdf:
            self.kd = optimizable_tensors["kd"]
            self.ks = optimizable_tensors["ks"]
            self.alpha_asg = optimizable_tensors["alpha_asg"]
            self.local_q = optimizable_tensors["local_q"]
            self.neural_material = optimizable_tensors["neural_material"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:, :1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_tex_color = self._tex_color[selected_pts_mask].repeat(N, 1, 1, 1) if self.use_textures else None
        new_tex_alpha = self._tex_alpha[selected_pts_mask].repeat(N, 1, 1, 1) if self.use_textures else None
        new_kd = self.kd[selected_pts_mask].repeat(N, 1) if self.use_mbrdf else None
        new_ks = self.ks[selected_pts_mask].repeat(N, 1) if self.use_mbrdf else None
        new_alpha_asg = self.alpha_asg[selected_pts_mask].repeat(N, 1, 1) if self.use_mbrdf else None
        new_local_q = self.local_q[selected_pts_mask].repeat(N, 1) if self.use_mbrdf else None
        new_neural_material = self.neural_material[selected_pts_mask].repeat(N, 1) if self.use_mbrdf else None

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_tex_color,
            new_tex_alpha,
            new_kd,
            new_ks,
            new_alpha_asg,
            new_local_q,
            new_neural_material,
        )

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_tex_color = self._tex_color[selected_pts_mask] if self.use_textures else None
        new_tex_alpha = self._tex_alpha[selected_pts_mask] if self.use_textures else None
        new_kd = self.kd[selected_pts_mask] if self.use_mbrdf else None
        new_ks = self.ks[selected_pts_mask] if self.use_mbrdf else None
        new_alpha_asg = self.alpha_asg[selected_pts_mask] if self.use_mbrdf else None
        new_local_q = self.local_q[selected_pts_mask] if self.use_mbrdf else None
        new_neural_material = self.neural_material[selected_pts_mask] if self.use_mbrdf else None

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_tex_color,
            new_tex_alpha,
            new_kd,
            new_ks,
            new_alpha_asg,
            new_local_q,
            new_neural_material,
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
