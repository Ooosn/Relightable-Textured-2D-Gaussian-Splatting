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
import torch.nn.functional as F
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
        texture_dynamic_resolution: bool = False,
        texture_min_resolution: int = 4,
        texture_max_resolution: int = 64,
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
        self.texture_dynamic_resolution = texture_dynamic_resolution
        self.texture_min_resolution = texture_min_resolution
        self.texture_max_resolution = texture_max_resolution
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
        self._texture_dims = torch.empty(0, dtype=torch.int32)
        self._rtg_score = torch.empty(0)
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
        self.texture_rtg_enabled = False
        self.texture_rtg_refine_from_iter = 30_000
        self.texture_rtg_refine_until_iter = 100_000
        self.texture_rtg_refine_interval = 1_000
        self.texture_rtg_refine_fraction = 0.02
        self.texture_rtg_ema = 0.9
        self.texture_rtg_alpha_weight = 1.0
        self.texture_rtg_min_score = 1e-5
        self.texture_rtg_resolution_gamma = 1.0
        self.texture_rtg_chunk_texels = 262_144
        self._last_rtg_refine_log = {}
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
            {
                "texture_dynamic_resolution": self.texture_dynamic_resolution,
                "texture_min_resolution": self.texture_min_resolution,
                "texture_max_resolution": self.texture_max_resolution,
                "texture_dims": self._texture_dims,
                "rtg_score": self._rtg_score,
            },
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
            extra_state = model_args[32] if len(model_args) > 32 and isinstance(model_args[32], dict) else {}
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
            ) = model_args[:32]
            self.texture_dynamic_resolution = bool(extra_state.get("texture_dynamic_resolution", self.texture_dynamic_resolution))
            self.texture_min_resolution = int(extra_state.get("texture_min_resolution", self.texture_min_resolution))
            self.texture_max_resolution = int(extra_state.get("texture_max_resolution", self.texture_max_resolution))
            texture_dims = extra_state.get("texture_dims", torch.empty(0))
            self._texture_dims = texture_dims.to(device="cuda", dtype=torch.int32) if isinstance(texture_dims, torch.Tensor) else torch.empty(0, device="cuda", dtype=torch.int32)
            rtg_score = extra_state.get("rtg_score", torch.empty(0))
            self._rtg_score = rtg_score.to(device="cuda", dtype=torch.float32) if isinstance(rtg_score, torch.Tensor) else torch.empty(0, device="cuda")
            if self.use_mbrdf:
                self._initialize_mbrdf_modules()
                if asg_state is not None:
                    self.asg_func.load_state_dict(asg_state)
                if phase_state is not None:
                    self.neural_phasefunc.load_state_dict(phase_state)

        self.training_setup(training_args)
        if self.use_textures and self._tex_color.numel() == 0:
            self.initialize_texture_state()
        if self.use_textures and self.has_dynamic_textures:
            self._validate_dynamic_texture_layout(self._tex_color, self._tex_alpha, self._texture_dims)
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
    def get_texture_dims(self):
        if not self.has_dynamic_textures:
            device = self.get_xyz.device if self.get_xyz.numel() > 0 else "cuda"
            return torch.empty(0, device=device, dtype=torch.int32)
        return self._texture_dims

    @property
    def has_dynamic_textures(self):
        return (
            bool(self.use_textures)
            and bool(self.texture_dynamic_resolution)
            and isinstance(self._texture_dims, torch.Tensor)
            and self._texture_dims.numel() > 0
        )

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

    def _build_texture_dims_from_resolutions(self, resolutions):
        resolutions = resolutions.to(device=self.get_xyz.device, dtype=torch.int32)
        counts = resolutions.to(torch.long) * resolutions.to(torch.long)
        offsets = torch.zeros_like(resolutions, dtype=torch.int32)
        if resolutions.numel() > 1:
            offsets[1:] = torch.cumsum(counts[:-1], dim=0).to(torch.int32)
        return torch.stack([resolutions, resolutions, offsets], dim=1)

    def _texture_counts(self, dims=None):
        dims = self._texture_dims if dims is None else dims
        if dims.numel() == 0:
            return torch.empty(0, device=self.get_xyz.device, dtype=torch.long)
        return dims[:, 0].to(torch.long) * dims[:, 1].to(torch.long)

    def _dynamic_texture_stats(self):
        if not self.has_dynamic_textures:
            return {
                "points": 0,
                "total_texels": 0,
                "avg_texels": 0.0,
                "hist": [],
            }
        dims = self._texture_dims.detach()
        resolutions = dims[:, 0].to(torch.long)
        counts = (dims[:, 0].to(torch.long) * dims[:, 1].to(torch.long)).clamp_min(1)
        unique_res, unique_counts = torch.unique(resolutions, sorted=True, return_counts=True)
        total_texels = int(counts.sum().item())
        points = int(resolutions.numel())
        return {
            "points": points,
            "total_texels": total_texels,
            "avg_texels": float(total_texels) / max(1, points),
            "hist": [(int(r.item()), int(c.item())) for r, c in zip(unique_res, unique_counts)],
        }

    def texture_rtg_log_string(self):
        log = getattr(self, "_last_rtg_refine_log", None)
        if not log:
            return ""
        hist = ", ".join(f"{res}x{res}:{count}" for res, count in log.get("hist", []))
        return (
            f"texels {log['before_texels']}->{log['after_texels']} "
            f"(+{log['texel_delta']}), avg {log['avg_texels']:.1f}/G, "
            f"score mean/max {log['score_mean']:.3e}/{log['score_max']:.3e}, "
            f"gate mean/max {log['gate_mean']:.3e}/{log['gate_max']:.3e}, "
            f"candidates {log['candidate_count']}/{log['eligible_count']}, "
            f"res {{{hist}}}"
        )

    def _validate_dynamic_texture_layout(self, tex_color=None, tex_alpha=None, texture_dims=None):
        dims = self._texture_dims if texture_dims is None else texture_dims
        if dims.numel() == 0:
            return
        if dims.ndim != 2 or dims.shape[1] != 3:
            raise ValueError("Dynamic texture_dims must have shape [num_points, 3].")
        if dims.shape[0] != self.get_xyz.shape[0]:
            raise ValueError("Dynamic texture_dims point count does not match Gaussian count.")
        dims_long = dims.to(device=self.get_xyz.device, dtype=torch.long)
        if torch.any(dims_long[:, :2] <= 0).item():
            raise ValueError("Dynamic texture dimensions must be positive.")
        counts = dims_long[:, 0] * dims_long[:, 1]
        offsets = dims_long[:, 2]
        expected_offsets = torch.zeros_like(offsets)
        if offsets.numel() > 1:
            expected_offsets[1:] = torch.cumsum(counts[:-1], dim=0)
        if not torch.equal(offsets, expected_offsets):
            raise ValueError("Dynamic texture offsets are not contiguous.")
        total_texels = int(counts.sum().item())
        if tex_color is not None and tex_color.numel() > 0 and tex_color.shape[0] != total_texels:
            raise ValueError("Dynamic texture_color first dimension does not match texture_dims.")
        if tex_alpha is not None and tex_alpha.numel() > 0 and tex_alpha.shape[0] != total_texels:
            raise ValueError("Dynamic texture_alpha first dimension does not match texture_dims.")

    def _ensure_rtg_buffers(self):
        n_points = self.get_xyz.shape[0]
        if self._rtg_score.numel() != n_points:
            self._rtg_score = torch.zeros((n_points,), dtype=torch.float32, device=self.get_xyz.device)

    def _replace_dynamic_texture_tensors(self, tex_color, tex_alpha, texture_dims):
        self._validate_dynamic_texture_layout(tex_color, tex_alpha, texture_dims)
        if self.optimizer is None:
            self._tex_color = nn.Parameter(tex_color.requires_grad_(True))
            self._tex_alpha = nn.Parameter(tex_alpha.requires_grad_(True))
        else:
            optimizable = self.replace_tensor_to_optimizer(tex_color, "tex_color")
            self._tex_color = optimizable["tex_color"]
            optimizable = self.replace_tensor_to_optimizer(tex_alpha, "tex_alpha")
            self._tex_alpha = optimizable["tex_alpha"]
        self._texture_dims = texture_dims.to(device=self.get_xyz.device, dtype=torch.int32)
        self._ensure_rtg_buffers()

    def _init_dynamic_textures_from_base_color(self, base_color):
        num_points = base_color.shape[0]
        tex_res = max(1, int(self.texture_min_resolution))
        tex_res = min(tex_res, max(1, int(self.texture_max_resolution)))
        resolutions = torch.full((num_points,), tex_res, dtype=torch.int32, device=base_color.device)
        self._texture_dims = self._build_texture_dims_from_resolutions(resolutions)
        texels_per_point = tex_res * tex_res
        tex_color = base_color[:, None, :].expand(num_points, texels_per_point, 3).reshape(-1, 3)
        tex_alpha = torch.full((num_points * texels_per_point, 1), 0.99, dtype=torch.float32, device=base_color.device)
        self._validate_dynamic_texture_layout(tex_color, tex_alpha, self._texture_dims)
        self._tex_color = nn.Parameter(inverse_sigmoid(tex_color.clamp(1e-6, 1 - 1e-6)).requires_grad_(True))
        self._tex_alpha = nn.Parameter(inverse_sigmoid(tex_alpha).requires_grad_(True))
        self._ensure_rtg_buffers()

    def _texel_owner_ids(self, dims=None):
        dims = self._texture_dims if dims is None else dims
        counts = self._texture_counts(dims)
        if counts.numel() == 0:
            return torch.empty(0, device=self.get_xyz.device, dtype=torch.long)
        return torch.repeat_interleave(torch.arange(counts.shape[0], device=counts.device), counts)

    def _accumulate_dynamic_texture_grad_score(self, grad, score, weight=1.0):
        if grad is None or not self.has_dynamic_textures:
            return
        texel_score = grad.detach().abs().mean(dim=1)
        dims = self._texture_dims
        counts = self._texture_counts(dims)
        offsets = dims[:, 2].to(torch.long)
        max_texels = max(1, int(getattr(self, "texture_rtg_chunk_texels", 262_144)))
        max_points = 4096
        num_points = int(dims.shape[0])
        start = 0
        while start < num_points:
            point_end = min(start + max_points, num_points)
            cumulative = torch.cumsum(counts[start:point_end], dim=0)
            valid = torch.nonzero(cumulative <= max_texels, as_tuple=False).flatten()
            end = start + 1 if valid.numel() == 0 else start + int(valid[-1].item()) + 1
            chunk_counts = counts[start:end]
            for count_value in torch.unique(chunk_counts).tolist():
                local_idx = torch.nonzero(chunk_counts == count_value, as_tuple=False).flatten()
                if local_idx.numel() == 0:
                    continue
                point_idx = local_idx + start
                local = torch.arange(count_value, device=dims.device, dtype=torch.long)
                src = offsets[point_idx, None] + local[None, :]
                score[point_idx] += float(weight) * texel_score[src.reshape(-1)].view(point_idx.numel(), -1).sum(dim=1)
            start = end

    def _gather_dynamic_texels_by_mask(self, point_mask):
        selected_idx = torch.nonzero(point_mask, as_tuple=False).flatten()
        if selected_idx.numel() == 0:
            return self._tex_color.detach()[:0], self._tex_alpha.detach()[:0]

        dims = self._texture_dims
        counts = self._texture_counts(dims)
        offsets = dims[:, 2].to(torch.long)
        selected_counts = counts[selected_idx]
        selected_offsets = torch.zeros_like(selected_counts)
        if selected_counts.numel() > 1:
            selected_offsets[1:] = torch.cumsum(selected_counts[:-1], dim=0)
        total_texels = int(selected_counts.sum().item())
        selected_color = torch.empty((total_texels, self._tex_color.shape[1]), dtype=self._tex_color.dtype, device=self._tex_color.device)
        selected_alpha = torch.empty((total_texels, self._tex_alpha.shape[1]), dtype=self._tex_alpha.dtype, device=self._tex_alpha.device)

        max_texels = max(1, int(getattr(self, "texture_rtg_chunk_texels", 262_144)))
        for count_value in torch.unique(selected_counts).tolist():
            local_idx = torch.nonzero(selected_counts == count_value, as_tuple=False).flatten()
            if local_idx.numel() == 0:
                continue
            charts_per_chunk = max(1, max_texels // max(1, int(count_value)))
            local = torch.arange(count_value, device=dims.device, dtype=torch.long)
            for chunk_start in range(0, int(local_idx.numel()), charts_per_chunk):
                chunk_local_idx = local_idx[chunk_start:chunk_start + charts_per_chunk]
                src_points = selected_idx[chunk_local_idx]
                src = offsets[src_points, None] + local[None, :]
                dst = selected_offsets[chunk_local_idx, None] + local[None, :]
                selected_color[dst.reshape(-1)] = self._tex_color.detach()[src.reshape(-1)]
                selected_alpha[dst.reshape(-1)] = self._tex_alpha.detach()[src.reshape(-1)]
        return selected_color, selected_alpha

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
            if self.texture_dynamic_resolution:
                self._init_dynamic_textures_from_base_color(torch.clamp(fused_color_rgb, 1e-6, 1 - 1e-6))
            else:
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
        self.texture_rtg_enabled = bool(getattr(training_args, "texture_rtg_enabled", False))
        self.texture_rtg_refine_from_iter = int(getattr(training_args, "texture_rtg_refine_from_iter", 30_000))
        self.texture_rtg_refine_until_iter = int(getattr(training_args, "texture_rtg_refine_until_iter", 100_000))
        self.texture_rtg_refine_interval = int(getattr(training_args, "texture_rtg_refine_interval", 1_000))
        self.texture_rtg_refine_fraction = float(getattr(training_args, "texture_rtg_refine_fraction", 0.02))
        self.texture_rtg_ema = float(getattr(training_args, "texture_rtg_ema", 0.9))
        self.texture_rtg_alpha_weight = float(getattr(training_args, "texture_rtg_alpha_weight", 1.0))
        self.texture_rtg_min_score = float(getattr(training_args, "texture_rtg_min_score", 1e-5))
        self.texture_rtg_resolution_gamma = float(getattr(training_args, "texture_rtg_resolution_gamma", 1.0))
        self.texture_rtg_chunk_texels = int(getattr(training_args, "texture_rtg_chunk_texels", 262_144))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._ensure_rtg_buffers()

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
        payload = {
            "texture_resolution": self.texture_resolution,
            "texture_dynamic_resolution": self.texture_dynamic_resolution,
            "texture_min_resolution": self.texture_min_resolution,
            "texture_max_resolution": self.texture_max_resolution,
        }
        if self.use_textures:
            payload["tex_color"] = self._tex_color.detach().cpu()
            payload["tex_alpha"] = self._tex_alpha.detach().cpu()
            payload["texture_dims"] = self._texture_dims.detach().cpu() if self.has_dynamic_textures else torch.empty(0, dtype=torch.int32)
            payload["rtg_score"] = self._rtg_score.detach().cpu() if self._rtg_score.numel() > 0 else torch.empty(0)
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
            self._texture_dims = torch.empty(0, device="cuda", dtype=torch.int32)
            self._rtg_score = torch.empty(0, device="cuda")
            return

        base_color = torch.clamp(SH2RGB(self._features_dc.detach().squeeze(1)), 1e-6, 1 - 1e-6)
        if self.texture_dynamic_resolution:
            self._init_dynamic_textures_from_base_color(base_color)
            return
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
        self.texture_dynamic_resolution = bool(payload.get("texture_dynamic_resolution", self.texture_dynamic_resolution))
        self.texture_min_resolution = int(payload.get("texture_min_resolution", self.texture_min_resolution))
        self.texture_max_resolution = int(payload.get("texture_max_resolution", self.texture_max_resolution))
        if self.use_textures:
            if "tex_color" in payload:
                self._tex_color = nn.Parameter(payload["tex_color"].to(device="cuda", dtype=torch.float).requires_grad_(True))
                self._tex_alpha = nn.Parameter(payload["tex_alpha"].to(device="cuda", dtype=torch.float).requires_grad_(True))
                texture_dims = payload.get("texture_dims", torch.empty(0, dtype=torch.int32))
                self._texture_dims = texture_dims.to(device="cuda", dtype=torch.int32) if isinstance(texture_dims, torch.Tensor) else torch.empty(0, device="cuda", dtype=torch.int32)
                rtg_score = payload.get("rtg_score", torch.empty(0))
                self._rtg_score = rtg_score.to(device="cuda", dtype=torch.float32) if isinstance(rtg_score, torch.Tensor) else torch.empty(0, device="cuda")
                if self.has_dynamic_textures:
                    self._validate_dynamic_texture_layout(self._tex_color, self._tex_alpha, self._texture_dims)
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

    def _prune_dynamic_textures(self, valid_points_mask):
        if not self.has_dynamic_textures:
            return
        tex_color, tex_alpha = self._gather_dynamic_texels_by_mask(valid_points_mask)
        resolutions = self._texture_dims[valid_points_mask, 0]
        texture_dims = self._build_texture_dims_from_resolutions(resolutions)
        kept = int(valid_points_mask.sum().item())
        self._rtg_score = self._rtg_score[valid_points_mask] if self._rtg_score.numel() == valid_points_mask.shape[0] else torch.zeros((kept,), device=self.get_xyz.device)
        self._replace_dynamic_texture_tensors(tex_color, tex_alpha, texture_dims)

    def _append_dynamic_textures_from_mask(self, selected_pts_mask, repeat_count=1):
        if not self.has_dynamic_textures or selected_pts_mask.sum() == 0:
            return
        repeat_count = max(1, int(repeat_count))
        selected_color, selected_alpha = self._gather_dynamic_texels_by_mask(selected_pts_mask)
        if repeat_count > 1:
            selected_color = selected_color.repeat(repeat_count, 1)
            selected_alpha = selected_alpha.repeat(repeat_count, 1)
        old_resolutions = self._texture_dims[:, 0]
        new_resolutions = old_resolutions[selected_pts_mask].repeat(repeat_count)
        texture_dims = self._build_texture_dims_from_resolutions(torch.cat([old_resolutions, new_resolutions], dim=0))
        tex_color = torch.cat([self._tex_color.detach(), selected_color], dim=0)
        tex_alpha = torch.cat([self._tex_alpha.detach(), selected_alpha], dim=0)
        if self._rtg_score.numel() == old_resolutions.shape[0]:
            self._rtg_score = torch.cat(
                [self._rtg_score, torch.zeros((new_resolutions.shape[0],), dtype=torch.float32, device=self._rtg_score.device)],
                dim=0,
            )
        self._replace_dynamic_texture_tensors(tex_color, tex_alpha, texture_dims)

    def _resize_dynamic_textures(self, selected_pts_mask, new_resolutions):
        if not self.has_dynamic_textures or selected_pts_mask.sum() == 0:
            return 0

        old_dims = self._texture_dims
        old_resolutions = old_dims[:, 0].to(torch.long)
        old_offsets = old_dims[:, 2].to(torch.long)
        new_resolutions = new_resolutions.to(device=old_dims.device, dtype=torch.long)
        new_counts = new_resolutions * new_resolutions
        new_offsets = torch.zeros_like(new_resolutions)
        if new_resolutions.numel() > 1:
            new_offsets[1:] = torch.cumsum(new_counts[:-1], dim=0)
        total_texels = int(new_counts.sum().item())

        color_channels = self._tex_color.shape[1]
        alpha_channels = self._tex_alpha.shape[1]
        new_color = torch.empty((total_texels, color_channels), dtype=self._tex_color.dtype, device=self._tex_color.device)
        new_alpha = torch.empty((total_texels, alpha_channels), dtype=self._tex_alpha.dtype, device=self._tex_alpha.device)
        unchanged = old_resolutions == new_resolutions
        max_texels = max(1, int(getattr(self, "texture_rtg_chunk_texels", 262_144)))

        for res_value in torch.unique(old_resolutions[unchanged]).tolist():
            idx = torch.nonzero(torch.logical_and(unchanged, old_resolutions == res_value), as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            texels_per_chart = int(res_value * res_value)
            charts_per_chunk = max(1, max_texels // max(1, texels_per_chart))
            local = torch.arange(texels_per_chart, device=old_dims.device, dtype=torch.long)
            for chunk_start in range(0, int(idx.numel()), charts_per_chunk):
                chunk_idx = idx[chunk_start:chunk_start + charts_per_chunk]
                src = old_offsets[chunk_idx, None] + local[None, :]
                dst = new_offsets[chunk_idx, None] + local[None, :]
                new_color[dst.reshape(-1)] = self._tex_color.detach()[src.reshape(-1)]
                new_alpha[dst.reshape(-1)] = self._tex_alpha.detach()[src.reshape(-1)]

        changed = old_resolutions != new_resolutions
        for res_value in torch.unique(old_resolutions[changed]).tolist():
            idx = torch.nonzero(torch.logical_and(changed, old_resolutions == res_value), as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            target_res = int(new_resolutions[idx[0]].item())
            old_texels = int(res_value * res_value)
            new_texels = int(target_res * target_res)
            charts_per_chunk = max(1, max_texels // max(1, max(old_texels, new_texels)))
            local_old = torch.arange(old_texels, device=old_dims.device, dtype=torch.long)
            local_new = torch.arange(new_texels, device=old_dims.device, dtype=torch.long)
            for chunk_start in range(0, int(idx.numel()), charts_per_chunk):
                chunk_idx = idx[chunk_start:chunk_start + charts_per_chunk]
                src = old_offsets[chunk_idx, None] + local_old[None, :]
                color_values = torch.sigmoid(self._tex_color.detach()[src.reshape(-1)]).view(chunk_idx.numel(), res_value, res_value, color_channels)
                alpha_values = torch.sigmoid(self._tex_alpha.detach()[src.reshape(-1)]).view(chunk_idx.numel(), res_value, res_value, alpha_channels)
                color_values = color_values.permute(0, 3, 1, 2).contiguous()
                alpha_values = alpha_values.permute(0, 3, 1, 2).contiguous()
                color_up = F.interpolate(color_values, size=(target_res, target_res), mode="bilinear", align_corners=False)
                alpha_up = F.interpolate(alpha_values, size=(target_res, target_res), mode="bilinear", align_corners=False)
                color_up = inverse_sigmoid(color_up.permute(0, 2, 3, 1).reshape(-1, color_channels).clamp(1e-6, 1 - 1e-6))
                alpha_up = inverse_sigmoid(alpha_up.permute(0, 2, 3, 1).reshape(-1, alpha_channels).clamp(1e-6, 1 - 1e-6))
                dst = new_offsets[chunk_idx, None] + local_new[None, :]
                new_color[dst.reshape(-1)] = color_up
                new_alpha[dst.reshape(-1)] = alpha_up

        texture_dims = torch.stack(
            [new_resolutions.to(torch.int32), new_resolutions.to(torch.int32), new_offsets.to(torch.int32)],
            dim=1,
        )
        refined = int(selected_pts_mask.sum().item())
        self._replace_dynamic_texture_tensors(new_color, new_alpha, texture_dims)
        if self._rtg_score.numel() == selected_pts_mask.shape[0]:
            self._rtg_score[selected_pts_mask] = 0.0
        return refined

    def accumulate_texture_rtg(self):
        if not (self.use_textures and self.texture_rtg_enabled):
            return
        if self._tex_color.grad is None and self._tex_alpha.grad is None:
            return
        self._ensure_rtg_buffers()
        with torch.no_grad():
            n_points = self.get_xyz.shape[0]
            score = torch.zeros((n_points,), dtype=torch.float32, device=self.get_xyz.device)
            if self.has_dynamic_textures:
                counts = self._texture_counts().to(torch.float32).clamp_min(1.0)
                self._accumulate_dynamic_texture_grad_score(self._tex_color.grad, score)
                self._accumulate_dynamic_texture_grad_score(
                    self._tex_alpha.grad,
                    score,
                    weight=self.texture_rtg_alpha_weight,
                )
                score = score / counts
            else:
                if self._tex_color.grad is not None:
                    score = score + self._tex_color.grad.detach().abs().flatten(start_dim=1).mean(dim=1)
                if self._tex_alpha.grad is not None:
                    score = score + self.texture_rtg_alpha_weight * self._tex_alpha.grad.detach().abs().flatten(start_dim=1).mean(dim=1)
            score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
            ema = min(max(float(self.texture_rtg_ema), 0.0), 0.9999)
            self._rtg_score.mul_(ema).add_(score, alpha=1.0 - ema)

    def refine_textures_by_rtg(self, iteration):
        self._last_rtg_refine_log = {}
        if not (self.use_textures and self.texture_dynamic_resolution and self.texture_rtg_enabled and self.has_dynamic_textures):
            return 0
        if iteration < self.texture_rtg_refine_from_iter or iteration > self.texture_rtg_refine_until_iter:
            return 0
        interval = max(1, int(self.texture_rtg_refine_interval))
        if iteration % interval != 0:
            return 0
        self._ensure_rtg_buffers()
        old_resolutions = self._texture_dims[:, 0].to(torch.long)
        eligible = old_resolutions < int(self.texture_max_resolution)
        num_eligible = int(eligible.sum().item())
        if num_eligible == 0:
            return 0
        min_res = max(1.0, float(getattr(self, "texture_min_resolution", 1)))
        gamma = max(0.0, float(getattr(self, "texture_rtg_resolution_gamma", 1.0)))
        base_gate = max(0.0, float(getattr(self, "texture_rtg_min_score", 1e-5)))
        resolution_scale = (old_resolutions.to(torch.float32) / min_res).clamp_min(1.0).pow(gamma)
        score_gate = resolution_scale * base_gate
        valid = eligible
        if base_gate > 0:
            valid = torch.logical_and(valid, self._rtg_score >= score_gate)
        candidate_count = int(valid.sum().item())
        if candidate_count == 0:
            return 0
        fraction = max(0.0, float(self.texture_rtg_refine_fraction))
        if fraction <= 0.0:
            return 0
        k = max(1, int(np.ceil(candidate_count * fraction)))
        valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
        priority = self._rtg_score / resolution_scale
        top_priority, top_order = torch.topk(priority[valid_idx], k=min(k, valid_idx.numel()), largest=True)
        if top_priority.numel() == 0 or float(top_priority.max().item()) <= 0.0:
            return 0
        selected = torch.zeros_like(valid)
        selected[valid_idx[top_order]] = True
        new_resolutions = old_resolutions.clone()
        new_resolutions[selected] = torch.clamp(new_resolutions[selected] * 2, max=int(self.texture_max_resolution))
        selected = torch.logical_and(selected, new_resolutions > old_resolutions)
        if selected.sum() == 0:
            return 0
        before_stats = self._dynamic_texture_stats()
        selected_scores = self._rtg_score[selected].detach()
        selected_gates = score_gate[selected].detach()
        refined = self._resize_dynamic_textures(selected, new_resolutions)
        if refined > 0:
            after_stats = self._dynamic_texture_stats()
            self._last_rtg_refine_log = {
                "iteration": int(iteration),
                "refined": int(refined),
                "before_texels": int(before_stats["total_texels"]),
                "after_texels": int(after_stats["total_texels"]),
                "texel_delta": int(after_stats["total_texels"] - before_stats["total_texels"]),
                "avg_texels": float(after_stats["avg_texels"]),
                "hist": after_stats["hist"],
                "score_mean": float(selected_scores.mean().item()) if selected_scores.numel() > 0 else 0.0,
                "score_max": float(selected_scores.max().item()) if selected_scores.numel() > 0 else 0.0,
                "gate_mean": float(selected_gates.mean().item()) if selected_gates.numel() > 0 else 0.0,
                "gate_max": float(selected_gates.max().item()) if selected_gates.numel() > 0 else 0.0,
                "candidate_count": int(candidate_count),
                "eligible_count": int(num_eligible),
            }
        return refined

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
        if self.use_textures and not self.has_dynamic_textures:
            self._tex_color = optimizable_tensors["tex_color"]
            self._tex_alpha = optimizable_tensors["tex_alpha"]
        elif self.use_textures and self.has_dynamic_textures:
            self._prune_dynamic_textures(valid_points_mask)
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
        if self.use_textures and not self.has_dynamic_textures:
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
        if self.use_textures and not self.has_dynamic_textures:
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
        new_tex_color = self._tex_color[selected_pts_mask].repeat(N, 1, 1, 1) if self.use_textures and not self.has_dynamic_textures else None
        new_tex_alpha = self._tex_alpha[selected_pts_mask].repeat(N, 1, 1, 1) if self.use_textures and not self.has_dynamic_textures else None
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
        if self.use_textures and self.has_dynamic_textures:
            self._append_dynamic_textures_from_mask(selected_pts_mask, repeat_count=N)

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
        new_tex_color = self._tex_color[selected_pts_mask] if self.use_textures and not self.has_dynamic_textures else None
        new_tex_alpha = self._tex_alpha[selected_pts_mask] if self.use_textures and not self.has_dynamic_textures else None
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
        if self.use_textures and self.has_dynamic_textures:
            self._append_dynamic_textures_from_mask(selected_pts_mask, repeat_count=1)

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
