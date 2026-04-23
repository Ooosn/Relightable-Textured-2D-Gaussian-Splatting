import importlib.util
import os
import sys
from pathlib import Path

import torch
from torch import nn


_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent.parent
_SIMPLE_KNN_PATH = _ROOT / "gs2dgs" / "submodules" / "simple-knn"
_SRC = _ROOT / "2dgs" / "scene" / "gaussian_model.py"

if _SIMPLE_KNN_PATH.is_dir():
    _simple_knn = str(_SIMPLE_KNN_PATH)
    if _simple_knn not in sys.path:
        sys.path.insert(0, _simple_knn)

_SPEC = importlib.util.spec_from_file_location("gaussian_model_2dgs_impl", _SRC)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(_MOD)
_BaseGaussianModel2DGS = _MOD.GaussianModel


class GaussianModel2DGSAdapter(_BaseGaussianModel2DGS):
    """Use the native 2DGS surfel model inside gs3's training loop."""

    def __init__(self, modelset, opt=None):
        super().__init__(
            sh_degree=getattr(modelset, "sh_degree", 0),
            use_textures=False,
            texture_resolution=4,
            use_mbrdf=getattr(modelset, "use_nerual_phasefunc", False),
            basis_asg_num=getattr(modelset, "basis_asg_num", 8),
            hidden_feature_size=getattr(modelset, "phasefunc_hidden_size", 32),
            hidden_feature_layers=getattr(modelset, "phasefunc_hidden_layers", 3),
            phase_frequency=getattr(modelset, "phasefunc_frequency", 4),
            neural_material_size=getattr(modelset, "neural_material_size", 6),
            asg_channel_num=getattr(modelset, "asg_channel_num", 1),
            asg_mlp=getattr(modelset, "asg_mlp", False),
            asg_alpha_num=getattr(modelset, "asg_alpha_num", 1),
        )
        self.rasterizer = getattr(modelset, "rasterizer", "2dgs")
        self.use_MBRDF = self.use_mbrdf
        self.use_hgs = False
        self.maximum_gs = getattr(modelset, "maximum_gs", 550_000)
        self.light_direction = torch.zeros(3, dtype=torch.float32, device="cuda")

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

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, bigsize_threshold=None, crop_extent=None):
        before = int(self.get_xyz.shape[0])
        super().densify_and_prune(max_grad, min_opacity, extent, max_screen_size)
        after = int(self.get_xyz.shape[0])
        # 2dgs native implementation does not report clone/split counts.
        return 0, max(after - before, 0)

    def add_densification_stats(self, viewspace_point_tensor, update_filter, image_width=None, image_height=None, out_weight=None):
        return super().add_densification_stats(viewspace_point_tensor, update_filter)

    def change_alpha_asg(self, alpha_asg):
        if not self.use_mbrdf:
            return
        alpha_asg_3 = alpha_asg.repeat(1, 1, 3)
        optimizable = self.replace_tensor_to_optimizer(alpha_asg_3, "alpha_asg")
        self.alpha_asg = optimizable["alpha_asg"]
        self.asg_alpha_num = 3

    def reset_local_q(self, temp):
        if not self.use_mbrdf:
            return
        local_q_new = temp * 0.6 + self.local_q * 0.4
        optimizable = self.replace_tensor_to_optimizer(local_q_new, "local_q")
        self.local_q = optimizable["local_q"]
