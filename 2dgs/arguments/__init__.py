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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self.use_textures = False
        self.cam_opt = False
        self.pl_opt = False
        self.texture_resolution = 4
        self.use_mbrdf = False
        self.basis_asg_num = 8
        self.phasefunc_hidden_size = 32
        self.phasefunc_hidden_layers = 3
        self.phasefunc_frequency = 4
        self.neural_material_size = 6
        self.asg_channel_num = 1
        self.asg_mlp = False
        self.asg_alpha_num = 1
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.view_num = -1
        self.render_items = ['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature']
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.depth_ratio = 0.0
        self.debug = False
        self.shadow_pass = False
        # diff-surfel-rasterization-texture: toggle texture sampling in CUDA
        # True  -> 2DGS + texture sampling
        # False -> 2DGS behavior (no texture sampling), still using texture rasterizer module
        self.enable_texture = True
        self.shadow_offset = 0.015
        self.shadow_light_scale = 10.0
        self.shadow_resolution_scale = 1.0
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.texture_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.kd_lr = 0.01
        self.ks_lr = 0.01
        self.asg_lr_init = 0.01
        self.asg_lr_final = 0.0001
        self.asg_lr_delay_mult = 0.01
        self.asg_lr_max_steps = 50_000
        self.local_q_lr_init = 0.01
        self.local_q_lr_final = 0.0001
        self.local_q_lr_delay_mult = 0.01
        self.local_q_lr_max_steps = 50_000
        self.neural_phasefunc_lr_init = 0.001
        self.neural_phasefunc_lr_final = 0.00001
        self.neural_phasefunc_lr_delay_mult = 0.01
        self.neural_phasefunc_lr_max_steps = 50_000
        # Camera & light pose optimization (match gs3)
        self.opt_cam_lr_init = 0.001
        self.opt_cam_lr_final = 0.00001
        self.opt_cam_lr_delay_step = 20_000
        self.opt_cam_lr_delay_mult = 0.2
        self.opt_cam_lr_max_steps = 80_000
        self.train_cam_freeze_step = 5_000

        self.opt_pl_lr_init = 1e-3
        self.opt_pl_lr_final = 0.00005
        self.opt_pl_lr_delay_step = 30_000
        self.opt_pl_lr_delay_mult = 0.1
        self.opt_pl_lr_max_steps = 80_000
        self.train_pl_freeze_step = 15_000

        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_dist = 0.0
        self.lambda_normal = 0.05
        # Match gs3's pruning floor. 0.05 was culling surfels far too
        # aggressively once pose/light optimization started to perturb
        # opacities, which starved late-stage regrowth.
        self.opacity_cull = 0.005

        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        # Match gs3's late-stage densification horizon. Stopping at 15k left
        # 2dgs in a prune-only regime for the rest of training, which explains
        # the 30k/100k point collapse we kept observing.
        self.densify_until_iter = 100_000
        self.densify_grad_threshold = 0.0002

        # Multi-phase training schedule (mirrors gs3)
        self.unfreeze_iterations = 5000
        self.spcular_freeze_step = 9000
        self.fit_linear_step = 7000
        self.asg_freeze_step = 22000
        self.asg_lr_freeze_step = 40_000
        self.local_q_lr_freeze_step = 40_000
        self.freeze_phasefunc_steps = 50_000
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
