import torch
from random import randint
from argparse import ArgumentParser

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.loss_utils import l1_loss, ssim


def main():
    parser = ArgumentParser()
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    opt_group = OptimizationParams(parser)
    args = parser.parse_args()

    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    opt = opt_group.extract(args)
    pipe.shadow_pass = True
    pipe.shadow_resolution_scale = 1.0
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda") if dataset.white_background else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    gaussians = GaussianModel(
        dataset.sh_degree,
        dataset.use_textures,
        dataset.texture_resolution,
        dataset.use_mbrdf,
        dataset.basis_asg_num,
        dataset.phasefunc_hidden_size,
        dataset.phasefunc_hidden_layers,
        dataset.phasefunc_frequency,
        dataset.neural_material_size,
        dataset.asg_channel_num,
        dataset.asg_mlp,
        dataset.asg_alpha_num,
    )
    scene = Scene(dataset, gaussians, load_iteration=None, shuffle=False)
    gaussians.training_setup(opt)
    cam = scene.getTrainCameras()[randint(0, len(scene.getTrainCameras()) - 1)]
    out = render(cam, gaussians, pipe, bg)
    img = out["render"]
    pre_shadow = out["pre_shadow"]
    print("render", tuple(img.shape), float(img.mean()), bool(torch.isfinite(img).all()), flush=True)
    print("pre_shadow", None if pre_shadow is None else tuple(pre_shadow.shape), None if pre_shadow is None else float(pre_shadow.mean()), flush=True)
    gt = cam.original_image.cuda()
    loss = (1.0 - opt.lambda_dssim) * l1_loss(img, gt) + opt.lambda_dssim * (1.0 - ssim(img, gt))
    loss.backward()
    print("loss", float(loss), flush=True)
    print("kd_grad", gaussians.kd.grad is not None, float(torch.nan_to_num(gaussians.kd.grad.norm()).item()) if gaussians.kd.grad is not None else -1.0, flush=True)
    print("mat_grad", gaussians.neural_material.grad is not None, float(torch.nan_to_num(gaussians.neural_material.grad.norm()).item()) if gaussians.neural_material.grad is not None else -1.0, flush=True)
    print("xyz_grad", gaussians._xyz.grad is not None, float(torch.nan_to_num(gaussians._xyz.grad.norm()).item()) if gaussians._xyz.grad is not None else -1.0, flush=True)


if __name__ == "__main__":
    main()
