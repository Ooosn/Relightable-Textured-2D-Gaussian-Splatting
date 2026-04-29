"""
One-step comparison: load gsplat-15k checkpoint, pick one camera,
do full MBRDF render with both backends, compare rendered_image and
dL_dcolors_precomp after loss.backward().
"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "submodules", "diff-gaussian-rasterization"))

from argparse import Namespace
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim

# Load config
ckpt_dir = r"D:\RTS\output\gs3_probe_from_gsplat15k_to30k_gsplat_nodensify_norreset"
with open(os.path.join(ckpt_dir, "cfg_args")) as f:
    cfg = eval(f.read())
cfg.source_path = r"E:\gsrelight-data\NRHints\Pixiu"
cfg.data_device = "cpu"
pipe = Namespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)

results = {}
for rast in ["gsplat", "3dgs"]:
    print(f"\n{'='*50}")
    print(f"Rasterizer: {rast}")
    print(f"{'='*50}")
    cfg.rasterizer = rast
    gaussians = GaussianModel(cfg)

    ckpt_path = os.path.join(ckpt_dir, "chkpnt15000.pth")
    if not os.path.exists(ckpt_path):
        # try the source checkpoint
        src = r"D:\RTS\output\gs3_NRHints_Pixiu"
        ckpt_path = os.path.join(src, "chkpnt15000.pth")
    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    if isinstance(ckpt, tuple):
        model_args, _ = ckpt
    else:
        model_args = ckpt
    # Manually restore state
    asg_keys = {"asg_sigma", "asg_rotation", "asg_scales"}
    skip_keys = {"optimizer", "neural_phasefunc"} | asg_keys
    for key, value in model_args.items():
        if key in skip_keys:
            continue
        try:
            setattr(gaussians, key, value)
        except:
            pass
    # Load asg_func params before training_setup
    for k in asg_keys:
        if k in model_args:
            setattr(gaussians.asg_func, k, model_args[k])
    # training_setup
    opt = Namespace(
        percent_dense=0.01, position_lr_init=0.00016, position_lr_final=0.0000016,
        position_lr_delay_mult=0.01, position_lr_max_steps=30000,
        feature_lr=0.0025, feature_rest_lr=0.000125, opacity_lr=0.05,
        scaling_lr=0.005, rotation_lr=0.001, kd_lr=0.01, ks_lr=0.01,
        asg_lr_init=0.01, asg_lr_final=0.0001, asg_lr_delay_mult=0.01,
        asg_lr_max_steps=50000, asg_lr_freeze_step=40000,
        local_q_lr_init=0.01, local_q_lr_final=0.0001, local_q_lr_delay_mult=0.01,
        local_q_lr_max_steps=50000, local_q_lr_freeze_step=40000,
        neural_phasefunc_lr_init=0.001, neural_phasefunc_lr_final=0.00001,
        neural_phasefunc_lr_delay_mult=0.01, neural_phasefunc_lr_max_steps=50000,
        freeze_phasefunc_steps=50000,
        densification_interval=100, opacity_reset_interval=3000,
        densify_from_iter=500, densify_until_iter=100000,
        densify_grad_threshold=0.0002, random_background=False,
        lambda_dssim=0.2,
        train_cam_freeze_step=5000, opt_cam_lr_init=0.001,
        opt_pl_lr_init=0.005,
        spcular_freeze_step=9000, fit_linear_step=7000,
        asg_freeze_step=22000,
        neural_material_lr_init=0.001,
    )
    gaussians.training_setup(opt)
    # Load neural_phasefunc state
    if "neural_phasefunc" in model_args:
        try:
            gaussians.neural_phasefunc.load_state_dict(model_args["neural_phasefunc"])
        except Exception as e:
            print(f"Warning: {e}")
    gaussians.rasterizer = rast

    scene = Scene(cfg, gaussians, load_iteration=-1)
    train_cams = scene.getTrainCameras()
    cam = train_cams[0]

    bg = torch.tensor([0,0,0,0,0,0,0], dtype=torch.float32, device="cuda")
    gaussians.optimizer.zero_grad()

    local_axises = gaussians.get_local_axises
    asg_scales = gaussians.asg_scales
    asg_axises = gaussians.get_asg_axises

    renderArgs = {
        "modelset": cfg, "pipe": pipe, "bg_color": bg,
        "fix_labert": False, "is_train": False,
        "asg_mlp": False, "iteration": 15001
    }

    render_pkg = render(cam, gaussians, None, None, local_axises, asg_scales, asg_axises, **renderArgs)

    image = render_pkg["render"]
    shadow = render_pkg["shadow"]
    other_effects = render_pkg["other_effects"]

    gt_image = cam.original_image.cuda()
    final_image = image * shadow + other_effects
    final_image = torch.clip(final_image, 0.0, 1.0)

    Ll1 = l1_loss(final_image, gt_image)
    loss = 0.8 * Ll1 + 0.2 * (1.0 - ssim(final_image, gt_image))

    print(f"loss = {loss.item():.10f}")
    print(f"rendered_image: shape={render_pkg['render'].shape}, "
          f"[{render_pkg['render'].min():.8f}, {render_pkg['render'].max():.8f}]")

    # Save rendered image for comparison
    results[rast] = {
        "loss": loss.item(),
        "rendered_image": render_pkg["render"].detach().clone(),
        "shadow": shadow.detach().clone(),
        "final_image": final_image.detach().clone(),
    }

    del scene, gaussians, render_pkg
    torch.cuda.empty_cache()

# Compare
ri_gs = results["gsplat"]["rendered_image"]
ri_3d = results["3dgs"]["rendered_image"]
diff = (ri_gs - ri_3d).abs()
print(f"\n{'='*50}")
print(f"FORWARD COMPARISON (rendered_image)")
print(f"{'='*50}")
print(f"mean diff: {diff.mean():.10e}")
print(f"max diff:  {diff.max():.10e}")
for ch in range(min(8, ri_gs.shape[0])):
    d = diff[ch]
    print(f"  ch{ch}: mean={d.mean():.6e}, max={d.max():.6e}")

fi_gs = results["gsplat"]["final_image"]
fi_3d = results["3dgs"]["final_image"]
diff_f = (fi_gs - fi_3d).abs()
print(f"\nFINAL IMAGE diff: mean={diff_f.mean():.10e}, max={diff_f.max():.10e}")
print(f"loss gsplat={results['gsplat']['loss']:.10f}, 3dgs={results['3dgs']['loss']:.10f}")

sh_gs = results["gsplat"]["shadow"]
sh_3d = results["3dgs"]["shadow"]
diff_s = (sh_gs - sh_3d).abs()
print(f"shadow diff: mean={diff_s.mean():.10e}, max={diff_s.max():.10e}")
