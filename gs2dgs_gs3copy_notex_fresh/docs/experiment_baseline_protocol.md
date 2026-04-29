# Experiment Baseline Protocol

This document records the current experiment contract so future runs do not
mix datasets, code trees, checkpoints, or texture branches.

## Current Goal

The current goal is not to invent another training setup.

The goal is:

1. Use the `2dgs` pipeline from `/home/wangyy/RTS/gs3` as the trusted baseline.
2. Verify that `/home/wangyy/RTS/gs2dgs_gs3copy_notex_fresh` matches that
   baseline in no-texture mode.
3. Only after no-texture is aligned, test texture / MicroTex changes.

If no-texture in `gs2dgs_gs3copy_notex_fresh` does not match the `gs3` baseline,
do not interpret texture results.

## Trusted Scripts

Use this script as the canonical Fish no-texture / texture comparison runner:

```text
/home/wangyy/RTS/gs2dgs_gs3copy_notex_fresh/experiments/fish_testsh_protocol.sh
```

The simple local baseline script is:

```text
/home/wangyy/RTS/gs2dgs_gs3copy_notex_fresh/test.sh
```

Do not hand-type a training command when one of these scripts can be used.

As of this note, the dataset line is:

```bash
model="Fish"
```

The corresponding source path is:

```text
/home/wangyy/data_download/gsrelight/NRHints/Fish/Fish
```

The same script originally used `model="Pixiu"`. If the experiment is about
Fish, do not use Pixiu results or Pixiu checkpoints as evidence.

## Fixed Baseline Parameters

These parameters come from `experiments/fish_testsh_protocol.sh` and must
remain aligned unless the change is deliberately documented:

```text
view_num=2000
data_device=cpu
iterations=100000
asg_freeze_step=22000
spcular_freeze_step=9000
fit_linear_step=7000
asg_lr_freeze_step=40000
asg_lr_max_steps=50000
asg_lr_init=0.01
asg_lr_final=0.0001
local_q_lr_freeze_step=40000
local_q_lr_init=0.01
local_q_lr_final=0.0001
local_q_lr_max_steps=50000
neural_phasefunc_lr_init=0.001
neural_phasefunc_lr_final=0.00001
freeze_phasefunc_steps=50000
neural_phasefunc_lr_max_steps=50000
position_lr_max_steps=70000
densify_until_iter=90000
test_iterations=10000 20000 30000 40000 50000 60000 70000 80000 90000 100000
save_iterations=10000 20000 30000 40000 50000 60000 70000 80000 90000 100000
checkpoint_iterations=10000 20000 30000 40000 50000 60000 70000 80000 90000 100000
unfreeze_iterations=5000
use_nerual_phasefunc=true
cam_opt=true
pl_opt=true
densify_grad_threshold=0.00015
eval=true
```

## Launch And Import Rules

Do not use `conda run` for long training in this project. It delays or buffers
logs and makes failures harder to localize. Use the environment Python directly:

```bash
/home/wangyy/miniconda3/envs/gs3/bin/python -u train.py ...
```

Do not judge CUDA extension health with a bare extension import. A command such
as this is not equivalent to the training import order:

```bash
python -c "import simple_knn._C"
```

The training process imports `torch` first, which loads the PyTorch shared
libraries such as `libc10.so`. If an import smoke test is needed, use the same
order:

```bash
ROOT=/home/wangyy/RTS/gs2dgs_gs3copy_notex_fresh
PYTHONPATH="$ROOT/submodules/simple-knn:$ROOT/submodules/diff-gaussian-rasterization:$ROOT/submodules/diff-gaussian-rasterization_light:$ROOT/submodules/diff-gaussian-rasterization_hgs:$ROOT/submodules/v_3dgs:$ROOT/submodules/v_3dgs_ortho:$ROOT/../2dgs/submodules/surfel-texture:$ROOT/../2dgs/submodules/surfel-texture-deferred:$ROOT/../2dgs/submodules/diff-surfel-rasterization-shadow:$ROOT" \
/home/wangyy/miniconda3/envs/gs3/bin/python -u -c "import torch; import simple_knn._C, diff_gaussian_rasterization, diff_gaussian_rasterization_light, v_3dgs; print('imports ok')"
```

For real validation, prefer a short training launch with the same script and
environment that will be used for the actual run. A synthetic import check is
only a smoke test.

## Known Good Fish Baseline Runs

These are the Fish runs that previously reached the expected high range.

### Fish no-texture warmup 30k

```text
/home/wangyy/data_download/gsrelight_runs/gs2dgs_gs3copy_texture_validation/fish_testsh_notex_warmup30k_20260428_032429
```

Important metric:

```text
iteration 30000 test PSNR = 29.55029851740057
```

### Fish no-texture continuation

```text
/home/wangyy/data_download/gsrelight_runs/gs2dgs_gs3copy_texture_validation/fish_testsh_notex_from_new30k_20260428_034827
```

Important metrics:

```text
iteration 50000 test PSNR = 30.975900534427527
iteration 60000 test PSNR = 31.178805640249543
iteration 100000 test PSNR = 31.140971299373742
```

### Fish texture continuation, old `per_uv`

```text
/home/wangyy/data_download/gsrelight_runs/gs2dgs_gs3copy_texture_validation/fish_testsh_tex_from_new30k_20260428_034835
```

Important metrics:

```text
iteration 50000 test PSNR = 31.07471801295425
iteration 60000 test PSNR = 31.30583786241936
iteration 100000 test PSNR = 31.312775640776664
```

## Known Bad / Non-Comparable Runs

The following run should not be used to judge the method:

```text
/home/wangyy/data_download/gsrelight_runs/gs2dgs_gs3copy_texture_validation/microtex_localq_exact_compare_20260430_005740
```

Reason:

It was launched from:

```text
/home/wangyy/data_download/gsrelight_runs/gs2dgs_gs3copy_texture_validation/clean_notex30k_pretexture_20260427/chkpnt30000.pth
```

That checkpoint has a weaker 30k Fish result:

```text
iteration 30000 test PSNR = 29.22087727171002
```

This is not the same starting point as the known good Fish `test.sh` run
(`29.55029851740057` at 30k). Its later PSNR cannot be compared directly
against the known good 31 dB runs.

The incorrect long-running `microtex_localq_exact_compare_20260430_005740`
processes were stopped.

## Texture Branch Names

Do not mix these modes when reporting results.

### `per_uv`

Older texture branch. Historical good Fish texture results used this mode.

### `uv_specular_gain`

Mean-shadow correction branch:

```text
shadow = clamp(decay_g + raw_shadow - point_shadow)
```

This branch uses Gaussian-level neural phase output plus per-UV shadow residual.
It was smoke-tested and resume-tested, but has not yet been fully validated
with the known good Fish 30k checkpoint.

### `per_uv_micro_normal`

Micro-normal branch. This is not the same as `uv_specular_gain`.

In the current implementation it computes per-UV local frames and calls the
neural phase function per texel. Therefore its results are not directly
comparable to old `per_uv` or `uv_specular_gain` unless the starting checkpoint
and all run parameters are identical.

## Required Run Order

For any new texture experiment:

1. Run or verify no-texture Fish baseline from `test.sh`.
2. Confirm the baseline reaches the known range.
3. Start texture from the known good Fish 30k checkpoint only.
4. Save the exact command or script used.
5. Record:
   - code directory,
   - git commit,
   - dataset,
   - checkpoint path,
   - texture mode,
   - eval iterations,
   - PSNR / SSIM / LPIPS,
   - Gaussian count,
   - texture resolution state if RTG is enabled.

If any of these fields are missing, the run should be treated as non-comparable.
