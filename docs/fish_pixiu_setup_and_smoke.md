# Fish/Pixiu Setup And Smoke Notes

Date: 2026-04-06

## Data

- Dataset source: `https://huggingface.co/datasets/gsrelight/gsrelight-data`
- Downloaded subset: `NRHints/Fish.zip`, `NRHints/Pixiu.zip`
- Local paths:
  - `/home/wangyy/data_download/gsrelight/NRHints/Fish/Fish`
  - `/home/wangyy/data_download/gsrelight/NRHints/Pixiu/Pixiu`

Each scene contains:

- `train/`
- `test/`
- `transforms_train.json`
- `transforms_test.json`
- `transforms_valid.json`

The JSON format matches the `gs3`/`NRHints` style:

- scene-level `camera_intrinsics`
- per-frame `transform_matrix`
- per-frame `pl_pos`
- per-frame `pl_intensity`
- per-frame `file_path` + `file_ext`

## Environment

Reference file: `/home/wangyy/conda.sh`

Important correction:

- `conda.sh` prefers `/home/wangyy/cuda`, but this node does not have that path.
- The working CUDA toolkit is `/opt/cuda/current`
- `nvcc` is `/opt/cuda/current/bin/nvcc`

Created env:

```bash
source /home/wangyy/miniconda3/etc/profile.d/conda.sh
conda create -y -n rts-relight python=3.10 pip
conda activate /home/wangyy/miniconda3/envs/rts-relight
export CUDA_HOME=/opt/cuda/current
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export TORCH_CUDA_ARCH_LIST=9.0
```

Installed and verified:

- `torch 2.5.1+cu121`
- `torchvision 0.20.1`
- `torchaudio 2.5.1`
- `numpy 1.26.3`
- `plyfile 0.8.1`
- `opencv-python 4.9.0.80`
- `tensorboard 2.16.2`
- `tqdm 4.66.2`
- `lpips 0.1.4`
- `jaxtyping`
- `gsplat 1.1.1` from local wheel

GPU probe result:

- `torch.cuda.is_available() == True`
- device: `NVIDIA H100 80GB HBM3`

## 2DGS Loader Smoke

The `2dgs` loader was extended earlier to parse `NRHints` metadata:

- global `camera_intrinsics`
- per-frame `file_ext`
- per-frame `pl_pos`
- per-frame `pl_intensity`

Smoke result on real data:

- `Fish`
  - train cameras: `526`
  - test cameras: `66`
  - first train frame: `r_0`
  - first train `pl_pos`: `[-4.0784, 2.3137, -2.2764]`
  - first train `pl_intensity`: `[4.9184, 4.9184, 4.9184]`
- `Pixiu`
  - train cameras: `562`
  - test cameras: `71`
  - first train frame: `r_0`
  - first train `pl_pos`: `[-3.9575, 5.6347, -0.7349]`
  - first train `pl_intensity`: `[6.6441, 6.6441, 6.6441]`

Notes:

- These scenes do not ship with a COLMAP point cloud.
- `2dgs` therefore generated `points3d.ply` automatically using its random initialization path.
- Generated files:
  - `/home/wangyy/data_download/gsrelight/NRHints/Fish/Fish/points3d.ply`
  - `/home/wangyy/data_download/gsrelight/NRHints/Pixiu/Pixiu/points3d.ply`

## GS3 Loader Smoke

`gs3` was also exercised on the real `Fish` scene using its native `Scene` and `GaussianModel`.

Smoke configuration:

- `view_num=12`
- `load_num=12`
- `data_device=cpu`
- `cam_opt=False`
- `pl_opt=False`

Result:

- train cameras: `12`
- test cameras: `12`
- initialized gaussians: `100000`
- first train frame: `r_0`
- first train `pl_pos`: `[[-4.0784, 2.3137, -2.2764]]`

This confirms the `gs3` camera/light metadata path is already compatible with `NRHints/Fish`.

## Shadow Smoke

Compiled and installed:

- `simple_knn`
- `diff_gaussian_rasterization_light`

Then ran a minimal CUDA shadow-rasterization smoke test directly against the installed extension.

Observed result:

- `radii_positive = 2`
- shadow values: `[1.0, 0.6871543526649475]`
- `means3d` gradient finite
- `opacities` gradient finite
- `cov3d` gradient finite
- `all_finite = True`

Interpretation:

- The light-space shadow rasterizer is compiled and executing correctly on GPU.
- Forward and backward both work on this node.

## GS3 Native Shadow Smoke

After the renderer submodules were built in-place from their own source directories, `gs3` could import:

- `gaussian_renderer`
- `gaussian_renderer.shadow_render`

Then a real `Fish` scene smoke test was run through `gs3`'s native `Scene -> GaussianModel -> shadow_render` path.

Smoke setup notes:

- scene: `NRHints/Fish`
- loader subset: `12` train + `12` test views
- because the random initialization path has no semantic labels, the smoke injected a temporary all-ones `object_id`
- this was only for validating the renderer path, not for semantic grouping

Observed result:

- render shape: `(3, 512, 512)`
- shadow shape: `(1, 512, 512)`
- shadow mean: `0.5312`
- shadow min/max: `0.3001 / 0.9986`
- positive radii: `69046`
- all finite: `True`

This is the first full native `gs3` shadow pass smoke on the real `Fish` data in this workspace.

## Current Limitation

`gs3/gaussian_renderer/__init__.py` injects submodule source directories into `sys.path`.

That means plain `pip install` is not enough for the main render stack.

The working approach is:

1. run `setup.py build_ext --inplace` from each submodule directory itself
2. let `gaussian_renderer` import the matching in-tree `_C...so` files

During this process, several real glue bugs in `gs3/gaussian_renderer/shadow_render.py` were also fixed locally:

- detached `object_center` before `.cpu().numpy()`
- passed an `(N, 3)` tensor to the light rasterizer instead of an `(N+2, 1)` object-id payload
- unpacked the light rasterizer's current 6-return signature correctly
- added missing `znear` / `zfar` when creating `v_3dgs_settings`

These fixes were necessary for the native `Fish` shadow smoke to complete.

## Recommended Next Step

Use `gs3` as the reference implementation for:

- dataset/light metadata ingestion
- camera/light optimization structure
- light-space shadow pass

Use `2dgs` as the integration target for:

- current geometry backbone
- the relighting-aware data interface already added
- future shadow-aware rendering path

Practical order:

1. finish the in-place builds for the remaining `gs3` renderer submodules
2. run one tiny native `gs3` render/shadow smoke on `Fish`
3. port the minimal light-camera and shadow-pass pieces into `2dgs`

## 2DGS Migration Cut

The smallest practical insertion point for bringing shadows into `2dgs` is:

- `/home/wangyy/RTS/2dgs/gaussian_renderer/__init__.py`

Why this is the right cut:

- the whole camera-space render path is centralized there
- `train.py` already consumes the returned render package
- the loader now already exposes per-view `pl_pos` / `pl_intensity`

Minimal migration sequence:

1. keep the current `2dgs` camera-space surfel render as the final composition pass
2. add a `gs3`-style light-space helper pass before final composition
3. return `shadow` and `pre_shadow` in the `render_pkg`
4. let `train.py` optionally supervise or regularize with that new shadow output

Files that should be touched first in the next round:

- `/home/wangyy/RTS/2dgs/gaussian_renderer/__init__.py`
- `/home/wangyy/RTS/2dgs/train.py`
- `/home/wangyy/RTS/2dgs/scene/cameras.py`
- `/home/wangyy/RTS/2dgs/scene/gaussian_model.py`

## 2DGS Shadow Port Status

The first end-to-end `2dgs` shadow port is now alive on the real `NRHints/Fish` data.

What was added on the `2dgs` side:

- new pipeline flags for `--shadow_pass`, `--shadow_offset`, `--shadow_light_scale`, and `--shadow_resolution_scale`
- a `gs3`-style light-space shadow helper inside `2dgs/gaussian_renderer/__init__.py`
- a compatibility shim for the current `v_3dgs` Python binding shape
- a safe light-buffer resolution policy so the first smoke does not explode memory

Observed forward smoke on `Fish`:

- render shape: `(3, 512, 512)`
- shadow shape: `(1, 512, 512)`
- shadow mean: `0.000104`
- all finite: `True`
- positive radii: `68419`

Observed minimal train-step smoke on `Fish`:

- loss: `0.514196`
- shadow mean: `0.896016`
- `xyz` gradient norm: `0.004285`
- optimizer step: completed

Important current limitation:

- the shadow branch is currently detached before final image modulation in `2dgs`

Why that temporary choice was made:

- the imported `v_3dgs` rasterizer backward path is not yet stable in this environment
- forward shadow rendering works
- direct backward through the shadow rasterizer currently hits binding / gradient-shape mismatches
- detaching the shadow image still lets the main `2dgs` render path train normally while using a real forward shadow image

So the current milestone is:

- forward shadow path: working
- image-space composition with shadow: working
- training loop with shadow-enabled render package: working
- backward through the shadow rasterizer itself: still pending

## 2DGS MBRDF Port Status

The next layer from `gs3` has now been ported into `2dgs` as a minimal material stack instead of only a shadow pass.

Ported pieces:

- per-Gaussian material tensors: `kd`, `ks`, `alpha_asg`, `local_q`, `neural_material`
- shared `Mixture_of_ASG`
- shared `Neural_phase` with a PyTorch fallback when `tinycudann` is unavailable
- Python-side MBRDF shading inside `2dgs/gaussian_renderer/__init__.py`

Important design choice:

- the final image still uses the native `2dgs` surfel rasterizer
- `gs3` was used as the material / shadow reference, not copied wholesale as the final renderer

Observed `Fish` MBRDF smoke:

- render shape: `(3, 512, 512)`
- render mean: `0.034718`
- shadow mean: `0.860112`
- loss: `0.226571`
- `kd` grad norm: `8.64e-06`
- `neural_material` grad norm: `2.32e-06`
- `xyz` grad norm: `4.83e-04`

Observed 5-step mini training smoke:

- iter 1 loss: `0.240683`
- iter 2 loss: `0.252696`
- iter 3 loss: `0.245504`
- iter 4 loss: `0.251363`
- iter 5 loss: `0.230980`

Interpretation:

- the new MBRDF branch is part of the computation graph
- material tensors and geometry both receive gradients
- the system is trainable in the practical sense of "forward + backward + optimizer step" with the new material stack enabled
- this is still only a short smoke, not yet evidence of long-horizon convergence
