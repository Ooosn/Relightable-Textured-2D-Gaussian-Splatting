# GS2DGS

`gs2dgs` is the cleaned training shell for the active `2dgs` workflow.

It keeps:
- the `gs3`-style training / evaluation shell,
- the native `2dgs` surfel renderer,
- the deferred `7ch` relighting path,
- optional texture training.
- local vendored CUDA / rasterization dependencies under [`submodules/`](/d:/RTS/gs2dgs/submodules)

It does **not** keep the old `gsplat / 3dgs / hgs / aaai / beijing / shadow_render / video` branches.

## Supported workflow

The supported training path is:

- `rasterizer=2dgs`
- deferred `7ch`
- `use_nerual_phasefunc=True`
- optional `use_textures=True`

`2dgs_3ch` still exists in the renderer as an ablation/debug path, but it is not the recommended training path.

## Standalone boundary

`gs2dgs` is intended to run without importing runtime code from sibling project folders.

Local runtime dependencies now live under:

- [submodules/surfel-texture](/d:/RTS/gs2dgs/submodules/surfel-texture)
- [submodules/surfel-texture-deferred](/d:/RTS/gs2dgs/submodules/surfel-texture-deferred)
- [submodules/diff-surfel-rasterization-shadow](/d:/RTS/gs2dgs/submodules/diff-surfel-rasterization-shadow)
- [submodules/simple-knn](/d:/RTS/gs2dgs/submodules/simple-knn)
- [submodules/diff-surfel-rasterization](/d:/RTS/gs2dgs/submodules/diff-surfel-rasterization)

The active model wrapper is:

- [scene/gaussian_model_2dgs_adapter.py](/d:/RTS/gs2dgs/scene/gaussian_model_2dgs_adapter.py)

It wraps the vendored native implementation:

- [scene/gaussian_model_native_2dgs.py](/d:/RTS/gs2dgs/scene/gaussian_model_native_2dgs.py)

Current submodules include the local `cp312` Windows extension binaries needed by the active environment.
If you move to a new machine or a different Python/CUDA build, rebuild these extensions locally from the vendored source under [`submodules/`](/d:/RTS/gs2dgs/submodules).

## Environment

Examples below assume:

- repo root: `D:\RTS`
- conda env: `mygs`

## Training

### 30k no texture

```powershell
conda run -n mygs python D:\RTS\gs2dgs\train.py `
  -s D:\RTS\data\Pixiu `
  -m D:\RTS\output\gs2dgs_run `
  --eval `
  --iterations 30000 `
  --rasterizer 2dgs `
  --use_nerual_phasefunc `
  --cam_opt `
  --pl_opt `
  --sh_degree 0 `
  --view_num 2000 `
  --resolution 1
```

### 30k texture

```powershell
conda run -n mygs python D:\RTS\gs2dgs\train.py `
  -s D:\RTS\data\Pixiu `
  -m D:\RTS\output\gs2dgs_tex_run `
  --eval `
  --iterations 30000 `
  --rasterizer 2dgs `
  --use_nerual_phasefunc `
  --use_textures `
  --texture_resolution 4 `
  --texture_effect_mode per_uv `
  --cam_opt `
  --pl_opt `
  --sh_degree 0 `
  --view_num 2000 `
  --resolution 1
```

### Texture pixel densification / RTG short test

This is the active experiment for **2DGS texture-level pixel densification**.
The goal is to keep each Gaussian as one surfel, but dynamically refine its
local texture chart when relighting-related texture gradients are high.

This is different from normal Gaussian densification:

- normal densification adds / splits Gaussians,
- RTG pixel densification keeps the Gaussian and grows its texture chart,
- dynamic texture charts are stored in a flat atlas with per-Gaussian
  `[height, width, offset]` metadata.

Required switches:

- `--use_textures`
- `--texture_dynamic_resolution`
- `--texture_rtg_enabled`
- `--texture_effect_mode per_uv`

For a quick PC sanity test, force RTG refinement to happen early:

```powershell
conda run -n mygs python D:\RTS\gs2dgs\train.py `
  -s D:\RTS\data\Pixiu `
  -m D:\RTS\output\gs2dgs_rtg_short `
  --eval `
  --iterations 3000 `
  --rasterizer 2dgs `
  --use_nerual_phasefunc `
  --use_textures `
  --texture_dynamic_resolution `
  --texture_min_resolution 4 `
  --texture_max_resolution 64 `
  --texture_effect_mode per_uv `
  --texture_phase_chunk_texels 262144 `
  --texture_rtg_enabled `
  --texture_rtg_refine_from_iter 500 `
  --texture_rtg_refine_until_iter 2000 `
  --texture_rtg_refine_interval 500 `
  --texture_rtg_refine_fraction 0.02 `
  --texture_rtg_min_score 0.0 `
  --texture_rtg_resolution_gamma 1.0 `
  --texture_rtg_alpha_weight 1.0 `
  --texture_rtg_chunk_texels 262144 `
  --cam_opt `
  --pl_opt `
  --sh_degree 0 `
  --view_num 2000 `
  --resolution 1
```

Expected RTG log lines look like:

```text
[ITER 500] RTG refined 123 texture charts | texels 480000->496384 (+16384), avg 18.2/G, score mean/max 1.23e-04/9.87e-04, gate mean/max 0.00e+00/0.00e+00, candidates 12000/12000, res {4x4:12000, 8x8:123}
```

What to check:

- At refine iterations, the log should print `RTG refined ...`.
- `texels old->new` should increase.
- `res {...}` should gradually move some charts from `4x4` to `8x8`,
  then later to `16x16`, up to `--texture_max_resolution`.
- For this forced short test, `--texture_rtg_min_score 0.0` disables the
  absolute gate so the path is easy to verify. Real runs should use the
  default nonzero gate.
- Loss should not jump to NaN after the first refinement.
- If no RTG log appears, check that both `--texture_dynamic_resolution`
  and `--texture_rtg_enabled` are present, and that the run reaches
  `--texture_rtg_refine_from_iter`.
- If VRAM spikes during dynamic texture training, first lower
  `--texture_phase_chunk_texels` to `131072` or `65536`. This caps the
  largest temporary texel batch used by the neural phase / texture shading
  path. `--texture_rtg_chunk_texels` applies the same idea to RTG gradient
  aggregation.

The default RTG schedule starts later:

- `--texture_rtg_refine_from_iter 30000`
- `--texture_rtg_refine_until_iter 100000`
- `--texture_rtg_refine_interval 1000`
- `--texture_rtg_refine_fraction 0.02`
- `--texture_rtg_min_score 1e-10`
- `--texture_rtg_resolution_gamma 1.0`

So short tests should override these values as shown above. The production
default starts after specular rendering is enabled and after the ASG
anisotropic parameters have had several thousand iterations to settle.

The production RTG gate is resolution-aware:

```text
RTG_i >= texture_rtg_min_score * (resolution_i / texture_min_resolution) ^ texture_rtg_resolution_gamma
```

After this gate, `texture_rtg_refine_fraction` is only used as a budget cap.
The ranking score is also divided by the same resolution scale, so larger
charts need stronger relighting-aware texture gradients to keep growing.

### Important defaults

`train.py` now always appends the final iteration to:

- `save_iterations`
- `test_iterations`
- `checkpoint_iterations`

It also appends the default resume milestones `30000` and `40000` whenever
the requested total iteration count reaches them.

That means a normal run will automatically:

- save the final point cloud,
- run final evaluation,
- save the final full checkpoint.

When RTG dynamic texture refinement is enabled, `train.py` also appends
`--texture_rtg_refine_from_iter` to `checkpoint_iterations`. With the
production default this writes `chkpnt30000.pth` immediately before the first
RTG texture-chart refinement, so it can be used as a clean full-state warmup
checkpoint for resumed RTG experiments.

Full training checkpoints are saved as `chkpntxxxxx.pth` and include both the
Gaussian/texture state and the scene camera/light optimization state. They are
intended for `--start_checkpoint` resume, not just offline rendering.
Before saving, current batch gradients are cleared with `set_to_none=True`;
optimizer moments and densification / RTG accumulators are preserved.

## Rendering

Use `gs2dgs/render.py`.

This script:

- loads the requested iteration from `point_cloud/iteration_xxx/point_cloud.ply`,
- restores `chkpntxxx.pth` if it exists,
- renders `train` and/or `test`,
- saves `render / base / shadow / other / gt`,
- writes split metrics to JSON.

### Render latest iteration

```powershell
conda run -n mygs python D:\RTS\gs2dgs\render.py `
  -m D:\RTS\output\gs2dgs_run `
  --iteration -1
```

### Render test only

```powershell
conda run -n mygs python D:\RTS\gs2dgs\render.py `
  -m D:\RTS\output\gs2dgs_run `
  --iteration 30000 `
  --skip_train
```

Render output is written to:

- `MODEL_PATH/render_eval/iteration_xxxxx/train/...`
- `MODEL_PATH/render_eval/iteration_xxxxx/test/...`

Each split contains:

- `render/*.png`
- `base/*.png`
- `shadow/*.png`
- `other/*.png`
- `gt/*.png`
- `results.json`

There is also a top-level:

- `render_eval/iteration_xxxxx/summary.json`

## Model saving

There are two kinds of saved artifacts:

### 1. Point cloud save

Path:

- `MODEL_PATH/point_cloud/iteration_xxxxx/point_cloud.ply`

This stores explicit Gaussian / surfel state such as:

- positions
- scales
- rotations
- opacity
- SH / texture tensors

### 2. Full checkpoint

Path:

- `MODEL_PATH/chkpntxxxxx.pth`

This stores the full model state, including:

- Gaussian state
- optimizer state
- mBRDF parameters
- neural phase function weights
- ASG state

For relighting models, final evaluation and offline rendering should use the checkpoint when it exists.

## Training outputs

A normal training run writes:

- `cfg_args`
- `events.out.tfevents...`
- `point_cloud/iteration_xxxxx/point_cloud.ply`
- `chkpntxxxxx.pth`
- `convergence/eval_metrics.jsonl`
- `convergence/train_health.jsonl`
- `convergence/densify_events.jsonl`
- `convergence/fixed_views/...`

## Current cleanup boundary

This directory intentionally keeps only the active `gs2dgs` workflow.

If you need any of the following, use another project directory instead:

- `gsplat`
- `3dgs`
- `hgs`
- legacy custom render scenes
- shadow-only legacy render entrypoints
