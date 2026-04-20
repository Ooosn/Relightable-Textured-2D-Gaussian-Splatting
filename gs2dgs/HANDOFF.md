# GS2DGS Handoff

## Current mainline

Use:

- `rasterizer=2dgs`
- deferred `7ch`
- `use_nerual_phasefunc=True`

Recommended branches:

- no texture: active baseline
- texture: active secondary branch

Do not use `2dgs_3ch` as the main training path.

## Best known 30k results

### No texture

Run:

- [output/_gs2dgs_impl30k_bootstrap](/d:/RTS/output/_gs2dgs_impl30k_bootstrap)

Metrics:

- `30k` test PSNR `29.6203`
- final points `92,384`

### Texture

Run:

- [output/_gs2dgs_tex30k_bootstrap_v2](/d:/RTS/output/_gs2dgs_tex30k_bootstrap_v2)

Metrics:

- `30k` test PSNR `29.3689`
- final points `79,725`

Known gap:

- texture is about `0.25 dB` below no-texture at `30k`

## Current texture status

Texture path is connected and stable.

Important fixes already included:

- texture `basecolor` includes specular
- `25 GB` VRAM guard in training
- per-UV `other_effects` path exists behind `texture_effect_mode`

Current observation:

- `texture_effect_mode=per_uv` is implemented
- but it has **not** beaten the stable texture baseline yet

## Texture RTG status

There is a texture-gradient driven dynamic-resolution path in code:

- `texture_rtg_enabled`
- `texture_dynamic_resolution`
- `_tex_color.grad / _tex_alpha.grad -> per-Gaussian _rtg_score`
- `refine_textures_by_rtg(iteration)`

This is **not enabled by default**. The stable texture baseline uses fixed
`texture_resolution=4` and does not run RTG refinement.

To actually trigger RTG texture refinement, use both:

- `--texture_dynamic_resolution`
- `--texture_rtg_enabled`

Default RTG schedule:

- accumulate `_rtg_score` whenever `use_textures` and `texture_rtg_enabled` are true
- refine from iteration `15000`
- refine until iteration `100000`
- refine every `1000` iterations
- refine top `2%` valid Gaussian texture charts each event
- double selected chart resolution up to `texture_max_resolution=64`

RTG smoke validation on `2026-04-20`:

- full-view short run with `view_num=2000`, `texture_max_resolution=64`,
  and early refine reached iter `2200` but pushed VRAM to about `32 GB`, so it
  was stopped as too heavy for a quick check
- lightweight RTG smoke with `view_num=4`, `load_num=4`,
  `texture_max_resolution=16`, refine `100..500` every `100`, completed
- observed RTG lines at iter `100`, `300`, and `500`
- texels increased `1600000 -> 2001328`
- resolution histogram progressed from all `4x4` to `{4x4:94791, 8x8:4421, 16x16:788}`
- no CUDA error or NaN was observed; final `chkpnt700.pth` was written

## Save / eval behavior

`train.py` now always appends the final iteration to:

- `save_iterations`
- `test_iterations`
- `checkpoint_iterations`

That means future runs should automatically produce:

- final `point_cloud`
- final `test` eval
- final `chkpnt*.pth`

## Windows validation status

Last checked on `2026-04-20` on the local Windows workstation.

Environment:

- data root: `E:\gsrelight-data\NRHints\Pixiu`
- conda env: `mygs`
- PyTorch `2.7.1+cu128`
- CUDA toolkit `12.8`
- GPU: RTX 4080 SUPER
- build arch: `TORCH_CUDA_ARCH_LIST=8.9`

Rebuilt local Windows `cp312` CUDA extensions under:

- `submodules/simple-knn`
- `submodules/diff-surfel-rasterization`
- `submodules/diff-surfel-rasterization-shadow`
- `submodules/surfel-texture`
- `submodules/surfel-texture-deferred`

Post-rebuild checks passed:

- all five `_C` extensions import in `mygs`
- no-texture 2-iteration smoke trains/evals/saves `chkpnt2.pth`
- texture 2-iteration smoke trains/evals/saves `chkpnt2.pth`
- `render.py --iteration 2 --skip_train` restores the texture checkpoint and writes `render_eval/iteration_00002/summary.json`

For future CUDA sanity checks, first reread [`README.md`](/d:/RTS/gs2dgs/README.md)
and this handoff, then inspect current Python/conda/GPU processes before running
rebuild or smoke commands.

## What to do next

If moving to a new environment:

1. confirm local CUDA extensions under [`submodules/`](/d:/RTS/gs2dgs/submodules) import correctly
2. run a short no-texture smoke test
3. then resume the `100k no-texture vs texture` comparison

If focusing on quality:

- no-texture baseline is the reference
- texture work should target `29.37 -> 29.62+`
- the next likely place to improve is texture-side relighting expression, not the old `3ch` path
