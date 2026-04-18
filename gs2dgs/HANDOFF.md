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

## Save / eval behavior

`train.py` now always appends the final iteration to:

- `save_iterations`
- `test_iterations`
- `checkpoint_iterations`

That means future runs should automatically produce:

- final `point_cloud`
- final `test` eval
- final `chkpnt*.pth`

## What to do next

If moving to a new environment:

1. confirm local CUDA extensions under [`submodules/`](/d:/RTS/gs2dgs/submodules) import correctly
2. run a short no-texture smoke test
3. then resume the `100k no-texture vs texture` comparison

If focusing on quality:

- no-texture baseline is the reference
- texture work should target `29.37 -> 29.62+`
- the next likely place to improve is texture-side relighting expression, not the old `3ch` path
