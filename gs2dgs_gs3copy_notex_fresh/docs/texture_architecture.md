# Texture Architecture Notes

This document is the working source of truth for the current
`gs2dgs_gs3copy_notex_fresh` texture branch. Read this before discussing or
changing texture training.

Last audited: 2026-04-30.

## Current Goal

The current texture line is not a generic rewrite of GS3 material modeling.
It keeps GS3's low-frequency Gaussian-level relighting structure and adds
texture-space high-frequency degrees of freedom.

Current default texture mode:

```text
texture_effect_mode = per_uv_micro_normal
texture_start_iter = 30000
texture_resolution = 4
texture_dynamic_resolution = optional
texture_rtg_enabled = optional
asg_mlp = False
```

The intended interpretation is:

```text
Gaussian level: kd, ks, alpha_asg, ASG basis, neural_material, shadow/other MLP
Texture level: tex_color, tex_alpha, tex_normal/local_q
```

`tex_color` and `tex_alpha` are default texture-rasterizer channels and are
expected to exist. Do not call their existence a bug.

Current per-step texture phase is intentionally unchunked: per-UV shadow,
normal/ASG, shadow MLP, and other-effect MLP are evaluated as full tensors.
Only RTG score/refine keeps `texture_rtg_chunk_texels`, because that is a
separate refinement/aggregation path rather than the normal render step.

## Main Files

- `arguments/__init__.py`
  - default training and texture arguments.
- `train.py`
  - texture start scheduling, checkpointing, RTG calls.
- `scene/gaussian_model_native_2dgs.py`
  - texture tensors, optimizer groups, checkpoint state, texture densify/RTG.
- `gaussian_renderer/texture_branch.py`
  - texture-aware shadow pass and mBRDF formula.
- `gaussian_renderer/textured.py`
  - final texture rasterization wrapper.
- `scene/neural_phase_function.py`
  - shadow/other/asg correction network. Current runs use `asg_mlp=False`.

## Parameters And Meaning

### Gaussian-level mBRDF parameters

These are per Gaussian:

- `kd`: diffuse BRDF coefficient.
- `ks`: specular coefficient.
- `alpha_asg`: Gaussian-level ASG sharpness/weight parameters.
- `local_q`: Gaussian-level local frame quaternion.
- `neural_material`: learned latent vector for neural phase function.
- `asg_func`: analytic ASG basis/function.
- `neural_phasefunc`: neural shadow and other-effect correction.

Important: `local_q` is independent learned material/frame state. It is not
the 2DGS spatial rotation tensor.

### Texture-level parameters

These are per texel when texture training is enabled:

- `_tex_color`
  - Stored as logits.
  - Read through `sigmoid`.
  - The renderer uses `2.0 * get_texture_color`, so initial value `0.5` is a
    neutral multiplier of `1.0`.
- `_tex_alpha`
  - Stored as logits.
  - Used by the texture rasterizer and shadow pass as a texture-space opacity
    or coverage channel.
- `_tex_normal`
  - Current meaning: per-texel `local_q` quaternion.
  - It is not a residual normal.
  - It is not scaled by `texture_normal_scale` in the current
    `per_uv_micro_normal` formula.
- `_tex_specular`
  - Used by the legacy `uv_specular_gain` mode.
  - In current `per_uv_micro_normal`, it is not used in the formula.
- `_texture_dims`
  - Dynamic texture atlas metadata `[height, width, offset]` per Gaussian.
- `_rtg_score`
  - Per-Gaussian score accumulated from texture gradients for RTG refinement.

## Texture Start At 30k

When running with a 30k no-texture checkpoint and:

```text
--use_textures --texture_start_iter 30000
```

`train.py` does this:

1. Load checkpoint.
2. If `first_iter <= texture_start_iter`, call `defer_texture_training()`.
3. At `iteration > texture_start_iter`, call `enable_texture_training(opt)`.

So for a checkpoint at iteration 30000, texture first becomes active at
iteration 30001.

At enable time:

- texture color is initialized as neutral multiplier:

```text
tex_color sigmoid value = 0.5
2.0 * tex_color = 1.0
```

- `tex_normal` is initialized by copying Gaussian `local_q`.
- `tex_normal` Adam state is initialized by copying/expanding Gaussian
  `local_q` Adam state.

This was verified from checkpoints:

```text
no-refine chkpnt30001:
  tex_normal shape = [73231, 4, 4, 4]
  max(abs(tex_normal - local_q expanded)) = 0.0
  tex_normal optimizer exp_avg/exp_avg_sq shape = [73231, 4, 4, 4]

RTG dynamic chkpnt30001:
  tex_normal shape = [1171696, 4]
  max(abs(tex_normal - repeat_interleave(local_q))) = 0.0
  tex_normal optimizer exp_avg/exp_avg_sq shape = [1171696, 4]
```

By 40000, `tex_normal` differs from Gaussian `local_q`, which means the texel
quaternions are training independently.

## per_uv_micro_normal Formula

There are static and dynamic texture paths, but their formulas are equivalent.

In `gaussian_renderer/texture_branch.py`, for each texel:

1. Read per-texel local quaternion:

```python
texture_local_q = gau.get_texture_local_q
```

2. Convert it to a local frame:

```python
local_axes_uv = _frame_from_texture_local_q(...)
```

3. Project light/view directions into this per-texel frame:

```python
wi_local_uv = _project_to_local(wi_uv, local_axes_uv)
wo_local_uv = _project_to_local(wo_uv, local_axes_uv)
```

4. Compute per-texel cosine and ASG:

```python
cos_theta_uv = _NdotWi(local_axes_uv[:, :, 2], wi_uv, ...)
asg_uv = gau.asg_func(
    wi_local_uv,
    wo_local_uv,
    gau.get_alpha_asg[texel_ids],
    asg_scales,
    asg_axises,
)
```

5. Call `neural_phasefunc`:

```python
decay_flat, other_effects_flat, asg_3_flat, _ = gau.neural_phasefunc(
    wi_uv,
    wo_uv,
    gau.get_xyz[texel_ids],
    gau.get_neural_material[texel_ids],
    hint=per_uv_shadow[...],
    asg_1=asg_uv,
    asg_mlp=gau.asg_mlp,
)
```

Important current fact:

```text
asg_mlp = False
```

Therefore `neural_phasefunc` does not run the ASG MLP. It returns:

```python
asg_3 = asg_1
```

So the current normal/specular data flow is:

```text
tex_normal/local_q
  -> local_axes_uv
  -> wi_local_uv / wo_local_uv / cos_theta_uv
  -> analytic ASG asg_uv
  -> specular = ks * asg_uv
```

Do not describe this as "normal goes through neural_phasefunc MLP". That is
incorrect.

What the neural phase function still does per texel:

- `shadow_func`: uses `per_uv_shadow` as hint and outputs corrected shadow
  decay.
- `other_effects_func`: outputs other effects.
- `asg_func` MLP: not used unless `asg_mlp=True`.

6. Final current formula:

```text
diffuse = kd * (2 * tex_color) / pi
specular = ks * asg_uv          # because asg_mlp=False
basecolor = (diffuse + specular) * cos_theta_uv * dist_2_inv
shadow = decay_flat
other_effects = other_effects_flat * dist_2_inv
```

With `fix_labert=True`, specular is forced to zero.

## Shadow Pass

The texture-aware shadow pass computes `per_uv_shadow` before the mBRDF
formula.

`tex_alpha` is passed to the shadow rasterizer. In the shadow CUDA path, sampled
texture alpha affects footprint/opacity in shadow accumulation.

This is expected behavior for the texture branch.

## Final Texture Rasterization

`gaussian_renderer/textured.py` passes:

```python
texture_color = mbrdf_colors
texture_alpha = pc.get_texture_alpha
texture_dims = pc.get_texture_dims if dynamic else None
```

The deferred texture rasterizer composites the 7-channel mBRDF texture result.
`tex_alpha` also participates in final textured rasterization. This is part of
the current texture-rasterizer design.

## Optimizer State Inheritance

### Texture start

When texture is enabled at 30001:

- `_tex_normal` values are copied from Gaussian `local_q`.
- `_tex_normal` Adam state is copied from Gaussian `local_q` by
  `_texture_local_q_optimizer_state()`.
- `_tex_color` and `_tex_alpha` are new texture parameters, so they start with
  fresh Adam state.

### Static Gaussian densify

For static textures:

- `densify_and_split()` and `densify_and_clone()` copy:
  - `_tex_color`
  - `_tex_alpha`
  - `_tex_specular`
  - `_tex_normal`
- `_static_texture_state_extensions()` copies Adam state for:
  - `tex_color`
  - `tex_alpha`
  - `tex_specular`
  - `tex_normal`

Gaussian-level mBRDF values such as `kd`, `ks`, `alpha_asg`, `local_q`,
`neural_material` are copied as values during Gaussian densification. Their
optimizer state is not currently specially inherited through
`state_extensions`; missing extension state falls back to zeros.

In the main 30k texture-start experiments, Gaussian densification is already
finished, so this does not affect the 30k texture activation path.

### Dynamic Gaussian prune/append

For dynamic textures:

- `_prune_dynamic_textures()` gathers kept texels and gathers optimizer state.
- `_append_dynamic_textures_from_mask()` appends selected texels and appends
  optimizer state.

So dynamic Gaussian clone/split/prune preserves texture tensor state and
texture Adam state.

### RTG texture refinement

RTG is texture-chart refinement, not Gaussian densification.

Current implementation:

- unchanged charts copy values exactly.
- refined charts are upsampled:
  - `tex_color`: sigmoid-space bilinear, then inverse sigmoid.
  - `tex_alpha`: sigmoid-space bilinear, then inverse sigmoid.
  - `tex_specular`: raw bilinear.
  - `tex_normal`: raw bilinear.
- optimizer state for refined charts is also bilinear-resized.

Important: for `tex_normal`, RTG refinement currently uses bilinear upsampling,
not exact 2x2 parent-pixel replication. If the desired rule is "each refined
child texel inherits exactly the parent texel value and Adam state", this is a
future change.

RTG scoring starts only after `texture_rtg_refine_from_iter`:

```python
if iteration <= texture_rtg_refine_from_iter:
    reset_texture_rtg_scores()
    return
```

After each RTG refine log, `train.py` resets RTG scores.

## Checkpoint State

Full checkpoint `capture()` stores:

- Gaussian state.
- optimizer state.
- mBRDF state.
- texture state:
  - `texture_dynamic_resolution`
  - `texture_min_resolution`
  - `texture_max_resolution`
  - `texture_dims`
  - `rtg_score`
  - `tex_specular`
  - `tex_normal`
  - `texture_normal_scale`

`restore()` supports:

- old no-texture checkpoints.
- static texture checkpoints.
- dynamic texture checkpoints.
- requested static-to-dynamic conversion when resuming with
  `--texture_dynamic_resolution`.
- partial optimizer restoration by group name when exact optimizer loading
  fails.

## Current Experiment Facts

Current Fish runs started from:

```text
/home/wangyy/data_download/gsrelight_runs/gs2dgs_gs3copy_texture_validation/
  clean_notex30k_pretexture_20260427/chkpnt30000.pth
```

Both exact-local-q texture experiments use:

```text
--texture_start_iter 30000
--texture_effect_mode per_uv_micro_normal
--texture_resolution 4
--densify_until_iter 30000
--densify_grad_threshold 0.00015
--asg_mlp False
```

Observed evals so far:

```text
no-refine:
  30001 test PSNR 29.2353
  40000 test PSNR 30.1198
  50000 test PSNR 30.4869

RTG refine:
  30001 test PSNR 29.2353
  40000 test PSNR 30.2428
```

The 30001 equality between no-refine and refine verifies that dynamic atlas
conversion did not change initial rendering at texture start.

## Known Caveats / Cleanup Items

1. `texture_normal_scale` is currently legacy metadata.
   - It is saved/restored.
   - It is not used by `per_uv_micro_normal`.
   - This is because current normal texture stores full local-q quaternions,
     not residual offsets.

2. `_tex_specular` is currently unused in `per_uv_micro_normal`.
   - It belongs to `uv_specular_gain`.
   - The optimizer group may still exist, but there should be no gradient in
     `per_uv_micro_normal`.

3. `asg_mlp=False` in the current runs.
   - `asg_uv` is analytic ASG from `gau.asg_func`, not a neural ASG MLP result.
   - `neural_phasefunc(..., asg_1=asg_uv, asg_mlp=False)` returns `asg_uv`
     unchanged as `asg_3`.

4. Runtime paths must not fall back to sibling `gs2dgs/submodules`.
   - The active Python path setup uses this tree first and the sibling `2dgs`
     vendored surfel modules as the temporary CUDA source.
   - Do not add `../gs2dgs/submodules` fallbacks back into renderer imports.
   - If this copy must be fully standalone, vendor the surfel texture/shadow
     submodules into this directory and keep this rule unchanged.

5. RTG normal refinement currently uses bilinear upsampling.
   - This is a design choice in the current code, not an exact parent-pixel
     copy.
   - If exact inheritance is desired, change `_resize_dynamic_textures()` and
     `_resize_dynamic_optimizer_state()` for `tex_normal` to nearest/replicate.

6. `--texture_freeze_gaussian_densify` is an experiment control, not a default.
   - Default is off, so no-texture and historical texture runs keep their normal
     densification schedule.
   - Enable it when testing whether texture-space parameters should absorb
     high-frequency relighting after `texture_start_iter` without continued
     Gaussian clone/split/prune changes.

## Common Mistakes To Avoid

- Do not say `tex_normal` goes through the neural phase MLP. It does not.
- Do not say ASG MLP is active unless `asg_mlp=True` in config/checkpoint.
- Do not treat `tex_color` and `tex_alpha` as accidental pollution; they are
  default texture-rasterizer channels.
- Do not judge texture-start correctness from memory. Check `chkpnt30001.pth`:
  `tex_normal` should exactly equal expanded/repeated Gaussian `local_q`.
- Do not compare texture and no-texture runs unless source path, data device,
  view count, densification threshold, checkpoint, and test iterations match.
