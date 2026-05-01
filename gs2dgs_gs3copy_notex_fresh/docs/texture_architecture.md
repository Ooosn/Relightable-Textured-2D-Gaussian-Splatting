# Texture Architecture Notes

Last audited: 2026-05-02.

This file is the working source of truth for the current texture branch.

## Default Route

The default texture mode is:

```text
texture_effect_mode = uvshadow_specular_residual
texture_start_iter = 30000
texture_resolution = 4
texture_dynamic_resolution = optional
texture_rtg_enabled = optional
```

The default route keeps GS3's Gaussian-level relighting model and adds only the
texture-space degrees of freedom we currently want:

```text
Gaussian level:
  kd/ks base material state
  local_q, alpha_asg, ASG basis
  neural_material and neural_phasefunc
  other_effects RGB

Texture level:
  tex_color      -> absolute per-UV kd/albedo, not a multiplier
  tex_specular   -> per-UV specular gain residual
  per_uv_shadow  -> per-UV raw shadow residual input
```

`tex_alpha` still exists for checkpoint/layout compatibility and future shadow
experiments, but default rendering and default shadow do not use it.

## Formula

For each texel:

```text
diffuse_uv  = kd_uv / pi
specular_uv = ks_g * ASG_g(wi, wo, local_q_g, alpha_asg_g) * gain_uv
base_uv     = (diffuse_uv + specular_uv) * cosTheta_g * dist_2_inv_g
shadow_uv   = clamp(decay_g + raw_shadow_uv - point_shadow_g, 0, 1)
other_g     = neural_other_g * dist_2_inv_g
```

Important consequences:

- `kd_uv` is a real per-texel value, not `2 * texture_color * kd_g`.
- Per-UV normal/local_q is not used in the default route.
- The neural phase function is evaluated per Gaussian, not per texel.
- `other_effects` stays Gaussian-level.
- The final deferred texture CUDA path samples only 4 texture channels:

```text
texture channels:  base RGB + shadow = 4ch
Gaussian channels: other RGB         = channels 4:7
screen compose:    RGB = base * shadow + other
```

## Alpha

Default:

```text
texture_render_use_alpha = False
texture_shadow_use_alpha = False
texture_shadow_output_uv = True
texture_shadow_alpha_bilinear = False
```

If shadow alpha is enabled later, it should use nearest/binning by default, not
bilinear interpolation. The bilinear path is kept behind
`--texture_shadow_alpha_bilinear` only as an explicit debug/ablation switch.

## Optimizer State

Texture tensors:

- `tex_color`: LR follows Gaussian `kd_lr`.
- `tex_specular`: LR follows Gaussian ASG schedule times
  `texture_specular_lr_scale`.
- `tex_alpha`: optimizer group exists but receives no gradient in the default
  route because alpha is not used.
- `tex_normal`: not created or optimized in the default route.

Texture start from a no-texture 30k checkpoint:

- `tex_color` copies Gaussian `kd`.
- `tex_specular` starts at zero, so `gain_uv = exp(2 * tanh(0)) = 1`.
- `tex_alpha` is initialized near one but is inactive by default.

Gaussian densification:

- duplicated texture values are copied to child Gaussians;
- newly appended texture Adam moments are zeroed, matching GS3-style behavior
  for new Gaussian parameters.

Texture pixel refinement / RTG:

- refined texel values inherit from the parent chart via the resize path;
- Adam moments are resized and scaled by `texture_rtg_optimizer_state_scale`
  (default `0.5`);
- RTG accumulators are reset after each refine window.

## Checkpoint/Save/Load

Full checkpoints preserve:

- texture tensors;
- optimizer groups and moments;
- dynamic texture dims;
- RTG score buffers;
- `texture_effect_mode`;
- RTG optimizer state scale.

Point-cloud appearance saves preserve texture and mBRDF tensors needed for
offline rendering, but full training resume should use `chkpnt*.pth`.

## Runtime Audit Logs

Training writes a structured texture audit stream to:

```text
MODEL_PATH/convergence/texture_architecture.jsonl
```

Rows are emitted at:

- `train_start`
- `texture_enabled`
- every RTG refine or skip event
- every full checkpoint save

Each row records:

- texture mode and active route;
- static/dynamic layout;
- Gaussian count, total texels, average texels per Gaussian;
- resolution histogram;
- RTG schedule and gate settings;
- tensor shapes for `tex_color`, `tex_alpha`, `tex_specular`, `tex_normal`,
  `texture_dims`, and `rtg_score`;
- optimizer groups, learning rates, and Adam moment tensor shapes;
- pipeline alpha flags.

The same state is available in code through:

```python
gaussians.texture_architecture_state()
gaussians.texture_architecture_log_string()
```

Use this file when checking whether a run is actually using the intended
algorithm. The default clean route should report:

```text
route kd_uv+shadow_residual_uv+specular_residual_uv
deferred_tex_ch 4
tex_normal present false
texture_render_use_alpha false
texture_shadow_use_alpha false
```

## Files To Read First

- `arguments/__init__.py`
- `gaussian_renderer/texture_branch.py`
- `gaussian_renderer/textured.py`
- `scene/gaussian_model_native_2dgs.py`
- `submodules/surfel-texture-deferred/cuda_rasterizer/texture_utils.cuh`
- `submodules/diff-surfel-rasterization-shadow/cuda_rasterizer/forward.cu`
