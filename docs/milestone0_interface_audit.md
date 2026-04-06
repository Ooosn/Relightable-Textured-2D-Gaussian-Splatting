# Milestone 0 Interface Audit

This note freezes the current implementation boundary before CUDA texture sampling work begins.

## Current Touchpoints

- `2dgs/scene/gaussian_model.py`
  Owns learnable Gaussian state, optimizer groups, densify/clone/split/prune behavior, and checkpoint capture/restore.
- `2dgs/scene/__init__.py`
  Owns scene-level save/load orchestration for per-iteration artifacts.
- `2dgs/train.py`
  Instantiates `GaussianModel`, drives training, and writes checkpoints and per-iteration saves.
- `2dgs/render.py`
  Loads saved scenes for offline rendering and mesh export.
- `2dgs/view.py`
  Loads saved scenes for visualization.
- `2dgs/gaussian_renderer/__init__.py`
  Python entrypoint for rasterizer invocation.
- `2dgs/submodules/diff-surfel-rasterization/*`
  C++/CUDA binding and rasterizer implementation that will eventually need texture inputs and UV-aware sampling.

## State Added In This Round

The model now has optional texture-state storage, guarded by `--use_textures`.

- `tex_color`: shape `[P, 3, T, T]`
- `tex_alpha`: shape `[P, 1, T, T]`
- `T`: controlled by `--texture_resolution`, default `4`

Initialization policy:

- Fresh scenes initialize `tex_color` from the input point colors.
- Fresh scenes initialize `tex_alpha` from the same base opacity prior used for Gaussian opacity.
- Older checkpoints or PLY-only loads without `appearance.pt` fall back to broadcasting SH DC color and opacity into the texture grids.

## Serialization Decision

Geometry-compatible scalar attributes stay in `point_cloud.ply`.

Dense appearance state is stored in a sidecar file:

- `appearance.pt`

Stored payload:

- `texture_resolution`
- `tex_color`
- `tex_alpha`

This keeps the PLY schema stable while allowing appearance tensors to evolve independently.

## Densification / Inheritance Rules

The first textured implementation uses direct inheritance:

- clone: copy textures exactly
- split: duplicate parent textures to children
- prune: remove matching entries from texture tensors

This rule is now implemented alongside the existing Gaussian tensors in `GaussianModel`.

## Remaining Work For Milestone 1

- Extend `gaussian_renderer` and rasterizer bindings to accept texture inputs.
- Implement local-plane `(u, v)` sampling in CUDA.
- Decide whether the first textured renderer should bypass SH entirely or mix textured color with existing SH fallback.
- Add visualization/debug outputs for texture-aware rendering.
- Validate memory impact for `T=4` and `T=8`.

## Environment Note

No environment-specific build step was required for this audit/infrastructure round. CUDA extension rebuilds should use the external `conda.sh` flow and avoid the old `gs3` environment, per project guidance.
