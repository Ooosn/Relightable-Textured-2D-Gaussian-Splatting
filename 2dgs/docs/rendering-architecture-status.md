# Rendering Architecture Status (mBRDF / Shadow / Texture)

Last update: 2026-04-08

## 1) Target Architecture (agreed)

The renderer is organized into two explicit stages:

1. mBRDF stage:
- Direct mode: compute 3-channel `rgb = base * shadow + other`.
- Deferred mode: compute 7-channel `[base(3), decay(1), other(3)]`.
- This stage is independent from the rasterization backend choice.

2. Rasterization stage:
- Texture-aware rasterizer(s) receive `shs` or `colors_precomp` and texture tensors.
- `use_textures` controls whether texture sampling is active inside CUDA.
- No Python-side early fusion of tex + mBRDF into a fake 3-channel substitute for deferred mode.

## 2) What is done

### 2.1 Shadow pass refactor in `gaussian_renderer/__init__.py`
- `_compute_shadow_pass(...)` now returns only:
  - `per_point_shadow` (`[N,1]`)
  - `per_uv_shadow` (`[N,R,R]` or `None`)
  - light matrices
- Removed extra image-level shadow rendering from the main render path.
- `_render_shadow_image(...)` function was removed.
- Output dictionary no longer exposes `"shadow"`; only `"pre_shadow"` is kept.

### 2.2 Shadow usage in mBRDF stage
- mBRDF uses:
  - `per_uv_shadow` as UV-level hint in textured phasefunc path.
  - `per_point_shadow` as per-Gaussian hint in no-texture path.
- Shadow is now treated as a hint/decay input and not a separately splatted final image channel.

### 2.3 7-channel deferred kernel branch (no-texture path)
- Deferred rasterization branch has been adapted to 7 channels in its module chain.
- The intended semantic layout is `[base(3), decay(1), other(3)]`.

### 2.4 Script compatibility update
- `scripts/mbrdf_smoke.py` reads `"pre_shadow"` instead of `"shadow"`.

## 3) What is NOT done yet

### 3.1 Full 7-channel deferred texture path integration
- A complete and validated "deferred + texture + 7ch" end-to-end route is still pending final consolidation/testing.
- Current code contains transitional behavior and needs one final pass to lock exact routing rules.

### 3.2 Renderer routing cleanup
- Some routing decisions still use `getattr(...)` style fallback checks.
- The agreed contract is explicit flags (e.g. `pipe.enable_texture`, `pc.use_textures`) without ambiguous fallback logic.

### 3.3 Documentation/code consistency polish
- Some inline docstrings/comments still describe older wording (e.g. legacy screen-space shadow wording) and should be aligned with current behavior.

## 4) Current practical behavior (as of this update)

- Shadow pass: computes per-point/per-uv shadow statistics only.
- Main output:
  - `"render"` remains the final rendered image path output.
  - `"pre_shadow"` exposes per-Gaussian shadow factors.
  - `"shadow"` image key is removed.
- Textured rendering helpers are present in `gaussian_renderer/textured.py` and currently represent an in-progress consolidation point for routing.

## 5) Next priorities

1. Freeze one canonical routing table in `render(...)`:
- Inputs: `deferred`, `use_mbrdf`, `pc.use_textures`, `pipe.enable_texture`, `override_color`.
- Outputs: exact backend call + exact tensor contract.

2. Finalize deferred+texture 7-channel path:
- Ensure one consistent CUDA API contract for `[base, decay, other]` with texture enabled.
- Verify forward/backward gradients and training stability.

3. Remove transitional branches:
- Eliminate fallback `getattr(...)` checks where explicit config fields are guaranteed.
- Clean stale comments/docstrings to reflect final behavior exactly.

## 6) Quick checklist for verification

- Smoke:
  - `scripts/mbrdf_smoke.py` runs and prints `pre_shadow`.
- Render dictionary:
  - no `"shadow"` key access remains in training/viewing scripts.
- Shadow pass:
  - textured case returns `per_uv_shadow` and valid `per_point_shadow`.
  - non-textured case returns `per_point_shadow` and `per_uv_shadow=None`.
- Deferred:
  - 7-channel path shape checks pass.
  - final composition semantics match `base * decay + other`.
