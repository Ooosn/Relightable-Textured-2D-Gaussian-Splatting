# Rendering Architecture Status (mBRDF / Shadow / Texture)

Last update: 2026-04-08

## 1) Canonical Architecture

The renderer is organized into three explicit stages:

1. Shadow statistics stage
- Light-space surfel rasterization computes only shadow hints.
- Outputs:
  - `per_point_shadow`: `[N, 1]`
  - `per_uv_shadow`: `[N, R, R]` or `None`
- No image-space `"shadow"` render target is produced anymore.

2. Appearance payload stage
- This stage converts scene state into the payload consumed by rasterization.
- Direct mode payload:
  - `rgb = basecolor * shadow + other_effects`
  - shape is either `[N, 3]` or `[N, 3, R, R]`
- Deferred mode payload:
  - `[base(3), decay(1), other(3)]`
  - shape is either `[N, 7]` or `[N, 7, R, R]`

3. Rasterization stage
- Rasterization consumes one of:
  - SH features via `shs`
  - precomputed per-Gaussian payload via `colors_precomp`
  - per-UV payload via `texture_color`
- Texture sampling is controlled only by `pc.use_textures`, and the same flag is passed down to the backend as `use_textures`.
- Deferred rasterization returns a 7-channel image which is composed in Python as:
  - `final_rgb = base_img * decay_img + other_img`

## 2) Current Implemented Behavior

### 2.1 Shadow pass
- Implemented in `gaussian_renderer/__init__.py`.
- With textures:
  - shadow backend returns `per_uv_shadow`
  - `per_point_shadow` is the UV mean
- Without textures:
  - shadow backend returns `per_point_shadow`
  - `per_uv_shadow=None`
- Main render output only exposes `"pre_shadow"`, not `"shadow"`.

### 2.2 mBRDF payload construction
- No-texture mBRDF path:
  - `basecolor`: `[N, 3]`
  - `shadow`: `[N, 1]`
  - `other_effects`: `[N, 3]`
- Texture-aware mBRDF path:
  - `basecolor`: `[N, 3, R, R]`
  - `shadow`: `[N, 1, R, R]`
  - `other_effects`: `[N, 3, R, R]`
- Phase-0 warmup now stays shape-consistent:
  - textured models use texture albedo as `basecolor`
  - non-textured models use `kd / pi`

### 2.3 Canonical raster routing

Inputs that determine routing:
- `deferred`
- `override_color`
- `pc.use_mbrdf`
- `pc.use_textures`

Routing rules:
- `override_color` has highest priority for payload selection.
- If `pc.use_mbrdf=True` and `override_color is None`, payload comes from mBRDF.
- If no explicit payload exists and `deferred=False`:
  - non-texture models fall back to SH via `shs`
  - texture models fall back to `pc.get_texture_color`
- If `deferred=True`, an explicit payload is required.
  - SH-only deferred fallback is not supported.

Raster payload mapping:
- Direct + per-Gaussian payload `[N, 3]`
  - routed to `colors_precomp`
- Direct + per-UV payload `[N, 3, R, R]`
  - if `pc.use_textures=True`: routed to `texture_color`, with zero `colors_precomp`
- Deferred + per-Gaussian payload `[N, 7]`
  - routed to `colors_precomp`
- Deferred + per-UV payload `[N, 7, R, R]`
  - if `pc.use_textures=True`: routed to `texture_color`, with zero `colors_precomp`

### 2.4 Deferred composition semantics
- Deferred backend stores 7 channels as:
  - `base(3), decay(1), other(3)`
- Python composes the final image after rasterization.
- Deferred background is packed as:
  - `[0, 0, 0, 0, bg_rgb]`
- This avoids background decay cross-terms during final composition.

## 3) CUDA / Wrapper Status

### 3.1 Direct texture backend
- Python and backend now both use the same `use_textures` flag.
- Texture-only rendering uses zero `colors_precomp` to avoid accidental double-counting with SH/precomputed colors.

### 3.2 Deferred texture backend
- Wrapper and CUDA signatures now expose `use_textures`.
- Forward/backward texture sampling paths are now guarded by:
  - `use_textures`
  - `texture_color != nullptr`
  - `texture_resolution > 0`
- This matches the direct backend switch semantics instead of relying on empty tensors implicitly.

### 3.3 Important kernel contract
- Both texture backends add:
  - `feature/color_precomp contribution + sampled texture contribution`
- Therefore pure texture payloads must be paired with zero `colors_precomp`.
- Python now enforces that convention for texture-driven paths.

## 4) Known Constraints

- Deferred mode does not support SH-only rendering.
- `override_color` is treated as an explicit raster payload.
- A 3-channel deferred override is lifted to 7 channels as:
  - `[0, 0, 0, 0, override_rgb]`
- The returned `"render"` image is always final RGB.
  - raw deferred 7-channel buffers are not currently exposed in the public render dict.

## 5) Files To Treat As Ground Truth

- `gaussian_renderer/__init__.py`
  - shadow statistics
  - mBRDF payload construction
  - top-level render routing inputs
- `gaussian_renderer/textured.py`
  - canonical routing table
  - direct/deferred payload mapping
  - deferred final composition
- `submodules/surfel-texture/surfel_texture/__init__.py`
  - direct backend Python wrapper
- `submodules/surfel-texture-deferred/surfel_texture_deferred/__init__.py`
  - deferred backend Python wrapper
- `submodules/surfel-texture/cuda_rasterizer/*`
- `submodules/surfel-texture-deferred/cuda_rasterizer/*`

## 6) Practical Verification Checklist

- Direct, no texture, no mBRDF:
  - falls back to SH rendering
- Direct, texture model:
  - uses zero `colors_precomp` plus texture sampling
- Deferred, no texture:
  - accepts `[N, 7]` payload and composes RGB in Python
- Deferred, texture model:
  - accepts `[N, 7, R, R]` payload and composes RGB in Python
- Shadow:
  - `"pre_shadow"` exists
  - `"shadow"` image key does not
