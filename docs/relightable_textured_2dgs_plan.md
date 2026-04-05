# Relightable Textured 2DGS Plan

## 1. Goal

We want to evolve this repository from a geometry-aware 2D Gaussian Splatting baseline into a
relightable textured 2DGS system.

The target system should have the following properties:

- Keep 2DGS as the geometric backbone.
- Use each Gaussian's local plane as a texture/material parameterization domain.
- Decouple geometry from high-frequency appearance.
- Support relighting under novel view and novel light conditions.
- Provide a clean path toward more graphics-aware effects such as normal mapping, roughness,
  anisotropy, procedural textures, and sharper shadows.

In short:

> Geometry stays in 2DGS surfels. Appearance moves into texture space. Lighting is handled by a
> mix of explicit graphics reasoning and lightweight learned residuals.

## 2. Why This Direction

This direction is attractive for several reasons.

- 2DGS already gives us stable local planes in 3D space, which is a natural home for texture-space
  graphics operations.
- Moving appearance into texture space should let us represent higher-frequency detail without
  exploding the Gaussian count.
- Texture-space shading can produce sharper and more structured shadows than per-primitive color
  alone.
- Once a Gaussian owns a local parameterization, we can attach richer graphics quantities to it:
  albedo, alpha, normal map, roughness, specular parameters, anisotropy, and learned features.
- Compared with a pure black-box neural appearance model, a texture-space representation is more
  interpretable and gives us better leverage for graphics priors.

## 3. References and Their Roles

This plan is informed by four sources with different purposes.

- Current repo: official 2DGS implementation. It provides the geometric representation and
  perspective-correct surfel rasterization.
- GS3: useful mainly for relighting structure, especially light-space splatting and shadow handling.
- RNG: useful for view/light-conditioned neural appearance, shadow cue design, and staged training.
- LGTM: useful for the textured Gaussian idea and geometry/appearance decoupling, even though its
  feed-forward setup is not our current goal.

We are **not** trying to reproduce LGTM end-to-end at first, since this repo is a per-scene
optimization codebase, not a feed-forward predictor.

## 4. Guiding Principles

- Preserve the current 2DGS geometry pipeline as much as possible in the first iteration.
- Build the system in stages, with a useful checkpoint after each stage.
- Prefer explicit, controllable graphics quantities before pushing complexity into an MLP.
- Make the first textured version work for static novel view synthesis before adding relighting.
- Keep save/load and densification behavior well-defined from the start.

## 5. Non-Goals for the First Version

The following are intentionally out of scope for the first milestone.

- A full feed-forward LGTM-style architecture.
- A full inverse-rendering decomposition with physically complete BRDF recovery.
- Perfect environment-light relighting.
- A large monolithic neural renderer that directly replaces the current rasterization logic.

## 6. Proposed Representation

### 6.1 Geometry

Geometry remains standard 2DGS.

Each Gaussian keeps:

- `xyz`
- `scaling` with 2 in-plane scales
- `rotation`
- `opacity`

The local plane and its `(u, v)` coordinates are already implicit in the current rasterizer through
ray-splat intersection.

### 6.2 Appearance and Material

We will extend each Gaussian with texture-space parameters.

#### Stage 1: Textured appearance MVP

- `tex_color`: per-Gaussian RGB texture
- `tex_alpha`: per-Gaussian alpha texture

Suggested starting shape:

- `tex_color`: `[P, 3, T, T]`
- `tex_alpha`: `[P, 1, T, T]`

with `T = 4` or `T = 8`.

#### Stage 2: Relightable material representation

After the textured MVP is stable, extend toward:

- `tex_albedo`
- `tex_feat`
- optional `tex_normal_ts`
- optional `tex_roughness`
- optional `tex_specular`
- optional anisotropy parameters

### 6.3 Shading Philosophy

The long-term shading design should be:

- explicit direct-light term for interpretable effects
- optional neural residual for effects not captured analytically

Conceptually:

`final_color = direct_lighting(material, normal, view, light, visibility) + residual`

This is preferable to a fully black-box radiance decoder because it keeps the system grounded in
graphics reasoning while still allowing flexibility.

## 7. Rendering Pipeline We Want

For a camera ray hitting a Gaussian plane:

1. Compute local `(u, v)` coordinates from the ray-splat intersection.
2. Sample per-Gaussian textures at `(u, v)` using bilinear sampling.
3. Use sampled alpha to modulate the primitive contribution.
4. Use sampled color or material features to compute shading.
5. Alpha-blend contributions in the existing 2DGS order.

For relighting, the shading inputs will later include:

- view direction
- light direction
- light intensity or falloff
- optional visibility or shadow cue
- optional tangent-space normal information

## 8. Development Roadmap

### Milestone 0: Baseline Freeze and Interface Audit

Purpose:

- Stabilize the current baseline before introducing new parameters.

Tasks:

- Document the current render path and tensor interfaces.
- Identify all places where Gaussian attributes are created, cloned, split, saved, and loaded.
- Decide how extra appearance tensors will be stored on disk.

Exit criteria:

- We have a written list of touched files and a save/load strategy.

### Milestone 1: Textured 2DGS MVP

Purpose:

- Add texture-space appearance without introducing relighting yet.

Tasks:

- Add `tex_color` and `tex_alpha` to `GaussianModel`.
- Decide on texture resolution hyperparameters.
- Extend densification and cloning so texture tensors are copied correctly.
- Add a sidecar save/load format for texture tensors.
- Extend the rasterizer interface to accept texture tensors.
- Implement bilinear texture sampling in CUDA using existing `(u, v)` values.
- Replace or bypass SH-based color with sampled texture color.
- Multiply alpha by sampled texture alpha.
- Train and evaluate on the current static-view setup.

Expected benefit:

- Higher-frequency appearance tied to local Gaussian planes.

Exit criteria:

- Static novel view synthesis works with texture-space appearance.
- The model can save and reload all new parameters.
- Densification does not break the new tensors.

### Milestone 2: Relightable Appearance

Purpose:

- Make appearance depend on view and light.

Tasks:

- Extend the dataset and camera structures to carry light metadata.
- Replace plain `tex_color` usage with a material-oriented representation.
- Introduce a small shading decoder or residual network.
- Start with a simple combination such as:
  - diffuse or albedo texture
  - optional learned feature texture
  - analytic light falloff
  - neural residual for complex transport
- Add training losses for relighting supervision.

Exit criteria:

- The same optimized scene can render under changed light position or direction.
- Appearance remains stable under novel views.

### Milestone 3: Shadow Pipeline

Purpose:

- Improve shadow sharpness and light transport consistency.

Tasks:

- Add a light-space pass inspired by GS3.
- Evaluate whether a direct light-space visibility pass is sufficient, or whether an RNG-style
  shadow cue and refinement path is needed.
- Feed visibility or shadow cue into the shading stage.
- Compare a pure raster shadow pass against cue-conditioned shading.

Exit criteria:

- Shadows are visibly sharper and more stable than a no-shadow baseline.
- The shadow mechanism integrates cleanly with the texture-space material path.

### Milestone 4: Graphics-Rich Extensions

Purpose:

- Exploit the value of the local plane representation.

Possible extensions:

- tangent-space normal maps
- roughness maps
- specular maps
- anisotropic shading aligned with local tangent directions
- procedural texture generation
- learned or analytic micro-shadowing
- texture-space filtering or mipmapping

Exit criteria:

- At least one material-space effect beyond plain color and alpha is demonstrated.

## 9. Repository Touchpoints

The most important code areas are listed below.

### Core geometry and state

- `scene/gaussian_model.py`
- `scene/__init__.py`

Responsibilities:

- new learnable tensors
- optimizer parameter groups
- clone/split/prune behavior
- save/load behavior

### Camera and dataset

- `scene/cameras.py`
- `scene/dataset_readers.py`
- `utils/camera_utils.py`

Responsibilities:

- lighting metadata
- camera-side relighting inputs

### Python render path

- `gaussian_renderer/__init__.py`
- `train.py`
- `render.py`
- `view.py`

Responsibilities:

- render branch selection
- relighting inputs
- new losses
- visualization and debugging

### CUDA and rasterizer bindings

- `submodules/diff-surfel-rasterization/diff_surfel_rasterization/__init__.py`
- `submodules/diff-surfel-rasterization/ext.cpp`
- `submodules/diff-surfel-rasterization/rasterize_points.h`
- `submodules/diff-surfel-rasterization/rasterize_points.cu`
- `submodules/diff-surfel-rasterization/cuda_rasterizer/forward.cu`
- `submodules/diff-surfel-rasterization/cuda_rasterizer/backward.cu`

Responsibilities:

- texture tensor inputs
- local `(u, v)` sampling
- texture-space alpha and color accumulation
- backward gradients for texture sampling

## 10. Data and Serialization Plan

PLY alone is not a good long-term container for per-Gaussian textures.

Recommended plan:

- keep geometry-compatible scalar attributes in `point_cloud.ply`
- store texture and other dense appearance tensors in a sidecar file such as:
  - `appearance.pt`
  - or `appearance.npz`

A sidecar file is easier to evolve as we add:

- color textures
- alpha textures
- feature textures
- normal maps
- material maps

This also avoids making the PLY schema too brittle.

## 11. Densification and Inheritance Rules

Densification must be explicitly defined for new appearance tensors.

Initial rule set:

- clone: copy textures exactly
- split: duplicate parent textures to children initially
- prune: remove corresponding texture tensors

This simple inheritance policy is enough for the first version and keeps behavior predictable.

Later, we may explore smarter texture initialization for split Gaussians.

## 12. Main Technical Risks

### Risk 1: CUDA complexity

Adding texture sampling inside the rasterizer is the most invasive part.

Mitigation:

- keep the first textured version minimal
- support only one texture format first
- avoid overloading the first implementation with relighting logic

### Risk 2: Memory growth

Per-Gaussian textures can quickly become expensive.

Mitigation:

- start with small `T`
- start with only color and alpha textures
- profile memory before adding feature textures

### Risk 3: Training instability

New tensors plus densification can destabilize optimization.

Mitigation:

- get a static textured baseline working before adding lights and shadows
- start with conservative learning rates for texture tensors

### Risk 4: Dataset mismatch for relighting

Relighting requires reliable light metadata.

Mitigation:

- define the dataset format early
- start from a known, simple point-light training setup

### Risk 5: Too much neuralization too early

If we immediately move everything into an MLP, we may lose interpretability and debugging ability.

Mitigation:

- prefer analytic direct-lighting terms first
- use a neural residual only where it clearly helps

## 13. Recommended First Coding Step

The best first implementation target is:

**Textured 2DGS MVP for static novel view synthesis.**

This step is the best leverage point because it answers the most important engineering question:

> Can the current 2DGS rasterization pipeline support stable per-primitive texture-space sampling
> and training?

If the answer is yes, then relighting and shadow work can be layered on top with much lower risk.

## 14. Immediate Action Items

1. Add `docs/` plan document to the repo.
2. Define the exact tensor shapes for `tex_color` and `tex_alpha`.
3. Extend `GaussianModel` with new tensors and optimizer groups.
4. Design the sidecar serialization format.
5. Extend rasterizer bindings to accept texture inputs.
6. Implement bilinear texture sampling in CUDA.
7. Reproduce static rendering quality with textured appearance before touching relighting.

## 15. Summary

The project should proceed in this order:

1. Textured 2DGS
2. Relightable appearance
3. Shadow pipeline
4. Advanced graphics-aware material effects

This keeps the plan practical, lets us validate the hardest infrastructure piece early, and gives us
a strong base for the more interesting graphics work we actually care about.
