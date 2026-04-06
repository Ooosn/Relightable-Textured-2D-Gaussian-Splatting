# GS3 / RNG Integration Notes For Relightable Textured 2DGS

## Purpose

This note summarizes what the newly-added `gs3` and `RNG_release` repositories are useful for,
how they differ from the current `2dgs` baseline, and what we should actually borrow from them.

The goal is not to merge these repos wholesale.

The goal is to extract the lowest-risk ideas that help us build:

- textured appearance on top of 2DGS
- relighting support
- a future shadow pipeline

while preserving 2DGS as the geometry backbone.

## Short Take

If we compress the answer:

- `2dgs` should remain the geometric core.
- `gs3` is most valuable for shadow structure and light-space rendering ideas.
- `RNG_release` is most valuable for dataset/light metadata conventions and staged neural relighting.
- Neither repo should be copied end-to-end into `2dgs`.

The correct strategy is:

1. finish textured 2DGS first
2. adopt a minimal light-metadata dataset format inspired by RNG/GS3
3. add a light-space visibility pass inspired mainly by GS3
4. only then add a lightweight neural residual inspired by RNG

## What GS3 Gives Us

## Core value

`gs3` is strongest where it treats shadows as a first-class rendering problem instead of only a
neural appearance problem.

Most relevant files:

- `gs3/STRUCTURE.md`
- `gs3/gaussian_renderer/__init__.py`
- `gs3/gaussian_renderer/shadow_render.py`
- `gs3/scene/cameras.py`

## Practical takeaways

### 1. A dedicated light-space pass

`gs3` explicitly renders from the light point of view before the final view render.

That is the main idea we care about.

For our project, this suggests:

- build a separate light camera from light position plus target scene center
- rasterize Gaussian geometry into light space
- derive a visibility / shadow signal there
- feed that signal into the final shaded render

This matches the shadow milestone in our project plan much better than RNG's more neuralized path.

### 2. Camera objects can carry relighting state

`gs3/scene/cameras.py` puts `pl_pos` on the camera and even allows light/camera refinement.

We probably do not want camera/light optimization in the first version, but the data model is
useful:

- each frame can carry point-light position
- optionally intensity
- future extensions can add direction or environment metadata

### 3. It separates visibility from final appearance

Even though the codebase is large and experimental, the architectural idea is solid:

- first compute shadow / visibility
- then combine with final appearance

This is exactly the separation we want once texture-space material exists in 2DGS.

## What Not To Copy From GS3

- The renderer is much broader than we need right now and mixes multiple branches, rasterizers, and
  experiments.
- The codebase carries a lot of research/debug history that would be expensive to port cleanly.
- It is built around 3DGS-style appearance/material branches rather than our planned texture-space
  2DGS representation.
- Camera/light joint optimization is not a first-milestone need.

So the right move is to copy the idea, not the whole implementation.

## What RNG Gives Us

## Core value

`RNG_release` is strongest where it defines a relighting-oriented training/data interface and a
staged appearance model.

Most relevant files:

- `RNG_release/README.md`
- `RNG_release/gaussian_renderer/__init__.py`
- `RNG_release/scene/cameras.py`
- `RNG_release/scene/dataset_readers.py`
- `RNG_release/arguments/__init__.py`

## Practical takeaways

### 1. A dataset convention we can actually reuse

`RNG_release` expects per-frame lighting metadata in the transform JSON.

Important fields already appear in the reader:

- `pl_pos`
- `pl_intensity`
- camera intrinsics
- camera pose

This is immediately useful for us.

If the upcoming dataset can provide per-image light metadata, we should strongly prefer a
Blender/NRHints-style JSON path over inventing a more ad hoc format.

### 2. Staged training is probably the right relighting recipe

RNG explicitly uses:

- stage 1: forward shading
- stage 2: deferred shading with shadow/depth-related components

That staged mentality is important even if we do not copy the same modules.

For our project the equivalent should be:

- stage A: textured static 2DGS
- stage B: texture-space material plus simple direct lighting
- stage C: add shadow visibility
- stage D: optional neural residual

### 3. Neural appearance should be an add-on, not the base representation

RNG uses color/depth MLPs, positional encodings, and optional shadow-map-driven features.

This is useful as inspiration for a later residual decoder, but it should not replace the texture
representation we already decided to build.

The useful lesson is:

- keep the neural part small
- let it refine or correct explicit shading
- do not make it responsible for all appearance from day one

## What Not To Copy From RNG

- The base appearance is still SH/features plus MLP, not per-Gaussian texture maps.
- Its shadow-map path depends on a view/depth reproject-and-query workflow that is more tightly tied
  to its renderer than to ours.
- It is optimized around the NRG/NRHints data path, not around 2DGS local-plane texturing.

So RNG is more useful as:

- data format guidance
- staged-training guidance
- lightweight residual-network inspiration

than as a direct renderer transplant.

## Direct Comparison For Our Use Case

### Geometry backbone

- `2dgs`: best fit
- `gs3`: not the right geometry backbone for us
- `RNG_release`: not the right geometry backbone for us

Decision:

- keep `2dgs` geometry unchanged as long as possible

### Texture-space appearance

- `2dgs`: local plane makes this natural
- `gs3`: does not solve our texture-space problem directly
- `RNG_release`: neural appearance, not texture-first

Decision:

- implement texture-space appearance ourselves in `2dgs`

### Light metadata and dataset design

- `gs3`: useful
- `RNG_release`: very useful

Decision:

- adopt a JSON dataset format that includes at least `pl_pos` and `pl_intensity`

### Shadows / visibility

- `gs3`: strongest reference
- `RNG_release`: useful secondary reference

Decision:

- shadow pipeline should be GS3-inspired first

### Neural residual for relighting

- `gs3`: optional and mixed with broader material machinery
- `RNG_release`: clearer reference

Decision:

- if we add a neural residual, follow RNG's spirit more than GS3's

## Recommended System Design After Reading Both Repos

## Phase 1: Textured 2DGS MVP

Stay entirely inside `2dgs`.

Build:

- per-Gaussian `tex_color`
- per-Gaussian `tex_alpha`
- sidecar appearance storage
- texture inheritance during clone/split/prune
- texture-aware rendering

No relighting yet.

## Phase 2: Minimal relighting-ready dataset format

Before training relighting, extend the dataset loader in `2dgs` so each frame can carry:

- `pl_pos`
- `pl_intensity`
- optional image-specific intrinsics if needed

Preferred source format:

- Blender/NRHints-style `transforms_*.json`

This is the cleanest bridge from the incoming dataset to our future relighting path.

## Phase 3: Explicit direct-light shading on texture-space materials

Use the sampled texture outputs as material terms, for example:

- base albedo from texture
- alpha from texture
- optional learned feature texture later

Then add simple direct lighting:

- Lambertian diffuse first
- distance falloff from point light
- optional view/light feature inputs later

This should happen before adding a neural residual.

## Phase 4: GS3-style shadow visibility pass

Add a light-space pass that uses the 2DGS geometry to estimate visibility.

High-level plan:

- create a light camera from `pl_pos`
- render geometry into light space
- derive a visibility or shadow ratio
- feed that signal into the final shaded view render

This is where GS3 is most relevant.

## Phase 5: RNG-style lightweight residual network

Only after explicit lighting and visibility work should we consider:

- a tiny residual MLP
- optional learned feature texture as input
- view direction / light direction / visibility as conditioning

This is where RNG becomes the best reference.

## Dataset Requirements We Should Ask For

When the dataset arrives, the most useful minimum payload is:

- images
- camera poses
- intrinsics
- train/test split or enough metadata to derive one
- per-image point-light position
- per-image light intensity

Strongly preferred extras:

- masks or alpha when available
- HDR images if relighting fidelity matters
- consistent scene scale
- a starter point cloud or COLMAP reconstruction

Optional but useful later:

- normals
- environment maps
- multiple lights
- shadow hints or relit target pairs under controlled lighting

## Concrete File-Level Plan For Our Repo

### Near-term files in `2dgs`

- `2dgs/scene/cameras.py`
  Add light metadata fields.
- `2dgs/scene/dataset_readers.py`
  Parse Blender/NRHints-style lighting metadata.
- `2dgs/utils/camera_utils.py`
  Thread those values into runtime `Camera` objects.
- `2dgs/gaussian_renderer/__init__.py`
  Later consume light inputs during shading.
- `2dgs/submodules/diff-surfel-rasterization/*`
  Later add texture sampling.

### Separate implementation stages

- Stage 1 work is mostly `GaussianModel`, `Scene`, and rasterizer texture plumbing.
- Stage 2 work is mostly camera/dataset/render path.
- Stage 3 shadow work likely needs a dedicated new rasterization path or an adapted light-space pass.

## Final Recommendation

The two external repos are useful, but in different ways:

- `gs3` should inform how we do shadows
- `RNG_release` should inform how we do relighting supervision and light-conditioned residuals

Our own implementation should stay centered on:

- `2dgs` geometry
- texture-space appearance
- explicit graphics terms first
- small learned residuals second

That is the lowest-risk and most coherent route toward a relightable textured 2DGS system.
