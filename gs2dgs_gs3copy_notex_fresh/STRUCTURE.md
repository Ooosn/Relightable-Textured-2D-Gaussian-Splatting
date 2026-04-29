# Project Structure

## Core Training Path
- `train.py`
  Main training loop for reconstruction and rendering experiments.
- `gaussian_renderer/__init__.py`
  Main render function used by training. It combines:
  - light-space shadow splatting
  - view-space Gaussian splatting
  - optional neural appearance refinement
- `scene/gaussian_model.py`
  Owns Gaussian parameters, optimizer setup, densification, pruning, checkpoint restore, and PLY IO.

## Differentiable Shadow Path
- `submodules/diff-gaussian-rasterization_light/cuda_rasterizer/forward.cu`
  Computes the light-space forward shadow statistics.
  The shadow signal is based on a normalized occlusion ratio rather than ordinary RGB alpha blending.
- `submodules/diff-gaussian-rasterization_light/cuda_rasterizer/backward.cu`
  Analytic backward for the light-space shadow pass.
  This is where reverse transmittance reconstruction happens.
- `submodules/diff-gaussian-rasterization_light/cuda_rasterizer/rasterizer_impl.cu`
  CUDA dispatch / glue code.
- `submodules/diff-gaussian-rasterization_light/diff_gaussian_rasterization_light/__init__.py`
  Torch extension wrapper.

## Renderer Data Flow
1. `render()` gathers Gaussian states from `GaussianModel`.
2. The light rasterizer computes per-Gaussian shadow numerator / denominator statistics.
3. The renderer normalizes them into a shadow ratio.
4. The view-space renderer uses:
   - base RGB
   - shadow term
   - optional `other_effects`
5. In `train.py`, after the warm-up phase, the image is rendered as:
   - `image = image * shadow + other_effects`

## Experiment Scripts
- `train.py`
  Main reconstruction path and the one most relevant to the paper.
- `train_light_direction.py`
  Separate experiment script that directly supervises the shadow output.
  Important: this file still needs extra wiring if light direction itself should be optimized through the renderer.
- `render.py`
  Rendering / evaluation entry point.
- `make_tiny_shadow_dataset.py`
  Generates a tiny Blender-format synthetic dataset with known light position and visible cast shadows.
- `run_tiny_shadow_smoke.py`
  Convenience wrapper that regenerates the tiny dataset and launches a short `train.py` smoke run.
- `make_mesh_shadow_dataset.py`
  Generates a mesh-based Blender-format synthetic dataset using Open3D ray casting and hard shadows from a point light.
- `run_mesh_shadow_smoke.py`
  Convenience wrapper that regenerates the mesh-based dataset and launches a short `train.py` smoke run.
- `shadow_validation_common.py`
  Shared helper layer for GT shadow reconstruction, synthetic-scene loading, fixed-view artifact export, and validation-time rendering.
- `shadow_gradient_probe.py`
  One-step synthetic probe for checking whether shadow loss reaches `xyz / scale / opacity / rotation`.
- `toy_shadow_fd_check.py`
  Finite-difference validation for the light rasterizer on controlled toy cases.
- `synthetic_shadow_validation.py`
  Multi-view shadow-only recovery experiments on `synthetic_shadow_single_object/` and `synthetic_shadow_mesh/`.

## Debug / Validation
- `debug_light_smoke.py`
  Synthetic tests for the differentiable shadow rasterizer.
  Use this first when CUDA shadow backward changes.
- `synthetic_shadow_tiny/`
  Small self-made dataset for end-to-end smoke tests.
- `synthetic_shadow_mesh/`
  Multi-object mesh scene with known light position, Blender-format poses, and a starter point cloud.
- `synthetic_shadow_single_object/`
  Single-object mesh scene intended for controlled-light object experiments.
- `output/shadow_gradient_validation/probe/`
  Formal gradient-connectivity results and per-view shadow comparisons for the existing synthetic datasets.
- `output/shadow_gradient_validation/toy_fd/`
  Formal toy finite-difference results for `xyz`, `opacity`, and `scale`.
- `output/shadow_gradient_validation/synthetic_recovery_v2/`
  Formal multi-view shadow-only recovery results. This is currently the strongest synthetic evidence folder for the paper claim.
- `output/synthetic_shadow_tiny_20iter/`
  Successful short training run on the tiny dataset.
- `output/synthetic_shadow_tiny_smoke_script/`
  Successful automated smoke run launched through `run_tiny_shadow_smoke.py`.
- `output/synthetic_shadow_mesh_smoke_v2/`
  Successful short training run on the mesh-based synthetic dataset after geometry cleanup.
- `output/synthetic_shadow_mesh_20iter/`
  Successful 20-iteration training run on the mesh-based synthetic dataset.

## Smoke Test Recipe
1. Activate the `mygs` environment.
2. Run `python run_tiny_shadow_smoke.py --iterations 5 --model-path output/synthetic_shadow_tiny_smoke_script`.
3. Confirm that training completes and that a point cloud is written under `point_cloud/iteration_5/`.
4. For a slightly stronger sanity check, run a 20-iteration job with `train.py` directly on `synthetic_shadow_tiny/`.

## Mesh Smoke Test Recipe
1. Activate the `mygs` environment.
2. Run `python run_mesh_shadow_smoke.py --preset studio --iterations 20 --model-path output/synthetic_shadow_mesh_smoke`.
3. Confirm that training completes and that a point cloud is written under the chosen output directory.
4. Use `--preset single_object --generate-only` when you only want the controlled-light object dataset without launching training.

## Shadow Validation Recipe
1. Activate the `mygs` environment.
2. Run `python shadow_gradient_probe.py --datasets synthetic_shadow_single_object synthetic_shadow_mesh`.
3. Run `python toy_shadow_fd_check.py`.
4. Run `python synthetic_shadow_validation.py --steps 200`.
5. Inspect:
   - `output/shadow_gradient_validation/probe/summary.json`
   - `output/shadow_gradient_validation/toy_fd/toy_fd_summary.json`
   - `output/shadow_gradient_validation/synthetic_recovery_v2/summary.json`
6. For paper figures, the most useful panels right now are under:
   - `output/shadow_gradient_validation/probe/*/*/*/`
   - `output/shadow_gradient_validation/synthetic_recovery_v2/synthetic_shadow_single_object/scale/`
   - `output/shadow_gradient_validation/synthetic_recovery_v2/synthetic_shadow_mesh/scale/`

## Paper / Writing
- `main.tex`
  Paper draft. It now contains the introduction, method section, and an initial experimental setup / ablation skeleton.

## Deprioritized Branches
- HGS
  Present in the codebase but not the current paper mainline.
- `ortho_light`
  Derived branch, not the current reference.
- `shadow_opacity`
  Old debug / experimental route. Removed from the active training-rendering path.

## Suggested File Reading Order
1. `AGENT.md`
2. `STRUCTURE.md`
3. `gaussian_renderer/__init__.py`
4. `submodules/diff-gaussian-rasterization_light/cuda_rasterizer/forward.cu`
5. `submodules/diff-gaussian-rasterization_light/cuda_rasterizer/backward.cu`
6. `scene/gaussian_model.py`
7. `train.py`
8. `make_tiny_shadow_dataset.py`
9. `run_tiny_shadow_smoke.py`
10. `make_mesh_shadow_dataset.py`
11. `run_mesh_shadow_smoke.py`
12. `main.tex`
