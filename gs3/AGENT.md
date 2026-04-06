# Agent Handoff

## Goal
This project studies shadow-supervised 3D Gaussian Splatting under controlled illumination.
The main paper path is the non-HGS differentiable shadow pipeline:

1. Splat Gaussians in light space.
2. Compute per-Gaussian cast-shadow visibility / occlusion ratio.
3. Back-propagate shadow errors to Gaussian geometry, scale, and opacity.
4. Use the shadow term as supervision and as a rendering cue.

## Current Status
- The trusted reference for shadow semantics is `submodules/diff-gaussian-rasterization_light/cuda_rasterizer/forward.cu`.
- The main non-HGS CUDA backward has already been debugged for the reverse-transmittance (`T`) chain.
- The old finite-difference debug branch has been removed from the main renderer.
- Legacy `shadow_opacity` / `use_shadow_gs` branches have been removed from the training/rendering path.
- Synthetic smoke tests exist and were used to validate finite gradients on the core light rasterizer path.
- A tiny self-made Blender-format dataset now exists at `synthetic_shadow_tiny/`.
- A richer mesh-based Blender-format dataset generator now exists at `make_mesh_shadow_dataset.py`.
- Two mesh presets are available right now: `studio` and `single_object`.
- Shadow-gradient validation utilities now exist at:
  - `shadow_validation_common.py`
  - `shadow_gradient_probe.py`
  - `toy_shadow_fd_check.py`
  - `synthetic_shadow_validation.py`
- The main training entry `train.py` has already been run successfully on this tiny dataset.
- An automated harness `run_tiny_shadow_smoke.py` can now generate the dataset and launch a short smoke training run in one command.
- A second harness `run_mesh_shadow_smoke.py` now does the same for the mesh-based dataset.

## Important Conclusions
- If `Inf` / `NaN` appears inside the light CUDA backward, the first assumption should be forward/backward state mismatch, not an invalid shadow formulation.
- Do not casually rewrite the main loop structure in `forward.cu`. Forward shadow behavior is treated as ground truth unless there is very strong evidence otherwise.
- The paper mainline should focus on the non-HGS path. HGS is optional and not the current priority.
- `train_light_direction.py` is not the canonical full training path. It is a separate experiment script and still has its own wiring issues.
- The current analytic shadow backward demonstrably reaches `xyz`, `scale`, `opacity`, and `rotation` in end-to-end rendering.
- On the toy light-rasterizer cases, finite-difference checks match the analytic gradient sign for `xyz`, `opacity`, and `scale`.
- On the mesh synthetic datasets, multi-view shadow-only optimization clearly reduces shadow loss. The `scale` branch is the cleanest geometry-oriented recovery case right now.

## Key Files
- `gaussian_renderer/__init__.py`
  Main render entry. This is where the light-space shadow pass and the view-space render meet.
- `submodules/diff-gaussian-rasterization_light/cuda_rasterizer/forward.cu`
  Ground-truth implementation of the light-space forward shadow accumulation.
- `submodules/diff-gaussian-rasterization_light/cuda_rasterizer/backward.cu`
  Analytic CUDA backward for the light-space shadow pass.
- `submodules/diff-gaussian-rasterization_light/diff_gaussian_rasterization_light/__init__.py`
  Python/CUDA wrapper for the light rasterizer.
- `scene/gaussian_model.py`
  Gaussian parameters, optimizer groups, densification, checkpoint/PLY IO.
- `train.py`
  Main training script. This is the primary path for paper experiments.
- `train_light_direction.py`
  Separate light-direction experiment script. Not yet fully aligned with the main renderer.
- `debug_light_smoke.py`
  Synthetic regression script for the light rasterizer.
- `make_tiny_shadow_dataset.py`
  Creates a tiny controlled-illumination Blender-format dataset for smoke testing.
- `run_tiny_shadow_smoke.py`
  One-command generator + smoke-training launcher built on top of `make_tiny_shadow_dataset.py` and `train.py`.
- `make_mesh_shadow_dataset.py`
  Mesh-based synthetic dataset generator using Open3D ray casting and Blender-format camera metadata.
- `run_mesh_shadow_smoke.py`
  One-command generator + smoke-training launcher for the mesh-based synthetic dataset.
- `main.tex`
  Paper draft.
- `shadow_validation_common.py`
  Shared utilities for synthetic shadow validation, GT shadow reconstruction, rendering helpers, and artifact export.
- `shadow_gradient_probe.py`
  One-step gradient connectivity check on existing synthetic datasets.
- `toy_shadow_fd_check.py`
  Small controlled finite-difference contract checks for `xyz`, `opacity`, and `scale`.
- `synthetic_shadow_validation.py`
  Multi-view shadow-only recovery experiments on the existing synthetic datasets.

## What Was Cleaned Up
- Removed the `DifferentiableShadow` finite-difference fallback from `gaussian_renderer/__init__.py`.
- Removed the iteration-gated `shadow_opacity` debug route from the renderer.
- Removed `use_shadow_gs` / `shadow_opacity` from the active model/optimizer/densification/PLY path.
- Kept one legacy guard in `scene/gaussian_model.py` so old checkpoint dicts with `shadow_opacity` keys are ignored instead of crashing immediately.

## Known Remaining Items
- Real data has not yet been used to run the final experiment validation.
- `train_light_direction.py` creates a learnable `gaussians.light_direction`, but the renderer currently does not actually consume it because `use_customized_light_direction` is still `False`.
- HGS and ortho-light are not the current target and should not be mixed into the main paper claim by accident.
- The tiny synthetic smoke run only validates stability and wiring. It is not a final convergence or paper-quality experiment.
- Shadow-only recovery improves shadow loss strongly, but final parameter error can overshoot if optimization runs too long. Read the recorded best-step statistics, not only the last step.

## Smoke-Test Status
- `python train.py -s synthetic_shadow_tiny -m output/synthetic_shadow_tiny_20iter --iterations 20 --unfreeze_iterations 0 --view_num 4 --load_num 4 --white_background --eval --test_iterations 999999 --save_iterations 999999 --checkpoint_iterations 999999`
  completed successfully, with training loss decreasing over the short run.
- `python run_tiny_shadow_smoke.py --iterations 5 --model-path output/synthetic_shadow_tiny_smoke_script`
  completed successfully and wrote outputs to `output/synthetic_shadow_tiny_smoke_script/`.
- `python make_mesh_shadow_dataset.py --output synthetic_shadow_mesh --preset studio --force`
  completed successfully and produced a Blender-format mesh scene with 12 train views, 6 test views, and `points3d.ply`.
- `python train.py -s synthetic_shadow_mesh -m output/synthetic_shadow_mesh_20iter --iterations 20 --unfreeze_iterations 0 --view_num 12 --load_num 12 --white_background --eval --test_iterations 999999 --save_iterations 999999 --checkpoint_iterations 999999`
  completed successfully on the mesh-based synthetic scene.
- `python make_mesh_shadow_dataset.py --output synthetic_shadow_single_object --preset single_object --force`
  completed successfully and produced a single-object controlled-light scene for future experiments.
- `python shadow_gradient_probe.py --datasets synthetic_shadow_single_object synthetic_shadow_mesh`
  completed successfully. Both datasets show finite nonzero gradients for `xyz`, `scale`, and `opacity`.
- `python toy_shadow_fd_check.py`
  completed successfully. Analytic and numeric gradients have matching signs for `xyz`, `opacity`, and `scale`.
- `python synthetic_shadow_validation.py --steps 200`
  completed successfully. On `synthetic_shadow_single_object`, all three groups reduce shadow loss; `scale` is the only group whose best-step parameter error also improves. The follow-up `synthetic_shadow_mesh` run with `scale` also shows both best-step shadow improvement and best-step parameter-error improvement.
- The smoke dataset contains:
  - 4 training views
  - 2 test views
  - a simple object-plane scene with a known light source
  - a starter `points3d.ply`

## Recommended Next Steps
1. Run a real-data single-step check through `train.py` once data is available.
2. Verify `loss`, `shadow`, and geometry-related gradients remain finite on real scenes.
3. Decide whether `train_light_direction.py` should be repaired for a dedicated light-direction experiment or left out of the main paper story.
4. Use the tiny synthetic dataset as the first-line regression harness whenever the light renderer or training glue changes.
5. Use the mesh-based synthetic dataset when a more realistic cast-shadow sanity check is needed.
6. Use `output/shadow_gradient_validation/` as the main evidence folder for gradient-connectivity and shadow-only recovery claims.
7. Expand `main.tex` around experiments and ablations using the non-HGS shadow path as the core formulation.

## Practical Rules For Future Sessions
- Treat `forward.cu` as the source of truth for shadow semantics.
- Prefer fixing metadata/state alignment over restructuring CUDA loops.
- Keep the mainline centered on the analytic light rasterizer.
- Do not reintroduce finite-difference shadow gradients unless explicitly needed for a separate experiment.
