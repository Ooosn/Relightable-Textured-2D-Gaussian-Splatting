/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include "config.h"

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,
		const float* orig_points,
		const glm::vec2* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		bool* clamped,
		const float* transMat_precomp,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* points_xy_image,
		float* depths,
		// float* isovals,
		// float3* normals,
		float* transMats,
		float* colors,
		float4* normal_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered);

	// Main rasterization method (shadow accumulation pass).
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* points_xy_image,
		const float* depths,
			const float* transMats,
			const float4* normal_opacity,
			const float* texture_alpha,
			const int* texture_dims,
			int texture_resolution,
		float texture_sigma_factor,
		float* final_T,
		uint32_t* n_contrib,
		float* out_color,
		float* out_weight,
		float* out_trans,
			float* non_trans,
			const float offset,
			const float thres,
			const bool is_train,
			const bool texture_shadow_use_alpha,
			const bool texture_shadow_output_uv,
			const bool texture_shadow_alpha_bilinear);

	// Simple view-space color render with UV-level shadow modulation.
	// No offset-point mechanism – intended as a second pass after shadow accumulation.
	void renderColor(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* points_xy_image,
		const float* features,
		const float* transMats,
		const float4* normal_opacity,
		const float* texture_color,    // [N, C, R, R] or nullptr
		const float* texture_alpha,    // [N, 1, R, R] or nullptr
		const float* texture_shadow,   // [N*R*R] flat or nullptr
		int texture_resolution,
		float texture_sigma_factor,
		const float* background,
		float* out_color);             // [C, H, W]
}


#endif
