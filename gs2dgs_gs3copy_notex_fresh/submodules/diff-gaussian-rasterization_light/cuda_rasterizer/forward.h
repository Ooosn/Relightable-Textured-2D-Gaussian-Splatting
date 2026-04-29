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

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,
		const float* orig_points,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		bool* clamped,
		const float* cov3D_precomp,
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
		float* cov3Ds,
		float* colors,
		float4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered,
		float low_pass_filter_radius,

		// 投影方式
		const bool ortho,
		
		//emmm
		float* radii_comp,
		

		// hgs 相关
		const bool hgs,
		bool conic_,
		const float* hgs_normals,
		const float* hgs_opacities,
		float3* save_normal,
		float* cov3D_smalls,
		float4* conic_opacity_hgs,
		float4* conic_opacity_hgs_temp,
		uint4* conic_opacity3,
		uint4* conic_opacity4,
		float3* conic_opacity5,
		uint4* conic_opacity6
		);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* points_xy_image,
		const float* point_depth,
		const float* features,
		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color,
		float* out_weight,
		float* out_trans,
		float* non_trans,
		const float offset,
		const float thres,
    	const bool is_train,
		
		//added
		const float* radii_comp,

		// hgs 相关
		const bool hgs,
		const float* hgs_opacities,
		float* hgs_opacities_shadow,
		float* hgs_opacities_light,
		const float3* saved_normal,
		const float4* conic_opacity_hgs,
		// for backward
		uint32_t* id_contrib,
		uint32_t* n_contrib_offset,
		uint32_t* id_contrib_offset
		);
}


#endif