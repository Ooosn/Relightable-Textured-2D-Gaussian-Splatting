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

#include "forward.h"
#include "auxiliary.h"
#include "texture_utils.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Compute a 2D-to-2D mapping matrix from a tangent plane into a image plane
// given a 2D gaussian parameters.
__device__ void compute_transmat(
	const float3& p_orig,
	const glm::vec2 scale,
	float mod,
	const glm::vec4 rot,
	const float* projmatrix,
	const float* viewmatrix,
	const int W,
	const int H, 
	glm::mat3 &T,
	float3 &normal
) {

	glm::mat3 R = quat_to_rotmat(rot);
	glm::mat3 S = scale_to_mat(scale, mod);
	glm::mat3 L = R * S;

	// center of Gaussians in the camera coordinate
	glm::mat3x4 splat2world = glm::mat3x4(
		glm::vec4(L[0], 0.0),
		glm::vec4(L[1], 0.0),
		glm::vec4(p_orig.x, p_orig.y, p_orig.z, 1)
	);

	glm::mat4 world2ndc = glm::mat4(
		projmatrix[0], projmatrix[4], projmatrix[8], projmatrix[12],
		projmatrix[1], projmatrix[5], projmatrix[9], projmatrix[13],
		projmatrix[2], projmatrix[6], projmatrix[10], projmatrix[14],
		projmatrix[3], projmatrix[7], projmatrix[11], projmatrix[15]
	);

	glm::mat3x4 ndc2pix = glm::mat3x4(
		glm::vec4(float(W) / 2.0, 0.0, 0.0, float(W-1) / 2.0),
		glm::vec4(0.0, float(H) / 2.0, 0.0, float(H-1) / 2.0),
		glm::vec4(0.0, 0.0, 0.0, 1.0)
	);

	T = glm::transpose(splat2world) * world2ndc * ndc2pix;
	normal = transformVec4x3({L[2].x, L[2].y, L[2].z}, viewmatrix);

}

// Computing the bounding box of the 2D Gaussian and its center
// The center of the bounding box is used to create a low pass filter
__device__ bool compute_aabb(
	glm::mat3 T, 
	float cutoff,
	float2& point_image,
	float2& extent
) {
	glm::vec3 t = glm::vec3(cutoff * cutoff, cutoff * cutoff, -1.0f);
	float d = glm::dot(t, T[2] * T[2]);
	if (d == 0.0) return false;
	glm::vec3 f = (1 / d) * t;

	glm::vec2 p = glm::vec2(
		glm::dot(f, T[0] * T[2]),
		glm::dot(f, T[1] * T[2])
	);

	glm::vec2 h0 = p * p - 
		glm::vec2(
			glm::dot(f, T[0] * T[0]),
			glm::dot(f, T[1] * T[1])
		);

	glm::vec2 h = sqrt(max(glm::vec2(1e-4, 1e-4), h0));
	point_image = {p.x, p.y};
	extent = {h.x, h.y};
	return true;
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
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
	const float tan_fovx, const float tan_fovy,
	const float focal_x, const float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* transMats,
	float* rgb,
	float4* normal_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;
	
	// Compute transformation matrix
	glm::mat3 T;
	float3 normal;
	if (transMat_precomp == nullptr)
	{
		compute_transmat(((float3*)orig_points)[idx], scales[idx], scale_modifier, rotations[idx], projmatrix, viewmatrix, W, H, T, normal);
		float3 *T_ptr = (float3*)transMats;
		T_ptr[idx * 3 + 0] = {T[0][0], T[0][1], T[0][2]};
		T_ptr[idx * 3 + 1] = {T[1][0], T[1][1], T[1][2]};
		T_ptr[idx * 3 + 2] = {T[2][0], T[2][1], T[2][2]};
	} else {
		glm::vec3 *T_ptr = (glm::vec3*)transMat_precomp;
		T = glm::mat3(
			T_ptr[idx * 3 + 0], 
			T_ptr[idx * 3 + 1],
			T_ptr[idx * 3 + 2]
		);
		normal = make_float3(0.0, 0.0, 1.0);
	}

#if DUAL_VISIABLE
	float cos = -sumf3(p_view * normal);
	if (cos == 0) return;
	float multiplier = cos > 0 ? 1: -1;
	normal = multiplier * normal;
#endif

#if TIGHTBBOX // no use in the paper, but it indeed help speeds.
	// the effective extent is now depended on the opacity of gaussian.
	float cutoff = sqrtf(max(9.f + 2.f * logf(opacities[idx]), 0.000001));
#else
	float cutoff = 3.0f;
#endif

	// Compute center and radius
	float2 point_image;
	float radius;
	{
		float2 extent;
		bool ok = compute_aabb(T, cutoff, point_image, extent);
		if (!ok) return;
		radius = ceil(max(max(extent.x, extent.y), cutoff * FilterSize));
	}

	uint2 rect_min, rect_max;
	getRect(point_image, radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// Compute colors 
	if (colors_precomp == nullptr) {
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	depths[idx] = p_view.z;
	radii[idx] = (int)radius;
	points_xy_image[idx] = point_image;
	normal_opacity[idx] = {normal.x, normal.y, normal.z, opacities[idx]};
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ depths,
	const float* __restrict__ transMats,
	const float4* __restrict__ normal_opacity,
	const float* __restrict__ texture_alpha,
	const int* __restrict__ texture_dims,
	int texture_resolution,
	float texture_sigma_factor,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	float* __restrict__ out_color,
	float* __restrict__ out_weight,
	float* __restrict__ out_trans,
	float* __restrict__ non_trans,
	const float offset,
	const float thres,
	const bool is_train)
{
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	bool inside = pix.x < W && pix.y < H;
	bool done = !inside;

	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float collected_depth[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_normal_opacity[BLOCK_SIZE];
	__shared__ float3 collected_Tu[BLOCK_SIZE];
	__shared__ float3 collected_Tv[BLOCK_SIZE];
	__shared__ float3 collected_Tw[BLOCK_SIZE];
	__shared__ int collected_id_pointT[BLOCK_SIZE];
	__shared__ float collected_depth_pointT[BLOCK_SIZE];
	__shared__ float2 collected_xy_pointT[BLOCK_SIZE];
	__shared__ float4 collected_normal_opacity_pointT[BLOCK_SIZE];
	__shared__ float3 collected_Tu_pointT[BLOCK_SIZE];
	__shared__ float3 collected_Tv_pointT[BLOCK_SIZE];
	__shared__ float3 collected_Tw_pointT[BLOCK_SIZE];

	float T = 1.0f;
	float temp = 1.0f;
	float depthT = 0.0f;
	uint32_t contributor = 0;
	int pointT = 0;
	int i_pointT = 0;
	int progress_pointT = i_pointT * BLOCK_SIZE + block.thread_rank();

	if (range.x + progress_pointT < range.y)
	{
		const int coll_id = point_list[range.x + progress_pointT];
		collected_id_pointT[block.thread_rank()] = coll_id;
		collected_depth_pointT[block.thread_rank()] = depths[coll_id];
		collected_xy_pointT[block.thread_rank()] = points_xy_image[coll_id];
		collected_normal_opacity_pointT[block.thread_rank()] = normal_opacity[coll_id];
		collected_Tu_pointT[block.thread_rank()] = {transMats[9 * coll_id + 0], transMats[9 * coll_id + 1], transMats[9 * coll_id + 2]};
		collected_Tv_pointT[block.thread_rank()] = {transMats[9 * coll_id + 3], transMats[9 * coll_id + 4], transMats[9 * coll_id + 5]};
		collected_Tw_pointT[block.thread_rank()] = {transMats[9 * coll_id + 6], transMats[9 * coll_id + 7], transMats[9 * coll_id + 8]};
	}
	block.sync();

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		int progress = i * BLOCK_SIZE + block.thread_rank();
		block.sync();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_depth[block.thread_rank()] = depths[coll_id];
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_normal_opacity[block.thread_rank()] = normal_opacity[coll_id];
			collected_Tu[block.thread_rank()] = {transMats[9 * coll_id + 0], transMats[9 * coll_id + 1], transMats[9 * coll_id + 2]};
			collected_Tv[block.thread_rank()] = {transMats[9 * coll_id + 3], transMats[9 * coll_id + 4], transMats[9 * coll_id + 5]};
			collected_Tw[block.thread_rank()] = {transMats[9 * coll_id + 6], transMats[9 * coll_id + 7], transMats[9 * coll_id + 8]};
		}
		block.sync();

		for (int j = 0; j < min(BLOCK_SIZE, toDo); j++)
		{
			const int current_id = collected_id[j];
			const float sort_depth_j = collected_depth[j];
			const float2 xy_j = collected_xy[j];
			const float3 Tu_j = collected_Tu[j];
			const float3 Tv_j = collected_Tv[j];
			const float3 Tw_j = collected_Tw[j];
			const float4 nor_o_j = collected_normal_opacity[j];
			float footprint_j = 0.0f;
			float alpha_j = 0.0f;
			bool current_valid = false;
			// UV coords of this gaussian's intersection — hoisted so contribution
			// writes can use them for UV-indexed shadow accumulation.
			float u_j = 0.5f;
			float v_j = 0.5f;

			float3 k_j = pix.x * Tw_j - Tu_j;
			float3 l_j = pix.y * Tw_j - Tv_j;
			float3 p_j = cross(k_j, l_j);
			if (p_j.z != 0.0f)
			{
				float2 s_j = {p_j.x / p_j.z, p_j.y / p_j.z};
				float rho3d_j = (s_j.x * s_j.x + s_j.y * s_j.y);
				float2 d_j = {xy_j.x - pixf.x, xy_j.y - pixf.y};
				float rho2d_j = FilterInvSquare * (d_j.x * d_j.x + d_j.y * d_j.y);
				float power_j = -0.5f * min(rho3d_j, rho2d_j);
				bool inside_thres = (thres <= 0.0f) || ((d_j.x * d_j.x + d_j.y * d_j.y) <= thres * thres);

				if (inside_thres && power_j <= 0.0f)
				{
					float tex_dummy_rgb[CHANNELS] = {0};
					float tex_dummy_du[CHANNELS] = {0};
					float tex_dummy_dv[CHANNELS] = {0};
					float tex_a_j = 1.0f;
					float tex_a_du_j = 0.0f;
					float tex_a_dv_j = 0.0f;
					float du_dsx_j = 0.0f;
					float dv_dsy_j = 0.0f;
					compute_texture_uv(s_j, texture_sigma_factor, u_j, v_j, du_dsx_j, dv_dsy_j);
					sample_texture_bilinear(
						nullptr,
							texture_alpha,
							current_id,
							texture_resolution,
							texture_dims,
							u_j,
						v_j,
						tex_dummy_rgb,
						tex_a_j,
						tex_dummy_du,
						tex_dummy_dv,
						tex_a_du_j,
						tex_a_dv_j);

					footprint_j = max(0.0f, tex_a_j) * exp(power_j);
					alpha_j = min(0.99f, nor_o_j.w * footprint_j);
					current_valid = alpha_j >= (1.0f / 255.0f);
				}
			}

			float depth_diff = sort_depth_j - depthT;
			if (depth_diff > offset)
			{
				while (true)
				{
					T = T * temp;
					if (!done && T <= 0.001f)
					{
						done = true;
						T = 0.0f;
					}

					const int offset_id = collected_id_pointT[pointT];
					const float2 xy_o = collected_xy_pointT[pointT];
					const float3 Tu_o = collected_Tu_pointT[pointT];
					const float3 Tv_o = collected_Tv_pointT[pointT];
					const float3 Tw_o = collected_Tw_pointT[pointT];
					const float4 nor_o = collected_normal_opacity_pointT[pointT];

					float3 k_o = pix.x * Tw_o - Tu_o;
					float3 l_o = pix.y * Tw_o - Tv_o;
					float3 p_o = cross(k_o, l_o);
					float alpha_o = 0.0f;

					if (p_o.z != 0.0f)
					{
						float2 s_o = {p_o.x / p_o.z, p_o.y / p_o.z};
						float rho3d_o = (s_o.x * s_o.x + s_o.y * s_o.y);
						float2 d_o = {xy_o.x - pixf.x, xy_o.y - pixf.y};
						float rho2d_o = FilterInvSquare * (d_o.x * d_o.x + d_o.y * d_o.y);
						float power_o = -0.5f * min(rho3d_o, rho2d_o);
						if (power_o <= 0.0f)
						{
							float tex_dummy_rgb_o[CHANNELS] = {0};
							float tex_dummy_du_o[CHANNELS] = {0};
							float tex_dummy_dv_o[CHANNELS] = {0};
							float tex_a_o = 1.0f;
							float tex_a_du_o = 0.0f;
							float tex_a_dv_o = 0.0f;
							float u_o = 0.5f;
							float v_o = 0.5f;
							float du_dsx_o = 0.0f;
							float dv_dsy_o = 0.0f;
							compute_texture_uv(s_o, texture_sigma_factor, u_o, v_o, du_dsx_o, dv_dsy_o);
							sample_texture_bilinear(
								nullptr,
									texture_alpha,
									offset_id,
									texture_resolution,
									texture_dims,
									u_o,
								v_o,
								tex_dummy_rgb_o,
								tex_a_o,
								tex_dummy_du_o,
								tex_dummy_dv_o,
								tex_a_du_o,
								tex_a_dv_o);
							alpha_o = min(0.99f, nor_o.w * max(0.0f, tex_a_o) * exp(power_o));
							if (alpha_o < 1.0f / 255.0f)
								alpha_o = 0.0f;
						}
					}

					depthT = collected_depth_pointT[pointT];
					pointT = pointT + 1;
					if (pointT >= BLOCK_SIZE)
					{
						i_pointT += 1;
						if (i_pointT < rounds)
						{
							block.sync();
							progress_pointT = i_pointT * BLOCK_SIZE + block.thread_rank();
							if (range.x + progress_pointT < range.y)
							{
								const int coll_id = point_list[range.x + progress_pointT];
								collected_id_pointT[block.thread_rank()] = coll_id;
								collected_depth_pointT[block.thread_rank()] = depths[coll_id];
								collected_xy_pointT[block.thread_rank()] = points_xy_image[coll_id];
								collected_normal_opacity_pointT[block.thread_rank()] = normal_opacity[coll_id];
								collected_Tu_pointT[block.thread_rank()] = {transMats[9 * coll_id + 0], transMats[9 * coll_id + 1], transMats[9 * coll_id + 2]};
								collected_Tv_pointT[block.thread_rank()] = {transMats[9 * coll_id + 3], transMats[9 * coll_id + 4], transMats[9 * coll_id + 5]};
								collected_Tw_pointT[block.thread_rank()] = {transMats[9 * coll_id + 6], transMats[9 * coll_id + 7], transMats[9 * coll_id + 8]};
							}
							block.sync();
						}
						pointT = 0;
					}

				temp = 1.0f - alpha_o;
				depth_diff = sort_depth_j - depthT;
				if (depth_diff <= offset)
				{
					if (!done && current_valid)
					{
						contributor++;
						float transition_trans = footprint_j * (T * (1.0f - (depth_diff / offset) + (depth_diff / offset) * temp));
							if (texture_resolution > 0 || texture_dims != nullptr)
							{
								int tex_h = 0;
								int tex_w = 0;
								int tex_offset = 0;
								if (texture_layout(current_id, texture_resolution, texture_dims, tex_h, tex_w, tex_offset))
								{
									int ui = min(max((int)(u_j * tex_w), 0), tex_w - 1);
									int vi = min(max((int)(v_j * tex_h), 0), tex_h - 1);
									int sidx = texture_alpha_index(current_id, vi, ui, texture_resolution, texture_dims);
									atomicAdd(&(out_trans[sidx]), transition_trans);
									atomicAdd(&(non_trans[sidx]), footprint_j);
								}
							}
							else
							{
							atomicAdd(&(out_trans[current_id]), transition_trans);
							atomicAdd(&(non_trans[current_id]), footprint_j);
						}
						if (is_train)
							atomicAdd(&(out_weight[current_id]), footprint_j * T);
					}
					break;
				}
			}
		}
		else
		{
			if (!done && current_valid)
			{
				contributor++;
				float transition_trans = footprint_j * (T * (1.0f - (depth_diff / offset) + (depth_diff / offset) * temp));
					if (texture_resolution > 0 || texture_dims != nullptr)
					{
						int tex_h = 0;
						int tex_w = 0;
						int tex_offset = 0;
						if (texture_layout(current_id, texture_resolution, texture_dims, tex_h, tex_w, tex_offset))
						{
							int ui = min(max((int)(u_j * tex_w), 0), tex_w - 1);
							int vi = min(max((int)(v_j * tex_h), 0), tex_h - 1);
							int sidx = texture_alpha_index(current_id, vi, ui, texture_resolution, texture_dims);
							atomicAdd(&(out_trans[sidx]), transition_trans);
							atomicAdd(&(non_trans[sidx]), footprint_j);
						}
					}
					else
					{
					atomicAdd(&(out_trans[current_id]), transition_trans);
					atomicAdd(&(non_trans[current_id]), footprint_j);
				}
				if (is_train)
					atomicAdd(&(out_weight[current_id]), footprint_j * T);
			}
		}
		}
	}

	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = 0.0f;
	}
}

// ─── Simple color render kernel ──────────────────────────────────────────────
// Standard front-to-back alpha blending with UV-level texture color + shadow.
// No offset-point mechanism. Intended as the view-space render pass after the
// shadow accumulation pass.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderColorCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float* __restrict__ transMats,
	const float4* __restrict__ normal_opacity,
	const float* __restrict__ texture_color,    // [N, C, R, R] or nullptr
	const float* __restrict__ texture_alpha,    // [N, 1, R, R] or nullptr
	const float* __restrict__ texture_shadow,   // [N, R, R] flat, or nullptr
	int texture_resolution,
	float texture_sigma_factor,
	const float* __restrict__ background,
	float* __restrict__ out_color)              // [C, H, W]
{
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
	uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = {(float)pix.x, (float)pix.y};

	bool inside = (pix.x < (uint32_t)W && pix.y < (uint32_t)H);
	bool done = !inside;

	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	__shared__ int    rc_id[BLOCK_SIZE];
	__shared__ float2 rc_xy[BLOCK_SIZE];
	__shared__ float3 rc_Tu[BLOCK_SIZE];
	__shared__ float3 rc_Tv[BLOCK_SIZE];
	__shared__ float3 rc_Tw[BLOCK_SIZE];
	__shared__ float4 rc_no[BLOCK_SIZE];

	float T = 1.0f;
	float C[CHANNELS] = {};

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		block.sync();
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int cid = point_list[range.x + progress];
			rc_id[block.thread_rank()] = cid;
			rc_xy[block.thread_rank()] = points_xy_image[cid];
			rc_no[block.thread_rank()] = normal_opacity[cid];
			rc_Tu[block.thread_rank()] = {transMats[9*cid+0], transMats[9*cid+1], transMats[9*cid+2]};
			rc_Tv[block.thread_rank()] = {transMats[9*cid+3], transMats[9*cid+4], transMats[9*cid+5]};
			rc_Tw[block.thread_rank()] = {transMats[9*cid+6], transMats[9*cid+7], transMats[9*cid+8]};
		}
		block.sync();

		for (int j = 0; j < min(BLOCK_SIZE, toDo); j++)
		{
			if (done) break;

			const int gid  = rc_id[j];
			const float2 xy_j = rc_xy[j];
			const float3 Tu_j = rc_Tu[j];
			const float3 Tv_j = rc_Tv[j];
			const float3 Tw_j = rc_Tw[j];
			const float4 no_j = rc_no[j];

			float3 k_j = pix.x * Tw_j - Tu_j;
			float3 l_j = pix.y * Tw_j - Tv_j;
			float3 p_j = cross(k_j, l_j);
			if (p_j.z == 0.0f) continue;

			float2 s_j = {p_j.x / p_j.z, p_j.y / p_j.z};
			float rho3d = s_j.x * s_j.x + s_j.y * s_j.y;
			float2 d_j  = {xy_j.x - pixf.x, xy_j.y - pixf.y};
			float rho2d = FilterInvSquare * (d_j.x * d_j.x + d_j.y * d_j.y);
			float power  = -0.5f * min(rho3d, rho2d);
			if (power > 0.0f) continue;

			float u_j = 0.5f, v_j = 0.5f, du = 0.0f, dv = 0.0f;
			compute_texture_uv(s_j, texture_sigma_factor, u_j, v_j, du, dv);

			float rgb_j[CHANNELS] = {};
			float dummy_du[CHANNELS] = {}, dummy_dv[CHANNELS] = {};
			float tex_a = 1.0f, tex_a_du = 0.0f, tex_a_dv = 0.0f;
			if (texture_color != nullptr && texture_resolution > 0)
			{
				sample_texture_bilinear(
					texture_color, texture_alpha,
					gid, texture_resolution,
					u_j, v_j,
					rgb_j, tex_a,
					dummy_du, dummy_dv,
					tex_a_du, tex_a_dv);
			}
			else
			{
				for (int ch = 0; ch < CHANNELS; ch++)
					rgb_j[ch] = features[gid * CHANNELS + ch];
			}

			float footprint = max(0.0f, tex_a) * exp(power);
			float alpha = min(0.99f, no_j.w * footprint);
			if (alpha < 1.0f / 255.0f) continue;

			// Per-UV shadow lookup
			float shadow = 1.0f;
			if (texture_shadow != nullptr && texture_resolution > 0)
			{
				int ui = min(max((int)(u_j * texture_resolution), 0), texture_resolution - 1);
				int vi = min(max((int)(v_j * texture_resolution), 0), texture_resolution - 1);
				shadow = texture_shadow[texture_alpha_index(gid, vi, ui, texture_resolution)];
			}

			float w = alpha * T;
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += w * rgb_j[ch] * shadow;

			T *= (1.0f - alpha);
			if (T <= 0.001f) done = true;
		}
	}

	if (inside)
	{
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * background[ch];
	}
}
// ─────────────────────────────────────────────────────────────────────────────

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
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
	const bool is_train)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		depths,
		transMats,
		normal_opacity,
		texture_alpha,
		texture_dims,
		texture_resolution,
		texture_sigma_factor,
		final_T,
		n_contrib,
		out_color,
		out_weight,
		out_trans,
		non_trans,
		offset,
		thres,
		is_train);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
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
	const int W, const int H,
	const float focal_x, const float focal_y,
	const float tan_fovx, const float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* transMats,
	float* rgb,
	float4* normal_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		transMat_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		transMats,
		rgb,
		normal_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}

void FORWARD::renderColor(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* features,
	const float* transMats,
	const float4* normal_opacity,
	const float* texture_color,
	const float* texture_alpha,
	const float* texture_shadow,
	int texture_resolution,
	float texture_sigma_factor,
	const float* background,
	float* out_color)
{
	renderColorCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		features,
		transMats,
		normal_opacity,
		texture_color,
		texture_alpha,
		texture_shadow,
		texture_resolution,
		texture_sigma_factor,
		background,
		out_color);
}
