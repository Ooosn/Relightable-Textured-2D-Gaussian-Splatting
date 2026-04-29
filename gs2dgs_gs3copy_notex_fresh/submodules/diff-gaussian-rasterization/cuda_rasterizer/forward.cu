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
#include "stream.h"

#include "forward.h"
#include "auxiliary.h"
#include <cuda_runtime.h>
#include <math_functions.h> 
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <math_constants.h>  // 对 expf, fabsf, copysignf 声明
namespace cg = cooperative_groups;



__device__ __forceinline__ float fast_erff(float x) {
    float sign = copysignf(1.0f, x);
    x = fabsf(x);
    const float a1 = 0.254829592f;
    const float a2 = -0.284496736f;
    const float a3 = 1.421413741f;
    const float a4 = -1.453152027f;
    const float a5 = 1.061405429f;
    const float p  = 0.3275911f;
    float t = 1.0f / (1.0f + p * x);
    float y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1)
                   * t * expf(-x * x);
    return sign * y;
}


__device__ __forceinline__ float approx_tanhf(float x) {
    // 限定范围
    x = fmaxf(fminf(x,  5.0f), -5.0f);
    float x2 = x*x;
    // 有理函数近似：tanh(x) ≈ x*(27+x²)/(27+9x²)
    // fmaf(a, b, c) 等同于 a*b + c
    float num = fmaf(x, 27.0f + x2, 0.0f);
    float den = 27.0f + 9.0f*x2;
    return num / den;
}

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


// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}


// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix, float low_pass_filter_radius = 0.3f)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += low_pass_filter_radius;
	cov[1][1] += low_pass_filter_radius;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Calculate the 2D splatting for half gaussian
__device__ float6 computeCov2D_halfgs(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix, float low_pass_filter_radius = 0.3f)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002).
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	// Change J for half gaussian
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		(focal_x / t.z), 0.0, (-(focal_x * t.x) / (t.z * t.z)),
		0.0, (focal_y / t.z), (-(focal_y * t.y) / (t.z * t.z)),
		0, 0, (focal_y / t.z));

	glm::mat3 W = glm::mat3(
		(viewmatrix[0]), (viewmatrix[4]), (viewmatrix[8]),
		(viewmatrix[1]), (viewmatrix[5]), (viewmatrix[9]),
		(viewmatrix[2]), (viewmatrix[6]), (viewmatrix[10]));

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. 
	// No discard 3rd row and column.
	cov[0][0] += low_pass_filter_radius;
	cov[1][1] += low_pass_filter_radius;
	cov[2][2] += low_pass_filter_radius;
	return { cov[0][0], cov[0][1], cov[1][1], cov[0][2], cov[1][2], cov[2][2] };


}



// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
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
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	float low_pass_filter_radius,

	// added
	float* radii_comp,

	// hgs 相关
	const bool hgs,
	bool conic_,
	const float* hgs_normals,
	const float* hgs_opacities,	// 占位，暂时没用
	float3* save_normal,	// 占位，暂时没用
	float* cov3D_smalls,
	float4* conic_opacity_hgs,
	float4* conic_opacity_hgs_temp,
	uint4* conic_opacity3,	// 占位，暂时没用
	uint4* conic_opacity4,	// 占位，暂时没用
	float3* conic_opacity5,	// 占位，暂时没用
	uint4* conic_opacity6	// 占位，暂时没用
	)
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

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;	// 每个高斯点的 3D 协方差矩阵 指针，3D 协方差矩阵用 6 个元素表示
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	/////
	// Compute 2D screen-space covariance matrix:
	float3 conic;
	float3 cov;
	float det;
	if (hgs)
	{	
		// 转化 3D 法向量
		float3 gs_normal = {hgs_normals[3*idx],hgs_normals[3*idx+1],hgs_normals[3*idx+2]};
		gs_normal = transformPoint4x3(gs_normal, viewmatrix);

		// computeCov2D_halfgs 保留 z 轴方向的协方差，默认选取 fy 用于量纲变化
		float6 cov_temp = computeCov2D_halfgs(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix, low_pass_filter_radius);
		cov = {cov_temp.x, cov_temp.y, cov_temp.z};

		det = cov.x * cov.z - cov.y * cov.y; 
		if (det == 0.0f)
			return;
		float det_inv = 1.f / det;
		conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };


		//////
		// calculate rectangle size for 2D ellipse:
		// float power = logf(256.f * max(opacities[2 * idx], opacities[2 * idx+1]));//logf(2.f) * 8.0f + logf(2.f) * log2_opacity;
		// int width = (int)(1.414214f * __fsqrt_rn(cov.x * power) + 1.0f);
		// int height = (int)(1.414214f * __fsqrt_rn(cov.z * power) + 1.0f);
		// if (width <= 0 || height <= 0){
		//	 return;
		//}


		// 计算 I(x,y)的参数：
		// 计算分母，sqrt(2)·sigmaz|xy， sigmaz|xy <-> 当前 xy平面下, z的方差，该值为常量，条件分布的方差在多变量高斯中是固定的，与被条件化的变量值无关
		float first_divide = 1/(1.4142135f*sqrtf(max(0.00000001f,cov_temp.v - (cov_temp.w*cov_temp.w* conic.x+2*cov_temp.w*cov_temp.u*conic.y+cov_temp.u*cov_temp.u* conic.z))));
		// 计算 miuz|xy 该值会随着 x，y 变化，因此此处求的是导数，即 给定 x,y 时，z 的条件分布均值的线性变化率，因为后面不同像素位置不同，到时候再计算最终的 miuz|xy
		float x_term = cov_temp.w* conic.x + cov_temp.u* conic.y;
		float y_term = cov_temp.u* conic.z + cov_temp.w* conic.y;
		gs_normal.z = (1.f/(gs_normal.z+0.000001f)); //just in case z is 0
		gs_normal.x = gs_normal.x*gs_normal.z;
		gs_normal.y = gs_normal.y*gs_normal.z;

		// conic_opacity_hgs[idx] = {x_term*first_divide, y_term*first_divide, first_divide, hgs_opacities[2 * idx + 1]};
		// conic_opacity_hgs 储存 I(x,y) 中 所有参数, opacities 暂时占位用
		conic_opacity_hgs[idx] = { x_term, y_term, first_divide, opacities[idx] };		
		save_normal[idx] = gs_normal;
		conic_ = false;
	}
	// 非 hgs 渲染
	else{
		conic_ = false;
		cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix, low_pass_filter_radius);
		/*
		* 源代码中近期加入了 抗锯齿处理
		* 原理就是：增大高斯的协方差，将高斯范围扩大（模糊），然后对不透明度做一个修正：
		* opacity * h_convolution_scaling
		* h_convolution_scaling = sqrt(max(0.000025f, det_cov / det_cov_plus_h_cov))
		* det_cov = cov.x * cov.z - cov.y * cov.y    原协方差行列式
		* h_var = 0.3f
		* cov.x += h_var;
		* cov.z += h_var;
		* det_cov_plus_h_cov = cov.x * cov.z - cov.y * cov.y;	模糊后协方差行列式
		*/


		// Invert covariance (EWA algorithm)
		det = (cov.x * cov.z - cov.y * cov.y);
		if (det == 0.0f)
			return;
		float det_inv = 1.f / det;
		conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };
	}

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.01f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.01f, mid * mid - det));
	float my_radius = 3.f * sqrt(max(lambda1, lambda2));
	float my_radius_ceil = ceil(my_radius);
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	// 用 圆心 加减 3*最长轴 即3sigma，得到 矩形范围，其实相当于先获得圆，然后获得包裹圆的矩形
	getRect(point_image, my_radius_ceil, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;


	// Store some useful helper data for the next steps.
	
	// change the z dist to the dist to camera center 这里计算的是 高斯点到相机中心的距离，原代码是 z 轴距离，因为我们要计算向光源点的遮挡，深度排序
	// depths[idx] = p_view.z;
	depths[idx] = p_view.z;
	radii[idx] = my_radius_ceil;
	radii_comp[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	// conic.x, conic.y, conic.z 2D 协方差矩阵的逆，conic.w 高斯点的不透明度
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}



// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
hgs_renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ geomState_depth,
	float* __restrict__ out_depth,
	float* __restrict__ out_alpha,
	
	const float* __restrict__ radii_comp,
	
	// hgs 相关
	const bool hgs,
	const float* __restrict__ hgs_opacities,	// 占位，暂时没用
	const float3* __restrict__ saved_normal,
	const float4* __restrict__ conic_opacity_hgs
	)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// smoother
	__shared__ float collected_radii_comp[BLOCK_SIZE];
	
	// hgs
	__shared__ float4 collected_conic_opacity_hgs[BLOCK_SIZE];
	__shared__ float3 collected_normal[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };
	float pix_depth = 0;
	float pix_alpha = 0;

	//hgs
	float tanh_part1;
	float tanh_part2;
	float ratio1;
	float ratio2;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			collected_conic_opacity_hgs[block.thread_rank()] = conic_opacity_hgs[coll_id];
			collected_normal[block.thread_rank()] = saved_normal[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f) continue;
			// if( power >= 0.0f || power <= -4.8f) continue;

			float4 con_o_hgs = collected_conic_opacity_hgs[j];
			float3 norm_hgs = collected_normal[j];

			// erfc(x) = 1 - erf(x), erf(-x) = -erf(x), erfc(x) = 1 + erf(-x)
			// erf ≈ tanh
			float dot = erff(((norm_hgs.x+con_o_hgs.x)*d.x + (norm_hgs.y+con_o_hgs.y)*d.y)*con_o_hgs.z);
			// float dot = tanhf(0.886f * ((norm_hgs.x+con_o_hgs.x)*d.x + (norm_hgs.y+con_o_hgs.y)*d.y));
			// float dot = fmaf(norm_hgs.x + con_o_hgs.x, d.x,
            //     fmaf(norm_hgs.y + con_o_hgs.y, d.y, 0.0f));

			// 使用 tanhf 代替 erff
			// dot = dot * 0.886f;
			// 1.1 用 approx_tanhf 代替 tanhf/erff
			// tanh_part1 = 1.0f + approx_tanhf(dot);
			// 1.2 用 cuda 自带的 tanhf 代替 erff
			// tanh_part1 = 1.0f + tanhf(dot);

			// 使用 erff 
			// 2.1 直接调用 fast_erff，多项式/有理近似
			// tanh_part1 = 1.0f + fast_erff(dot);
			// 2.2 直接调用 erff
			tanh_part1 = 1.0f+erff(dot);
			tanh_part2 = 2.0f-tanh_part1;
			
		
			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.999f, con_o.w * exp(power));
            if (alpha < 0.f || alpha < 1.f / 255.f) {
                continue;
            }
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < 3; ch++)
				C[ch] += features[collected_id[j] * (CHANNELS+1) + ch] * alpha * T;
			C[3] += (features[collected_id[j] * (CHANNELS+1) + 3] * tanh_part1 + 
 					 features[collected_id[j] * (CHANNELS+1) + 4] * tanh_part2) * alpha * T;
			for (int ch = 4; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * (CHANNELS+1) + (ch+1)] * alpha * T;
			

			// pix_depth += geomState_depth[collected_id[j]] * alpha * T; 
			// pix_alpha += alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
	// rendering depth and alpha to the frame
	// if (inside){out_depth[pix_id] = pix_depth;out_alpha[pix_id] = pix_alpha;}
}



// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
original_renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ geomState_depth,
	float* __restrict__ out_depth,
	float* __restrict__ out_alpha,
	
	const float* radii_comp
	)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };
	float pix_depth = 0;
	float pix_alpha = 0;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.999f, con_o.w * exp(power));
            if (alpha < 0.f || alpha < 1.f / 255.f) {
                continue;
            }
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			pix_depth += geomState_depth[collected_id[j]] * alpha * T; 
			pix_alpha += alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
	// rendering depth and alpha to the frame
	if (inside)
	{
		out_depth[pix_id] = pix_depth;
		out_alpha[pix_id] = pix_alpha;
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* geomState_depth,
	float* out_depth,
	float* out_alpha,

	const float* radii_comp,

	const bool hgs,
	const float* hgs_opacities,
	const float3* saved_normal, 
	const float4* conic_opacity_hgs)
{	
	if (hgs)
	{
		hgs_renderCUDA<NUM_CHANNELS> << <grid, block, 0, MY_STREAM>> > (
			ranges,
			point_list,
			W, H,
			means2D,
			colors,
			conic_opacity,
			final_T,
			n_contrib,
			bg_color,
			out_color,
			geomState_depth,
			out_depth,
			out_alpha,
			
			radii_comp,
			
			hgs,
			hgs_opacities,
			saved_normal,
			conic_opacity_hgs);
	}
	else
	{
		original_renderCUDA<NUM_CHANNELS> << <grid, block, 0, MY_STREAM>> > (
			ranges,
			point_list,
			W, H,
			means2D,
			colors,
			conic_opacity,
			final_T,
			n_contrib,
			bg_color,
			out_color,
			geomState_depth,
			out_depth,
			out_alpha,
			
			radii_comp);
	}
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
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
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	float low_pass_filter_radius,

	// added
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
	uint4* conic_opacity6)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256, 0, MY_STREAM>> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
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
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered,
		low_pass_filter_radius,

		// added
		radii_comp,

		// hgs 相关
		hgs,
		conic_,
		hgs_normals,
		hgs_opacities,
		save_normal,
		cov3D_smalls,
		conic_opacity_hgs,
		conic_opacity_hgs_temp,
		conic_opacity3,
		conic_opacity4,
		conic_opacity5,
		conic_opacity6
		);
}

