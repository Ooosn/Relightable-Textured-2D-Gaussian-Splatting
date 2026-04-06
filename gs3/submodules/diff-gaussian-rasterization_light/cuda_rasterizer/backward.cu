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

#include "backward.h"
#include "auxiliary.h"
#include "stream.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
__global__ void Persp_computeCov2DCUDA(int P,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float h_x, float h_y,
	const float tan_fovx, float tan_fovy,
	const float* viewmatrix,
	const float* projmatrix,
	const float* dL_dconics,
	float3* dL_dmeans,
	float* dL_dcov,
	float* dL_dproj,
	float* dL_dview,
	const float low_pass_filter_radius
	)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.
	float3 mean = means[idx];
	float3 dL_dconic = { 
		dL_dconics[4 * idx + 0], 
		dL_dconics[4 * idx + 1], 
		dL_dconics[4 * idx + 3] 
	};

	float3 t = transformPoint4x3(mean, viewmatrix);
	
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
		0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;

	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Use helper variables for 2D covariance entries. More compact.
	float a = cov2D[0][0] += low_pass_filter_radius;
	float b = cov2D[0][1];
	float c = cov2D[1][1] += low_pass_filter_radius;

	float denom = a * c - b * b;
	float dL_da = 0, dL_db = 0, dL_dc = 0;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	if (denom2inv != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
		dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
		dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
		dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
		// dL_d
		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}

	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	// cov2D = transpose(T) * transpose(Vrk) * T;
	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
		(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
		(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
		(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
		(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
		(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
		(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;

	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
	// T = W * J
	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	float tz = 1.f / t.z;
	float tz2 = tz * tz;
	float tz3 = tz2 * tz;

	// Gradients of loss w.r.t. transformed Gaussian mean t
	float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
	float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
	float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;

	// Account for transformation of mean to t
	// t = transformPoint4x3(mean, view_matrix);
	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, viewmatrix);

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in BACKWARD::preprocess.
	dL_dmeans[idx] = dL_dmean;
}


__global__ void Ortho_computeCov2DCUDA(
    int P,
    const float3* __restrict__ means,      // 为了一致性保留，但此核不使用
    const int*    __restrict__ radii,
    const float*  __restrict__ cov3Ds,     // packed 6: [xx,xy,xz,yy,yz,zz]
    float h_x, float h_y,
	const float tan_fovx, float tan_fovy,
    const float*  __restrict__ viewmatrix, // 为了一致性保留，但此核不使用
    const float*  __restrict__ projmatrix, // 用到 3x3 部分
    const float*  __restrict__ dL_dconics, // packed 4: [A,B,?,C] → 取 [0,1,3]
    float3*       __restrict__ dL_dmeans,  // 经由本链路为 0
    float*        __restrict__ dL_dcov,    // 写回 6 个分量
	float*        __restrict__ dL_dproj,   // 写回 3x3 部分
	float*        __restrict__ dL_dview,   // 写回 3x3 部分
    const float low_pass_filter_radius     // 必须与前向一致	
	)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P || !(radii[idx] > 0)) return;

	// 计算 backward 准备数据：

	// 1. 投影变换
    glm::mat3 Proj = glm::mat3(
        projmatrix[0], projmatrix[4], projmatrix[8],
        projmatrix[1], projmatrix[5], projmatrix[9],
        projmatrix[2], projmatrix[6], projmatrix[10]
    );

   
	// 2. 构造 A = Proj * S
    // Reconstruct the orthographic screen scale used in forward.
    const float sx = h_x * tan_fovx;
    const float sy = h_y * tan_fovy;
	
    glm::mat3 S  = glm::mat3(
        sx, 0,  0,
        0,  sy, 0,
        0,  0,  1
    );

    glm::mat3 A = Proj * S;

	// 取 3D 协方差（对称）指针（packed 6: [xx,xy,xz,yy,yz,zz]）
	const float* cov3D = cov3Ds + 6*idx;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]
	);
	
    // cov2D = A^T * Vrk^T * A  （Vrk 对称，T不影响）
    glm::mat3 cov2D = glm::transpose(A) * glm::transpose(Vrk) * A;

	// 从 conic = inv(cov2D) 的梯度回传到 (a,b,c)
    // dL_dconic 打包为 [A,B,*,C]（与透视版一致：取 0,1,3）
    const float3 dL_dconic = {
        dL_dconics[4*idx + 0],   // dL/dA
        dL_dconics[4*idx + 1],   // dL/dB
        dL_dconics[4*idx + 3]    // dL/dC
    };

    // 取 2x2：a,b,c，并加 low_pass_filter_radius
    float a = cov2D[0][0] += low_pass_filter_radius;
    float b = cov2D[0][1];
    float c = cov2D[1][1] += low_pass_filter_radius;

    const float denom   = a*c - b*b;
	float dL_da = 0.f, dL_db = 0.f, dL_dc = 0.f;
    const float denom2  = denom * denom + 1e-7f;
    const float denom2inv = 1.0f / denom2;

	// 开始回传
    if (denom2inv != 0.f) {
        // 与透视版完全一致
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
        dL_da = denom2inv * ( -c*c * dL_dconic.x + 2*b*c * dL_dconic.y + (denom - a*c) * dL_dconic.z );
        dL_dc = denom2inv * ( -a*a * dL_dconic.z + 2*a*b * dL_dconic.y + (denom - a*c) * dL_dconic.x );
        dL_db = denom2inv * 2.0f * ( b*c * dL_dconic.x - (denom + 2*b*b) * dL_dconic.y + a*b * dL_dconic.z );

        // 回传到 3D 协方差（与透视版相同公式，只把 T → A）
		// 比直接 6 * idx + ？ 更高效
		// 1) 回传到 Vrk（把 dL/dcov2D 乘以 A，形式与原版 T 一致）
		// 先按原版分量写法，保持一致性
		dL_dcov[6*idx+0] = (A[0][0]*A[0][0]*dL_da + A[0][0]*A[1][0]*dL_db + A[1][0]*A[1][0]*dL_dc);
		dL_dcov[6*idx+3] = (A[0][1]*A[0][1]*dL_da + A[0][1]*A[1][1]*dL_db + A[1][1]*A[1][1]*dL_dc);
		dL_dcov[6*idx+5] = (A[0][2]*A[0][2]*dL_da + A[0][2]*A[1][2]*dL_db + A[1][2]*A[1][2]*dL_dc);

		// Off-diagonal（出现两次 → ×2），与原版完全同逻辑
		dL_dcov[6*idx+1] = 2*A[0][0]*A[0][1]*dL_da + (A[0][0]*A[1][1]+A[0][1]*A[1][0])*dL_db + 2*A[1][0]*A[1][1]*dL_dc;
		dL_dcov[6*idx+2] = 2*A[0][0]*A[0][2]*dL_da + (A[0][0]*A[1][2]+A[0][2]*A[1][0])*dL_db + 2*A[1][0]*A[1][2]*dL_dc;
		dL_dcov[6*idx+4] = 2*A[0][2]*A[0][1]*dL_da + (A[0][1]*A[1][2]+A[0][2]*A[1][1])*dL_db + 2*A[1][1]*A[1][2]*dL_dc;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}
	// 2) 求 dL/dA —— 这一步等价于原版的 dL_dT00...dL_dT12
	// 用矩阵式更清楚：G = [[dL_xx, 0.5*dL_xy, 0],[0.5*dL_xy, dL_yy, 0],[0,0,0]]
	glm::mat3 G = glm::mat3(
		dL_da, 0.5f*dL_db, 0.0f,
		0.5f*dL_db, dL_dc, 0.0f,
		0.0f, 0.0f, 0.0f
	);
	// Vrk 对称，故 dL/dA = 2 * Vrk * A * G
	glm::mat3 dL_dA = 2.0f * Vrk * A * G;

	// 3) A = P * S ⇒ dL/dP = dL/dA * S^T = dL/dA * S（S 对角）
	glm::mat3 dL_dP = dL_dA * glm::transpose(S); // S 对角，转置等于自身
	CHECK_NAN(dL_dP[0][0]);
	CHECK_NAN(dL_dP[0][1]);
	CHECK_NAN(dL_dP[0][2]);
	CHECK_NAN(dL_dP[1][0]);
	CHECK_NAN(dL_dP[1][1]);
	CHECK_NAN(dL_dP[1][2]);
	CHECK_NAN(dL_dP[2][0]);
	CHECK_NAN(dL_dP[2][1]);
	CHECK_NAN(dL_dP[2][2]);
	// 4) 把 3x3 的 dL_dP 写回 projmatrix 的相应槽位（其他槽位可清零或不动）
	// 注意构造时的列主序映射：前向是 (0,4,8; 1,5,9; 2,6,10)
	atomicAdd(&dL_dproj[0],  dL_dP[0][0]); // m00
	atomicAdd(&dL_dproj[4],  dL_dP[0][1]); // m01
	atomicAdd(&dL_dproj[8],  dL_dP[0][2]); // m02

	atomicAdd(&dL_dproj[1],  dL_dP[1][0]); // m10
	atomicAdd(&dL_dproj[5],  dL_dP[1][1]); // m11
	atomicAdd(&dL_dproj[9],  dL_dP[1][2]); // m12

	atomicAdd(&dL_dproj[2],  dL_dP[2][0]); // m20
	atomicAdd(&dL_dproj[6],  dL_dP[2][1]); // m21
	atomicAdd(&dL_dproj[10], dL_dP[2][2]); // m22
}

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}


__device__ void computeProjGrad(const float3 m, const float m_w, const float mul1, const float mul2, 
								const float gu, const float gv, float* __restrict__ dL_dproj)
								
{
	const float x = m.x, y = m.y, z = m.z;
    const float s    = gu*mul1 + gv*mul2;
	CHECK_NAN(gu);
	CHECK_NAN(gv);
	CHECK_NAN(m_w);
	CHECK_NAN(mul1);
	CHECK_NAN(mul2);

    // row0 -> u 的分子（索引 0,4,8,12）
    atomicAdd(&dL_dproj[0],  gu * x * m_w);
    atomicAdd(&dL_dproj[4],  gu * y * m_w);
    atomicAdd(&dL_dproj[8],  gu * z * m_w);
    atomicAdd(&dL_dproj[12], gu * 1.f * m_w);

    // row1 -> v 的分子（索引 1,5,9,13）
    atomicAdd(&dL_dproj[1],  gv * x * m_w);
    atomicAdd(&dL_dproj[5],  gv * y * m_w);
    atomicAdd(&dL_dproj[9],  gv * z * m_w);
    atomicAdd(&dL_dproj[13], gv * 1.f * m_w);

    // row3 -> 分母（索引 3,7,11,15）
    atomicAdd(&dL_dproj[3],  -s * x);
    atomicAdd(&dL_dproj[7],  -s * y);
    atomicAdd(&dL_dproj[11], -s * z);
    atomicAdd(&dL_dproj[15], -s * 1.f);

    // row2（2,6,10,14）只有在 loss 用到了 z=a_z/d 时才需要（见下“扩展”）。
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	glm::vec3* dL_dmeans,
	float* dL_dcolor,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	float* dL_dproj,
	float* dL_dview, // 当考虑 z/depth 的梯度时，会纳入
	const bool ortho,
	const float low_pass_filter_radius)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = means[idx];

	// Taking care of gradients from the screenspace points
	float4 m_hom = transformPoint4x4(m, projmatrix);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	glm::vec3 dL_dmean;
	const float gu = dL_dmean2D[idx].x;
	const float gv = dL_dmean2D[idx].y;
	float mul1 = (projmatrix[0] * m.x + projmatrix[4] * m.y + projmatrix[8] * m.z + projmatrix[12]) * m_w * m_w;
	float mul2 = (projmatrix[1] * m.x + projmatrix[5] * m.y + projmatrix[9] * m.z + projmatrix[13]) * m_w * m_w;
	dL_dmean.x = (projmatrix[0] * m_w - projmatrix[3] * mul1) * gu + (projmatrix[1] * m_w - projmatrix[3] * mul2) * gv;
	dL_dmean.y = (projmatrix[4] * m_w - projmatrix[7] * mul1) * gu + (projmatrix[5] * m_w - projmatrix[7] * mul2) * gv;
	dL_dmean.z = (projmatrix[8] * m_w - projmatrix[11] * mul1) * gu + (projmatrix[9] * m_w - projmatrix[11] * mul2) * gv;


	computeProjGrad(m, m_w, mul1, mul2, gu, gv, dL_dproj);

	// That's the second part of the mean gradient. Previous computation
	// of cov2D and following SH conversion also affects it.
	dL_dmeans[idx] += dL_dmean;

	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh);

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ depths,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_invdepths,

	const float* __restrict__ out_trans,
	const float* __restrict__ non_trans,
	const float* __restrict__ dL_dout_trans,
	const float* __restrict__ dL_dnon_trans,

	float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float* __restrict__ dL_dinvdepths, // 占位

	// added
	const float* __restrict__ radii_comp,
	const float offset,

	// for accelerate and verify point order
	const uint32_t* __restrict__ id_contrib,
	const uint32_t* __restrict__ n_contrib_offset,
	const uint32_t* __restrict__ id_contrib_offset
	)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int range_start = static_cast<int>(range.x);
	const int range_end = static_cast<int>(range.y);

	const int rounds = ((range_end - range_start + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range_end - range_start;

	// original
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_depth[BLOCK_SIZE];
	__shared__ float collected_dL_dout_trans[BLOCK_SIZE];
	__shared__ float collected_dL_dnon_trans[BLOCK_SIZE];
	// __shared__ float collected_dL_dopacity[BLOCK_SIZE];
	// __shared__ float collected_out_trans[BLOCK_SIZE];
	__shared__ float collected_non_trans[BLOCK_SIZE];

	// accelerate
	__shared__ float collected_radii_comp[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	uint32_t contributor_pointT = 0;
	uint32_t last_contributor_pointT = 0;
	float w = T;
	// float C[CHANNELS] = { 0 };

	// Iterate over batches until all done or range is complete
	// offset point id
	int pointT = 0;
	// offset point depth
	float depthT = 0.0f;

	// addded
	float temp = 1.0f;
	float depth_diff = 0.0f;
	float test_T2 = 0.0f;
	float alpha = 0.0f;


   
	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	contributor = toDo;
	last_contributor = inside ? n_contrib[pix_id] : 0;

	//backward offset transmitance
	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	const int final_T_chain_count = inside ? static_cast<int>(n_contrib_offset[pix_id]) : 0;
	float T_offset = T_final;
	float T_self = T_final;
	// T_final only contains factors before the last offset window boundary.
	// Reconstruct T_* from that boundary instead of from the last contributor.
	int T_self_index = range_start + final_T_chain_count - 1;
	int offset_index = range_start + last_contributor - 1;
	int offset_T_index = range_start + final_T_chain_count - 1;

	// backward component
	float accd_Grad = 0.0f;
	float dL_dT = 0.0f;

	int gradient_rrepeat = 0;



	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		// 因为反向读取，每个线程不是读取自己的数据，所以需要 block.sync() 等待所有线程都写入完毕
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range_start + progress < range_end)
		{
			const int point_index = range_end - progress - 1;
			const int coll_id = point_list[point_index];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			collected_dL_dout_trans[block.thread_rank()] = dL_dout_trans[coll_id];
			collected_dL_dnon_trans[block.thread_rank()] = dL_dnon_trans[coll_id];
			collected_non_trans[block.thread_rank()] = non_trans[coll_id];
			collected_depth[block.thread_rank()] = depths[coll_id];
			collected_radii_comp[block.thread_rank()] = radii_comp[coll_id];
			// for (int i = 0; i < C; i++)
			// 	collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; j < min(BLOCK_SIZE, toDo); j++)
		{
			if (!inside || contributor == 0)
				break;

			contributor--;
			const bool current_has_out = contributor < last_contributor;

			// 计算高斯粒子属性
			const float2 xy_j = collected_xy[j];
			const float2 d_j = { xy_j.x - pixf.x, xy_j.y - pixf.y };
			float dd =d_j.x*d_j.x+d_j.y*d_j.y;
			if (dd >= collected_radii_comp[j]) continue;
			const float4 con_o_j = collected_conic_opacity[j];
			const float power_j = -0.5f * (con_o_j.x * d_j.x * d_j.x + con_o_j.z * d_j.y * d_j.y) - con_o_j.y * d_j.x * d_j.y;
			if (power_j > 0.0f) continue;
			


			// 非法值处理
			const float G = exp(power_j);
			float alpha = min(0.99f, con_o_j.w * G);
			bool alpha_valid = true;
			if (alpha < 1.0f / 255.0f)
			{
				alpha = 0.0f;
				alpha_valid = false;
			}
			if (current_has_out)
			{
				while (T_self_index >= range_start &&
				collected_depth[j] - depths[point_list[T_self_index]] <= offset ) {
				const int coll_id_self_T = point_list[T_self_index];
				const float2 xy_self_T = points_xy_image[coll_id_self_T];
				const float2 d_self_T = { xy_self_T.x - pixf.x, xy_self_T.y - pixf.y };
				const float4 con_o_self_T = conic_opacity[coll_id_self_T];
				const float power_self_T = -0.5f * (con_o_self_T.x * d_self_T.x * d_self_T.x + con_o_self_T.z * d_self_T.y * d_self_T.y) - con_o_self_T.y * d_self_T.x * d_self_T.y;

				float alpha_self_T=0.0f;
				float G_self_T=0.0f;
				if (power_self_T <= 0.0f)
				{
					G_self_T = exp(power_self_T);
					alpha_self_T = min(0.99f, con_o_self_T.w * G_self_T);
					if (alpha_self_T < 1.0f / 255.0f)
						alpha_self_T = 0.0;
				}
				float temp = 1.0f - alpha_self_T;
				T_self = T_self/temp;
				CHECK_NAN(T_self)
				--T_self_index;
			}
			

			while (offset_index >= range_start && 
				depths[point_list[offset_index]] - collected_depth[j] > offset) {
				const int coll_id_offset = point_list[offset_index];
				const float2 xy_offset = points_xy_image[coll_id_offset];
				const float2 d_offset = { xy_offset.x - pixf.x, xy_offset.y - pixf.y };
				const float4 con_o_offset = conic_opacity[coll_id_offset];
				const float power_offset = -0.5f * (con_o_offset.x * d_offset.x * d_offset.x + con_o_offset.z * d_offset.y * d_offset.y) - con_o_offset.y * d_offset.x * d_offset.y;		
				
				float G_offset=0.0f;
				if (power_offset <= 0.0f)
				{
					G_offset = exp(power_offset);
					if (G_offset < 1.0f / 255.0f)
						G_offset = 0.0f;
				}

				// accd_Grad(i) = Σ_{k>i, d_k-d_i>offset} (G_k T_k) * dL_dout_k
				// T_k = Π_{k>j, d_k-d_j>offset} (1 - α_j)
				while (offset_T_index >= range_start && 
					depths[point_list[offset_index]] - depths[point_list[offset_T_index]] <= offset) {
					const int coll_id_offset_T = point_list[offset_T_index];
					const float2 xy_offset_T = points_xy_image[coll_id_offset_T];
					const float2 d_offset_T = { xy_offset_T.x - pixf.x, xy_offset_T.y - pixf.y };
					const float4 con_o_offset_T = conic_opacity[coll_id_offset_T];
					const float power_offset_T= -0.5f * (con_o_offset_T.x * d_offset_T.x * d_offset_T.x + con_o_offset_T.z * d_offset_T.y * d_offset_T.y) - con_o_offset_T.y * d_offset_T.x * d_offset_T.y;
					
					float alpha_offset_T=0.0f;
					float G_offset_T=0.0f;
					if (power_offset_T <= 0.0f)
					{
						G_offset_T = exp(power_offset_T);
						alpha_offset_T = min(0.99f, con_o_offset_T.w * G_offset_T);
						if (alpha_offset_T < 1.0f / 255.0f)
							alpha_offset_T = 0.0f;
					}
					float temp = 1.0f - alpha_offset_T;
					T_offset = T_offset/temp;
					--offset_T_index;
				}

				const float dd_offset = d_offset.x * d_offset.x + d_offset.y * d_offset.y;
				if (dd_offset < radii_comp[coll_id_offset])
					accd_Grad += G_offset * T_offset * dL_dout_trans[coll_id_offset];
				--offset_index;
			}
			}
			
			
			// Gradient of pixel coordinate w.r.t. normalized 
			// screen-space viewport corrdinates (-1 to 1)
			const float ddelx_dx = 0.5f * W;
			const float ddely_dy = 0.5f * H;

			const int global_id = collected_id[j];
			float self_grad_term = collected_dL_dnon_trans[j];
			float other_grad_term_opacity = 0.0f;
			float other_grad_term = 0.0f;
			if (current_has_out)
			{
				const float inv_denom = 1.f / fmaxf(1.f - alpha, 1e-6f);
				self_grad_term += collected_dL_dout_trans[j] * T_self;
				other_grad_term_opacity = - accd_Grad * inv_denom;
				const float dalpha_dG = alpha_valid ? con_o_j.w : 0.0f; // gating
				other_grad_term = - accd_Grad * (dalpha_dG * inv_denom);
				CHECK_NAN(accd_Grad);
				CHECK_NAN(T_self);
			}
			// 合并得到 dL/dG_i
			const float dL_dG_total = self_grad_term + other_grad_term;
			const float gdx = G * d_j.x;
			const float gdy = G * d_j.y;

			// 对 2D 均值（屏幕坐标）
			// dG/dx_ndc = dG/ddelx, 乘上 ddelx/dx_pix = 0.5*W
			// 在这里已经考虑了前向的 ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H)
			const float dG_ddelx = -(con_o_j.x * gdx + con_o_j.y * gdy);
			const float dG_ddely = -(con_o_j.y * gdx + con_o_j.z * gdy);
			atomicAdd(&dL_dmean2D[global_id].x, dL_dG_total * dG_ddelx * ddelx_dx);
			atomicAdd(&dL_dmean2D[global_id].y, dL_dG_total * dG_ddely * ddely_dy);

			// 对协方差（conic: A=xx, B=xy, C=yy）
			// power = -0.5*A*dx^2 - B*dx*dy - 0.5*C*dy^2
			// => ∂G/∂A = -0.5*G*dx^2, ∂G/∂B = -G*dx*dy, ∂G/∂C = -0.5*G*dy^2
			atomicAdd(&dL_dconic2D[global_id].x, dL_dG_total * (-0.5f) * gdx * d_j.x);  // dA
			atomicAdd(&dL_dconic2D[global_id].y, dL_dG_total * (-0.5f) * gdx * d_j.y);  // dB
			atomicAdd(&dL_dconic2D[global_id].w, dL_dG_total * (-0.5f) * gdy * d_j.y);  // dC
			CHECK_NAN(dL_dG_total);
			CHECK_NAN(dG_ddelx);
			CHECK_NAN(ddely_dy);

			atomicAdd(&dL_dopacity[global_id], (current_has_out && alpha_valid) ? (G * other_grad_term_opacity) : 0.0f);

		}
	}
}






void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const float* opacities,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	const float* dL_dconic,
	const float* dL_dinvdepth,
	float* dL_dopacity,
	glm::vec3* dL_dmean3D,
	float* dL_dcolor,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	float* dL_dproj,
	float* dL_dview,
	const bool ortho,
	const float low_pass_filter_radius)
{
	// Propagate gradients for the path of 2D conic matrix computation. 
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	

	if (ortho)
	{
		Ortho_computeCov2DCUDA << <(P + 255) / 256, 256, 0, MY_STREAM >> > (
			P,
			means3D,
			radii,
			cov3Ds,
			focal_x,
			focal_y,
			tan_fovx,
			tan_fovy,
			viewmatrix,
			projmatrix,
			dL_dconic,
			(float3*)dL_dmean3D,
			dL_dcov3D,
			dL_dproj,
			dL_dview,
			low_pass_filter_radius);
	}
	else
	{
		Persp_computeCov2DCUDA << <(P + 255) / 256, 256, 0, MY_STREAM >> > (
			P,
			means3D,
			radii,
			cov3Ds,
			focal_x,
			focal_y,
			tan_fovx,
			tan_fovy,
			viewmatrix,
			projmatrix,
			dL_dconic,
			(float3*)dL_dmean3D,
			dL_dcov3D,
			dL_dproj,
			dL_dview,
			low_pass_filter_radius);
	}

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256, 0, MY_STREAM >> > (
		P, D, M,
		(float3*)means3D,
		radii,
		shs,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		viewmatrix,
		projmatrix,
		campos,
		(float3*)dL_dmean2D,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		dL_dscale,
		dL_drot,
		dL_dproj,
		dL_dview,
		ortho,
		low_pass_filter_radius);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float* bg_color,
	const float2* means2D,
	const float4* conic_opacity,
	const float* colors,
	const float* depths,
	const float* final_Ts,
	const uint32_t* n_contrib,

	const float* dL_dpixels,
	const float* dL_invdepths,

	const float* out_trans,
	const float* non_trans,
	const float* dL_dout_trans,
	const float* dL_dnon_trans,

	float3* dL_dmean2D,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dcolors,
	float* dL_dinvdepths,

	// added
	const float* radii_comp,
	const float offset,

	const uint32_t* id_contrib,
	const uint32_t* n_contrib_offset,
	const uint32_t* id_contrib_offset)
{
	renderCUDA<NUM_CHANNELS> << <grid, block, 0, MY_STREAM >> >(
		ranges,
		point_list,
		W, H,
		bg_color,
		means2D,
		conic_opacity,
		colors,
		depths,
		final_Ts,
		n_contrib,
		dL_dpixels,
		dL_invdepths,

		out_trans,
		non_trans,
		dL_dout_trans,
		dL_dnon_trans,

		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dcolors,
		dL_dinvdepths,

		// added
		radii_comp,
		offset,

		// for accelerate and verify point order
		id_contrib,
		n_contrib_offset,
		id_contrib_offset
		);
}
