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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

#define CHECK_INPUT(x)											\
	AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
	// AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
	auto lambda = [&t](size_t N) {
		t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
	};
	return lambda;
}

torch::Tensor RasterizeColorCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& colors,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& transMat_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& texture_color,
	const torch::Tensor& texture_alpha,
	const torch::Tensor& texture_shadow,
	const float texture_sigma_factor,
	const bool prefiltered,
	const bool debug)
{
	const int P = means3D.size(0);
	const int H = image_height;
	const int W = image_width;

	auto float_opts = means3D.options().dtype(torch::kFloat32);
	torch::Tensor out_color = torch::zeros({NUM_CHANNELS, H, W}, float_opts);

	torch::Device device(torch::kCUDA);
	torch::TensorOptions options(torch::kByte);
	torch::Tensor geomBuffer   = torch::empty({0}, options.device(device));
	torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
	torch::Tensor imgBuffer    = torch::empty({0}, options.device(device));
	std::function<char*(size_t)> geomFunc    = resizeFunctional(geomBuffer);
	std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
	std::function<char*(size_t)> imgFunc     = resizeFunctional(imgBuffer);

	if (P != 0)
	{
		int M = (sh.size(0) != 0) ? sh.size(1) : 0;
		const int tex_res = static_cast<int>(
			texture_alpha.numel() == 0 ? 0 : texture_alpha.size(2));

		CudaRasterizer::Rasterizer::forwardColor(
			geomFunc, binningFunc, imgFunc,
			P, degree, M,
			background.contiguous().data<float>(),
			W, H,
			means3D.contiguous().data<float>(),
			sh.contiguous().data_ptr<float>(),
			colors.contiguous().data<float>(),
			opacity.contiguous().data<float>(),
			scales.contiguous().data_ptr<float>(),
			scale_modifier,
			rotations.contiguous().data_ptr<float>(),
			transMat_precomp.contiguous().data<float>(),
			viewmatrix.contiguous().data<float>(),
			projmatrix.contiguous().data<float>(),
			campos.contiguous().data<float>(),
			texture_color.numel() == 0 ? nullptr : texture_color.contiguous().data_ptr<float>(),
			texture_alpha.numel() == 0 ? nullptr : texture_alpha.contiguous().data_ptr<float>(),
			texture_shadow.numel() == 0 ? nullptr : texture_shadow.contiguous().data_ptr<float>(),
			tex_res,
			texture_sigma_factor,
			tan_fovx, tan_fovy,
			prefiltered,
			out_color.contiguous().data<float>(),
			nullptr,   // radii – not needed for color-only render
			debug);
	}
	return out_color;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& colors,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& transMat_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& texture_alpha,
	const float texture_sigma_factor,
	const bool prefiltered,
	const bool debug,
	const torch::Tensor& non_trans,
	const float offset,
	const float thres,
	const bool is_train)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
	AT_ERROR("means3D must have dimensions (num_points, 3)");
  }

  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  CHECK_INPUT(background);
  CHECK_INPUT(means3D);
  CHECK_INPUT(colors);
  CHECK_INPUT(opacity);
  CHECK_INPUT(scales);
  CHECK_INPUT(rotations);
  CHECK_INPUT(transMat_precomp);
  CHECK_INPUT(viewmatrix);
  CHECK_INPUT(projmatrix);
  CHECK_INPUT(sh);
  CHECK_INPUT(campos);
  if (texture_alpha.numel() != 0) CHECK_INPUT(texture_alpha);
  if (non_trans.numel() != 0) CHECK_INPUT(non_trans);

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::zeros({NUM_CHANNELS, H, W}, float_opts);
  torch::Tensor out_weight = torch::zeros({P, 1}, float_opts);
  // When a texture_alpha with a valid resolution is provided, accumulate shadow
  // per UV texel (layout matches texture_alpha_index: [P * res * res] flat).
  // Otherwise fall back to the original per-Gaussian scalar layout [P].
  const int tex_res_shadow = static_cast<int>(
      texture_alpha.numel() == 0 ? 0 : texture_alpha.size(2));
  const long shadow_buf_size = (tex_res_shadow > 0)
      ? static_cast<long>(P) * tex_res_shadow * tex_res_shadow
      : static_cast<long>(P);
  torch::Tensor out_trans = torch::zeros({shadow_buf_size}, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
	  }

	  rendered = CudaRasterizer::Rasterizer::forward(
		geomFunc,
		binningFunc,
		imgFunc,
		P, degree, M,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(), 
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		transMat_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		texture_alpha.numel() == 0 ? nullptr : texture_alpha.contiguous().data_ptr<float>(),
		static_cast<int>(texture_alpha.numel() == 0 ? 0 : texture_alpha.size(2)),
		texture_sigma_factor,
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data<float>(),
		out_weight.contiguous().data<float>(),
		out_trans.contiguous().data<float>(),
		non_trans.contiguous().data<float>(),
		offset,
		thres,
		is_train,
		radii.contiguous().data<int>(),
		debug);
  }
  return std::make_tuple(rendered, out_color, out_weight, radii, out_trans, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
	 const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
	const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& transMat_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_others,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& texture_color,
	const torch::Tensor& texture_alpha,
	const float texture_sigma_factor,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug) 
{
  AT_ERROR("diff_surfel_rasterization_shadow backward is not implemented yet.");
  return std::make_tuple(
      torch::Tensor(),
      torch::Tensor(),
      torch::Tensor(),
      torch::Tensor(),
      torch::Tensor(),
      torch::Tensor(),
      torch::Tensor(),
      torch::Tensor(),
      torch::Tensor(),
      torch::Tensor());
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}
