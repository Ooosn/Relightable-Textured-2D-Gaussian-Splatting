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

// added: trans



// char* 也许会认为是 字符串 的指针，但其实他也是最低级的指针类型，可用于 逐字节操作数据 的指针。
// 指针的最小单位 是 字节，char* 是 1字节，可以保证 指针运算的正确性

// 返回 std::function<char*(size_t N)>  类型 ————  返回一个 函数，这个函数接受一个 size_t 类型的参数 N，返回一个 char* 类型的指针。
// 这个函数的作用是 ————  当需要调整张量 t 的大小时，使用这个函数来调整张量的大小，并返回一个指向调整后张量数据的指针。
// size_t ————  unsigned int 类型，一般用于表示一个对象的大小（以字节为单位），也就是表达一个变量的 size
// 因此 resizeFunctional 函数 ————  接受一个 torch::Tensor 类型的参数 t，并返回一个 std::function<char*(size_t N)> 类型的函数。 
std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    // 内部 定义 lambda 表达式
	// auto lambda = ... 让编译器推导 lambda 的类型
	/* 
	* lamda 表达式 ————  [] 捕捉外部变量，() 需要输入的参数，{} 函数体
	* lamda 表达式 默认是 const 的，如果需要修改外部变量，需要使用 mutable & 关键字
	* & 的优先级 高于 const ，即便 变量 是 const 的，也可以修改
	*/
	auto lambda = [&t](size_t N) {

		// torch 函数后面跟_，表示 原地修改
		// (type)variable ————  将 variable 转换为 type 类型
		// long long ———— int64_t 类型
		// {} ———— 统一初始化方式，根据上下文自动推导
		/*
		 * size_t ————  unsigned int 类型，但是 resize_ 的参数需要 long long 类型，
		 * 因此需要将 size_t 转换为 long long 类型
		 * 所以这里的指令，是 将 t 的大小 调整为 N 字节
		*/
        t.resize_({(long long)N});
		// reinterpret_cast<char*> ————  直接改变指针类型，将 t 的内存地址转换为 char* 类型
		// t.contiguous().data_ptr<T>() ————  返回 t 所控制的数据的内存地址，类型为 T*
		/*
		 * data_ptr<>() 是 torch::Tensor 的成员函数，返回其控制的数据在内存中的地址，类型为 T*，
		 *			     用于将数据以裸指针形式传入核函数（CUDA kernel），因为 CUDA 只能处理最基础的 C 风格指针。
		 * void* 只能用于存储地址，不能用于运算
		 * reinterpret_cast<> ————  让 void* 变成可用的类型，比如 reinterpret_cast<> 告诉编译器：“我明确知道这个 void* 其实是 int* 类型，请放心使用”
		*/ 
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
	// 返回 lambda 表达式
    return lambda;
}


/*
 * 函数：用 cuda 实现 高斯分布的 光栅化
 * 接收多个 torch::Tensor 作为输入（比如 3D 坐标、颜色、透明度等）
 * 创建 CUDA 设备上的缓冲区（geomBuffer、binningBuffer、imgBuffer）
 * 调用 CudaRasterizer::Rasterizer::forward() 进行 CUDA 计算
 * 返回多个 torch::Tensor 结果，比如颜色、权重、半径、变换信息等
*/ 
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	
	const torch::Tensor& background,// 背景颜色
	const torch::Tensor& means3D,	// 3D 坐标
    const torch::Tensor& colors,	// 颜色
    const torch::Tensor& opacities,	// 透明度
	const torch::Tensor& scales,	// 缩放
	const torch::Tensor& rotations,	// 旋转
	const float scale_modifier,	// 缩放比例
	const torch::Tensor& cov3Ds_precomp,	// 协方差矩阵
	const torch::Tensor& viewmatrix,	// 视图矩阵
	const torch::Tensor& projmatrix,	// 投影矩阵
	const float tan_fovx, 	// 视图矩阵的 x 方向的切线
	const float tan_fovy,	// 视图矩阵的 y 方向的切线
    const int image_height,	// 图像高度
    const int image_width,	// 图像宽度
	const torch::Tensor& shs,	// sh 系数
	const int degree,	// sh 阶数
	const torch::Tensor& campos,	// 相机位置
	const bool prefiltered,	// 预过滤， 默认 false
	const bool debug,	// 调试， 默认 false
	const float low_pass_filter_radius,

	// 投影方式
	const bool ortho,	// 是否为正交投影

	// 阴影相关
	const torch::Tensor& non_trans,	// 非透明
	
	const float offset,	// 偏移点深度
	const float thres,	// 高斯阈值

	// prune 相关
	const bool is_train,

	// hgs 相关
	const bool hgs,
	const torch::Tensor& hgs_normals,
	const torch::Tensor& hgs_opacities,
	const torch::Tensor& hgs_opacities_shadow,
	const torch::Tensor& hgs_opacities_light
	)
{
// 检查 means3D 的维度是否为 2，且第二维的大小是否为 3
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
// 定义常量：高斯点数量
  const int P = means3D.size(0);
// 定义常量：图像高度
  const int H = image_height;
// 定义常量：图像宽度
  const int W = image_width;

/*
* torch::TensorOptions ————  用于指定张量的数据类型、设备类型等属性。
* options = a.options() ————  返回 a 的 torch::TensorOptions 对象
* options.dtype() ————  返回该对象的 dtype 属性
* options.device() ————  返回该对象的 device 属性
*  ......
* 如果 options.dtype(?) 并不是空括号，则返回 a 的 torch::TensorOptions 对象，但 dtype 属性为 ?，其他同理
* 相当于在修改的基础上，继承了 a 的其他属性
*/
// 获得 means3D 的 TensorOptions ， 并设置 dtype 为 int32
// 获得 means3D 的 TensorOptions ， 并设置 dtype 为 float32
// 然后 可以直接用 int_opts 和 float_opts 来创建张量，从而继承这些其他的 means3D 的属性
  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

/*
* torch::full(size, value, options) ————  创建一个全为 value 的张量，并指定其数据类型和设备类型
* size ————  张量的形状 {}
* value ————  张量中每个元素的值，比如 0.0 表示每个元素的值为 0.0
* options ————  张量的数据类型和设备类型，用 torch::TensorOptions 对象来指定。
*/
// const int P = means3D.size(0);
// 初始化 radii，out_color，out_weight，trans
  torch::Tensor radii = torch::full({P}, 0, int_opts);
  torch::Tensor out_color = torch::full({2,2,2}, 0.0, float_opts);
  torch::Tensor out_weight = torch::full({P, 1}, 0.0, float_opts);
// out_trans 的维度 ———— 如果 hgs 为 true，则维度为 (P, 2)，否则为 (P, 1)
  torch::Tensor trans = torch::full({P, 1}, 0.0, float_opts);

  

  // torch::Device 和 torch::TensorOptions class 类型，（）内为构造函数声明
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);

  // 初始化 三个缓冲区
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  // 初始化 三个缓冲区的 调整函数
  // 调整函数 ————  接受一个 size_t 类型的参数 N，返回一个 char* 类型的指针，这里用于调整 缓冲区的大小
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  // .data<float>() 和 .data_ptr<float>() 的区别 ———— 没有区别，data_ptr() 更新

  int rendered = 0;
  // const int P = means3D.size(0);
  // 如果高斯点数量不为 0，则进行光栅化
  if(P != 0)
  {
	  // 根据 sh 系数的维度，初始化 M
	  // sh 维度为 （num_points, num_coefficients）
	  int M = 0;
	  if(shs.size(0) != 0)
	  {
		M = shs.size(1);
      }


	  // 调用 CudaRasterizer::Rasterizer::forward() 进行 CUDA 计算
	  rendered = CudaRasterizer::Rasterizer::forward(

		// 缓冲区调整函数
	    geomFunc,
		binningFunc,
		imgFunc,

		// 输入参数
		// 高斯点数量、sh 阶数、sh 系数数量
	    P, degree, M,
		// 背景颜色
		background.contiguous().data<float>(),
		// 图像高度和宽度
		W, H,
		means3D.contiguous().data<float>(),
		shs.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(), 
		opacities.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3Ds_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		debug,
		low_pass_filter_radius,

		// return 的 四个张量
		out_color.contiguous().data<float>(),	// FORWARD::render 
		out_weight.contiguous().data<float>(),	// FORWARD::render
		radii.contiguous().data<int>(),		// FORWARD::preprocess
		trans.contiguous().data<float>(),	// FORWARD::render

		// 其他参数
		non_trans.contiguous().data<float>(),	// FORWARD::render, 也会被修改,加权平均使用
		ortho,
		offset,
		thres,
		is_train,

		// hgs 相关
		hgs,
		hgs_normals.contiguous().data<float>(),
		hgs_opacities.contiguous().data<float>(),	
		hgs_opacities_shadow.contiguous().data<float>(),
		hgs_opacities_light.contiguous().data<float>()
		);
  }

  // 返回 前向传播的 结果
  return std::make_tuple(rendered, out_color, out_weight, radii, trans, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& opacities,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3Ds_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,

    const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_invdepth, 

	const torch::Tensor& out_trans,
	const torch::Tensor& non_trans,
	const torch::Tensor& dL_dout_trans,
	const torch::Tensor& dL_dnon_trans,

	const torch::Tensor& shs,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R, // num_rendered,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug,
	const float low_pass_filter_radius,
	const int image_height,
    const int image_width,

	// 阴影相关参数
	const bool ortho,
	const float offset) 
{
  const int P = means3D.size(0);
//   const int H = dL_dout_color.size(1);
//   const int W = dL_dout_color.size(2);
  const int H = image_height;
  const int W = image_width;
  
  int M = 0;
  if(shs.size(0) != 0)
  {	
	M = shs.size(1);
  }

// 初始化 返回的张量，也是我们需要的 梯度
  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacities = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3Ds = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dshs = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  torch::Tensor dL_dinvdepths = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dproj = torch::zeros({4, 4}, means3D.options());
  torch::Tensor dL_dview = torch::zeros({4, 4}, means3D.options());
  
  float* dL_dinvdepthsptr = nullptr;
  float* dL_dout_invdepthptr = nullptr;
  if(dL_dout_invdepth.size(0) != 0)
  {
	dL_dinvdepths = torch::zeros({P, 1}, means3D.options());
	dL_dinvdepths = dL_dinvdepths.contiguous();
	dL_dinvdepthsptr = dL_dinvdepths.data<float>();
	dL_dout_invdepthptr = dL_dout_invdepth.data<float>();
  }

  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data<float>(),
	  W, H, 
	  means3D.contiguous().data<float>(),
	  shs.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  opacities.contiguous().data<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3Ds_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data<int>(),
	  // 以 char* 类型 返回 缓冲区
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  
	  dL_dout_color.contiguous().data<float>(),
	  dL_dout_invdepthptr,

	  out_trans.contiguous().data<float>(),
	  non_trans.contiguous().data<float>(),
	  dL_dout_trans.contiguous().data<float>(),
	  dL_dnon_trans.contiguous().data<float>(),

	  
	  
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dconic.contiguous().data<float>(),  
	  dL_dopacities.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),

	  dL_dinvdepthsptr,
	  
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dcov3Ds.contiguous().data<float>(),
	  dL_dshs.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  dL_dproj.contiguous().data<float>(),
	  dL_dview.contiguous().data<float>(),

	  debug,
	  low_pass_filter_radius,

	  ortho,
	  offset
	  );
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacities, dL_dmeans3D, dL_dcov3Ds, dL_dshs, dL_dscales, dL_drotations, dL_dproj, dL_dview);
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