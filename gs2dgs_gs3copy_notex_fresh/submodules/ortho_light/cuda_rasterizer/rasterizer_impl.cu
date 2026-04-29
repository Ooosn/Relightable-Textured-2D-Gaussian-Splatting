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


#include "rasterizer_impl.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>


#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

namespace cg = cooperative_groups;



/**
 * @brief markVisible()：标记可见的高斯点
 * 		  forward()：前向渲染流程
 * 		  backward()：反向渲染流程
 * 		  GeometryState / ImageState / BinningState：存储渲染所需的几何、图像和排序状态
 *  	  cub::DeviceRadixSort：使用 CUB 库 进行 GPU 并行排序
 */


// Helper function to find the next-highest bit of the MSB 计算一个数的最高位（按 1 计数）
// on the CPU.
// sizeof(n) * 4 ———— 计算 n 的 字节，因为 uint32_t 是 32 位，4字节，所以 sizeof(n) * 4 = 16，利用 2 分法 计算 最高位
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)	// 直到步长为 1 为止
	{
		step /= 2;	// 下一次步长
		if (n >> msb)	// n 右移 msb 位，如果不为 0 ，说明左半边存在 1，则继续朝左边找，否则 朝右边找   // >> ———— 右移运算符，高位补 0，低位丢弃
			msb += step;	// 增加 n 右移的位数
		else
			msb -= step;	// 减少 n 右移的位数
	}
	if (n >> msb)	// 最后一次单独的判断，即最终步长为 1 时的右移位数
		msb++;		// msb 最小为 1，最大为 31，因此 如果 n 是 0，依然返回 1，其他的按最高位返回（按 1 计数）
	return msb;
}


// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
/**
 * @brief 判断高斯点是否在视图范围内
 * 
 * @note __global__ 声明一个 GPU 核函数
 *                  - 这意味着 checkFrustum() 不是在 CPU 上顺序执行的，而是在 GPU 上 "一次性启动成千上万个线程，并行计算"。
 * 		            - 因此同时启动了可能成千上万个线程，只保留 P 个线程，其他线程返回
 * @note __global__ 函数不支持 & 传递引用，因此必须用指针
 * @note 使用时，通过 核函数名<<<（线程数量+线程块大小-1）/线程块大小, 线程块大小 >>> 向上取整创建线程块，启动核函数
 */
__global__ void checkFrustum(int P, // 高斯点数量
	const float* orig_points, // 高斯点坐标
	const float* viewmatrix, // 视图矩阵
	const float* projmatrix, // 投影矩阵
	bool* present) // 是否可见 // 通过指针直接修改数据
{											  
	// idx = 网格（Grid/）blockIdx * 线程块（Block）blockDim + 线程（Thread）threadIdx
	// 网格代表不同kernel（任务），blockDim 是生产线，idx 是流水线上加工的零件
	// 因此当运行该核函数时，就相当于启动了一个 kernel，kernel 中有 P 个任务（逻辑线程），每个任务都有一个 idx，它是线程的唯一编号
	//     他们在不同的生产线（blockDim）上并行运行，所以他们会进入队列，等待 GPU 核心（工人）来处理
	// 因此所有任务都能满足 从 0 到 P-1 的索引，并且不同任务的 idx 是唯一的，不会冲突
	// 最终通过 不同的 idx 计算需要访问的内存地址，进行并行计算
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P) 			// 如果当前线程的索引大于等于高斯点数量，则返回，即每个 cuda 线程处理一个高斯点
		return;
	
	// 每个线程 初始化 自己的 p_view 变量作为局部变量，函数要求嘛，虽然后续不使用，但是不能不传嘛
	float3 p_view;	
	//  in_frustum （auxiliary 函数中） ———— 判断高斯点是否在视图范围内
	// 最终 得到 present[idx] ———— 一个 bool 值，表示当前高斯点是否在视图范围内
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}


// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
/**
 * @brief 针对每个“高斯”（Gaussian），根据其在图像或场景中的位置、半径以及深度，并行计算出它们在 网格（tiles/grid）中所覆盖的区域
 * 		  对于高斯覆盖的每个 tile，生成一对 key/value 对。
 * 		  - Key：由 tile 的 ID（来自 tile 坐标）和该高斯的深度信息组成
 * 		  - Value：该高斯的索引（ID）
 * 
 * @note 后续可以对这些 key/value 对进行排序，可以让同一个 tile 内的高斯按照深度顺序排列，方便后续处理（比如渲染或遮挡计算）
 * 
 * @note 高斯椭球指泼溅前，高斯椭圆指当前视角泼溅后
 * @param offsets 高斯椭圆写入结束位置
 * @param gaussian_keys_unsorted 高斯椭圆 key 未排序列表
 * @param gaussian_values_unsorted 高斯椭圆 value 未排序列表
 * @param grid 网格大小.tile 布局，grid.x 是 tile 的列数，grid.y 是 tile 的行数
 */
__global__ void duplicateWithKeys(
	int P, // 高斯椭球数量
	const float2* points_xy, // 高斯椭圆中心坐标
	const float* depths, // 高斯椭圆深度
	const uint32_t* offsets, // 高斯椭圆写入结束位置
	uint64_t* gaussian_keys_unsorted, // 高斯椭圆 key 未排序
	uint32_t* gaussian_values_unsorted, // 高斯椭圆 value 未排序
	uint4* conic_opacity3,
	uint4* conic_opacity4,
	uint4* conic_opacity6,
	int* radii, // 高斯椭圆半径
	dim3 grid,
	bool conic_
	) // 网格大小.tile 布局，grid.x 是 tile 的列数，grid.y 是 tile 的行数
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	// 对每一个视角，只对可见的高斯点进行处理
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		// 找到写入的开始位置，offsets[idx] 是 结束位置
		// 这里用 三元运算符，如果 idx 为 0，则 off 为 0，否则为 offsets[idx - 1]
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;
		// 计算高斯椭圆在图像中的 tile 范围，rect_min 和 rect_max 是 tile 的左下角和右上角坐标
		if (conic_)
		{
			rect_min.x = conic_opacity3[idx].x;
			rect_min.y = conic_opacity3[idx].y;
			rect_max.x = conic_opacity3[idx].z;
			rect_max.y = conic_opacity3[idx].w;
		}
		else{
			getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);
		}

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		// 遍历覆盖的所有 tile，计算每个tile的key，idx 是高斯椭圆的索引
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				// 计算 tile 的 key
				uint64_t key = y * grid.x + x;
				// 高32位 存储 tile 的 ID，低32位 存储 高斯椭圆的深度
				// 因此 他们排序时，可以先按 tile 排序，tile 内再按深度排序
				key <<= 32;
				// 先将 float* 强制转换为 uint32_t*，再 * 取值，这样取得的是 32 位无符号整数，而不是转译后的浮点数
				// 由于都是正数，不用担心最高位 01 问题
				key |= *((uint32_t*)&depths[idx]); 
				// 存储 key 和 value
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}

}


// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
/**
 * @brief 得到每个 tile 对应的数据范围（即某个 tile 所有的 Gaussians 在列表中的起始和结束 index）
 * 
 * @note 因此，ranges 的每个元素，表示一个 tile 的开始和结束 index
 * 
 * @param L 排序后的 key 列表长度（也就是 total duplicated Gaussians 的数量），即 idx 最大值 -1 
 * @param point_list_keys 排序后的 key 列表，即 gaussian_keys_unsorted 排序后
 * @param ranges 目标地址，写入每个 tile 的开始和结束 index
 * @note 这里的 线程 不是高斯椭圆，而是所有 tile 中键值对（duplicateWithKeys 生成的）的数量
 * @note 结束idx 一般是在列表中指的是下一个元素，而开始idx 就是列表中当前元素
 */
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	// 读取当前高斯椭圆的 tile ID
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	//如果当前为第一个，则一定是 第一个 tile 深度最小的
	if (idx == 0)
		ranges[currtile].x = 0;
	// 对于非第一个元素：
	// 看当前 tile 和前一个是否是同一个 tile；
	/* 如果 tile 发生变化（说明前一个 tile 到此为止），就：
	** - 把前一个 tile 的结束位置设为当前 idx；
	** - 把当前 tile 的开始位置设为当前 idx。	*/
	else
	{	
		// 读取前一个元素的 tile ID
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		// 如果当前 tile 和前一个 tile 不是同一个 tile，说明到新的 tile 了
		if (currtile != prevtile)
		{
			// 把前一个 tile 的结束位置设为当前 idx
			ranges[prevtile].y = idx;
			// 把当前 tile 的开始位置设为当前 idx
			ranges[currtile].x = idx;
		}
	}
	// 如果当前是最后一个元素，则把当前 tile 的结束位置设为 L
	if (idx == L - 1)
		ranges[currtile].y = L;
}


// Mark Gaussians as visible/invisible, based on view frustum testing
/**
 * @brief 标记高斯点是否可见
 * 
 * @param P 高斯点数量
 * @param means3D 高斯点坐标
 * @param viewmatrix 视图矩阵
 * @param projmatrix 投影矩阵
 * @param present 是否可见
 */
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{	
	// 启动 checkFrustum 核函数 ———— 检查每个高斯点是否在视图范围内
	// checkFrustum 会为每个高斯点调用 in_frustum 函数
	checkFrustum << <(P + 255) / 256, 256, 0, MY_STREAM>> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}


// obtain(a, b, c, d) 函数 ———— 在考虑 对齐 d 的情况下，根据 c = 需要分配的元素个数，得到 b = 当前内存块新的起始地址 和 a = 下一内存块的起始地址
// cub ———— NVIDIA 官方提供的一个 高性能 GPU 并行算法库。

// 对于大小在运行时决定的内存需求（如输入数量不定的运算），编译器无法在编译期分配临时空间。
// 因此必须先在 Host 侧计算所需的 buffer 大小，使用 cudaMalloc 分配临时空间，然后通过 指针 传入函数中，用于计算。
// 因为在函数的编译过程总中，一般（或者即便可以也一般不使用）只能编译 静态中间变量，不能编译动态变量。


// 定义 GeometryState fromChunk 函数
// cub::DeviceScan::InclusiveSum() 前缀和并行运算函数
CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{	
	// 按顺序 初始化或者更新 各自属性的 指针
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128); // 每个高斯点覆盖 多少个 tile 

	// added
	obtain(chunk, geom.radii_comp, P, 128);

	// hgs 相关
	obtain(chunk, geom.normal, P, 128);
	obtain(chunk, geom.cov3D_smalls, P * 6, 128);
	obtain(chunk, geom.conic_opacity1, P, 128);
	obtain(chunk, geom.conic_opacity2, P, 128);
	obtain(chunk, geom.conic_opacity3, P, 128);
	obtain(chunk, geom.conic_opacity4, P, 128);
	obtain(chunk, geom.conic_opacity5, P, 128);
	obtain(chunk, geom.conic_opacity6, P, 128);

	/** 
	* @brief 分配临时buffer 给 前缀和运算 库函数
	* @details cub::DeviceScan::InclusiveSum(
					temp_storage = nullptr,
					temp_storage_bytes =  geom.scan_size,
					input = geom.tiles_touched,
					output = geom.tiles_touched,
					num_items =  P);
	*
	* @note 传入 nullptr ———— cub 不对 tiles_touched 做前缀和运算，而是只计算需要的临时空间，把结果写进 geom.scan_size
	* @note 当传入指针不为 nullptr 时，cub 会对 tiles_touched 做前缀和运算，并把结果写进 tiles_touched
	* @note 前缀和运算 ———— 给定一个数组，输出一个新数组，其中每个元素是前面所有元素的“累加和”，这正是之前参数 offsets 的实现
	*/
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P, MY_STREAM	);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}


// 定义 ImageState fromChunk 函数
// N ———— 像素数量
CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}


// 定义 BinningState fromChunk 函数
// cub::DeviceRadixSort::SortPairs() 对一个key-value 对键值对进行排序，按照 key 排序，value 跟着 key 同步移动
CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	/** 
	* @brief 分配临时buffer 给 键值对排序 库函数
	* @details cub::DeviceRadixSort::SortPairs(
					temp_storage = nullptr,
					temp_storage_bytes =  binning.sorting_size,
					input1 = binning.point_list_keys_unsorted,
					output1 = binning.point_list_keys,
					input2 = binning.point_list_unsorted,
					output2 = binning.point_list,
					num_items =  P);
	*
	* @brief 传入 nullptr ———— 同理，只计算需要的临时空间，把结果写进 binning.sorting_size
	* @note 当传入指针不为 nullptr 时，cub 会对 键值对 进行排序，按照 key 排序，value 跟着 key 同步移动
	*/
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P, int(0), int(64), MY_STREAM);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}



// ------------------------------------------------------------------------------------------------


// ！！！ 前向过程 ！！！
// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(

	// 缓冲区调整函数 ———— 接受一个 size_t 类型的参数 N，返回一个 char* 类型的指针，调整缓冲区的大小
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,

	// 原输入参数
	const int P, int D, int M, // 高斯点数量、sh 阶数、sh 系数数量
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	bool debug,
	float low_pass_filter_radius,

	// 输出参数
	float* out_color,
	float* out_weight,
	int* radii,
	float* out_trans,

	// 添加参数
	float* non_trans,
	const float offset,
	const float thres,
    const bool is_train,
	
	// hgs 相关
	const bool hgs,
	const float* hgs_normals,
	const float* hgs_opacities,
	float* hgs_opacities_shadow,
	float* hgs_opacities_light
	)
{

	// 计算焦距
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);


	// 计算几何状态所需的内存大小
	size_t chunk_size = required<GeometryState>(P);
	// 分配/调整 geometryBuffer 内存
	char* chunkptr = geometryBuffer(chunk_size);
	// 根据缓冲区内存位置 初始化几何状态
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);


	// 如果 radii 为 nullptr，则使用几何状态中的 internal_radii，其实也就是初始化为 0
	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}


	// dim3 是 CUDA 中的一个结构体，用于定义 3D 网格和块的维度。
	// 默认 BLOCK_X = 16, BLOCK_Y = 16，即 每个tile 的像素为 16x16
	// 构造两个 dim3 对象，分别表示 tile 维度 和 block 维度。
	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1); // 向上取整获得 tile 的个数
	// 输入每个 block 的 维度 ———— 像素长宽
	dim3 block(BLOCK_X, BLOCK_Y, 1);


	// Dynamically resize image-based auxiliary buffers during training
	// 计算图片状态所需的内存大小，width * height ———— 像素数量
	size_t img_chunk_size = required<ImageState>(width * height);
	// 分配/调整 imageBuffer 内存
	char* img_chunkptr = imageBuffer(img_chunk_size);
	// 根据缓冲区内存位置 初始化图片状态
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);


	// 如果 不是 RGB 图像，并且 colors_precomp 为 nullptr，则抛出异常
	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}


	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	// 预处理 每个高斯点（变换、边界、将 SHs 转换为 RGB），转移数据到 buffer/GPU 上
	bool conic_ = false;
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered,
		low_pass_filter_radius,

		// added
		geomState.radii_comp,

		// hgs 相关
		hgs,
		conic_,
		hgs_normals,
		hgs_opacities,
		geomState.normal,
		geomState.cov3D_smalls,
		geomState.conic_opacity1,
		geomState.conic_opacity2,
		geomState.conic_opacity3,
		geomState.conic_opacity4,
		geomState.conic_opacity5,
		geomState.conic_opacity6
	), debug)
	

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	// 之前分配了 geomState.scanning_space 内存，现在使用 cub::DeviceScan::InclusiveSum 函数 真正的 计算前缀和，并存入 geomState.point_offsets
	// geomState.point_offsets[p-1] ———— 所有 [ tile | depth ] - idx 键值对 的数量，也就是所有需要计算/渲染的高斯点数量
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P, MY_STREAM), debug)


	// 获得需要计算/渲染的所有高斯点实例，每个高斯对象可能覆盖多个 tile，因此会重复计算，即创建多个实例
	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	/**
	 * @brief 因为缓冲区在 GPU 上，而该函数是 host 函数，不能从 GPU 中直接获取 geomState.point_offsets[p-1]，因此要先把这个值拷贝到 host 上
	 * 
	 * @param geomState.point_offsets + P - 1 ———— 指向最后一个元素的指针，也是源地址
	 * @param sizeof(int) ———— 取出一个 int 的大小
	 * @param cudaMemcpyDeviceToHost ———— 从 GPU 拷贝到 host
	 * @param &num_rendered ———— 目标地址
	 * @note num_rendered = geomState.point_offsets[P-1] ———— 所有需要计算/渲染的高斯点数量
	 * 
	 * @attention 下面有 host 函数，因此需要同步
	 */
	int num_rendered;
	CHECK_CUDA(cudaMemcpyAsync(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost, MY_STREAM), debug);
	// CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);
	cudaStreamSynchronize(MY_STREAM);	// 下面有 host 函数，因此需要同步

	/**
	 * @brief binningstate ———— 用于给每个高斯点分到所属的 tile 中
	 * 
	 * @note num_rendered = geomState.point_offsets[P-1]，有了这个值后，才可以计算所需内存，分配所需内存，并初始化 binningState
	 */
	// 有了 num_rendered 后，分配 binningBuffer 内存，因为在排序以及建立 tile分组 时，需要考虑所有需要计算的高斯
	// 初始化 binningState，用于存放 所有的 [ tile | depth ] key 和 对应的 重复高斯点索引
	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);


	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	/**
	 * @brief 生成 每个高斯点 的 [ tile | depth ] key 和 对应的 重复高斯点索引，
	 *		 - 并存放在 binningState.point_list_keys_unsorted 和 binningState.point_list_unsorted 中
	 */
	
	duplicateWithKeys << <(P + 255) / 256, 256, 0, MY_STREAM>> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		geomState.conic_opacity3,
		geomState.conic_opacity4,
		geomState.conic_opacity6,
		radii,
		tile_grid,
		conic_
		)
	CHECK_CUDA(, debug)


	// 获取 tile_grid 的 最高有效位，按1计数，减少后续计算量
	int bit = getHigherMsb(tile_grid.x * tile_grid.y);
	// Sort complete list of (duplicated) Gaussian indices by keys
	/** 
	 * @brief 同理，初始化时分配了计算临时内存，这里真正的键值对排序函数
	 * 
	 * @param 0 ———— begin_bit = 0 ———— 开始的位数，按0计数
	 * @param 32 + bit ———— end_bit = 32 + bit ———— 结束的位数，按0计数
	 * @param bit 是 tile_grid 的 最高有效位，按1计数
	 * @details 范围左闭右开，因此只用排序 32 + bit 位，即 [0, 32 + bit)，减少排序位数，减少排序时间
	 * @result 排序后的结果 写入 binningState.point_list_keys 和 binningState.point_list 中
	*/ 
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit, MY_STREAM), debug)


	/**
	 * @brief 初始化 imgState.ranges 为 0
	 * 
	 * @details cudaMemset(...) 在 GPU 设备内存上，把指定范围的内存设置为某个值
	 * @param imgState.ranges ———— 目标地址
	 * @param 0 ———— 设置为 0
	 * @param tile_grid.x * tile_grid.y * sizeof(uint2) ———— 设置的内存大小，即这个是要清零的总字节数
	 * @note uint2 是 2 个 uint 类型，即 2 个 32 位无符号整数，共 64 位，即 8 个字节，需要用 .x 和 .y 访问
	 * @note imgState.ranges 是用像素数量初始化的，相比于 tile 数量，过大
	 * 
	 * @attention 下面是 核函数，因此需要不需要同步
	 */
	CHECK_CUDA(cudaMemsetAsync(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2), MY_STREAM), debug);
	// CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);


	/* identifyTileRanges 确定每个 tile 对应的数据范围：
	 * 在当前使用中，即计算其在 point_list 列表中的起始与结束索引。这些索引标识了属于该 tile 的点数据子集
	 *
	 * 回忆，binningState.point_list 是按照 [ tile | depth ] 键值对 排序后的 高斯点 索引列表
	 * 因此，imgState.ranges 中每个元素表示一个 tile 的按 深度排序后的 高斯点 索引范围
	 */ 
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256, 0, MY_STREAM>> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)


	// 如果使用了 precomputed colors，则使用 colors_precomp，否则使用 geomState.rgb
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
// 随机生成100个在0~P以内的数，然后打印它们对应的depth
	if (false){
	float* h_depths = new float[P];
	float2* h_means2D = new float2[P];
	float3* h_means3D = new float3[P];
	cudaMemcpy(h_depths, geomState.depths, P * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_means2D, geomState.means2D, P * sizeof(float2), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_means3D, means3D, P * sizeof(float3), cudaMemcpyDeviceToHost);
	printf("--------------------------------\n");
	int rand_indices[100];
	float rand_depths[100];
	// 初始化随机种子
	unsigned int seed = (unsigned int)time(NULL);
	for (int i = 0; i < 100; ++i) {
		rand_indices[i] = rand() % P;
		printf("--------------------------------\n");
		rand_depths[i] = h_depths[rand_indices[i]];
	}
	printf("随机采样的100个depth值如下：\n");
	for (int i = 0; i < 100; ++i) {
		printf("depth: %d: %f\n", rand_indices[i], rand_depths[i]);
		printf("2d positon: %f, %f\n", h_means2D[rand_indices[i]].x, h_means2D[rand_indices[i]].y);
		printf("3d positon: %f, %f, %f\n", h_means3D[rand_indices[i]].x, h_means3D[rand_indices[i]].y, h_means3D[rand_indices[i]].z);
	}
	}
	// ！！！ 这里是 前向渲染 ！！！
	// Let each tile blend its range of Gaussians independently in parallel
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,	// tile_grid： tile 的数量，block： tile 的像素大小，用于启动核函数
		P,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		geomState.depths,
		feature_ptr,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		// 输出
		out_color,
		out_weight,
		out_trans,
		non_trans,
		// 其他参数
		offset,
		thres,
	    is_train,
		
		// added
		geomState.radii_comp,
		
		// hgs 相关
		hgs,
		hgs_opacities,
		hgs_opacities_shadow,
		hgs_opacities_light,
		geomState.normal,
		geomState.conic_opacity1
		), debug)


	// 回归 host，同步流，其实可忽略，因为 return 会触发隐式同步
	cudaStreamSynchronize(MY_STREAM);

	return num_rendered;
}




// ------------------------------------------------------------------------------------------------


// ！！！ 反向过程 ！！！
// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,

    // const float* opacities, 为什么这个没了
	
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,

	const float* dL_dtrans,	// add
	const float* trans,		// add

	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,

	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,

	bool debug,
	const float* non_trans	/*add*/)
{
	// 初始化缓冲区
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);


	// 如果 radii 为 nullptr，则使用几何状态中的 internal_radii，其实也就是初始化为 0
	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}


	// 计算焦距
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);


	// 计算 tile 网格和 block 的维度
	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);


	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	// 如果使用了 precomputed colors，则使用 colors_precomp，否则使用 geomState.rgb
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;


	//！！！ 这里是 反向渲染 ！！！
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
								// 未添加 geomState.depths,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
								// 未添加 dL_invdepths,
		dL_dtrans,				//add
		trans, 					// add
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor,
								// 未添加 dL_dinvdepth
		non_trans), debug)		// add

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	// 如果使用了 precomputed covariance，则使用 cov3D_precomp，否则使用 geomState.cov3D
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;


	// 反向预处理
	CHECK_CUDA(BACKWARD::preprocess(
		P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
								// opacities 为什么没了
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
								// dL_dinvdepth,
								// dL_dopacity,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot
								/*antialiasing*/
		), debug)
}