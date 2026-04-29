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

#pragma once

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>
#include <cstdint>

namespace CudaRasterizer
{   
    /**
     * @brief  在考虑 对齐 的情况下，为传入的指针修改值（即修改指针指向的地址），并将 更新传入的地址
	 * 
     * @tparam T  指定的类型（如 int, float, double 等），同时要求 函数 需要在头文件中定义，但也有例外，比如在这个函数中，T 可以在调用中自己推导
     * @param chunk   原始内存块的起始地址（char*），会被更新 下一个内存块的起始地址
     * @param ptr     输出指针，指向对齐后的地址，即 当前内存块的起始地址
     * @param count   需要分配的 `T` 类型元素个数
     * @param alignment  需要对齐的字节数（通常为 2 的次幂，如 16、32）
	 * @typedef char* ———— 保证跨平台性，char* 是 1字节，可以保证 指针运算的正确性
	 * @typedef uintptr_t ———— 保证跨平台性，专为指针运算设计的无符号整数指针类型，保证指针运算安全
	 * @note static 在这里不是 静态成员函数，而是 限制函数 不出现在 全局符号表，不和其他同名函数冲突，为什么会有这个限制？
	 * 		- 主要是因为 模板函数的特殊性，模板函数 在 编译时 会生成 多个 实例，如果不加 static，则 会出现 多个 同名函数，导致 链接错误
	 * 		- 因此，static 使不同的 .cpp 都会得到自己独立的 static 版本，不会在全局符号表中共享从而避免冲突
	 *      - 但其实 现版本 不用 static 也可以，因为 模板函数 在 编译时 会生成 多个实例，不会在全局符号表中共享从而避免冲突
	 *      - 现版本多用 namespace{} 匿名函数 来 限制 函数 不出现在 全局符号表，不和 其他同名函数冲突
     */
    template <typename T>
    static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
    {   
        // 1. 计算对齐后的新地址：
        //    - 先将 `chunk` 转换为 `uintptr_t`
        //    - `+ alignment - 1` 确保即使 `chunk` 已对齐，也不会错过正确的对齐边界
        //    - `& ~(alignment - 1)` 清除低 `alignment` 位，使地址成为 `alignment` 的倍数
        std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);

        // 2. `ptr` 指向对齐后的新地址
        ptr = reinterpret_cast<T*>(offset);

        // 3. 更新 `chunk` 指向下一块可用内存
		//    - 指针加整数 会自动考虑 不同数据类型的 偏移量，ptr + i == (T*)((char*)ptr + i * sizeof(T))
        //    - `ptr + count` 计算出 `count` 个 `T` 类型对象之后的地址，
        //    - 重新转换为 `char*`，确保 `chunk` 始终指向字节级别的地址
        chunk = reinterpret_cast<char*>(ptr + count);
    }

	/** 
	 * @brief 建立一个公用几何状态，用于存储几何信息
	 * 
	 * @note 结构体对象和类对象不包括静态成员函数，静态函数属于 类型 GeometryState，而不是某个实例 
	 * @note 声明 静态成员函数，用于从 内存块中 获取几何状态 
	 * @else static ———— 这里是 静态成员函数，使其可以直接访问，直接通过 GeometryState::fromChunk() 调用
	*/
	struct GeometryState
	{
		size_t scan_size;
		float* depths;
		char* scanning_space;
		bool* clamped;
		int* internal_radii;
		float2* means2D;
		float* cov3D;
		float4* conic_opacity;	// conic.x, conic.y, conic.z 2D 协方差矩阵的逆，conic.w 高斯点的不透明度
		float* rgb;
		uint32_t* point_offsets;
		uint32_t* tiles_touched;
		
		// added
		float* radii_comp;

		// hgs 相关
		float3* normal;
		float* cov3D_smalls;
		float4* conic_opacity1;
		float4* conic_opacity2;
		uint4* conic_opacity3;
		uint4* conic_opacity4;
		float3* conic_opacity5;
		uint4* conic_opacity6;

		// 声明 一个返回 GeometryState 的静态函数 fromChunk
		static GeometryState fromChunk(char*& chunk, size_t P);
	};

	/** 
	 * @brief 建立一个公用图像状态，以像素为单位存储图像信息
	 * 
	 * @param ranges ———— 存储 每个 tile 的 开始和结束 index
	 * @param n_contrib ———— 存储 每个像素的 贡献数
	 * @param accum_alpha ———— 存储 每个像素的 最终累积透明度
	 * @note 声明 静态成员函数，用于从 内存块中 获取图像状态
	*/
	struct ImageState
	{
		uint2* ranges;
		uint32_t* n_contrib;
		float* accum_alpha;

		static ImageState fromChunk(char*& chunk, size_t N);
	};

	/** 
	 * @brief 建立一个公用排序状态，用于存储排序信息
	 * 
	 * @note 声明 静态成员函数，用于从 内存块中 获取排序状态
	*/
	struct BinningState
	{
		size_t sorting_size;
		uint64_t* point_list_keys_unsorted;
		uint64_t* point_list_keys;
		uint32_t* point_list_unsorted;
		uint32_t* point_list;
		char* list_sorting_space;

		static BinningState fromChunk(char*& chunk, size_t P);
	};

	/** 
	 * @brief 计算 几何状态、图像状态、排序状态 所需的内存大小
	 * @note T::fromChunk ———— 调用 几何状态、图像状态、排序状态 的 fromChunk 函数
	 * 			- 引用时 使用 required<T>(P)
	 * @details 设定 size = nullptr，即 从 0 开始分配内存，当 fromChunk 函数 返回时，size 指向 分配的内存块的 起始地址，即所需要的 内存大小
	 * @tparam T ———— 指定类型，可以是 GeometryState、ImageState、BinningState 等
	 * @param P ———— 指定大小，用于计算内存大小
	 * @return 返回 所需内存大小
	*/
	template<typename T> 
	size_t required(size_t P)
	{
		char* size = nullptr; // 即地址为 0
		T::fromChunk(size, P);
		return ((size_t)size) + 128;
	}
};