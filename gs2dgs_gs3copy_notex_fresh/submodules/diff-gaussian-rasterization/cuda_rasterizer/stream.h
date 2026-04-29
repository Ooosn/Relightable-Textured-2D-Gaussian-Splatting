#pragma once
#include <c10/cuda/CUDAStream.h>

// 定义一个宏，每次使用 MY_STREAM 就会调用 c10::cuda::getCurrentCUDAStream().stream()
#define MY_STREAM c10::cuda::getCurrentCUDAStream().stream()
