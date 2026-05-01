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

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS 7 // base(3) + decay(1) + other(3); screen compose: rgb = base*decay + other
#define TEXTURE_CHANNELS 4 // sampled texture carries only base RGB + decay/shadow
#define BLOCK_X 16
#define BLOCK_Y 16

#endif
