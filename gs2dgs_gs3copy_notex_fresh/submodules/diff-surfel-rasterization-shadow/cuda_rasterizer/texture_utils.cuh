#pragma once

#include <cuda_runtime.h>
#include "config.h"

__device__ inline float texture_clamp(float v, float lo, float hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ inline int texture_color_index(int gid, int ch, int y, int x, int tex_res)
{
    return (((gid * NUM_CHANNELS + ch) * tex_res + y) * tex_res + x);
}

__device__ inline int texture_alpha_index(int gid, int y, int x, int tex_res)
{
    return (((gid * 1 + 0) * tex_res + y) * tex_res + x);
}

__device__ inline bool texture_layout(
    const int gid,
    const int tex_res,
    const int* texture_dims,
    int& tex_h,
    int& tex_w,
    int& tex_offset)
{
    if (texture_dims != nullptr)
    {
        tex_h = texture_dims[gid * 3 + 0];
        tex_w = texture_dims[gid * 3 + 1];
        tex_offset = texture_dims[gid * 3 + 2];
        return tex_h > 0 && tex_w > 0;
    }

    tex_h = tex_res;
    tex_w = tex_res;
    tex_offset = gid * tex_res * tex_res;
    return tex_res > 0;
}

__device__ inline int texture_color_index(int gid, int ch, int y, int x, int tex_res, const int* texture_dims)
{
    if (texture_dims != nullptr)
    {
        const int tex_w = texture_dims[gid * 3 + 1];
        const int tex_offset = texture_dims[gid * 3 + 2];
        return ((tex_offset + y * tex_w + x) * NUM_CHANNELS + ch);
    }
    return texture_color_index(gid, ch, y, x, tex_res);
}

__device__ inline int texture_alpha_index(int gid, int y, int x, int tex_res, const int* texture_dims)
{
    if (texture_dims != nullptr)
    {
        const int tex_w = texture_dims[gid * 3 + 1];
        const int tex_offset = texture_dims[gid * 3 + 2];
        return tex_offset + y * tex_w + x;
    }
    return texture_alpha_index(gid, y, x, tex_res);
}

__device__ inline void compute_texture_uv(
    const float2& s,
    const float sigma_factor,
    float& u,
    float& v,
    float& du_dsx,
    float& dv_dsy)
{
    const float inv_extent = 0.5f / max(sigma_factor, 1e-6f);
    u = s.x * inv_extent + 0.5f;
    v = s.y * inv_extent + 0.5f;

    const bool u_clamped = (u <= 0.0f || u >= 1.0f);
    const bool v_clamped = (v <= 0.0f || v >= 1.0f);
    u = texture_clamp(u, 0.0f, 1.0f);
    v = texture_clamp(v, 0.0f, 1.0f);

    du_dsx = u_clamped ? 0.0f : inv_extent;
    dv_dsy = v_clamped ? 0.0f : inv_extent;
}

__device__ inline void sample_texture_bilinear(
    const float* texture_color,
    const float* texture_alpha,
    const int gid,
    const int tex_res,
    const int* texture_dims,
    const float u,
    const float v,
    float out_rgb[NUM_CHANNELS],
    float& out_alpha,
    float d_rgb_du[NUM_CHANNELS],
    float d_rgb_dv[NUM_CHANNELS],
    float& d_alpha_du,
    float& d_alpha_dv)
{
    for (int ch = 0; ch < NUM_CHANNELS; ++ch)
    {
        out_rgb[ch] = 0.0f;
        d_rgb_du[ch] = 0.0f;
        d_rgb_dv[ch] = 0.0f;
    }
    out_alpha = 1.0f;
    d_alpha_du = 0.0f;
    d_alpha_dv = 0.0f;

    int tex_h = 0;
    int tex_w = 0;
    int tex_offset = 0;
    if (!texture_layout(gid, tex_res, texture_dims, tex_h, tex_w, tex_offset))
    {
        return;
    }

    if (tex_h == 1 && tex_w == 1)
    {
        if (texture_color != nullptr)
        {
            for (int ch = 0; ch < NUM_CHANNELS; ++ch)
            {
                out_rgb[ch] = texture_color[texture_color_index(gid, ch, 0, 0, tex_res, texture_dims)];
            }
        }
        if (texture_alpha != nullptr)
        {
            out_alpha = texture_alpha[texture_alpha_index(gid, 0, 0, tex_res, texture_dims)];
        }
        return;
    }

    const float x = tex_w > 1 ? u * (tex_w - 1) : 0.0f;
    const float y = tex_h > 1 ? v * (tex_h - 1) : 0.0f;
    const int x0 = max(0, min(tex_w - 1, static_cast<int>(floorf(x))));
    const int y0 = max(0, min(tex_h - 1, static_cast<int>(floorf(y))));
    const int x1 = min(x0 + 1, tex_w - 1);
    const int y1 = min(y0 + 1, tex_h - 1);
    const float tx = x - x0;
    const float ty = y - y0;

    const float w00 = (1.0f - tx) * (1.0f - ty);
    const float w10 = tx * (1.0f - ty);
    const float w01 = (1.0f - tx) * ty;
    const float w11 = tx * ty;
    const float scale_x = tex_w > 1 ? static_cast<float>(tex_w - 1) : 0.0f;
    const float scale_y = tex_h > 1 ? static_cast<float>(tex_h - 1) : 0.0f;

    if (texture_color != nullptr)
    {
        for (int ch = 0; ch < NUM_CHANNELS; ++ch)
        {
            const float c00 = texture_color[texture_color_index(gid, ch, y0, x0, tex_res, texture_dims)];
            const float c10 = texture_color[texture_color_index(gid, ch, y0, x1, tex_res, texture_dims)];
            const float c01 = texture_color[texture_color_index(gid, ch, y1, x0, tex_res, texture_dims)];
            const float c11 = texture_color[texture_color_index(gid, ch, y1, x1, tex_res, texture_dims)];

            out_rgb[ch] = w00 * c00 + w10 * c10 + w01 * c01 + w11 * c11;
            d_rgb_du[ch] = scale_x * ((1.0f - ty) * (c10 - c00) + ty * (c11 - c01));
            d_rgb_dv[ch] = scale_y * ((1.0f - tx) * (c01 - c00) + tx * (c11 - c10));
        }
    }

    if (texture_alpha == nullptr)
    {
        return;
    }

    const float a00 = texture_alpha[texture_alpha_index(gid, y0, x0, tex_res, texture_dims)];
    const float a10 = texture_alpha[texture_alpha_index(gid, y0, x1, tex_res, texture_dims)];
    const float a01 = texture_alpha[texture_alpha_index(gid, y1, x0, tex_res, texture_dims)];
    const float a11 = texture_alpha[texture_alpha_index(gid, y1, x1, tex_res, texture_dims)];

    out_alpha = w00 * a00 + w10 * a10 + w01 * a01 + w11 * a11;
    d_alpha_du = scale_x * ((1.0f - ty) * (a10 - a00) + ty * (a11 - a01));
    d_alpha_dv = scale_y * ((1.0f - tx) * (a01 - a00) + tx * (a11 - a10));
}

__device__ inline float sample_texture_alpha_nearest(
    const float* texture_alpha,
    const int gid,
    const int tex_res,
    const int* texture_dims,
    const float u,
    const float v)
{
    if (texture_alpha == nullptr)
    {
        return 1.0f;
    }
    int tex_h = 0;
    int tex_w = 0;
    int tex_offset = 0;
    if (!texture_layout(gid, tex_res, texture_dims, tex_h, tex_w, tex_offset))
    {
        return 1.0f;
    }
    const int x = max(0, min(tex_w - 1, static_cast<int>(floorf(u * tex_w))));
    const int y = max(0, min(tex_h - 1, static_cast<int>(floorf(v * tex_h))));
    return texture_alpha[texture_alpha_index(gid, y, x, tex_res, texture_dims)];
}

__device__ inline void sample_texture_bilinear(
    const float* texture_color,
    const float* texture_alpha,
    const int gid,
    const int tex_res,
    const float u,
    const float v,
    float out_rgb[NUM_CHANNELS],
    float& out_alpha,
    float d_rgb_du[NUM_CHANNELS],
    float d_rgb_dv[NUM_CHANNELS],
    float& d_alpha_du,
    float& d_alpha_dv)
{
    sample_texture_bilinear(
        texture_color, texture_alpha, gid, tex_res, nullptr, u, v,
        out_rgb, out_alpha, d_rgb_du, d_rgb_dv, d_alpha_du, d_alpha_dv);
}

__device__ inline void accumulate_texture_bilinear_vjp(
    const int gid,
    const int tex_res,
    const int* texture_dims,
    const float u,
    const float v,
    const float dL_drgb[NUM_CHANNELS],
    const float dL_dalpha,
    float* dL_dtex_color,
    float* dL_dtex_alpha)
{
    int tex_h = 0;
    int tex_w = 0;
    int tex_offset = 0;
    if (!texture_layout(gid, tex_res, texture_dims, tex_h, tex_w, tex_offset))
    {
        return;
    }

    if (tex_h == 1 && tex_w == 1)
    {
        if (dL_dtex_color != nullptr)
        {
            for (int ch = 0; ch < NUM_CHANNELS; ++ch)
            {
                atomicAdd(&dL_dtex_color[texture_color_index(gid, ch, 0, 0, tex_res, texture_dims)], dL_drgb[ch]);
            }
        }
        if (dL_dtex_alpha != nullptr)
        {
            atomicAdd(&dL_dtex_alpha[texture_alpha_index(gid, 0, 0, tex_res, texture_dims)], dL_dalpha);
        }
        return;
    }

    const float x = tex_w > 1 ? u * (tex_w - 1) : 0.0f;
    const float y = tex_h > 1 ? v * (tex_h - 1) : 0.0f;
    const int x0 = max(0, min(tex_w - 1, static_cast<int>(floorf(x))));
    const int y0 = max(0, min(tex_h - 1, static_cast<int>(floorf(y))));
    const int x1 = min(x0 + 1, tex_w - 1);
    const int y1 = min(y0 + 1, tex_h - 1);
    const float tx = x - x0;
    const float ty = y - y0;

    const float w00 = (1.0f - tx) * (1.0f - ty);
    const float w10 = tx * (1.0f - ty);
    const float w01 = (1.0f - tx) * ty;
    const float w11 = tx * ty;

    if (dL_dtex_color != nullptr)
    {
        for (int ch = 0; ch < NUM_CHANNELS; ++ch)
        {
            const float grad = dL_drgb[ch];
            atomicAdd(&dL_dtex_color[texture_color_index(gid, ch, y0, x0, tex_res, texture_dims)], w00 * grad);
            atomicAdd(&dL_dtex_color[texture_color_index(gid, ch, y0, x1, tex_res, texture_dims)], w10 * grad);
            atomicAdd(&dL_dtex_color[texture_color_index(gid, ch, y1, x0, tex_res, texture_dims)], w01 * grad);
            atomicAdd(&dL_dtex_color[texture_color_index(gid, ch, y1, x1, tex_res, texture_dims)], w11 * grad);
        }
    }

    if (dL_dtex_alpha != nullptr)
    {
        atomicAdd(&dL_dtex_alpha[texture_alpha_index(gid, y0, x0, tex_res, texture_dims)], w00 * dL_dalpha);
        atomicAdd(&dL_dtex_alpha[texture_alpha_index(gid, y0, x1, tex_res, texture_dims)], w10 * dL_dalpha);
        atomicAdd(&dL_dtex_alpha[texture_alpha_index(gid, y1, x0, tex_res, texture_dims)], w01 * dL_dalpha);
        atomicAdd(&dL_dtex_alpha[texture_alpha_index(gid, y1, x1, tex_res, texture_dims)], w11 * dL_dalpha);
    }
}

__device__ inline void accumulate_texture_bilinear_vjp(
    const int gid,
    const int tex_res,
    const float u,
    const float v,
    const float dL_drgb[NUM_CHANNELS],
    const float dL_dalpha,
    float* dL_dtex_color,
    float* dL_dtex_alpha)
{
    accumulate_texture_bilinear_vjp(
        gid, tex_res, nullptr, u, v, dL_drgb, dL_dalpha,
        dL_dtex_color, dL_dtex_alpha);
}
