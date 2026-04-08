import math
import os
import sys

import torch

from utils.point_utils import depth_to_normal

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_TEX_RAST_ROOT = os.path.join(_ROOT, "submodules", "diff-surfel-rasterization-texture")
if os.path.isdir(_TEX_RAST_ROOT) and _TEX_RAST_ROOT not in sys.path:
    sys.path.insert(0, _TEX_RAST_ROOT)

from diff_surfel_rasterization_texture import (  # noqa: E402
    GaussianRasterizationSettings,
    GaussianRasterizer,
)


def render_textured(viewpoint_camera, pc, pipe, bg_color, scaling_modifier=1.0,
                    override_color=None, shadow_pkg=None, mbrdf=None, deferred=False):
    """Textured 2DGS render via diff-surfel-rasterization-texture.

    mbrdf dict keys: basecolor [N,3,R,R], shadow [N,R,R], other_effects [N,3]

    Direct  (deferred=False):
        final_tex = basecolor * shadow.unsqueeze(1) + other_effects[:,:,None,None]
        → single textured rasterizer pass → 3-ch image

    Deferred (deferred=True):
        TODO: 7-channel CUDA kernel: concat [basecolor(3)+shadow(1)+other_effects(3)]
        Python composition: out[:3] * out[3:4] + out[4:7]
    """
    from gaussian_renderer import _build_output_dict

    means2D = torch.zeros_like(pc.get_xyz, requires_grad=True, device="cuda") + 0
    try:
        means2D.retain_grad()
    except Exception:
        pass

    # ── Raster settings (uses textured module's own Settings class) ────────
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=getattr(pipe, "debug", False),
    )

    scales, rotations, cov3D_precomp = None, None, None
    if pipe.compute_cov3D_python:
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W - 1) / 2],
            [0, H / 2, 0, (H - 1) / 2],
            [0, 0, far - near, near],
            [0, 0, 0, 1],
        ]).float().cuda().T
        splat2world = pc.get_covariance(scaling_modifier)
        cov3D_precomp = (
            splat2world[:, [0, 1, 3]] @ (viewpoint_camera.full_proj_transform @ ndc2pix)[:, [0, 1, 3]]
        ).permute(0, 2, 1).reshape(-1, 9)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    def _splat(tex_color):
        img, radii, allmap = rasterizer(
            means3D=pc.get_xyz, means2D=means2D,
            shs=None, colors_precomp=None,
            opacities=pc.get_opacity,
            scales=scales, rotations=rotations, cov3D_precomp=cov3D_precomp,
            texture_color=tex_color, texture_alpha=pc.get_texture_alpha,
        )
        return img, radii, allmap

    # ── Route ──────────────────────────────────────────────────────────────
    if override_color is not None:
        # Debug override: broadcast [N,3] → [N,3,R,R]
        R = pc.get_texture_color.shape[-1]
        tex = override_color[:, :, None, None].expand(-1, -1, R, R).contiguous()
        rendered_image, radii, allmap = _splat(tex)

    elif deferred:
        # TODO: 7-channel CUDA kernel
        #   concat [basecolor(3), shadow(1), other_effects(3)] → rasterizer_7ch
        #   rendered_image = raw[:3] * raw[3:4] + raw[4:7]
        raise NotImplementedError("Deferred 7-channel kernel not yet implemented.")

    else:
        # Direct: final_tex = basecolor * shadow + other_effects  [N,3,R,R]
        if mbrdf is not None:
            basecolor = mbrdf["basecolor"]                         # [N, 3, R, R]
            shadow    = mbrdf["shadow"]                            # [N, R, R]
            oe        = mbrdf["other_effects"]                     # [N, 3] or None
            tex = basecolor * shadow.unsqueeze(1)
            if oe is not None:
                tex = tex + oe[:, :, None, None]
            tex = torch.clamp_min(tex, 0.0)
        else:
            tex = pc.get_texture_color                             # [N, 3, R, R]
        rendered_image, radii, allmap = _splat(tex)

    return _build_output_dict(means2D, radii, rendered_image, allmap,
                              viewpoint_camera, pc, pipe, shadow_pkg)
