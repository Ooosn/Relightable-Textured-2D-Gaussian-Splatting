import torch
import torch.nn.functional as F
import numpy as np
import math

from gsplat import rasterization
from utils.graphics_utils import fov2focal
from diff_gaussian_rasterization import  GaussianRasterizationSettings as basic_settings
from diff_gaussian_rasterization import  GaussianRasterizer as basic_rasterizer
from diff_gaussian_rasterization_light import GaussianRasterizationSettings as light_settings
from diff_gaussian_rasterization_light import  GaussianRasterizer as light_rasterizer
from v_3dgs import GaussianRasterizationSettings as v_3dgs_settings
from v_3dgs import GaussianRasterizer as persp_rasterizer
from v_3dgs_ortho import GaussianRasterizer as ortho_rasterizer
from diff_gaussian_rasterization_hgs import  GaussianRasterizationSettings as hgs_settings
from diff_gaussian_rasterization_hgs import  GaussianRasterizer as hgs_rasterizer

from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.graphics_utils import getProjectionMatrixWithPrincipalPoint, look_at, getProjectionMatrix, getOrthographicMatrixFromBounds, computeCov2D_ortho_python
import time
from utils.general_utils import  get_expon_lr, GradientScaler



debug = False

def has_nan_or_inf(tensor, dim):
    return tensor.isnan().any(dim=dim) | tensor.isinf().any(dim=dim)




def shadow_render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, 
           separate_sh = False, override_color = None, use_trained_exp = False,
           updated_light_pos = None, offset = 0.15, 
           scene_offset_coefficient = 10,
           vehicle_offset_coefficient = 5,
           fix_pkg = None, move = None, pre_shadow = None, transparent_layer = False,
           scale_factor = 1.0, need_shadow = False,
           light_dir = None
           ):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means



    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)


    # light

    # calculate the fov and projmatrix of light
    fx_origin = viewpoint_camera.image_width / (2. * tanfovx)
    fy_origin = viewpoint_camera.image_height / (2. * tanfovy)

    # calculate the fov for shadow splatting:
    # 计算光源和相机的距离比 f_scale_ratio
    
    # 处理 nan 值
    # print("num of gaussian points: ", pc.get_xyz.shape[0])
    prune = (
        has_nan_or_inf(pc.get_xyz, dim=1) |
        has_nan_or_inf(pc.get_opacity, dim=1) |
        has_nan_or_inf(pc.get_scaling, dim=1) |
        has_nan_or_inf(pc.get_rotation, dim=1) |
        has_nan_or_inf(pc.get_features, dim=(1, 2))
    )
    
    # object_id = pc.object_id
    # mask = object_id[:, 0] == 1
    # prune = prune | mask
    
    # print("prune_sum: ", prune.sum())
    xyz = pc.get_xyz
    opacity = pc.get_opacity
    scaling = pc.get_scaling
    rotation = pc.get_rotation
    features = pc.get_features
    object_id = pc.object_id
    
    xyz = xyz[~prune]
    opacity = opacity[~prune]
    scaling = scaling[~prune]
    rotation = rotation[~prune]
    features = features[~prune]
    object_id = object_id[~prune]
    object_id = object_id.squeeze().unsqueeze(1)
    object_id = torch.cat((object_id, torch.tensor([[scene_offset_coefficient], [vehicle_offset_coefficient]], dtype=torch.float32, device="cuda")))
    # print("object_id: ", object_id)
    # print("object_id.shape: ", object_id.shape)
    # print("xyz.shape: ", xyz.shape)
    
    screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    ortho = True
    light_box = True
    shadow_map = False
    depth_image = None
    ratio = 1.0
    ortho_rendered_image = None
    
    xyz_scene = xyz[(object_id[:-2,]!=0).squeeze()]
    
    
    if fix_pkg is None:
        object_center = xyz_scene.mean(dim=0).detach().cpu().numpy()
    # 应对极端情况，光源跟着轨迹移动
    else:
        object_center = fix_pkg["object_center"]
        print("fix_pkg[object_center]:", fix_pkg["object_center"])
        if move is not None:
            if not ortho:
                move = move.detach().cpu().numpy() if isinstance(move, torch.Tensor) else np.asarray(move)
                print("move: ", move)
                object_center = object_center + move
                
    if updated_light_pos is None:
        light_position = object_center + (0, 0, -2333)
    else:
        light_position = updated_light_pos + object_center
        # print("light_position: ", light_position)
 
    
    
    world_view_transform_light=look_at(light_position,
                                       object_center,
                                       up_dir=np.array([0, 0, 1]),
                                       camera_direction=light_dir)
    
    world_view_transform_light=torch.tensor(world_view_transform_light,
                                            device=viewpoint_camera.world_view_transform.device,
                                            dtype=viewpoint_camera.world_view_transform.dtype) 
    
    camera_position = viewpoint_camera.camera_center.detach().cpu().numpy() * 10
    f_scale_ratio = np.sqrt(np.sum(light_position * light_position) / np.sum(camera_position * camera_position))
    f_scale_ratio *= 0.2
    # print("f_scale_ratio", f_scale_ratio)
    # 计算光源的焦距
    fx_far = fx_origin * f_scale_ratio
    fy_far = fy_origin * f_scale_ratio
    cx = viewpoint_camera.image_width / 2.0
    cy = viewpoint_camera.image_height / 2.0

    # 先计算 FoV 对应的正切值，然后通过 arctan 得出 FoV
    tanfovx_far = 0.5 * viewpoint_camera.image_width / fx_far
    tanfovy_far = 0.5 * viewpoint_camera.image_height / fy_far
    # 将焦距反推回视场角（FoV）
    fovx_far = 2 * math.atan(tanfovx_far)
    fovy_far = 2 * math.atan(tanfovy_far)


    
    mask_sky = object_id[:-2, ] == 0
    mask_object = object_id[:-2, ] == 1
    mask_scene = (object_id[:-2, ] == 1) | mask_sky
    mask_blanket = object_id[:-2, ] >= 100
    
    
    if ortho:
        xyz_4 = torch.cat([xyz_scene, torch.ones_like(xyz_scene[:, 0:1])], dim=1)
        xyz_ortho_scene = xyz_4 @ world_view_transform_light
        x_min, y_min, z_min = xyz_ortho_scene[:, 0].min(), xyz_ortho_scene[:, 1].min(), xyz_ortho_scene[:, 2].min()
        x_max, y_max, z_max = xyz_ortho_scene[:, 0].max(), xyz_ortho_scene[:, 1].max(), xyz_ortho_scene[:, 2].max()
        extent_scene = torch.stack([
            x_max - x_min,
            y_max - y_min,
            z_max - z_min
        ])
        # print("x_max, y_max, z_max: ", x_max, y_max, z_max)
        # print("x_min, y_min, z_min: ", x_min, y_min, z_min)
        
        if light_box:
            mask_sky_scene = mask_scene[~mask_sky]
            xyz_ortho_object = xyz_ortho_scene[~mask_sky_scene]
            try:
                x_min, y_min, z_min = xyz_ortho_object[:, 0].min(), xyz_ortho_object[:, 1].min(), xyz_ortho_object[:, 2].min()
                x_max, y_max, z_max = xyz_ortho_object[:, 0].max(), xyz_ortho_object[:, 1].max(), xyz_ortho_object[:, 2].max()
                extent_object = torch.stack([
                        x_max - x_min,
                        y_max - y_min,
                        z_max - z_min 
                        ])
                # ratio = torch.sqrt( (extent_object / extent_scene).abs().max())
                # print("ratio: ", ratio)
                # print("x_max, y_max, z_max: ", x_max, y_max, z_max)
                # print("x_min, y_min, z_min: ", x_min, y_min, z_min)
            except:
                pass

        else:
            ratio = 1.0

            
        # padding_factor = 0.1 # 10% 扩展
        # x_min -= extent_scene[0] * padding_factor
        # y_min -= extent_scene[1] * padding_factor
        # z_min -= extent_scene[2] * padding_factor
        # x_max += extent_scene[0] * padding_factor
        # y_max += extent_scene[1] * padding_factor
        # z_max += extent_scene[2] * padding_factor
        light_orthoproj_matrix = getOrthographicMatrixFromBounds(
            xmin=x_min,
            xmax=x_max,
            ymin=y_min,
            ymax=y_max,
            zmin=z_min,
            zmax=z_max
        ).transpose(0, 1).cuda()
    else:
        ratio = 1.0
        
    
    if not ortho:
        light_persp_proj_matrix = getProjectionMatrix(znear=viewpoint_camera.znear, zfar=viewpoint_camera.zfar, fovX=fovx_far, fovY=fovy_far).transpose(0,1).cuda()
        full_persp_proj_transform_light = (world_view_transform_light.unsqueeze(0).bmm(light_persp_proj_matrix.unsqueeze(0))).squeeze(0)
        full_ortho_proj_transform_light = None
    else:
        full_ortho_proj_transform_light = (world_view_transform_light.unsqueeze(0).bmm(light_orthoproj_matrix.unsqueeze(0))).squeeze(0)
        full_persp_proj_transform_light = None
    

    # 你猜这是什么？ 
    d3_ortho_cov = None
    if False:
        d3_ortho_cov = computeCov2D_ortho_python(xyz, viewpoint_camera.image_width, viewpoint_camera.image_height, \
            light_orthoproj_matrix, fx_far, fy_far, scaling, rotation, world_view_transform_light, scaling_modifier,\
            full_ortho_proj_transform_light)
    
    
    
    H_light = int(10 * ratio * viewpoint_camera.image_height)
    W_light = int(10 * ratio * viewpoint_camera.image_width)
    W = int(viewpoint_camera.image_width)
    H = int(viewpoint_camera.image_height)
    
    
    opacity_light = torch.zeros([xyz.shape[0],1], dtype=torch.float32, device='cuda')
    means3D = xyz
    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda")
    
    if transparent_layer:
        # 暂存
        original_opacity = opacity.clone()
        mask_blanket = object_id[:-2, 0] >= 100
        opacity[mask_blanket] = 1
        mask_scene = (object_id[:-2, 0] == 1) | (object_id[:-2, 0] == 0)
        opacity[mask_scene] = 0
    
    scales = None
    rotations = None
    cov3Ds_precomp = None
    if pipe.compute_cov3D_python:
        cov3Ds_precomp = pc.get_covariance(scaling_modifier)[~prune]
    else:
        scales = scaling
        rotations = rotation
            
    temp_opacity = opacity.clone()
    # opacity[object_id[:-2]==2] = 1          
    if shadow_map:
        raster_settings = v_3dgs_settings(
            image_height = H_light,
            image_width = W_light,
            tanfovx = tanfovx_far,
            tanfovy = tanfovy_far,
            bg = bg_color[:3],
            scale_modifier = scaling_modifier,
            viewmatrix = world_view_transform_light,
            projmatrix = full_ortho_proj_transform_light if ortho else full_persp_proj_transform_light,
            sh_degree = pc.active_sh_degree,
            campos = light_position,
            prefiltered = False,
            debug = pipe.debug,
            antialiasing = False
            )
        
        rasterizer_inputs = {
                # 高斯点相关
                "means3D": means3D,
                "means2D": screenspace_points,
                "shs": None,
                "colors_precomp": torch.zeros((2, 3), dtype=torch.float32, device='cuda'),
                "opacities": opacity,
                "scales": None if d3_ortho_cov is not None else scales,
                "rotations": None if d3_ortho_cov is not None else rotations,
                "cov3D_precomp": d3_ortho_cov if ortho else cov3Ds_precomp,
                "shadow": None,
            }
        
        if ortho:
            shadow_map_rasterizer = ortho_rasterizer(raster_settings=raster_settings )
        else:
            shadow_map_rasterizer = persp_rasterizer(raster_settings=raster_settings )
        _, __, depth_image, final_T = shadow_map_rasterizer(**rasterizer_inputs)
        shadow = None
    
    else:
        # 设置光源的高斯泼溅参数
        raster_settings_light = light_settings(
            image_height = H_light,
            image_width = W_light,
            tanfovx = tanfovx_far,
            tanfovy = tanfovy_far,
            bg = bg_color[:3],
            scale_modifier = scaling_modifier,
            viewmatrix = world_view_transform_light,
            projmatrix = full_ortho_proj_transform_light if ortho else full_persp_proj_transform_light, 
            sh_degree = pc.active_sh_degree,
            campos = torch.tensor(light_position, dtype=torch.float32, device='cuda'),
            prefiltered = False,
            debug = pipe.debug,
            low_pass_filter_radius = 0.3,
            ortho = ortho
        )

        rasterizer_light = light_rasterizer(raster_settings=raster_settings_light)
                

        
        
        light_object_ids = object_id[:-2].repeat(1, 3)
        light_inputs = {
                    # 高斯点相关
                    "means3D": means3D,
                    "means2D": screenspace_points,
                    "shs": None,
                    "colors_precomp": light_object_ids,
                    "opacities": opacity,
                    "scales": None if d3_ortho_cov is not None else scales,
                    "rotations": None if d3_ortho_cov is not None else rotations,
                    "cov3Ds_precomp": d3_ortho_cov if ortho else cov3Ds_precomp,

                    # 阴影相关
                    "non_trans": opacity_light,
                    "offset": offset,
                    "thres": -1,

                    # prune 相关
                    "is_train": False,
                    
                    # hgs 相关
                    "hgs": False,
                    "hgs_normals": None,
                    "hgs_opacities": None,
                    "hgs_opacities_shadow": None, 
                    "hgs_opacities_light": None, 

                    # 流
                    "streams": None # 暂时没用，（用于内部多个流）

                }
        if ortho:
            _, out_weight, _, shadow, opacity_light, _ = rasterizer_light(**light_inputs)
            
        opacity_light = torch.clamp_min(opacity_light, 1e-6)
        
        # print("num of nan: ", torch.isnan(shadow).sum())
        # print("num of inf: ", torch.isnan(opacity_light1).sum())
        shadow = shadow / opacity_light 
        print("shadow1: ", shadow.mean())
        # print("shadow: ", shadow)
        print((opacity_light<1e-4).sum())
        print(xyz.shape)
        shadow[opacity_light<1e-4] = 1
        print("shadow1: ", shadow.mean())
        # print("num of nan: ", torch.isnan(shadow).sum())
        # print("shadow: ", shadow)
        assert not torch.isnan(shadow).any()
        shadow = torch.clamp(shadow, 0, 1)
    
    
    
    
    # 不要地面阴影
    
    print("shadow2: ", shadow.mean())
    mask = (object_id[:-2, 0] == 1) | (object_id[:-2, 0] >= 100)
    shadow[mask] += 0.3
    shadow[~mask] += 0.577
    shadow = torch.clamp(shadow, 0, 1)
    print("shadow3: ", shadow.mean())
    # 测试 layer
    mask = object_id[:-2, 0] > 100
    # shadow[mask] = 0
    # shadow[~mask] = 1

    
    #------------------------------------------------------------------------
    # 视角方向渲染
    raster_settings = v_3dgs_settings(
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
        debug=pipe.debug,
        antialiasing=True,
        znear=viewpoint_camera.znear,
        zfar=viewpoint_camera.zfar,
    )

    raster_settings_dict = {
        "image_height": int(viewpoint_camera.image_height),
        "image_width": int(viewpoint_camera.image_width),
        "tanfovx": tanfovx,
        "tanfovy": tanfovy,
        "bg": bg_color,
        "scale_modifier": scaling_modifier,
        "viewmatrix": viewpoint_camera.world_view_transform,
        "projmatrix": viewpoint_camera.full_proj_transform,
        "sh_degree": pc.active_sh_degree,
        "campos": viewpoint_camera.camera_center
    }

    rasterizer = persp_rasterizer(raster_settings=raster_settings)

    means3D = xyz
    means2D = screenspace_points
    opacity = opacity
    cov3D_precomp = None

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)[~prune]
    else:
        scales = scaling
        rotations = rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (xyz - viewpoint_camera.camera_center.repeat(xyz.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = features
    else:
        colors_precomp = override_color
    
    if not need_shadow:
        print("no shadow")
        print("no shadow")
        print("no shadow")
        shadow = torch.ones_like(opacity)
        
    if shadow_map:
        focalx = fov2focal(viewpoint_camera.FoVx, viewpoint_camera.image_width)
        focaly = fov2focal(viewpoint_camera.FoVy, viewpoint_camera.image_height)
        K = torch.tensor([[focalx, 0, viewpoint_camera.cx], [0, focaly, viewpoint_camera.cy], [0., 0., 1.]], device="cuda")
        
        splat_inputs = {
            "means": means3D, 
            "quats": rotations, 
            "scales": scales, 
            "opacities": opacity.squeeze(-1), 
            "colors": shs, 
            "viewmats": viewpoint_camera.world_view_transform.transpose(0, 1)[None, ...], 
            "Ks": K[None, ...], 
            "width": int(viewpoint_camera.image_width),
            "height": int(viewpoint_camera.image_height),
            "near_plane": viewpoint_camera.znear,
            "far_plane": viewpoint_camera.zfar,
            "eps2d": 0.3,
            "sh_degree": pc.active_sh_degree,
            "packed": False,
            "backgrounds": bg_color[None, ...]   
            } 
        
        splat_inputs["render_mode"] = "RGB+ED"
        rendered_image, alphas, meta = rasterization(**splat_inputs)
        rendered_image = rendered_image[0].permute(2, 0, 1)
        
        
        expected_depth = rendered_image[3, :, :].detach() if shadow_map else None
        x_pix = torch.arange(W, device="cuda")
        y_pix = torch.arange(H, device="cuda")
        fx, fy = focalx, focaly
        cx, cy = viewpoint_camera.cx, viewpoint_camera.cy
        x_dir = ((x_pix + 0.5) - cx) / fx      # (W,)
        y_dir = ((y_pix + 0.5) - cy) / fy      # (H,)
        X = x_dir.unsqueeze(0) * expected_depth     # (1,W) * (H,W) -> (H,W)
        Y = y_dir.unsqueeze(1) * expected_depth     # (H,1) * (H,W) -> (H,W)
        Z = expected_depth 
        points_cam = torch.stack([X, Y, Z], dim=-1)
        # homo 
        points_cam = torch.cat([points_cam, torch.ones_like(Z).unsqueeze(-1)], dim=-1)
        pts_world = points_cam @ viewpoint_camera.world_view_transform.inverse()
        pts_light = pts_world @ full_persp_proj_transform_light
        pts_light_ndc = pts_light[ : , : , : 2] / pts_light[ : , : , 3:4]
        pts_light_pix_w = ((pts_light_ndc[...,0] + 1.0) * W_light - 1.0) * 0.5
        pts_light_pix_h = ((pts_light_ndc[...,1] + 1.0) * H_light - 1.0) * 0.5
        dpts_light_depth = pts_light[ : , : , 2] 
        smooth = True
        if smooth: 
            u_norm = pts_light_pix_w.div(W_light-1).mul(2).sub(1)  # [-1,1]
            v_norm = pts_light_pix_h.div(H_light-1).mul(2).sub(1)  # [-1,1]
            grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0) 
            shadow_map_4d = depth_image.squeeze().unsqueeze(0).unsqueeze(0)
            sampled = F.grid_sample(
                shadow_map_4d,               # (1,1,H_light,W_light) depth map
                grid,                        # normalized coords
                mode='bilinear',             # bilinear interpolation of the *depth* texels
                padding_mode='border',
                align_corners=True
            )
            stored_depth = sampled.squeeze(0).squeeze(0)
            print("stored_depth", stored_depth)
        else:
            pts_light_pix_h = pts_light_pix_h.round().clamp(0, H_light-1).long()
            pts_light_pix_w = pts_light_pix_w.round().clamp(0, W_light-1).long()
            flat_idx   = (pts_light_pix_h * W_light + pts_light_pix_w).view(-1)     # (H*W,)
            depth_flat = depth_image.contiguous().view(-1)                         # (H_light*W_light,)
            stored_depth = depth_flat[flat_idx]                                     # (H*W,)
            stored_depth = stored_depth.view(pts_light_pix_h.shape)                # (H, W)
        bias = 0.1
        mash_nan = stored_depth.isnan().unsqueeze(0)
        shadow_mask = (dpts_light_depth - bias) <= stored_depth
        shadow_map_shadow = shadow_mask.float().unsqueeze(0)
        shadow_map_shadow[mash_nan] = 0
        shadow_map_shadow += final_T
        shadow_map_shadow = torch.clamp(shadow_map_shadow, 0, 1)
        
    else:
        opacity = temp_opacity                  
        if transparent_layer:
            opacity = original_opacity
            opacity[mask_blanket] = 0
            opacity[mask_scene] = 1
            rendered_image, radii, depth_image, final_T  = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = shs,
                colors_precomp = colors_precomp,
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp,
                shadow = shadow,
                )

        
            opacity[mask_scene] = 0
            opacity[mask_blanket] = 1
            rendered_image_shadow, radii, depth_image, final_T  = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = shs,
                colors_precomp = colors_precomp,
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp,
                shadow = shadow,
                )
        else:
            rendered_image_shadow = None
            rendered_image, radii, depth_image, final_T  = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = shs,
                colors_precomp = colors_precomp,
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp,
                shadow = shadow,
                )
                
                
        
        
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image[0:3, :, :],
        "shadow": rendered_image_shadow[3:4, :, :] if transparent_layer else rendered_image[3:4, :, :],
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image,
        "object_center": object_center,
        "world_view_transform_light": world_view_transform_light,
        "full_ortho_proj_transform_light": full_ortho_proj_transform_light,
        "full_persp_proj_transform_light": full_persp_proj_transform_light,
        "tanfovx_far": tanfovx_far,
        "tanfovy_far": tanfovy_far,
        "camera_position": viewpoint_camera.camera_center,
        "pre_shadow": shadow,
        "ortho_rendered_image": ortho_rendered_image,
        "raster_settings_dict": raster_settings_dict
        }
    
    return out
