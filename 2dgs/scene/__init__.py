#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
import torch
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func
from scene.dataset_readers import sceneLoadTypeCallbacks, readCamerasFromTransforms
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.graphics_utils import fov2focal

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], opt=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            view_num = getattr(args, "view_num", -1)
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, view_num)
        else:
            assert False, "Could not recognize scene type!"

        if self.loaded_iter:
            iteration_path = os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter))
            train_pose_path = os.path.join(iteration_path, "transforms_train.json")
            test_pose_path = os.path.join(iteration_path, "transforms_test.json")
            train_cameras = scene_info.train_cameras
            test_cameras = scene_info.test_cameras
            if os.path.exists(train_pose_path):
                train_cameras = readCamerasFromTransforms(
                    iteration_path,
                    "transforms_train.json",
                    args.white_background,
                    getattr(args, "view_num", -1),
                )
            if os.path.exists(test_pose_path):
                test_cameras = readCamerasFromTransforms(
                    iteration_path,
                    "transforms_test.json",
                    args.white_background,
                    -1,
                )
            scene_info = scene_info._replace(
                train_cameras=train_cameras,
                test_cameras=test_cameras,
            )

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
        
        if self.loaded_iter:
            iteration_path = os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter))
            self.gaussians.load_ply(os.path.join(iteration_path, "point_cloud.ply"))
            self.gaussians.load_appearance(os.path.join(iteration_path, "appearance.pt"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

        # ── Camera & light pose optimizer ────────────────────────────────────
        self.cam_opt = getattr(args, "cam_opt", False)
        self.pl_opt  = getattr(args, "pl_opt",  False)
        self.pose_optimizer = None

        if (self.cam_opt or self.pl_opt) and opt is not None:
            self.cam_scheduler_args = get_expon_lr_func(
                lr_init=opt.opt_cam_lr_init,
                lr_final=opt.opt_cam_lr_final,
                lr_delay_steps=opt.opt_cam_lr_delay_step,
                lr_delay_mult=opt.opt_cam_lr_delay_mult,
                max_steps=opt.opt_cam_lr_max_steps,
            )
            self.pl_scheduler_args = get_expon_lr_func(
                lr_init=opt.opt_pl_lr_init,
                lr_final=opt.opt_pl_lr_final,
                lr_delay_steps=opt.opt_pl_lr_delay_step,
                lr_delay_mult=opt.opt_pl_lr_delay_mult,
                max_steps=opt.opt_pl_lr_max_steps,
            )
            cam_params, pl_params = [], []
            for scale in resolution_scales:
                for cam in self.train_cameras[scale]:
                    cam_params.append(cam.cam_pose_adj)
                    pl_params.append(cam.pl_adj)
                for cam in self.test_cameras[scale]:
                    cam_params.append(cam.cam_pose_adj)
                    pl_params.append(cam.pl_adj)
            self.pose_optimizer = torch.optim.Adam(
                [
                    {"params": cam_params, "lr": 0.0, "name": "cam_adj"},
                    {"params": pl_params,  "lr": 0.0, "name": "pl_adj"},
                ],
                lr=0, eps=1e-15,
            )

    def update_lr(self, iteration, opt):
        """Unfreeze and update learning rates for camera/light optimizers."""
        if self.pose_optimizer is None:
            return
        for pg in self.pose_optimizer.param_groups:
            if pg["name"] == "cam_adj":
                pg["lr"] = 0.0 if (not self.cam_opt or iteration < opt.train_cam_freeze_step) \
                           else self.cam_scheduler_args(iteration)
            elif pg["name"] == "pl_adj":
                pg["lr"] = 0.0 if (not self.pl_opt or iteration < opt.train_pl_freeze_step) \
                           else self.pl_scheduler_args(iteration)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_appearance(os.path.join(point_cloud_path, "appearance.pt"))
        if self.pose_optimizer is not None:
            scale = 1.0
            ref_cam = self.train_cameras[scale][0]
            intrinsics = [
                ref_cam.cx,
                ref_cam.cy,
                fov2focal(ref_cam.FoVx, ref_cam.image_width),
                fov2focal(ref_cam.FoVy, ref_cam.image_height),
            ]

            def _dump(split_name, cameras):
                payload = {"camera_intrinsics": intrinsics, "frames": []}
                for cam in cameras:
                    R, T, pl_pos = cam.get()
                    payload["frames"].append(
                        {
                            "file_path": cam.image_name,
                            "img_path": cam.image_path,
                            "R_opt": R.tolist(),
                            "T_opt": T.tolist(),
                            "pl_pos": None if pl_pos is None else pl_pos[0].tolist(),
                            "camera_intrinsics": [
                                cam.cx,
                                cam.cy,
                                fov2focal(cam.FoVx, cam.image_width),
                                fov2focal(cam.FoVy, cam.image_height),
                            ],
                        }
                    )
                with open(os.path.join(point_cloud_path, f"transforms_{split_name}.json"), "w") as f:
                    json.dump(payload, f)

            _dump("train", self.train_cameras[scale])
            _dump("test", self.test_cameras[scale])

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
