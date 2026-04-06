import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image


BG_COLOR = np.array([1.0, 1.0, 1.0], dtype=np.float32)
DEFAULT_LIGHT_INTENSITY = [1.0, 1.0, 1.0]


@dataclass
class MeshObject:
    name: str
    mesh: o3d.geometry.TriangleMesh
    color: np.ndarray
    sample_count: int


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, 1e-8, None)


def look_at_c2w(camera_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = normalize((target - camera_pos)[None, :])[0]
    up_guess = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(forward, up_guess))) > 0.98:
        up_guess = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = normalize(np.cross(forward, up_guess)[None, :])[0]
    up = normalize(np.cross(right, forward)[None, :])[0]

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = -forward
    c2w[:3, 3] = camera_pos
    return c2w


def make_transform(translation, rotation_deg=(0.0, 0.0, 0.0)) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz(
        np.radians(np.asarray(rotation_deg, dtype=np.float64))
    )
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def apply_transform(mesh: o3d.geometry.TriangleMesh, transform: np.ndarray) -> o3d.geometry.TriangleMesh:
    mesh_out = o3d.geometry.TriangleMesh(mesh)
    mesh_out.transform(transform)
    mesh_out.compute_vertex_normals()
    return mesh_out


def create_floor(size_xy, thickness, color) -> MeshObject:
    width, depth = size_xy
    mesh = o3d.geometry.TriangleMesh.create_box(width=width, height=depth, depth=thickness)
    mesh = apply_transform(mesh, make_transform((-width * 0.5, -depth * 0.5, -thickness)))
    return MeshObject("floor", mesh, np.asarray(color, dtype=np.float32), 2800)


def create_centered_box(width, height, depth) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=depth)
    mesh.translate((-width * 0.5, -height * 0.5, -depth * 0.5))
    mesh.compute_vertex_normals()
    return mesh


def create_studio_scene():
    objects = [create_floor((4.8, 4.2), 0.08, [0.86, 0.84, 0.80])]

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.45, resolution=48)
    objects.append(
        MeshObject(
            "sphere",
            apply_transform(sphere, make_transform((-0.85, -0.15, 0.45))),
            np.array([0.68, 0.80, 0.96], dtype=np.float32),
            2200,
        )
    )

    box = create_centered_box(width=0.80, height=0.65, depth=1.00)
    objects.append(
        MeshObject(
            "box",
            apply_transform(box, make_transform((0.55, -0.10, 0.48), (14.0, 11.0, 35.0))),
            np.array([0.96, 0.70, 0.53], dtype=np.float32),
            2000,
        )
    )

    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.26, height=1.15, resolution=48, split=8)
    objects.append(
        MeshObject(
            "cylinder",
            apply_transform(cylinder, make_transform((0.05, 0.95, 0.58), (15.0, 0.0, -22.0))),
            np.array([0.58, 0.85, 0.64], dtype=np.float32),
            1800,
        )
    )

    look_at = np.array([0.0, 0.15, 0.48], dtype=np.float32)
    light_pos = np.array([2.9, -2.1, 3.7], dtype=np.float32)
    return objects, look_at, light_pos


def create_single_object_scene():
    objects = [create_floor((3.8, 3.8), 0.08, [0.88, 0.87, 0.83])]

    torus = o3d.geometry.TriangleMesh.create_torus(torus_radius=0.55, tube_radius=0.18, radial_resolution=44, tubular_resolution=64)
    objects.append(
        MeshObject(
            "torus",
            apply_transform(torus, make_transform((0.0, 0.0, 0.62), (70.0, 15.0, 18.0))),
            np.array([0.69, 0.76, 0.95], dtype=np.float32),
            3200,
        )
    )

    look_at = np.array([0.0, 0.0, 0.55], dtype=np.float32)
    light_pos = np.array([2.5, -1.8, 3.4], dtype=np.float32)
    return objects, look_at, light_pos


def build_scene(preset: str):
    if preset == "single_object":
        return create_single_object_scene()
    return create_studio_scene()


def build_camera_dirs(width: int, height: int, fov_x: float) -> np.ndarray:
    xs = (2.0 * ((np.arange(width, dtype=np.float32) + 0.5) / width) - 1.0) * math.tan(fov_x * 0.5)
    ys = (1.0 - 2.0 * ((np.arange(height, dtype=np.float32) + 0.5) / height)) * math.tan(fov_x * 0.5) * (height / width)
    grid_x, grid_y = np.meshgrid(xs, ys)
    dirs = np.stack([grid_x, grid_y, -np.ones_like(grid_x)], axis=-1)
    return normalize(dirs.reshape(-1, 3)).astype(np.float32)


def build_camera_poses(train_views: int, test_views: int, radius: float, look_at: np.ndarray):
    train = []
    for idx in range(train_views):
        theta = 2.0 * math.pi * idx / max(train_views, 1)
        height = 1.05 + 0.18 * math.sin(theta * 1.7)
        camera_pos = np.array(
            [radius * math.cos(theta), radius * math.sin(theta), height],
            dtype=np.float32,
        )
        train.append(look_at_c2w(camera_pos, look_at))

    test = []
    for idx in range(test_views):
        theta = 2.0 * math.pi * (idx + 0.5) / max(test_views, 1)
        height = 0.95 + 0.15 * math.cos(theta * 1.3)
        camera_pos = np.array(
            [radius * math.cos(theta), radius * math.sin(theta), height],
            dtype=np.float32,
        )
        test.append(look_at_c2w(camera_pos, look_at))

    return train, test


def write_ply(path: Path, points: np.ndarray, normals: np.ndarray, colors: np.ndarray):
    header = "\n".join(
        [
            "ply",
            "format ascii 1.0",
            f"element vertex {points.shape[0]}",
            "property float x",
            "property float y",
            "property float z",
            "property float nx",
            "property float ny",
            "property float nz",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
        ]
    )
    with path.open("w", encoding="utf-8") as f:
        f.write(header + "\n")
        for p, n, c in zip(points, normals, colors):
            f.write(
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                f"{n[0]:.6f} {n[1]:.6f} {n[2]:.6f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )


def sample_scene_point_cloud(objects):
    points_all = []
    normals_all = []
    colors_all = []
    for obj in objects:
        sampled = obj.mesh.sample_points_uniformly(number_of_points=obj.sample_count, use_triangle_normal=True)
        points = np.asarray(sampled.points, dtype=np.float32)
        normals = np.asarray(sampled.normals, dtype=np.float32)
        colors = np.tile((obj.color * 255.0).astype(np.uint8)[None, :], (points.shape[0], 1))
        points_all.append(points)
        normals_all.append(normals)
        colors_all.append(colors)
    return (
        np.concatenate(points_all, axis=0),
        normalize(np.concatenate(normals_all, axis=0)).astype(np.float32),
        np.concatenate(colors_all, axis=0),
    )


def render_view(scene, geometry_colors, c2w, width, height, fov_x, light_pos):
    cam_dirs = build_camera_dirs(width, height, fov_x)
    rotation = c2w[:3, :3]
    camera_pos = c2w[:3, 3]
    world_dirs = normalize(cam_dirs @ rotation.T).astype(np.float32)
    origins = np.repeat(camera_pos[None, :], width * height, axis=0).astype(np.float32)

    rays = np.concatenate([origins, world_dirs], axis=1)
    cast = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))

    t_hit = cast["t_hit"].numpy()
    geometry_ids = cast["geometry_ids"].numpy().astype(np.int64)
    normals = normalize(cast["primitive_normals"].numpy()).astype(np.float32)

    rgb = np.ones((width * height, 3), dtype=np.float32) * BG_COLOR[None, :]
    alpha = np.zeros((width * height, 1), dtype=np.float32)

    hit_mask = np.isfinite(t_hit)
    if np.any(hit_mask):
        hit_points = origins[hit_mask] + world_dirs[hit_mask] * t_hit[hit_mask, None]
        hit_normals = normals[hit_mask]
        hit_geom = geometry_ids[hit_mask]
        hit_colors = np.stack([geometry_colors[g] for g in hit_geom], axis=0)

        light_vec = light_pos[None, :] - hit_points
        light_dist = np.linalg.norm(light_vec, axis=1)
        light_dir = normalize(light_vec).astype(np.float32)

        shadow_origins = hit_points + hit_normals * 2e-3 + light_dir * 1e-3
        shadow_rays = np.concatenate([shadow_origins, light_dir], axis=1)
        shadow_cast = scene.cast_rays(o3d.core.Tensor(shadow_rays, dtype=o3d.core.Dtype.Float32))
        shadow_t = shadow_cast["t_hit"].numpy()
        shadow_mask = np.isfinite(shadow_t) & (shadow_t < (light_dist - 5e-3))

        ndotl = np.clip(np.sum(hit_normals * light_dir, axis=1), 0.0, 1.0)
        view_dir = normalize(camera_pos[None, :] - hit_points).astype(np.float32)
        half_dir = normalize(light_dir + view_dir).astype(np.float32)
        specular = np.power(np.clip(np.sum(hit_normals * half_dir, axis=1), 0.0, 1.0), 56.0)
        specular = np.where(shadow_mask, 0.0, specular * 0.08)

        ambient = 0.20
        diffuse = 0.90 * ndotl
        diffuse = np.where(shadow_mask, diffuse * 0.12, diffuse)
        linear_rgb = hit_colors * (ambient + diffuse[:, None]) + specular[:, None]
        rgb[hit_mask] = np.clip(linear_rgb, 0.0, 1.0)
        alpha[hit_mask] = 1.0

    rgba = np.concatenate([np.power(np.clip(rgb, 0.0, 1.0), 1.0 / 2.2), alpha], axis=1)
    return (rgba.reshape(height, width, 4) * 255.0 + 0.5).astype(np.uint8)


def frame_dict(file_path: str, c2w: np.ndarray, light_pos: np.ndarray, light_intensity):
    return {
        "file_path": file_path,
        "transform_matrix": c2w.tolist(),
        "pl_pos": light_pos.tolist(),
        "pl_intensity": light_intensity,
    }


def save_image(path: Path, image_rgba: np.ndarray):
    Image.fromarray(image_rgba, mode="RGBA").save(path)


def make_dataset(
    output_root: Path,
    preset: str,
    width: int,
    height: int,
    fov_x_deg: float,
    train_views: int,
    test_views: int,
    radius: float,
    light_intensity,
):
    objects, look_at, light_pos = build_scene(preset)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "train").mkdir(exist_ok=True)
    (output_root / "test").mkdir(exist_ok=True)

    scene = o3d.t.geometry.RaycastingScene()
    geometry_colors = {}
    for obj in objects:
        geom_id = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(obj.mesh))
        geometry_colors[int(geom_id)] = obj.color

    fov_x = math.radians(fov_x_deg)
    train_poses, test_poses = build_camera_poses(train_views, test_views, radius, look_at)

    train_frames = []
    for idx, c2w in enumerate(train_poses):
        image = render_view(scene, geometry_colors, c2w, width, height, fov_x, light_pos)
        rel_path = f"train/r_{idx:03d}.png"
        save_image(output_root / rel_path, image)
        train_frames.append(frame_dict(rel_path, c2w, light_pos, light_intensity))

    test_frames = []
    for idx, c2w in enumerate(test_poses):
        image = render_view(scene, geometry_colors, c2w, width, height, fov_x, light_pos)
        rel_path = f"test/r_{idx:03d}.png"
        save_image(output_root / rel_path, image)
        test_frames.append(frame_dict(rel_path, c2w, light_pos, light_intensity))

    with (output_root / "transforms_train.json").open("w", encoding="utf-8") as f:
        json.dump({"camera_angle_x": fov_x, "frames": train_frames}, f, indent=2)
    with (output_root / "transforms_test.json").open("w", encoding="utf-8") as f:
        json.dump({"camera_angle_x": fov_x, "frames": test_frames}, f, indent=2)

    points, normals, colors = sample_scene_point_cloud(objects)
    write_ply(output_root / "points3d.ply", points, normals, colors)

    metadata = {
        "preset": preset,
        "width": width,
        "height": height,
        "train_views": train_views,
        "test_views": test_views,
        "fov_x_deg": fov_x_deg,
        "light_pos": light_pos.tolist(),
        "light_intensity": light_intensity,
        "look_at": look_at.tolist(),
        "point_count": int(points.shape[0]),
    }
    with (output_root / "scene_meta.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote mesh shadow dataset to {output_root.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Generate a Blender-format synthetic shadow dataset from meshes.")
    parser.add_argument("--output", default="synthetic_shadow_mesh")
    parser.add_argument("--preset", choices=["studio", "single_object"], default="studio")
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=160)
    parser.add_argument("--fov-x-deg", type=float, default=45.0)
    parser.add_argument("--train-views", type=int, default=12)
    parser.add_argument("--test-views", type=int, default=6)
    parser.add_argument("--radius", type=float, default=3.4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output)
    if output_root.exists() and args.force:
        shutil.rmtree(output_root)

    make_dataset(
        output_root=output_root,
        preset=args.preset,
        width=args.width,
        height=args.height,
        fov_x_deg=args.fov_x_deg,
        train_views=args.train_views,
        test_views=args.test_views,
        radius=args.radius,
        light_intensity=DEFAULT_LIGHT_INTENSITY,
    )


if __name__ == "__main__":
    main()
