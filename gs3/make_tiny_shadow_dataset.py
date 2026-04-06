import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path("synthetic_shadow_tiny")
W = 64
H = 64
FOV_X = math.radians(50.0)
LIGHT_POS = np.array([1.8, 1.2, 2.5], dtype=np.float32)
LIGHT_INTENSITY = [1.0, 1.0, 1.0]

SPHERE_CENTER = np.array([0.0, 0.0, 0.0], dtype=np.float32)
SPHERE_RADIUS = 0.33
PLANE_Z = -0.33
LOOK_AT = np.array([0.0, 0.0, -0.05], dtype=np.float32)

SPHERE_COLOR = np.array([0.62, 0.73, 0.92], dtype=np.float32)
PLANE_COLOR = np.array([0.85, 0.82, 0.76], dtype=np.float32)
BG_COLOR = np.array([1.0, 1.0, 1.0], dtype=np.float32)


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-8:
        return v
    return v / n


def look_at_c2w(camera_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    # OpenGL / NeRF style camera: local -Z points forward.
    forward = normalize(target - camera_pos)
    up_guess = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(np.dot(forward, up_guess)) > 0.98:
        up_guess = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = normalize(np.cross(forward, up_guess))
    up = normalize(np.cross(right, forward))

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = -forward
    c2w[:3, 3] = camera_pos
    return c2w


def ray_dir_camera(px: int, py: int) -> np.ndarray:
    x = (2.0 * ((px + 0.5) / W) - 1.0) * math.tan(FOV_X * 0.5)
    y = (1.0 - 2.0 * ((py + 0.5) / H)) * math.tan(FOV_X * 0.5) * (H / W)
    return normalize(np.array([x, y, -1.0], dtype=np.float32))


def intersect_sphere(ray_o: np.ndarray, ray_d: np.ndarray):
    oc = ray_o - SPHERE_CENTER
    b = np.dot(oc, ray_d)
    c = np.dot(oc, oc) - SPHERE_RADIUS * SPHERE_RADIUS
    disc = b * b - c
    if disc < 0.0:
        return None
    s = math.sqrt(float(disc))
    t0 = -b - s
    t1 = -b + s
    eps = 1e-4
    if t0 > eps:
        return t0
    if t1 > eps:
        return t1
    return None


def intersect_plane(ray_o: np.ndarray, ray_d: np.ndarray):
    denom = ray_d[2]
    if abs(denom) < 1e-6:
        return None
    t = (PLANE_Z - ray_o[2]) / denom
    if t <= 1e-4:
        return None
    return t


def occluded_by_sphere(point: np.ndarray, light_pos: np.ndarray) -> bool:
    to_light = light_pos - point
    max_t = np.linalg.norm(to_light)
    ray_d = normalize(to_light)
    ray_o = point + ray_d * 1e-3
    hit_t = intersect_sphere(ray_o, ray_d)
    return hit_t is not None and hit_t < max_t - 1e-3


def shade(point: np.ndarray, normal: np.ndarray, base_color: np.ndarray, in_shadow: bool) -> np.ndarray:
    light_dir = normalize(LIGHT_POS - point)
    ndotl = max(0.0, float(np.dot(normal, light_dir)))
    ambient = 0.28
    diffuse = 0.72 * ndotl
    if in_shadow:
        diffuse *= 0.12
    color = base_color * (ambient + diffuse)
    return np.clip(color, 0.0, 1.0)


def render_view(c2w: np.ndarray) -> np.ndarray:
    image = np.zeros((H, W, 4), dtype=np.uint8)
    cam_pos = c2w[:3, 3]
    rot = c2w[:3, :3]

    for py in range(H):
        for px in range(W):
            ray_d = normalize(rot @ ray_dir_camera(px, py))
            ray_o = cam_pos

            t_sphere = intersect_sphere(ray_o, ray_d)
            t_plane = intersect_plane(ray_o, ray_d)

            hit_color = BG_COLOR
            alpha = 1.0

            choose_sphere = False
            if t_sphere is not None:
                if t_plane is None or t_sphere < t_plane:
                    choose_sphere = True

            if choose_sphere:
                point = ray_o + t_sphere * ray_d
                normal = normalize(point - SPHERE_CENTER)
                in_shadow = False
                hit_color = shade(point, normal, SPHERE_COLOR, in_shadow)
            elif t_plane is not None:
                point = ray_o + t_plane * ray_d
                if abs(point[0]) <= 1.3 and abs(point[1]) <= 1.3:
                    normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                    in_shadow = occluded_by_sphere(point, LIGHT_POS)
                    hit_color = shade(point, normal, PLANE_COLOR, in_shadow)
                else:
                    hit_color = BG_COLOR

            image[py, px, :3] = np.clip(hit_color * 255.0, 0, 255).astype(np.uint8)
            image[py, px, 3] = int(alpha * 255)
    return image


def sample_point_cloud():
    rng = np.random.default_rng(0)

    sphere_pts = []
    sphere_cols = []
    for _ in range(1200):
        d = rng.normal(size=3)
        d = d / np.linalg.norm(d)
        p = SPHERE_CENTER + d * SPHERE_RADIUS
        sphere_pts.append(p)
        sphere_cols.append((SPHERE_COLOR * 255.0).astype(np.uint8))

    plane_pts = []
    plane_cols = []
    xs = np.linspace(-1.1, 1.1, 28)
    ys = np.linspace(-1.1, 1.1, 28)
    for x in xs:
        for y in ys:
            plane_pts.append(np.array([x, y, PLANE_Z], dtype=np.float32))
            plane_cols.append((PLANE_COLOR * 255.0).astype(np.uint8))

    points = np.asarray(sphere_pts + plane_pts, dtype=np.float32)
    colors = np.asarray(sphere_cols + plane_cols, dtype=np.uint8)
    return points, colors


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray):
    normals = np.zeros_like(points, dtype=np.float32)
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


def frame_dict(file_path: str, c2w: np.ndarray):
    return {
        "file_path": file_path,
        "transform_matrix": c2w.tolist(),
        "pl_pos": LIGHT_POS.tolist(),
        "pl_intensity": LIGHT_INTENSITY,
    }


def orbit_pose(angle_deg: float, radius: float = 2.2, height: float = 0.9) -> np.ndarray:
    theta = math.radians(angle_deg)
    cam_pos = np.array(
        [radius * math.cos(theta), radius * math.sin(theta), height],
        dtype=np.float32,
    )
    return look_at_c2w(cam_pos, LOOK_AT)


def main():
    ROOT.mkdir(exist_ok=True)
    (ROOT / "train").mkdir(exist_ok=True)
    (ROOT / "test").mkdir(exist_ok=True)

    train_angles = [0.0, 90.0, 180.0, 270.0]
    test_angles = [45.0, 225.0]

    train_frames = []
    for idx, angle in enumerate(train_angles):
        c2w = orbit_pose(angle)
        image = render_view(c2w)
        name = f"train/r_{idx:03d}.png"
        Image.fromarray(image, mode="RGBA").save(ROOT / name)
        train_frames.append(frame_dict(name, c2w))

    test_frames = []
    for idx, angle in enumerate(test_angles):
        c2w = orbit_pose(angle)
        image = render_view(c2w)
        name = f"test/r_{idx:03d}.png"
        Image.fromarray(image, mode="RGBA").save(ROOT / name)
        test_frames.append(frame_dict(name, c2w))

    transforms_train = {
        "camera_angle_x": FOV_X,
        "frames": train_frames,
    }
    transforms_test = {
        "camera_angle_x": FOV_X,
        "frames": test_frames,
    }

    with (ROOT / "transforms_train.json").open("w", encoding="utf-8") as f:
        json.dump(transforms_train, f, indent=2)
    with (ROOT / "transforms_test.json").open("w", encoding="utf-8") as f:
        json.dump(transforms_test, f, indent=2)

    points, colors = sample_point_cloud()
    write_ply(ROOT / "points3d.ply", points, colors)

    print(f"Wrote tiny synthetic dataset to {ROOT.resolve()}")


if __name__ == "__main__":
    main()
