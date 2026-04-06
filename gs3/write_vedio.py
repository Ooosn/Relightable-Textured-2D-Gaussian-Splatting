import os, re
from pathlib import Path
import cv2
import numpy as np

def natural_key(s):
    # 自然排序：frame2.png < frame10.png
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))]

def letterbox(img, target_wh, bg_color=(0,0,0)):
    """按比例缩放并居中填充到目标尺寸（避免拉伸）"""
    h, w = img.shape[:2]
    tw, th = target_wh
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((th, tw, 3), bg_color, dtype=np.uint8)
    y0 = (th - nh) // 2
    x0 = (tw - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

def images_to_video(
    image_folders,                       # list[str] 或 dict[str, any] 的 keys（文件夹路径）
    out_path,                            # 输出视频路径，如 'out.mp4'
    fps=30,
    size='auto',                         # 'auto' 用第一张图定尺寸；或传 (W, H)
    codec='mp4v',                        # 常用：'mp4v'（兼容广），或 'H264'（需要编码器）
    letterbox_fill=True,                 # True: 等比不拉伸；False: 直接拉伸
    bg_color=(0,0,0)
):
    # 把传入的 image_folders 统一成列表
    if isinstance(image_folders, dict):
        folders = list(image_folders.keys())
    elif isinstance(image_folders, (list, tuple)):
        folders = list(image_folders)
    else:
        folders = [image_folders]

    # 收集所有图片（按文件夹顺序 + 自然排序）
    all_imgs = []
    for key in folders:
        folder = Path(key)
        files = sorted([p for p in folder.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}],
                       key=natural_key)
        all_imgs.extend(files)

    if not all_imgs:
        raise RuntimeError("未找到任何图片。")

    # 确定帧尺寸
    if size == 'auto':
        probe = cv2.imread(str(all_imgs[0]))
        if probe is None:
            raise RuntimeError(f"无法读取首帧：{all_imgs[0]}")
        H, W = probe.shape[:2]
    else:
        W, H = size
    print("W, H:", W, H)
    # 初始化写入器
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
    if not out.isOpened():
        raise RuntimeError("VideoWriter 打不开。可能是编码器不支持，换个 codec（例如 'mp4v'）或换扩展名试试。")

    # 写帧
    for p in all_imgs:
        img = cv2.imread(str(p))  # BGR
        if img is None:
            print(f"[Warn] 跳过无法读取的图片：{p}")
            continue
        if img.shape[1] != W or img.shape[0] != H:
            img = letterbox(img, (W, H), bg_color) if letterbox_fill else cv2.resize(img, (W, H))
        out.write(img)

    out.release()
    print(f"✅ 已保存：{out_path}")


# render_path = "path/to/render"
# video_path = os.path.join(render_path, 'video')
# os.makedirs(video_path, exist_ok=True)

# # 假设 image_folder 是 dict，key 就是文件夹路径
# image_folder = {
#     os.path.join(render_path, "seq1"): None,
#     os.path.join(render_path, "seq2"): None,
# }
# out_file = os.path.join(video_path, "merged.mp4")

# images_to_video(image_folder, out_file, fps=30, size=(1920,1080), codec='mp4v')