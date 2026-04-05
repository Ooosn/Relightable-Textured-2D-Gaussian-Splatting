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

from errno import EEXIST
from os import makedirs, path
import os

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def searchForMaxIteration(folder):
    """
    在给定的 point_cloud 目录下，寻找形如：
        iteration_7000
        iteration_30000
    这样的子目录 / 文件，并返回最大的迭代数。

    原版代码会直接对所有名字做 int(...)，如果里面有 "images" 之类的目录，
    就会触发 ValueError。这里做一次过滤 + try/except，忽略掉非数字的名字。
    """
    saved_iters = []
    for fname in os.listdir(folder):
        # 只考虑包含下划线的名字，例如 "iteration_7000"
        if "_" not in fname:
            continue
        last_part = fname.split("_")[-1]
        try:
            it = int(last_part)
            saved_iters.append(it)
        except ValueError:
            # 比如 "images"、"foo_bar" 这类就直接跳过
            continue

    if not saved_iters:
        raise RuntimeError(f"No valid iteration folders/files found in '{folder}'. "
                           f"Expected names like 'iteration_7000'. Found: {os.listdir(folder)}")

    return max(saved_iters)
