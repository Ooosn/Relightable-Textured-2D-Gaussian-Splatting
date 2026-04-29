from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))


if False:
    extra_compile_args={
                "cxx": ["-g"],
                "nvcc": [
                    "-g", "-G",  # 添加这两个用于 device 调试
                    "-lineinfo",  # 可选但推荐：Nsight Compute/Systems 用
                    "-I" + os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "third_party/glm/")
                ]
            }
else:
    extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]}


setup(
    name="ortho_light",
    packages=['ortho_light'],
    ext_modules=[
        CUDAExtension(
            name="ortho_light._C",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
                "rasterize_points.cu",
                "ext.cpp"
            ],
            extra_compile_args=extra_compile_args
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
