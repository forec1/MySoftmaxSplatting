from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='sumsplat_cuda',
    ext_modules=[
        CUDAExtension('sumsplat_cuda', [
            'sumsplat_cuda.cpp',
            'sumsplat_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })