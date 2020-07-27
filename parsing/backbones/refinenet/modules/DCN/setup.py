from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='DCN',
    ext_modules=[
        CUDAExtension('DCNv2', [
            './ModulatedDeformConv_kernel.cu',
            './ModulatedDeformConv.cpp',
        ])
    ],
    cmdclass={'build_ext': BuildExtension})


#python setup.py build_ext --inplace