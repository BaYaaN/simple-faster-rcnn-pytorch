from distutils.core import setup, Extension

import numpy
from Cython.Distutils import build_ext

setup(
    name="Hello pyx",
    cmdclass={'build_ext': build_ext},
    ext_modules=[
        Extension("_nms_gpu_post", ["_nms_gpu_post.pyx"],
                  include_dirs=[numpy.get_include()]),
    ],
)
