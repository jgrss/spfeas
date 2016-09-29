#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np


file2build = '_rolling_stats'

setup(name=file2build,
      ext_modules=[Extension(file2build,
                             ['{}.pyx'.format(file2build)],
                             libraries=['m'],
                             extra_compile_args=['-O3', '-ffast-math', '-march=native', '-fopenmp'],
                             extra_link_args=['-fopenmp'])],
      cmdclass={'build_ext': build_ext},
      include_dirs=[np.get_include()])

# setup(name=file2build, ext_modules=[Extension(file2build, ['{}.pyx'.format(file2build)])],
#       cmdclass={'build_ext': build_ext}, include_dirs=[np.get_include()])
