#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np


file2build = '_stats'

setup(name=file2build, ext_modules=[Extension(file2build, ['%s.pyx' % file2build])], \
      cmdclass={'build_ext': build_ext}, include_dirs=[np.get_include()])