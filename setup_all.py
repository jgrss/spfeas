#!/usr/bin/env python

import os
import shutil
import argparse

# try:
#      from Cython.distutils import build_ext
# except ImportError:
#      from distutils.command import build_ext

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np


# def get_pyx_list(pyx_dirs):
#     return [os.path.join(pyx_d, '*.pyx') for pyx_d in pyx_dirs]


# pyx_dirs = ['spfeas/spfeas/helpers']


def get_extensions(extension_key):

    extension_dict = {os.path.join('spfeas', 'moving'): Extension(os.path.join('spfeas', 'spfeas', 'helpers', '_moving_window'),
                                                                  [os.path.join('spfeas', 'spfeas', 'helpers', '_moving_window.pyx')]),
                      'spfeas/chunk': Extension(os.path.join('spfeas', 'spfeas', 'sphelpers', '_chunk'),
                                                [os.path.join('spfeas', 'spfeas', 'sphelpers', '_chunk.pyx')]),
                      'spfeas/stats': Extension(os.path.join('spfeas', 'spfeas', 'sphelpers', '_stats'),
                                                [os.path.join('spfeas', 'spfeas', 'sphelpers', '_stats.pyx')]),
                      'mpglue/rolling': Extension(os.path.join('mpglue', 'mpglue', 'stats', '_rolling_stats'),
                                                  [os.path.join('spfeas', 'spfeas', 'stats', '_rolling_stats.pyx')]),
                      'mpglue/interp': Extension(os.path.join('mpglue', 'mpglue', 'stats', '_lin_interp'),
                                                 [os.path.join('spfeas', 'spfeas', 'stats', '_lin_interp.pyx')])
                      }

    return extension_dict[extension_key]


# extensions = [Extension('spfeas', ['spfeas/spfeas/helpers/*.pyx'])]


# def main():
#
#     parser = argparse.ArgumentParser(description='Setup pyx',
#                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#
#     parser.add_argument('--name', dest='name', help='The setup name', default=None,
#                         choices=['spfeas/moving', 'spfeas/chunk', 'spfeas/stats', 'mpglue/rolling'])
#
#     args = parser.parse_args()
print os.path.join('spfeas', 'spfeas', 'helpers', '_moving_window')
# name = os.path.join('spfeas', 'moving')
#
# # print get_extensions(name)
# # print os.path.join('spfeas', 'spfeas', 'helpers', '_moving_window.pyx')
# setup(ext_modules=cythonize(get_extensions(name)),
#       cmdclass={'build_ext': build_ext},
#       include_dirs=[np.get_include()])
#
# build_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'build')
#
# if os.path.isdir(build_dir):
#     shutil.rmtree(build_dir)

# if __name__ == '__main__':
#     main()
