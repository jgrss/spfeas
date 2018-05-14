# -*- coding: utf-8 -*-

import setuptools
from distutils.core import setup
import platform

from Cython.Build import cythonize

try:
    from Cython.Distutils import build_ext
except:
    from distutils.command import build_ext

import numpy as np


__version__ = '0.3.0b'

spfeas_name = 'SpFeas'
maintainer = 'Jordan Graesser'
maintainer_email = 'graesser@bu.edu'
description = 'A Python library for processing spatial (contextual) image features and image classification'
git_url = 'http://github.com/jgrss/spfeas.git'

with open('README.md') as f:
    long_description = f.read()

with open('LICENSE.txt') as f:
    license_file = f.read()

with open('AUTHORS.txt') as f:
    author_file = f.read()

required_packages = ['matplotlib>=2.0.0',
                     'psutil>=4.3.1',
                     'joblib>=0.11.0',
                     'BeautifulSoup4>=4.5.1',
                     'PyYAML>=3.12',
                     'colorama>=0.3.7',
                     'xmltodict',
                     'retrying',
                     'future',
                     'PySAL>=1.11.2',
                     'six>=1.11.0']

if platform.system() != 'Windows':

    for pkg in ['numpy>=1.14.0',
                'scipy>=0.19.0',
                'scikit-image>=0.13',
                'Rtree>=0.8.2',
                'gdal>=2.1',
                'numexpr>=2.6.2',
                'tables>=3.4.2',
                'statsmodels>=0.8.0',
                'opencv-python>=3.4.0',
                'cython>=0.28.0',
                'scikit-learn>=0.19.0',
                'pandas>=0.22.0']:

        required_packages.append(pkg)


def get_packages():
    return setuptools.find_packages()


def get_pyx_list():
    return ['spfeas/sphelpers/*.pyx']


def get_package_data():

    return {'': ['*.md', '*.txt'],
            'spfeas': ['sphelpers/*.pyx',
                       'notebooks/*.ipynb',
                       'notebooks/*.png',
                       'data/*.tif',
                       'data/*.tfw',
                       'data/_features/*.yaml',
                       'data/_features/*.txt',
                       'data/_features/*.vrt',
                       'data/_features/test_image__BD1_BK4_SC8_TRmean/*.tif',
                       'data/_features/test_image__BD1_BK2_SC8-16_TRdmp-hog-mean-saliency-rbvi/*.tif',
                       'data/shp/grid*.tif']}


def get_console_dict():
    return {'console_scripts': ['spfeas=spfeas.spfeas:main']}


def setup_package():

    metadata = dict(name=spfeas_name,
                    maintainer=maintainer,
                    maintainer_email=maintainer_email,
                    description=description,
                    license=license_file,
                    version=__version__,
                    long_description=long_description,
                    author=author_file,
                    packages=get_packages(),
                    package_data=get_package_data(),
                    ext_modules=cythonize(get_pyx_list()),
                    include_dirs=[np.get_include()],
                    cmdclass=dict(build_ext=build_ext),
                    zip_safe=False,
                    download_url=git_url,
                    install_requires=required_packages,
                    entry_points=get_console_dict())

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
