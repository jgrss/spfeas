# -*- coding: utf-8 -*-

"""
To build .tar.gz:
    1. Setup directory structure
        build/
            ...
            mappy/
            setup.py
                ...
    2. > cd build/
    3. python setup.py sdist
    4. Upload dist/MapPy-XXX.tar.gz

To install:
    1. Download tar.gz file
    2. > pip install MapPy-XXX.tar.gz

Windows:
    Build the files
        1. setup.py build
    Create the executable installer
        2. setup.py bdist_wininst --target-version=2.7
"""

import setuptools
from distutils.core import setup
import platform

from Cython.Build import cythonize

try:
    from Cython.Distutils import build_ext
except:
    from distutils.command import build_ext

import numpy as np


__version__ = '0.2.0b'

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

required_packages = ['matplotlib>=2.0',
                     'psutil>=4.3.1',
                     'joblib>=0.11.0',
                     'BeautifulSoup4>=4.5.1',
                     'PyYAML>=3.12',
                     'colorama>=0.3.7',
                     'xmltodict',
                     'retrying',
                     'PySAL>=1.11.2',
                     'six>=1.10.0']

if platform.system() != 'Windows':

    for pkg in ['numpy>=1.13.0',
                'scipy>=0.19.0',
                'scikit-image>=0.13',
                'Rtree>=0.8.2',
                'gdal>=2.1',
                'numexpr>=2.6.2',
                'tables>=3.4.2',
                'bottleneck>=1.2.0',
                'statsmodels>=0.8.0',
                'opencv-python>=3.2',
                'cython>=0.26',
                'scikit-learn>=0.18.1',
                'pandas>=0.20.0']:

        required_packages.append(pkg)


def get_packages():
    return setuptools.find_packages()


def get_pyx_list():

    return ['spfeas/helpers/*.pyx',
            'spfeas/sphelpers/*.pyx']


def get_package_data():

    if platform.system() == 'Windows':

        return {'': ['*.md', '*.txt'],
                'spfeas': ['helpers/*.pyd',
                           'sphelpers/*.pyd',
                           'notebooks/*.ipynb',
                           'notebooks/*.png']}

    else:

        return {'': ['*.md', '*.txt'],
                'spfeas': ['helpers/*.so',
                           'helpers/*.pyx',
                           'sphelpers/*.so',
                           'sphelpers/*.pyx',
                           'notebooks/*.ipynb',
                           'notebooks/*.png']}


def get_console_dict():

    return {'console_scripts': ['density=spfeas.density:main',
                                'spfeas=spfeas.spfeas:main']}


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
