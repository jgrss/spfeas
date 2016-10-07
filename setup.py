#!/usr/bin/env python

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


__version__ = '0.0.1'

spfeas_name = 'SpFeas'
maintainer = 'Jordan Graesser'
maintainer_email = 'jordan.graesser@mail.mcgill.ca'
description = 'A Python library for processing spatial (contextual) image features and image classification'
git_url = 'http://github.com/spfeas.git'

with open('spfeas/README.md') as f:
    long_description = f.read()

with open('spfeas/LICENSE.txt') as f:
    license_file = f.read()

with open('spfeas/AUTHORS.txt') as f:
    author_file = f.read()

required_packages = ['numpy>=1.11.0', 'scipy>=0.17.1', 'scikit-learn>=0.17.1', 'scikit-image>=0.12.3', 'gdal>=2.1',
                     'numexpr>=2.6.1', 'tables>=3.2.2', 'pandas>=0.18.1', 'matplotlib>=1.5.1', 'psutil>=4.3.1',
                     'joblib>=0.10.2', 'BeautifulSoup4>=4.5.1', 'PyYAML>=3.12', 'colorama>=0.3.7',
                     'cython>=0.24.1', 'bottleneck>=1.1.0', 'xmltodict', 'Rtree>=0.8.2', 'retrying',
                     'opencv-python>=3.1.0', 'PySAL>=1.11.2', 'six>=1.10.0']


def get_packages():
    return setuptools.find_packages()


def get_package_data():

    return {'spfeas': ['*.md',
                       '*.txt',
                       'sphelpers/*.pyx',
                       'sphelpers/*.c',
                       'sphelpers/*.so',
                       'sphelpers/stats/*.pyx',
                       'sphelpers/stats/*.c',
                       'sphelpers/stats/*.so',
                       'notebooks/*.ipynb',
                       'notebooks/*.png']}


def get_pyx_list():
    return ['spfeas/sphelpers/*.pyx', 'spfeas/sphelpers/stats/*.pyx']


def get_console_dict():

    return {'console_scripts': ['spfeas_raster_tools=mappy.raster_tools:main',
                                'spfeas_vector_tools=mappy.vector_tools:main',
                                'spfeas_classification=mappy.classifiers.classification:main',
                                'spfeas=spfeas.spfeas:main',
                                'spfeas_veg_indices=mappy.features.veg_indices:main',
                                'spfeas_sample_raster=mappy.sample.sample_raster:main']}


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
                    zip_safe=False,
                    download_url=git_url,
                    install_requires=required_packages,
                    entry_points=get_console_dict())

    setup(**metadata)

if __name__ == '__main__':
    setup_package()
