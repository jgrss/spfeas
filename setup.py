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


__version__ = '0.1.0'

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

required_packages = ['numpy>=1.12.0', 'scipy>=0.18.1', 'scikit-learn>=0.18.1', 'scikit-image>=0.12.3',
                     'numexpr>=2.6.2', 'tables>=3.3', 'pandas>=0.19.2', 'matplotlib>=1.5.1', 'psutil>=4.3.1',
                     'joblib>=0.10.3', 'BeautifulSoup4>=4.5.1', 'PyYAML>=3.12', 'colorama>=0.3.7',
                     'cython>=0.25.2', 'bottleneck>=1.2.0', 'xmltodict', 'retrying',
                     'opencv-python>=3.1.0', 'PySAL>=1.11.2', 'six>=1.10.0', 'pyFFTW>=0.10.4']

if platform.system() == 'Darwin':

    for pkg in ['Rtree>=0.8.2', 'gdal>=2.1']:
        required_packages.append(pkg)


def get_packages():
    return setuptools.find_packages()


def get_package_data():

    return {'spfeas': ['*.md',
                       '*.txt',
                       'helpers/*.so',
                       'helpers/*.pyd',
                       'sphelpers/*.so',
                       'sphelpers/*.pyd',
                       'notebooks/*.ipynb',
                       'notebooks/*.png']}


# def get_pyx_list():
#
#     return ['spfeas/helpers/*.pyx',
#             'spfeas/sphelpers/*.pyx']


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
                    zip_safe=False,
                    download_url=git_url,
                    install_requires=required_packages,
                    entry_points=get_console_dict())

    setup(**metadata)

if __name__ == '__main__':
    setup_package()
