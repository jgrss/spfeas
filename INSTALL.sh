#!/bin/bash

brew tap osgeo/osgeo4mac
brew tap homebrew/science
brew tap homebrew/versions
brew install gcc python hdf4 hdf5 spatialindex
brew install gdal-20 --with-hdf4 --with-hdf5
echo /usr/local/opt/gdal-20/lib/python2.7/site-packages >> /usr/local/lib/python2.7/site-packages/gdal-20.pth
brew link --force gdal-20
brew linkapps python
pip install beautifulsoup4 Bottleneck colorama cython joblib matplotlib numpy opencv-python pandas PySAL retrying Rtree scikit-image scikit-learn scipy tables xmltodict
pip install SpFeas-0.0.1.tar.gz
