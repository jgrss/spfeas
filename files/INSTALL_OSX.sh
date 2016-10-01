#!/bin/bash

brew tap osgeo/osgeo4mac
brew tap homebrew/science
brew tap homebrew/versions
brew install gcc python hdf4 hdf5 spatialindex
brew linkapps python
sudo -H pip install beautifulsoup4 Bottleneck colorama cython joblib matplotlib numexpr numpy opencv-python pandas psutil PySAL retrying Rtree scikit-image scikit-learn scipy six tables xmltodict
brew install gdal2 --with-hdf4 --with-hdf5
echo /usr/local/opt/gdal2/lib/python2.7/site-packages >> /usr/local/lib/python2.7/site-packages/gdal2.pth
brew link --force gdal2
pip install SpFeas-0.0.1.tar.gz
