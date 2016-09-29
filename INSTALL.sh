#!/bin/bash

brew tap osgeo/osgeo4mac
brew tap homebrew/science
brew tap homebrew/versions
brew install gcc python hdf4 hdf5 spatialindex
brew install gdal-20 --with-hdf4 --with-hdf5
brew linkapps python
pip install beautifulsoup4 Bottleneck colorama joblib matplotlib numpy opencv-python pandas PySAL scikit-image scikit-learn scipy tables xmltodict
pip install SpFeas-0.0.1.tar.gz
