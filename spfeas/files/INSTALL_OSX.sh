#!/bin/bash

# Ensure the working directory is /Downloads
the_cwd=${PWD##*/}

if [ '$the_cwd' != 'Downloads' ]; then
  cd ~/Downloads/
fi

# When public?
# curl -sL https://github.com/jgrss/spfeas/archive/SpFeas-0.0.1.tar.gz | tar xz

# Upgrade or install Homebrew
if which brew >/dev/null; then
  brew update
  brew upgrade
else
  /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
fi

# Update the .profile
if [ ! -f ~/.profile ]; then
  touch ~/.profile
fi

echo 'export CFLAGS=-I/usr/local/lib/python2.7/site-packages/numpy/core/include/' >>~/.profile
source ~/.profile

# Homebrew
brew tap osgeo/osgeo4mac
brew tap homebrew/science
brew tap homebrew/versions
#brew install gcc python hdf4 hdf5 spatialindex
brew install python hdf4 hdf5 spatialindex
brew linkapps python

# Python pip
if which pip >/dev/null; then
  echo 'pip is installed'
else
  sudo -H easy_install pip
fi

pip install --upgrade setuptools

# Python libraries
sudo -H pip install beautifulsoup4 Bottleneck colorama cython joblib matplotlib numexpr numpy opencv-python pandas psutil PySAL PyYAML retrying Rtree scikit-image scikit-learn scipy six tables xmltodict

brew install gdal2 --with-hdf4 --with-hdf5
echo /usr/local/opt/gdal2/lib/python2.7/site-packages >> /usr/local/lib/python2.7/site-packages/gdal2.pth
brew link --force gdal2

# MpGlue
pip uninstall mpglue
pip install ~/Downloads/MpGlue-0.0.1.tar.gz

# SpFeas
pip uninstall spfeas
pip install ~/Downloads/SpFeas-0.0.1.tar.gz

echo

if which spfeas >/dev/null; then
  echo 'The installation has finished!'
else
  echo 'SpFeas failed to install.'
fi
