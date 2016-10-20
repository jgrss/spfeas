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

if [ -z ${CFLAGS} ]; then 
  echo 'export CFLAGS=-I/usr/local/lib/python2.7/site-packages/numpy/core/include/' >>~/.profile
  source ~/.profile
else
  echo "CFLAGS is already set to '$CFLAGS'"
fi

# Homebrew
brew tap osgeo/osgeo4mac
brew tap homebrew/science
brew tap homebrew/versions
#brew install gcc python hdf4 hdf5 spatialindex

# Python from Homebrew
if which python >/dev/null; then
  echo 'Python is already installed'
else
  brew install python
  brew linkapps python
fi

# Python pip
if which pip >/dev/null; then
  echo 'pip is already installed'
else
  sudo -H easy_install pip
fi

# Upgrade Python setuptools
pip install --upgrade setuptools

LINE_BREAK1='========================================='

echo
echo $LINE_BREAK1
echo 'If prompted, enter your computer password'
echo $LINE_BREAK1
echo

# Python libraries
sudo -H pip install beautifulsoup4 Bottleneck colorama cython joblib matplotlib numexpr numpy opencv-python pandas psutil PySAL PyYAML retrying Rtree scikit-image scikit-learn scipy six tables xmltodict

# GDAL from Homebrew
if which gdalinfo >/dev/null; then
  echo 'GDAL is already installed'
else
  brew install hdf4 hdf5 spatialindex
  brew install gdal2 --with-hdf4 --with-hdf5
  echo /usr/local/opt/gdal2/lib/python2.7/site-packages >> /usr/local/lib/python2.7/site-packages/gdal2.pth
  brew link --force gdal2
fi

LINE_BREAK2='======================================='

echo
echo $LINE_BREAK2
echo "Type 'y' to uninstall MpGlue and SpFeas"
echo $LINE_BREAK2
echo

# MpGlue
if which classify >/dev/null; then
  pip uninstall mpglue
fi

LINE_BREAK3='===================================================='

echo
echo $LINE_BREAK3
echo 'If prompted, enter your GitHub username and password'
echo $LINE_BREAK3
echo

pip install git+https://github.com/jgrss/mpglue.git
# pip install ~/Downloads/MpGlue-0.0.1.tar.gz

# SpFeas
if which classify >/dev/null; then
  pip uninstall spfeas
fi

echo
echo $LINE_BREAK
echo 'If prompted, enter your GitHub username and password'
echo $LINE_BREAK
echo

pip install git+https://github.com/jgrss/spfeas.git
# pip install ~/Downloads/SpFeas-0.0.1.tar.gz

echo

if which spfeas >/dev/null; then
  echo 'The installation has finished!'
else
  echo 'SpFeas failed to install.'
fi
