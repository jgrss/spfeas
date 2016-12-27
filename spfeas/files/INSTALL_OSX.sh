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

# Create the .profile
if [ ! -f ~/.profile ]; then
  touch ~/.profile
fi

#export PATH=/usr/local/bin:/usr/local/sbin:$PATH

# Add the NumPy core directory to .profile
if [ -z ${CFLAGS} ]; then 
  echo 'export CFLAGS=-I/usr/local/lib/python2.7/site-packages/numpy/core/include/' >>~/.profile
  source ~/.profile
else
  echo "CFLAGS is already set to '$CFLAGS'"
fi

# gcc 6
if [ -z ${CC} ]; then
  echo 'export CC=gcc-6' >>~/.profile
  source ~/.profile
else
  echo "CC is already set to '$CC'"
fi

# Homebrew
brew tap osgeo/osgeo4mac
brew tap homebrew/science
brew tap homebrew/versions
#brew install wget
brew install gcc hdf4 hdf5 spatialindex

# Python from Homebrew
#if which python >/dev/null; then
#  echo 'Python is already installed'
brew install python
brew linkapps python

# Python pip
if which pip >/dev/null; then
  echo 'pip is already installed'
  sudo -H pip install --upgrade pip
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
sudo -H pip install beautifulsoup4 Bottleneck colorama cython joblib matplotlib numexpr numpy opencv-python pandas psutil PySAL PyYAML retrying Rtree scikit-image scikit-learn scipy six tables xmltodict GDAL

# Statsmodels (0.8 is not available from pip)
git clone https://github.com/statsmodels/statsmodels.git
cd statsmodels/
python setup.py build
python setup.py install
cd ..
rm -rf statsmodels/

# GDAL from Homebrew
if which gdalinfo >/dev/null; then
  echo 'GDAL is already installed'
else
  brew install gdal2 --with-hdf4 --with-hdf5
  # echo /usr/local/opt/gdal2/lib/python2.7/site-packages >> /usr/local/lib/python2.7/site-packages/gdal2.pth
  brew link --force gdal2

  if [ -z ${GDAL_DRIVER_PATH} ]; then
    echo 'export GDAL_DRIVER_PATH=/usr/local/lib/gdalplugins' >>~/.profile
    source ~/.profile
  else
    echo "GDAL_DRIVER_PATH is already set to '$GDAL_DRIVER_PATH'"
  fi

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
