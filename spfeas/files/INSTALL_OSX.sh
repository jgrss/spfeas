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

# Create the .profile if it doesn't exist
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

# gcc
if [ -z ${CC} ]; then
  echo 'export CC=gcc' >>~/.profile
  source ~/.profile
else
  echo "CC is already set to '$CC'"
fi

# Homebrew
brew tap osgeo/osgeo4mac
brew tap homebrew/science
brew tap homebrew/versions
brew install gcc hdf5 spatialindex

# Python from Homebrew
if which python >/dev/null; then
  echo 'Python is already installed'
else
  brew install python
  brew linkapps python
fi

# GDAL from Homebrew
if which gdalinfo --version >/dev/null; then
  echo 'GDAL binaries are already installed'
else

  # Upgrade with Homebrew and Sierra breaks privileges.
  mkdir /usr/local/lib/gdalplugins
  sudo chown -R $(whoami) /usr/local/lib/gdalplugins

  mkdir /usr/local/lib/gdalplugins/2.2
  sudo chown -R $(whoami) /usr/local/lib/gdalplugins/2.2

  brew install gdal2 --with-complete --with-unsupported
  # echo /usr/local/opt/gdal2/lib/python2.7/site-packages >> /usr/local/lib/python2.7/site-packages/gdal2.pth
  brew link --force gdal2

  if [ -z ${GDAL_DRIVER_PATH} ]; then
    echo 'export GDAL_DRIVER_PATH=/usr/local/lib/gdalplugins' >>~/.profile
    source ~/.profile
  else
    echo "GDAL_DRIVER_PATH is already set to '$GDAL_DRIVER_PATH'"
  fi

fi

# Python pip
if which pip >/dev/null; then
  echo 'pip is already installed'
else
  sudo -H easy_install pip
fi

LINE_BREAK1='========================================='

echo
echo $LINE_BREAK1
echo 'If prompted, enter your computer password'
echo $LINE_BREAK1
echo

# Upgrade Python setuptools
sudo -H pip install --upgrade pip
sudo -H pip install --upgrade setuptools

# Python libraries
sudo -H pip install --upgrade --no-cache-dir cython
sudo -H pip install --upgrade --no-cache-dir beautifulsoup4 colorama joblib psutil retrying six xmltodict PyYAML
sudo -H pip install --upgrade --no-cache-dir Bottleneck matplotlib numexpr PySAL Rtree tables
sudo -H pip install --upgrade --no-cache-dir numpy GDAL
sudo -H pip install --upgrade --no-cache-dir scikit-image scikit-learn
sudo -H pip install --upgrade --no-cache-dir opencv-python pandas statsmodels
sudo -H pip install --upgrade --no-cache-dir scipy

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

cd ~/Documents/
git clone https://github.com/jgrss/mpglue.git
cd mpglue/
python setup.py build
python setup.py install
cd ~/Documents/
rm -rf mpglue/

# SpFeas
if which spfeas >/dev/null; then
  pip uninstall spfeas
fi

echo
echo $LINE_BREAK
echo 'If prompted, enter your GitHub username and password'
echo $LINE_BREAK
echo

cd ~/Documents/
git clone https://github.com/jgrss/spfeas.git
cd spfeas/
python setup.py build
python setup.py install
cd ~/Documents/
rm -rf spfeas/

echo

if which spfeas >/dev/null; then
  echo 'The installation has finished!'
else
  echo 'SpFeas failed to install.'
fi
