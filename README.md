SpFeas
-----

**SpFeas** is a Python library for processing spatial (contextual) image features and image classification.

SpFeas has only been tested on Python 2.7. 

Version 0.2.0b
-----

Refer to the [Wiki](https://github.com/jgrss/spfeas/wiki/SpFeas-updates) for changes coming in `0.2.0`. 

#### Naming conventions

##### YAML status

```text
<OUT_DIRECTORY>/<FILENAME>__BD#_BK#_SC#_TR%.yaml

Example:
out_dir/image_name__BD1_BK4_SC4-8_TRmean-hog.yaml
```

##### Tiled files

```text
<OUT_DIRECTORY>/<FILENAME>__BD#_BK#_SC#_TR%/<FILENAME>__BD#_BK#_SC#_ST1-###_TL######.tif

Example:
out_dir/image_name__BD1_BK4_SC4-8_TRmean-hog/image_name__BD1_BK4_SC4-8_ST1-012_TL000001.tif
out_dir/image_name__BD1_BK4_SC4-8_TRmean-hog/image_name__BD1_BK4_SC4-8_ST1-012_TL000002.tif
out_dir/image_name__BD1_BK4_SC4-8_TRmean-hog/image_name__BD1_BK4_SC4-8_ST1-012_TL000003.tif
```

```text
BD = input band position
BK = input block size
SC = input scales
TR = input triggers
ST = statistics
TL = tile position
```

Installation
------------

#### Installing or upgrading from source

##### Upgrade [`mpglue`](https://github.com/jgrss/mpglue)

```commandline
pip uninstall spfeas
git clone https://github.com/jgrss/spfeas.git
cd spfeas/
python setup.py build
python setup.py install
```

##### Dependencies:

```commandline
pip install beautifulsoup4 Bottleneck colorama cython joblib matplotlib numexpr numpy opencv-python pandas psutil PySAL PyYAML retrying Rtree scikit-image scikit-learn scipy six tables xmltodict GDAL
```

#### Installing with the bash or CMD installers

1) Open INSTALLATION.ipynb under [**/notebooks**](https://github.com/jgrss/spfeas/tree/master/spfeas/notebooks).

2) Follow the instructions to install SpFeas for your operating system.

3) On OSX, the following line should print **/usr/local/bin/spfeas**:

```bash
which spfeas
```

4) On Windows, the following line should print **C:\<Python path>\Scripts\spfeas**:

```bash
where spfeas
```

5) To uninstall SpFeas, type the following line in the terminal:

```commandline
pip uninstall spfeas
```

Usage examples
-----

#### Print help:

```commandline
spfeas -h
```

#### Print examples:

```commandline
spfeas -e
```

#### Full usage:

##### Traditional single pixel moving window

```commandline
spfeas -i /input_image.tif -o /output_directory -tr mean hog --block 1 --scales 5 7 --sect-size 1000 --n-jobs -1 --overviews
```

##### Pixel block moving window

```commandline
spfeas -i /input_image.tif -o /output_directory -tr mean hog --block 4 --scales 4 8 --sect-size 1000 --n-jobs -1 --overviews
```

#### Detailed examples

Please refer to [**/notebooks/examples.ipynb**](https://github.com/jgrss/spfeas/tree/master/spfeas/notebooks/examples.ipynb).

#### Spatial features

##### Processing
> Graesser, Jordan, Cheriyadat, Anil, Vatsavai, Ranga Raju, Chandola, Varun, Long, Jordan, and Bright, Eddie (2012) Image based characterization of formal and informal neighborhoods in an urban landscape. _IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing_, 5(4), 1164--1176.

##### Differential Morphological Profiles (dmp)
`mean` `variance`
> Pesaresi, Martino and Benediktsson, Jon Atli (2001) A New Approach for the Morphological Segmentation of High-Resolution Satellite Imagery. _IEEE Transactions on Geoscience and Remote Sensing_, 39(2).

> Pesaresi, Martino et al. (2013) A Global Human Settlement Layer From Optical HR/VHR RS Data: Concept and First Results. _IEEE Transactions on Geoscience and Remote Sensing_, 6(5).

##### Two-band Enhanced Vegetation Index (evi2)
`mean` `variance`
> Jiang, Zhangyan, Huete, Alfredo R, Didan, Kamel, and Miura, Tomoaki (2008) Development of a two-band enhanced vegetation index without a blue band. _Remote Sensing of Environment_, 112, 3833--3845.

##### Fourier Transform (fourier)
`mean` `variance`
> Measure of local energy (power spectrum) 

##### Gabor filter (gabor)
`mean` `variance` x `n filters`
> Image convolution with a series of 2d Gabor filters

##### Edge gradient magnitude (grad)
`mean` `variance`
> Edge gradient magnitude
  
##### Histogram of Oriented Gradients (hog)
`max` `mean` `variance` `skew` `kurtosis`
> Dalal, N and Triggs, B (2005) Histograms of Oriented Gradients for Human Detection. _IEEE Computer Society Conference on Computer Vision and Pattern Recognition_, 2005, San Diego, CA, USA.

##### Lacunarity (lac)
`lacunarity`
> Myint, Soe W, Mesev, Victor, and Lam, Nina (2006) Urban Textural Analysis from Remote Sensor Data: Lacunarity Measurements Based on the Differential Box Counting Method. _Geographical Analysis_, ISSN 0016-7363.

##### Local Binary Patterns (lbpm)
`max` `mean` `variance` `skew` `kurtosis`
> Ojala, Timo, Pietikainen, Matti, and Maenpaa, Topi (2002) Multiresolution gray-scale and rotation invariant texture classification with local binary patterns. _IEEE Transactions on Pattern Analysis and Machine Intelligence_, 24(7), 971--987.

##### Line Support Regions (lsr)
`line length` `line mean` `line contrast`
> Ünsalan, Cem and Boyer, KL (2004) Classifying Land Development in High-Resolution Satellite Imagery Using Hybrid Structural–Multispectral Features. _IEEE Transactions on Geoscience and Remote Sensing_, 42(12). 

> Ünsalan, Cem (2006) Gradient-Magnitude-Based Support Regions in Structural Land Use Classification. _IEEE Geoscience and Remote Sensing Letters_, 3(4).

##### Inverse distance weighted mean and variance (mean)
`mean` `variance`
> Pixels near the center of the local scale/window are given inversely higher weights to the farthest pixel in the window.

##### Normalized Difference Vegetation Index (ndvi)
`mean` `variance`
> Tucker, CJ (1979) Red and photographic red linear combinations for monitoring vegetation. _Remote Sensing of Environment_, 8, 127--150.

##### Built-up presence index (pantex)
`min contrast`
> Pesaresi, Martino, Gerhardinger, Andrea, and Kayitakire, François (2008) A Robust Built-Up Area Presence Index by
Anisotropic Rotation-Invariant Textural Measure. _IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing_, 1(3).

##### Oriented FAST and Rotated BRIEF (orb)
`max` `mean` `variance` `skew` `kurtosis`
> Rublee, Ethan, Rabaud, Vincent, Konolige, Kurt, and Bradski, Gary R (2011) ORB: An efficient alternative to SIFT or SURF. _ICCV 2011_, 2564-2571.

##### Image saliency (saliency)
`mean` `variance`
> Perazzi, Federico, Krahenb, Philipp, Pritch, Yael, and Hornung, Alexander (2012) Contrast Based Filtering for Salient Region Detection. _IEEE CVPR_, Providence, Rhode Island, USA, 16-21.

> Cheng, Ming-Ming, Mitra, Niloy J, Huang, Xiaolei, Torr, Philip HS, and Hu, Shi-Min (2015) Global Contrast based Salient Region detection. _IEEE Transactions on Pattern Analysis and Machine Intelligence_, 37(3), 569--582.

##### Structural Feature Sets (sfs)
`max line length` `min line length` `mean` `w-mean` `standard deviation` ` max ratio of orthogonal angles`
> Huang, Xin, Zhang, Liangpei, and Li, Pingxiang (2007) Classification and Extraction of Spatial Features in Urban Areas Using High-Resolution Multispectral Imagery. _IEEE Geoscience and Remote Sensing Letters_, 4(2).

#### 'To Read' list
> Hoberg, Thorsten, Rottensteiner, Franz, Feitosa, Raul Queiroz, and Heipke, Christian (2015) Conditional Random Fields for multitemporal and multiscale classification of optical satellite imagery. _IEEE Transactions on Geoscience and Remote Sensing_, 53(2). 

> Kenduiywo, Benson Kipkemboi, Bargiel, Damian, and Soergel, Uwe (2017) Higher order Dynamic Conditional Random Fields ensemble for crop type classification in radar images. _IEEE Transactions on Geoscience and Remote Sensing_, 55(8).

> Volpi, Michele and Ferrari, Vittorio (2015) Semantic segmentation of urban scenes by learning local class interactions. _Computer Vision Foundation_.

> Benedeka, Csaba, Shadaydeha, Maha, Katob, Zoltan, Sziranyi, Tamas, and Zerubia, Josiane (2015) Multilayer Markov Random Field models for change detection in optical remote sensing images. _ISPRS Journal of Photogrammetry and Remote Sensing_.

> Wehmann, Adam and Liu, Desheng (2015) A spatial–temporal contextual Markovian kernel method for multi-temporal land cover mapping. _ISPRS Journal of Photogrammetry and Remote Sensing_, 107, 77--89.

Development
-----------
For questions or bugs, please [**submit an issue**](https://github.com/jgrss/spfeas/issues).
