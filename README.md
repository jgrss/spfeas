SpFeas
-----

**SpFeas** is a Python library for processing spatial (contextual) image features and image classification.

SpFeas has only been tested on Python 2.7. 

Version 0.2.0
-----

Refer to the [Wiki](https://github.com/jgrss/spfeas/wiki/SpFeas-updates) for changes in `0.2.0`. 

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

Upgrade [`mpglue`](https://github.com/jgrss/mpglue)

```text
> pip uninstall spfeas
> git clone https://github.com/jgrss/spfeas.git
> cd spfeas/
> python setup.py build && python setup.py install
```

#### Installing with bash or CMD installers

1) Open INSTALLATION.ipynb under [**/notebooks**](https://github.com/jgrss/spfeas/tree/master/spfeas/notebooks).

2) Follow the instructions to install SpFeas for your operating system.

3) On OSX, the following line should print **/usr/local/bin/spfeas**:

```text
which spfeas
```

4) On Windows, the following line should print **C:\<Python path>\Scripts\spfeas**:

```text
where spfeas
```

5) To uninstall SpFeas, type the following line in the terminal

```text
pip uninstall spfeas
```

Usage examples
-----

Print help:

```text
spfeas -h
```

Print examples:

```text
spfeas -e
```

Full usage:
```text
spfeas -i /input_image.tif -o /output_directory -tr mean hog --block 4 --scales 4 8 --sect-size 1000 --chunk-size 100 --n-jobs-section -1 --n-jobs-chunk -1 --overviews
```

#### Detailed examples

Please refer to [**/notebooks/examples.ipynb**](https://github.com/jgrss/spfeas/tree/master/spfeas/notebooks/examples.ipynb).

Development
-----------
For questions or bugs, contact Jordan Graesser (graesser@bu.edu).
