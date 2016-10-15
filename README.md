SpFeas
-----

**SpFeas** is a Python library for processing spatial (contextual) image features and image classification.

SpFeas has only been tested on Python 2.7. 

Installation
------------
#### Installation instructions

1) Open INSTALLATION.ipynb under [**/notebooks**.](https://github.com/jgrss/spfeas/tree/master/notebooks)

2) Follow the instructions to install SpFeas for your system.

3) On OSX, the following line should print /usr/local/bin/spfeas:

```
which spfeas
```

4) On Windows, the following line should print C:\??\spfeas:

```
where spfeas
```

5) To uninstall SpFeas, type the following line in the terminal

```
pip uninstall spfeas
```

Usage examples
-----

Print help for spatial variable calculations:

```
spfeas -h
```

Print examples of spatial variable calculations:

```
spfeas -e
```

Create spatial variables:

```
spfeas -i /your_image.tif -o /your_output_directory
```

Sample land cover data:

```
sp_sample_raster -s /land_cover_samples.shp -i /value_image.tif -o /output_directory
```

Image classification:

```
sp_classify -i /value_image.tif -o /output_image.tif -s /land_cover_samples.txt
```

Test classification accuracy:

```
sp_sample_raster -s /land_cover_samples.shp -i /value_image.tif --accuracy
```


Development
-----------
For questions or bugs, contact Jordan Graesser (graesser@bu.edu).


