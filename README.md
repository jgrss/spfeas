SpFeas
-----

**SpFeas** is a Python library for processing spatial (contextual) image features and image classification.

SpFeas has only been tested on Python 2.7. 

Installation
------------
#### Dependencies
- [GDAL](http://www.gdal.org) binaries
- Python third-party libraries (see /notebooks/01_installation.pynb)

##### Optional
- [libspatialindex](https://libspatialindex.github.io) and RTree

#### Installation instructions

1) Either download the INSTALLATION.pdf under **/files** or open INSTALLATION.ipynb under **/notebooks**.

2) Follow the instructions to install SpFeas.

Usage examples
-----

Create spatial variables:

    > sp_spfeas -i /your_image.tif -o /your_output_directory  

Sample land cover data:

    > sp_sample_raster -s /land_cover_samples.shp -i /value_image.tif -o /output_directory

Image classification:

    > sp_classify -i /value_image.tif -o /output_image.tif -s /land_cover_samples.txt


Development
-----------
For questions or bugs, contact Jordan Graesser (jordan.graesser@bu.edu).


