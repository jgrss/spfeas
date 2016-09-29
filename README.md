MapPy
-----

**MapPy** is a Python library for image processing and remote sensing.

See AUTHORS.txt for contributors and HISTORY.txt or the [MapPy wiki](https://github.com/jgrss/mappy/wiki) for the log.

MapPy provides an interface for image processing through
NumPy, GDAL, OpenCV, SciPy, and others. It is intended as a 
user friendly interface to file handling (GDAL), computer
vision (OpenCV), image processing (NumPy), and machine learning
(Scikit-learn). 

MapPy has only been tested on Python 2.7. 

Usage examples
-----

Load the MapPy object:

    >>> import mappy as mp
    >>> i_info = mp.rinfo('image')

or load sub-classes:

    >>> from mappy.classifiers import classification
    >>> cl = classification()

Call the same functions through Python:

    >>> from mappy.features import veg_indices
    >>> 
    >>> veg_indices('/some_image.tif', '/some_image_indice.tif', 'NDVI', 'Landsat')

or via command line:

    > veg_indices.py -i /some_image.tif -o /some_image_indice.tif --index ndvi --sensor Landsat

Image handling:

    >>> import mappy as mp
    >>>
    >>> # Load an image and get information.
    >>> i_info = mp.rinfo('/your/image.tif')
    >>> print(i_info.bands)
    >>> print(i_info.shape)
    >>>
    >>> # Open an image as an array.
    >>> my_array = i_info.mparray()
    >>>
    >>> # Open specific bands, starting indexes, and row/column dimensions.
    >>> my_array = i_info.mparray(bands2open=[2, 3, 4], i=1000, j=2000, rows=500, cols=500)
    >>> my_array[0]     # 1st index = band 2
    >>>
    >>> # Open all bands and index by map coordinates.
    >>> my_array = i_info.mparray(bands2open=-1, y=1200000, x=4230000, rows=500, cols=500)
    >>>
    >>> # Open image bands as arrays with dictionary mappings.
    >>> my_band_dict = i_info.mparray(bands2open={'red': 2, 'green': 3, 'nir': 4})
    >>> my_band_dict['red']
    >>>
    >>> # Compute the NDVI.
    >>> ndvi = i_info.mparray(compute_index='ndvi', sensor='Landsat')
    >>>
    >>> # Writing to file
    >>>
    >>> # Copy an image info object and modify it.
    >>> o_info = i_info.copy()
    >>> o_info.update_info(bands=3, storage='float32')
    >>>
    >>> # Create the raster object
    >>> out_raster = mp.create_raster('/output_image.tif', o_info)
    >>>
    >>> # Write an array block to band 1
    >>> array2write = <some 2d array data>
    >>> out_raster.write_array(array2write, i=0, j=0, band=1)
    >>> out_raster.close()

Land cover sampling:

    > sample_raster.py -s /land_cover_samples.shp -i /value_image.tif -o /output_directory

Image classification:

    >>> #########################################
    >>> # With Scikit-learn or OpenCV classifiers
    >>> #########################################
    >>>
    >>> from mappy.classifiers import classification
    >>>
    >>> cl = classification()
    >>>
    >>> # Check available models
    >>> print(cl.model_options())
    >>>
    >>> # Load training samples
    >>> cl.split_samples('/samples.txt')
    >>>
    >>> # Train a Random Forest classification model
    >>> cl.construct_model(classifier_info={'classifier': 'RF', 'n_estimators': 500},
    >>>                    perc_samp_each=.7)
    >>>
    >>> # Test model accuracy on withheld samples
    >>> cl.test_accuracy()
    >>> print(cl.emat.accuracy)
    >>> print(cl.emat.e_matrix)
    >>>
    >>> # Make predictions on an image
    >>> cl.predict('/image_variables.tif', '/image_labels.tif')
    >>>
    >>> ####################
    >>> # With R classifiers
    >>> ####################
    >>>
    >>> from mappy.classifiers import classification_r
    >>>
    >>> cl = classification_r()
    >>>
    >>> # Load training samples
    >>> cl.split_samples('/samples.txt')
    >>>
    >>> # Train a C5 classification model
    >>> cl.construct_r_model(classifier_info={'classifier': 'C5', 'trials': 10})

Order, download, and composite Landsat imagery:

    > sift_landsat_metadata.py -o /download_links.txt --path_rows "{225: [77, 79, 81]}" --start 2005/06/03 --end 2007/01/20
    > download_landsat.py -t /download_links.txt -o /Landsat/downloads -u username
    > composite_landsat.py -i /LE72250772000003-SC20151019142328.tar.gz -od /comp_dir --fmask

Raster calculator:

    >>> from mappy.utilities import raster_calc
    >>>
    >>> raster_calc('/output.tif', equation='A * B', A='/imageA.tif', B='/imageB.tif')

Zonal statistics:

    # Get land cover class totals
    > zonal_stats.py -i /land_cover_image.tif -z /vector_zones.shp -o /output_table.csv -c 1 -t 

If installed from source (see below), all tools are available from the command line and begin with **mp_**.

    > mp_download_landsat -h

See ``/notebooks`` for more detailed examples.

Installation
------------
#### Dependencies
- [GDAL](http://www.gdal.org)
- [OpenCV](http://opencv.org)
- [HDF 4 & 5](https://www.hdfgroup.org)
- [libspatialindex](https://libspatialindex.github.io)
- Python third-party libraries (see /notebooks/01_installation.pynb)

#### There are multiple installation options

**Install stable release with pip (recommended)**

1) Update setuptools:

    > pip install -U setuptools

2) [Acquire the latest MapPy tarball](https://github.com/jgrss/mappy/releases)

3) To install:

    > pip install MapPy-<version>.tar.gz
    > e.g., pip install MapPy-0.4.9.tar.gz

4) To update:

    > pip install -U MapPy-<new version>.tar.gz

5) To uninstall:

    > pip uninstall mappy

**PIP install from GitHub**

    > pip install https://github.com/jgrss/mappy/archive/mappy-<version>.tar.gz

**Use the latest MapPy on GitHub**

1) Navigate to location where you want to save MapPy (change */scripts/Python* as desired):

    > cd /scripts/Python

2) Clone the GitHub MapPy repository:

    > git clone https://github.com/jgrss/mappy.git

3) Add /scripts/Python/mappy to the PYTHONPATH

4) To update:

    > cd /scripts/Python/mappy
    > git pull

**Dependencies**

    See ``/notebooks/01_installation.ipynb`` for dependencies. Or, install with pip:

    > pip install -r ../mappy/requirements.txt

Modules & highlights
--------------------
- ``raster_tools``
  - Image handling
- ``vector_tools``
  - Vector handling
- ``/calibrate``
  - Radiometric calibration (support for Landsat, ASTER, CBERS, WorldView2)
  - Image to image normalization, Landsat cross-track adjustments (using MODIS)
- ``/classifiers``
  - Supervised classification (Random Forests, C5, SVM, QDA)
  - Thematic map post-classification processes 
- ``/features``
  - Band transformations, Tasseled Cap, PCA, contextual image statistics
- ``/sample``
  - Land cover sampling, map accuracy, point sampling, object accuracy
- ``/tables``
  - Shapefile table joins
- ``/utilities``
  - Compositing, geo-referencing, image masking, raster calculator, zonal statistics
- ``/utilities/landsat``
  - Compositing, scene download, scene search, tarball extraction
- ``/utilities/modis``
  - Compositing, scene download

Development
-----------
For questions or bugs, contact Jordan Graesser (jordan.graesser@mail.mcgill.ca).


