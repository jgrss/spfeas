SpFeas
-----

**SpFeas** is a Python library for

See AUTHORS.txt for contributors and HISTORY.txt or the [MapPy wiki](https://github.com/jgrss/mappy/wiki) for the log.

SpFeas has only been tested on Python 2.7. 

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

Usage examples
-----

Create spatial variables:

> sp_spfeas 

Sample land cover data:

> sp_sample_raster -s /land_cover_samples.shp -i /value_image.tif -o /output_directory


Image classification in Python:

>>> #########################################
>>> # With Scikit-learn or OpenCV classifiers
>>> #########################################
>>>
>>> from spfeas.classifiers import classification
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
>>> from spfeas.classifiers import classification_r
>>>
>>> cl = classification_r()
>>>
>>> # Load training samples
>>> cl.split_samples('/samples.txt')
>>>
>>> # Train a C5 classification model
>>> cl.construct_r_model(classifier_info={'classifier': 'C5', 'trials': 10})

Development
-----------
For questions or bugs, contact Jordan Graesser (jordan.graesser@bu.edu).


