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

2) Follow the instructions to download SpFeas.

Usage examples
-----

Create spatial variables:

> sp_spfeas -i /your_image.tif -o /your_output_directory  

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


