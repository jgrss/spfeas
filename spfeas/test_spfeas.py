import os
import shutil

from .errors import logger
from .spfeas import spatial_features
from .paths import get_path

import mpglue as gl

import numpy as np


SPFEAS_PATH = get_path()


def test_features():

    """
    Test SpFeas features
    """

    data_dir = os.path.join(SPFEAS_PATH, 'data')
    good_features_dir = os.path.join(data_dir, '_features')
    test_features_dir = os.path.join(data_dir, 'features')

    good_features_mean = os.path.join(good_features_dir, 'test_image__BD1_BK4_SC8_TRmean.vrt')
    test_features_mean = os.path.join(test_features_dir, 'test_image__BD1_BK4_SC8_TRmean.vrt')

    assert os.path.isfile(good_features_mean)

    if os.path.isdir(test_features_dir):
        os.remove(test_features_dir)

    image = os.path.join(data_dir, 'test_image.tif')

    spatial_features(image,
                     test_features_dir,
                     band_positions=[1],
                     block=4,
                     scales=[8],
                     triggers=['mean'])

    with gl.ropen(good_features_mean) as good_info:

        good_bands = good_info.bands

        good_band1 = good_info.read(bands2open=1,
                                    d_type='float32')

        good_band2 = good_info.read(bands2open=2,
                                    d_type='float32')

    del good_info

    with gl.ropen(test_features_mean) as test_info:

        test_bands = test_info.bands

        test_band1 = test_info.read(bands2open=1,
                                    d_type='float32')

        test_band2 = test_info.read(bands2open=2,
                                    d_type='float32')

    del test_info

    if test_bands != good_bands:
        logger.error('  The output band number did not match the test.')

    if not np.allclose(test_band1, good_band1):
        logger.error('  Band 1 did not match the test.')

    if not np.allclose(test_band2, good_band2):
        logger.error('  Band 2 did not match the test.')

    logger.info('')
    logger.info('  SpFeas tests were OK.')

    shutil.rmtree(test_features_dir)
