from .spfeas import spatial_features
from .test_spfeas import test_features

from .data import test_image, \
    training_01_4m, training_02_4m, training_03_4m, training_04_4m, training_05_4m, \
    training_01_2m, training_02_2m, training_03_2m, training_04_2m, training_05_2m, \
    features_01__BD1_BK4_SC8, \
    features_02__BD1_BK2_SC8_16

from .version import __version__


__all__ = ['spatial_features',
           'test_features',
           'test_image',
           'training_01_4m',
           'training_02_4m',
           'training_03_4m',
           'training_04_4m',
           'training_05_4m',
           'training_01_2m',
           'training_02_2m',
           'training_03_2m',
           'training_04_2m',
           'training_05_2m',
           'features_01__BD1_BK4_SC8',
           'features_02__BD1_BK2_SC8_16',
           '__version__']
