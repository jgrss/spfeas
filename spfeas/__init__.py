from .spfeas import spatial_features
from .test_spfeas import test_features

from .data import test_image, training_1m_01, training_1m_02, training_4m_01, training_4m_02, features_4m_01

from .version import __version__


__all__ = ['spatial_features',
           'test_features',
           'test_image',
           'training_1m_01',
           'training_1m_02',
           'training_4m_01',
           'training_4m_02',
           'features_4m_01',
           '__version__']
