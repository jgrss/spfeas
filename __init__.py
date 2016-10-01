from .raster_tools import rinfo, mparray, create_raster, write2raster, batch_manage_overviews, pixel_stats
from .classification import classification, classification_r
from .veg_indices import veg_indices, VegIndicesEquations, VegIndices
from .error_matrix import error_matrix, object_accuracy

__all__ = ['__version__', 'rinfo', 'mparray', 'create_raster', 'write2raster', 'batch_manage_overviews', 'pixel_stats',
           'classification', 'classification_r', 'veg_indices', 'VegIndicesEquations', 'VegIndices',
           'lsr', 'spfunctions', 'spprocess', 'spreshape', 'spsplit', 'sputilities', 'error_matrix', 'object_accuracy']

__version__ = '0.0.1'
