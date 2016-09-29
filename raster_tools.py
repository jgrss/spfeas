#!/usr/bin/env python

"""
@author: Jordan Graesser
Date Created: 9/29/2016
"""

import os
import sys
import copy
import fnmatch
import time
import argparse
import inspect
import atexit
from joblib import Parallel, delayed

from .helpers.utilities import random_float, overwrite_file
from .helpers.other.progress_iter import _iteration_parameters
from .vector_tools import get_xy_offsets
from .helpers.utilities import check_and_create_dir
from .helpers.errors import LenError, RinfoError
from vector_tools import vinfo, intersects_boundary

# GDAL
try:
    from osgeo import gdal, osr, gdal_array
    from osgeo.gdalconst import *
except ImportError:
    raise ImportError('GDAL must be installed')

# NumPy
try:
    import numpy as np
except ImportError:
    raise ImportError('NumPy must be installed')

# Bottleneck
try:
    import bottleneck as bn
except ImportError:
    raise ImportError('Bottleneck must be installed')

# OpenCV
try:
    import cv2
except ImportError:
    raise ImportError('OpenCV must be installed')

# Matplotlib
try:
    import matplotlib as mpl
    mpl.use('TkAgg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib import ticker, colors, colorbar
    import matplotlib.cm as cm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except ImportError:
    raise ImportError('Matplotlib must be installed')

# Scikit-learn
try:
    from sklearn import decomposition
    from sklearn.preprocessing import StandardScaler
except ImportError:
    raise ImportError('Scikit-learn must be installed')

# Scikit-image
try:
    from skimage import exposure
except ImportError:
    raise ImportError('Scikit-image must be installed')

# Pandas
try:
    import pandas as pd
except ImportError:
    raise ImportError('Pandas must be installed')

# SciPy
try:
    from scipy.stats import mode as sci_mode
    from scipy.ndimage.measurements import label as lab_img
except ImportError:
    raise ImportError('SciPy must be installed')

# BeautifulSoup4
try:
    from bs4 import BeautifulSoup
except ImportError:
    raise ImportError('BeautifulSoup4 must be installed')

# xmltodict
try:
    import xmltodict
except ImportError:
    raise ImportError('xmltodict must be installed')


DRIVER_DICT = {'.tif': 'GTiff',
               '.img': 'HFA',
               '.hdf4': 'HDF4',
               '.hdf5': 'HDF5',
               '.vrt': 'VRT',
               '.dat': 'ENVI',
               '.bin': 'ENVI',
               '.kea': 'KEA',
               '.sid': 'MrSID',
               '.jp2': 'JPEG2000',
               '.mem': 'MEM'}

FORMAT_DICT = dict((v, k) for k, v in DRIVER_DICT.iteritems())

STORAGE_DICT = {'byte': 'uint8',
                'int16': 'int16',
                'uint16': 'uint16',
                'int32': 'int32',
                'uint32': 'uint32',
                'int64': 'int64',
                'uint64': 'uint64',
                'float32': 'float32',
                'float64': 'float64'}

STORAGE_DICT_GDAL = {'unknown': gdal.GDT_Unknown,
                     'byte': gdal.GDT_Byte,
                     'uint16': gdal.GDT_UInt16,
                     'int16': gdal.GDT_Int16,
                     'uint32': gdal.GDT_UInt32,
                     'int32': gdal.GDT_Int32,
                     'float32': gdal.GDT_Float32,
                     'float64': gdal.GDT_Float64,
                     'cint16': gdal.GDT_CInt16,
                     'cint32': gdal.GDT_CInt32,
                     'cfloat32': gdal.GDT_CFloat32,
                     'cfloat64': gdal.GDT_CFloat64}

STORAGE_DICT_NUMPY = {'byte': np.uint8,
                      'int16': np.int16,
                      'uint16': np.uint16,
                      'int32': np.int32,
                      'uint32': np.uint32,
                      'int64': np.int64,
                      'uint64': np.uint64,
                      'float32': np.float32,
                      'float64': np.float64}


def get_sensor_dict():

    return {'landsat_tm': 'tm',
            'lt4': 'tm',
            'lt5': 'tm',
            'tm': 'tm',
            'landsat_etm_slc_off': 'etm',
            'landsat_etm': 'etm',
            'landsat_et': 'etm',
            'landsat_etm_slc_on': 'etm',
            'etm': 'etm',
            'le7': 'etm',
            'lt7': 'etm',
            'landsat_oli_tirs': 'oli_tirs',
            'oli_tirs': 'oli_tirs',
            'oli': 'oli_tirs',
            'lc8': 'oli_tirs',
            'lt8': 'oli_tirs'}


class DataChecks(object):

    """
    A class for spatial and cloud checks
    """

    def contains(self, iinfo):

        """
        Test whether the open image contains another image.

        Args:
            iinfo (object): An image instance of ``rinfo`` to test.
        """

        if (iinfo.left > self.left) and (iinfo.right < self.right) \
                and (iinfo.top < self.top) and (iinfo.bottom > self.bottom):

            return True

        else:
            return False

    def contains_value(self, value):

        """
        Tests whether a value is within the array (the array must be open)

        Args:
            value (int): The value to search for.
        """

        return np.in1d(np.array([value]), self.array)[0]

    def intersects(self, iinfo):

        """
        Test whether the open image intersects another image.

        Args:
            iinfo (object): An image instance of ``rinfo`` to test.
        """

        image_intersects = False

        # At least within the longitude frame.
        if ((iinfo.left > self.left) and (iinfo.left < self.right)) or \
                ((iinfo.right < self.right) and (iinfo.right > self.left)):

            # Also within the latitude frame.
            if ((iinfo.bottom > self.bottom) and (iinfo.bottom < self.top)) or \
                    ((iinfo.top < self.top) and (iinfo.top > self.bottom)):

                image_intersects = True

        return image_intersects

    def within(self, iinfo):

        """
        Test whether the open image falls within another image.

        Args:
            iinfo (object or dict): An image instance of ``rinfo`` to test.
        """

        if isinstance(iinfo, rinfo):
            iinfo = self._info2dict(iinfo)

        if (self.left > iinfo['left']) and (self.right < iinfo['right']) \
                and (self.top < iinfo['top']) and (self.bottom > iinfo['bottom']):

            return True

        else:
            return False

    def outside(self, iinfo):

        """
        Test whether the open image falls outside coordinates

        Args:
            iinfo (object or dict): An image instance of ``rinfo`` to test.
        """

        if isinstance(iinfo, rinfo):
            iinfo = self._info2dict(iinfo)

        if (self.right < iinfo['left']) or (self.left > iinfo['right']) or \
                (self.top < iinfo['bottom']) or (self.bottom > iinfo['top']):

            return True

        else:
            return False

    def _info2dict(self, info_obect):
        return dict(left=info_obect.left, right=info_obect.right, bottom=info_obect.bottom, top=info_obect.top)

    def check_clouds(self, cloud_band=7, clear_value=0, background_value=255):

        """
        Checks cloud information

        Args:
            cloud_band (Optional[int]): The cloud band position. Default is 7.
            clear_value (Optional[int]): The clear pixel value. Default is 0.
            background_value (Optional[int]): The background pixel value. Default is 255.
        """

        cloud_array = self.mparray(bands2open=cloud_band)

        clear_pixels = (cloud_array == clear_value).sum()

        non_background_pixels = (cloud_array != background_value).sum()

        self.clear_percent = (float(clear_pixels) / float(non_background_pixels)) * 100.


class RegisterDriver(object):

    """
    Class handler for driver registration

    Args:
        out_name (str): The file to register.
        in_memory (bool): Whether to create the file in memory.

    Attributes:
        out_name (str)
        driver (object)
        file_format (str)
    """

    def __init__(self, out_name, in_memory):

        gdal.AllRegister()

        if not in_memory:

            self._get_file_format(out_name)

            self.driver = gdal.GetDriverByName(self.file_format)

        else:
            self.driver = gdal.GetDriverByName('MEM')

        self.driver.Register

    def _get_file_format(self, image_name):

        __, f_name = os.path.split(image_name)
        __, file_extension = os.path.splitext(f_name)

        self.file_format = self._get_driver_name(file_extension)

    @staticmethod
    def _get_driver_name(file_extension):

        if file_extension.lower() not in DRIVER_DICT:
            raise TypeError('{} is not an image, or is not a supported raster format.'.format(file_extension))
        else:
            return DRIVER_DICT[file_extension.lower()]


class CreateDriver(RegisterDriver):

    """
    Class handler for driver creation

    Args:
        out_name (str): The output file name.
        out_rows (int): The output number of rows.
        out_cols (int): The output number of columns.
        n_bands (int): The output number of bands.
        storage_type (str): The output storage type.
        in_memory (bool): Whether to create the file in memory.
        overwrite (bool): Whether to overwrite an existing file.
        parameters (str list): A list of GDAL creation parameters.

    Attributes:
        datasource (object)
    """

    def __init__(self, out_name, out_rows, out_cols, n_bands, storage_type, in_memory, overwrite, parameters):

        RegisterDriver.__init__(self, out_name, in_memory)

        if overwrite and not in_memory:

            if os.path.isfile(out_name):
                os.remove(out_name)

        # Create the output driver.
        if in_memory:
            self.datasource = self.driver.Create('', out_cols, out_rows, n_bands, storage_type)
        else:
            self.datasource = self.driver.Create(out_name, out_cols, out_rows, n_bands, storage_type, parameters)


class FileManager(DataChecks, RegisterDriver):

    """
    Class for file handling

    Args:
        raster_object (object)

    Attributes:
        band (object)
        chunk_size (int)

    Methods:
        build_overviews
        get_band
        write_array
        close_band
        close_file
        close_all
        get_chunk_size
        remove_overviews

    Returns:
        None
    """

    def get_image_info(self, open2read, hdf_band, check_corrupted):

        if not os.path.isfile(self.file_name):
            raise IOError('\n{} does not exist.\n'.format(self.file_name))

        self._get_file_format(self.file_name)

        # Open input raster.
        try:

            if open2read:

                self.datasource = gdal.Open(self.file_name, GA_ReadOnly)
                self.image_mode = 'read only'

            else:

                self.datasource = gdal.Open(self.file_name, GA_Update)
                self.image_mode = 'update'

            self.file_open = True

        except:
            print '\nCould not open {}.\n'.format(self.file_name)
            return

        if self.file_name.lower().endswith('.hdf'):

            if self.datasource is None:
                print '\n{} appears to be empty.\n'.format(self.file_name)
                return

            self.hdf_layers = self.datasource.GetSubDatasets()

            if self.open2read:
                self.datasource = gdal.Open(self.datasource.GetSubDatasets()[hdf_band - 1][0], GA_ReadOnly)
            else:
                self.datasource = gdal.Open(self.datasource.GetSubDatasets()[hdf_band - 1][0], GA_Update)

        if self.datasource is None:
            raise OSError('\n{} appears to be empty.\n'.format(self.file_name))

        try:
            self.meta_dict = self.datasource.GetMetadata_Dict()
        except:
            self.meta_dict = 'none'

        try:
            self.storage = gdal.GetDataTypeName(self.datasource.GetRasterBand(1).DataType)
        except:
            self.storage = 'none'

        self.directory, self.filename = os.path.split(self.file_name)

        self.bands = self.datasource.RasterCount

        # Initiate the data checks object.
        # DataChecks.__init__(self)

        # Check if any of the bands are corrupted.
        if check_corrupted:
            self.check_corrupted_bands()

        self.projection = self.datasource.GetProjection()

        self.sp_ref = osr.SpatialReference()
        self.sp_ref.ImportFromWkt(self.projection)

        self.color_interpretation = self.datasource.GetRasterBand(1).GetRasterColorInterpretation()

        try:

            if 'PROJ' in self.projection[:4]:

                try:
                    self.epsg = int(self.sp_ref.GetAttrValue('PROJCS|AUTHORITY', 1))
                except:
                    pass

            elif 'GEOG' in self.projection[:4]:

                try:
                    self.epsg = int(self.sp_ref.GetAttrValue('GEOGCS|AUTHORITY', 1))
                except:
                    if 'WGS' in self.sp_ref.GetAttrValue('GEOGCS') and '84' in self.sp_ref.GetAttrValue('GEOGCS'):
                        self.epsg = 4326  # WGS 1984

        except:
            self.epsg = 'none'

        # Set georeference and projection.
        self.geo_transform = self.datasource.GetGeoTransform()

        # adfGeoTransform[0] :: top left x
        # adfGeoTransform[1] :: w-e pixel resolution
        # adfGeoTransform[2] :: rotation, 0 if image is north up
        # adfGeoTransform[3] :: top left y
        # adfGeoTransform[4] :: rotation, 0 if image is north up
        # adfGeoTransform[5] :: n-s pixel resolution

        self.left = self.geo_transform[0]  # get left extent
        self.top = self.geo_transform[3]  # get top extent
        self.cellY = self.geo_transform[1]  # pixel height
        self.cellX = self.geo_transform[5]  # pixel width

        self.rotation1 = self.geo_transform[2]
        self.rotation2 = self.geo_transform[4]

        self.rows = self.datasource.RasterYSize  # get number of rows
        self.cols = self.datasource.RasterXSize  # get number of columns

        self.shape = dict(bands=self.bands,
                          rows='{:,d}'.format(self.rows),
                          columns='{:,d}'.format(self.cols),
                          pixels='{:,d}'.format(self.bands * self.rows * self.cols),
                          row_units='{:,.2f}'.format(self.rows * self.cellY),
                          col_units='{:,.2f}'.format(self.cols * self.cellY))

        self.right = self.left + (float(self.cols) * float(self.cellY))  # get right extent
        self.bottom = self.top - (float(self.rows) * float(self.cellY))  # get bottom extent

        self.extent = dict(left=self.left, right=self.right, bottom=self.bottom, top=self.top)

        self.name = self.datasource.GetDriver().ShortName

        try:
            self.block_x = self.datasource.GetRasterBand(1).GetBlockSize()[0]
            self.block_y = self.datasource.GetRasterBand(1).GetBlockSize()[1]
        except:
            self.block_x = 'none'
            self.block_y = 'none'

    def build_overviews(self, sampling_method='nearest', levels=[2, 4, 8, 16]):

        """
        Builds image overviews

        Args:
            sampling_method (Optional[str]): The sampling method to use. Default is 'nearest'.
            levels (Optional[int list]): The levels to build. Default is [2, 4, 8, 16].
        """

        if not isinstance(levels, list):
            raise TypeError('\nThe overviews must be a list.\n')

        levels = map(int, levels)

        try:
            self.datasource.BuildOverviews(sampling_method.upper(), overviewlist=levels)
        except:
            raise ValueError('\nFailed to build overviews.\n')

    def get_band(self, band_position):

        """
        Loads a raster band pointer

        Args:
            band_position (int): The band position to load.
        """

        if not isinstance(band_position, int) or band_position < 1:
            raise TypeError('\nThe band position must be an integer > 0.\n')

        try:

            self.band = self.datasource.GetRasterBand(band_position)
            self.band_open = True

        except:
            raise ValueError('\nFailed to load the band.\n')

    def check_corrupted_bands(self):

        self.corrupted_bands = []

        for band in range(1, self.bands+1):

            self.datasource.GetRasterBand(band).Checksum()

            if gdal.GetLastErrorType() != 0:
                print('\nBand {:d} of {} appears to be corrupted.\n'.format(band, self.file_name))

                self.corrupted_bands.append(band)

    def write_array(self, array2write, i=0, j=0, band=None):

        """
        Writes array to the loaded band object (``self.band`` of ``get_band``).

        Args:
            array2write (ndarray): The array to write.
            i (Optional[int]): The starting row position to write to. Default is 0.
            j (Optional[int]): The starting column position to write to. Default is 0.
            band (Optional[int]): The band position to write to. Default is None. If None, an object of
                ``get_band`` must be open.
        """

        if not isinstance(array2write, np.ndarray):
            raise TypeError('\nThe array must be an ndarray.\n')

        if not isinstance(i, int) or (i < 0):
            raise TypeError('\nThe row index must be a positive integer.\n')

        if not isinstance(j, int) or (j < 0):
            raise TypeError('\nThe column index must be a positive integer.\n')

        if isinstance(band, int):
            self.get_band(band_position=band)

        try:
            self.band.WriteArray(array2write, j, i)
        except:

            if (array2write.shape[0] > self.rows) or (array2write.shape[1] > self.cols):
                raise ValueError('\nThe array is larger than the file size.\n')
            else:
                raise ValueError('\nThe band must be set either with get_band() or write_array(band=)\n')

    def close_band(self):

        """
        Closes a band object
        """

        if hasattr(self, 'band'):

            try:
                self.band.SetColorInterpretation(self.color_interpretation)
                self.band.SetRasterColorInterpretation(self.color_interpretation)

                self.band.GetStatistics(0, 1)
                self.band.FlushCache()

                self.band = None
            except:
                self.band = None

            self.band_open = False

    def close_file(self):

        """
        Closes file object
        """

        if hasattr(self, 'datasource'):

            try:
                self.datasource.FlushCache()
                self.datasource = None
            except:
                self.datasource = None

            self.file_open = False

    def close_all(self):

        """
        Closes a band object and a file object
        """

        self.close_band()
        self.close_file()

    def fill(self, fill_value, band=None):

        """
        Fills a band with a specified value

        Args:
            fill_value (int): The value to fill.
            band (Optional[int]): The band to fill. Default is None.
        """

        if isinstance(band, int):
            self.get_band(band_position=band)

        self.band.Fill(fill_value)

    def get_chunk_size(self):

        """
        Gets the band block size
        """

        try:
            self.chunk_size = self.band.GetBlockSize()[0]
        except:
            raise IOError('\nFailed to get the block size.\n')

    def remove_overviews(self):

        """
        Removes image overviews
        """

        if self.image_mode != 'update':
            raise NameError('\nOpen the image in update mode (open2read=False) to remove overviews.\n')
        else:
            self.build_overviews(levels=[])

    def calculate_stats(self, band=1):

        """
        Calculates image statistics and can be used to check for empty images.
        """

        self.get_band(band_position=band)

        image_metadata = self.band.GetMetadata()

        use_exceptions = gdal.GetUseExceptions()
        gdal.UseExceptions()

        try:

            image_min, image_max, image_mu, image_std = self.band.GetStatistics(False, True)

            image_metadata['STATISTICS_MINIMUM'] = repr(image_min)
            image_metadata['STATISTICS_MAXIMUM'] = repr(image_max)
            image_metadata['STATISTICS_MEAN'] = repr(image_mu)
            image_metadata['STATISTICS_STDDEV'] = repr(image_std)

            image_metadata['STATISTICS_SKIPFACTORX'] = '1'
            image_metadata['STATISTICS_SKIPFACTORY'] = '1'

            if not use_exceptions:
                gdal.DontUseExceptions()

            self.band.SetMetadata(image_metadata)

            return True

        except:

            if not use_exceptions:
                gdal.DontUseExceptions()

            return False


class LandsatParser(object):

    """
    A class to parse Landsat metadata
    """

    def __init__(self, metadata, band_order=[]):

        self.bo = copy.copy(band_order)

        if metadata.endswith('MTL.txt'):
            self.parse_mtl(metadata)
        elif metadata.endswith('.xml'):
            self.parse_xml(metadata)
        else:
            raise NameError('Parser type not supported')

    def parse_mtl(self, metadata):

        df = pd.read_table(metadata, header=None, sep='=')
        df.rename(columns={0: 'Variable', 1: 'Value'}, inplace=True)

        df['Variable'] = df['Variable'].str.strip()
        df['Value'] = df['Value'].str.strip()

        self.scene_id = df.loc[df['Variable'] == 'LANDSAT_SCENE_ID', 'Value'].values[0].replace('"', '').strip()

        if not df.loc[df['Variable'] == 'DATE_ACQUIRED', 'Value'].empty:
            self.date = df.loc[df['Variable'] == 'DATE_ACQUIRED', 'Value'].values[0].replace('"', '').strip()
        else:
            self.date = df.loc[df['Variable'] == 'ACQUISITION_DATE', 'Value'].values[0].replace('"', '').strip()

        self.date_ = self.date.split('-')
        self.year = self.date_[0]
        self.month = self.date_[1]
        self.day = self.date_[2]

        self.sensor = df.loc[df['Variable'] == 'SENSOR_ID', 'Value'].values[0].replace('"', '').strip()
        self.series = df.loc[df['Variable'] == 'SPACECRAFT_ID', 'Value'].values[0].replace('"', '').strip()

        self.path = df.loc[df['Variable'] == 'WRS_PATH', 'Value'].astype(int).astype(str).values[0].strip()

        if not df.loc[df['Variable'] == 'WRS_ROW', 'Value'].empty:
            self.row = df.loc[df['Variable'] == 'WRS_ROW', 'Value'].astype(int).astype(str).values[0].strip()
        else:
            self.row = df.loc[df['Variable'] == 'STARTING_ROW', 'Value'].astype(int).astype(str).values[0].strip()

        self.elev = df.loc[df['Variable'] == 'SUN_ELEVATION', 'Value'].astype(float).values[0]
        self.azimuth = df.loc[df['Variable'] == 'SUN_AZIMUTH', 'Value'].astype(float).values[0]
        self.cloudCover = df.loc[df['Variable'] == 'CLOUD_COVER', 'Value'].astype(float).astype(str).values[0].strip()

        try:
            self.imgQuality = df.loc[df['Variable'] == 'IMAGE_QUALITY', 'Value'].astype(int).astype(str).values[0].strip()
        except:

            self.img_quality_oli = df.loc[df['Variable'] ==
                                          'IMAGE_QUALITY_OLI', 'Value'].astype(int).astype(str).values[0].strip()

            self.img_quality_tirs = df.loc[df['Variable'] ==
                                           'IMAGE_QUALITY_TIRS', 'Value'].astype(int).astype(str).values[0].strip()

        self.LMAX_dict = {1: 0., 2: 0., 3: 0., 4: 0., 5: 0., 6: 0., 7: 0., 8: 0., 9: 0.}
        self.LMIN_dict = {1: 0., 2: 0., 3: 0., 4: 0., 5: 0., 6: 0., 7: 0., 8: 0., 9: 0.}
        self.no_coeff = 999

        # Landsat 8 radiance
        self.rad_mult_dict = {1: 0., 2: 0., 3: 0., 4: 0., 5: 0., 6: 0., 7: 0., 8: 0., 9: 0., 10: 0., 11: 0.}
        self.rad_add_dict = {1: 0., 2: 0., 3: 0., 4: 0., 5: 0., 6: 0., 7: 0., 8: 0., 9: 0., 10: 0., 11: 0.}

        # Landsat 8 reflectance
        self.refl_mult_dict = {1: 0., 2: 0., 3: 0., 4: 0., 5: 0., 6: 0., 7: 0., 8: 0., 9: 0.}
        self.refl_add_dict = {1: 0., 2: 0., 3: 0., 4: 0., 5: 0., 6: 0., 7: 0., 8: 0., 9: 0.}

        self.k1 = 0
        self.k2 = 0

        if self.sensor.lower() == 'oli_tirs':

            if not self.bo:
                self.bo = [2, 3, 4, 5, 6, 7]

        else:

            if not self.bo:
                self.bo = [1, 2, 3, 4, 5, 7]

        for bi in self.bo:

            if not df.loc[df['Variable'] == 'RADIANCE_MAXIMUM_BAND_{:d}'.format(bi), 'Value'].empty:

                self.LMAX_dict[bi] = df.loc[df['Variable'] == 'RADIANCE_MAXIMUM_BAND_{:d}'.format(bi),
                                            'Value'].astype(float).values[0]

                self.LMIN_dict[bi] = df.loc[df['Variable'] == 'RADIANCE_MINIMUM_BAND_{:d}'.format(bi),
                                            'Value'].astype(float).values[0]

                self.no_coeff = 1000

            else:

                self.LMAX_dict[bi] = df.loc[df['Variable'] == 'LMAX_BAND{:d}'.format(bi),
                                            'Value'].astype(float).values[0]

                self.LMIN_dict[bi] = df.loc[df['Variable'] == 'LMIN_BAND{:d}'.format(bi),
                                            'Value'].astype(float).values[0]

                self.no_coeff = 1000

            if self.sensor.lower() == 'oli_tirs':

                self.rad_mult_dict[bi] = df.loc[df['Variable'] == 'RADIANCE_MULT_BAND_{:d}'.format(bi),
                                                'Value'].astype(float).values[0]

                self.rad_add_dict[bi] = df.loc[df['Variable'] == 'RADIANCE_ADD_BAND_{:d}'.format(bi),
                                               'Value'].astype(float).values[0]

                self.refl_mult_dict[bi] = df.loc[df['Variable'] == 'REFLECTANCE_MULT_BAND_{:d}'.format(bi),
                                                'Value'].astype(float).values[0]

                self.refl_add_dict[bi] = df.loc[df['Variable'] == 'REFLECTANCE_ADD_BAND_{:d}'.format(bi),
                                               'Value'].astype(float).values[0]

                # TODO: add k1 and k2 values
                # self.k1 =
                # self.k2 =

    def parse_xml(self, metadata):

        with open(metadata) as mo:

            meta = mo.read()

            soup = BeautifulSoup(meta)

            wrs = soup.find('wrs')

            self.scene_id = soup.find('lpgs_metadata_file').text
            si_index = self.scene_id.find('_')
            self.scene_id = self.scene_id[:si_index]

            self.path = wrs['path']

            self.row = wrs['row']

            self.sensor = soup.find('instrument').text

            self.series = soup.find('satellite').text

            self.date = soup.find('acquisition_date').text

            self.date_ = self.date.split('-')
            self.year = self.date_[0]
            self.month = self.date_[1]
            self.day = self.date_[2]

            solar_angles = soup.find('solar_angles')

            self.elev = solar_angles['zenith']

            self.azimuth = solar_angles['azimuth']


class SentinelParser(object):

    """
    A class to parse Sentinel 2 metadata
    """

    def __init__(self, metadata, band_order=[]):

        self.bo = copy.copy(band_order)

        if metadata.endswith('.xml'):
            self.parse_xml(metadata)
        else:
            raise NameError('Parser type not supported')

    def parse_xml(self, metadata):

        with open(metadata) as xml_tree:
            xml_object = xmltodict.parse(xml_tree.read())

        base_xml = xml_object['n1:Level-1C_User_Product']

        general_info = base_xml['n1:General_Info']

        quality_info = base_xml['n1:Quality_Indicators_Info']

        self.cloud_cover = float(quality_info['Cloud_Coverage_Assessment'])

        product_info = general_info['Product_Info']

        self.year, self.month, self.day = product_info['GENERATION_TIME'][:10].split('-')

        self.band_list = product_info['Query_Options']['Band_List']

        self.band_list = [bn for bn in self.band_list['BAND_NAME']]

        granule_list = product_info['Product_Organisation']['Granule_List']

        self.granule_dict = {}

        for granule in granule_list:

            tile = granule['Granules']
            tile_id = tile['@granuleIdentifier']
            image_ids = tile['IMAGE_ID']

            image_format = tile['@imageFormat']

            self.granule_dict[tile_id] = image_ids

        self.image_ext = FORMAT_DICT[image_format]

        # print self.granule_dict

        self.level = product_info['PROCESSING_LEVEL']
        self.product = product_info['PRODUCT_TYPE']

        self.series = product_info['Datatake']['SPACECRAFT_NAME']

        self.no_data = int(general_info['Product_Image_Characteristics']['Special_Values'][0]['SPECIAL_VALUE_INDEX'])
        self.saturated = int(general_info['Product_Image_Characteristics']['Special_Values'][1]['SPECIAL_VALUE_INDEX'])


class rinfo(FileManager, LandsatParser, SentinelParser):

    """
    Gets image information and a file pointer object.

    Args:
        file_name (Optional[str]): Image location, name, and extension. Default is 'none'.
        open2read (Optional[bool]): Whether to open image as 'read only' (True) or writeable (False).
            Default is True.
        metadata (Optional[str]): A metadata file. Default is None.
        sensor (Optional[str]): The satellite sensor to parse with ``metadata``. Default is 'Landsat'. Choices are
            ['Landsat', 'Sentinel2']. This is only used for inplace spectral transformations. It will not
            affect the image otherwise.
        hdf_band (Optional[int])

    Attributes:
        file_name (str)
        datasource (object)
        directory (str)
        filename (str)
        bands (int)
        projection (str)
        geo_transform (list)
        left (float)
        top (float)
        right (float)
        bottom (float)
        cellY (float)
        cellX (float)
        rows (int)
        cols (int)
        shape (str)
        name (str)
        block_x (int)
        block_y (int)

    Returns:
        None

    Examples:
        >>> # typical usage
        >>> import mappy as mp
        >>>
        >>> i_info = mp.rinfo('/some_raster.tif')
        >>> # <rinfo> has its own array instance
        >>> array = i_info.mparray()    # opens band 1, all rows and columns
        >>> print array
        >>>
        >>> # use the <mparray> function
        >>> # open specific rows and columns
        >>> array = mp.mparray(i_info, \
        >>>                    bands2open=[-1], \
        >>>                    i=100, j=100, \
        >>>                    rows=500, cols=500)
        >>>
        >>> # compute the NDVI (for Landsat-like band channels only)
        >>> i_info.mparray(compute_index='ndvi')
        >>> print i_info.ndvi
        >>> print i_info.array.shape    # note that the image array is a 2xrowsxcolumns array
        >>> # display the NDVI
        >>> i_info.show('ndvi')
        >>> # display band 1 of the image (band 1 of <array> is the red band)
        >>> i_info.show(band=1)
        >>> # write the NDVI to file
        >>> i_info.write2raster('/ndvi.tif', write_which='ndvi', \
        >>>                     o_info=i_info.copy(), storage='float32')
        >>>
        >>> # write an array to file
        >>> array = np.random.randn(3, 1000, 1000)
        >>> i_info.write2raster('/array.tif', write_which=array, \
        >>>                     o_info=i_info.copy(), storage='float32')
        >>>
        >>> # create info from scratch
        >>> i_info = mp.rinfo('create', left=, right=, top=, bottom=, \
        >>>                   cellY=, cellX=, bands=, storage=, projection=, \
        >>>                   rows=, cols=)
        >>>
        >>> # build overviews
        >>> i_info = mp.rinfo('/some_raster.tif')
        >>> i_info.build_overviews()
        >>> i_info.close()
        >>>
        >>> # remove overviews
        >>> i_info = mp.rinfo('/some_raster.tif', open2read=False)
        >>> i_info.remove_overviews()
        >>> i_info.close()
    """

    def __init__(self, file_name='none', open2read=True, metadata=None, sensor='Landsat',
                 hdf_band=1, check_corrupted=False, **kwargs):

        self.file_name = file_name

        passed = True

        # Initiate the file manager object.
        # FileManager.__init__(self)

        if file_name == 'create':
            self.update_info(**kwargs)
        elif file_name != 'none':
            self.get_image_info(open2read, hdf_band, check_corrupted)
        else:
            passed = False

        if isinstance(metadata, str):
            self.get_metadata(metadata, sensor)
        else:
            if not passed:
                print('\nNo image or metadata file was given.\n')

        # Check open files before closing.
        atexit.register(self.exit)

    def exit(self):

        if hasattr(self, 'band_open') and self.band_open:
            self.close_band()

        if hasattr(self, 'file_open') and self.file_open:
            self.close_file()

    def get_metadata(self, metadata, sensor):

        """
        Args:
            metadata (str): The metadata file.
            sensor (str): The satellite sensor to search. Default is 'Landsat'. Choices are ['Landsat', 'Sentinel2'].
        """

        if sensor == 'Landsat':
            LandsatParser.__init__(self, metadata)
        elif sensor == 'Sentinel2':
            SentinelParser.__init__(self, metadata)
        else:
            raise NameError('The {} sensor is not an option.'.format(sensor))

    def update_info(self, **kwargs):

        for k, v in kwargs.iteritems():
            setattr(self, k, v)

    def copy(self):

        return copy.copy(self)

    def close(self):

        """
        Closes the dataset
        """
        
        self.close_all()

    def mparray(self, bands2open=1, i=0, j=0, rows=-1, cols=-1, d_type=None,
                compute_index='none', sensor='Landsat', sort_bands2open=True, predictions=False,
                y=0., x=0., check_x=None, check_y=None):

        """
        Reads a raster as an array

        Args:
            bands2open (Optional[int or int list or dict]: Band position to open, list of bands to open, or a
                dictionary of name-band pairs. Default is 1.

                Examples:
                    bands2open = 1        (open band 1)
                    bands2open = [1,2,3]  (open first three bands)
                    bands2open = [4,3,2]  (open bands in a specific order)
                        *When opening bands in a specific order, be sure to set ``sort_bands2open`` as ``False``.
                    bands2open = -1       (open all bands)
                    bands2open = {'blue': 1, 'green': 2, 'nir': 4}  (open bands 1, 2, and 4)

            i (Optional[int]): Starting row position. Default is 0, or first row.
            j (Optional[int]): Starting column position. Default is 0, or first column.
            rows (Optional[int]): Number of rows to extract. Default is -1, or all rows.
            cols (Optional[int]): Number of columns to extract. Default is -1, or all columns.
            d_type (Optional[str]): Type of array to return. Choices are ['byte', 'int16', 'uint16',
                'int32', 'uint32', 'int64', 'uint64', 'float32', 'float64']. Default is None, or gathered
                from <i_info>.
            compute_index (Optional[str]): A spectral index to compute. Default is 'none'.
            sensor (Optional[str]): The input sensor type (used with ``compute_index``). Default is 'Landsat'.
            sort_bands2open (Optional[bool]): Whether to sort ``bands2open``. Default is True.
            predictions (Optional[bool]): Whether to return reshaped array for predictions.
            y (Optional[float]): A y index coordinate (latitude, in map units). Default is 0.
                If greater than 0, overrides ``i``.
            x (Optional[float]): A x index coordinate (longitude, in map units). Default is 0.
                If greater than 0, overrides ``j``.
            check_x (Optional[float]): Check the x offset against ``check_x``. Default is None.
            check_y (Optional[float]): Check the y offset against ``check_y``. Default is None.

        Attributes:
            array (ndarray)

        Returns:
            ``ndarray``, where shape is (rows x cols) if 1 band or (bands x rows x cols) if more than 1 band.

        Examples:
            >>> import mappy as mp
            >>>
            >>> i_info = mp.rinfo('image.tif')
            >>>
            >>> # Open 1 band.
            >>> array = i_info.mparray(bands2open=1)
            >>>
            >>> # Open multiple bands.
            >>> array = i_info.mparray(bands2open=[1, 2, 3])
            >>> band_1 = array[0]
            >>>
            >>> # Open as a dictionary of arrays.
            >>> bands = i_info.mparray(bands2open={'blue': 1, 'red': 2, 'nir': 4})
            >>> red = bands['red']
            >>>
            >>> # Index an image by pixel positions.
            >>> array = i_info.mparray(i=1000, j=4000, rows=500, cols=500)
            >>>
            >>> # Index an image by map coordinates.
            >>> array = i_info.mparray(y=1200000., x=4230000., rows=500, cols=500)
        """

        self.i = i
        self.j = j

        # `self.rows` and `self.cols` are the
        #   image dimension info, so don't overwrite.
        self.rrows = rows
        self.ccols = cols

        self.sort_bands2open = sort_bands2open

        if isinstance(bands2open, dict):

            if isinstance(d_type, str):
                self.d_type = STORAGE_DICT_NUMPY[d_type]
            else:
                self.d_type = STORAGE_DICT_NUMPY[self.storage.lower()]

        else:

            if isinstance(d_type, str):
                self.d_type = STORAGE_DICT[d_type]
            else:
                self.d_type = STORAGE_DICT[self.storage.lower()]

        if compute_index != 'none':

            from features.veg_indices import BandHandler

            bh = BandHandler(sensor)

            bh.get_band_order()

            bands2open = bh.get_band_positions(bh.wavelength_lists[compute_index.upper()])

            self.d_type = 'float32'

        if self.rrows == -1:
            self.rrows = self.rows
        else:
            if self.rrows > self.rows:
                raise ValueError('\nThe requested rows cannot be larger than the image rows.\n')

        if self.ccols == -1:
            self.ccols = self.cols
        else:
            if self.ccols > self.cols:
                raise ValueError('\nThe requested columns cannot be larger than the image columns.\n')

        # Index the image by x, y coordinates (in map units).
        if abs(y) > 0:
            __, __, __, self.i = get_xy_offsets(self, x=x, y=y)

        if abs(x) > 0:
            __, __, self.j, __ = get_xy_offsets(self, x=x, y=y)

        if isinstance(check_x, float) and isinstance(check_y, float):

            __, __, x_offset, y_offset = get_xy_offsets(self, x=check_x, y=check_y, check_position=False)

            self.i += y_offset
            self.j += x_offset

        #################
        # Bounds checking
        #################

        # Row indices
        if self.i < 0:
            self.i = 0

        if self.i >= self.rows:
            self.i = self.rows - 1

        # Number of rows
        self.rrows = n_rows_cols(self.i, self.rrows, self.rows)

        # Column indices
        if self.j < 0:
            self.j = 0

        if self.j >= self.cols:
            self.j = self.cols - 1

        # Number of columns
        self.ccols = n_rows_cols(self.j, self.ccols, self.cols)

        #################

        # format_dict = {'byte': 'B', 'int16': 'i', 'uint16': 'I', 'float32': 'f', 'float64': 'd'}
        # values = struct.unpack('%d%s' % ((rows * cols * len(bands2open)), format_dict[i_info.storage.lower()]),
        #     i_info.datasource.ReadRaster(yoff=i, xoff=j, xsize=cols, ysize=rows, band_list=bands2open))

        if hasattr(self, 'band'):

            self.array = self.band.ReadAsArray(self.j,
                                               self.i,
                                               self.ccols,
                                               self.rrows).astype(self.d_type)

            self.array_shape = [1, self.rrows, self.ccols]

            if predictions:
                self._reshape2predictions(1)

        else:

            # Check ``bands2open`` type.
            bands2open = self._check_band_list(bands2open)

            # Open the array.
            self._open_array(bands2open)

            if predictions:
                self._reshape2predictions(len(bands2open))

        if compute_index != 'none':

            from features import VegIndicesEquations

            vie = VegIndicesEquations(self.array, chunk_size=-1)

            exec 'self.{} = vie.compute(compute_index.upper())'.format(compute_index)

        return self.array

    def _open_array(self, bands2open):

        # Open the image as a dictionary of arrays.
        if isinstance(bands2open, dict):

            self.array = {}

            for band_name, band_position in bands2open.iteritems():

                self.array[band_name] = self.datasource.GetRasterBand(band_position).ReadAsArray(self.j,
                                                                                                 self.i,
                                                                                                 self.ccols,
                                                                                                 self.rrows).astype(self.d_type)

        # Open the image as an array.
        else:

            self.array = np.asarray([self.datasource.GetRasterBand(band).ReadAsArray(self.j,
                                                                                     self.i,
                                                                                     self.ccols,
                                                                                     self.rrows)
                                     for band in bands2open], dtype=self.d_type)

            self.array = self._reshape(self.array, bands2open)

    def predictions2norm(self):

        """
        Reshapes predictions back to 2d array
        """

        self.array = self.array.reshape(self.ccols, self.rrows).T

    def _reshape2predictions(self, n_bands):

        if n_bands == 1:

            self.array = self.array.reshape(self.rrows,
                                            self.ccols).T.reshape(self.rrows * self.ccols, n_bands)

        else:

            self.array = self.array.reshape(n_bands,
                                            self.rrows,
                                            self.ccols).T.reshape(self.rrows * self.ccols, n_bands)

        self.array_shape = [1, self.rrows*self.ccols, n_bands]

    def _reshape(self, array2reshape, band_list):

        if len(band_list) == 1:
            array2reshape = array2reshape.reshape(self.rrows, self.ccols)
        else:
            array2reshape = array2reshape.reshape(len(band_list), self.rrows, self.ccols)

        self.array_shape = [len(band_list), self.rrows, self.ccols]

        return array2reshape

    def _check_band_list(self, bands2open):

        if isinstance(bands2open, dict):

            return bands2open

        elif isinstance(bands2open, list):

            if len(bands2open) == 0:
                raise ValueError('\nA band list must be declared.\n')

            if max(bands2open) > self.bands:
                raise ValueError('\nThe requested band position cannot be greater than the image bands.\n')

        elif isinstance(bands2open, int):

            if bands2open > self.bands:
                raise ValueError('\nThe requested band position cannot be greater than the image bands.\n')

            if bands2open == -1:
                bands2open = range(1, self.bands+1)
            else:
                bands2open = [bands2open]

        else:
            raise TypeError('The ``bands2open`` parameter must be a dict, list, or int.')

        if self.sort_bands2open and not isinstance(bands2open, dict):
            bands2open = sorted(bands2open)

        return bands2open

    def write2raster(self, out_name, write_which='array', o_info=None, x=0, y=0, out_rst=None, write2bands=[],
                     compress='LZW', tile=False, close_band=True, flush_final=False, write_chunks=False, **kwargs):

        """
        Writes array to file

        Args:
            out_name (str): Output image name.
            o_info (Optional[object]): Output image information, instance of ``rinfo``.
                Needed if <out_rst> not given. Default is None.
            x (Optional[int]): Column starting position. Default is 0.
            y (Optional[int]): Row starting position. Default is 0.
            out_rst (Optional[object]): GDAL object to right to, otherwise created. Default is None.
            write2bands (Optional[int or int list]): Band positions to write to, otherwise takes the order of the input
                array dimensions. Default is None.
            compress (Optional[str]): Needed if <out_rst> not given. Default is 'LZW'.
            tile (Optional[bool]): Needed if <out_rst> not given. Default is False.
            close_band (Optional[bool]): Whether to flush the band cache. Default is True.
            flush_final (Optional[bool]): Whether to flush the raster cache. Default is False.
            write_chunks (Optional[bool]): Whether to write to file in <write_chunks> chunks. Default is False.

        Returns:
            None, writes <out_name>.
        """

        if isinstance(write_which, str):

            if write_which == 'ndvi':
                out_arr = self.ndvi
            elif write_which == 'evi2':
                out_arr = self.evi2
            elif write_which == 'pca':
                out_arr = self.pca_components
            else:
                out_arr = self.array

        elif isinstance(write_which, np.ndarray):
            out_arr = write_which

        gdal.SetCacheMax(2.56e+8)

        d_name, f_name = os.path.split(out_name)

        if not os.path.isdir(d_name):
            os.makedirs(d_name)

        array_shape = out_arr.shape

        if len(array_shape) == 2:

            out_rows, out_cols = out_arr.shape
            out_dims = 1

        else:

            out_dims, out_rows, out_cols = out_arr.shape

        new_file = False

        if not out_rst:

            new_file = True

            if kwargs:

                try:
                    o_info.storage = kwargs['storage']
                except:
                    pass

                try:
                    o_info.bands = kwargs['bands']
                except:
                    o_info.bands = out_dims

            o_info.rows = out_rows
            o_info.cols = out_cols

            out_rst = create_raster(out_name, o_info, compress=compress, tile=tile)

        # Specify a band to write to.
        if isinstance(write2bands, int) or isinstance(write2bands, list):

            if isinstance(write2bands, int):
                write2bands = [write2bands]

            for n_band in write2bands:

                out_rst.get_band(n_band)

                if write_chunks:

                    out_rst.get_chunk_size()

                    for i in xrange(0, out_rst.rows, out_rst.chunk_size):

                        n_rows = n_rows_cols(i, out_rst.chunk_size, out_rst.rows)

                        for j in xrange(0, out_rst.cols, out_rst.chunk_size):

                            n_cols = n_rows_cols(j, out_rst.chunk_size, out_rst.cols)

                            out_rst.write_array(out_arr[i:i+n_rows, j:j+n_cols], i=i, j=j)

                else:

                    out_rst.write_array(out_arr, i=y, j=x)

                if close_band:
                    out_rst.close_band()

        # Write in order of the 3rd array dimension.
        else:

            arr_shape = out_arr.shape

            if len(arr_shape) > 2:

                out_bands = arr_shape[0]

                for n_band in xrange(1, out_bands+1):

                    out_rst.write_array(out_arr[n_band-1], i=y, j=x, band=n_band)

                    if close_band:
                        out_rst.close_band()

            else:

                out_rst.write_array(out_arr, i=y, j=x, band=1)

                if close_band:
                    out_rst.close_band()

        # close the dataset if it was created or prompted by <flush_final>
        if flush_final or new_file:
            out_rst.close_file()

    def pca(self, n_components=3):

        """
        Computes Principle Components Analysis

        Args:
            n_components (Optional[int]): The number of components to return. Default is 3.

        Attributes:
            pca_components (ndarray)

        Returns:
            None
        """

        if n_components > self.bands:
            n_components = self.bands

        embedder = decomposition.PCA(n_components=n_components)

        dims, rs, cs = self.array.shape

        x = self.array.T.reshape(rs*cs, dims)

        scaler = StandardScaler().fit(x)
        x = scaler.transform(x.astype(np.float32)).astype(np.float32)

        x_fit = embedder.fit(x.astype(np.float32))
        x_reduced = x_fit.transform(x)

        self.pca_components = x_reduced.reshape(cs, rs, n_components).T

    def show(self, show_which='array', band=1, color_map='gist_stern', discrete=False,
             class_list=[], out_fig=None, dpi=300, clip_percentiles=(2, 98), equalize_hist=False,
             equalize_adapthist=False, gammas=[], sigmoid=[]):

        """
        Displays an array

        Args:
            show_which (Optional[str]): Which array to display. Default is 'array'. Choices are ['array',
                'evi2', 'gndvi', 'ndbai', 'ndvi', 'ndwi', 'savi'].
            band (Optional[int]): The band to display. Default is 1.
            color_map (Optional[str]): The colormap to use. Default is 'gist_stern'. For more colormaps, visit
                http://matplotlib.org/examples/color/colormaps_reference.html.
            discrete (Optional[bool]): Whether the colormap is discrete. Otherwise, continuous. Default is False.
            class_list (Optional[int list]): A list of the classes to display. Default is [].
            out_fig (Optional[str]): An output image to save to. Default is None.
            dpi (Optional[int]): The DPI of the output figure. Default is 300.
            clip_percentiles (Optional[tuple]): The lower and upper clip percentiles to rescale RGB images.
                Default is (2, 98).
            equalize_hist (Optional[bool]): Whether to equalize the histogram. Default is False.
            equalize_adapthist (Optional[bool]): Whether to equalize the histogram using a localized approach.
                Default is False.
            gammas (Optional[float list]): A list of gamma corrections for each band. Default is [].
            sigmoid (Optional[float list]): A list of sigmoid contrast and gain values. Default is [].

        Examples:
            >>> import mappy as mp
            >>> i_info = mp.rinfo('image')
            >>>
            >>> # Plot a discrete map with specified colors
            >>> color_map = ['#000000', '#DF7401', '#AEB404', '#0B6121', '#610B0B', '#A9D0F5',
            >>>              '#8181F7', '#BDBDBD', '#3A2F0B', '#F2F5A9', '#5F04B4']
            >>> i_info.show(color_map=color_map, discrete=True,
            >>>             class_list=[0,1,2,3,4,5,6,7,8,9,10])
            >>>
            >>> # Plot the NDVI
            >>> i_info.mparray(compute_index='ndvi')
            >>> i_info.show(show_which='ndvi')
            >>>
            >>> # Plot a single band array as greyscale
            >>> i_info.mparray(bands2open=4)
            >>> i_info.show(color_map='Greys')
            >>>
            >>> # Plot a 3-band array as RGB true color
            >>> i_info.mparray(bands2open=[3, 2, 1], sort_bands2open=False)
            >>> i_info.show(band='rgb')

        Returns:
            None
        """

        fig = plt.figure()
        ax = fig.add_subplot(111)

        plt.axis('off')

        if show_which == 'ndvi':

            self.ndvi[self.ndvi != 0] += 1.1

            if equalize_hist:
                self.ndvi = exposure.equalize_hist(self.ndvi)

            ip = ax.imshow(self.ndvi)
            im_min = np.percentile(self.ndvi, clip_percentiles[0])
            im_max = np.percentile(self.ndvi, clip_percentiles[1])

        elif show_which == 'evi2':

            if equalize_hist:
                self.evi2 = exposure.equalize_hist(self.evi2)

            ip = ax.imshow(self.evi2)
            im_min = np.percentile(self.evi2, clip_percentiles[0])
            im_max = np.percentile(self.evi2, clip_percentiles[1])

        elif show_which == 'pca':

            if equalize_hist:
                self.pca_components[band-1] = exposure.equalize_hist(self.pca_components[band-1])

            ip = ax.imshow(self.pca_components[band-1])
            im_min = np.percentile(self.pca_components[band-1], clip_percentiles[0])
            im_max = np.percentile(self.pca_components[band-1], clip_percentiles[1])

        else:

            if self.array_shape[0] > 1:

                if band == 'rgb':

                    for ii, im in enumerate(self.array):

                        pl, pu = np.percentile(im, clip_percentiles)
                        self.array[ii] = exposure.rescale_intensity(im, in_range=(pl, pu), out_range=(0, 255))

                        if equalize_hist:
                            self.array[ii] = exposure.equalize_hist(im)

                        if equalize_adapthist:
                            self.array[ii] = exposure.equalize_adapthist(im, ntiles_x=4, ntiles_y=4, clip_limit=0.5)

                        if gammas:
                            self.array[ii] = exposure.adjust_gamma(im, gammas[ii])

                        if sigmoid:
                            self.array[ii] = exposure.adjust_sigmoid(im, cutoff=sigmoid[0], gain=sigmoid[1])

                    # ip = ax.imshow(cv2.merge([self.array[2], self.array[1], self.array[0]]))
                    ip = ax.imshow(np.ascontiguousarray(self.array.transpose(1, 2, 0)))
                    # ip = ax.imshow(np.dstack((self.array[0], self.array[1], self.array[2])), interpolation='nearest')

                else:
                    ip = ax.imshow(self.array[band-1])
                    im_min = np.percentile(self.array[band-1], clip_percentiles[0])
                    im_max = np.percentile(self.array[band-1], clip_percentiles[1])

            else:
                ip = ax.imshow(self.array)
                im_min = np.percentile(self.array, clip_percentiles[0])
                im_max = np.percentile(self.array, clip_percentiles[1])

        ip.axes.get_xaxis().set_visible(False)
        ip.axes.get_yaxis().set_visible(False)

        if discrete:

            if isinstance(color_map, list):
                color_map = colors.ListedColormap(color_map)
                # color_map = colorbar.ColorbarBase(ax, cmap=color_map_)
                ip.set_cmap(color_map)
            elif color_map.lower() == 'random':
                ip.set_cmap(colors.ListedColormap(np.random.rand(len(class_list), 3)))
            else:
                ip.set_cmap(_discrete_cmap(len(class_list), base_cmap=color_map))

            ip.set_clim(min(class_list), max(class_list))

        else:
            if band != 'rgb':
                ip.set_cmap(color_map)
                ip.set_clim(im_min, im_max)

        cbar = plt.colorbar(ip, fraction=0.046, pad=0.04, orientation='horizontal')

        cbar.solids.set_edgecolor('face')

        # Remove colorbar container frame
        cbar.outline.set_visible(False)

        # cbar.set_ticks([])
        # cbar.set_ticklabels(class_list)

        if band == 'rgb':
            colorbar_label = 'RGB'
        else:
            if show_which == 'array':
                colorbar_label = 'Band {:d} of {:d} bands'.format(band, self.array_shape[0])
            else:
                colorbar_label = show_which.upper()

        cbar.ax.set_xlabel(colorbar_label)

        # Remove color bar tick lines, while keeping the tick labels
        cbarytks = plt.getp(cbar.ax.axes, 'xticklines')
        plt.setp(cbarytks, visible=False)

        if isinstance(out_fig, str):
            plt.savefig(out_fig, dpi=dpi, bbox_inches='tight', pad_inches=.1, transparent=True)
        else:
            plt.show()

        if show_which == 'ndvi':
            self.ndvi[self.ndvi != 0] -= 1.1

        plt.clf()
        plt.close(fig)


def gdal_open(image2open, band):

    """
    A direct file open from GDAL
    """

    driver_o = gdal.Open(image2open, GA_ReadOnly)

    return driver_o, driver_o.GetRasterBand(band)


def gdal_read(image2open, band, i, j, rows, cols):

    """
    A direct array read from GDAL
    """

    driver_o = gdal.Open(image2open, GA_ReadOnly)

    if isinstance(band, list):

        band_array = []

        for bd in band:

            band_object_o = driver_o.GetRasterBand(bd)
            band_array.append(band_object_o.ReadAsArray(j, i, cols, rows))

            band_object_o = None

        driver_o = None

        return np.array(band_array, dtype='float32').reshape(len(band), rows, cols)

    else:

        band_object_o = driver_o.GetRasterBand(band)
        band_array = np.float32(band_object_o.ReadAsArray(j, i, cols, rows))

    band_object_o = None
    driver_o = None

    return band_array


def gdal_write(band_object_w, array2write, io=0, jo=0):
    band_object_w.WriteArray(array2write, jo, io)


def gdal_close_band(band_object_c):

    try:
        band_object_c.FlushCache()
    except:
        pass

    del band_object_c

    band_object_c = None

    return band_object_c


def gdal_close_datasource(datasource_d):

    try:
        datasource_d.FlushCache()
    except:
        pass

    del datasource_d

    datasource_d = None

    return datasource_d


def gdal_register(image_name, in_memory=False):

    __, f_name = os.path.split(image_name)
    __, file_extension = os.path.splitext(f_name)

    if file_extension.lower() not in DRIVER_DICT:
        raise TypeError('{} is not an image, or is not a supported raster format.'.format(file_extension))
    else:
        file_format = DRIVER_DICT[file_extension.lower()]

    gdal.AllRegister()

    if in_memory:
        driver_r = gdal.GetDriverByName('MEM')
    else:
        driver_r = gdal.GetDriverByName(file_format)

    driver_r.Register

    return driver_r


def gdal_create(image_name, driver_cr, out_rows, out_cols, n_bands, storage_type,
                left, top, cellY, cellX, projection,
                in_memory=False, overwrite=False, parameters=[]):

    if overwrite:

        if os.path.isfile(image_name):
            os.remove(image_name)

    # Create the output driver.
    if in_memory:
        return driver_cr.Create('', out_cols, out_rows, n_bands, storage_type)
    else:
        ds = driver_cr.Create(image_name, out_cols, out_rows, n_bands, storage_type, parameters)

        # Set the geo-transformation.
        ds.SetGeoTransform([left, cellY, 0., top, 0., cellX])

        # Set the projection.
        ds.SetProjection(projection)

        return ds


def gdal_get_band(datasource_b, band_position):
    return datasource_b.GetRasterBand(band_position)


def _parallel_blocks(out_image, band_list, ii, jj, y_offset, x_offset,
                     nn_rows, nn_cols, left, top, cellY, cellX, projection, **kwargs):

    """
    Args:
        out_image:
        band_list:
        ii:
        jj:
        y_offset:
        x_offset:
        n_rows:
        n_cols:
        **kwargs:

    Returns:

    """

    # out_info_tile = out_info.copy()
    # out_info_tile.update_info(rows=nn_rows, cols=nn_cols,
    #                           left=out_info.left+(jj*out_info.cellY),
    #                           top=out_info.top-(ii*out_info.cellY))

    d_name_, f_name_ = os.path.split(out_image)
    f_base_, f_ext_ = os.path.splitext(f_name_)

    d_name_ = '{}/temp'.format(d_name_)

    rsn = '{:f}'.format(abs(np.random.randn(1)[0]))[-4:]

    out_image_tile = '{}/{}_{}{}'.format(d_name_, f_base_, rsn, f_ext_)

    datasource = gdal_create(out_image_tile, driver_pp, nn_rows, nn_cols, 1, STORAGE_DICT_GDAL['float32'],
                             left, top, cellY, cellX, projection)

    band_object = gdal_get_band(datasource, 1)

    # out_raster = create_raster(out_image_tile, out_info_tile)
    # out_raster.get_band(1)

    image_arrays = [gdal_read(image_infos_list[imi],
                              band_list[imi],
                              ii+y_offset[imi],
                              jj+x_offset[imi],
                              nn_rows,
                              nn_cols) for imi in xrange(0, len(image_infos_list))]

    output = block_func(image_arrays, **kwargs)

    gdal_write(band_object, output)

    band_object = gdal_close_band(band_object)
    datasource = gdal_close_datasource(datasource)

    # out_raster.write_array(output)
    #
    # out_raster.close_all()
    #
    # out_raster = None
    #
    # out_info_tile.close()
    #
    # out_info_tile = None

    return out_image_tile


class BlockFunc(object):

    """
    A class for block by block processing

    Args:
        func
        image_infos (list): A list of ``rinfo`` instances.
        out_image (str): The output image.
        out_info (object): An instance of ``rinfo``.
        band_list (Optional[list]): A list of band positions. Default is [].
        proc_info (Optional[object]): An instance of ``rinfo``. Overrides image_infos[0]. Default is None.
        y_offset (Optional[list]): The row offset. Default is [0].
        x_offset (Optional[list]): The column offset. Default is [0].
        block_rows (Optional[int]): The block row chunk size. Default is 2048.
        block_cols (Optional[int]): The block column chunk size. Default is 2048.
        d_type (Optional[str]): The read data type. Default is 'byte'.
        be_quiet (Optional[bool]): Whether to be quiet and do not print progress. Default is False.
        print_statement (Optional[str]): A string to print. Default is None.
        out_attributes (Optional[list]): A list of output attribute names. Default is [].
        write_array (Optional[bool]): Whether to write the output array to file. Default is True.
        boundary_file (Optional[str]): A file to use for block intersection. Default is None.
            Skip blocks that do not intersect ``boundary_file``.
        mask_file (Optional[str]): An file to use for block masking. Default is None.
            Recode blocks to binary 1 and 0 that intersect ``mask_file``.
        n_jobs (Optional[int]): The number of blocks to process in parallel. Default is 1.
        kwargs (Optional[dict]): Function specific parameters.

    Returns:
        None, writes to ``out_image``.
    """

    def __init__(self, func, image_infos, out_image, out_info,
                 band_list=[], proc_info=None, y_offset=[0], x_offset=[0],
                 block_rows=2048, block_cols=2048, be_quiet=False,
                 d_type='byte', print_statement=None, out_attributes=[],
                 write_array=True, boundary_file=None, mask_file=None,
                 n_jobs=1, **kwargs):

        self.func = func
        self.image_infos = image_infos
        self.out_image = out_image
        self.out_info = out_info
        self.band_list = band_list
        self.proc_info = proc_info
        self.y_offset = y_offset
        self.x_offset = x_offset
        self.block_rows = block_rows
        self.block_cols = block_cols
        self.d_type = d_type
        self.be_quiet = be_quiet
        self.print_statement = print_statement
        self.out_attributes = out_attributes
        self.write_array = write_array
        self.boundary_file = boundary_file
        self.mask_file = mask_file
        self.n_jobs = n_jobs
        self.kwargs = kwargs

        if not isinstance(self.out_image, str) and write_array:
            raise NameError('The output image was not given.')

        if self.n_jobs in [0, 1]:

            if not self.proc_info:
                self.proc_info = self.image_infos[0]

            for imi in xrange(0, len(self.image_infos)):
                if not isinstance(self.image_infos[imi], rinfo):
                    if not isinstance(self.image_infos[imi], GetMinExtent):
                        raise RinfoError('The image info list should be instances of rinfo or GetMinExtent.')

        if not isinstance(self.band_list, list) and isinstance(self.band_list, int):
            self.band_list = [self.band_list] * len(self.image_infos)
        else:

            if self.band_list:
                if len(self.band_list) != len(self.image_infos):
                    raise LenError('The band list and image info list much be the same length.')
            else:
                self.band_list = [1] * len(self.image_infos)

        if not isinstance(self.out_info, rinfo):
            if not isinstance(self.out_info, GetMinExtent):
                raise RinfoError('The output image object is not a MapPy instance.')

        if not isinstance(self.image_infos, list):
            raise TypeError('The image infos must be given as a list.')

        if not len(self.y_offset) == len(self.x_offset) == len(self.image_infos):
            raise LenError('The offset lists and input image info lists must be the same length.')

    def run(self):

        global block_func, image_infos_list, driver_pp

        if self.n_jobs in [0, 1]:

            for imi in xrange(0, len(self.image_infos)):
                if isinstance(self.band_list[imi], int):
                    self.image_infos[imi].get_band(self.band_list[imi])

            self._process_blocks()

        else:

            block_func = self.func
            image_infos_list = self.image_infos

            self._get_pairs()

            dn, __ = os.path.split(self.out_image)

            check_and_create_dir('{}/temp'.format(dn))

            driver_pp = gdal_register(self.out_image)

            tile_list = Parallel(n_jobs=self.n_jobs,
                                 max_nbytes=None)(delayed(_parallel_blocks)(self.out_image,
                                                                            self.band_list,
                                                                            pair[0], pair[1],
                                                                            self.y_offset,
                                                                            self.x_offset,
                                                                            pair[2], pair[3],
                                                                            self.out_info.left+(pair[1]*self.out_info.cellY),
                                                                            self.out_info.top-(pair[0]*self.out_info.cellY),
                                                                            self.out_info.cellY,
                                                                            self.out_info.cellX,
                                                                            self.out_info.projection,
                                                                            **self.kwargs) for pair in self.pairs)

    def _get_pairs(self):

        self.pairs = []

        for i in xrange(0, self.proc_info.rows, self.block_rows):

            n_rows = n_rows_cols(i, self.block_rows, self.proc_info.rows)

            for j in xrange(0, self.proc_info.cols, self.block_cols):

                n_cols = n_rows_cols(j, self.block_cols, self.proc_info.cols)

                self.pairs.append((i, j, n_rows, n_cols))

    def _process_blocks(self):

        if self.write_array:

            out_raster = create_raster(self.out_image, self.out_info)
            out_raster.get_band(1)

        # n_blocks = 0
        # for i in xrange(0, self.proc_info.rows, self.block_rows):
        #     for j in xrange(0, self.proc_info.cols, self.block_cols):
        #         n_blocks += 1
        #
        # n_block = 1

        if isinstance(self.print_statement, str):
            print(self.print_statement)

        # set widget and pbar
        if not self.be_quiet:
            ctr, pbar = _iteration_parameters(self.proc_info.rows, self.proc_info.cols,
                                              self.block_rows, self.block_cols)

        # iterate over the images and get change pixels
        for i in xrange(0, self.proc_info.rows, self.block_rows):

            n_rows = n_rows_cols(i, self.block_rows, self.proc_info.rows)

            for j in xrange(0, self.proc_info.cols, self.block_cols):

                n_cols = n_rows_cols(j, self.block_cols, self.proc_info.cols)

                if isinstance(self.boundary_file, str):

                    # Get the extent of the current block.
                    self.get_block_extent(i, j, n_rows, n_cols)

                    # Check if the block intersects the boundary file.
                    if not intersects_boundary(self.extent_dict, self.boundary_file):
                        continue

                # if not self.be_quiet:
                #
                #     if n_block == 1:
                #         print 'Blocks 1--19 of {:,d} ...'.format(n_blocks)
                #     elif n_block % 20 == 0:
                #         n_block_ = n_block + 19 if n_blocks - n_block > 20 else n_blocks
                #         print 'Block {:,d}--{:,d} of {:,d} ...'.format(n_block, n_block_, n_blocks)
                #
                #     n_block += 1

                image_arrays = [self.image_infos[imi].mparray(bands2open=self.band_list[imi],
                                                              i=i+self.y_offset[imi],
                                                              j=j+self.x_offset[imi],
                                                              rows=n_rows, cols=n_cols,
                                                              d_type=self.d_type)
                                for imi in xrange(0, len(self.image_infos))]

                if isinstance(self.mask_file, str):

                    self.get_block_extent(i, j, n_rows, n_cols)

                    orw = create_raster('none', None, in_memory=True, rows=n_rows, cols=n_cols,
                                        bands=1, projection=self.proc_info.projection,
                                        cellY=self.proc_info.cellY, cellX=self.proc_info.cellX,
                                        left=self.extent_dict['UL'][0], top=self.extent_dict['UL'][1],
                                        storage='byte')

                    # Rasterize the vector at the current block.
                    with vinfo(self.mask_file) as v_info:

                        gdal.RasterizeLayer(orw.datasource, [1], v_info.lyr, burn_values=[1])
                        block_array = orw.datasource.GetRasterBand(1).ReadAsArray(0, 0, n_cols, n_rows)

                        for imib, image_array in enumerate(image_arrays):

                            image_array[block_array == 0] = 0
                            image_arrays[imib] = image_array

                output = self.func(image_arrays, **self.kwargs)

                if isinstance(output, tuple):

                    if self.write_array:
                        out_raster.write_array(output[0], i=i, j=j)

                    # Get the other results.
                    for ri in xrange(1, len(output)):

                        self.kwargs[self.out_attributes[ri-1]] = output[ri]
                        setattr(self, self.out_attributes[ri-1], output[ri])

                else:

                    if self.write_array:
                        out_raster.write_array(output, i=i, j=j)

                if not self.be_quiet:

                    pbar.update(ctr)
                    ctr += 1

        if not self.be_quiet:
            pbar.finish()

        for imi in xrange(0, len(self.image_infos)):
            self.image_infos[imi].close()

        if self.write_array:
            out_raster.close_all()

    def get_block_extent(self, ii, jj, nn_rows, nn_cols):

        adj_left = self.proc_info.left + (jj * self.proc_info.cellY)
        adj_right = adj_left + (nn_cols * self.proc_info.cellY) + self.proc_info.cellY
        adj_top = self.proc_info.top - (ii * self.proc_info.cellY)
        adj_bottom = adj_top - (nn_rows * self.proc_info.cellY) - self.proc_info.cellY

        self.extent_dict = {'UL': [adj_left, adj_top],
                            'UR': [adj_right, adj_top],
                            'LL': [adj_left, adj_bottom],
                            'LR': [adj_right, adj_bottom]}


def _mparray_parallel(image, image_info, bands2open, y, x, rows2open, columns2open, n_jobs, d_type, predictions):

    """
    Opens image bands into arrays using multiple processes

    Args:
        image (str): The image to open.
        image_info (instance)
        bands2open (int or int list: Band position to open or list of bands to open.
        y (int): Starting row position.
        x (int): Starting column position.
        rows2open (int): Number of rows to extract.
        columns2open (int): Number of columns to extract.
        n_jobs (int): The number of jobs to run in parallel.
        d_type (str): Type of array to return.
        predictions (bool): Whether to return reshaped array for predictions.

    Returns:
        Ndarray where [rows, cols] if 1 band and [bands, rows, cols] if more than 1 band
    """

    if isinstance(bands2open, list):
        if max(bands2open) > image_info.bands:
            raise ValueError('\nCannot open more bands than exist in the image.\n')
    else:
        if bands2open == -1:
            bands2open = range(1, image_info.bands+1)

    if rows2open == -1:
        rows2open = image_info.rows

    if columns2open == -1:
        columns2open = image_info.cols

    image_info.close()

    band_arrays = Parallel(n_jobs=n_jobs)(delayed(gdal_read)(image, band2open, y, x, rows2open, columns2open)
                                          for band2open in bands2open)

    if predictions:

        return np.array(band_arrays, dtype=d_type).reshape(len(bands2open),
                                                           rows2open,
                                                           columns2open).T.reshape(rows2open*columns2open,
                                                                                   len(bands2open))

    else:
        return np.array(band_arrays, dtype=d_type).reshape(len(bands2open), rows2open, columns2open)


def mparray(image2open=None, i_info=None, bands2open=1, i=0, j=0,
            rows=-1, cols=-1, d_type=None, n_jobs=0,
            predictions=False, sort_bands2open=True, y=0., x=0.):

    """
    Reads a raster as an array

    Args:
        image2open (Optional[str]): An image to open. Default is None.
        i_info (Optional[object]): An instance of ``rinfo``. Default is None
        bands2open (Optional[int list or int]: Band position to open or list of bands to open. Default is 1.
            Examples:
                bands2open = 1        (open band 1)
                bands2open = [1,2,3]  (open first three bands)
                bands2open = -1       (open all bands)
        i (Optional[int]): Starting row position. Default is 0, or first row.
        j (Optional[int]): Starting column position. Default is 0, or first column.
        rows (Optional[int]): Number of rows to extract. Default is all rows.
        cols (Optional[int]): Number of columns to extract. Default is all columns.
        d_type (Optional[str]): Type of array to return. Default is None, or gathered from <i_info>.
            Choices are ['uint8', 'int8', 'uint16', 'uint32', 'int16', 'float32', 'float64'].
        n_jobs (Optional[int]): The number of bands to open in parallel. Default is 0.
        predictions (Optional[bool]): Whether to return reshaped array for predictions.
        sort_bands2open (Optional[bool]): Whether to sort ``bands2open``. Default is True.
        y (Optional[float]): A y index coordinate. Default is 0. If greater than 0, overrides ``i``.
        x (Optional[float]): A x index coordinate. Default is 0. If greater than 0, overrides ``j``.

    Attributes:
        array (ndarray)

    Returns:
        Ndarray where [rows, cols] if 1 band and [bands, rows, cols] if more than 1 band

    Examples:
        >>> import mappy as mp
        >>>
        >>> array = mp.mparray('image.tif')
        >>>
        >>> array = mp.mparray('image.tif', bands2open=[1, 2, 3])
        >>> print(a.shape)
        >>>
        >>> array = mp.mparray('image.tif', bands2open={'green': 3, 'nir': 4})
        >>> print(len(array))
        >>> print(array['nir'].shape)
    """

    if not isinstance(i_info, rinfo) and not isinstance(image2open, str):
        raise NameError('\nEither i_info or image2open must be declared.\n')
    elif isinstance(i_info, rinfo) and isinstance(image2open, str):
        raise NameError('\nChoose either i_info or image2open, but not both.\n')
    elif not isinstance(i_info, rinfo) and isinstance(image2open, str):
        i_info = rinfo(image2open)

    if (n_jobs == 0) and not predictions:

        kwargs = dict(bands2open=bands2open, i=i, j=j, rows=rows, cols=cols, d_type=d_type,
                      sort_bands2open=sort_bands2open, y=y, x=x)

        return i_info.mparray(**kwargs)

    else:

        if isinstance(d_type, str):
            d_type = STORAGE_DICT[d_type]
        else:
            d_type = STORAGE_DICT[i_info.storage.lower()]

        if rows == -1:
            rows = i_info.rows
        else:
            if rows > i_info.rows:
                raise ValueError('\nThe requested rows cannot be larger than the image rows.\n')

        if cols == -1:
            cols = i_info.cols
        else:
            if cols > i_info.cols:
                raise ValueError('\nThe requested columns cannot be larger than the image columns.\n')

        # format_dict = {'byte': 'B', 'int16': 'i', 'uint16': 'I', 'float32': 'f', 'float64': 'd'}

        if isinstance(bands2open, list):

            if len(bands2open) == 0:
                raise ValueError('\nA band list must be declared.\n')

            if max(bands2open) > i_info.bands:
                raise ValueError('\nThe requested band position cannot be greater than the image bands.\n')

        elif isinstance(bands2open, int):

            if bands2open > i_info.bands:
                raise ValueError('\nThe requested band position cannot be greater than the image bands.\n')

            if bands2open == -1:
                bands2open = range(1, i_info.bands+1)
            else:
                bands2open = [bands2open]

        if sort_bands2open:
            bands2open = sorted(bands2open)

        # Index the image by x, y coodinates (in map units).
        if abs(y) > 0:
            __, __, __, i = get_xy_offsets(i_info, x=x, y=y)

        if abs(x) > 0:
            __, __, j, __ = get_xy_offsets(i_info, x=x, y=y)

        if n_jobs == 0:

            values = np.asarray([i_info.datasource.GetRasterBand(band).ReadAsArray(j, i, cols, rows)
                                 for band in bands2open], dtype=d_type)

            # values = struct.unpack('%d%s' % ((rows * cols * len(bands2open)), format_dict[i_info.storage.lower()]),
            #                        i_info.datasource.ReadRaster(yoff=i, xoff=j, xsize=cols, ysize=rows, band_list=bands2open))

            if predictions:
                return values.reshape(len(bands2open), rows, cols).T.reshape(rows*cols, len(bands2open))
            else:

                if len(bands2open) == 1:
                    return values.reshape(rows, cols)
                else:
                    return values.reshape(len(bands2open), rows, cols)

            # only close the image if it was opened internally
            # if isinstance(image2open, str):
            #     i_info.close()

        else:

            return _mparray_parallel(image2open, i_info, bands2open, i, j, rows, cols, n_jobs, d_type, predictions)


class create_raster(CreateDriver, FileManager):

    """
    Creates a raster driver to write to.

    Args:
        out_name (str): Output raster name.
        o_info (object): Instance of ``rinfo``.
        compress (Optional[str]): The type of compression to use. Default is 'lzw'.
            Choices are ['none' 'lzw', 'packbits', 'deflate'].
        bigtiff (Optional[str]): How to manage large TIFF files. Default is 'no'.
            Choices are ['yes', 'no', 'if_needed', 'if_safer'].
        tile (Optional[bool]): Whether to tile the new image. Default is True.
        project_epsg (Optional[int]): Project the new raster to an EPSG code projection.
        create_tiles (Optional[str]): If positive, image is created in separate file tiles. Default is 0.
        overwrite (Optional[str]): Whether to overwrite an existing file. Default is False.
        in_memory (Optional[str]): Whether to create the raster dataset in memory. Default is False.

    Attributes:
        filename (str)
        rows (int)
        cols (int)
        bands (int)
        storage (str)

    Returns:
        Raster driver GDAL object or list of GDAL objects (if create_tiles > 0).
    """

    def __init__(self, out_name, o_info, compress='lzw', tile=True, bigtiff='no', project_epsg=None,
                 create_tiles=0, overwrite=False, in_memory=False, **kwargs):

        if not in_memory:

            d_name, f_name = os.path.split(out_name)
            f_base, f_ext = os.path.splitext(f_name)

            check_and_create_dir(d_name)

        storage_type = STORAGE_DICT_GDAL[o_info.storage.lower()] if 'storage' not in kwargs \
            else STORAGE_DICT_GDAL[kwargs['storage'].lower()]

        out_rows = o_info.rows if 'rows' not in kwargs else kwargs['rows']
        out_cols = o_info.cols if 'cols' not in kwargs else kwargs['cols']
        n_bands = o_info.bands if 'bands' not in kwargs else kwargs['bands']
        projection = o_info.projection if 'projection' not in kwargs else kwargs['projection']
        cellY = o_info.cellY if 'cellY' not in kwargs else kwargs['cellY']
        cellX = o_info.cellX if 'cellX' not in kwargs else kwargs['cellX']
        left = o_info.left if 'left' not in kwargs else kwargs['left']
        top = o_info.top if 'top' not in kwargs else kwargs['top']

        if tile:
            tile = 'YES'
        else:
            tile = 'NO'

        if abs(cellY) == 0:
            raise ValueError('The cell y size must be greater than 0.')

        if abs(cellX) == 0:
            raise ValueError('The cell x size must be greater than 0.')

        if cellX > 0:
            cellX *= -1.

        if cellY < 0:
            cellY *= -1.

        if out_name.lower().endswith('.img'):

            if compress.upper() == 'NONE':
                parameters = ['COMPRESS=NO']
            else:
                parameters = ['COMPRESS=YES']

        elif out_name.lower().endswith('.tif'):

            if compress.upper() == 'NONE':
                parameters = ['TILED={}'.format(tile), 'BIGTIFF={}'.format(bigtiff.upper())]
            else:
                parameters = ['TILED={}'.format(tile), 'COMPRESS={}'.format(compress.upper()),
                              'BIGTIFF={}'.format(bigtiff.upper())]

        elif (out_name.lower().endswith('.dat')) or (out_name.lower().endswith('.bin')):

            parameters = ['INTERLEAVE=BSQ']

        elif out_name.lower().endswith('.kea'):

            parameters = ['DEFLATE=1']

        else:
            parameters = []

        if isinstance(project_epsg, int):

            osng = osr.SpatialReference()
            osng.ImportFromWkt(o_info.projection)

            srs = osr.SpatialReference()
            srs.ImportFromEPSG(project_epsg)
            new_projection = srs.ExportToWkt()

            tx = osr.CoordinateTransformation(osng, srs)

            # Work out the boundaries of the new dataset in the target projection
            ulx, uly, ulz = tx.TransformPoint(o_info.left, o_info.top)
            lrx, lry, lrz = tx.TransformPoint(o_info.left + o_info.cellY*o_info.cols,
                                              o_info.top + o_info.cellX*o_info.rows)

            # project_rows = int((uly - lry) / o_info.cellY)
            # project_cols = int((lrx - ulx) / o_info.cellY)

            # Calculate the new geotransform
            new_geo = [ulx, o_info.cellY, o_info.rotation1, uly, o_info.rotation2, o_info.cellX]

            # out_rows = int((uly - lry) / o_info.cellY)
            # out_cols = int((lrx - ulx) / o_info.cellY)

        # Create driver for output image.
        if create_tiles > 0:

            d_name_tiles = '{}/{}_tiles'.format(d_name, f_base)

            if not os.path.isdir(d_name_tiles):
                os.makedirs(d_name_tiles)

            out_rst = {}

            if out_rows >= create_tiles:
                blk_size_rows = create_tiles
            else:
                blk_size_rows = copy.copy(out_rows)

            if out_cols >= create_tiles:
                blk_size_cols = create_tiles
            else:
                blk_size_cols = copy.copy(out_cols)

            topo = copy.copy(top)

            image_counter = 1

            for i in xrange(0, out_rows, blk_size_rows):

                lefto = copy.copy(left)

                out_rows = n_rows_cols(i, blk_size_rows, out_rows)

                for j in xrange(0, out_cols, blk_size_cols):

                    out_cols = n_rows_cols(j, blk_size_cols, out_cols)

                    out_name = '{}/{}_{:d}_{:d}{}'.format(d_name_tiles, f_base, i, j, f_ext)

                    out_rst[image_counter] = out_name

                    image_counter += 1

                    if overwrite:

                        if os.path.isfile(out_name):

                            try:
                                os.remove(out_name)
                            except OSError:
                                raise OSError('\nCould not delete {}.'.format(out_name))

                    else:

                        if os.path.isfile(out_name):

                            print('\n{} already exists.'.format(out_name))

                            continue

                    CreateDriver.__init__(self, out_name, out_rows, out_cols, n_bands,
                                          storage_type, in_memory, overwrite, parameters)

                    # FileManager.__init__(self)

                    # out_rst_ = self.driver.Create(out_name, out_cols, out_rows, bands, storage_type, parameters)

                    # set the geo-transformation
                    self.datasource.SetGeoTransform([lefto, cellY, 0., topo, 0., cellX])

                    # set the projection
                    self.datasource.SetProjection(projection)

                    self.close_file()

                    lefto += (out_cols * cellY)

                topo -= (out_rows * cellY)

        else:

            if not in_memory:

                if overwrite:

                    if os.path.isfile(out_name):

                        try:
                            os.remove(out_name)
                        except OSError:
                            raise OSError('\nCould not delete {}.'.format(out_name))

                else:

                    if os.path.isfile(out_name):
                        print('\n{} already exists.\n'.format(out_name))

            CreateDriver.__init__(self, out_name, out_rows, out_cols, n_bands,
                                  storage_type, in_memory, overwrite, parameters)

            # FileManager.__init__(self)

            # self.datasource = self.driver.Create(out_name, out_cols, out_rows, bands, storage_type, parameters)

            if isinstance(project_epsg, int):

                # set the geo-transformation
                self.datasource.SetGeoTransform(new_geo)

                # set the projection
                self.datasource.SetProjection(new_projection)

                # gdal.ReprojectImage(o_info.datasource, out_rst, o_info.proj, new_projection, GRA_NearestNeighbour)

            else:

                # Set the geo-transformation.
                self.datasource.SetGeoTransform([left, cellY, 0., top, 0., cellX])

                # Set the projection.
                self.datasource.SetProjection(projection)

            self.filename = out_name
            self.rows = out_rows
            self.cols = out_cols
            self.bands = n_bands
            self.storage = storage_type


def write2raster(out_arr, out_name, o_info=None, x=0, y=0, out_rst=None, write2bands=[], compress='lzw',
                 tile=True, close_band=True, flush_final=False, write_chunks=False, **kwargs):

    """
    Writes an ndarray to file.

    Args:
        out_arr (ndarray): The array to write to file.
        out_name (str): The output image name.
        o_info (Optional[object]): Output image information. Needed if ``out_rst`` not given. Default is None.
        x (Optional[int]): Column starting position. Default is 0.
        y (Optional[int]): Row starting position. Default is 0.
        out_rst (Optional[object]): GDAL object to right to, otherwise created. Default is None.
        write2bands (Optional[int or int list]): Band positions to write to, otherwise takes the order of the input
            array dimensions. Default is None.
        compress (Optional[str]): Needed if <out_rst> not given. Default is 'lzw'.
        tile (Optional[bool]): Needed if <out_rst> not given. Default is False.
        close_band (Optional[bool]): Whether to flush the band cache. Default is True.
        flush_final (Optional[bool]): Whether to flush the raster cache. Default is False.
        write_chunks (Optional[bool]): Whether to write to file in <write_chunks> chunks. Default is False.

    Returns:
        None, writes <out_name>.

    Examples:
        >>> # Example
        >>> import mappy as mp
        >>> i_info = mp.rinfo('/in_raster.tif')
        >>> out_array = np.random.randn(3, 100, 100).astype(np.float32)
        >>> mp.write2raster(out_array, '/out_name.tif', o_info=copy(i_info),
        >>>                 flush_final=True)
    """

    gdal.SetCacheMax(2.56e+8)

    d_name, f_name = os.path.split(out_name)

    if not os.path.isdir(d_name):
        os.makedirs(d_name)

    array_shape = out_arr.shape

    if len(array_shape) == 2:

        out_rows, out_cols = out_arr.shape
        out_dims = 1

    else:

        out_dims, out_rows, out_cols = out_arr.shape

    new_file = False

    if not out_rst:

        new_file = True

        if kwargs:

            try:
                o_info.storage = kwargs['storage']
            except:
                pass

            try:
                o_info.bands = kwargs['bands']
            except:
                o_info.bands = out_dims

        o_info.rows = out_rows
        o_info.cols = out_cols

        out_rst = create_raster(out_name, o_info, compress=compress, tile=tile)

    ##########################
    # pack the data to binary
    ##########################
    # format_dict = {'byte': 'B', 'int16': 'i', 'uint16': 'I', 'float32': 'f', 'float64': 'd'}

    # specifiy a band to write to
    if isinstance(write2bands, int) or (isinstance(write2bands, list) and write2bands):

        if isinstance(write2bands, int):
            write2bands = [write2bands]

        for n_band in write2bands:

            out_rst.get_band(n_band)

            if write_chunks:

                out_rst.get_chunk_size()

                for i in xrange(0, out_rst.rows, out_rst.chunk_size):

                    n_rows = n_rows_cols(i, out_rst.chunk_size, out_rst.rows)

                    for j in xrange(0, out_rst.cols, out_rst.chunk_size):

                        n_cols = n_rows_cols(j, out_rst.chunk_size, out_rst.cols)

                        out_rst.write_array(out_arr[i:i+n_rows, j:j+n_cols], i=i, j=j)

            else:

                out_rst.write_array(out_arr, i=y, j=x)

            if close_band:

                out_rst.close_band()

    # write in order of the 3rd array dimension
    else:

        arr_shape = out_arr.shape

        if len(arr_shape) > 2:

            out_bands = arr_shape[0]

            for n_band in xrange(1, out_bands+1):

                out_rst.write_array(out_arr[n_band-1], i=y, j=x, band=n_band)

                if close_band:
                    out_rst.close_band()

        else:

            out_rst.write_array(out_arr, i=y, j=x, band=1)

            if close_band:
                out_rst.close_band()

    # close the dataset if it was created or prompted by <flush_final>
    if flush_final or new_file:
        out_rst.close_file()


class GetMinExtent(object):

    """
    Args:
        info1 (rinfo or GetMinExtent object)
        info2 (rinfo or GetMinExtent object)

    Attributes:
        Inherits from ``info1``.
    """

    def __init__(self, info1, info2):

        if not isinstance(info1, rinfo):
            if not isinstance(info1, GetMinExtent):
                raise TypeError('The first info argument must be an instance of rinfo or GetMinExtent.')

        if not isinstance(info2, rinfo):
            if not isinstance(info2, GetMinExtent):
                if not isinstance(info2, vinfo):
                    raise TypeError('The second info argument must be an instance of rinfo, vinfo, or GetMinExtent.')

        # Pass the image info properties.
        attributes = inspect.getmembers(info1, lambda ia: not (inspect.isroutine(ia)))
        attributes = [ia for ia in attributes if not (ia[0].startswith('__') and ia[0].endswith('__'))]

        for attribute in attributes:
            setattr(self, attribute[0], attribute[1])

        setattr(self, 'update_info', info1.update_info)

        self.get_overlap_info(info2)

    def copy(self):
        return copy.copy(self)

    def get_overlap_info(self, info2):

        self.left = np.maximum(self.left, info2.left)
        self.right = np.minimum(self.right, info2.right)
        self.top = np.minimum(self.top, info2.top)
        self.bottom = np.maximum(self.bottom, info2.bottom)

        if (self.left < 0) and (self.right < 0) or (self.left >= 0) and (self.right >= 0):
            self.cols = int(abs(abs(self.right) - abs(self.left)) / self.cellY)
        elif (self.left < 0) and (self.right >= 0):
            self.cols = int(abs(abs(self.right) + abs(self.left)) / self.cellY)

        if (self.top < 0) and (self.bottom < 0) or (self.top >= 0) and (self.bottom >= 0):
            self.rows = int(abs(abs(self.top) - abs(self.bottom)) / self.cellY)
        elif (self.top >= 0) and (self.bottom < 0):
            self.rows = int(abs(abs(self.top) + abs(self.bottom)) / self.cellY)

        # Rounded dimensions for aligning pixels.
        left_max = np.minimum(self.left, info2.left)
        top_max = np.maximum(self.top, info2.top)

        if (left_max < 0) and (self.left < 0):
            n_col_pixels = int((abs(left_max) - abs(self.left)) / self.cellY)
            self.left_rounded = left_max + (n_col_pixels * self.cellY)
        elif (left_max >= 0) and (self.left >= 0):
            n_col_pixels = int((abs(left_max) - abs(self.left)) / self.cellY)
            self.left_rounded = left_max + (n_col_pixels * self.cellY)
        elif (left_max < 0) and (self.left >= 0):
            n_col_pixels1 = int(abs(left_max) / self.cellY)
            n_col_pixels2 = int(abs(self.left) / self.cellY)
            self.left_rounded = left_max + (n_col_pixels1 * self.cellY) + (n_col_pixels2 * self.cellY)

        if (top_max >= 0) and (self.top >= 0):
            n_row_pixels = int((abs(top_max) - abs(self.top)) / self.cellY)
            self.top_rounded = top_max - (n_row_pixels * self.cellY)
        elif (top_max < 0) and (self.top < 0):
            n_row_pixels = int((abs(top_max) - abs(self.top)) / self.cellY)
            self.top_rounded = top_max - (n_row_pixels * self.cellY)
        elif (top_max >= 0) and (self.top < 0):
            n_row_pixels1 = int(abs(top_max) / self.cellY)
            n_row_pixels2 = int(abs(self.top) / self.cellY)
            self.top_rounded = top_max - (n_row_pixels1 * self.cellY) - (n_row_pixels2 * self.cellY)

        if (self.left_rounded < 0) and (self.right < 0):
            n_col_pixels_r = int((abs(self.left_rounded) - abs(self.right)) / self.cellY)
            self.right_rounded = self.left_rounded + (n_col_pixels_r * self.cellY)
        elif (self.left_rounded >= 0) and (self.right >= 0):
            n_col_pixels_r = int((abs(self.left_rounded) - abs(self.right)) / self.cellY)
            self.right_rounded = self.left_rounded + (n_col_pixels_r * self.cellY)
        elif (self.left_rounded < 0) and (self.right >= 0):
            n_col_pixels_r1 = int(abs(self.left_rounded) / self.cellY)
            n_col_pixels_r2 = int(abs(self.right) / self.cellY)
            self.right_rounded = self.left_rounded + (n_col_pixels_r1 * self.cellY) + (n_col_pixels_r2 * self.cellY)

        if (self.top_rounded < 0) and (self.bottom < 0):
            n_row_pixels_r = int((abs(self.top_rounded) - abs(self.bottom)) / self.cellY)
            self.bottom_rounded = self.top_rounded - (n_row_pixels_r * self.cellY)
        elif (self.top_rounded >= 0) and (self.bottom >= 0):
            n_row_pixels_r = int((abs(self.top_rounded) - abs(self.bottom)) / self.cellY)
            self.bottom_rounded = self.top_rounded - (n_row_pixels_r * self.cellY)
        elif (self.top_rounded >= 0) and (self.bottom < 0):
            n_row_pixels_r1 = int(abs(self.top_rounded) / self.cellY)
            n_row_pixels_r2 = int(abs(self.bottom) / self.cellY)
            self.bottom_rounded = self.top_rounded - (n_row_pixels_r1 * self.cellY) + (n_row_pixels_r2 * self.cellY)


def get_min_extent(image1, image2):

    """
    Finds the minimum extent of two rasters

    Args:
        image1 (dict or object): The first image. If a ``dict``, {left: <left>, right: <right>,
            top: <top>, bottom: <bottom>}.
        image2 (dict or object): The second image. If a ``dict``, {left: <left>, right: <right>,
            top: <top>, bottom: <bottom>}.

    Returns:
        List as [left, right, top, bottom].
    """

    if isinstance(image1, rinfo):
        left1 = image1.left
        top1 = image1.top
        right1 = image1.right
        bottom1 = image1.bottom
    else:
        left1 = image1['left']
        top1 = image1['top']
        right1 = image1['right']
        bottom1 = image1['bottom']

    if isinstance(image2, rinfo):
        left2 = image2.left
        top2 = image2.top
        right2 = image2.right
        bottom2 = image2.bottom
    else:
        left2 = image2['left']
        top2 = image2['top']
        right2 = image2['right']
        bottom2 = image2['bottom']

    left = np.maximum(left1, left2)
    right = np.minimum(right1, right2)
    top = np.minimum(top1, top2)
    bottom = np.maximum(bottom1, bottom2)

    return left, right, top, bottom


def get_min_extent_list(image_list):

    lefto = image_list[0].left
    righto = image_list[0].right
    topo = image_list[0].top
    bottomo = image_list[0].bottom
    cell_size = image_list[0].cellY

    for img in image_list[1:]:

        lefto, righto, topo, bottomo = \
            get_min_extent(dict(left=lefto, right=righto, top=topo, bottom=bottomo),
                           dict(left=img.left, right=img.right, top=img.top, bottom=img.bottom))

    # Check for East/West, positive/negative dividing line.
    if (righto >= 0) and (lefto <= 0):
        cs = int((abs(lefto) + righto) / cell_size)
    else:
        cs = int(abs(abs(righto) - abs(lefto)) / cell_size)

    if (topo >= 0) and (bottomo <= 0):
        rs = int((abs(bottomo) + topo) / cell_size)
    else:
        rs = int(abs(abs(topo) - abs(bottomo)) / cell_size)

    return [lefto, topo, righto, bottomo, -cell_size, cell_size, rs, cs]


def get_new_dimensions(image_info, kernel_size):

    """
    Gets new [output] image dimensions based on kernel size used in processing.

    Args:
        image_info (object)
        kernel_size (int)

    Returns:
        ``new rows``, ``new columns``, ``new cell size y``, ``new cell size x``
    """

    image_info.rows = int(np.ceil(float(image_info.rows) / float(kernel_size)))
    image_info.cols = int(np.ceil(float(image_info.cols) / float(kernel_size)))

    image_info.cellY = float(kernel_size) * float(image_info.cellY)
    image_info.cellX = float(kernel_size) * float(image_info.cellX)

    return image_info


def n_rows_cols(pixel_index, block_size, rows_cols):

    """
    Adjusts block size for the end of image rows and columns.

    Args:
        pixel_index (int): The current pixel row or column index.
        block_size (int): The image block size.
        rows_cols (int): The total number of rows or columns in the image.

    Example:
        >>> n_rows = 5000
        >>> block_size = 1024
        >>> i = 4050
        >>> adjusted_block_size = n_rows_cols(i, block_size, n_rows)

    Returns:
        Adjusted block size as int.
    """

    if (pixel_index + block_size) < rows_cols:
        samp_out = block_size
    else:
        samp_out = rows_cols - pixel_index

    return samp_out


def n_i_j(pixel_index, offset):

    """
    Args:
        pixel_index (int): Current pixel index.
        block_size (int): Block size to use.

    Returns:
        int
    """

    if pixel_index - offset < 0:
        samp_out = 0
    else:
        samp_out = pixel_index - offset

    return samp_out


def block_dimensions(image_rows, image_cols, row_block_size=1024, col_block_size=1024):

    """
    Args:
        image_rows (int): The number of image rows.
        image_cols (int): The number of image columns.
        row_block_size (Optional[int]): Default is 1024.
        col_block_size (Optional[int]): Default is 1024.

    Returns:
        Row dimensions, Column dimensions
    """

    # set the block dimensions
    if image_rows >= row_block_size:
        row_blocks = row_block_size
    else:
        row_blocks = copy.copy(image_rows)

    if image_cols >= col_block_size:
        col_blocks = col_block_size
    else:
        col_blocks = copy.copy(image_cols)

    return row_blocks, col_blocks


def stats_func(im, ignore_value=None, stat=None, stats_functions=None,
               set_below=None, set_above=None, set_common=None, no_data_value=None):

    im = im[0][:]

    if isinstance(ignore_value, int):

        stat = 'nan{}'.format(stat)

        im[im == ignore_value] = np.nan

    if stat in stats_functions:
        out_array = stats_functions[stat](im, axis=0)
    elif stat == 'nancv':
        out_array = stats_functions['nanstd'](im, axis=0)
        out_array /= stats_functions['nanmean'](im, axis=0)
    elif stat == 'nanmode':
        out_array = sci_mode(im, axis=0, nan_policy='omit')
    elif stat == 'cv':
        out_array = im.std(axis=0)
        out_array /= im.mean(axis=0)
    elif stat == 'min':
        out_array = im.max(axis=0)
    elif stat == 'min':
        out_array = im.max(axis=0)
    elif stat == 'mean':
        out_array = im.mean(axis=0)
    elif stat == 'var':
        out_array = im.var(axis=0)
    elif stat == 'std':
        out_array = im.std(axis=0)
    elif stat == 'sum':
        out_array = im.sum(axis=0)

    # Filter values.
    if isinstance(set_below, int):
        out_array[out_array < set_below] = no_data_value

    if isinstance(set_above, int):

        if set_common:

            # Mask unwanted to 1 above threshold.
            out_array[out_array > set_above] = set_above + 1

            # Invert the array values.
            __, out_array = cv2.threshold(np.uint8(out_array), 0, 1, cv2.THRESH_BINARY_INV)

            # Add the common value among all bands.
            out_array *= np.uint8(im[0])

        else:
            out_array[out_array > set_above] = no_data_value

    # Reset no data pixels
    out_array[np.isnan(out_array) | np.isinf(out_array)] = no_data_value

    return out_array


def pixel_stats(input_image, output_image, stat='mean', bands2process=-1,
                ignore_value=None, no_data_value=0, set_below=None,
                set_above=None, set_common=False, be_quiet=False,
                block_rows=2048, block_cols=2048, out_storage='float32', n_jobs=1):

    """
    Computes statistics on n-dimensions

    Args:
        input_image (str): The (bands x rows x columns) input image to process.
        output_image (str): The output image.
        stat (Optional[str]): The statistic to calculate. Default is 'mean'.
            Choices are ['min', 'max', 'mean', 'median', 'mode', 'var', 'std', 'cv', 'sum'].
        bands2process (Optional[int or int list]): The bands to include in the statistics. Default is -1, or
            include all bands.
        ignore_value (Optional[int]): A value to ignore in the calculations. Default is None.
        no_data_value (Optional[int]): A no data value to set in ``output_image``. Default is 0.
        set_below (Optional[int]): Set values below ``set_below`` to ``no_data_values``. Default is None.
        set_above (Optional[int]): Set values above ``set_above`` to ``no_data_values``. Default is None.
        set_common (Optional[bool]): Whether to set threshold values to the common pixel among all bands.
            Default is False.
        be_quiet (Optional[bool]): Whether to be quiet and do not report progress status. Default is False.
        out_storage (Optional[str]): The output raster storage. Default is 'float32'.
        n_jobs (Optional[int]): The number of blocks to process in parallel. Default is 1.

    Examples:
        >>> import mappy as mp
        >>>
        >>> # Coefficient of variation on all dimensions.
        >>> mp.pixel_stats('/image.tif', '/output.tif', stat='cv')
        >>>
        >>> # Calculate the mean of the first 3 bands, ignoring zeros, and
        >>> #   set the output no data pixels as -999.
        >>> mp.pixel_stats('/image.tif', '/output.tif', stat='mean', \
        >>>                bands2process=[1, 2, 3], ignore_value=0, \
        >>>                no_data_value=-999)

    Returns:
        None, writes to ``output_image``.
    """

    if stat not in ['min', 'max', 'mean', 'median', 'mode', 'var', 'std', 'cv', 'sum']:
        raise NameError('{} is not an option.'.format(stat))

    i_info = rinfo(input_image)

    info_list = [input_image]

    if isinstance(bands2process, list):
        bands2process = [bands2process]
    elif isinstance(bands2process, int):

        if bands2process == -1:
            bands2process = [range(1, i_info.bands+1)]
        else:
            bands2process = [bands2process]

    if i_info.bands <= 1:
        raise ValueError('The input image only has {:d} band. It should have at least 2.'.format(i_info.bands))

    # Copy the input information.
    o_info = i_info.copy()
    o_info.update_info(bands=1, storage=out_storage)

    stats_functions = dict(nanmean=bn.nanmean, nanmedian=bn.nanmedian, nanvar=bn.nanvar,
                           nanstd=bn.nanstd, nanmin=bn.nanmin, nanmax=bn.nanmax,
                           nansum=bn.nansum, median=np.median, mode=sci_mode)

    params = dict(ignore_value=ignore_value, stat=stat,
                  stats_functions=stats_functions, set_below=set_below,
                  set_above=set_above, set_common=set_common,
                  no_data_value=no_data_value)

    bp = BlockFunc(stats_func, info_list, output_image, o_info,
                   proc_info=i_info,
                   print_statement='\nGetting pixel stats for {} ...\n'.format(input_image),
                   d_type='float32', be_quiet=be_quiet, band_list=bands2process,
                   n_jobs=n_jobs, block_rows=block_rows, block_cols=block_cols, **params)

    bp.run()


# def hist_equalization(img, n_bins=256):
#
#     """
#     Computes histogram equalization on an image array
#
#     Args:
#         img (ndarray)
#         n_bins (Optional[int])
#
#     Returns:
#         Histogram equalized image & normalized cumulative distribution function
#     """
#
#     rows, cols = img.shape
#
#     imhist, bins = np.histogram(img.flat, n_bins, normed=True)  # get image histogram
#     cdf = imhist.cumsum()                               # cumulative distribution function
#     cdf = 255 * cdf / cdf[-1]                           # normalize
#
#     img = np.interp(img.flat, bins[:-1], cdf)           # use linear interpolation of cdf to find new pixel values
#
#     img = img.reshape(rows, cols)                        # reshape
#
#     return img, cdf


def match_histograms(source_array, target_hist, n_bins):

    image_rows, image_cols = source_array.shape

    source_array_flat = source_array.flatten()

    hist1, bins = np.histogram(source_array_flat, n_bins, range=[1, 255])

    # Cumulative distribution function.
    cdf1 = hist1.cumsum()
    cdf2 = target_hist.cumsum()

    # Normalize
    cdf1 = (255. * cdf1 / cdf1[-1]).astype(np.uint8)
    cdf2 = (255. * cdf2 / cdf2[-1]).astype(np.uint8)

    # cdf_m = np.ma.masked_equal(cdf,0)
    # 2 cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    # 3 cdf = np.ma.filled(cdf_m,0).astype('uint8')

    matched_image = np.interp(source_array_flat, bins[:-1], cdf1)

    matched_image = np.interp(matched_image, cdf2, bins[:-1]).reshape(image_rows, image_cols)

    matched_image[source_array == 0] = 0

    return matched_image, hist1


def fill_ref_histogram(image_array, n_bins):

    source_array_flat = image_array.flatten()

    hist, __ = np.histogram(source_array_flat, n_bins, range=[1, 255])

    return hist


def histogram_matching(image2adjust, reference_list, output_image, band2match=-1, n_bins=254,
                       overwrite=False, vis_hist=False):

    """
    Adjust one reference image to another using image using histogram matching

    Args:
        image2adjust (str): The image to adjust.
        reference_list (str list): A list of reference images.
        output_image (str): The output adjusted image.
        band2match (Optional[int]): The band or bands to adjust. Default is -1, or all bands.
        n_bins (Optional[int]): The number of bins. Default is 254 (ignores 0).
        overwrite (Optional[bool]): Whether to overwrite an existing ``output_image``. Default is False.
        vis_hist (Optional[bool]): Whether to plot the band histograms. Default is False.

    Returns:
        None, writes to ``output_image``.
    """

    # Open the images
    match_info = rinfo(image2adjust)

    if band2match == -1:
        bands = range(1, match_info.bands+1)
    else:
        bands = [band2match]

    # Copy the input information.
    o_info = match_info.copy()
    o_info.bands = len(bands)

    if overwrite:
        overwrite_file(output_image)

    # Create the output.
    out_rst = create_raster(output_image, o_info)

    color_list = ['r', 'g', 'b', 'o', 'c', 'k', 'y']

    # Match each band.
    for bi, band in enumerate(bands):

        match_array = match_info.mparray(bands2open=band)

        for ri, reference_image in enumerate(reference_list):

            ref_info = rinfo(reference_image)
            ref_array = ref_info.mparray(bands2open=band)

            if ri == 0:
                h2 = fill_ref_histogram(ref_array, n_bins)
            else:
                h2 += fill_ref_histogram(ref_array, n_bins)

        adjusted_array, h1 = match_histograms(match_array, h2, n_bins)

        out_rst.write_array(adjusted_array, band=band)

        out_rst.close_band()

        if vis_hist:
            plt.plot(range(len(h1+1)), [0]+h1, color=color_list[bi], linestyle='-')
            plt.plot(range(len(h2+1)), [0]+h2, color=color_list[bi], linestyle='--')

    # Close the input image.
    ref_info.close()
    match_info.close()

    # Close the output drivers.
    out_rst.close_all()

    out_rst = None

    if vis_hist:
        plt.show()

    plt.close()


def _add_unique_values(segmented_objects):

    object_image_array = np.copy(segmented_objects)

    # binarize
    segmented_objects[segmented_objects > 0] = 1

    # Label the objects, in sequential order.
    objects, n_objects = lab_img(segmented_objects)

    index = np.unique(objects)

    # Get random values.
    random_values = np.random.uniform(2, 255, size=len(index))

    # Here we give each object a random value between 2-255.
    for noi, n_object in enumerate(index):

        if n_object == 0:
            continue

        # Check if any object has been labeled.
        # object_image_array[object_image_array > 1] = object_image_array

        # Give any value <= 1 a random value.
        object_image_array[(object_image_array <= 1) & (objects == n_object)] = int(random_values[noi])

    return object_image_array


def quick_plot(image_arrays, titles=['Field estimates'], colorbar_labels=['ha'], color_maps=['gist_stern'],
               out_fig=None, unique_values=False, dpi=300, font_size=12, colorbar_font_size=7,
               font_face='Calibri', fig_size=(5, 5), image_mins=[None], image_maxes=[None], discrete_list=[],
               class_list=[], layout='by', tile_size=256, clip_limit=1.):

    """
    Args:
        image_array (ndarray list): A list of image arrays to plot.
        titles (Optional[str list]): A list of subplot title labels. Default is ['Field estimates'].
        colorbar_labels (Optional[str list]): A list of colorbar labels. Default is ['ha'].
        color_maps (Optional[str list]): A list of colormaps to plot. Color maps can be found at
            http://matplotlib.org/examples/color/colormaps_reference.html. e.g., 'ocean', 'gist_earth',
            'terrain', 'gist_stern', 'brg', 'cubehelix', 'gnuplot', 'CMRmap'. Default is 'gist_stern'.
            Default is ['gist_stern'].
        out_fig (Optional[str]): An output figure to write to. Default is None.
        unique_values (Optional[bool]): Whether to create unique values for each object. Default is False.
        dpi (Optional[int]): The plot DPI. Default is 300.
        font_size (Optional[int]): The plot font size. Default is 12.
        font_face (Optional[str]): The plot font face type. Default is 'Calibri'.
        fig_size (Optional[int tuple]): The plot figure size (width, height). Default is (5, 5).
        discrete_list (Optional[bool]): Whether the colormap is discrete. Otherwise, continuous. Default is False.
        tile_size (Optional[int]): The tile size (in pixels) for CLAHE. Default is 256.
        clip_limit (Optional[float]): The clip percentage limit for CLAHE. Default is 1.

    Examples:
        >>> import mappy as mp
        >>> from mappy import raster_tools
        >>>
        >>> i_info = mp.rinfo('/image.tif')
        >>> arr = mp.mparray(i_info)
        >>> raster_tools.quick_plot([arr], colorbar_labels=['Hectares'], color_maps=['gist_earth'])
    """

    # set the parameters
    mpl.rcParams['font.size'] = font_size
    mpl.rcParams['font.family'] = font_face
    mpl.rcParams['axes.labelsize'] = font_size  # controls colorbar label size
    mpl.rcParams['xtick.labelsize'] = colorbar_font_size        # controls colorbar tick label size
    mpl.rcParams['ytick.labelsize'] = 9.
    mpl.rcParams['figure.facecolor'] = 'white'
    mpl.rcParams['figure.edgecolor'] = 'white'

    mpl.rcParams['savefig.dpi'] = dpi
    mpl.rcParams['savefig.facecolor'] = 'white'
    mpl.rcParams['savefig.edgecolor'] = 'white'
    mpl.rcParams['savefig.bbox'] = 'tight'
    mpl.rcParams['savefig.pad_inches'] = .05

    if fig_size:
        mpl.rcParams['figure.figsize'] = fig_size[0], fig_size[1]      # width, height

    fig = plt.figure(frameon=False)

    if layout == 'over':
        gs = gridspec.GridSpec(len(image_arrays), 1)
    elif layout == 'by':
        gs = gridspec.GridSpec(1, len(image_arrays))
    else:
        raise NameError('The layout should by "by" or "over".')

    gs.update(hspace=.01)
    gs.update(wspace=.01)

    if not discrete_list or len(discrete_list) != len(image_arrays):
        discrete_list = [False] * len(image_arrays)

    zip_list = [range(0, len(image_arrays)), image_arrays, titles, colorbar_labels, color_maps,
                discrete_list, image_mins, image_maxes]

    for ic, image_array, title, colorbar_label, color_map, discrete, image_min, image_max in zip(*zip_list):

        # ax = fig.add_subplot(1, len(image_arrays), ic)
        if layout == 'over':
            ax = fig.add_subplot(gs[ic, 0])
        else:
            ax = fig.add_subplot(gs[0, ic])

        ax.set_title(title)

        image_shape = image_array.shape

        if unique_values:
            image_array = _add_unique_values(image_array)

        if len(image_shape) > 2:

            for ii, im in enumerate(image_array):

                # im_min = np.percentile(im, 2)
                # im_max = np.percentile(im, 98)

                # image_array[ii] = exposure.rescale_intensity(im,
                #                                     in_range=(mins[ii], maxs[ii]),
                #                                     out_range=(0, 255)).astype(np.uint8)

                # image_array[ii] = exposure.equalize_hist(im)
                image_array[ii] = exposure.equalize_adapthist(np.uint8(im),
                                                              kernel_size=tile_size,
                                                              clip_limit=clip_limit)

            image_array = np.ascontiguousarray(image_array.transpose(1, 2, 0))

        else:

            if isinstance(image_min, int) or isinstance(image_min, float):
                im_min = image_min
            else:
                im_min = np.percentile(image_array, 2)

            if isinstance(image_max, int) or isinstance(image_max, float):
                im_max = image_max
            else:
                im_max = np.percentile(image_array, 98)

            # image_array[image_array < im_min] = 0
            image_array[image_array > im_max] = im_max

        plt.axis('off')

        if len(image_shape) == 2:

            if discrete:

                ip = ax.imshow(image_array)

                if isinstance(color_map, list):
                    color_map = colors.ListedColormap(color_map)
                    # color_map = colorbar.ColorbarBase(ax, cmap=color_map_)
                    ip.set_cmap(color_map)
                elif color_map.lower() == 'random':
                    ip.set_cmap(colors.ListedColormap(np.random.rand(len(class_list), 3)))
                else:
                    ip.set_cmap(_discrete_cmap(len(class_list), base_cmap=color_map))

                ip.set_clim(min(class_list), max(class_list))

            else:

                # my_cmap = cm.gist_stern
                # my_cmap.set_under('#E6E6E6', alpha=1)
                ip = ax.imshow(image_array, vmin=im_min, vmax=im_max, clim=[im_min, im_max])
                # modest_image.imshow(ax, image_array, vmin=im_min, vmax=im_max, clim=[im_min, im_max])
                ip.set_cmap(color_map)
                ip.set_clim(im_min, im_max)

        else:
            ip = ax.imshow(image_array)

        ip.axes.get_xaxis().set_visible(False)
        ip.axes.get_yaxis().set_visible(False)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('bottom', size='3%', pad=.05)

        cbar = plt.colorbar(ip, orientation='horizontal', cax=cax)

        # cbar = plt.colorbar(ip, fraction=0.046, pad=0.04, orientation='horizontal')
        # cbar = plt.colorbar(ip, orientation='horizontal')#ticks=[-1, 0, 1],
        # cbar.ax.set_xticklabels(['Low', 'Medium', 'High'])
        # cbar.set_label(colorbar_label)
        cbar.outline.set_linewidth(0)
        ticklines = cbar.ax.get_xticklines()
        for line in ticklines:
            line.set_visible(False)
        cbar.update_ticks()
        cbar.ax.set_xlabel(colorbar_label)

        # cbar.set_clim(im_min, im_max)

    gs.tight_layout(fig)

    if isinstance(out_fig, str):

        plt.savefig(out_fig)

        plt.clf()

    else:
        plt.show()

    plt.close(fig)


def cumulative_plot_table(table, label_field, data_field='DATA_LYR1', lw_field='MEAN_LYR1',
                          threshold_field='MAX_LYR1', threshold='none', small2large=True, out_fig=None,
                          plot_hist=False, labels2exclude=[], log_data=False, standardize=False,
                          color_by_adm1=False, line_weight_weighting=.00001, **kwargs):

    """
    Plots histograms from table data

    Args:
        table (str): The table with the data.
        label_field (str): The column in ``table`` with the label.
        data_field (Optional[str]): The column in ``table`` with the data. Default is 'DATA_LYR1'.
        lw_field (Optional[str]): The column in ``table`` with the line weights. Default is 'MEAN_LYR1'.
        threshold_field (Optional[str]): The column in ``table`` with the threshold cutoff data.
            Default is 'MAX_LYR1'.
        threshold (Optional[str]): The threshold with ``threshold_field``. Default is 'none'.
        small2large (Optional[bool]): Whether to sort the fields small to large. Default is True.
        out_fig (Optional[str]): The output figure (otherwise pyplot.show). Default is None.
        plot_hist (Optional[bool]): Whether to plot the regular histogram (otherwise cumulative histogram).
            Default is False.
        labels2exclude (Optional[str list]): A list of labels to exclude from the plot. Default is [].
        log_data (Optional[bool]): Whether to log the data. Default is False.
        standardize (Optional[bool]): Whether to standardize the data. Default is False.
        color_by_adm1 (Optional[bool]): Whether to color by ADM (otherwise random colors). Default is False.

    Examples:
        >>> from mappy import raster_tools
        >>> raster_tools.cumulative_plot_table('/PRY_all_bands_fields_centroids_stats_join.csv',
        >>>                                    'NAME_1', labels2exclude=['asuncin', 'central', 'cordillera'])
        >>> # or
        >>> raster_tools.cumulative_plot_table('/PRY_all_bands_fields_centroids_stats_join.csv', 'NAME_1',
        >>>                                    labels2exclude=['asuncin', 'central', 'cordillera'],
        >>>                                    log_data=True)
        >>> # or
        >>> # threshold defines what is shown
        >>> # here, only zones with less than 200 max are shown
        >>> raster_tools.cumulative_plot_table('/zonal_stats.csv', 'UNQ', lw_field='MED_LYR1',
        >>>                                    threshold_field='MAX_LYR1', threshold='<200')
    """

    # Pandas
    try:
        import pandas
    except ImportError:
        raise ImportError('Pandas must be installed')

    from itertools import cycle

    col_gen = cycle('bgrcmk')

    if isinstance(table, str):

        try:
            df = pandas.read_csv(table)
        except:
            df = pandas.read_excel(table)

    else:
        df = table

    if isinstance(out_fig, str):

        dpi = 300
        fig = plt.figure(figsize=(10, 7), dpi=dpi, facecolor='white')
        ax = fig.add_subplot(111, axisbg='white')

    else:

        ax = plt.subplot(111)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    text_positions = []

    if log_data:
        range_value = 1
        xr = .5
        yr1 = .25
        yr2 = .5
    else:
        range_value = 50
        xr = 10
        yr1 = 2.5
        yr2 = 5

    adm1_colors = {'BUENOS AIRES': '#8181F7', 'LA PAMPA': '#2ECCFA', 'CORDOBA': '#7401DF', 'ENTRE RIOS': '#00FF00',
                   'SANTIAGO DEL ESTERO': '#B40404', 'SAN LUIS': '#AEB404', 'MENDOZA': '#FF8000',
                   'RIO NEGRO': '#088A29', 'SANTA FE': '#0B615E'}

    # plot the data
    for di, df_row in df.iterrows():

        x = df_row[data_field].split(',')

        if log_data:
            x = [np.log(float(d)) for d in x]
        else:
            x = [float(d) for d in x]

        if threshold != 'none':

            # get the sign
            the_sign = threshold[0]

            # the threshold
            the_threshold = int(threshold[1:])

            if the_sign == '<':

                if df_row[threshold_field] >= the_threshold:
                    continue

            else:

                if df_row[threshold_field] < the_threshold:
                    continue

        try:
            line_weight = float(df_row[lw_field]) * line_weight_weighting
        except:
            line_weight = 1

        try:
            label = u''.join(df_row[label_field])
        except:
            label = df_row[label_field].decode('utf-8')

        if label.encode('ascii', 'ignore').lower() in labels2exclude:
            continue

        n_area = sum(x)

        y = [(float(n) / n_area) * 100. for n in x]

        y = np.sort(y).cumsum()

        # color = np.random.rand(3,)
        # color = col_gen.next()
        if color_by_adm1:
            try:
                color = adm1_colors[df_row.provincia]
            except:
                color = 'black'
        else:
            color = cm.nipy_spectral(random_float(0, 1))

        if plot_hist:

            h, b = np.histogram(x, bins=1000, range=(1, max(x)), density=True)
            h /= float(max(h))
            ax.plot(b[1:], h)

        else:

            if standardize:
                ax.plot(np.sort(x) / float(max(x)), y, c=color, lw=line_weight, alpha=.5)
            else:
                ax.plot(np.sort(x), y, c=color, lw=line_weight, alpha=.5)

            plot_x_position = np.sort(x)[-1]
            plot_y_position = y[-1]

            if text_positions:

                for text_position in text_positions:

                    if ((plot_x_position - text_position) > -range_value) and \
                            ((plot_x_position - text_position) < range_value):

                        plot_x_position -= xr
                        plot_y_position += yr1

                    elif ((plot_x_position - text_position) >= range_value) and \
                            ((plot_x_position - text_position) < range_value):

                        plot_x_position += xr
                        plot_y_position += yr2

            text_positions.append(plot_x_position)

            ax.text(plot_x_position, plot_y_position, label.title(), color=color, fontsize=20, alpha=.5)

    if not plot_hist:
        ax.set_ylim(0, 100)

    # ax.set_xlim(0, 500)

    plt.tick_params(axis='both',
                    which='both',
                    bottom='off',
                    top='off',
                    labelbottom='on',
                    left='off',
                    right='off',
                    labelleft='on')

    plt.ylabel('Percent of cropland total', fontsize=16)
    plt.xlabel('Field size (ha)', fontsize=16)

    if isinstance(out_fig, str):

        plt.savefig(out_fig, dpi=dpi, bbox_inches='tight', pad_inches=.1)
        plt.clf()

    else:
        plt.show()

    plt.close()


def cumulative_plot_array(image_array, small2large=True, out_fig=None):

    """
    Args:
        image_array (ndarray): Segments are area.
        small2large (Optional[bool]): Whether to sort the x-axis from small to large fields as range,
            otherwise sort by size. Default is True.
        out_fig (Optional[str])
    """

    # SciPy
    try:
        from scipy.ndimage.measurements import label as nd_label
    except ImportError:
        raise ImportError('SciPy must be installed')

    # Scikit-learn
    try:
        from skimage.measure import regionprops
    except ImportError:
        raise ImportError('Scikit-learn must be installed')

    image_shape = image_array.shape

    plot_multiple = False
    if len(image_shape) > 2:
        plot_multiple = True

    if isinstance(out_fig, str):

        dpi = 300
        fig = plt.figure(figsize=(10, 7), dpi=dpi)
        ax = fig.add_subplot(111)

    else:

        ax = plt.subplot(111)

    ax.spines["top"].set_visible(False)
    # ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # ax.spines["left"].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    if plot_multiple:

        cs = []
        xs = []

        for image_band in image_array:

            # create binary
            arr_b = np.where(image_band > 0, 1, 0).astype(np.uint8)
            o, n_o = nd_label(arr_b)
            p = regionprops(o, intensity_image=image_band)

            # convert ha to square kilometers
            l = np.asarray([(pp.max_intensity / 100.) for pp in p]).astype(np.float32).sort()

            n_features = len(list(l))
            n_area = l.sum()

            x = []
            for i in xrange(1, n_features+1):
                x.append((i / n_features) * 100.)

            y = [(n / n_area) * 100. for n in l]

            c = np.sort(l).cumsum()

            if small2large:
                x = np.arange(np.sort(l).size)
            else:
                x = np.sort(l)

            cs.append([c[0], c[-1]])
            xs.append([x[0], x[-1]])

            ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
            ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

            ax.plot(x, c)

            ax.fill_between(x, c, facecolor=np.random.rand(3,), alpha=.5)

            cc = np.multiply(c, 100.)

            i5 = np.percentile(cc, 50)
            for i5_index in xrange(0, len(cc)):
                if i5+5 > cc[i5_index] > i5-5:
                    break

            i75 = np.percentile(cc, 75)
            for i75_index in xrange(0, len(cc)):
                if i75+5 > cc[i75_index] > i75-5:
                    break

            i90 = np.percentile(cc, 90)
            for i90_index in xrange(0, len(cc)):
                if i90+50 > cc[i90_index] > i90-50:
                    break

            bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="white", lw=.5)

            # ax.scatter(x[i5_index], c[i5_index], c='black', s=40, marker='o', edgecolor='black')
            # ax.text(x[i5_index], c[i5_index], '50%', size=10, bbox=bbox_props)
            #
            # ax.scatter(x[i75_index], c[i75_index], c='black', s=40, marker='o', edgecolor='black')
            # ax.text(x[i75_index], c[i75_index], '75%', size=10, bbox=bbox_props)

            ax.stem([x[i90_index]], [c[i90_index]], linefmt='b-.', markerfmt='bo')

        plt.tick_params(axis="both",
                        which="both",
                        bottom="off",
                        top="off",
                        labelbottom="on",
                        left="off",
                        right="off",
                        labelleft="on")

        min_c, max_c, min_x, max_x = 0, 0, 0, 0

        for c, x in zip(cs, xs):

            min_c = min(c[0], min_c)
            max_c = max(c[1], max_c)

            min_x = min(x[0], min_x)
            max_x = max(x[1], max_x)

        plt.ylim(min_c, max_c)
        plt.xlim(min_x, max_x)

        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)

    else:

        # create binary
        arr_b = np.where(image_array > 0, 1, 0).astype(np.uint8)
        o, n_o = nd_label(arr_b)
        p = regionprops(o, intensity_image=image_array)

        # convert ha to square kilometers
        l = np.asarray([(pp.max_intensity / 100.) for pp in p])

        c = np.sort(l).cumsum()

        if small2large:
            x = np.arange(np.sort(l).size)
        else:
            x = np.sort(l)

        plt.ylim(c[0], c[-1])
        plt.xlim(x[0], x[-1])

        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)

        plt.tick_params(axis="both",
                        which="both",
                        bottom="off",
                        top="off",
                        labelbottom="on",
                        left="off",
                        right="off",
                        labelleft="on")

        ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

        ax.plot(x, c)

        ax.fill_between(x, c, facecolor='#3104B4', alpha=.7)#np.random.rand(3,))

        # h5 = np.percentile(c, 50)
        # i5 = np.where(c == h5)
        i5 = len(c) / 2
        i25 = int(len(c) * .25)
        i75 = int(len(c) * .75)

        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="white", lw=.5)

        ax.scatter(x[i5], c[i5], c='black', s=40, marker='o', edgecolor='black')
        ax.text(x[i5]-100, c[i5]-100, '50%', size=10, bbox=bbox_props)

        ax.scatter(x[i25], c[i25], c='black', s=40, marker='o', edgecolor='black')
        ax.text(x[i25]-10, c[i25]-10, '25%', size=10, bbox=bbox_props)

        ax.scatter(x[i75], c[i75], c='black', s=40, marker='o', edgecolor='black')
        ax.text(x[i75]-10, c[i75]-10, '75%', size=10, bbox=bbox_props)

        # ax2 = ax.twinx()
        # ax2.hist(c, 100, color="#3F5D7D", alpha=.7)

    if small2large:
        plt.xlabel('Sorted fields, small to large (order)', fontsize=16)
    else:
        plt.xlabel('Sorted fields, small to large (Square km)', fontsize=16)

    plt.ylabel('Square km', fontsize=16)

    if isinstance(out_fig, str):

        plt.savefig(out_fig, dpi=dpi, bbox_inches='tight', pad_inches=.1)
        plt.clf()

    else:
        plt.show()

    plt.close()


def rasterize_vector(in_vector, out_raster, burn_id='Id', cell_size=None, storage='float32',
                     match_raster=None, bigtiff='no', in_memory=False, **kwargs):

    """
    Rasterizes a vector dataset

    Args:
        in_vector (str): The vector to rasterize.
        out_raster (str): The output image.
        burn_id (Optional[str]): The attribute id of ``in_vector`` to burn into ``out_raster``. Default is 'Id'.
        cell_size (Optional[float]): The output raster cell size. Default is None. *Needs to be given if
            ``match_raster``=None.
        match_raster (Optional[str]): A raster to match cell size. Default is None.
        bigtiff (Optional[str]): How to handle big TIFF creation option. Default is 'no'.
        in_memory (Optional[bool]): Whether to build ``out_raster`` in memory. Default is False.

    Examples:
        >>> # rasterize to the extent of the matching raster
        >>> from mappy.raster_tools import rasterize_vector
        >>>
        >>> rasterize_vector('/in_vector.shp', 'out_image.tif',
        >>>                  match_raster='/some_image.tif')
        >>>
        >>> # rasterize to the extent of the input vector
        >>> rasterize_vector('/in_vector.shp', 'out_image.tif',
        >>>                  burn_id='UNQ', cell_size=30.)
        >>>
        >>> # rasterize to a given extent
        >>> rasterize_vector('/in_vector.shp', 'out_image.tif', burn_id='UNQ',
        >>>                  cell_size=30., top=10000, bottom=5000, left=-8000,
        >>>                  right=-2000, projection='')

    Returns:
        None, writes to ``out_raster``.
    """

    v_info = vinfo(in_vector)

    if match_raster:

        o_info = rinfo(match_raster)

    elif kwargs:

        if not isinstance(cell_size, float):
            raise ValueError('The cell size must be given.')

        if 'projection' not in kwargs:
            raise ValueError('The projection must be given.')

        if 'right' not in kwargs:

            if 'cols' not in kwargs:
                raise ValueError('Either right or cols must be given.')

            kwargs['right'] = kwargs['left'] + (kwargs['cols'] * cell_size) + cell_size

        if 'bottom' not in kwargs:

            if 'rows' not in kwargs:
                raise ValueError('Either bottom or rows must be given.')

            kwargs['bottom'] = kwargs['top'] - (kwargs['rows'] * cell_size) - cell_size

        # get rows and columns
        if 'rows' not in kwargs:

            if (kwargs['top'] > 0) and (kwargs['bottom'] >= 0):
                kwargs['rows'] = int((kwargs['top'] - kwargs['bottom']) / cell_size)
            elif (kwargs['top'] > 0) and (kwargs['bottom'] < 0):
                kwargs['rows'] = int((kwargs['top'] + abs(kwargs['bottom'])) / cell_size)
            elif (kwargs['top'] < 0) and (kwargs['bottom'] < 0):
                kwargs['rows'] = int((abs(kwargs['bottom']) - abs(kwargs['top'])) / cell_size)

        if 'cols' not in kwargs:

            if (kwargs['right'] > 0) and (kwargs['left'] >= 0):
                kwargs['cols'] = int((kwargs['right'] - kwargs['left']) / cell_size)
            elif (kwargs['right'] > 0) and (kwargs['left'] < 0):
                kwargs['cols'] = int((kwargs['right'] + abs(kwargs['left'])) / cell_size)
            elif (kwargs['right'] < 0) and (kwargs['left'] < 0):
                kwargs['cols'] = int((abs(kwargs['left']) - abs(kwargs['right'])) / cell_size)

        o_info = rinfo('create', left=kwargs['left'], right=kwargs['right'], top=kwargs['top'],
                       bottom=kwargs['bottom'], projection=kwargs['projection'], storage=storage, bands=1,
                       cellY=cell_size, cellX=-cell_size, rows=kwargs['rows'], cols=kwargs['cols'])

    else:

        if not isinstance(cell_size, float):
            raise ValueError('The cell size must be given.')

        # get rows and columns
        rows = abs(int((abs(v_info.top) - abs(v_info.bottom)) / cell_size))
        cols = abs(int((abs(v_info.left) - abs(v_info.right)) / cell_size))

        o_info = rinfo('create', left=v_info.left, right=v_info.right, top=v_info.top, bottom=v_info.bottom,
                       proj=v_info.projection, storage=storage, bands=1, cellY=cell_size, cellX=-cell_size,
                       rows=rows, cols=cols)

    orw = create_raster(out_raster, o_info, bigtiff=bigtiff, in_memory=in_memory)

    # raster dataset, band(s) to rasterize, vector layer to rasterize,
    # burn a specific value, or values, matching the bands :: burn_values=[100]

    gdal.RasterizeLayer(orw.datasource, [1], v_info.lyr, options=['ATTRIBUTE={}'.format(burn_id)])

    if in_memory:
        return orw
    else:
        orw.close_file()
        return None


def batch_manage_overviews(image_directory, build=True, image_extensions=['tif'], wildcard=None):

    """
    Creates images overviews for each image in a directory

    Args:
        image_directory (str): The directory to search in.
        build (Optional[bool]): Whether to build overviews (otherwise, remove overviews). Default is True.
        image_extensions (Optional[str list]): A list of image extensions to limit the search to. Default is ['tif'].
        wildcard (Optional[str]): A wildcard search parameter to limit the search to. Default is None.

    Examples:
        >>> import mappy as mp
        >>>
        >>> # build overviews
        >>> mp.batch_manage_overviews('/image_directory', wildcard='p224*')
        >>>
        >>> # remove overviews
        >>> mp.batch_manage_overviews('/image_directory', build=False, wildcard='p224*')

    Returns:
        None, builds overviews in place for each image in ``image_directory``.
    """

    image_list = os.listdir(image_directory)

    image_extensions = ['*.{}'.format(se) for se in image_extensions]

    images_filtered = []
    for se in image_extensions:
        [images_filtered.append(fn) for fn in fnmatch.filter(image_list, se)]

    if isinstance(wildcard, str):
        images_filtered = fnmatch.filter(images_filtered, wildcard)

    for image in images_filtered:

        if build:
            info = rinfo('{}/{}'.format(image_directory, image))
            info.build_overviews()
        else:
            info = rinfo('{}/{}'.format(image_directory, image), open2read=False)
            info.remove_overviews()

        info.close()


def _discrete_cmap(n_classes, base_cmap='cubehelix'):

    """
    @original author: Jake VanderPlas
    License: BSD-style

    Creates an N-bin discrete colormap from the specified input map

    Args:
        n_classes (int): The number of classes in the colormap.
        base_cmap (Optional[str]): The colormap to use. Default is 'cubehelix'.
    """

    if not isinstance(n_classes, int):
        raise ValueError('\nThe number of classes must be given as an integer.\n')

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, n_classes))
    cmap_name = base.name + str(n_classes)

    return base.from_list(cmap_name, color_list, n_classes)


def _examples():

    sys.exit("""\

    # Get basic image information
    raster_tools.py -i /image.tif --method info

    # Compute the variance over all bands
    raster_tools.py -i /image.tif -o /output.tif --stat var

    # Compute the average over three bands
    raster_tools.py -i /image.tif -o /output.tif --stat mean --bands 1 2 3

    # Set pixels with variance to 0 and keep the common value among all layers.
    raster_tools.py -i /image.tif -o /output.tif --stat var --set-above 0 --set-common --out-storage int16 --no-data -1

    # Compute the majority value among all bands
    raster_tools.py -i /image.tif -o /output.tif --stat mode

    """)


def main():

    parser = argparse.ArgumentParser(description='Raster tools',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-e', '--examples', dest='examples', action='store_true', help='Show usage examples and exit')
    parser.add_argument('-i', '--input', dest='input', help='The input image', default=None)
    parser.add_argument('-o', '--output', dest='output', help='The output image', default=None)
    parser.add_argument('-m', '--method', dest='method', help='The method to run', default='pixel-stats',
                        choices=['info', 'pixel-stats'])
    parser.add_argument('-b', '--bands', dest='bands', help='A list of bands to process', default=[-1], nargs='+',
                        type=int)
    parser.add_argument('--stat', dest='stat', help='The statistic to compute', default='mean',
                        choices=['min', 'max', 'mean', 'median', 'mode', 'var', 'std', 'cv', 'sum'])
    parser.add_argument('--ignore', dest='ignore', help='A value to ignore', default=None, type=int)
    parser.add_argument('--no-data', dest='no_data', help='The output no data value', default=0, type=int)
    parser.add_argument('--set-above', dest='set_above', help='Set values above threshold to no-data', default=None,
                        type=int)
    parser.add_argument('--set-below', dest='set_below', help='Set values below threshold to no-data', default=None,
                        type=int)
    parser.add_argument('--set-common', dest='set_common',
                        help='Set values above or below thresholds to common values among all bands',
                        action='store_true')
    parser.add_argument('--out-storage', dest='out_storage', help='The output raster storage', default='float32')

    args = parser.parse_args()

    if args.examples:
        _examples()

    print('\nStart date & time --- (%s)\n' % time.asctime(time.localtime(time.time())))

    start_time = time.time()

    if args.method == 'info':

        i_info = rinfo(args.input)

        print '\nThe projection:\n'
        print i_info.projection

        print '\n======================================\n'

        print 'The extent (left, right, top, bottom):\n'
        print '{:f}, {:f}, {:f}, {:f}'.format(i_info.left, i_info.right, i_info.top, i_info.bottom)

        storage_string = 'The data type: {}\n'.format(i_info.storage)

        print '\n{}\n'.format(''.join(['=']*(len(storage_string)-1)))

        print storage_string

        print '=========\n'

        print 'The size:\n'
        print '{:,d} rows'.format(i_info.rows)
        print '{:,d} columns'.format(i_info.cols)

        if i_info.bands == 1:
            print '{:,d} band'.format(i_info.bands)
        else:
            print '{:,d} bands'.format(i_info.bands)

        print '{:.2f} meter cell size'.format(i_info.cellY)

        i_info.close()

    elif args.method == 'pixel-stats':

        pixel_stats(args.input, args.output, stat=args.stat, bands2process=args.bands,
                    ignore_value=args.ignore, no_data_value=args.no_data,
                    set_below=args.set_below, set_above=args.set_above,
                    set_common=args.set_common, out_storage=args.out_storage)

    print('\nEnd data & time -- (%s)\nTotal processing time -- (%.2gs)\n' %
          (time.asctime(time.localtime(time.time())), (time.time()-start_time)))

if __name__ == '__main__':
    main()
