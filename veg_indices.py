#!/usr/bin/env python

"""
@authors: Jordan Graesser, Jordan Long
Date Created: 9/24/2011
"""

from __future__ import division

import os
import sys
import subprocess
import time
from copy import copy
import multiprocessing as mpr
import argparse
import fnmatch
from collections import OrderedDict

# MapPy
import raster_tools
# from mappy.utilities import composite
from helpers.other.progress_iter import _iteration_parameters

# Numpy    
try:
    import numpy as np
except ImportError:
    raise ImportError('NumPy must be installed')

# Numexpr
try:
    import numexpr as ne
    ne.set_num_threads(mpr.cpu_count())
    numexpr_installed = True
except:
    numexpr_installed = False

# Carray
# try:
#     import carray as ca
#     carray_installed = True
# except:
#     carray_installed = False

# GDAL
try:
    from osgeo import gdal
    from osgeo.gdalconst import *
except ImportError:
    raise ImportError('GDAL must be installed')

# Scikit-image
try:
    from skimage.exposure import rescale_intensity
except ImportError:
    raise ImportError('Scikit-image must be installed')

old_settings = np.seterr(all='ignore')


class SensorInfo(object):

    """
    A class to hold sensor names, wavelengths, and equations.
    """

    def __init__(self):

        self.sensors = ['ASTER', 'ASTER-VNIR', 'CBERS2', 'CitySphere', 'GeoEye1', 'IKONOS',
                        'Landsat', 'Landsat8', 'Landsat-thermal', 'MODIS', 'RapidEye',
                        'Sentinel2', 'Sentinel2-10m', 'Sentinel2-20m', 'Quickbird', 'WorldView2',
                        'RGB', 'BGR']

        # Landsat 8
        #   coastal blue, blue, gree, red, NIR, SWIR 1, SWIR 2, cirrus
        #   skips panchromatic (otherwise 8) and 2 thermal bands
        #
        # Sentinel2
        #   The full, 10 (10m + 20m) or 13 (10m + 20m + 60m) band image.
        #   blue, green, red, NIR, --, --, --, red edge, MidIR, FarIR
        #   2, 3, 4, 8, 5, 6, 7, 8A, 11, 12, 1, 9, 10
        self.band_orders = {'ASTER': {'green': 1, 'red': 2, 'nir': 3, 'midir': 4, 'farir': 5},
                            'ASTER-VNIR': {'green': 1, 'red': 2, 'nir': 3},
                            'CBERS': {'blue': 1, 'green': 2, 'red': 3, 'nir': 4},
                            'CBERS2': {'blue': 1, 'green': 2, 'red': 3, 'nir': 4},
                            'CitySphere': {'blue': 3, 'green': 2, 'red': 1},
                            'GeoEye1': {'blue': 1, 'green': 2, 'red': 3, 'nir': 4},
                            'IKONOS': {'blue': 1, 'green': 2, 'red': 3, 'nir': 4},
                            'Quickbird': {'blue': 1, 'green': 2, 'red': 3, 'nir': 4},
                            'WorldView2': {'cblue': 1, 'blue': 2, 'green': 3, 'yellow': 4, 'red': 5,
                                           'rededge': 6, 'nir': 7, 'midir': 8},
                            'Landsat': {'blue': 1, 'green': 2, 'red': 3, 'nir': 4, 'midir': 5, 'farir': 6},
                            'Landsat8': {'cblue': 1, 'blue': 2, 'green': 3, 'red': 4, 'nir': 5, 'midir': 6,
                                         'farir': 7, 'cirrus': 8},
                            'Landsat-thermal': {'blue': 1, 'green': 2, 'red': 3, 'nir': 4, 'midir': 5, 'farir': 7},
                            'MODIS': {'blue': 3, 'green': 4, 'red': 1, 'nir': 2, 'midir': 6, 'farir': 7},
                            'RapidEye': {'blue': 1, 'green': 2, 'red': 3, 'rededge': 4, 'nir': 5},
                            'Sentinel2': {'cblue': 1, 'blue': 2, 'green': 3, 'red': 4, 'rededge': 5, 'niredge': 8,
                                          'nir': 9, 'midir': 12, 'farir': 13},
                            'Sentinel2-10m': {'blue': 1, 'green': 2, 'red': 3, 'nir': 4},
                            'Sentinel2-20m': {'rededge': 1, 'niredge': 4, 'midir': 5, 'farir': 6},
                            'RGB': {'blue': 3, 'green': 2, 'red': 1},
                            'BGR': {'blue': 1, 'green': 2, 'red': 3}}

        # The wavelengths needed to compute the index.
        # The wavelengths are loaded in order, so the
        #   order should match the equations in
        #   ``self.equations``.
        self.wavelength_lists = {'ARVI': ['blue', 'red', 'nir'],
                                 'CBI': ['cblue', 'blue'],
                                 'EVI': ['blue', 'red', 'nir'],
                                 'EVI2': ['red', 'nir'],
                                 'IPVI': ['red', 'nir'],
                                 'GNDVI': ['green', 'nir'],
                                 'MNDWI': ['midir', 'green'],
                                 'MSAVI': ['red', 'nir'],
                                 'NDSI': ['nir', 'midir'],
                                 'NDBAI': ['midir', 'farir'],
                                 'NDVI': ['red', 'nir'],
                                 'RENDVI': ['rededge', 'nir'],
                                 'ONDVI': ['red', 'nir'],
                                 'NDWI': ['nir', 'green'],
                                 'PNDVI': ['green', 'red'],
                                 'RBVI': ['blue', 'red'],
                                 'GBVI': ['blue', 'green'],
                                 'SATVI': ['red', 'midir', 'farir'],
                                 'SAVI': ['red', 'nir'],
                                 'OSAVI': ['red', 'nir'],
                                 'SVI': ['red', 'nir'],
                                 'TNDVI': ['red', 'nir'],
                                 'TVI': ['green', 'nir'],
                                 'YNDVI': ['yellow', 'nir'],
                                 'VCI': ['red', 'nir']}

        # The vegetation index equations. The arrays are
        #   loaded from ``self.wavelength_lists``. For example,
        #   ``array01`` of 'ARVI' would be the 'blue' wavelength.
        self.equations = \
            {'ARVI': '(array03 - (array02 - y*(array01 - array02))) / (array03 + (array02 - y*(array01 - array02)))',
             'CBI': '(array02 - array01) / (array02 + array01)',
             'EVI2': '((array02 - array01) / ((array02 + (c1 * array01)) + c2)) * c3',
             'IPVI': 'array02 / (array02 + array01)',
             'MSAVI': '((2 * array02 + 1) - ((((2 * array02 + 1)**2) - (8 * (array02 - array01)))**.5)) / 2',
             'GNDVI': '(array02 - array01) / (array02 + array01)',
             'MNDWI': '(array02 - array01) / (array02 + array01)',
             'NDSI': '(array02 - array01) / (array02 + array01)',
             'NDBAI': '(array02 - array01) / (array02 + array01)',
             'NDVI': '(array02 - array01) / (array02 + array01)',
             'RENDVI': '(array02 - array01) / (array02 + array01)',
             'NDWI': '(array02 - array01) / (array02 + array01)',
             'PNDVI': '(array02 - array01) / (array02 + array01)',
             'RBVI': '(array02 - array01) / (array02 + array01)',
             'GBVI': '(array02 - array01) / (array02 + array01)',
             'ONDVI': '(4. / pi) * arctan((array02 - array01) / (array02 + array01))',
             'SATVI': '(((array02 - array01) / (array02 + array01 + L)) * (1. + L)) - (array03 / 2.)',
             'SAVI': '((array02 - array01) / (array02 + array01 + L)) * (1. + L)',
             'OSAVI': 'arctan((((array02 - array01) / (array02 + array01 + L)) * (1. + L)) / 1.5) * 2.',
             'SVI': 'array02 / array01',
             'TNDVI': 'sqrt(((array02 - array01) / (array02 + array01)) * .5)',
             'TVI': 'sqrt(((array02 - array01) / (array02 + array01)) + .5)',
             'YNDVI': '(array02 - array01) / (array02 + array01)',
             'VCI': '(((array02 - array01) / (array02 + array01)) - min_ndvi) / (max_ndvi - min_ndvi)'}

        # The data ranges for scaling, but only
        #   used if the output storage type is not
        #   equal to 'float32'.
        self.data_ranges = {'ARVI': (),
                            'CBI': (-1., 1.),
                            'EVI2': (-1., 2.),
                            'IPVI': (),
                            'MSAVI': (),
                            'GNDVI': (-1., 1.),
                            'MNDWI': (-1., 1.),
                            'NDSI': (-1., 1.),
                            'NDBAI': (-1., 1.),
                            'NDVI': (-1., 1.),
                            'RENDVI': (-1., 1.),
                            'NDWI': (-1., 1.),
                            'PNDVI': (-1., 1.),
                            'RBVI': (-1., 1.),
                            'GBVI': (-1., 1.),
                            'ONDVI': (),
                            'SATVI': (),
                            'SAVI': (),
                            'OSAVI': (),
                            'SVI': (),
                            'TNDVI': (),
                            'TVI': (),
                            'YNDVI': (-1., 1.),
                            'VCI': ()}

    def list_expected_band_order(self, sensor):

        # Return the dictionary sorted by values
        self.expected_band_order = OrderedDict(sorted(self.band_orders[sensor].items(), key=lambda sbo: sbo[1]))

        print '\nExpected band order for {}:\n'.format(sensor)
        print '  WAVELENGTH  Band'
        print '  ----------  ----'

        sp = ' '

        for w, b in self.expected_band_order.iteritems():

            gap_string = ''

            gap_len = 12 - len(w)
            for gx in xrange(0, gap_len):
                gap_string += sp

            print '  {}{}{:d}'.format(w.upper(), gap_string, b)

        print

    def list_indice_options(self, sensor):

        """
        Lists the vegetation indices that can be computed from the given sensor.

        Args:
            sensor (str): The sensor.
        """

        if sensor not in self.sensors:
            raise NameError('{} not a sensor option. Choose one of {}'.format(sensor, ', '.join(self.sensors)))

        self.sensor_indices = []

        # A list of wavelengths in the
        #   current sensor.
        sensor_wavelengths = self.band_orders[sensor].keys()

        # All of the vegetation index wavelengths must
        #   be in the sensor wavelength.
        for veg_index, indice_wavelengths in self.wavelength_lists.iteritems():

            if set(indice_wavelengths).issubset(sensor_wavelengths):

                self.sensor_indices.append(veg_index)


class VegIndicesEquations(SensorInfo):

    """
    A class to compute vegetation indices

    Args:
        image_array (ndarray)
        no_data (Optional[int]): An output 'no data' value. Overflows and NaNs are filled with ``no_data``.
            Default is 0.
        chunk_size (Optional[int]): The chunk size to determine whether to use ``ne.evaluate``. Default is -1, or
            use ``numexpr``.
    """

    def __init__(self, image_array, no_data=0, chunk_size=-1):

        self.image_array = image_array
        self.no_data = no_data
        self.chunk_size = chunk_size

        SensorInfo.__init__(self)

        try:
            self.array_dims, self.array_rows, self.array_cols = image_array.shape
        except:
            raise ValueError('The input array must be at least 3d.')

    def rescale_range(self, array2rescale, in_range=()):

        if self.out_type > 3:
            raise ValueError('The output type cannot be greater than 3.')

        if self.out_type == 2:

            if in_range:

                array2rescale_ = rescale_intensity(array2rescale,
                                                   in_range=in_range,
                                                   out_range=(0, 254)).astype(np.uint8)

            else:
                array2rescale_ = rescale_intensity(array2rescale, out_range=(0, 254)).astype(np.uint8)

        elif self.out_type == 3:

            if in_range:

                array2rescale_ = rescale_intensity(array2rescale,
                                                   in_range=in_range,
                                                   out_range=(0, 9999)).astype(np.uint16)

            else:

                array2rescale_ = rescale_intensity(array2rescale, out_range=(0, 9999)).astype(np.uint16)

        return np.where(array2rescale == self.no_data, self.no_data, array2rescale_)

    def compute(self, index2compute, out_type=1, **kwargs):

        """
        Args:
            index2compute (str): The vegetation index to compute.
            out_type (Optional[int]): This controls the output scaling. Default is 1, or return 'as is'. Choices
                are [1, 2, 3].

                1 = raw values (float32)
                2 = scaled (byte)
                3 = scaled (uint16)

        Example:
            >>> from mappy.features import VegIndicesEquations
            >>>
            >>> # Create a fake 2-band array.
            >>> image_stack = np.random.randn(2, 100, 100, dtype='float32')
            >>>
            >>> # Setup the vegetation index object.
            >>> vie = VegIndicesEquations(image_stack)
            >>>
            >>> # Calculate the NDVI vegetation index.
            >>> ndvi = vie.compute('NDVI')
        """

        self.index2compute = index2compute
        self.out_type = out_type

        self.n_bands = len(self.wavelength_lists[self.index2compute.upper()])

        # Use ``numexpr``.
        if self.chunk_size == -1:

            if kwargs:
                return self.run_index(**kwargs)
            else:
                return self.run_index()

        else:

            vi_functions = {'ARVI': self.ARVI,
                            'CBI': self.CBI,
                            'EVI': self.EVI,
                            'EVI2': self.EVI2,
                            'IPVI': self.IPVI,
                            'GNDVI': self.GNDVI,
                            'MNDWI': self.MNDWI,
                            'MSAVI': self.MSAVI,
                            'NDSI': self.NDSI,
                            'NDBAI': self.NDBAI,
                            'NDVI': self.NDVI,
                            'RENDVI': self.RENDVI,
                            'ONDVI': self.ONDVI,
                            'NDWI': self.NDWI,
                            'PNDVI': self.PNDVI,
                            'RBVI': self.RBVI,
                            'GBVI': self.GBVI,
                            'SATVI': self.SATVI,
                            'SAVI': self.SAVI,
                            'OSAVI': self.OSAVI,
                            'SVI': self.SVI,
                            'TNDVI': self.TNDVI,
                            'TVI': self.TVI,
                            'YNDVI': self.YNDVI,
                            'VCI': self.VCI}

            if self.index2compute.upper() not in vi_functions:
                raise NameError('{} is not a vegetation index option.'.format(self.index2compute))

            vi_function = vi_functions[self.index2compute.upper()]

            if kwargs:
                return vi_function(kwargs)
            else:
                return vi_function()

    def run_index(self, y=1., c1=2.4, c2=1., c3=2.5, L=.5, min_ndvi=-1, max_ndvi=1):

        no_data = self.no_data
        pi = np.pi

        if self.n_bands == 2:

            try:
                array01 = self.image_array[0]
                array02 = self.image_array[1]
            except:
                raise ValueError('\nThe input array should have {:d} dimensions.\n'.format(self.n_bands))

            mask_equation = 'where((array01 == 0) | (array02 == 0), no_data, index_array)'

        elif self.n_bands == 3:

            try:
                array01 = self.image_array[0]
                array02 = self.image_array[1]
                array03 = self.image_array[2]
            except:
                raise ValueError('\nThe input array should have {:d} dimensions.\n'.format(self.n_bands))

            mask_equation = 'where((array01 == 0) | (array02 == 0) | (array03 == 0), no_data, index_array)'

        index_array = ne.evaluate(self.equations[self.index2compute.upper()])

        if self.data_ranges[self.index2compute.upper()] == (-1, 1):
            index_array = ne.evaluate('where(index_array < -1, -1, index_array)')
            index_array = ne.evaluate('where(index_array > 1, 1, index_array)')

        index_array = ne.evaluate(mask_equation)

        index_array[np.isinf(index_array) | np.isnan(index_array)] = self.no_data

        if self.out_type > 1:
            index_array = self.rescale_range(index_array, in_range=self.data_ranges[self.index2compute.upper()])

        return index_array

    def ARVI(self, y=1):

        """
        Atmospherically Resistant Vegetation Index (ARVI)

        Equation:
            (nir - rb) / (nir + rb)
                where, rb = red - y(blue - red)
                    where, y = gamma value (weighting factor depending on aersol type), (0.7 to 1.3)
        """

        try:
            blue = self.image_array[0]
            red = self.image_array[1]
            nir = self.image_array[2]
        except:
            raise ValueError('\nThe input array should have {:d} dimensions.\n'.format(self.n_bands))

        rb1 = np.multiply(np.subtract(blue, red), y)
        rb = np.subtract(red, rb1)

        arvi = self.NDVI()

        arvi[(blue == 0) | (red == 0) | (nir == 0)] = self.no_data

        arvi[np.isinf(arvi) | np.isnan(arvi)] = self.no_data

        if self.out_type > 1:
            arvi = self.rescale_range(arvi)

        return arvi

    def CBI(self):

        """
        Coastal-Blue Index

        Equation:
            CBI = (blue - cblue) / (blue + cblue)
        """

        try:
            cblue = self.image_array[0]
            blue = self.image_array[1]
        except:
            raise ValueError('\nThe input array should have {:d} dimensions.\n'.format(self.n_bands))

        cbi = self.main_index(cblue, blue)

        cbi[(cblue == 0) | (blue == 0)] = self.no_data

        cbi[np.isinf(cbi) | np.isnan(cbi)] = self.no_data

        if self.out_type > 1:
            cbi = self.rescale_range(cbi, in_range=(-1., 1.))

        return cbi

    def EVI(self, C1=6., C2=7.5, L=1.):

        """
        Enhanced Vegetation Index (EVI)

        Equation:
            2.5 * [         nir - Red
                    ------------------------------
                    nir + C1 * Red - C2 * Blue + L

                    ]

                    C1 = 6
                    C2 = 7.5
                    L = 1
        """

        try:
            blue = self.image_array[0]
            red = self.image_array[1]
            nir = self.image_array[2]
        except:
            raise ValueError('\nThe input array should have {:d} dimensions.\n'.format(self.n_bands))

        top = np.subtract(nir, red)
        red_c1 = np.multiply(C1, red)
        blue_c2 = np.multiply(C2, blue)
        bottom = np.add(np.add(np.subtract(red_c1, blue_c2), nir), L)

        evi = np.divide(top, bottom)
        evi = np.multiply(evi, 2.5)

        evi[(blue == 0) | (red == 0) | (nir == 0)] = self.no_data

        evi[np.isinf(evi) | np.isnan(evi)] = self.no_data

        if self.out_type > 1:
            evi = self.rescale_range(evi)

        return evi

    def EVI2(self, c1=2.4, c2=1., c3=2.5):

        """
        Enhanced Vegetation Index (EVI2)

        Reference:
            Jiang, Zhangyan, Alfredo R. Huete, Kamel Didan, and Tomoaki Miura. 2008. "Development of a
                two-band enhanced vegetation index without a blue band." Remote Sensing of Environment 112: 3833-3845.

        Equation:
            2.5 * [         nir - Red
                    ---------------------
                    nir + (2.4 * Red) + 1
                    ]
        """

        try:
            red = self.image_array[0]
            nir = self.image_array[1]
        except:
            raise ValueError('\nThe input array should have {:d} dimensions.\n'.format(self.n_bands))

        top = np.subtract(nir, red)
        bottom = np.add(np.add(np.multiply(red, c1), nir), c2)

        evi2 = np.divide(top, bottom)
        evi2 = np.multiply(evi2, c3)

        evi2[(red == 0) | (nir == 0)] = self.no_data

        evi2[np.isinf(evi2) | np.isnan(evi2)] = self.no_data

        if self.out_type > 1:
            evi2 = self.rescale_range(evi2, in_range=(-1., 2.))

        return evi2

    def IPVI(self):

        """
        Equation:
            IPVI = nir / (nir + red)
        """

        try:
            red = self.image_array[0]
            nir = self.image_array[1]
        except:
            raise ValueError('\nThe input array should have {:d} dimensions.\n'.format(self.n_bands))

        bottom = np.add(nir, red)

        ipvi = np.divide(nir, bottom)

        ipvi[(red == 0) | (nir == 0)] = self.no_data

        ipvi[np.isinf(ipvi) | np.isnan(ipvi)] = self.no_data

        if self.out_type > 1:
            ipvi = self.rescale_range(ipvi)

        return ipvi

    def MSAVI(self):

        """
        Modified Soil Adjusted Vegetation Index (MSAVI2)

        Equation:
            ((2 * nir + 1) - sqrt(((2 * nir + 1)^2) - (8 * (nir - Red)))) / 2
        """

        try:
            red = self.image_array[0]
            nir = self.image_array[1]
        except:
            raise ValueError('\nThe input array should have {:d} dimensions.\n'.format(self.n_bands))

        topR1 = np.add(np.multiply(nir, 2.), 1.)

        topR2 = np.power(topR1, 2.)
        topR4 = np.multiply(np.subtract(nir, red), 8.)

        topR5 = np.subtract(topR2, topR4)

        topR6 = np.sqrt(topR5)
        msavi = np.subtract(topR1, topR6)

        msavi = np.divide(msavi, 2.)

        msavi[(red == 0) | (nir == 0)] = self.no_data

        msavi[np.isinf(msavi) | np.isnan(msavi)] = self.no_data

        if self.out_type > 1:
            msavi = self.rescale_range(msavi)

        return msavi

    def GNDVI(self):

        """
        Green Normalised Difference Vegetation Index (GNDVI)

        Equation:
            GNDVI = (NIR - green) / (NIR + green)
        """

        try:
            green = self.image_array[0]
            nir = self.image_array[1]
        except:
            raise ValueError('\nThe input array should have {:d} dimensions.\n'.format(self.n_bands))

        gndvi = self.main_index(green, nir)

        gndvi[(gndvi < -1.)] = -1.
        gndvi[(gndvi > 1.)] = 1.

        gndvi[(green == 0) | (nir == 0)] = self.no_data

        gndvi[np.isinf(gndvi) | np.isnan(gndvi)] = self.no_data

        if self.out_type > 1:
            gndvi = self.rescale_range(gndvi, in_range=(-1., 1.))

        return gndvi

    def MNDWI(self):

        """
        Modified Normalised Difference Water Index (MNDWI)

        Equation:
            MNDWI = (green - MidIR) / (green + MidIR)

        Reference:
            Xu, Hanqiu (2006) Modification of normalised difference water index (NDWI) to enhance
                open water features in remotely sensed imagery. IJRS 27:14.
        """

        try:
            midir = self.image_array[0]
            green = self.image_array[1]
        except:
            raise ValueError('\nThe input array should have {:d} dimensions.\n'.format(self.n_bands))

        mndwi = self.main_index(midir, green)

        mndwi[(mndwi < -1.)] = -1.
        mndwi[(mndwi > 1.)] = 1.

        mndwi[(green == 0) | (midir == 0)] = self.no_data

        mndwi[np.isinf(mndwi) | np.isnan(mndwi)] = self.no_data

        if self.out_type > 1:
            mndwi = self.rescale_range(mndwi, in_range=(-1., 1.))

        return mndwi

    def NDSI(self):

        """
        Normalised Difference Soil Index (NDSI) (Rogers) or
        Normalised Difference Water Index (NDWI) (Gao)

        Equation:
            NDSI = (MidIR - NIR) / (MidIR + NIR)

        References:
            Rogers, A.S. & Kearney, M.S. (2004) 'Reducing signature
                variability in unmixing coastal marsh Thematic
                Mapper scenes using spectral indices' International
                Journal of Remote Sensing, 25(12), 2317-2335.

            Gao, Bo-Cai (1996) 'NDWI A Normalized Difference Water
                Index for Remote Sensing of Vegetation Liquid Water
                From Space' Remote Sensing of Environment.
        """

        try:
            nir = self.image_array[0]
            midir = self.image_array[1]
        except:
            raise ValueError('\nThe input array should have {:d} dimensions.\n'.format(self.n_bands))

        ndsi = self.main_index(nir, midir)

        ndsi[(ndsi < -1.)] = -1.
        ndsi[(ndsi > 1.)] = 1.

        ndsi[(nir == 0) | (midir == 0)] = self.no_data

        ndsi[np.isinf(ndsi) | np.isnan(ndsi)] = self.no_data

        if self.out_type > 1:
            ndsi = self.rescale_range(ndsi, in_range=(-1., 1.))

        return ndsi

    def NDBAI(self):

        """
        Normalised Difference Bareness Index (NDBaI)

        Equation:
            NDBaI = (FarIR - MidIR) / (FarIR + MidIR)

        Reference:
            Zhao, Hongmei, Chen, Xiaoling (1995) 'Use of Normalized
                Difference Bareness Index in Quickly Mapping Bare
                Areas from TM/ETM+' IEEE.
        """

        try:
            midir = self.image_array[0]
            farir = self.image_array[1]
        except:
            raise ValueError('\nThe input array should have {:d} dimensions.\n'.format(self.n_bands))

        ndbai = self.main_index(midir, farir)

        ndbai[(ndbai < -1.)] = -1.
        ndbai[(ndbai > 1.)] = 1.

        ndbai[(midir == 0) | (farir == 0)] = self.no_data

        ndbai[np.isinf(ndbai) | np.isnan(ndbai)] = self.no_data

        if self.out_type > 1:
            ndbai = self.rescale_range(ndbai, in_range=(-1., 1.))

        return ndbai

    def NDVI(self):

        """
        Normalised Difference Vegetation Index (NDVI)

        Equation:
            NDVI = (NIR - red) / (NIR + red)
        """

        try:
            red = self.image_array[0]
            nir = self.image_array[1]
        except:
            raise ValueError('\nThe input array should have {:d} dimensions.\n'.format(self.n_bands))

        ndvi = self.main_index(red, nir)

        ndvi[(ndvi < -1.)] = -1.
        ndvi[(ndvi > 1.)] = 1.

        ndvi[(red == 0) | (nir == 0)] = self.no_data

        ndvi[np.isinf(ndvi) | np.isnan(ndvi)] = self.no_data

        if self.out_type > 1:
            ndvi = self.rescale_range(ndvi, in_range=(-1., 1.))

        return ndvi

    def RENDVI(self):

        """
        Rededge Normalised Difference Vegetation Index (RENDVI)

        Equation:
            RENDVI = (NIR - rededge) / (NIR + rededge)
        """

        try:
            rededge = self.image_array[0]
            nir = self.image_array[1]
        except:
            raise ValueError('\nThe input array should have {:d} dimensions.\n'.format(self.n_bands))

        rendvi = self.main_index(rededge, nir)

        rendvi[(rendvi < -1.)] = -1.
        rendvi[(rendvi > 1.)] = 1.

        rendvi[(rededge == 0) | (nir == 0)] = self.no_data

        rendvi[np.isinf(rendvi) | np.isnan(rendvi)] = self.no_data

        if self.out_type > 1:
            rendvi = self.rescale_range(rendvi, in_range=(-1., 1.))

        return rendvi

    def NDWI(self):

        """
        Normalised Difference Water Index (NDWI)

        Equation:
            NDWI = (green - NIR) / (green + NIR)

        Reference:
            McFeeters, S.K. (1996) 'The use of the Normalized Difference
                Water Index (NDWI) in the delineation of open water
                features, International Journal of Remote Sensing, 17(7),
                1425-1432.
        """

        try:
            nir = self.image_array[0]
            green = self.image_array[1]
        except:
            raise ValueError('\nThe input array should have {:d} dimensions.\n'.format(self.n_bands))

        ndwi = self.main_index(nir, green)

        ndwi[(ndwi < -1.)] = -1.
        ndwi[(ndwi > 1.)] = 1.

        ndwi[(green == 0) | (nir == 0)] = self.no_data

        ndwi[np.isinf(ndwi) | np.isnan(ndwi)] = self.no_data

        if self.out_type > 1:
            ndwi = self.rescale_range(ndwi, in_range=(-1., 1.))

        return ndwi

    def PNDVI(self):

        """
        Pseudo Normalised Difference Vegetation Index (PNDVI)

        Equation:
            PNDVI = (red - green) / (red + green)
        """

        try:
            green = self.image_array[0]
            red = self.image_array[1]
        except:
            raise ValueError('\nThe input array should have {:d} dimensions.\n'.format(self.n_bands))

        pndvi = self.main_index(green, red)

        pndvi[(pndvi < -1.)] = -1.
        pndvi[(pndvi > 1.)] = 1.

        pndvi[(green == 0) | (red == 0)] = self.no_data

        pndvi[np.isinf(pndvi) | np.isnan(pndvi)] = self.no_data

        if self.out_type > 1:
            pndvi = self.rescale_range(pndvi, in_range=(-1., 1.))

        return pndvi

    def RBVI(self):

        """
        Red Blue Vegetation Index (RBVI)

        Equation:
            RBVI = (red - blue) / (red + blue)
        """

        try:
            blue = self.image_array[0]
            red = self.image_array[1]
        except:
            raise ValueError('\nThe input array should have {:d} dimensions.\n'.format(self.n_bands))

        rbvi = self.main_index(blue, red)

        rbvi[(rbvi < -1.)] = -1.
        rbvi[(rbvi > 1.)] = 1.

        rbvi[(blue == 0) | (red == 0)] = self.no_data

        rbvi[np.isinf(rbvi) | np.isnan(rbvi)] = self.no_data

        if self.out_type > 1:
            rbvi = self.rescale_range(rbvi, in_range=(-1., 1.))

        return rbvi

    def GBVI(self):

        """
        Green Blue Vegetation Index (GBVI)

        Equation:
            GBVI = (green - blue) / (green + blue)
        """

        try:
            blue = self.image_array[0]
            green = self.image_array[1]
        except:
            raise ValueError('\nThe input array should have {:d} dimensions.\n'.format(self.n_bands))

        gbvi = self.main_index(blue, green)

        gbvi[(gbvi < -1.)] = -1.
        gbvi[(gbvi > 1.)] = 1.

        gbvi[(blue == 0) | (green == 0)] = self.no_data

        gbvi[np.isinf(gbvi) | np.isnan(gbvi)] = self.no_data

        if self.out_type > 1:
            gbvi = self.rescale_range(gbvi, in_range=(-1., 1.))

        return gbvi

    def ONDVI(self):

        """
        Theta Normalised Difference Vegetation Index (0NDVI)

        Equation:
            (4 / pi) * arctan(NDVI)
        """

        original_type = copy(self.out_type)

        self.out_type = 1

        ndvi = self.NDVI()

        self.out_type = original_type

        red = self.image_array[0]
        nir = self.image_array[1]

        ondvi = np.multiply(np.arctan(ndvi), 4. / np.pi)

        ondvi[(red == 0) | (nir == 0)] = self.no_data

        ondvi[np.isinf(ondvi) | np.isnan(ondvi)] = self.no_data

        if self.out_type > 1:
            ondvi = self.rescale_range(ondvi)

        return ondvi

    def SATVI(self, L=.5):

        """
        Soil Adjusted Total Vegetation Index (SATVI)

        Equation:
            [((Mid-IR - Red) / (Mid-IR + Red + 0.1)) * 1.1)] - (Far-IR / 2)
        """

        try:
            red = self.image_array[0]
            midir = self.image_array[1]
            farir = self.image_array[2]
        except:
            raise ValueError('\nThe input array should have {:d} dimensions.\n'.format(self.n_bands))

        top_r0 = np.subtract(midir, red)
        top_r1 = np.add(np.add(midir, red), L)
        top_r2 = np.divide(top_r0, top_r1)
        satvi = np.multiply(top_r2, 1.+L)

        satvi = np.subtract(satvi, np.divide(farir, 2.))

        satvi[(red == 0) | (midir == 0) | (farir == 0)] = self.no_data

        satvi[np.isinf(satvi) | np.isnan(satvi)] = self.no_data

        if self.out_type > 1:
            satvi = self.rescale_range(satvi)

        return satvi

    def SAVI(self, L=.5):

        """
        Soil Adjusted Vegetation Index (SAVI)

        Equation:
            ((NIR - red) / (NIR + red + L)) * (1 + L)
        """

        try:
            red = self.image_array[0]
            nir = self.image_array[1]
        except:
            raise ValueError('\nThe input array should have {:d} dimensions.\n'.format(self.n_bands))

        top_r = np.subtract(nir, red)
        top_rx = np.multiply(top_r, 1.+L)

        bottom = np.add(np.add(red, nir), L)

        savi = np.divide(top_rx, bottom)

        savi[(red == 0) | (nir == 0)] = self.no_data

        savi[np.isinf(savi) | np.isnan(savi)] = self.no_data

        if self.out_type > 1:
            savi = self.rescale_range(savi)

        return savi

    def OSAVI(self, L=.5):

        """
        Theta Soil Adjusted Vegetation Index (0SAVI)

        Equation:
                   ((NIR - red) / (NIR + red + L)) * (1 + L)
            arctan(-----------------------------------------) * 2
                                    1.5
        """

        try:
            red = self.image_array[0]
            nir = self.image_array[1]
        except:
            raise ValueError('\nThe input array should have {:d} dimensions.\n'.format(self.n_bands))

        original_type = copy(self.out_type)

        self.out_type = 1

        osavi = self.SAVI()

        self.out_type = original_type

        osavi = np.multiply(np.arctan(np.divide(osavi, 1.5)), 2.)

        osavi[(red == 0) | (nir == 0)] = self.no_data

        osavi[np.isinf(osavi) | np.isnan(osavi)] = self.no_data

        if self.out_type > 1:
            osavi = self.rescale_range(osavi)

        return osavi

    def SVI(self):

        """
        Simple Vegetation Index (SVI)

        Equation:
            NIR / red
        """

        try:
            red = self.image_array[0]
            nir = self.image_array[1]
        except:
            raise ValueError('\nThe input array should have {:d} dimensions.\n'.format(self.n_bands))

        svi = np.divide(nir, red)

        svi[(red == 0) | (nir == 0)] = self.no_data

        svi[np.isinf(svi) | np.isnan(svi)] = self.no_data

        if self.out_type > 1:
            svi = self.rescale_range(svi)

        return svi

    def TNDVI(self):

        """
        Transformed Normalised Difference Vegetation Index (TNDVI)

        Equation:
            Square Root(((NIR - red) / (NIR + red)) * 0.5)
        """

        try:
            red = self.image_array[0]
            nir = self.image_array[1]
        except:
            raise ValueError('\nThe input array should have {:d} dimensions.\n'.format(self.n_bands))

        original_type = copy(self.out_type)

        self.out_type = 1

        tndvi = self.NDVI()

        self.out_type = original_type

        tndvi = self.NDVI()
        tndvi = np.sqrt(np.multiply(tndvi, .5))

        tndvi[(red == 0) | (nir == 0)] = self.no_data

        tndvi[np.isinf(tndvi) | np.isnan(tndvi)] = self.no_data

        if self.out_type > 1:
            tndvi = self.rescale_range(tndvi)

        return tndvi

    def TVI(self):

        """
        Transformed Vegetation Index (TVI)

        Equation:
            Square Root(((NIR - green) / (NIR + green)) + 0.5)
        """

        original_type = copy(self.out_type)

        self.out_type = 1

        tvi = self.GNDVI()

        self.out_type = original_type

        green = self.image_array[0]
        nir = self.image_array[1]

        tvi = np.sqrt(np.add(tvi, .5))

        tvi[(green == 0) | (nir == 0)] = self.no_data

        tvi[np.isinf(tvi) | np.isnan(tvi)] = self.no_data

        if self.out_type > 1:
            tvi = self.rescale_range(tvi)

        return tvi

    def YNDVI(self):

        """
        Yellow Normalized Difference Vegetation Index (YNDVI)

        Equation:
            YNDVI = (nir - yellow) / (nir + yellow)
        """

        try:
            yellow = self.image_array[0]
            nir = self.image_array[1]
        except:
            raise ValueError('\nThe input array should have {:d} dimensions.\n'.format(self.n_bands))

        yndvi = self.main_index(yellow, nir)

        yndvi[(yndvi < -1.)] = -1.
        yndvi[(yndvi > 1.)] = 1.

        yndvi[(yellow == 0) | (nir == 0)] = self.no_data

        yndvi[np.isinf(yndvi) | np.isnan(yndvi)] = self.no_data

        if self.out_type > 1:
            yndvi = self.rescale_range(yndvi, in_range=(-1., 1.))

        return yndvi

    def VCI(self, min_ndvi=-1, max_ndvi=1):

        """
        Vegetation Condition Index (VCI)

        Reference:
            Kogan (1997) & Kogan et al. (2011)

        Equation:
            (NDVI - NDVI_min) / (NDVI_max - NDVI_min)
        """

        original_type = copy(self.out_type)

        self.out_type = 1

        ndvi = self.NDVI()

        self.out_type = original_type

        red = self.image_array[0]
        nir = self.image_array[1]

        vci = np.subtract(ndvi, min_ndvi)
        vci_bot = np.subtract(max_ndvi, min_ndvi)

        vci = np.divide(vci, vci_bot)

        vci[(red == 0) | (nir == 0)] = self.no_data

        vci[np.isinf(vci) | np.isnan(vci)] = self.no_data

        if self.out_type > 1:
            vci = self.rescale_range(vci)

        return vci

    def main_index(self, array01, array02):

        top = np.subtract(array02, array01)
        bottom = np.add(array02, array01)

        return np.divide(top, bottom)


class BandHandler(SensorInfo):

    def __init__(self, sensor):

        self.sensor = sensor

        SensorInfo.__init__(self)

    def get_band_order(self):

        try:
            self.band_order = self.band_orders[self.sensor]
        except:
            raise ValueError('\n{} is not supported. Choose from: {}'.format(self.sensor, ','.join(self.sensors)))

    def stack_bands(self, band_list):

        """
        Returns stacked bands in sorted (smallest band to largest band) order.
        """

        band_positions = self.get_band_positions(band_list)

        return self.meta_info.mparray(bands2open=band_positions, i=self.i, j=self.j,
                                      rows=self.n_rows, cols=self.n_cols, d_type='float32')

    def get_band_positions(self, band_list):

        return [self.band_order[img_band] for img_band in band_list]


class VegIndices(BandHandler):

    """
    Args:
        input_image (str)
        input_indice (str)
        sensor (str)
    """

    def __init__(self, input_image, input_indice, sensor):

        self.input_indice = input_indice

        # Get the sensor band order.
        BandHandler.__init__(self, sensor)

        self.get_band_order()

        # Open the image.
        self.meta_info = raster_tools.rinfo(input_image)

        self.rows, self.cols = self.meta_info.rows, self.meta_info.cols

    def run(self, output_image, storage='float32', no_data=0, chunk_size=1024, k=0,
            be_quiet=False, overwrite=False, overviews=False):

        """
        Args:
            output_image (str)
            storage (Optional[str])
            no_data (Optional[int])
            chunk_size (Optional[int])
            k (Optional[int])
            be_quiet (Optional[bool])
            overwrite (Optional[bool])
            overviews (Optional[bool])
        """

        self.output_image = output_image
        self.storage = storage
        self.no_data = no_data
        self.chunk_size = chunk_size
        self.k = k
        self.be_quiet = be_quiet

        print_progress = True

        if self.storage == 'float32':
            self.out_type = 1
        elif self.storage == 'byte':

            if (self.no_data < 0) or (self.no_data > 255):

                raise ValueError("""
                The 'no data' value cannot be less than 0 or
                greater than 255 with Byte storage.
                """)

            self.out_type = 2

        elif self.storage == 'uint16':

            if self.no_data < 0:

                raise ValueError("""
                The 'no data' value cannot be less than 0
                with UInt16 storage.
                """)

            self.out_type = 3

        else:
            raise NameError('{} is not a supported storage option.'.format(self.storage))

        d_name, f_name = os.path.split(self.output_image)
        __, f_ext = os.path.splitext(f_name)

        if not os.path.isdir(d_name):
            os.makedirs(d_name)

        o_info = self.meta_info.copy()

        o_info.storage = self.storage
        o_info.bands = 1

        if self.chunk_size == -1:

            block_size_rows = copy(self.rows)
            block_size_cols = copy(self.cols)

            print_progress = False

        else:

            # set the block size
            block_size_rows, block_size_cols = raster_tools.block_dimensions(self.rows, self.cols,
                                                                             row_block_size=self.chunk_size,
                                                                             col_block_size=self.chunk_size)

        if overwrite:

            associated_files = fnmatch.filter(os.listdir(d_name), '*{}*'.format(f_name))
            associated_files = fnmatch.filter(associated_files, '*{}*'.format(f_ext))

            if associated_files:

                for associated_file in associated_files:

                    associated_file_full = '{}/{}'.format(d_name, associated_file)

                    if os.path.isfile(associated_file_full):
                        os.remove(associated_file_full)

        if os.path.isfile(self.output_image):
            print '\n{} already exists ...'.format(self.output_image)
        else:

            out_rst = raster_tools.create_raster(self.output_image, o_info, compress='none')

            out_rst.get_band(1)

            if not self.be_quiet:

                print '\n{} ...\n'.format(self.input_indice.upper())

                if print_progress:
                    ctr, pbar = _iteration_parameters(self.rows, self.cols, block_size_rows, block_size_cols)

            max_ndvi, min_ndvi = 0., 0.

            # Iterate over the image block by block.
            for self.i in xrange(0, self.rows, block_size_rows):

                self.n_rows = raster_tools.n_rows_cols(self.i, block_size_rows, self.rows)

                for self.j in xrange(0, self.cols, block_size_cols):

                    self.n_cols = raster_tools.n_rows_cols(self.j, block_size_cols, self.cols)

                    # Stack the image array with the
                    #   appropriate bands.
                    try:
                        if self.input_indice.upper() == 'VCI':
                            image_stack = self.stack_bands(self.wavelength_lists['NDVI'])
                        else:
                            image_stack = self.stack_bands(self.wavelength_lists[self.input_indice.upper()])
                    except:
                        raise NameError('{} cannot be computed for {}.'.format(self.input_indice.upper(), self.sensor))

                    # Setup the vegetation index object.
                    vie = VegIndicesEquations(image_stack, chunk_size=self.chunk_size, no_data=self.no_data)

                    # Calculate the vegetation index.
                    veg_indice_array = vie.compute(self.input_indice, out_type=self.out_type)

                    if self.input_indice.upper() == 'VCI':

                        # get the maximum NDVI value for the image
                        max_ndvi = max(veg_indice_array.max(), max_ndvi)

                        # get the maximum NDVI value for the image
                        min_ndvi = min(veg_indice_array.min(), min_ndvi)

                    if self.input_indice != 'VCI':

                        out_rst.write_array(veg_indice_array, i=self.i, j=self.j)

                    if not self.be_quiet:

                        if print_progress:
                            pbar.update(ctr)
                            ctr += 1

            if not self.be_quiet:
                if print_progress:
                    pbar.finish()

            if self.input_indice.upper() == 'VCI':

                if not self.be_quiet:
                    print '\nComputing VCI ...'

                # iterative over entire image with row blocks
                for self.i in xrange(0, self.rows, block_size_rows):

                    self.n_rows = raster_tools.n_rows_cols(self.i, block_size_rows, self.rows)

                    for self.j in xrange(0, self.cols, block_size_cols):

                        self.n_cols = raster_tools.n_rows_cols(self.j, block_size_cols, self.cols)

                        # Stack the image array with the
                        #   appropriate bands.
                        try:
                            image_stack = self.stack_bands(self.wavelength_lists[self.input_indice.upper()])
                        except:
                            raise NameError('{} cannot be computed for {}.'.format(self.input_indice.upper(),
                                                                                   self.sensor))

                        # Setup the vegetation index object.
                        vie = VegIndicesEquations(image_stack, chunk_size=self.chunk_size)

                        # Calculate the VCI index.
                        veg_indice_array = vie.compute(self.input_indice, out_type=self.out_type,
                                                       min_ndvi=min_ndvi, max_ndvi=max_ndvi)

                        out_rst.write_array(veg_indice_array, i=self.i, j=self.j)

            out_rst.close_all()

            if self.k > 0:

                print

                d_name, f_name = os.path.split(self.output_image)
                f_base, f_ext = os.path.splitext(f_name)

                outImgResamp = '{}/{}_resamp{}'.format(d_name, f_base, f_ext)

                comResamp = 'gdalwarp -tr {:f} {:f} -r near {} {}'.format(self.k, self.k,
                                                                          self.output_image, outImgResamp)

                subprocess.call(comResamp, shell=True)

        if overviews:

            print '\nComputing overviews ...\n'

            v_info = raster_tools.rinfo(output_image)
            v_info.build_overviews()
            v_info.close()


def _compute_as_list(img, out_img, sensor, k, storage, no_data, chunk_size,
                     overwrite, overviews, veg_indice_list=[]):

    if (len(veg_indice_list) == 1) and (veg_indice_list[0].lower() == 'all'):

        si = SensorInfo()
        si.list_indice_options(sensor)

        veg_indice_list = si.sensor_indices

    d_name, f_name = os.path.split(out_img)
    f_base, f_ext = os.path.splitext(f_name)

    name_list = []

    for input_indice in veg_indice_list:

        out_img_indice = '{}/{}_{}{}'.format(d_name, f_base, input_indice.lower(), f_ext)

        name_list.append(out_img_indice)

        vio = VegIndices(img, input_indice, sensor)

        vio.run(out_img_indice, k=k, storage=storage, no_data=no_data, chunk_size=chunk_size, overwrite=overwrite)

    out_stack = '{}/{}_STACK.vrt'.format(d_name, f_base)

    # Stack all the indices.
    composite(d_name, out_stack, stack=True, image_list=name_list, build_overviews=overviews, no_data=no_data)

    # Save a list of vegetation indice names.
    index_order = '{}/{}_STACK_order.txt'.format(d_name, f_base)

    with open(index_order, 'w') as tio:

        for bi, vi in enumerate(veg_indice_list):
            tio.write('{:d}: {}\n'.format(bi+1, vi))


def veg_indices(input_image, output_image, input_index, sensor, k=0., storage='float32', no_data=0,
                chunk_size=1024, be_quiet=False, overwrite=False, overviews=False):

    """
    Computes vegetation indexes

        Assumes standard band orders for available sensors.

    Args:
        input_image (str): The input image.
        output_image (str): The output image.
        input_index (str or str list): Vegetation index or list of indices to compute.
        sensor (str): Input sensor. Choices are [ASTER VNIR, CBERS2, CitySphere, GeoEye1, IKONOS, Landsat, Landsat8, 
            Landsat thermal, MODIS, Pan, RapidEye, Sentinel2-10m (coming), Sentinel2-20m (coming),
            Quickbird, WorldView2, WorldView2 PS FC].
        k (Optional[float]): Resample size. Default is 0., or no resampling.
        storage (Optional[str]): Storage type of ``output_image``. Default is 'float32'. Choices are
            ['byte', 'uint16', 'float32].
        no_data (Optional[int]): The output 'no data' value for ``output_image``. Default is 0.
        chunk_size (Optional[int]): Size of image chunks. Default is 1024. *chunk_size=-1 will use Numexpr
            threading. This should give faster results on larger imagery.
        be_quiet (Optional[bool]): Whether to print progress (False) or be quiet (True). Default is False.
        overwrite (Optional[bool]): Whether to overwrite an existing ``output_image`` file. Default is False.
        overviews (Optional[bool]): Whether to build pyramid overviews for ``output_image``. Default is False.

    Examples:
        >>> from mappy.features import veg_indices
        >>>
        >>> # Compute the NDVI for Landsat (4, 5, or 7).
        >>> veg_indices('/some_image.tif', '/some_image_indice.tif', 'NDVI', 'Landsat')
        >>>
        >>> # Compute the NDVI, but save as Byte (0-255) storage.
        >>> veg_indices('/some_image.tif', '/some_image_indice.tif', 'NDVI', 'Landsat', \
        >>>             storage='byte', overviews=True)
        >>>
        >>> # Compute the NDVI for Landsat 8.
        >>> veg_indices('/some_image.tif', '/some_image_indice.tif', 'NDVI', 'Landsat8')
        >>>
        >>> # Compute the NDVI for Sentinel 2.
        >>> veg_indices('/some_image.tif', '/some_image_indice.tif', 'NDVI', 'Sentinel2')

    Returns:
        None, writes to ``output_image``.

    Vegetation Indices:
        ARVI
            Name:
                Atmospheric resistant vegetation index
        CNDVI
            Name:
                Corrected normalized difference vegetation index
            Eq:
                CNDVI = [(nir - Red) / (nir + Red)] * (1 - [(SWIR - SWIRmin) / (SWIRmax - SWIRmin)])
            Ref:
                Nemani et al. 1993.
        EVI
            Name:
                Enhanced vegetation index
        EVI2
        GNDVI
            Name:
                Green normalized difference vegetation index
            Eq:
                GNDVI = (nir - Green) / (nir + Green)
        IPVI
            Name:
                Infrared Percentage Vegetation Index
            Eq:
                IPVI = nir / (nir + Red)
            Ref:
                Crippen, R.E. 1990. "Calculating the Vegetation Index Faster."
                    Remote Sensing of Environment 34: 71-73.
        MNDWI
        MSAVI2 -- MSAVI in veg_indice_arrayndices.py
            Name:
                Modified Soil Adjusted Vegetation Index
            Eq:
                MSAVI = ((2 * (nir + 1)) - sqrt(((2 * nir + 1)^2) - (8 * (nir - Red)))) / 2
            Ref:
                Qi, J., Chehbouni, A., Huete, A.R., and Kerr, Y.H. 1994.
                    "Modified Soil Adjusted Vegetation Index (MSAVI)." Remote Sensing
                    of Environment 48: 119-126.
        NDBI
        NDBaI
            Name:
                Normalized difference bareness index
        NDVI
            Name:
                Normalized Difference Vegetation Index
            Eq:
                NDVI = (nir - Red) / (nir + Red)
            Ref:
                Rouse, J.W., Haas, R.H., Schell, J.A., and Deering, D.W. 1973.
                    "Monitoring vegetation systems in the great plains with rERTS."
                    Third ERTS Symposium, NASA SP-351 1: 309-317.
                Kriegler, F.J., Malila, W.A., Nalepka, R.F., and Richardson, W.
                    1969. "Preprocessing transformations and their effects on
                    multispectral recognition." in Proceedings of the Sixth
                    International Symposium on Remote Sensing of Environment,
                    University of Michigan, Ann Arbor, MI: 97-131.
        ONDVI
            Name:
                Theta normalized difference vegetation index
            Eq:
                (4 / pi) * arctan(NDVI)
        NDWI
            Name:
                Normalized difference water index or moisture index
        pNDVI
            Name:
                Pseudo normalized difference vegetation index
            Eq:
                pNDVI = (Red - Green) / (Red + Green)
        RBVI
        SATVI
        SAVI
            Name:
                Soil Adjusted Vegetation Index
            Eq:
                SAVI = ((nir - Red) / (nir + Red + L)) * (1 + L)
                    where, L=0.--1. (high--low vegetation cover)
            Ref:
                Huete, A.R. 1988. "A soil-adjusted vegetation index (SAVI)." Remote Sensing of
                    Environment 25, 295-309.
        SVI (or RVI)
            Name:
                Simple Vegetation Index (or Ratio Vegetation Index)
            Eq:
                RVI = nir / Red
            Ref:
                Jordan, C.F. 1969. "Derivation of leaf area index from quality of
                    light on the forest floor." Ecology 50: 663-666.
        TSAVI -- not used
            Name:
                Transformed Soil Adjusted Vegetation Index
            Eq:
                TSAVI = s(nir - s * Red - a) / (a * nir + red - a * s + X * (1 + s * s))
            Ref:
                Baret, F., Guyot, G., and Major, D. 1989. "TSAVI: A vegetation
                    index which minimizes soil brightness effects on LAI or APAR
                    estimation." in 12th Canadian Symposium on Remote Sensing and
                    IGARSS 1990, Vancouver, Canada, July 10-14.
                Baret, F. and Guyot, G. 1991. "Potentials and limits of vegetation
                    indices for LAI and APAR assessment." Remote Sensing of
                    Environment 35: 161-173.
        TNDVI
        TVI
    """

    if isinstance(input_index, str):

        if input_index.lower() == 'all':

            _compute_as_list(input_image, output_image, sensor, k, storage, no_data,
                             chunk_size, overwrite, overviews)

        else:

            vio = VegIndices(input_image, input_index, sensor)

            vio.run(output_image, k=k, storage=storage, no_data=no_data, chunk_size=chunk_size,
                    be_quiet=be_quiet, overwrite=overwrite, overviews=overviews)

    if isinstance(input_index, list):

        if (len(input_index) == 1) and (input_index[0].lower() != 'all'):

            vio = VegIndices(input_image, input_index[0], sensor)

            vio.run(output_image, k=k, storage=storage, no_data=no_data, chunk_size=chunk_size,
                    be_quiet=be_quiet, overwrite=overwrite, overviews=overviews)

        else:

            _compute_as_list(input_image, output_image, sensor, k, storage, no_data,
                             chunk_size, overwrite, overviews, veg_indice_list=input_index)


def _examples():

    sys.exit("""\

    # List vegetation index options for a sensor.
    veg_indices.py --sensor Landsat --options

    # List the expected band order for a sensor.
    veg_indices.py --sensor Landsat --band_order
    veg_indices.py --sensor Landsat8 --band_order
    veg_indices.py --sensor Landsat-thermal --band_order
    veg_indices.py --sensor Sentinel2 --band_order
    veg_indices.py --sensor Quickbird --band_order
    veg_indices.py --sensor RGB --band_order

    # Compute NDVI for a Landsat image.
    veg_indices.py -i /some_image.tif -o /ndvi.tif --index ndvi --sensor Landsat

    # Compute NDVI for a Landsat8 image.
    veg_indices.py -i /some_image.tif -o /ndvi.tif --index ndvi --sensor Landsat8

    # Compute NDVI for a Sentinel 2 image.
    veg_indices.py -i /some_image.tif -o /ndvi.tif --index ndvi --sensor Sentinel2

    # Compute NDWI for a Landsat image and save as Byte (0-255) storage.
    veg_indices.py -i /some_image.tif -o /ndwi.tif --index ndwi --sensor Landsat --storage byte --overviews

    # Compute NDSI for a Landsat image, save as float32 storage, and set 'no data' pixels to -999.
    veg_indices.py -i /some_image.tif -o /ndsi.tif --index ndsi --sensor Landsat --overviews --no_data -999

    # Compute NDVI and SAVI for a Landsat image. The --chunk -1 parameter tells the
    #   system to use 'numexpr' for calculations.
    #
    #   *Each output image will be saved as /output_ndvi.tif, /output_savi.tif AND
    #   a VRT multi-band image will be saved as /output_STACK.vrt. Unlike single index
    #   triggers, if --index is a list of more than one index, the --overviews parameter
    #   will only build overviews for the VRT file.
    veg_indices.py -i /some_image.tif -o /output.tif --index ndvi savi --sensor Landsat --overviews --chunk -1

    # Compute all available indices for Landsat.
    veg_indices.py -i /some_image.tif -o /output.tif --index all --sensor Landsat
    """)


def main():

    parser = argparse.ArgumentParser(description='Computes spectral indices and band ratios',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-e', '--examples', dest='examples', action='store_true', help='Show usage examples and exit')
    parser.add_argument('-i', '--input', dest='input', help='The input image', default=None)
    parser.add_argument('-o', '--output', dest='output', help='The output image', default=None)
    parser.add_argument('--index', dest='index', help='The vegetation index to compute', default=['ndvi'], nargs='+')
    parser.add_argument('--sensor', dest='sensor', help='The input sensor', default='Landsat',
                        choices=SensorInfo().sensors)
    parser.add_argument('-k', '--resample', dest='resample', help='Resample cell size', default=0., type=float)
    parser.add_argument('-s', '--storage', dest='storage', help='The storage type', default='float32')
    parser.add_argument('-n', '--no_data', dest='no_data', help='The output "no data" value', default=0, type=int)
    parser.add_argument('-c', '--chunk', dest='chunk', help='The chunk size', default=1024, type=int)
    parser.add_argument('-q', '--be_quiet', dest='be_quiet', help='Whether to be quiet', action='store_true')
    parser.add_argument('--overwrite', dest='overwrite', help='Whether to overwrite an existing file',
                        action='store_true')
    parser.add_argument('--overviews', dest='overviews', help='Whether to build pyramid overviews',
                        action='store_true')
    parser.add_argument('--options', dest='options',
                        help='Whether to list the vegetation index options for the sensor, --sensor',
                        action='store_true')
    parser.add_argument('--band_order', dest='band_order',
                        help='Whether to list the expected band order for the sensor, --sensor',
                        action='store_true')

    args = parser.parse_args()

    if args.options:

        si = SensorInfo()
        si.list_indice_options(args.sensor)

        sys.exit("""\

        Available indices for {}:
        {}

        """.format(args.sensor, ', '.join(si.sensor_indices)))

    if args.band_order:

        si = SensorInfo()

        sys.exit(si.list_expected_band_order(args.sensor))

    if args.examples:
        _examples()

    print('\nStart date & time --- (%s)\n' % time.asctime(time.localtime(time.time())))

    start_time = time.time()

    veg_indices(args.input, args.output, args.index, args.sensor, k=args.resample, storage=args.storage,
                no_data=args.no_data, chunk_size=args.chunk, be_quiet=args.be_quiet,
                overwrite=args.overwrite, overviews=args.overviews)

    print('\nEnd data & time -- (%s)\nTotal processing time -- (%.2gs)\n' %
          (time.asctime(time.localtime(time.time())), (time.time()-start_time)))

if __name__ == '__main__':
    main()
