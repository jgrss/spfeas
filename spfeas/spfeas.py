#!/usr/bin/env python

"""
@authors: Jordan Graesser
Date Created: 9/29/2016
"""

from __future__ import print_function
from future.utils import viewitems
from builtins import dict

import os
import sys
import argparse
import time
import copy

from .errors import logger
from . import spprocess
from .sphelpers.sputilities import set_yaml_file

from mpglue.raster_tools import DRIVER_DICT
from mpglue import utils

try:
    import colorama
    from colorama import Fore, Style
except ImportError:
    raise ImportError('Colorama must be installed')


class SPParameters(object):

    def __init__(self, input_image, output_dir):

        self.input_image = input_image
        self.output_dir = output_dir

        self.features_dict = None
        self.out_bands_dict = None

        self.get_defaults()

        # Set the default parameters.
        for k, v in viewitems(self._defaults):
            setattr(self, k, v)

    def copy(self):
        return copy.copy(self)

    def get_defaults(self):

        self._defaults = dict(format='GTiff',
                              band_positions=[1],
                              block=2,
                              scales=[8],
                              triggers=['mean'],
                              hline_threshold=40,
                              hline_min=10,
                              hline_gap=2,
                              weight=False,
                              sfs_threshold=40,
                              sfs_skip=4,
                              sfs_resample=0,
                              smooth=0,
                              equalize=False,
                              equalize_adapt=False,
                              visualize=False,
                              convert=False,
                              use_rgb=False,
                              vis_order='bgr',
                              sat_sensor='Quickbird',
                              stack=False,
                              full_path=False,
                              stack_only=False,
                              neighbors=False,
                              n_jobs=-1,
                              reset=False,
                              image_min=-999.0,
                              image_max=-999.0,
                              lac_r=2,
                              section_size=1000,
                              gdal_cache=256,
                              overwrite=False,
                              overviews=False)

        # Set the features dictionary.
        self.features_dict = dict(ctr=1,
                                  dmp=2,
                                  fourier=2,
                                  gabor=8*2,
                                  grad=2,
                                  hough=4,
                                  hog=5,
                                  lac=1,
                                  lbp=62,
                                  lbpm=5,
                                  lsr=3,
                                  mean=2,
                                  orb=5,
                                  pantex=1,
                                  saliency=2,
                                  seg=2,
                                  sfs=6,
                                  sfsorf=6,
                                  surf=4,
                                  xy=2)

        self._update_bands_dict(self._defaults['scales'])

    def _update_bands_dict(self, scales_list):

        # Set the output bands based on the trigger.
        self.out_bands_dict = dict(ctr=len(scales_list) * self.features_dict['ctr'],
                                   dmp=len(scales_list) * self.features_dict['dmp'],
                                   fourier=len(scales_list) * self.features_dict['fourier'],
                                   gabor=len(scales_list) * self.features_dict['gabor'],
                                   grad=len(scales_list) * self.features_dict['grad'],
                                   hog=len(scales_list) * self.features_dict['hog'],
                                   hough=len(scales_list) * self.features_dict['hough'],
                                   lac=len(scales_list) * self.features_dict['lac'],
                                   lbp=len(scales_list) * self.features_dict['lbp'],
                                   lbpm=len(scales_list) * self.features_dict['lbpm'],
                                   lsr=len(scales_list) * self.features_dict['lsr'],
                                   mean=len(scales_list) * self.features_dict['mean'],
                                   orb=len(scales_list) * self.features_dict['orb'],
                                   pantex=len(scales_list) * self.features_dict['pantex'],
                                   saliency=len(scales_list) * self.features_dict['saliency'],
                                   seg=len(scales_list) * self.features_dict['seg'],
                                   sfs=len(scales_list) * self.features_dict['sfs'],
                                   sfsorf=len(scales_list) * self.features_dict['sfsorf'],
                                   surf=len(scales_list) * self.features_dict['surf'],
                                   xy=len(scales_list) * self.features_dict['xy'])

    def _crosscheck_sensor(self):

        for trigger in self.triggers:

            if trigger.upper() in utils.SUPPORTED_VIS:

                if self.sat_sensor not in utils.SENSOR_BAND_DICT:

                    logger.error('  The satellite sensor, {}, is not supported'.format(self.sat_sensor))
                    raise NameError

                vi_wvs = utils.VI_WAVELENGTHS[trigger.upper()]
                sensor_wvs = list(utils.SENSOR_BAND_DICT[self.sat_sensor])

                wvs_diff = list(set(vi_wvs).difference(set(sensor_wvs)))

                if wvs_diff:

                    logger.error('  The satellite sensor does not support the requested spectral indices.')
                    raise NameError

    def set_params(self, **kwargs):

        """
        Sets user-defined parameters
        """

        for k, v in viewitems(kwargs):
            setattr(self, k, v)

        # Check spectral indices
        #   against the sensor.
        self._crosscheck_sensor()

        for vi in utils.SUPPORTED_VIS:
            self.features_dict[vi.lower()] = 2

        if 'scales' in kwargs:
            self._update_bands_dict(self.scales)

        for vi in utils.SUPPORTED_VIS:
            self.out_bands_dict[vi.lower()] = len(self.scales) * self.features_dict[vi.lower()]

        if self.use_rgb:
            self.band_positions = [1]

        self.n_bands = len(self.band_positions)

        self.band_info = dict(band_count=0)

        self.spectral_indices = [i.lower() for i in utils.SUPPORTED_VIS]

        for trigger in self.triggers:

            # The starting band position for each trigger.
            self.band_info[trigger] = copy.copy(self.band_info['band_count'])

            # The total band count.
            self.band_info['band_count'] += self.out_bands_dict[trigger] * self.n_bands

        # Update the feature dictionary for feature neighbors.
        if self.neighbors:

            for key, val in viewitems(self.features_dict):
                self.features_dict[key] *= 5

        self.d_name, self.f_name = os.path.split(self.input_image)
        self.f_base = os.path.splitext(self.f_name)[0]

        self.f_ext = '.tif'

        # The log file.
        self.log_txt = os.path.join(self.output_dir, '{}_log.txt'.format(self.f_base))

        # The status file.
        self.status_file = set_yaml_file(self)

        if self.use_rgb:
            self.rgb2write = 'RGB'
        else:
            self.rgb2write = ','.join([str(bpos) for bpos in self.band_positions])

        if self.neighbors:
            self.write_neighbors = 'DID'
        else:
            self.write_neighbors = 'Did NOT'

        if self.equalize:
            self.write_equalize = 'DID'
        else:
            self.write_equalize = 'Did NOT'

        if self.equalize_adapt:
            self.write_equalize_adapt = 'DID'
        else:
            self.write_equalize_adapt = 'Did NOT'

        self.relative_path = True if not self.full_path else False

    def update_info(self, **kwargs):

        for k, v in viewitems(kwargs):
            setattr(self, k, v)

    def run(self):
        spprocess.run(self)
        

def spatial_features(input_image, output_dir, **kwargs):
    
    spp = SPParameters(input_image, output_dir)
    
    spp.set_params(**kwargs)
    
    spp.run()


def _examples():

    sys.exit("""\

    # Compute PanTex on band 3 with 2x2 pixel block, at scale 8.
    spfeas -i image.tif -o out_dir -bp 3 --block 2 --scales 8 -tr pantex

    # Compute HoG and LBP on band 1 with 4x4 pixel block, at scales 16 and 32.
    spfeas -i image.tif -o out_dir -bp 1 --block 4 --scales 16 32 -tr hog lbp

    # Compute the mean NDVI, with a 16-bit image that is scaled to 0-10,000. The `image_max`
    #   parameter ensures scaling across images.
    spfeas -i image.tif -o out_dir --equalize_adapt --image_max 10000 -tr ndvi

    # Compute Structural Feature Sets on band 4, with pre-smoothing
    spfeas -i image.tif -o out_dir -bp 4 -sfs_th 10 -tr sfs --smooth 5

    """)


def _options():

    colorama.init()

    text_lines = [Fore.GREEN + Style.BRIGHT + 'dmp' + Style.RESET_ALL + '     -- Differential morphological profiles (2 x n scales)',
                  Fore.GREEN + Style.BRIGHT + 'evi2' + Style.RESET_ALL + '    -- EVI2 mean (2 x n scales)',
                  Fore.GREEN + Style.BRIGHT + 'fourier' + Style.RESET_ALL + ' -- Fourier transform (2 x n scales)',
                  Fore.GREEN + Style.BRIGHT + 'gabor' + Style.RESET_ALL + '   -- Gabor filter bank (n scales x 2 x kernels(Default=24))',
                  Fore.GREEN + Style.BRIGHT + 'gndvi' + Style.RESET_ALL + '   -- GNDVI mean (2 x n scales)',
                  Fore.GREEN + Style.BRIGHT + 'grad' + Style.RESET_ALL + '    -- Edge gradient magnitude (2 x n scales)',
                  Fore.GREEN + Style.BRIGHT + 'hog' + Style.RESET_ALL + '     -- Histogram of Oriented Gradients (5 (max,mean,variance,skew,kurtosis) x n scales)',
                  Fore.RED + Style.BRIGHT + 'hough' + Style.RESET_ALL + '   -- Local line statistics from Probabilistic Hough Transform (4 x n scales)' + Fore.RED + ' **Currently out of order**' + Style.RESET_ALL,
                  Fore.GREEN + Style.BRIGHT + 'lac' + Style.RESET_ALL + '     -- Lacunarity (n scales)',
                  Fore.GREEN + Style.BRIGHT + 'lbp' + Style.RESET_ALL + '     -- Local Binary Patterns (59 x n scales)',
                  Fore.GREEN + Style.BRIGHT + 'lbpm' + Style.RESET_ALL + '    -- Local Binary Patterns moments (5 (max,mean,variance,skew,kurtosis) x n scales)',
                  Fore.GREEN + Style.BRIGHT + 'lsr' + Style.RESET_ALL + '     -- Line support regions (3 x n scales)',
                  Fore.GREEN + Style.BRIGHT + 'mean' + Style.RESET_ALL + '    -- Local inverse distance weighted mean and variance (2 x n scales)',
                  Fore.GREEN + Style.BRIGHT + 'ndvi' + Style.RESET_ALL + '    -- NDVI mean (2 x n scales)',
                  Fore.GREEN + Style.BRIGHT + 'pantex' + Style.RESET_ALL + '  -- Built-up presence index (n scales)',
                  Fore.GREEN + Style.BRIGHT + 'orb' + Style.RESET_ALL + '     -- Oriented BRIEF key point pyramid histogram (5 (max,mean,variance,skew,kurtosis) x n scales)',
                  Fore.GREEN + Style.BRIGHT + 'saliency' + Style.RESET_ALL + '-- Saliency features (2 x n scales)',
                  Fore.GREEN + Style.BRIGHT + 'sfs' + Style.RESET_ALL + '     -- Structural Feature Sets (6 (max,min,mean,w-mean,std,max ratio) x n scales)',
                  Fore.RED + Style.BRIGHT + 'surf' + Style.RESET_ALL + '    -- SURF key point descriptors (4 x n scales)' + Fore.RED + ' **Currently out of order**' + Style.RESET_ALL]

    for text_line in text_lines:
        logger.info(text_line)

    sys.exit(Style.RESET_ALL)


def _raster_options():

    print('=========  ======')
    print('Extension  Format')
    print('=========  ======')

    for k, v in viewitems(DRIVER_DICT):
        print('{:9}  {}'.format(k, v))

    sys.exit()


def _version():

    from . import __version__

    sys.exit(__version__)


def main():

    colorama.init()

    parser = argparse.ArgumentParser(description=Fore.GREEN + Style.BRIGHT + 'Contextual image features'
                                                 + Style.RESET_ALL,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-e', '--examples', dest='examples', action='store_true', help='Show usage examples and exit')
    parser.add_argument('-i', '--input', dest='input', help='The input image', default=None)
    parser.add_argument('-o', '--output', dest='output', help='The output directory', default=None)
    parser.add_argument('-bp', '--band-positions', dest='band_positions', help='The bands to process',
                        default=[1], type=int, nargs='+')
    parser.add_argument('--rgb', dest='use_rgb',
                        help='Whether to use the RGB spectrum 3-band average (*overrides -bp/--band-positions and reduces the visible spectrum to 1 channel)',
                        action='store_true')
    parser.add_argument('--vis-order', dest='vis_order',
                        help='The visible spectrum (red, green, blue) band order (Only required with -tr saliency)',
                        default='bgr')
    parser.add_argument('--sensor', dest='sat_sensor', help='The satellite sensor (--input)',
                        default='Quickbird',
                        choices=utils.SUPPORTED_SENSORS)
    parser.add_argument('--format', dest='format', help='The output raster format',
                        default='GTiff',
                        choices=list(DRIVER_DICT.values()))
    parser.add_argument('--block', dest='block', help='The block size', default=2, type=int)
    parser.add_argument('--scales', dest='scales', help='The scales', default=[8], type=int, nargs='+')
    parser.add_argument('-tr', '--triggers', dest='triggers', help='The feature triggers', default=['mean'],
                        nargs='+', choices=['dmp', 'fourier', 'gabor', 'grad', 'hog', 'lac',
                                            'lbpm', 'lsr', 'mean', 'orb',
                                            'pantex', 'saliency', 'seg', 'sfs'] +
                                           [vi.lower() for vi in utils.SUPPORTED_VIS])
    parser.add_argument('-lth', '--hline-threshold', dest='hline_threshold', help='The Hough line threshold',
                        default=40, type=int)
    parser.add_argument('-mnl', '--hline-min', dest='hline_min', help='The Hough line minimum length',
                        default=10, type=int)
    parser.add_argument('-lgp', '--hline-gap', dest='hline_gap', help='The Hough line gap',
                        default=2, type=int)
    parser.add_argument('--weight', dest='weight', help='Whether to weight PanTex by mean DN', action='store_true')
    parser.add_argument('--sfs-th', dest='sfs_threshold', help='The SFS stopping threshold',
                        default=40, type=int)
    parser.add_argument('--sfs-skip', dest='sfs_skip', help='The SFS angle skip factor',
                        default=4, type=int)
    parser.add_argument('--sfs-rs', dest='sfs_resample', help='The SFS resample size',
                        default=0., type=float)
    parser.add_argument('--lac-r', dest='lac_r', help='The lacunarity box r parameter', default=2, type=int)
    parser.add_argument('--smooth', dest='smooth', help='The smoothing kernel size', default=0, type=int)
    parser.add_argument('--image-min', dest='image_min', help='A user-defined image minimum', default=-999.0, type=float)
    parser.add_argument('--image-max', dest='image_max', help='A user-defined image maximum', default=-999.0, type=float)
    parser.add_argument('--equalize', dest='equalize', help='Whether to do histogram equalization', action='store_true')
    parser.add_argument('--equalize-adapt', dest='equalize_adapt',
                        help='Whether to do adaptive histogram equalization', action='store_true')
    parser.add_argument('--visualize', dest='visualize', help='Whether to visualize', action='store_true')
    parser.add_argument('--convert', dest='convert', help='Whether to convert the feature stack', action='store_true')
    parser.add_argument('--stack', dest='stack', help='Whether to stack features', action='store_true')
    parser.add_argument('--full-path', dest='full_path',
                        help='Whether to use full path names in the VRT composite (otherwise uses relative paths)',
                        action='store_true')
    parser.add_argument('--stack-only', dest='stack_only', help='Whether to only stack features', action='store_true')
    parser.add_argument('--neighbors', dest='neighbors', help='Whether to add features as neighbors',
                        action='store_true')
    parser.add_argument('--n-jobs', dest='n_jobs', help='The number of parallel jobs for sections',
                        default=-1, type=int)
    parser.add_argument('--sect-size', dest='section_size', help='The section size', default=1000, type=int)
    parser.add_argument('--gdal-cache', dest='gdal_cache', help='The GDAL cache size (MB)', default=256, type=int)
    parser.add_argument('--reset', dest='reset', help='Whether to reset section memory', action='store_true')
    parser.add_argument('--overwrite', dest='overwrite', help='Whether to overwrite output files', action='store_true')
    parser.add_argument('--overviews', dest='overviews', help='Whether to build pyramid overviews for the VRT mosaic',
                        action='store_true')
    parser.add_argument('--options', dest='options', help='Whether to show trigger options', action='store_true')
    parser.add_argument('--raster-options', dest='raster_options',
                        help='Whether to show available raster formats for writing', action='store_true')
    parser.add_argument('--version', dest='version', help='Whether to show the SpFeas version', action='store_true')

    args = parser.parse_args()

    if args.examples:
        _examples()

    if args.options:
        _options()

    if args.raster_options:
        _raster_options()

    if args.version:
        _version()

    logger.info('\nStart date & time --- (%s)\n' % time.asctime(time.localtime(time.time())))

    start_time = time.time()

    spatial_features(args.input,
                     args.output,
                     format=args.format,
                     band_positions=args.band_positions,
                     block=args.block,
                     scales=args.scales,
                     triggers=args.triggers,
                     hline_threshold=args.hline_threshold,
                     hline_min=args.hline_min,
                     hline_gap=args.hline_gap,
                     weight=args.weight,
                     sfs_threshold=args.sfs_threshold,
                     sfs_skip=args.sfs_skip,
                     sfs_resample=args.sfs_resample,
                     smooth=args.smooth,
                     equalize=args.equalize,
                     equalize_adapt=args.equalize_adapt,
                     visualize=args.visualize,
                     convert=args.convert,
                     use_rgb=args.use_rgb,
                     vis_order=args.vis_order,
                     sat_sensor=args.sat_sensor,
                     stack=args.stack,
                     full_path=args.full_path,
                     stack_only=args.stack_only,
                     neighbors=args.neighbors,
                     n_jobs=args.n_jobs,
                     reset=args.reset,
                     image_min=args.image_min,
                     image_max=args.image_max,
                     lac_r=args.lac_r,
                     section_size=args.section_size,
                     gdal_cache=args.gdal_cache,
                     overwrite=args.overwrite,
                     overviews=args.overviews)

    logger.info('\nEnd data & time -- (%s)\nTotal processing time -- (%.2gs)\n' %
                (time.asctime(time.localtime(time.time())), (time.time() - start_time)))


if __name__ == '__main__':
    main()
