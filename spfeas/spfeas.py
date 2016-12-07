#!/usr/bin/env python

"""
@authors: Jordan Graesser
Date Created: 9/29/2016
"""

import os
import sys
import argparse
import time

from . import spprocess

try:
    import colorama
    from colorama import Fore, Style
except ImportError:
    raise ImportError('Colorama must be installed')


class SPParameters(object):

    def __init__(self, input_image, output_dir):

        self.input_image = input_image
        self.output_dir = output_dir

    def set_defaults(self, **kwargs):
        
        for k, v in kwargs.iteritems():
            setattr(self, k, v)

        # Set the features dictionary.
        self.features_dict = dict(mean=1, pantex=1, ctr=1, lsr=3, hough=4, hog=4, lbp=62,
                                  lbpm=4, gabor=2 * 8, surf=4, seg=1, fourier=2, sfs=5, ndvi=1,
                                  objects=1, dmp=1, xy=2, lac=1)

        # Set the output bands based on the trigger.
        self.out_bands_dict = {'mean': len(self.scales) * self.features_dict['mean'],
                               'pantex': len(self.scales) * self.features_dict['pantex'],
                               'ctr': len(self.scales) * self.features_dict['ctr'],
                               'lsr': len(self.scales) * self.features_dict['lsr'],
                               'hough': len(self.scales) * self.features_dict['hough'],
                               'hog': len(self.scales) * self.features_dict['hog'],
                               'lbp': len(self.scales) * self.features_dict['lbp'],
                               'lbpm': len(self.scales) * self.features_dict['lbpm'],
                               'gabor': len(self.scales) * self.features_dict['gabor'],
                               'surf': len(self.scales) * self.features_dict['surf'],
                               'seg': len(self.scales) * self.features_dict['seg'],
                               'fourier': len(self.scales) * self.features_dict['fourier'],
                               'sfs': len(self.scales) * self.features_dict['sfs'],
                               'ndvi': len(self.scales) * self.features_dict['ndvi'],
                               'objects': len(self.scales) * self.features_dict['objects'],
                               'dmp': len(self.scales) * self.features_dict['dmp'],
                               'xy': len(self.scales) * self.features_dict['xy'],
                               'lac': len(self.scales) * self.features_dict['lac']}

        # Update the feature dictionary for feature neighbors.
        if self.neighbors:

            for key, val in self.features_dict.iteritems():
                self.features_dict[key] *= 5

        self.d_name, self.f_name = os.path.split(self.input_image)
        self.f_base, __ = os.path.splitext(self.f_name)

        self.f_ext = '.tif'

        # The status dictionary file.
        self.status_dict_txt = os.path.join(self.output_dir, '{}_status.yaml'.format(self.f_base))

        # The log file.
        self.log_txt = os.path.join(self.output_dir, '{}_log.txt'.format(self.f_base))

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

    def run(self):
        spprocess.run(self)
        

def spatial_features(input_image, output_dir, **kwargs):
    
    spp = SPParameters(input_image, output_dir)
    
    spp.set_defaults(**kwargs)
    
    spp.run()


def _examples():

    sys.exit("""\

    # Compute PanTex on band 3 with 2x2 pixel block, at scale 8.
    spfeas -i /image.tif -o /out_dir -bp 3 --block 2 --scales 8 -tr pantex

    # Compute HoG and LBP on bands 1, 2, and 3 with 4x4 pixel block, at scales 16 and 32.
    spfeas -i /image.tif -o /out_dir -bp 1 2 3 --block 4 --scales 16 32 -tr hog lbp

    # Compute the mean NDVI, with a 16-bit image that is scaled to 0-10,000. The `image_max`
    #   parameter ensures scaling across images.
    spfeas -i /image.tif -o /out_dir --equalize_adapt --image_max 10000 -tr ndvi

    # Compute Structural Feature Sets on band 4, with pre-smoothing
    spfeas -i /image.tif -o /out_dir -bp 4 -sfs_th 10 -tr sfs --smooth 5

    """)


def _options():

    colorama.init()

    text_lines = [Fore.GREEN + Style.BRIGHT + 'ctr' + Style.RESET_ALL + '     -- Copy scale centers',
                  Fore.GREEN + Style.BRIGHT + 'dmp' + Style.RESET_ALL + '     -- Differential morphological profiles (n scales)' + Fore.RED + ' **EXPERIMENTAL**',
                  Fore.GREEN + Style.BRIGHT + 'fourier' + Style.RESET_ALL + ' -- Fourier transform (n scales x 2)',
                  Fore.GREEN + Style.BRIGHT + 'gabor' + Style.RESET_ALL + '   -- Gabor filter bank (n scales x 2 x kernels(Default=24))',
                  Fore.GREEN + Style.BRIGHT + 'hog' + Style.RESET_ALL + '     -- Histogram of Oriented Gradients (4 (mean,var,skew,kurtosis) x n scales)',
                  Fore.GREEN + Style.BRIGHT + 'hough' + Style.RESET_ALL + '   -- Local line statistics from Probabilistic Hough Transform (4 x n scales)',
                  Fore.GREEN + Style.BRIGHT + 'lac' + Style.RESET_ALL + '     -- Lacunarity (n scales)',
                  Fore.GREEN + Style.BRIGHT + 'lbp' + Style.RESET_ALL + '     -- Local Binary Patterns (59 x n scales)',
                  Fore.GREEN + Style.BRIGHT + 'lbpm' + Style.RESET_ALL + '    -- Local Binary Patterns moments (4 x n scales)',
                  Fore.GREEN + Style.BRIGHT + 'lsr' + Style.RESET_ALL + '     -- Line support regions (3 x n scales)',
                  Fore.GREEN + Style.BRIGHT + 'mean' + Style.RESET_ALL + '    -- Local mean (n scales)',
                  Fore.GREEN + Style.BRIGHT + 'ndvi' + Style.RESET_ALL + '    -- NDVI mean (n scales)',
                  Fore.GREEN + Style.BRIGHT + 'pantex' + Style.RESET_ALL + '  -- Built-up presence index (n scales)',
                  Fore.GREEN + Style.BRIGHT + 'sfs' + Style.RESET_ALL + '     -- Structural Feature Sets (4)',
                  Fore.GREEN + Style.BRIGHT + 'surf' + Style.RESET_ALL + '    -- SURF key point descriptors (4 x n scales)' + Fore.RED + ' **Currently out of order**']

    for text_line in text_lines:
        print text_line

    sys.exit(Style.RESET_ALL)


def main():

    colorama.init()

    parser = argparse.ArgumentParser(description=Fore.GREEN + Style.BRIGHT + 'Contextual image features'
                                                 + Style.RESET_ALL,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-e', '--examples', dest='examples', action='store_true', help='Show usage examples and exit')
    parser.add_argument('-i', '--input', dest='input', help='The input image', default=None)
    parser.add_argument('-o', '--output', dest='output', help='The output directory', default=None)
    parser.add_argument('-bp', '--band-positions', dest='band_positions', help='The band to process', default=[1],
                        type=int, nargs='+')
    parser.add_argument('--rgb', dest='use_rgb', help='Whether to use the full RGB spectrum in place of -bp',
                        action='store_true')
    parser.add_argument('--block', dest='block', help='The block size', default=2, type=int)
    parser.add_argument('--scales', dest='scales', help='The scales', default=[8], type=int, nargs='+')
    parser.add_argument('-tr', '--triggers', dest='triggers', help='The feature triggers', default=['mean'],
                        nargs='+', choices=['dmp', 'fourier', 'gabor', 'hog', 'lac',
                                            'lbp', 'lbpm', 'lsr', 'mean', 'ndvi', 'pantex', 'sfs'])
    parser.add_argument('-lth', '--hline-threshold', dest='hline_threshold', help='The Hough line threshold',
                        default=20, type=int)
    parser.add_argument('-mnl', '--hline-min', dest='hline_min', help='The Hough line minimum length',
                        default=10, type=int)
    parser.add_argument('-lgp', '--hline-gap', dest='hline_gap', help='The Hough line gap',
                        default=2, type=int)
    parser.add_argument('--weight', dest='weight', help='Whether to weight PanTex by DN', action='store_true')
    parser.add_argument('-sfs-th', '--sfs-threshold', dest='sfs_threshold', help='The SFS stopping threshold',
                        default=80, type=int)
    parser.add_argument('-sfs-rs', '--sfs-resample', dest='sfs_resample', help='The SFS resample size',
                        default=0., type=float)
    parser.add_argument('-sfs-ag', '--sfs-angles', dest='sfs_angles', help='The SFS angles',
                        default=8, type=int, choices=[8, 16])
    parser.add_argument('--lac-r', dest='lac_r', help='The lacunarity box r parameter', default=2, type=int)
    parser.add_argument('--smooth', dest='smooth', help='The smoothing kernel size', default=0, type=int)
    parser.add_argument('--image-max', dest='image_max', help='A user-defined image maximum', default=0, type=int)
    parser.add_argument('--equalize', dest='equalize', help='Whether to do histogram equalization', action='store_true')
    parser.add_argument('--equalize-adapt', dest='equalize_adapt',
                        help='Whether to do adaptive histogram equalization', action='store_true')
    parser.add_argument('--visualize', dest='visualize', help='Whether to visualize', action='store_true')
    parser.add_argument('--pca', dest='pca', help='Whether to run PCA', action='store_true')
    parser.add_argument('--convert', dest='convert', help='Whether to convert the feature stack', action='store_true')
    parser.add_argument('--stack', dest='stack', help='Whether to stack features', action='store_true')
    parser.add_argument('--stack-only', dest='stack_only', help='Whether to only stack features', action='store_true')
    parser.add_argument('--band-red', dest='band_red', help='The red band position', default=3, type=int)
    parser.add_argument('--band-nir', dest='band_nir', help='The NIR band position', default=4, type=int)
    parser.add_argument('--neighbors', dest='neighbors', help='Whether to add features as neighbors',
                        action='store_true')
    parser.add_argument('--n-jobs', dest='n_jobs', help='The number of parallel jobs', default=1, type=int)
    parser.add_argument('--sect-size', dest='section_size', help='The section size', default=8000, type=int)
    parser.add_argument('--chunk-size', dest='chunk_size', help='The section chunk size', default=512, type=int)
    parser.add_argument('--gdal-cache', dest='gdal_cache', help='The GDAL cache size (MB)', default=256, type=int)
    parser.add_argument('--reset', dest='reset', help='Whether to reset section memory', action='store_true')
    parser.add_argument('--options', dest='options', help='Whether to show trigger options', action='store_true')

    args = parser.parse_args()

    if args.examples:
        _examples()

    if args.options:
        _options()

    print('\nStart date & time --- (%s)\n' % time.asctime(time.localtime(time.time())))

    start_time = time.time()

    spatial_features(args.input, args.output, band_positions=args.band_positions, block=args.block,
                     scales=args.scales, triggers=args.triggers, hline_threshold=args.hline_threshold,
                     hline_min=args.hline_min, hline_gap=args.hline_gap, weight=args.weight,
                     sfs_threshold=args.sfs_threshold, sfs_resample=args.sfs_resample, sfs_angles=args.sfs_angles,
                     smooth=args.smooth, equalize=args.equalize, equalize_adapt=args.equalize_adapt,
                     visualize=args.visualize, convert=args.convert, do_pca=args.pca, use_rgb=args.use_rgb,
                     stack=args.stack, stack_only=args.stack_only, band_red=args.band_red, band_nir=args.band_nir,
                     neighbors=args.neighbors, n_jobs=args.n_jobs, reset=args.reset, image_max=args.image_max,
                     lac_r=args.lac_r, section_size=args.section_size, chunk_size=args.chunk_size,
                     gdal_cache=args.gdal_cache)

    print('\nEnd data & time -- (%s)\nTotal processing time -- (%.2gs)\n'
          % (time.asctime(time.localtime(time.time())), (time.time() - start_time)))


if __name__ == '__main__':
    main()
