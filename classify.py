#!/usr/bin/env python

"""
@authors: Jordan Graesser
Date Created: 10/1/2016
"""

import time
import argparse
import ast

from classification import classification
# from classification import classification_r


class Classify(object):

    def __init__(self):

        self.cl = classification()

    def split(self, lc_samples, **kwargs):
        self.cl.split_samples(lc_samples, **kwargs)

    def construct(self, **kwargs):
        self.cl.construct_model(**kwargs)

    def predict(self, input_image, output_image, **kwargs):
        self.cl.predict(input_image, output_image, **kwargs)


def _examples():
    return


def main():

    parser = argparse.ArgumentParser(description='Image classification',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-e', '--examples', dest='examples', action='store_true', help='Show usage examples and exit')
    parser.add_argument('-i', '--input', dest='input_image', help='The input image to classify', default=None)
    parser.add_argument('-o', '--output', dest='output_image', help='The output image', default=None)
    parser.add_argument('-s', '--samples', dest='lc_samples', help='The input land cover samples', default=None)
    parser.add_argument('--perc-samp', dest='perc_samp', help='The percentage to sample', default=.9, type=float)
    parser.add_argument('--perc-samp-each', dest='perc_samp_each',
                        help='The number or percentage of each land cover class to sample',
                        default=0., type=float)
    parser.add_argument('--scale', dest='scale_data', help='Whether to scale the data', action='store_true')
    parser.add_argument('--class-subs', dest='class_subs', help='A dictionary of class subs', default=None)
    parser.add_argument('--labs-type', dest='labs_type', help='The data type of the labels', default='int',
                        choices=['int', 'float'])
    parser.add_argument('--recode-dict', dest='recode_dict', help='A dictionary of land cover recode pairs',
                        default=None)
    parser.add_argument('--classes2remove', dest='classes2remove', help='A list of land cover classes to remove',
                        default=[], type='int', nargs='+')
    parser.add_argument('--sample-weight', dest='sample_weight', help='A list of sample weights', default=[],
                        type=float, nargs='+')
    parser.add_argument('--ignore_feas', dest='ignore_feas', help='A list of features to ignore', default=[],
                        type=int, nargs='+')
    parser.add_argument('--use-xy', dest='use_xy', help='Whether to use x, y data', action='store_true')
    parser.add_argument('--stratified', dest='stratified', help='Whether to spatially stratify samples',
                        action='store_true')
    parser.add_argument('--spacing', dest='spacing', help='The stratification sampling', default=1000., type=float)
    parser.add_argument('--x-label', dest='x_label', help='The x column label', default='X')
    parser.add_argument('--y-label', dest='y_label', help='The y column label', default='Y')
    parser.add_argument('--response-label', dest='response_label', help='The response column label', default='response')

    parser.add_argument('--input-model', dest='input_model', help='A model to load', default=None)
    parser.add_argument('--output-model', dest='output_model', help='A model to save', default=None)
    parser.add_argument('--classifier-info', dest='classifier_info', help='The classification parameters',
                        default="{'classifier': 'RF'}")
    parser.add_argument('--class-weight', dest='class_weight', help='Individual class weights', default=None,
                        choices=['percent', 'inverse'])
    parser.add_argument('--cal-proba', dest='calibrate_proba', help='Whether to calibrate posterior probabilities',
                        action='store_true')
    parser.add_argument('--be-quiet', dest='be_quiet', help='Whether to be quiet', action='store_true')
    parser.add_argument('--jobs', dest='n_jobs', help='The number of parallel jobs', default=-1, type=int)
    parser.add_argument('--band-check', dest='band_check', help='The band to check for no data', default=-1, type=int)
    parser.add_argument('--mask-background', dest='mask_background',
                        help='An image to use as a background mask, applied post-classification', default=None)
    parser.add_argument('--background-band', dest='background_band',
                        help='The band from --mask-background to use for null background value', default=2, type=int)
    parser.add_argument('--background-value', dest='background_value',
                        help='The background value in --mask-background', default=0, type=int)
    parser.add_argument('--min-observations', dest='minimum_observations',
                        help='A minimum number of observations in --mask-background to be recoded to 0',
                        default=0, type=int)
    parser.add_argument('--observation-band', dest='observation_band',
                        help='The band position in --mask-background of the --min-observations counts',
                        default=0, type=int)

    args = parser.parse_args()

    if args.examples:
        _examples()

    print('\nStart date & time --- (%s)\n' % time.asctime(time.localtime(time.time())))

    start_time = time.time()

    clo = Classify()

    clo.split(args.lc_samples, perc_samp=args.perc_samp, perc_samp_each=args.perc_samp_each,
              scale_data=args.scale_data, class_subs=ast.literal_eval(args.class_subs),
              labs_type=args.labs_type, recode_dict=ast.literal_eval(args.recode_dict),
              classes2remove=args.classes2remove, sample_weight=args.sample_weight,
              ignore_feas=args.ignore_feas, use_xy=args.use_xy, stratified=args.stratified,
              spacing=args.spacing, x_label=args.x_label, y_label=args.y_label, response_label=args.response_label)

    clo.construct(input_model=args.input_model, output_model=args.output_model,
                  classifier_info=ast.literal_eval(args.classifier_info),
                  class_weight=args.class_weight, calibrate_proba=args.calibrate_proba,
                  be_quiet=args.be_quiet, n_jobs=args.n_jobs)

    clo.predict(args.input_image, args.output_image, scale_data=args.scale_data,
                band_check=args.band_check, ignore_feas=args.ignore_feas, use_xy=args.use_xy,
                in_model=args.input_model, mask_background=args.mask_background,
                background_band=args.background_band, background_value=args.background_value,
                minimum_observations=args.minimum_observations, observation_band=args.observation_band,
                row_block_size=1024, col_block_size=1024, n_jobs=args.n_jobs, gdal_cache=256)

    print('\nEnd data & time -- (%s)\nTotal processing time -- (%.2gs)\n'
          % (time.asctime(time.localtime(time.time())), (time.time() - start_time)))


if __name__ == '__main__':
    main()
