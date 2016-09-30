#!/usr/bin/env python

import os
import time
import platform
import copy

from .sputilities import parameter_checks, scale_fea_check, stack_features

# NumPy
try:
    import numpy as np
except ImportError:
    raise ImportError('NumPy must be installed')


def run(parameter_object):

    """
    Args:
        input_image, output_dir, band_positions=[1], rgb2gray=None, block_size=2, scales=[8], triggers=['mean'],
        threshold=20, min_len=10, line_gap=2, weighted=False, sfs_thresh=80, resamp_sfs=0., n_angles=8,
        equalize=False, equalize_adapt=False, smooth=0, visualize=False, convert_stk=False, gdal_cache=256,
        do_pca=False, stack_feas=True, stack_only=False, band_red=3, band_nir=4, neighbors=False, n_jobs=-1,
        reset_sects=False, image_max=0, lac_r=2, section_size=8000, chunk_size=512
    """

    parameter_checks(parameter_object)

    # Setup the log file.
    if os.path.isfile(parameter_object.log_txt):

        with open(parameter_object.log_txt, 'rb') as log_txt_wr:
            log_hist = log_txt_wr.readlines()

        log_txt_wr = open(parameter_object.log_txt, 'wb')
        log_txt_wr.writelines(log_hist)

    else:
        log_txt_wr = open(parameter_object.log_txt, 'wb')

    log_txt_wr.writelines("""\
    =================================================
    Start date & time --- ({})
    =================================================
    Input image: {}
    Output directory: {}
    Bands: {}
    Smoothing: {:d}
    Block size: {:d}
    Scales: {}
    Feature triggers: {}
    SFS stopping threshold: {:d}
    SFS angles: {:d}
    Red band position: {:d}
    NIR band position: {:d}
    {} compute features as neighbors
    {} perform histogram equalization
    {} perform adaptive histogram equalization
    """.format(time.asctime(time.localtime(time.time())), parameter_object.input_image, parameter_object.output_dir,
               parameter_object.rgb2write, parameter_object.smooth, parameter_object.block_size,
               ','.join([str(bpos) for bpos in parameter_object.scales]), ','.join(parameter_object.triggers),
               parameter_object.sfs_thresh, parameter_object.n_angles, parameter_object.band_red,
               parameter_object.band_nir, parameter_object.write_neighbors, parameter_object.write_equalize,
               parameter_object.write_equalize_adapt))

    log_txt_wr.close()

    if platform.system() == 'Windows':

        new_feas_list = '{}/{}_win_feas_list.txt'.format(parameter_object.output_dir, parameter_object.f_base)

        win_feas_list_o = open(new_feas_list, 'w')

    else:
        new_feas_list = []

    if parameter_object.stack_only:

        for trigger in parameter_object.triggers:

            # output features folder
            feas_dir = '{}/{}'.format(parameter_object.output_dir, trigger)

            if isinstance(parameter_object.rgb2gray, str):
                parameter_object.band_positions = [parameter_object.rgb2gray.lower()]

            for band_p in parameter_object.band_positions:

                # get feature names
                obds = 1
                for scale in parameter_object.scales:

                    for feature in xrange(1, parameter_object.features_dict[trigger]+1):

                        out_img, out_img_base = scale_fea_check(trigger, feas_dir, band_p, scale, feature,
                                                                parameter_object)

                        # skip the feature if it doesn't exist
                        if not os.path.isfile(out_img):
                            continue

                        # append new features to a list to stack
                        if platform.system() == 'Windows':
                            win_feas_list_o.write('%s\n' % out_img)
                        else:
                            new_feas_list.append(out_img)

                        obds += 1

        if platform.system() == 'Windows':
            win_feas_list_o.close()

        # If prompted, stack features only.
        out_vrt = stack_features(parameter_object, new_feas_list)

    else:
