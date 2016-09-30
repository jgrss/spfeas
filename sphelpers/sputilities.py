#!/usr/bin/env python

import os
import platform
import subprocess

import numpy as np


def parameter_checks(parameter_object):

    # Ensure the input image exists.
    if not os.path.isfile(parameter_object.input_image):
        raise OSError('The input image does not exist.')

    # Ensure the block size is smaller than
    #   the maximum scale size.
    if parameter_object.block_size > np.max(parameter_object.scales):
        raise ValueError('The block size (block_size) cannot be greater than the maximum scale <scales>.')

    # Ensure the block size is even if
    #   the scales are even.
    if (parameter_object.block_size % 2 != 0) and (parameter_object.scales[0] % 2 == 0):
        raise ValueError('Please pass an even number for the <block_size> parameter if your <scales> are also even.')

    # Ensure the correct smoothing parameters.
    if parameter_object.smooth > 0:

        if parameter_object.smooth <= 2:
            raise ValueError('The <smooth> parameter should be 3 or greater.')

        if parameter_object.smooth % 2 == 0:
            raise ValueError('The <smooth> parameter should be an odd number.')

    # Create the output directory.
    if not os.path.isdir(parameter_object.output_dir):

        try:
            os.makedirs(parameter_object.output_dir)
        except OSError:
            raise OSError('Could not create the output directory.')


def scale_fea_check(trigger, feas_dir, band_p, scale, feature, parameter_object):

    """
    Checks the scale and feature to set the string name.

    Returns:
        Image name as a string
    """

    band_pos_str = str(band_p)

    if band_pos_str == 'rgb' or band_pos_str == 'bgr':
        band_pos_str = '-{}'.format(band_pos_str)
    else:
        band_pos_str = '-'.join(band_pos_str)

    if feature < 10:

        out_img = '{}/{}_{}_bd{}_blk{:d}_sc{:d}_fea00{:d}{}'.format(feas_dir, parameter_object.f_base, trigger,
                                                                    band_pos_str, parameter_object.block_size,
                                                                    scale, feature, parameter_object.f_ext)

    elif 10 <= feature < 100:

        out_img = '{}/{}_{}_bd{}_blk{:d}_sc{:d}_fea0{:d}{}'.format(feas_dir, parameter_object.f_base, trigger,
                                                                   band_pos_str, parameter_object.block_size,
                                                                   scale, feature, parameter_object.f_ext)

    else:

        out_img = '{}/{}_{}_bd{}_blk{:d}_sc{:d}_fea{:d}{}'.format(feas_dir, parameter_object.f_base, trigger,
                                                                  band_pos_str, parameter_object.block_size,
                                                                  scale, feature, parameter_object.f_ext)

    out_img_d_name, out_img_f_name = os.path.split(out_img)
    out_img_f_base, out_img_f_ext = os.path.splitext(out_img_f_name)

    return out_img, out_img_f_base


def stack_features(parameter_object, new_feas_list):

    """
    Stack features
    """

    scs_str = [str(sc) for sc in parameter_object.scales]
    band_pos_str = [str(bp) for bp in parameter_object.band_positions]

    # write band list to text
    fea_list_txt = '{}/{}.{}.stk.bd{}.block{}.scales{}_fea_list.txt'.format(parameter_object.output_dir,
                                                                            parameter_object.f_base,
                                                                            '-'.join(parameter_object.triggers),
                                                                            '-'.join(band_pos_str),
                                                                            parameter_object.block_size,
                                                                            '-'.join(scs_str))

    # remove stacked VRT list
    if os.path.isfile(fea_list_txt):
        os.remove(fea_list_txt)

    fea_list_txt_wr = open(fea_list_txt, 'wb')
    fea_list_txt_wr.write('Layer,Name\n')

    if platform.system() == 'Windows':

        with open(new_feas_list) as f:

            fea_ctr = 1
            for line in f:
                fea_list_txt_wr.write('{:d},{}'.format(fea_ctr, line))
                fea_ctr += 1

        f.close()

    else:

        fea_ctr = 1
        for fea_name in new_feas_list:
            fea_list_txt_wr.write('{:d},{}\n'.format(fea_ctr, fea_name))
            fea_ctr += 1

    fea_list_txt_wr.close()

    # stack features here
    out_vrt = '{}/{}.{}.stk.bd{}.block{}.scales{}.vrt'.format(parameter_object.output_dir,
                                                              parameter_object.f_base,
                                                              '-'.join(parameter_object.triggers),
                                                              '-'.join(band_pos_str),
                                                              parameter_object.block_size,
                                                              '-'.join(scs_str))

    if os.path.isfile(out_vrt):
        os.remove(out_vrt)

    # create the stack list
    if platform.system() == 'Windows':
        com = 'gdalbuildvrt -separate -input_file_list {} {}'.format(new_feas_list, out_vrt)
    else:
        com = 'gdalbuildvrt -separate {} {}'.format(out_vrt, ' '.join(new_feas_list))

    print '\nMosaicking {:d} features ...\n'.format(len(new_feas_list))

    subprocess.call(com, shell=True)

    return out_vrt
