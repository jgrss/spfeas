#!/usr/bin/env python

import os
import sys
import platform
import subprocess
import copy

import raster_tools

import numpy as np
import numexpr as ne


def parameter_checks(parameter_object):

    # Ensure the input image exists.
    if not os.path.isfile(parameter_object.input_image):
        raise OSError('The input image does not exist.')

    # Ensure the block size is smaller than
    #   the maximum scale size.
    if parameter_object.block > np.max(parameter_object.scales):
        raise ValueError('The block size (block_size) cannot be greater than the maximum scale <scales>.')

    # Ensure the block size is even if
    #   the scales are even.
    if (parameter_object.block % 2 != 0) and (parameter_object.scales[0] % 2 == 0):
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
                                                                    band_pos_str, parameter_object.block,
                                                                    scale, feature, parameter_object.f_ext)

    elif 10 <= feature < 100:

        out_img = '{}/{}_{}_bd{}_blk{:d}_sc{:d}_fea0{:d}{}'.format(feas_dir, parameter_object.f_base, trigger,
                                                                   band_pos_str, parameter_object.block,
                                                                   scale, feature, parameter_object.f_ext)

    else:

        out_img = '{}/{}_{}_bd{}_blk{:d}_sc{:d}_fea{:d}{}'.format(feas_dir, parameter_object.f_base, trigger,
                                                                  band_pos_str, parameter_object.block,
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
                                                                            parameter_object.block,
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
                                                              parameter_object.block,
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


def set_feas_dir(parameter_object, trigger):

    feas_dir = '{}/{}'.format(parameter_object.output_dir, trigger)

    if not os.path.isdir(feas_dir):
        os.makedirs(feas_dir)

    if isinstance(parameter_object.rgb2gray, str):
        parameter_object.band_positions = [parameter_object.rgb2gray.lower()]

    return feas_dir, parameter_object


def convert_rgb2gray(i_info, j_sect, i_sect, n_cols, n_rows, rgb='BGR', stats=False):

    """
    Convert RGB to gray scale array

    0.2125 R + 0.7154 G + 0.0721 B

    Args:
        i_info (object of rinfo)
        j_sec (int): Starting column index.
        i_sect (int): Starting row index.
        n_cols (int)
        n_rows (int)
        rgb (Optional[str]): The order of the visible spectrum bands. Many RGB images or photos
            are stored as red, green, blue. However, with multi-band satellite imagery common storage
            is blue, green, red. Though it may be unorthodox, the default here is blue, green, red, or 'BGR'.
        stats (Optional[bool]
    """

    print '\nConverting RGB to grayscale ...'

    if stats:
        luminosity = np.zeros((i_info.rows, i_info.cols), dtype='float32')
    else:
        luminosity = np.zeros((n_rows, n_cols), dtype='float32')

    gray_min, gray_max = 0, 0

    if rgb == 'RGB':
        coeff_dict = {1: .2125, 2: .7154, 3: .0721}
    elif rgb == 'BGR':
        coeff_dict = {1: .0721, 2: .7154, 3: .2125}

    for band_p in xrange(1, 4):

        coeff = coeff_dict[band_p]

        if stats:
            temp_bd_sect = i_info.mparray(bands2open=band_p, d_type='float32')
        else:
            temp_bd_sect = i_info.mparray(bands2open=band_p, i=i_sect, j=j_sect,
                                          rows=n_rows, cols=n_cols, d_type='float32')

        luminosity = ne.evaluate('(temp_bd_sect * coeff) + luminosity')

    if stats:

        gray_min = luminosity.min()
        gray_max = luminosity.max()

        return None, gray_min, gray_max

    else:
        return luminosity, gray_min, gray_max


def get_sect_chunk_size(img_info, max_section_size):

    # get section and chunk size
    if img_info.rows <= max_section_size:
        sect_row_size = copy.copy(img_info.rows)
    else:
        sect_row_size = max_section_size

    if img_info.cols <= max_section_size:
        sect_col_size = copy.copy(img_info.cols)
    else:
        sect_col_size = max_section_size

    return sect_row_size, sect_col_size


def get_adj_info(meta_info, i_info, parameter_object):

    """
    Get the adjusted output image information

    Args:
        meta_info -- MapPy class object
        i_info -- MapPy class object
        max_sc -- int
            : maximum scale used
        blk_size -- int
            : block size to write to

    Returns:
        Updated MapPy class information object
    """

    i_info.rows = len([i for i in xrange(0, meta_info.rows, parameter_object.block)])
    i_info.cols = len([i for i in xrange(0, meta_info.cols, parameter_object.block)])

    i_info.left = meta_info.left
    i_info.top = meta_info.top
    i_info.right = meta_info.right
    i_info.bottom = meta_info.bottom

    i_info.cellY = float(parameter_object.block) * meta_info.cellY
    i_info.cellX = float(parameter_object.block) * meta_info.cellX

    return i_info


def create_band(meta_info, out_img, parameter_object, out_bands, blocks=True):

    """
    Args:
        meta_info (object)
        out_img (str): The output image name.
        blk (int): The size of block to write to.
        scs (list): A list of scales to use.
        out_bands (int)
        blocks (bool)

    Returns:
        None, creates raster as ``out_img``.
    """

    i_info = meta_info.copy()

    if blocks:
        i_info = get_adj_info(meta_info, i_info, parameter_object)

    i_info.bands = out_bands
    i_info.storage = 'float32'

    out_rst = raster_tools.create_raster(out_img, i_info)

    out_rst.close_file()
    out_rst = None
