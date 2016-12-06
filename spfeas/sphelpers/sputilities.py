#!/usr/bin/env python

import os
import copy

from mpglue import raster_tools, vrt_builder

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
        feature_str = 'fea00'
    elif 10 <= feature < 100:
        feature_str = 'fea0'
    else:
        feature_str = 'fea'

    out_img = os.path.join(feas_dir, '{}_{}_bd{}_blk{:d}_sc{:d}_{}{:d}{}'.format(parameter_object.f_base,
                                                                                 trigger,
                                                                                 band_pos_str,
                                                                                 parameter_object.block,
                                                                                 scale,
                                                                                 feature_str,
                                                                                 feature,
                                                                                 parameter_object.f_ext))

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
    fea_list_txt = os.path.join(parameter_object.output_dir,
                                '{}.{}.stk.bd{}.block{}.scales{}_fea_list.txt'.format(parameter_object.f_base,
                                                                                      '-'.join(parameter_object.triggers),
                                                                                      '-'.join(band_pos_str),
                                                                                      parameter_object.block,
                                                                                      '-'.join(scs_str)))

    # remove stacked VRT list
    if os.path.isfile(fea_list_txt):
        os.remove(fea_list_txt)

    with open(fea_list_txt, 'wb') as fea_list_txt_wr:

        fea_list_txt_wr.write('Layer,Name\n')

        # if platform.system() == 'Windows':
        #
        #     with open(new_feas_list) as f:
        #
        #         fea_ctr = 1
        #         for line in f:
        #             fea_list_txt_wr.write('{:d},{}'.format(fea_ctr, line))
        #             fea_ctr += 1
        #
        #     f.close()
        #
        # else:

        fea_ctr = 1
        for fea_name in new_feas_list:
            fea_list_txt_wr.write('{:d},{}\n'.format(fea_ctr, fea_name))
            fea_ctr += 1

    # stack features here
    out_vrt = os.path.join(parameter_object.output_dir,
                           '{}.{}.stk.bd{}.block{}.scales{}.vrt'.format(parameter_object.f_base,
                                                                        '-'.join(parameter_object.triggers),
                                                                        '-'.join(band_pos_str),
                                                                        parameter_object.block,
                                                                        '-'.join(scs_str)))

    if os.path.isfile(out_vrt):
        os.remove(out_vrt)

    # create the stack list
    # if platform.system() == 'Windows':
    #     com = 'gdalbuildvrt -separate -input_file_list {} {}'.format(new_feas_list, out_vrt)
    # else:
    #     com = 'gdalbuildvrt -separate {} {}'.format(out_vrt, ' '.join(new_feas_list))
    #
    # print '\nMosaicking {:d} features ...\n'.format(len(new_feas_list))
    #
    # subprocess.call(com, shell=True)

    stack_dict = {}

    for ni, new_feas in enumerate(new_feas_list):
        stack_dict[str(ni+1)] = [new_feas]

    vrt_builder(stack_dict, out_vrt, force_type='float32')

    return out_vrt


def set_feas_dir(parameter_object, trigger):

    feas_dir = os.path.join(parameter_object.output_dir, trigger)

    if not os.path.isdir(feas_dir):
        os.makedirs(feas_dir)

    if isinstance(parameter_object.rgb2gray, str):
        parameter_object.band_positions = [parameter_object.rgb2gray.lower()]

    return feas_dir, parameter_object


def min_max_func(im, im_min, im_max):

    try:

        im_min = np.minimum(im_min, im.min())
        im_max = np.maximum(im_max, im.max())

    except ValueError:  # raised if `im_min` is empty.
        pass

    return im_min, im_max


def get_luminosity(im_block, rows_, cols_, rgb):

    # coeff_dict = dict(B=.0721, G=.7154, R=.2125)

    luminosity = np.zeros((rows_, cols_), dtype='float32')

    for band_p, band_l in enumerate(rgb.upper()):

        # coeff = coeff_dict[band_l]

        luminosity += im_block[band_p]

        # luminosity = ne.evaluate('(im_block_ * coeff) + luminosity')

    return luminosity / 3.


def convert_rgb2gray(i_info, j_sect, i_sect, n_rows, n_cols, rgb='BGR', stats=False):

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

    if stats:

        print '\nCalculating image min and max ...\n'

        im_min = 1000000
        im_max = -1000000

        for i_ in xrange(0, i_info.rows, 512):

            n_rows_ = raster_tools.n_rows_cols(i_, 512, i_info.rows)

            for j_ in xrange(0, i_info.cols, 512):

                n_cols_ = raster_tools.n_rows_cols(j_, 512, i_info.cols)

                im_block = i_info.mparray(bands2open=[1, 2, 3],
                                          i=i_, j=j_,
                                          rows=n_rows_, cols=n_cols_,
                                          d_type='float32')

                luminosity = get_luminosity(im_block, n_rows_, n_cols_, rgb)

                im_min, im_max = min_max_func(luminosity, im_min, im_max)

        # bp = raster_tools.BlockFunc(min_max_func, [i_info], None, None,
        #                             out_attributes=['im_min', 'im_max'],
        #                             print_statement='\nGetting image statistics ...\n',
        #                             write_array=False,
        #                             close_files=False,
        #                             be_quiet=False)
        #
        # bp.run()

        return None, im_min, im_max

    else:

        print '\nConverting {} to luminosity ...\n'.format(rgb.upper())

        im_block = i_info.mparray(bands2open=[1, 2, 3],
                                  i=i_sect, j=j_sect,
                                  rows=n_rows, cols=n_cols,
                                  d_type='float32')

        luminosity = get_luminosity(im_block, n_rows, n_cols, rgb)

        return luminosity, None, None


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

    i_info.update_info(bands=out_bands, storage='float32')

    out_rst = raster_tools.create_raster(out_img, i_info)

    out_rst.close_file()
    out_rst = None
