#!/usr/bin/env python

import os
import sys
import copy
import time
import psutil
import itertools

from mpglue import raster_tools, vrt_builder

import numpy as np

# YAML
try:
    import yaml
except ImportError:
    raise ImportError('YAML must be installed')


def write_log(parameter_object):

    # Setup the log file.
    if os.path.isfile(parameter_object.log_txt):

        with open(parameter_object.log_txt, 'rb') as log_txt_wr:
            starter = log_txt_wr.readlines()

    else:
        starter = []

    lines2write = starter + ['\n',
                             '=================================================\n',
                             'Start date & time --- ({})\n'.format(time.asctime(time.localtime(time.time()))),
                             '=================================================\n',
                             'Input image: {}\n'.format(parameter_object.input_image),
                             'Output directory: {}\n'.format(parameter_object.output_dir),
                             'Bands: {}\n'.format(parameter_object.rgb2write),
                             'Smoothing: {:d}\n'.format(parameter_object.smooth),
                             'Block size: {:d}\n'.format(parameter_object.block),
                             'Scales: {}\n'.format(','.join([str(bpos) for bpos in parameter_object.scales])),
                             'Contextual features: {}\n'.format(','.join(parameter_object.triggers)),
                             'SFS stopping threshold: {:d}\n'.format(parameter_object.sfs_threshold),
                             'Red band position: {:d}\n'.format(parameter_object.band_red),
                             'NIR band position: {:d}\n'.format(parameter_object.band_nir),
                             '{} compute features as neighbors\n'.format(parameter_object.write_neighbors),
                             '{} perform histogram equalization\n'.format(parameter_object.write_equalize),
                             '{} perform adaptive histogram equalization\n'.format(parameter_object.write_equalize_adapt)]

    with open(parameter_object.log_txt, 'wb') as log_txt_wr:
        log_txt_wr.writelines(lines2write)


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


def scale_fea_check(parameter_object):

    """
    Checks the scale and feature to set the string name.

    Returns:
        Image name as a string
    """

    band_pos_str = str(parameter_object.band_position)

    if band_pos_str == 'rgb' or band_pos_str == 'bgr':
        band_pos_str = '-{}'.format(band_pos_str)
    else:
        band_pos_str = '-'.join(band_pos_str)

    feature_str = 'fea{:03d}'.format(parameter_object.feature)

    out_img = os.path.join(parameter_object.feas_dir,
                           '{}_{}_bd{}_blk{:d}_sc{:d}_{}{}'.format(parameter_object.f_base,
                                                                   parameter_object.trigger,
                                                                   band_pos_str,
                                                                   parameter_object.block,
                                                                   parameter_object.scale,
                                                                   feature_str,
                                                                   parameter_object.f_ext))

    out_img_d_name, out_img_f_name = os.path.split(out_img)
    out_img_base, out_img_f_ext = os.path.splitext(out_img_f_name)

    parameter_object.update_info(out_img=out_img,
                                 out_img_base=out_img_base)

    return parameter_object


def stack_features(parameter_object, new_feas_list):

    """
    Stack features
    """

    for trigger in parameter_object.triggers:

        parameter_object.update_info(trigger=trigger)

        # Set the output features folder.
        parameter_object = set_feas_dir(parameter_object)

        for band_p in parameter_object.band_positions:

            parameter_object.update_info(band_position=band_p)

            # Get feature names
            obds = 1
            for scale in parameter_object.scales:

                parameter_object.update_info(scale=scale)

                for feature in xrange(1, parameter_object.features_dict[trigger]+1):

                    parameter_object.update_info(feature=feature)

                    parameter_object = scale_fea_check(parameter_object)

                    # skip the feature if it doesn't exist
                    if not os.path.isfile(parameter_object.out_img):
                        continue

                    new_feas_list.append(parameter_object.out_img)

                    obds += 1

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

        fea_list_txt_wr.write('Layer Name\n')

        for fea_ctr, fea_name in enumerate(new_feas_list):
            fea_list_txt_wr.write('{:d} {}\n'.format(fea_ctr+1, fea_name))

    # stack features here
    out_vrt = os.path.join(parameter_object.output_dir,
                           '{}.{}.stk.bd{}.block{}.scales{}.vrt'.format(parameter_object.f_base,
                                                                        '-'.join(parameter_object.triggers),
                                                                        '-'.join(band_pos_str),
                                                                        parameter_object.block,
                                                                        '-'.join(scs_str)))

    if os.path.isfile(out_vrt):
        os.remove(out_vrt)

    stack_dict = dict()

    for ni, new_feas in enumerate(new_feas_list):
        stack_dict[str(ni+1)] = [new_feas]

    print('Stacking variables ...')

    vrt_builder(stack_dict, out_vrt, force_type='float32', be_quiet=True)

    parameter_object.update_info(out_vrt=out_vrt)

    return parameter_object


def set_feas_dir(parameter_object):

    feas_dir = os.path.join(parameter_object.output_dir, parameter_object.trigger)

    parameter_object.update_info(feas_dir=feas_dir)

    if not os.path.isdir(feas_dir):
        os.makedirs(feas_dir)

    if parameter_object.use_rgb:
        parameter_object.update_info(band_positions=[parameter_object.rgb2write.lower()])

    return parameter_object


def min_max_func(im, im_min, im_max):

    try:

        im_min = np.minimum(im_min, im.min())
        im_max = np.maximum(im_max, im.max())

    except ValueError:  # raised if `im_min` is empty.
        pass

    return im_min, im_max


def get_luminosity(im_block):

    """
    Get the pixel-wise average in the visible spectrum
    """

    return im_block.mean(axis=0)


def get_layer_min_max(i_info, layers=[1, 2, 3], rgb=False, block_size=2048):

    min_max = []

    if rgb:

        layer_min = 999999.
        layer_max = -999999.

        for i in range(0, i_info.rows, block_size):
            n_rows = raster_tools.n_rows_cols(i, block_size, i_info.rows)

            for j in range(0, i_info.cols, block_size):
                n_cols = raster_tools.n_rows_cols(j, block_size, i_info.cols)

                sect = i_info.read(bands2open=layers,
                                   i=i, j=j,
                                   rows=n_rows, cols=n_cols,
                                   d_type='float32')

                sect = get_luminosity(sect)

                layer_min = min(layer_min, np.percentile(sect, 1))
                layer_max = max(layer_max, np.percentile(sect, 99))

        min_max.append((layer_min, layer_max))

    else:

        for lb in layers:

            layer_min = 999999.
            layer_max = -999999.

            for i in range(0, i_info.rows, block_size):
                n_rows = raster_tools.n_rows_cols(i, block_size, i_info.rows)

                for j in range(0, i_info.cols, block_size):
                    n_cols = raster_tools.n_rows_cols(j, block_size, i_info.cols)

                    sect = i_info.read(bands2open=lb,
                                       i=i, j=j,
                                       rows=n_rows, cols=n_cols,
                                       d_type='float32')

                    layer_min = min(layer_min, np.percentile(sect, 1))
                    layer_max = max(layer_max, np.percentile(sect, 99))

            min_max.append((layer_min, layer_max))

    return min_max


def convert_rgb2gray(i_info, i_sect, j_sect, n_rows, n_cols, rgb='BGR', stats=False):

    """
    Convert RGB to gray scale array

    0.2125 R + 0.7154 G + 0.0721 B

    Args:
        i_info (object of ropen)
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

        print('\nCalculating image min and max ...\n')

        min_max = get_layer_min_max(i_info, rgb=True)

        im_min = min_max[0][0]
        im_max = min_max[0][1]

        # im_min = 1000000
        # im_max = -1000000
        #
        # for i_ in xrange(0, i_info.rows, 512):
        #
        #     n_rows_ = raster_tools.n_rows_cols(i_, 512, i_info.rows)
        #
        #     for j_ in xrange(0, i_info.cols, 512):
        #
        #         n_cols_ = raster_tools.n_rows_cols(j_, 512, i_info.cols)
        #
        #         im_block = i_info.read(bands2open=[1, 2, 3],
        #                                i=i_, j=j_,
        #                                rows=n_rows_, cols=n_cols_,
        #                                d_type='float32')
        #
        #         luminosity = get_luminosity(im_block, n_rows_, n_cols_, rgb)
        #
        #         im_min, im_max = min_max_func(luminosity, im_min, im_max)

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

        print('\nCalculating average RGB ...\n'.format(rgb.upper()))

        im_block = i_info.read(bands2open=[1, 2, 3],
                               i=i_sect, j=j_sect,
                               rows=n_rows, cols=n_cols,
                               d_type='float32')

        luminosity = get_luminosity(im_block)

        return luminosity, None, None


def get_sect_chunk_size(image_info, parameter_object):

    # get section and chunk size
    if image_info.rows <= parameter_object.section_size:
        sect_row_size = copy.copy(image_info.rows)
    else:
        sect_row_size = parameter_object.section_size

    if image_info.cols <= parameter_object.section_size:
        sect_col_size = copy.copy(image_info.cols)
    else:
        sect_col_size = parameter_object.section_size

    parameter_object.update_info(sect_row_size=sect_row_size,
                                 sect_col_size=sect_col_size)

    return parameter_object


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


def create_band(meta_info, parameter_object, out_bands, blocks=True):

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

    if os.path.isfile(parameter_object.out_img):
        return True
    else:

        i_info = meta_info.copy()

        if blocks:
            i_info = get_adj_info(meta_info, i_info, parameter_object)

        i_info.update_info(bands=out_bands, storage='float32')

        out_rst = raster_tools.create_raster(parameter_object.out_img, i_info)

        out_rst.close_file()
        out_rst = None

        return False


def get_stats(image_info, parameter_object):

    # Check available cpu memory
    available_space = psutil.virtual_memory().available * 9.53674e-7

    rows_and_cols = image_info.rows * image_info.cols

    if (25000000 < rows_and_cols < 64000000) and (available_space > 8000):

        if parameter_object.use_rgb:

            __, mn, mx = convert_rgb2gray(image_info, None, None, None, None, stats=True)

        else:

            if parameter_object.image_max > 0:
                mx = parameter_object.image_max
                mn = 0
            else:
                mx = image_info.read(bands2open=parameter_object.band_position).max()
                mn = image_info.read(bands2open=parameter_object.band_position).min()

    else:

        if parameter_object.use_rgb:

            __, mn, mx = convert_rgb2gray(image_info, None, None, None, None, stats=True)

        else:

            if parameter_object.image_max > 0:
                mx = parameter_object.image_max
                mn = 0
            else:
                mn, mx, __, __ = image_info.get_stats(parameter_object.band_position)

    parameter_object.update_info(min=mn, max=mx)

    return parameter_object


def set_status(parameter_object):

    if os.path.isfile(parameter_object.status_dict_txt):

        # open the status dictionary
        with open(parameter_object.status_dict_txt, 'r') as pf:
            status_dict = yaml.load(pf)

        # get the feature status
        try:
            feature_status = status_dict[parameter_object.out_img_base]
        except:
            status_dict[parameter_object.out_img_base] = -999
            feature_status = -999

    else:

        status_dict = dict()

        # set the layer feature status as non-existent
        status_dict[parameter_object.out_img_base] = -999

        feature_status = -999

    if parameter_object.reset:
        feature_status = -999

    parameter_object.update_info(feature_status=feature_status,
                                 status_dict=status_dict)

    return parameter_object


def get_n_sects(image_info, parameter_object):

    n_row_sects = len([i_sect
                       for i_sect in xrange(0, image_info.rows,
                                            parameter_object.sect_row_size -
                                            (parameter_object.scales[-1] - parameter_object.block))])

    n_col_sects = len([j_sect
                       for j_sect in xrange(0, image_info.cols,
                                            parameter_object.sect_col_size -
                                            (parameter_object.scales[-1] -
                                             parameter_object.block))])

    n_sects = len([j_sect
                   for (i_sect, j_sect) in
                   itertools.product(xrange(0, image_info.rows,
                                            parameter_object.sect_row_size -
                                            (parameter_object.scales[-1] -
                                             parameter_object.block)),
                                     xrange(0, image_info.cols,
                                            parameter_object.sect_col_size -
                                            (parameter_object.scales[-1] -
                                             parameter_object.block)))])

    parameter_object.update_info(n_row_sects=n_row_sects,
                                 n_col_sects=n_col_sects,
                                 n_sects=n_sects)

    return parameter_object


def pad_array(parameter_object, array_section, n_rows, n_cols):

    # pad left and top
    if parameter_object.scales[-1] != parameter_object.block:

        pad_len = (parameter_object.scales[-1] / 2) - (parameter_object.block / 2)

        if (parameter_object.i_sect_blk_ctr == 1) and (parameter_object.j_sect_blk_ctr == 1):

            if parameter_object.trigger == 'dmp':

                array_section = np.asarray([np.pad(array_section[pos], ((pad_len, 0), (pad_len, 0)), 'wrap')
                                            for pos in xrange(0, array_section.shape[0])]).reshape(array_section.shape[0],
                                                                                                   n_rows + pad_len,
                                                                                                   n_cols + pad_len)

            else:
                array_section = np.pad(array_section, ((pad_len, 0), (pad_len, 0)), 'wrap')

        # pad top only
        elif (parameter_object.i_sect_blk_ctr == 1) and (parameter_object.j_sect_blk_ctr > 1) and \
                (parameter_object.j_sect_blk_ctr < parameter_object.n_col_sects):

            if parameter_object.trigger == 'dmp':

                array_section = np.asarray([np.pad(array_section[pos], ((pad_len, 0), (0, 0)), 'wrap')
                                      for pos in xrange(0, array_section.shape[0])]).reshape(array_section.shape[0],
                                                                                             n_rows + pad_len, n_cols)

            else:
                array_section = np.pad(array_section, ((pad_len, 0), (0, 0)), 'wrap')

        # pad top and right
        elif (parameter_object.i_sect_blk_ctr == 1) and (parameter_object.j_sect_blk_ctr == parameter_object.n_col_sects):

            if parameter_object.trigger == 'dmp':

                array_section = np.asarray([np.pad(array_section[pos], ((pad_len, 0), (0, pad_len)), 'wrap')
                                      for pos in xrange(0, array_section.shape[0])]).reshape(array_section.shape[0],
                                                                                       n_rows + pad_len,
                                                                                       n_cols + pad_len)

            else:
                array_section = np.pad(array_section, ((pad_len, 0), (0, pad_len)), 'wrap')

        # pad left only
        elif (parameter_object.i_sect_blk_ctr > 1) and (parameter_object.i_sect_blk_ctr < parameter_object.n_row_sects) and \
                (parameter_object.j_sect_blk_ctr == 1):

            if parameter_object.trigger == 'dmp':

                array_section = np.asarray([np.pad(array_section[pos], ((pad_len, 0), (pad_len, 0)), 'wrap')
                                      for pos in xrange(0, array_section.shape[0])]).reshape(array_section.shape[0],
                                                                                       n_rows + pad_len,
                                                                                       n_cols + pad_len)

            else:
                array_section = np.pad(array_section, ((0, 0), (pad_len, 0)), 'wrap')

        # pad right only
        elif (parameter_object.i_sect_blk_ctr > 1) and (parameter_object.i_sect_blk_ctr < parameter_object.n_row_sects) \
                and (parameter_object.j_sect_blk_ctr == parameter_object.n_col_sects):

            if parameter_object.trigger == 'dmp':

                array_section = np.asarray([np.pad(array_section[pos], ((0, 0), (0, pad_len)), 'wrap')
                                      for pos in xrange(0, array_section.shape[0])]).reshape(array_section.shape[0],
                                                                                       n_rows, n_cols + pad_len)

            else:
                array_section = np.pad(array_section, ((0, 0), (0, pad_len)), 'wrap')

        # pad left and bottom
        elif (parameter_object.i_sect_blk_ctr == parameter_object.n_row_sects) \
                and (parameter_object.j_sect_blk_ctr == 1):

            if parameter_object.trigger == 'dmp':

                array_section = np.asarray([np.pad(array_section[pos], ((0, pad_len), (pad_len, 0)), 'wrap')
                                      for pos in xrange(0, array_section.shape[0])]).reshape(array_section.shape[0],
                                                                                       n_rows + pad_len,
                                                                                       n_cols + pad_len)

            else:
                array_section = np.pad(array_section, ((0, pad_len), (pad_len, 0)), 'wrap')

        # pad bottom only
        elif (parameter_object.i_sect_blk_ctr == parameter_object.n_row_sects) and (parameter_object.j_sect_blk_ctr > 1) \
                and (parameter_object.j_sect_blk_ctr < parameter_object.n_col_sects):

            if parameter_object.trigger == 'dmp':

                array_section = np.asarray([np.pad(array_section[pos], ((0, pad_len), (0, 0)), 'wrap')
                                      for pos in xrange(0, array_section.shape[0])]).reshape(array_section.shape[0],
                                                                                       n_rows + pad_len, n_cols)

            else:
                array_section = np.pad(array_section, ((0, pad_len), (0, 0)), 'wrap')

        # pad right and bottom
        elif (parameter_object.i_sect_blk_ctr == parameter_object.n_row_sects) \
                and (parameter_object.j_sect_blk_ctr == parameter_object.n_col_sects):

            if parameter_object.trigger == 'dmp':

                array_section = np.asarray([np.pad(array_section[pos], ((0, pad_len), (0, pad_len)), 'wrap')
                                      for pos in xrange(0,
                                                        array_section.shape[0])]).reshape(array_section.shape[0],
                                                                                    n_rows + pad_len,
                                                                                    n_cols + pad_len)

            else:
                array_section = np.pad(array_section, ((0, pad_len), (0, pad_len)), 'wrap')

    return array_section
