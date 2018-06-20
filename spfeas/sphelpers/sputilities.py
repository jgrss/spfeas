#!/usr/bin/env python

from __future__ import division
from future.utils import viewitems
from builtins import map

import os
import copy
import time
import itertools

from ..errors import logger

from mpglue import raster_tools, vrt_builder
from mpglue import utils

import numpy as np

# GDAL
try:
    from osgeo import gdal
except:

    logger.error('GDAL must be installed')
    raise ImportError

# YAML
try:
    import yaml
except:

    logger.error('YAML must be installed')
    raise ImportError

# retry
try:
    from retrying import retry
except:

    logger.error('retrying must be installed')
    raise ImportError


def write_log(parameter_object):

    """
    Writes a parameter log to file

    Args:
        parameter_object (class)
    """

    # Setup the log file.
    if os.path.isfile(parameter_object.log_txt):

        with open(parameter_object.log_txt, 'r') as log_txt_wr:
            starter = log_txt_wr.readlines()

        os.remove(parameter_object.log_txt)

    else:
        starter = list()

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
                             '{} compute features as neighbors\n'.format(parameter_object.write_neighbors),
                             '{} perform histogram equalization\n'.format(parameter_object.write_equalize),
                             '{} perform adaptive histogram equalization\n'.format(parameter_object.write_equalize_adapt)]

    with open(parameter_object.log_txt, 'w') as log_txt_wr:
        log_txt_wr.writelines(lines2write)


def parameter_checks(parameter_object):

    """
    Checks parameters

    Args:
        parameter_object (class)
    """

    # Ensure the input image exists.
    if not os.path.isfile(parameter_object.input_image):

        logger.error('The input image, {}, does not exist.'.format(parameter_object.input_image))
        raise OSError

    # Ensure the block size is smaller than
    #   the maximum scale size.
    if parameter_object.block > np.max(parameter_object.scales):

        logger.error('The block size ({:d}) cannot be greater than the maximum scale <scales>.'.format(parameter_object.block))
        raise ValueError

    # Ensure the block size is even if
    #   the scales are even.
    if (parameter_object.block % 2 != 0) and (parameter_object.scales[0] % 2 == 0):

        logger.error('Please pass an even number for the `block` parameter if your `scales` are also even.')
        raise ValueError

    # Ensure all scales are either odd or even.
    first_scale = parameter_object.scales[0]

    if len(parameter_object.scales) > 1:

        # Even
        if first_scale % 2 == 0:
            for scale in parameter_object.scales:
                if scale % 2 != 0:

                    logger.error('All scales should be even or odd.')
                    raise ValueError

        # Odd
        if first_scale % 2 != 0:
            for scale in parameter_object.scales:
                if scale % 2 == 0:

                    logger.error('All scales should be even or odd.')
                    raise ValueError

    # Ensure the section size is divisible
    #   by the largest scale size.
    if parameter_object.section_size % parameter_object.scales[-1] != 0:

        # Increase the section size.
        while parameter_object.section_size % parameter_object.scales[-1] != 0:
            parameter_object.section_size += 1

    # Ensure the correct smoothing parameters.
    if parameter_object.smooth > 0:

        if parameter_object.smooth <= 2:

            logger.error('The `smooth` parameter should be 3 or greater.')
            raise ValueError('The `smooth` parameter should be 3 or greater.')

        if parameter_object.smooth % 2 == 0:

            logger.error('The `smooth` parameter should be an odd number.')
            raise ValueError

    # Ensure the smallest scale is
    #   >= 16 when using Gabor.
    # if 'gabor' in parameter_object.triggers:
    #
    #     if min(parameter_object.scales) < 16:
    #         logger.error('The Gabor feature cannot be computed with scales < 16.')
    #         raise ValueError

    # Create the output directory.
    if not os.path.isdir(parameter_object.output_dir):

        try:
            os.makedirs(parameter_object.output_dir)
        except OSError:

            logger.error('Could not create the output directory.')
            raise OSError


def set_yaml_file(parameter_object):

    """
    Sets the output YAML status file

    Args:
        parameter_object (class)
    """

    band_pos_str = list(map(str, parameter_object.band_positions))

    if band_pos_str[0] in ['rgb', 'bgr']:
        band_pos_str = '-{}'.format(band_pos_str[0])
    else:
        band_pos_str = '-'.join(band_pos_str)

    return os.path.join(parameter_object.output_dir,
                        '{}__BD{}_BK{:d}_SC{}_TR{}.yaml'.format(parameter_object.f_base,
                                                                band_pos_str,
                                                                parameter_object.block,
                                                                '-'.join(list(map(str, parameter_object.scales))),
                                                                '-'.join(parameter_object.triggers)))


def class2dict(class2convert):

    """
    Converts a class to a dictionary

    Args:
        class2convert (class)
    """

    parameter_dict = dict()

    for attribute in [a for a in dir(class2convert) if not a.startswith('__')]:

        if attribute not in ['copy', 'set_defaults', 'run', 'update_info']:
            parameter_dict[attribute] = getattr(class2convert, attribute)

    return parameter_dict


class DictClass(object):

    """A class to convert a dictionary to a class object"""

    def __init__(self, input_dict):
        self._convert(input_dict)

    def _convert(self, input_dict):

        for ks, vs in viewitems(input_dict):
            setattr(self, ks, vs)

    def copy(self):
        return copy.copy(self)

    def update_info(self, **kwargs):

        for k, v in viewitems(kwargs):
            setattr(self, k, v)


def dict2class(dict2convert):

    """
    Converts a dictionary to a class

    Args:
        dict2convert (dict)
    """

    return DictClass(dict2convert)


def scale_fea_check(parameter_object, is_image=True):

    """
    Checks the scale and feature to set the string name.

    Args:
        parameter_object (class)
        is_image (Optional[bool])

    Returns:
        Image name as a string
    """

    band_pos_str = parameter_object.band_positions

    if isinstance(band_pos_str, str):
        band_pos_str = '-' + band_pos_str
    else:
        band_pos_str = '-'.join(list(map(str, band_pos_str)))

    feature_str = 'ST1-{:03}'.format(parameter_object.band_info['band_count'])

    # Get the output image extension.
    driver_dict_r = {v: k for k, v in viewitems(raster_tools.DRIVER_DICT)}
    image_extension = driver_dict_r[parameter_object.format]

    if is_image:

        section_counter_ = 'TL{:06}'.format(parameter_object.section_counter)

        out_img = os.path.join(parameter_object.feas_dir,
                               '{}__BD{}_BK{:d}_SC{}__{}__{}{}'.format(parameter_object.f_base,
                                                                       band_pos_str,
                                                                       parameter_object.block,
                                                                       '-'.join(list(map(str, parameter_object.scales))),
                                                                       feature_str,
                                                                       section_counter_,
                                                                       image_extension))

        out_img_d_name, out_img_f_name = os.path.split(out_img)
        out_img_base, out_img_f_ext = os.path.splitext(out_img_f_name)

        parameter_object.update_info(out_img=out_img,
                                     out_img_base=out_img_base)

    else:

        search_wildcard = '{}__BD{}_BK{:d}_SC{}__{}__*{}'.format(parameter_object.f_base,
                                                                 band_pos_str,
                                                                 parameter_object.block,
                                                                 '-'.join(list(map(str, parameter_object.scales))),
                                                                 feature_str,
                                                                 image_extension)

        parameter_object.update_info(search_wildcard=search_wildcard)

    return parameter_object


def stack_features(parameter_object, new_feas_list):

    """
    Stacks features
    """

    for trigger in parameter_object.triggers:

        parameter_object.update_info(trigger=trigger)

        # Set the output features folder.
        parameter_object = set_feas_dir(parameter_object)

        for band_p in parameter_object.band_positions:

            parameter_object.update_info(band_position=band_p)

            for sect_counter in range(1, parameter_object.n_sects+1):

                parameter_object.update_info(section_counter=sect_counter)

                parameter_object = scale_fea_check(parameter_object)

                # skip the feature if it doesn't exist
                if not os.path.isfile(parameter_object.out_img):
                    continue

                new_feas_list.append(parameter_object.out_img)

    # write band list to text
    fea_list_txt = parameter_object.status_file.replace('.yaml', '_feature_list.txt')

    # remove stacked VRT list
    if os.path.isfile(fea_list_txt):
        os.remove(fea_list_txt)

    with open(fea_list_txt, 'w') as fea_list_txt_wr:

        fea_list_txt_wr.write('Layer Name\n')

        for fea_ctr, fea_name in enumerate(new_feas_list):
            fea_list_txt_wr.write('{:d} {}\n'.format(fea_ctr+1, fea_name))

    vrt_mosaic = parameter_object.status_file.replace('.yaml', '.vrt')

    stack_dict = dict()

    for ni, new_feas in enumerate(new_feas_list):
        stack_dict[str(ni+1)] = [new_feas]

    logger.info('  Stacking variables ...')

    gdal.BuildVRT(vrt_mosaic,
                  new_feas_list)

    parameter_object.update_info(out_vrt=vrt_mosaic)

    return parameter_object


def set_feas_dir(parameter_object):

    """
    Prepares directory names

    Args:
        parameter_object (class)
    """

    feas_dir = os.path.join(parameter_object.output_dir, parameter_object.status_file.replace('.yaml', ''))

    parameter_object.update_info(feas_dir=feas_dir)

    if not os.path.isdir(parameter_object.feas_dir):
        os.makedirs(parameter_object.feas_dir)

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
    Gets the pixel-wise average in the visible spectrum

    Args:
        im_block (array)
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


def convert_rgb2gray(i_info, i_sect, j_sect, n_rows, n_cols, the_sensor, stats=False):

    """
    Converts RGB to gray scale array

    Args:
        i_info (object of ropen)
        j_sect (int): Starting column index.
        i_sect (int): Starting row index.
        n_cols (int)
        n_rows (int)
        the_sensor (str): The satellite sensor.
        stats (Optional[bool])

    Equation:
        0.2125 R + 0.7154 G + 0.0721 B
    """

    utils.sensor_wavelength_check(the_sensor, ['blue', 'green', 'red'])

    bands2open = [utils.SENSOR_BAND_DICT[the_sensor]['blue'],
                  utils.SENSOR_BAND_DICT[the_sensor]['green'],
                  utils.SENSOR_BAND_DICT[the_sensor]['red']]

    if stats:

        logger.info('\nCalculating image min and max ...\n')

        min_max = get_layer_min_max(i_info, rgb=True)

        im_min = min_max[0][0]
        im_max = min_max[0][1]

        # im_min = 1000000
        # im_max = -1000000
        #
        # for i_ in range(0, i_info.rows, 512):
        #
        #     n_rows_ = raster_tools.n_rows_cols(i_, 512, i_info.rows)
        #
        #     for j_ in range(0, i_info.cols, 512):
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

        logger.info('\nCalculating average RGB ...\n')

        im_block = i_info.read(bands2open=bands2open,
                               i=i_sect,
                               j=j_sect,
                               rows=n_rows,
                               cols=n_cols,
                               d_type='float32')

        luminosity = get_luminosity(im_block)

        return luminosity, None, None


def _retry_if_not_dict(result):

    if not isinstance(result, dict):
        return True
    else:
        return False


def _retry_if_not_open(result):

    if not result:
        return True
    else:
        return False


class ManageStatus(object):

    """A class to manage the processing status with YAML
    @retry(wait_fixed=2000, retry_on_result=_retry_if_not_dict, stop_max_attempt_number=50)
    """

    def copy(self):
        return copy.copy(self)

    def load_status(self, status2load):

        """Loads the processing status from file"""

        self.status_dict = self._load_status(status2load)

        # if not isinstance(self.status_dict, dict):
        #     self.status_dict = dict()

        # if not isinstance(self.status_dict, dict):
        #     logger.error('  The loaded object was not a dictionary.')

    @staticmethod
    def _load_status(status2load):

        """Open the file in 'read' mode"""

        with open(status2load, 'r') as pf:
            return yaml.load(pf)

    def dump_status(self, status2dump):

        """Dumps the processing status to file"""

        if not hasattr(self, 'status_dict'):
            logger.error('  The object does not have a status dictionary.')

        self._dump_status(status2dump)

    def _dump_status(self, status2dump):

        """Dumps the status file, waiting 1/2 a second between retrying"""

        with open(status2dump, 'wb') as pf:

            pf.write(yaml.dump(self.status_dict,
                               default_flow_style=False,
                               encoding='utf-8'))


def create_outputs(parameter_object, new_feas_list, image_info):

    obds = 1
    for scale in parameter_object.scales:

        parameter_object.update_info(scale=scale)

        for feature in range(1, parameter_object.features_dict[parameter_object.trigger]+1):

            parameter_object.update_info(feature=feature)

            parameter_object = scale_fea_check(parameter_object)

            new_feas_list.append(parameter_object.out_img)

            # Only create a new feature if the file does not exist
            if create_band(image_info, parameter_object, 1):
                continue

            obds += 1

    return parameter_object


def get_section_size(image_info, parameter_object):

    """
    Gets section and chunk sizes

    Args:
        image_info (`rinfo` object)
        parameter_object (class)
    """

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


def get_output_info_tile(meta_info, image_info, tile_parameter_object, i_sect, j_sect, out_rows, out_cols):

    """
    Gets the adjusted output image information

    Args:
        meta_info (`rinfo` object)
        image_info (`rinfo` object)
        tile_parameter_object
        i_sect (int)
        j_sect (int)
        out_rows (int)
        out_cols (int)

    Returns:
        Updated `rinfo` object
    """

    # The output cell size.
    cell_size_y = float(tile_parameter_object.block) * meta_info.cellY
    cell_size_x = float(tile_parameter_object.block) * meta_info.cellX

    block_offset = (tile_parameter_object.scales[-1] / 2) - (tile_parameter_object.block / 2)

    # Adjust the output left and right coordinates.
    left_coord = meta_info.left + abs(j_sect * meta_info.cellY) + (block_offset * abs(meta_info.cellY))
    top_coord = meta_info.top - abs(i_sect * meta_info.cellX) - (block_offset * abs(meta_info.cellY))

    image_info.update_info(rows=out_rows,
                           cols=out_cols,
                           left=left_coord,
                           top=top_coord,
                           cellY=cell_size_y,
                           cellX=cell_size_x,
                           bands=tile_parameter_object.band_info['band_count'],
                           storage='float32')

    # image_info.update_info(right=image_info.left+(cols*meta_info.cellY),
    #                        bottom=image_info.top-(rows*meta_info.cellY))

    return image_info


def get_adj_info(meta_info, image_info, parameter_object):

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

    image_info.update_info(rows=len([i for i in range(0, meta_info.rows, parameter_object.block)]),
                           cols=len([i for i in range(0, meta_info.cols, parameter_object.block)]),
                           left=meta_info.left,
                           top=meta_info.top,
                           right=meta_info.right,
                           bottom=meta_info.bottom,
                           cellY=float(parameter_object.block) * meta_info.cellY,
                           cellX=float(parameter_object.block) * meta_info.cellX)

    return image_info


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

        i_info.update_info(bands=out_bands,
                           storage='float32')

        out_rst = raster_tools.create_raster(parameter_object.out_img,
                                             i_info,
                                             bigtiff='yes')

        out_rst.close_file()
        del out_rst

        return False


def get_stats(image_info, parameter_object):

    """
    Sets the image statistics

    Args:
         image_info (`rinfo` object)
         parameter_object (class)
    """

    # Set the image minimum.
    if parameter_object.image_min == -999:
        parameter_object.update_info(image_min=0)

    # Set the image maximum.
    if parameter_object.image_max == -999:

        # Let's make some assumptions
        if image_info.storage.lower() == 'byte':
            image_max = 255
        elif image_info.storage.lower() == 'uint16':
            image_max = 10000
        elif image_info.storage.lower() in ['float32', 'float64']:
            image_max = 1.
        else:

            logger.error('The input storage, `{}`, of {} is not supported.'.format(image_info.storage,
                                                                                   image_info.file_name))

            raise NotImplementedError

        parameter_object.update_info(image_max=image_max)

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

    """
    Gets the section information

    Args:
        image_info (`rinfo` object)
        parameter_object (class)
    """

    rw = image_info.rows
    cl = image_info.cols
    bl = parameter_object.block
    sc = parameter_object.scales[-1]
    srw = parameter_object.sect_row_size
    scl = parameter_object.sect_col_size
    scale_block_diff = sc - bl

    row_range = range(0, rw, srw-scale_block_diff)
    col_range = range(0, cl, scl-scale_block_diff)

    # The section index pairs.
    section_idx_pairs = [(idx, jdx) for idx, jdx in itertools.product(row_range, col_range)]

    n_row_sects = len(row_range)
    n_col_sects = len(col_range)
    n_sects = len(section_idx_pairs)

    parameter_object.update_info(n_row_sects=n_row_sects,
                                 n_col_sects=n_col_sects,
                                 n_sects=n_sects,
                                 section_idx_pairs=section_idx_pairs)

    return parameter_object


def pad_array(parameter_object, array_section, n_rows, n_cols):

    """
    Pads the array

    Args:
        parameter_object (class)
        array_section (2d array)
        n_rows (int)
        n_cols (int)
    """

    # pad left and top
    if parameter_object.scales[-1] != parameter_object.block:

        pad_len = (parameter_object.scales[-1] / 2) - (parameter_object.block / 2)

        if (parameter_object.i_sect_blk_ctr == 1) and (parameter_object.j_sect_blk_ctr == 1):

            if parameter_object.trigger == 'dmp':

                array_section = np.asarray([np.pad(array_section[pos], ((pad_len, 0), (pad_len, 0)), 'wrap')
                                            for pos in range(0, array_section.shape[0])]).reshape(array_section.shape[0],
                                                                                                  n_rows + pad_len,
                                                                                                  n_cols + pad_len)

            else:
                array_section = np.pad(array_section, ((pad_len, 0), (pad_len, 0)), 'wrap')

        # pad top only
        elif (parameter_object.i_sect_blk_ctr == 1) and (parameter_object.j_sect_blk_ctr > 1) and \
                (parameter_object.j_sect_blk_ctr < parameter_object.n_col_sects):

            if parameter_object.trigger == 'dmp':

                array_section = np.asarray([np.pad(array_section[pos], ((pad_len, 0), (0, 0)), 'wrap')
                                      for pos in range(0, array_section.shape[0])]).reshape(array_section.shape[0],
                                                                                            n_rows + pad_len, n_cols)

            else:
                array_section = np.pad(array_section, ((pad_len, 0), (0, 0)), 'wrap')

        # pad top and right
        elif (parameter_object.i_sect_blk_ctr == 1) and (parameter_object.j_sect_blk_ctr == parameter_object.n_col_sects):

            if parameter_object.trigger == 'dmp':

                array_section = np.asarray([np.pad(array_section[pos], ((pad_len, 0), (0, pad_len)), 'wrap')
                                      for pos in range(0, array_section.shape[0])]).reshape(array_section.shape[0],
                                                                                            n_rows + pad_len,
                                                                                            n_cols + pad_len)

            else:
                array_section = np.pad(array_section, ((pad_len, 0), (0, pad_len)), 'wrap')

        # pad left only
        elif (parameter_object.i_sect_blk_ctr > 1) and (parameter_object.i_sect_blk_ctr < parameter_object.n_row_sects) and \
                (parameter_object.j_sect_blk_ctr == 1):

            if parameter_object.trigger == 'dmp':

                array_section = np.asarray([np.pad(array_section[pos], ((pad_len, 0), (pad_len, 0)), 'wrap')
                                      for pos in range(0, array_section.shape[0])]).reshape(array_section.shape[0],
                                                                                            n_rows + pad_len,
                                                                                            n_cols + pad_len)

            else:
                array_section = np.pad(array_section, ((0, 0), (pad_len, 0)), 'wrap')

        # pad right only
        elif (parameter_object.i_sect_blk_ctr > 1) and (parameter_object.i_sect_blk_ctr < parameter_object.n_row_sects) \
                and (parameter_object.j_sect_blk_ctr == parameter_object.n_col_sects):

            if parameter_object.trigger == 'dmp':

                array_section = np.asarray([np.pad(array_section[pos], ((0, 0), (0, pad_len)), 'wrap')
                                      for pos in range(0, array_section.shape[0])]).reshape(array_section.shape[0],
                                                                                            n_rows,
                                                                                            n_cols + pad_len)

            else:
                array_section = np.pad(array_section, ((0, 0), (0, pad_len)), 'wrap')

        # pad left and bottom
        elif (parameter_object.i_sect_blk_ctr == parameter_object.n_row_sects) \
                and (parameter_object.j_sect_blk_ctr == 1):

            if parameter_object.trigger == 'dmp':

                array_section = np.asarray([np.pad(array_section[pos], ((0, pad_len), (pad_len, 0)), 'wrap')
                                      for pos in range(0, array_section.shape[0])]).reshape(array_section.shape[0],
                                                                                            n_rows + pad_len,
                                                                                            n_cols + pad_len)

            else:
                array_section = np.pad(array_section, ((0, pad_len), (pad_len, 0)), 'wrap')

        # pad bottom only
        elif (parameter_object.i_sect_blk_ctr == parameter_object.n_row_sects) and (parameter_object.j_sect_blk_ctr > 1) \
                and (parameter_object.j_sect_blk_ctr < parameter_object.n_col_sects):

            if parameter_object.trigger == 'dmp':

                array_section = np.asarray([np.pad(array_section[pos], ((0, pad_len), (0, 0)), 'wrap')
                                      for pos in range(0, array_section.shape[0])]).reshape(array_section.shape[0],
                                                                                            n_rows + pad_len,
                                                                                            n_cols)

            else:
                array_section = np.pad(array_section, ((0, pad_len), (0, 0)), 'wrap')

        # pad right and bottom
        elif (parameter_object.i_sect_blk_ctr == parameter_object.n_row_sects) \
                and (parameter_object.j_sect_blk_ctr == parameter_object.n_col_sects):

            if parameter_object.trigger == 'dmp':

                array_section = np.asarray([np.pad(array_section[pos], ((0, pad_len), (0, pad_len)), 'wrap')
                                      for pos in range(0, array_section.shape[0])]).reshape(array_section.shape[0],
                                                                                            n_rows + pad_len,
                                                                                            n_cols + pad_len)

            else:
                array_section = np.pad(array_section, ((0, pad_len), (0, pad_len)), 'wrap')

    return array_section
