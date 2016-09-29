#!/usr/bin/env python

"""
@authors: Jordan Graesser, Rafael Alvarez
Date Created: 7/2/2013
"""

import os
import sys
import itertools
import time
import subprocess
import platform
from copy import copy
import multiprocessing as M
from joblib import Parallel, delayed
import argparse

# Pickle
try:
    import cPickle as pickle
except:
    from six.moves import cPickle as pickle
else:
   import pickle

from mappy import raster_tools
from mappy.features import pca
from mappy.features.veg_indices import VegIndicesEquations
from mappy.features.helpers import reshape_feas
from mappy.features.helpers import split_sects
from mappy.paths import gdal_path

try:
    import psutil
except:
    raise ImportError('Psutil must be installed')

# OpenCV
try:
    import cv2
except:
    raise ImportError('OpenCV must be installed')

# NumPy
try:
    import numpy as np
except:
    raise ImportError('NumPy must be installed')

# Numexpr
try:
    import numexpr as ne
except ImportError:
    raise ImportError('Numexpr must be installed')

# Scikit-image
try:
    from skimage.morphology import skeletonize, remove_small_objects#, erosion, square
except ImportError:
    raise ImportError('Scikit-image must be installed')

# Ndimage
try:
    from scipy.ndimage.measurements import label as lab_img
except ImportError:
    raise ImportError('Ndimage must be installed')

# Matplotlib
try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError('Matplotlib must be installed')

# Colorama
try:
    import colorama
    from colorama import Fore, Style
except:
    print('Colorama not found. Attempting to install ...')
    subprocess.call('pip install colorama', shell=True)

    import colorama
    from colorama import Fore, Style

# For Numexpr depracated warnings
import warnings
warnings.filterwarnings('ignore')

old_settings = np.seterr(all='ignore')


def _get_edge_prob(img):

    """
    img -- 2d array
    """

    rows, cols = img.shape

    img_max = img.max()
    img_min = img.min()

    img = cv2.bilateralFilter(img, 5, 5, 5)

    # pad by one pixel on each edge
    img = np.pad(img, ((1, 1), (1, 1)), 'edge')

    offsets = [1, 1, -1, -1]
    axes = [0, 1, 1, 0]
    offsets_diags = [1, -1, 1, -1]
    axes_diags = [1, 0, 0, 1]

    edge_prob = np.zeros((rows, cols)).astype(np.float32)

    for ngbr in xrange(0, 4):

        temp = np.roll(img, offsets[ngbr], axis=axes[ngbr])[1:rows+1, 1:cols+1]
        temp = np.subtract(img[1:rows+1, 1:cols+1], temp)
        temp = np.power(temp, 2)

        edge_prob = np.add(edge_prob, temp)

    for ngbr in xrange(0, 4):

        temp = np.roll(img, offsets[ngbr], axis=axes[ngbr])
        temp = np.roll(temp, offsets_diags[ngbr], axis=axes_diags[ngbr])[1:rows+1, 1:cols+1]
        temp = np.subtract(img[1:rows+1, 1:cols+1], temp)

        edge_prob = np.add(edge_prob, temp)

    max_possible = ((img_max - img_min) * 4) + ((img_max - img_min) * 4)

    edge_prob = np.divide(edge_prob, max_possible)

    return edge_prob


def _convert_rgb2gray(i_info, j_sect, i_sect, n_cols, n_rows, RGB='BGR', stats=False):

    """
    Convert RGB to gray scale array

    0.2125 R + 0.7154 G + 0.0721 B

    Args:
        i_info -- MapPy instance
            : image information object
        j_sec -- int
            : starting column index
        i_sect -- int
            : starting row index
        n_cols -- int
        n_rows -- int
        RGB -- str, optional
            : The order of the visible spectrum bands. Many RGB images or photos are stored as red, green, blue. However,
            with multi-band satellite imagery common storage is blue, green, red. Though it may be unorthodox, the default
              here is blue, green, red, or 'BGR'.
    """

    print '\nConverting RGB to grayscale ...'

    if stats:
        luminosity = np.zeros((i_info.rows, i_info.cols), dtype='float32')
    else:
        luminosity = np.zeros((n_rows, n_cols), dtype='float32')

    gray_min, gray_max = 0, 0

    if RGB == 'RGB':
        coeff_dict = {1: .2125, 2: .7154, 3: .0721}
    elif RGB == 'BGR':
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


def _get_adj_info(meta_info, i_info, blk_size):

    """
    Get the adjusted output image information

    Parameters
    ----------
    meta_info -- MapPy class object
    i_info -- MapPy class object
    max_sc -- int
        : maximum scale used
    blk_size -- int
        : block size to write to

    Returns
    -------
    Updated MapPy class information object
    """

    i_info.rows = len([i for i in xrange(0, meta_info.rows, blk_size)])
    i_info.cols = len([i for i in xrange(0, meta_info.cols, blk_size)])

    i_info.left = meta_info.left
    i_info.top = meta_info.top
    i_info.right = meta_info.right
    i_info.bottom = meta_info.bottom

    i_info.cellY, i_info.cellX = float(blk_size)*meta_info.cellY, float(blk_size)*meta_info.cellX

    return i_info


def _create_band(meta_info, out_img, file_ext, blk, scs, out_bands, blocks=True):

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
        i_info = _get_adj_info(meta_info, i_info, blk)

    i_info.bands = out_bands
    i_info.storage = 'float32'

    out_rst = raster_tools.create_raster(out_img, i_info)

    out_rst.close_file()
    out_rst = None


def _get_sect_chunk_size(img_info, max_section_size):

    # get section and chunk size
    if img_info.rows <= max_section_size:
        sect_row_size = copy(img_info.rows)
    else:
        sect_row_size = max_section_size

    if img_info.cols <= max_section_size:
        sect_col_size = copy(img_info.cols)
    else:
        sect_col_size = max_section_size

    return sect_row_size, sect_col_size


def _scale_fea_check(obds, feature, feas_dir, f_base, trigger, blk_size, scale, f_ext, band_p):

    """
    Checks the scale and feature to set the string name.

    Returns:
        Image name as a string
    """

    band_pos_str = str(band_p)

    if band_pos_str == 'rgb' or band_pos_str == 'bgr':
        band_pos_str = '-%s' % band_pos_str
    else:
        band_pos_str = '-'.join(band_pos_str)

    if feature < 10:

        out_img = '{}/{}_{}_bd{}_blk{:d}_sc{:d}_fea00{:d}{}'.format(feas_dir, f_base, trigger,
                                                                    band_pos_str, blk_size, scale, feature, f_ext)

    elif 10 <= feature < 100:

        out_img = '{}/{}_{}_bd{}_blk{:d}_sc{:d}_fea0{:d}{}'.format(feas_dir, f_base, trigger,
                                                                   band_pos_str, blk_size, scale, feature, f_ext)

    else:

        out_img = '{}/{}_{}_bd{}_blk{:d}_sc{:d}_fea{:d}{}'.format(feas_dir, f_base, trigger, band_pos_str,
                                                                  blk_size, scale, feature, f_ext)

    out_img_d_name, out_img_f_name = os.path.split(out_img)
    out_img_f_base, out_img_f_ext = os.path.splitext(out_img_f_name)

    return out_img, out_img_f_base


def _stack_features(blk_size, scs, band_pos, out_dir, f_base, triggers, new_feas_list):

    """
    Stack features
    """

    scs_str = [str(sc) for sc in scs]
    band_pos_str = [str(bp) for bp in band_pos]

    # write band list to text
    fea_list_txt = '{}/{}.{}.stk.bd{}.block{}.scales{}_fea_list.txt'.format(out_dir, f_base, '-'.join(triggers),
                                                                            '-'.join(band_pos_str),
                                                                            blk_size, '-'.join(scs_str))

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
    out_vrt = '{}/{}.{}.stk.bd{}.block{}.scales{}.vrt'.format(out_dir, f_base, '-'.join(triggers),
                                                              '-'.join(band_pos_str), blk_size, '-'.join(scs_str))

    if os.path.isfile(out_vrt):
        os.remove(out_vrt)

    # create the stack list
    if platform.system() == 'Windows':
        com = 'gdalbuildvrt -separate -input_file_list {} {}'.format(new_feas_list, out_vrt)
    else:
        com = 'gdalbuildvrt -separate {} {}'.format(out_vrt, ' '.join(new_feas_list))

    print '\nMosaicking {:d} features ...\n'.format(len(new_feas_list))

    try:
        subprocess.call(com, shell=True)
    except:

        com = r'{}/helpers/{}/apps/{}'.format(os.path.realpath('..'), gdal_path, com)
        subprocess.call(com, shell=True)

    return out_vrt


def _area_of_labeled(sect_in, hist):

    # fill labeled regions with pixel count
    for n in xrange(1, len(hist)+1):
        sect_in[sect_in == n] = hist[n-1]

    return sect_in


def _start_area_labeled(sectin_hist):
    return _area_of_labeled(*sectin_hist)


def _clean_edges(img):

    """
    Clean edges

    This is a replacement of skeletonize, because skeletons erode edges that are greater than 1 pixel wide
    """

    rows, cols = img.shape

    # pad by one pixel on each edge
    img = np.pad(img, ((1, 1), (1, 1)), 'edge')

    offsets = [1, 1, -1, -1]
    axes = [0, 1, 1, 0]

    temp_add = np.zeros((rows, cols)).astype(np.uint8)

    for ngbr in xrange(0, 4):

        temp = np.roll(img, offsets[ngbr], axis=axes[ngbr])[1:rows+1, 1:cols+1]
        temp_add = np.add(temp_add, temp)

    temp_add = np.add(temp_add, img[1:rows+1, 1:cols+1])

    # no edges
    # If the sum adds to 5, mark as no edge,
    #   which means the center pixel was 1
    #   and all surrounding pixels were 1s.
    temp_add = np.where(temp_add == 5, 0, temp_add)

    # only pixels previously marked as potential edges can still be an edge pixel
    return np.where((img[1:rows+1, 1:cols+1] == 1) & (temp_add > 0), 1, 0).astype(np.uint8)


def _close_gaps(img):

    """
    A solution to fill close small gaps between edges
    """

    rows, cols = img.shape

    # pad by one pixel on each edge
    img = np.pad(img, ((1, 1), (1, 1)), 'edge')

    offsets = [1, -1, 1, -1]
    axes = [0, 0, 1, 1]

    temp_add_1 = np.zeros((rows, cols)).astype(np.uint8)
    temp_add_2 = np.zeros((rows, cols)).astype(np.uint8)

    for ngbr in xrange(0, 2):

        temp = np.roll(img, offsets[ngbr], axis=axes[ngbr])[1:rows+1, 1:cols+1]
        temp_add_1 = np.add(temp_add_1, temp)

    for ngbr in xrange(2, 4):

        temp = np.roll(img, offsets[ngbr], axis=axes[ngbr])[1:rows+1, 1:cols+1]
        temp_add_2 = np.add(temp_add_2, temp)

    temp = np.where((temp_add_1 == 2) | (temp_add_2 == 2), 1, 0)

    offsets_diags = [1, -1, -1, 1]
    axes_diags = [1, 1, 0, 0]

    temp_add_1 = np.zeros((rows, cols)).astype(np.uint8)
    temp_add_2 = np.zeros((rows, cols)).astype(np.uint8)

    for ngbr in xrange(0, 2):

        temp = np.roll(img, offsets[ngbr], axis=axes[ngbr])
        temp = np.roll(temp, offsets_diags[ngbr], axis=axes_diags[ngbr])[1:rows+1, 1:cols+1]
        temp_add_1 = np.add(temp_add_1, temp)

    for ngbr in xrange(2, 4):

        temp = np.roll(img, offsets[ngbr], axis=axes[ngbr])
        temp = np.roll(temp, offsets_diags[ngbr], axis=axes_diags[ngbr])[1:rows+1, 1:cols+1]
        temp_add_2 = np.add(temp_add_2, temp)

    temp = np.where(((temp_add_1 == 2) | (temp_add_2 == 2)) & (temp == 1), 1, 0)

    return np.where((temp == 1) & (img[1:rows+1, 1:cols+1] == 0), 1, img[1:rows+1, 1:cols+1]).astype(np.uint8)


def feas(in_img, out_dir, band_positions=[1], rgb2gray=None, block_size=2, scales=[8], triggers=['mean'],
         threshold=20, min_len=10, line_gap=2, weighted=False, sfs_thresh=80, resamp_sfs=0., n_angles=8,
         equalize=False, equalize_adapt=False, smooth=0, visualize=False, convert_stk=False, gdal_cache=256,
         do_pca=False, stack_feas=True, stack_only=False, band_red=3, band_nir=4, neighbors=False, n_jobs=-1,
         reset_sects=False, image_max=0, lac_r=2, section_size=8000, chunk_size=512):

    """
    Computes spatial (contextual) features.

    Args:
        in_img (str): The input image to process.
        out_dir (str): The output directory where features will be saved.
        band_positions (Optional[int list]): A list of band positions to process (e.g., [1,2,3]). Default is [1].
        rgb2gray (Optional[str]): Pre-processing RGB to grayscale, or luminosity. Options are 'BGR' (default) or 'RGB'.
        block_size (Optional[int]): The block size, in pixels, to write features. Default is 2.
            e.g., block_size=2, and input cell resolution is 2 meters, then output feature spatial resolution will be
            4 meters (2 pixels * 2 meters)
        scales (Optional[int list]): A list of feature scales to use (e.g., [8,16,32]). Default is [8].
        triggers (Optional[str list]): A list of features to process. Default is ['mean'].
            Options:
                fourier, gabor, hog, hough, lbp, lbpm, lsr, mean, pantex, sfs, surf, seg
        threshold (Optional[int]): Maximum line threshold for Hough Transform. Default is 20 pixels.
        min_len (Optional[int])
        line_gap (Optional[int])
        weighted (Optional[bool]): Brightness weight for PanTex. Default is False.
        equalize (Optional[bool]): Pre-processing histogram equalization. Default is False.
        equalize_adapt (Optional[bool]): Pre-processing adaptive histogram equalization. Default is False.
        smooth (Optional[int]): Pre-processing bilateral smoothing. Default is False.
        visualize (Optional[int bool])
        gdal_cache (Optional[int]): The GDAL maximum cache, in MB. Default is 256.
        do_pca (Optional[bool])
        stack_feas (Optional[bool]): Post-processing feature stacking. Default is True.
        stack_only (Optional[bool]): Whether to only stack features, and no feature processing. Obviously, 
            it requires features. Default is False.
        band_red (Optional[int]): Red band of multi-spectral imagery to be used for veg. indices. Default is 3.
        band_nir (Optional[int]): NIR band of multi-spectral imagery to be used for veg. indices. Default is 4.
        neighbors (Optional[bool]): Add neighbors to the same feature vector (i.e., center pixel and 4 neighbors 
            become "stacked" features)
        n_jobs (Optional[int]): Number of processes to run in parallel. Default is -1, or use all processors.
        reset_sects (Optional[bool]): Reset the section status dictionary. Default is False.
        image_max (Optional[int]): The image maximum value. Default is 0. If you have a 16-bit image with a
            standardized maximum (e.g., 10,000), then you should declare the ``image_max``.
        lac_r (Optional[int]): The lacunarity box r parameter. Default is 2.
            
    Description:
    
        Image A
        -------
            1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
           ______________________________________________
        1  |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|
        2  |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|
        3  |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|
        4  |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|
        5  |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|
        6  |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|
        7  |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|
        8  |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|
        9  |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|
        10 |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|
    
        Example
        -------
        1) Get section
            A_sect = A[1:6, 1:6] of A[1:15, 1:10]
        2) Split section into chunks
            A_chunks = [A_sect[1:2, 1:2], A_sect[3:4, 3:4], A_sect[4:6, 4:6], ...]
        3) Feed chunks to multiple cores            
    """

    # file check
    if not os.path.isfile(in_img):
        raise OSError('\The input image does not exist.\n')

    # Parameter checks
    if block_size > np.max(scales):
        raise ValueError('\The block size (block_size) cannot be greater than the maximum scale <scales>.\n')

    if (block_size % 2 != 0) and (scales[0] % 2 == 0):
        raise ValueError('\nPlease pass an even number for the <block_size> parameter if your <scales> are also even.\n')

    if smooth > 0:

        if smooth <= 2:
            raise ValueError('\nThe <smooth> parameter should be 3 or greater.\n')
        if smooth % 2 == 0:
            raise ValueError('\nThe <smooth> parameter should be an odd number.\n')

    if not os.path.isdir(out_dir):

        try:
            os.makedirs(out_dir)
        except OSError:
            raise OSError('\nCould not create the output directory.\n')

    # Set the features dictionary.
    features_dict = {'mean': 1, 'pantex': 1, 'ctr': 1, 'lsr': 3, 'hough': 4, 'hog': 4, 'lbp': 62,
                     'lbpm': 4, 'gabor': 2*9, 'surf': 4, 'seg': 1, 'fourier': 2, 'sfs': 5, 'ndvi': 1, 'objects': 1,
                     'dmp': 1, 'xy': 2, 'lac': 1}

    # Set the output bands based on the trigger.
    out_bands_dict = {'mean': len(scales)*features_dict['mean'], 'pantex': len(scales)*features_dict['pantex'],
                      'ctr': len(scales)*features_dict['ctr'], 'lsr': len(scales)*features_dict['lsr'],
                      'hough': len(scales)*features_dict['hough'], 'hog': len(scales)*features_dict['hog'],
                      'lbp': len(scales)*features_dict['lbp'], 'lbpm': len(scales)*features_dict['lbpm'],
                      'gabor': len(scales)*features_dict['gabor'], 'surf': len(scales)*features_dict['surf'],
                      'seg': len(scales)*features_dict['seg'], 'fourier': len(scales)*features_dict['fourier'],
                      'sfs': len(scales)*features_dict['sfs'], 'ndvi': len(scales)*features_dict['ndvi'],
                      'objects': len(scales)*features_dict['objects'], 'dmp': len(scales)*features_dict['dmp'],
                      'xy': len(scales)*features_dict['xy'], 'lac': len(scales)*features_dict['lac']}

    # Update the feature dictionary for feature neighbors.
    if neighbors:

        for key, val in features_dict.iteritems():
            features_dict[key] *= 5

    d_name, f_name = os.path.split(in_img)
    f_base, __ = os.path.splitext(f_name)

    f_ext = '.tif'

    # status dictionary file
    status_dict_txt = '{}/{}_status.txt'.format(out_dir, f_base)

    # log file
    log_txt = '{}/{}_log.txt'.format(out_dir, f_base)

    if os.path.isfile(log_txt):

        with open(log_txt, 'rb') as log_txt_wr:
            log_hist = log_txt_wr.readlines()

        log_txt_wr = open(log_txt, 'wb')
        log_txt_wr.writelines(log_hist)

    else:
        log_txt_wr = open(log_txt, 'wb')

    if isinstance(rgb2gray, str):
        rgb2write = rgb2gray
    else:
        rgb2write = ','.join([str(bpos) for bpos in band_positions])

    if neighbors:
        write_neighbors = 'DID'
    else:
        write_neighbors = 'Did NOT'

    if equalize:
        write_equalize = 'DID'
    else:
        write_equalize = 'Did NOT'

    if equalize_adapt:
        write_equalize_adapt = 'DID'
    else:
        write_equalize_adapt = 'Did NOT'

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
    """.format(time.asctime(time.localtime(time.time())), in_img, out_dir, rgb2write, smooth, block_size,
               ','.join([str(bpos) for bpos in scales]), ','.join(triggers), sfs_thresh, n_angles,
               band_red, band_nir, write_neighbors, write_equalize, write_equalize_adapt))

    log_txt_wr.close()

    if platform.system() == 'Windows':

        new_feas_list = '{}/{}_win_feas_list.txt'.format(out_dir, f_base)

        win_feas_list_o = open(new_feas_list, 'w')

    else:
        new_feas_list = []

    if stack_only:

        for trigger in triggers:

            # output features folder
            feas_dir = '{}/{}'.format(out_dir, trigger)

            if isinstance(rgb2gray, str):
                band_positions = [rgb2gray.lower()]

            for band_p in band_positions:

                # get feature names
                obds = 1
                for scale in scales:

                    for feature in xrange(1, features_dict[trigger]+1):

                        in_trig_name = copy(trigger)

                        out_img, out_img_base = _scale_fea_check(obds, feature, feas_dir, f_base,
                                                                 in_trig_name, block_size, scale, f_ext, band_p)

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
        out_vrt = _stack_features(block_size, scales, band_positions, out_dir, f_base, triggers, new_feas_list)

    else:

        trigger_orig_seg = False

        # Iterate over each feature trigger.
        for trigger in triggers:

            # output features folder
            feas_dir = '{}/{}'.format(out_dir, trigger)

            if not os.path.isdir(feas_dir):
                os.makedirs(feas_dir)

            # trigger segmentation
            # !!!! experimental5

            if trigger == 'seg':

                from helpers.segmentation import feaRegion

                if len(band_positions) > 1:
                    print('\nWarning!! Segmentation will only use the first band in the list\n')

                i_info = raster_tools.rinfo(in_img)

                # get the band and segment image
                bd = i_info.mparray(bands2open=band_positions[0])

                # write segmentation to file
                out_name = '{}/{}_seg.tif'.format(feas_dir, f_base)

                if not os.path.isfile(out_name):

                    print '\n  Segmenting image ...'

                    out_arr = feaRegion(bd, i_info.cellY)

                    i_info.bands = 4
                    i_info.storage = 'float32'

                    raster_tools.write2raster(out_arr, out_name, i_info)

                trigger_orig_seg = True
                trigger = 'mean'

                i_info.close()

                i_info = None

                in_img = copy(out_name)

                # reset the band numbers -- one for each segmentation feature
                band_positions = [1, 2, 3, 4]

            if isinstance(rgb2gray, str):
                band_positions = [rgb2gray.lower()]

            for band_p in band_positions:

                # get image information
                i_info = raster_tools.rinfo(in_img)

                # check available memory
                avl = psutil.virtual_memory().available * 9.53674e-7
                # free = psutil.virtual_memory().free * 9.53674e-7

                # There seems to be bugs with GDAL
                #   get statistics, so we will try
                #   to avoid it if possible.
                #
                # 8,000 row x 8,000 column image
                # 8 GB memory
                rows_and_cols = i_info.rows * i_info.cols

                if (rows_and_cols < 25000000) or (rows_and_cols < 64000000) and (avl > 8000):

                    if isinstance(rgb2gray, str):
                        __, mn, mx = _convert_rgb2gray(i_info, 0, 0, 0, 0, RGB=rgb2gray, stats=True)
                    else:

                        if image_max > 0:
                            mx = image_max
                            mn = 0
                        else:
                            mx = i_info.mparray(bands2open=band_p).max()
                            mn = i_info.mparray(bands2open=band_p).min()
                        # mx = i_info.datasource.GetRasterBand(1).ReadAsArray(0, 0).max()
                        # mn = i_info.datasource.GetRasterBand(1).ReadAsArray(0, 0).min()

                else:

                    if isinstance(rgb2gray, str):
                        __, mn, mx = _convert_rgb2gray(i_info, 0, 0, 0, 0, RGB=rgb2gray, stats=True)
                    else:

                        if image_max > 0:
                            mx = image_max
                            mn = 0
                        else:
                            mn, mx, mnn, stdev = i_info.datasource.GetRasterBand(band_p).GetStatistics(1, 1)

                # Get section and chunk size.
                sect_row_size, sect_col_size = _get_sect_chunk_size(i_info, section_size)

                # Create the output feature bands.
                obds = 1
                for scale in scales:

                    for feature in xrange(1, features_dict[trigger]+1):

                        if trigger_orig_seg:
                            in_trig_name = 'seg'
                        else:
                            in_trig_name = copy(trigger)

                        out_img, out_img_base = _scale_fea_check(obds, feature, feas_dir, f_base,
                                                                 in_trig_name, block_size, scale, f_ext, band_p)

                        # status dictionary
                        if os.path.isfile(status_dict_txt):

                            # open the status dictionary
                            status_dict_txt_o = open(status_dict_txt, 'rb')

                            # pickle the status dictionary
                            status_dict = pickle.load(status_dict_txt_o)

                            status_dict_txt_o.close()

                            # get the feature status
                            try:
                                feature_status = status_dict[out_img_base]
                            except:
                                status_dict[out_img_base] = -999
                                feature_status = -999

                            status_dict_txt_o = open(status_dict_txt, 'wb')

                        else:

                            # create the status dictionary
                            status_dict_txt_o = open(status_dict_txt, 'wb')

                            status_dict = {}

                            # set the layer feature status as non-existent
                            status_dict[out_img_base] = -999

                            feature_status = -999

                        if reset_sects:
                            feature_status = -999

                        # append new features to a list to stack
                        if platform.system() == 'Windows':
                            win_feas_list_o.write('{}\n'.format(out_img))
                        else:
                            new_feas_list.append(out_img)

                        # only create a new feature if the file does not exist
                        if feature_status == -999:

                            _create_band(i_info, out_img, f_ext, block_size, scales, 1)

                            # set the status as created
                            status_dict[out_img_base] = 0

                        # pickle the status dictionary
                        pickle.dump(status_dict, status_dict_txt_o)

                        status_dict_txt_o.close()

                        obds += 1

                # Get the number of sections in
                #   the image (only used as a counter).
                n_row_sects = len([i_sect for i_sect in xrange(0, i_info.rows, sect_row_size-(scales[-1]-block_size))])
                n_col_sects = len([j_sect for j_sect in xrange(0, i_info.cols, sect_col_size-(scales[-1]-block_size))])

                n_sects = len([j_sect for (i_sect, j_sect) in
                               itertools.product(xrange(0, i_info.rows, sect_row_size-(scales[-1]-block_size)),
                                                 xrange(0, i_info.cols, sect_col_size-(scales[-1]-block_size))) ])

                # Here we loop through the
                #   image by sections.
                n_sect = 1
                i_sect_ctr = 0
                i_sect_blk_ctr = 1

                for i_sect in xrange(0, i_info.rows, sect_row_size-(scales[-1]-block_size)):

                    numRws = raster_tools.n_rows_cols(i_sect, sect_row_size, i_info.rows)

                    j_sect_ctr = 0
                    j_sect_blk_ctr = 1
                    for j_sect in xrange(0, i_info.cols, sect_col_size-(scales[-1]-block_size)):

                        print '\nSection {:d} of {:d} ...'.format(n_sect, n_sects)

                        numCols = raster_tools.n_rows_cols(j_sect, sect_col_size, i_info.cols)

                        #####################################
                        # Check if the section has been
                        # processed for all feature scales.
                        #####################################

                        sects_good = True

                        obds = 1
                        for scale in scales:

                            for feature in xrange(1, features_dict[trigger]+1):

                                if trigger_orig_seg:
                                    in_trig_name = 'seg'
                                else:
                                    in_trig_name = copy(trigger)

                                out_img, out_img_base = _scale_fea_check(obds, feature, feas_dir,
                                                                         f_base, in_trig_name,
                                                                         block_size, scale, f_ext, band_p)

                                # Open the status dictionary.
                                status_dict_txt_o = open(status_dict_txt, 'rb')

                                # Pickle the status dictionary.
                                status_dict = pickle.load(status_dict_txt_o)

                                status_dict_txt_o.close()

                                sect_status = status_dict[out_img_base]

                                obds += 1

                                # If any of the sections are not current,
                                #   continue with feature extraction.
                                if sect_status < n_sect:
                                    sects_good = False

                        # Open the image array.
                        if trigger == 'ndvi':

                            sect_in = i_info.mparray(bands2open=[band_red, band_nir], i=i_sect, j=j_sect,
                                                     rows=numRws, cols=numCols, d_type='float32')

                            vie = VegIndicesEquations(sect_in, chunk_size=-1)
                            sect_in = vie.compute('NDVI', out_type=2)

                            mn = 0
                            mx = 255

                        elif trigger == 'dmp':

                            sect_in = np.asarray([i_info.mparray(bands2open=dmp_bd, i=i_sect, j=j_sect,
                                                                 rows=numRws, cols=numCols, d_type='float32')
                                                  for dmp_bd in xrange(1, i_info.bands+1)]).reshape(i_info.bands,
                                                                                                    numRws, numCols)

                        elif trigger == 'hough' or trigger == 'objects':

                            img_arr = i_info.mparray(bands2open=1, i=i_sect, j=j_sect,
                                                     rows=numRws, cols=numCols, d_type='float32')

                            # import matplotlib.cm as cm
                            # import pymeanshift as pms
                            # from skimage.exposure import rescale_intensity
                            #
                            # img_arr = rescale_intensity(img_arr, in_range=(img_arr.min(), img_arr.max()), out_range=(0, 255)).astype(np.uint8)
                            # img_arr, labels_image, number_regions = pms.segment(img_arr, spatial_radius=3,
                            #                                                               range_radius=4.5, min_density=100)
                            #
                            #
                            # plt.imshow(img_arr, cmap=cm.gray, interpolation='nearest')
                            # plt.show()
                            # sys.exit()

                            sect_in = _get_edge_prob(img_arr)

                            # Iterate over the remaining bands
                            #   to find edges.
                            for band_ed in xrange(1, i_info.bands):

                                img_arr = i_info.mparray(bands2open=band_ed+1, i=i_sect, j=j_sect,
                                                         rows=numRws, cols=numCols, d_type='float32')

                                sect_in = np.add(_get_edge_prob(img_arr), sect_in)

                            vie = VegIndicesEquations(sect_in, chunk_size=-1)
                            ndvi = vie.compute('NDVI')

                            sect_in = np.divide(sect_in, i_info.bands)

                            # threshold edges
                            # urban, 1m, 40
                            sect_in = np.where(sect_in >= np.percentile(sect_in, 75), 1, 0).astype(np.uint8)

                            sect_in = _clean_edges(sect_in)

                            sect_in = _close_gaps(sect_in)

                            sect_in = skeletonize(sect_in).astype(np.uint8)

                            # my_cmap = cm.autumn
                            # my_cmap.set_under('k', alpha=0)
                            # ax = plt.subplot(111)
                            # ax.imshow(img_arr, cmap=cm.gray, interpolation='nearest', clim=[img_arr.min(), img_arr.max()])
                            # # ax.imshow(sect_in, cmap=my_cmap, interpolation='nearest', clim=[0, 1])
                            # ax.imshow(sect_in, cmap=cm.gray, interpolation='nearest', clim=[sect_in.min(), sect_in.max()])
                            # plt.show()
                            # sys.exit()

                            # thin the edges
                            # sect_in = skeletonize(sect_in)

                            # remove small objects
                            sect_in = remove_small_objects(sect_in, min_size=20, connectivity=2).astype(np.uint8)

                            if trigger == 'objects':

                                # invert the edges
                                dummy, sect_in = cv2.threshold(sect_in, 0, 1, cv2.THRESH_BINARY_INV)

                                # erode the edges
                                # sect_in = erosion(sect_in, square(3))

                                # label each component
                                # sect_in = label(sect_in, background=0)
                                sect_in, num_objs = lab_img(sect_in)

                                hist, __ = np.histogram(sect_in, bins=sect_in.max(), range=(1, sect_in.max()))

                                print '\n  Calculating object area ...\n'

                                # tile the section for multiprocessing
                                sect_in_tiles = [sect_in[i:i+chunk_size, j:j+chunk_size]
                                                 for i in xrange(0, sect_in.shape[0], chunk_size)
                                                 for j in xrange(0, sect_in.shape[1], chunk_size)]

                                lab_areas = Parallel(n_jobs=n_jobs,
                                                     max_nbytes=None)(delayed(_start_area_labeled)(sect_in_tile,
                                                                                                   hist)
                                                                      for sect_in_tile in sect_in_tiles)

                                # piece the tiles back together
                                chunk_iter = 0
                                for i in xrange(0, sect_in.shape[0], chunk_size):
                                    for j in xrange(0, sect_in.shape[1], chunk_size):
                                        sect_in[i:i+chunk_size, j:j+chunk_size] = lab_areas[chunk_iter]

                                        chunk_iter += 1

                        elif isinstance(rgb2gray, str):
                            sect_in, __, __ = _convert_rgb2gray(i_info, j_sect, i_sect, numCols, numRws, RGB=rgb2gray)
                        else:
                            sect_in = i_info.mparray(bands2open=band_p, i=i_sect, j=j_sect,
                                                     rows=numRws, cols=numCols)

                        # pad array here
                        # (top, bottom), (left, right)

                        # pad left and top
                        if (scales[-1] != block_size):

                            pad_len = (scales[-1] / 2) - (block_size / 2)

                            if (i_sect_blk_ctr == 1) and (j_sect_blk_ctr == 1):

                                if trigger == 'dmp':

                                    sect_in = np.asarray([np.pad(sect_in[pos], ((pad_len, 0), (pad_len, 0)), 'wrap')
                                                          for pos in xrange(0, sect_in.shape[0])]).reshape(sect_in.shape[0],
                                                                                                           numRws+pad_len,
                                                                                                           numCols+pad_len)

                                else:
                                    sect_in = np.pad(sect_in, ((pad_len, 0), (pad_len, 0)), 'wrap')

                            # pad top only
                            elif (i_sect_blk_ctr == 1) and (j_sect_blk_ctr > 1) and (j_sect_blk_ctr < n_col_sects):

                                if trigger == 'dmp':

                                    sect_in = np.asarray([np.pad(sect_in[pos], ((pad_len, 0), (0, 0)), 'wrap')
                                                          for pos in xrange(0, sect_in.shape[0])]).reshape(sect_in.shape[0], numRws+pad_len, numCols)

                                else:
                                    sect_in = np.pad(sect_in, ((pad_len, 0), (0, 0)), 'wrap')

                            # pad top and right
                            elif (i_sect_blk_ctr == 1) and (j_sect_blk_ctr == n_col_sects):

                                if trigger == 'dmp':

                                    sect_in = np.asarray([np.pad(sect_in[pos], ((pad_len, 0), (0, pad_len)), 'wrap')
                                                          for pos in xrange(0, sect_in.shape[0])]).reshape(sect_in.shape[0], numRws+pad_len, numCols+pad_len)

                                else:
                                    sect_in = np.pad(sect_in, ((pad_len, 0), (0, pad_len)), 'wrap')

                            # pad left only
                            elif (i_sect_blk_ctr > 1) and (i_sect_blk_ctr < n_row_sects) and (j_sect_blk_ctr == 1):

                                if trigger == 'dmp':

                                    sect_in = np.asarray([np.pad(sect_in[pos], ((pad_len, 0), (pad_len, 0)), 'wrap')
                                                          for pos in xrange(0, sect_in.shape[0])]).reshape(sect_in.shape[0], numRws+pad_len, numCols+pad_len)

                                else:
                                    sect_in = np.pad(sect_in, ((0, 0), (pad_len, 0)), 'wrap')

                            # pad right only
                            elif (i_sect_blk_ctr > 1) and (i_sect_blk_ctr < n_row_sects) and (j_sect_blk_ctr == n_col_sects):

                                if trigger == 'dmp':

                                    sect_in = np.asarray([np.pad(sect_in[pos], ((0, 0), (0, pad_len)), 'wrap')
                                                          for pos in xrange(0, sect_in.shape[0])]).reshape(sect_in.shape[0], numRws, numCols+pad_len)

                                else:
                                    sect_in = np.pad(sect_in, ((0, 0), (0, pad_len)), 'wrap')

                            # pad left and bottom
                            elif (i_sect_blk_ctr == n_row_sects) and (j_sect_blk_ctr == 1):

                                if trigger == 'dmp':

                                    sect_in = np.asarray([np.pad(sect_in[pos], ((0, pad_len), (pad_len, 0)), 'wrap')
                                                          for pos in xrange(0, sect_in.shape[0])]).reshape(sect_in.shape[0], numRws+pad_len, numCols+pad_len)

                                else:
                                    sect_in = np.pad(sect_in, ((0, pad_len), (pad_len, 0)), 'wrap')

                            # pad bottom only
                            elif (i_sect_blk_ctr == n_row_sects) and (j_sect_blk_ctr > 1) and (j_sect_blk_ctr < n_col_sects):

                                if trigger == 'dmp':

                                    sect_in = np.asarray([np.pad(sect_in[pos], ((0, pad_len), (0, 0)), 'wrap')
                                                          for pos in xrange(0, sect_in.shape[0])]).reshape(sect_in.shape[0], numRws+pad_len, numCols)

                                else:
                                    sect_in = np.pad(sect_in, ((0, pad_len), (0, 0)), 'wrap')

                            # pad right and bottom
                            elif (i_sect_blk_ctr == n_row_sects) and (j_sect_blk_ctr == n_col_sects):

                                if trigger == 'dmp':

                                    sect_in = np.asarray([np.pad(sect_in[pos], ((0, pad_len), (0, pad_len)), 'wrap')
                                                          for pos in xrange(0,
                                                                            sect_in.shape[0])]).reshape(sect_in.shape[0],
                                                                                                        numRws+pad_len,
                                                                                                        numCols+pad_len)

                                else:
                                    sect_in = np.pad(sect_in, ((0, pad_len), (0, pad_len)), 'wrap')

                        if trigger == 'dmp':
                            lRows, lCols = sect_in[0].shape

                            oR, oC, out_rows, out_cols = split_sects.get_out_dims(sect_in[0].astype(np.uint8), lRows,
                                                                                  lCols, block_size, chunk_size, scales)
                        else:
                            lRows, lCols = sect_in.shape

                            oR, oC, out_rows, out_cols = split_sects.get_out_dims(sect_in.astype(np.uint8), lRows,
                                                                                  lCols, block_size, chunk_size, scales)

                        # Only extract features if the section hasn't
                        #   been completed or if the section does not
                        #   contain all zeros.
                        if sects_good or sect_in.max() == 0:
                            pass
                        else:

                            # Here we split the current section into
                            #   chunks and process the features.

                            # Split image and compute features.
                            tk = split_sects.get_sect_feas(sect_in, lRows, lCols, block_size, scales, trigger,
                                                           chunk_size, mn, mx, i_info.cellY, threshold=threshold,
                                                           min_len=min_len, line_gap=line_gap, weighted=weighted,
                                                           sfs_thresh=sfs_thresh, n_angles=n_angles,
                                                           equalize=equalize, equalize_adapt=equalize_adapt,
                                                           smooth=smooth, vis=visualize, n_jobs=n_jobs,
                                                           image_max=image_max, lac_r=lac_r)

                            # Reshape list of features into
                            #   <features x rows x columns> array.
                            out_sect_arr = reshape_feas.reshape_feas(trigger, tk, block_size, scales, oR, oC, lRows,
                                                                     lCols, out_rows, out_cols, chunk_size,
                                                                     out_bands_dict, neighbors)

                            print '  Writing features to file ...'

                            obds = 1
                            for scale in scales:

                                for feature in xrange(1, features_dict[trigger]+1):

                                    if trigger_orig_seg:
                                        in_trig_name = 'seg'
                                    else:
                                        in_trig_name = copy(trigger)

                                    out_img, out_img_base = _scale_fea_check(obds, feature, feas_dir,
                                                                             f_base, in_trig_name,
                                                                             block_size, scale, f_ext, band_p)

                                    o_info = raster_tools.rinfo(out_img, open2read=False)
                                    out_band_obj = o_info.datasource.GetRasterBand(1)

                                    # write array to file
                                    if (i_sect_blk_ctr < n_row_sects) and (j_sect_blk_ctr == n_col_sects):

                                        try:
                                            out_band_obj.WriteArray(out_sect_arr[obds-1], j_sect_ctr, i_sect_ctr)
                                        except:
                                            out_band_obj.WriteArray(out_sect_arr[obds-1], j_sect_ctr-1, i_sect_ctr)

                                    elif (i_sect_blk_ctr == n_row_sects) and (j_sect_blk_ctr < n_col_sects):

                                        try:
                                            out_band_obj.WriteArray(out_sect_arr[obds-1], j_sect_ctr, i_sect_ctr)
                                        except:
                                            out_band_obj.WriteArray(out_sect_arr[obds-1], j_sect_ctr, i_sect_ctr-1)

                                    elif (i_sect_blk_ctr == n_row_sects) and (j_sect_blk_ctr == n_col_sects):

                                        try:
                                            out_band_obj.WriteArray(out_sect_arr[obds-1], j_sect_ctr, i_sect_ctr)
                                        except:
                                            out_band_obj.WriteArray(out_sect_arr[obds-1], j_sect_ctr-1, i_sect_ctr-1)

                                    else:
                                        out_band_obj.WriteArray(out_sect_arr[obds-1], j_sect_ctr, i_sect_ctr)

                                    out_band_obj = None

                                    o_info.close()

                                    o_info = None

                                    obds += 1

                            ###################################################
                            # Update status dictionary with the section number.
                            ###################################################

                            obds_t = 1
                            for scale in scales:

                                for feature in xrange(1, features_dict[trigger]+1):

                                    if trigger_orig_seg:
                                        in_trig_name = 'seg'
                                    else:
                                        in_trig_name = copy(trigger)

                                    out_img, out_img_base = _scale_fea_check(obds_t, feature, feas_dir,
                                                                             f_base, in_trig_name,
                                                                             block_size, scale, f_ext, band_p)

                                    # open the status dictionary
                                    status_dict_txt_o = open(status_dict_txt, 'rb')

                                    # pickle the status dictionary
                                    status_dict = pickle.load(status_dict_txt_o)

                                    status_dict_txt_o.close()

                                    status_dict[out_img_base] = n_sect

                                    # open the status dictionary
                                    status_dict_txt_o = open(status_dict_txt, 'wb')

                                    # pickle the status dictionary
                                    pickle.dump(status_dict, status_dict_txt_o)

                                    status_dict_txt_o.close()

                                    obds_t += 1

                            del tk, oR, oC

                        j_sect_ctr += out_cols
                        j_sect_blk_ctr += 1

                        n_sect += 1

                    i_sect_ctr += out_rows
                    i_sect_blk_ctr += 1

                obds = 1
                for scale in scales:

                    for feature in xrange(1, features_dict[trigger]+1):

                        if trigger_orig_seg:
                            in_trig_name = 'seg'
                        else:
                            in_trig_name = copy(trigger)

                        out_img, out_img_base = _scale_fea_check(obds, feature, feas_dir, f_base,
                                                                 in_trig_name, block_size,
                                                                 scale, f_ext, band_p)

                        # Resample the SFS image.
                        #
                        # SFS radiates from a center pixel, so is
                        #   more useful when computed with a
                        #   smaller block size.
                        if resamp_sfs > 0:

                            out_img_d_name, out_img_f_name = os.path.split(out_img)

                            out_img_resamp = '{}/{}_resamp{}'.format(out_img_d_name, out_img_base, f_ext)

                            # Replace the block size.
                            out_img_resamp = out_img_resamp.replace('blk{:d}'.format(block_size),
                                                                    'blk{:d}'.format(int(resamp_sfs)))

                            if 'img' in f_ext.lower():

                                sfs_resamp_com = 'gdalwarp -multi -wo NUM_THREADS=ALL_CPUS \
                                --config GDAL_CACHEMAX {:d} -co COMPRESS=YES \
                                -of HFA -tr {:f} {:f} -r average {} {}'.format(gdal_cache, resamp_sfs, resamp_sfs,
                                                                               out_img, out_img_resamp)

                            else:

                                sfs_resamp_com = 'gdalwarp -multi -wo NUM_THREADS=ALL_CPUS \
                                --config GDAL_CACHEMAX {:d} -co COMPRESS=DEFLATE \
                                -co TILED=YES -co BIGTIFF=YES -tr {:f} {:f} \
                                -r average {} {}'.format(gdal_cache, resamp_sfs, resamp_sfs, out_img, out_img_resamp)

                            print '\nResampling SFS to {:.1f}m x {:.1f}m cell size ...\n'.format(resamp_sfs, resamp_sfs)

                            try:
                                subprocess.call(sfs_resamp_com, shell=True)
                            except:

                                sfs_resamp_com = r'{}/helpers/{}/apps/{}'.format(os.path.realpath('..'), gdal_path,
                                                                                 sfs_resamp_com)

                                subprocess.call(sfs_resamp_com, shell=True)

                            out_img_new = out_img_resamp.replace('_resamp', '')

                            # Replace the block size.
                            out_img_new = out_img_new.replace('blk{:d}'.format(int(resamp_sfs)),
                                                              'blk{:d}'.format(block_size))

                            os.remove(out_img)

                            os.rename(out_img_resamp, out_img_new)

                        obds += 1

        if platform.system() == 'Windows':
            win_feas_list_o.close()

        # Stack the features
        if stack_feas:

            out_vrt = _stack_features(block_size, scales, band_positions, out_dir, f_base, triggers, new_feas_list)

        # Optional conversion to GeoTiff.
        if convert_stk:

            out_gtiff = '{}/{}.{}.stk.bd{}.block{}.scales{}.tif'.format(out_dir, f_base, '-'.join(triggers),
                                                                        '-'.join(band_pos_str), block_size,
                                                                        '-'.join(scs_str))

            com = 'gdal_translate --config GDAL_CACHEMAX {:d} \
            -of GTiff -co TILED=YES -co COMPRESS=LZW {} {}'.format(gdal_cache, out_vrt, out_gtiff)

            try:
                subprocess.call(com, shell=True)
            except:

                com = r'{}/helpers/{}/apps/{}'.format(os.path.realpath('..'), gdal_path, com)
                subprocess.call(com, shell=True)

        # Run PCA on features.
        if do_pca:

            out_pca = '{}/{}.{}.stk.bd{}.block{}.scales{}_pca.tif'.format(out_dir, f_base, '-'.join(triggers),
                                                                          '-'.join([str(bpos)
                                                                                    for bpos in band_positions]),
                                                                          block_size,
                                                                          '-'.join([str(bpos) for bpos in scales]))

            # Execute PCA.
            pca(out_vrt, out_pca, do_pca)


def _examples():
    
    sys.exit("""\

    # Compute PanTex on band 3 with 2x2 pixel block, at scale 8.
    feas.py -i /image.tif -o /out_dir -bp 3 --block 2 --scales 8 -tr pantex

    # Compute HoG and LBP on bands 1, 2, and 3 with 4x4 pixel block, at scales 16 and 32.
    feas.py -i /image.tif -o /out_dir -bp 1 2 3 --block 4 --scales 16 32 -tr hog lbp

    # Compute the mean NDVI, with a 16-bit image that is scaled to 0-10,000. The `image_max`
    #   parameter ensures scaling across images.
    feas.py -i /image.tif -o /out_dir --equalize_adapt --image_max 10000 -tr ndvi

    # Compute Structural Feature Sets on band 4, with pre-smoothing
    feas.py -i /image.tif -o /out_dir -bp 4 -sfs_th 10 -tr sfs --smooth 5

    """)


def _options():

    colorama.init()

    text_lines = [Fore.GREEN + Style.BRIGHT + 'ctr' + Style.RESET_ALL + '     -- Copy scale centers', \
                  Fore.GREEN + Style.BRIGHT + 'dmp' + Style.RESET_ALL + '     -- Differential morphological profiles (n scales)' + Fore.RED + ' **EXPERIMENTAL**', \
                  Fore.GREEN + Style.BRIGHT + 'fourier' + Style.RESET_ALL + ' -- Fourier transform (n scales x 2)', \
                  Fore.GREEN + Style.BRIGHT + 'gabor' + Style.RESET_ALL + '   -- Gabor filter bank (n scales x 2 x kernels(Default=24))', \
                  Fore.GREEN + Style.BRIGHT + 'hog' + Style.RESET_ALL + '     -- Histogram of Oriented Gradients (4 (mean,var,skew,kurtosis) x n scales)', \
                  Fore.GREEN + Style.BRIGHT + 'hough' + Style.RESET_ALL + '   -- Local line statistics from Probabilistic Hough Transform (4 x n scales)', \
                  Fore.GREEN + Style.BRIGHT + 'lac' + Style.RESET_ALL + '     -- Lacunarity (n scales)', \
                  Fore.GREEN + Style.BRIGHT + 'lbp' + Style.RESET_ALL + '     -- Local Binary Patterns (59 x n scales)', \
                  Fore.GREEN + Style.BRIGHT + 'lbpm' + Style.RESET_ALL + '    -- Local Binary Patterns moments (4 x n scales)', \
                  Fore.GREEN + Style.BRIGHT + 'lsr' + Style.RESET_ALL + '     -- Line support regions (3 x n scales)', \
                  Fore.GREEN + Style.BRIGHT + 'mean' + Style.RESET_ALL + '    -- Local mean (n scales)', \
                  Fore.GREEN + Style.BRIGHT + 'ndvi' + Style.RESET_ALL + '    -- NDVI mean (n scales)', \
                  Fore.GREEN + Style.BRIGHT + 'pantex' + Style.RESET_ALL + '  -- Built-up presence index (n scales)', \
                  Fore.GREEN + Style.BRIGHT + 'sfs' + Style.RESET_ALL + '     -- Structural Feature Sets (4)', \
                  Fore.GREEN + Style.BRIGHT + 'surf' + Style.RESET_ALL + '    -- SURF key point descriptors (4 x n scales)' + Fore.RED + ' **EXPERIMENTAL**']

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
    parser.add_argument('--block', dest='block', help='The block size', default=2, type=int)
    parser.add_argument('--scales', dest='scales', help='The scales', default=[8], type=int, nargs='+')
    parser.add_argument('-tr', '--triggers', dest='triggers', help='The feature triggers', default=['mean'],
                        nargs='+', choices=['ctr', 'dmp', 'fourier', 'gabor', 'hog', 'hough', 'lac',
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
    parser.add_argument('--rgb2gray', dest='rgb2gray', help='RGB conversion', default=None,
                        choices=[None, 'BGR', 'RGB'])
    parser.add_argument('--convert', dest='convert', help='Whether to convert the feature stack', action='store_true')
    parser.add_argument('--stack', dest='stack', help='Whether not to stack features', action='store_false')
    parser.add_argument('--stack-only', dest='stack_only', help='Whether to only stack features', action='store_true')
    parser.add_argument('--band-red', dest='band_red', help='The red band position', default=3, type=int)
    parser.add_argument('--band-nir', dest='band_nir', help='The NIR band position', default=4, type=int)
    parser.add_argument('--neighbors', dest='neighbors', help='Whether to add features as neighbors',
                        action='store_true')
    parser.add_argument('--n-jobs', dest='n_jobs', help='The number of parallel jobs', default=-1, type=int)
    parser.add_argument('--sect-size', dest='section_size', help='The section size', default=8000, type=int)
    parser.add_argument('--chunk-size', dest='chunk_size', help='The section chunk size', default=512, type=int)
    parser.add_argument('--reset', dest='reset', help='Whether to reset section memory', action='store_true')
    parser.add_argument('--options', dest='options', help='Whether to show trigger options', action='store_true')

    args = parser.parse_args()

    if args.examples:
        _examples()

    if args.options:
        _options()

    print('\nStart date & time --- (%s)\n' % time.asctime(time.localtime(time.time())))

    start_time = time.time()

    feas(args.input, args.output, band_positions=args.band_positions, block_size=args.block, scales=args.scales,
         triggers=args.triggers, threshold=args.hline_threshold, min_len=args.hline_min, line_gap=args.hline_gap,
         weighted=args.weight, sfs_thresh=args.sfs_threshold, resamp_sfs=args.sfs_resample,
         n_angles=args.sfs_angles, smooth=args.smooth, equalize=args.equalize, equalize_adapt=args.equalize_adapt,
         visualize=args.visualize, convert_stk=args.convert, do_pca=args.pca, rgb2gray=args.rgb2gray,
         stack_feas=args.stack, stack_only=args.stack_only, band_red=args.band_red, band_nir=args.band_nir,
         neighbors=args.neighbors, n_jobs=args.n_jobs, reset_sects=args.reset, image_max=args.image_max,
         lac_r=args.lac_r, section_size=args.section_size, chunk_size=args.chunk_size)

    print('\nEnd data & time -- (%s)\nTotal processing time -- (%.2gs)\n'
          % (time.asctime(time.localtime(time.time())), (time.time()-start_time)))

if __name__ == '__main__':
    main()	
