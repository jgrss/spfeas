"""
@author: Jordan Graesser
Date Created: 7/2/2013
"""

from __future__ import division
from builtins import int, map

import os
import sys
import subprocess

from .errors import logger
from . import spfunctions
from .paths import get_path

from mpglue import raster_tools

SPFEAS_PATH = get_path()

try:
    from .sphelpers import _stats
except:
    raise ImportError('The stats functions did not load')

# Scikit-image
try:
    from skimage.exposure import equalize_hist, rescale_intensity, equalize_adapthist
except ImportError:
    raise ImportError('Scikit-learn must be installed')

# NumPy
try:
    import numpy as np
except ImportError:
    raise ImportError('NumPy must be installed')

# OpenCV
try:
    import cv2
except ImportError:
    raise ImportError('OpenCV must be installed')

# Matplotlib
try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError('Matplotlib must be installed')

# Numexpr
try:
    import numexpr as ne
except ImportError:
    raise ImportError('Numexpr must be installed')

# SciPy
try:
    from scipy.stats import linregress
except ImportError:
    raise ImportError('SciPy must be installed')

# Dask
# try:
#     import dask.array as da
# except ImportError:
#     raise ImportError('Dask must be installed')

# Pymorph
# try:
#     import pymorph
# except ImportWarning:
#     raise ImportWarning('Pymorph must be installed')

import warnings
warnings.filterwarnings('ignore')


def call_gabor(block_array_, block_size_, scales_, end_scale_):
    return _stats.feature_gabor(np.float32(block_array_), block_size_, scales_, end_scale_)


def call_fourier(block_array_, block_size_, scales_, end_scale_):
    return spfunctions.feature_fourier(block_array_, block_size_, scales_, end_scale_)


def call_dmp(block_array_, block_size_, scales_, end_scale_):
    return _stats.feature_dmp(np.float32(block_array_), block_size_, scales_, end_scale_)


# def call_hog(gradient_array_, orientation_array_, block_size_, scales_, end_scale_):
#     return _hog.feature_hog(gradient_array_, orientation_array_, block_size_, scales_, end_scale_)


def call_hog(block_array_, block_size_, scales_, end_scale_):
    return _stats.feature_hog(np.float32(block_array_), block_size_, scales_, end_scale_)


def call_hough(block_array_, block_size_, scales_, end_scale_, threshold_, min_len_, line_gap_):
    return _stats.feature_hough(block_array_, block_size_, scales_, end_scale_, threshold_, min_len_, line_gap_)


def call_lbp(block_array_, block_size_, scales_, end_scale_):
    return _stats.feature_lbp(block_array_, block_size_, scales_, end_scale_)


def call_lbpm(block_array_, block_size_, scales_, end_scale_):
    return _stats.feature_lbpm(block_array_, block_size_, scales_, end_scale_)


def call_lacunarity(block_array_, block_size_, scales_, end_scale_, lac_r_):
    return _stats.feature_lacunarity(np.uint8(block_array_), block_size_, scales_, end_scale_, lac_r_)


def call_lsr(block_array_, block_size_, scales_, end_scale_):
    return spfunctions.feature_lsr(block_array_, block_size_, scales_, end_scale_)


def call_mean(block_array_, block_size_, scales_, end_scale_):
    return _stats.feature_mean(np.float32(block_array_), block_size_, scales_, end_scale_)


def call_orb(block_array_, block_size_, scales_, end_scale_):
    return _stats.feature_orb(np.uint8(np.ascontiguousarray(block_array_)), block_size_, scales_, end_scale_)


def call_pantex(block_array_, block_size_, scales_, end_scale_, weighted_):
    return _stats.feature_pantex(np.uint8(block_array_), block_size_, scales_, end_scale_, weighted_)


def call_sfs(block_array_, block_size_, scales_, end_scale_, sfs_thresh_, sfs_skip_):
    return _stats.feature_sfs(np.uint8(block_array_), block_size_, scales_, end_scale_, sfs_thresh_, skip_factor=sfs_skip_)


def call_func(block_array_, block_size_, scales_, end_scale_, trigger_, **kwargs):

    if trigger_ in ['grad', 'mean', 'saliency', 'seg']:
        return call_mean(block_array_, block_size_, scales_, end_scale_)
    elif trigger_ == 'dmp':
        return call_dmp(block_array_, block_size_, scales_, end_scale_)
    elif trigger_ == 'fourier':
        return call_fourier(block_array_, block_size_, scales_, end_scale_)
    elif trigger_ == 'gabor':
        return call_gabor(block_array_, block_size_, scales_, end_scale_)
    elif trigger_ == 'hog':
        return call_hog(block_array_, block_size_, scales_, end_scale_)
    elif trigger_ == 'lbp':
        return call_lbp(block_array_, block_size_, scales_, end_scale_)
    elif trigger_ == 'lbpm':
        return call_lbpm(block_array_, block_size_, scales_, end_scale_)
    elif trigger_ == 'lac':
        return call_lacunarity(block_array_, block_size_, scales_, end_scale_, kwargs['lac_r'])
    elif trigger_ == 'lsr':
        return call_lsr(block_array_, block_size_, scales_, end_scale_)
    elif trigger_ == 'orb':
        return call_orb(block_array_, block_size_, scales_, end_scale_)
    elif trigger_ == 'pantex':
        return call_pantex(block_array_, block_size_, scales_, end_scale_, kwargs['weight'])
    elif trigger_ == 'sfs':
        return call_sfs(block_array_, block_size_, scales_, end_scale_, kwargs['sfs_threshold'], kwargs['sfs_skip'])

# def call_surf(block_array_, block_size_, scales_, end_scale_):
#     return _stats.feature_surf(block_array_, block_size_, scales_, end_scale_)

# def startCtr(bd_blk_scs):
#     return feaCtr(*bd_blk_scs)


def get_out_rows(in_bd, blk, end_scale):

    rows = in_bd[1] - in_bd[0]

    return len([i for i in range(0, rows-(end_scale-blk), blk)])


def get_out_cols(in_bd, blk, end_scale):

    cols = in_bd[3] - in_bd[2]

    return len([j for j in range(0, cols-(end_scale-blk), blk)])


def get_chunk_indices(rows, cols, block_size, chunk_size, scale):

    index_list = list()

    for i in range(0, rows, chunk_size-scale-block_size):

        n_rows = raster_tools.n_rows_cols(i, chunk_size, rows)

        for j in range(0, cols, chunk_size-scale-block_size):

            n_cols = raster_tools.n_rows_cols(j, chunk_size, cols)

            index_list.append((i, i+n_rows, j, j+n_cols))

    return index_list


def get_out_dims(section_rows, section_cols, parameter_object):

    """
    Gets the output section dimensions

    Args:
        section_rows (int)
        section_cols (int)
        parameter_object (class object)

    Returns:
        rows, columns
    """

    bl = parameter_object.block
    sc = parameter_object.scales[-1]
    scale_block_diff = sc - bl

    out_rows = len(range(0, section_rows-scale_block_diff, bl))
    out_cols = len(range(0, section_cols-scale_block_diff, bl))

    return out_rows, out_cols


def _get_out_dims(section_rows, section_cols, parameter_object):

    bd_idx = get_chunk_indices(section_rows, section_cols,
                               parameter_object.block,
                               parameter_object.chunk_size,
                               parameter_object.scales[-1])

    # get the number of output row and columns for each chunk
    oR = list(map(get_out_rows, bd_idx, [parameter_object.block]*len(bd_idx), [parameter_object.scales[-1]]*len(bd_idx)))
    oC = list(map(get_out_cols, bd_idx, [parameter_object.block]*len(bd_idx), [parameter_object.scales[-1]]*len(bd_idx)))

    # get the output section row and column size
    iR, jR = 0, 0
    colsR = True
    
    out_rows, out_cols = [], []
    
    for i in range(0, section_rows, parameter_object.chunk_size-(parameter_object.scales[-1]-parameter_object.block)):
    
        out_rows.append(oR[iR])
        
        for j in range(0, section_cols, parameter_object.chunk_size-(parameter_object.scales[-1]-parameter_object.block)):
        
            if colsR:
                out_cols.append(oC[jR])
                
            iR += 1
            jR += 1

        colsR = False

    out_rows = int(np.sum(out_rows))
    out_cols = int(np.sum(out_cols))

    return oR, oC, out_rows, out_cols


def sfs_orfeo(parameter_object):

    com = 'otbcli_SFSTextureExtraction -in {} -channel {:d} -ram 512 ' \
          '-parameters.spethre {:d} -parameters.spathre {:d} -parameters.nbdir 40 ' \
          '-out {}'.format(parameter_object.input_image,
                           int(parameter_object.band_position),
                           int(parameter_object.sfs_threshold),
                           int(parameter_object.scales[-1]),
                           parameter_object.out_img)

    if not os.path.isfile(parameter_object.out_img):
        subprocess.call(com, shell=True)

    with raster_tools.ropen(parameter_object.out_img) as i_info:

        # 6 layers
        for bd in range(1, i_info.bands+1):

            raster_tools.translate(parameter_object.out_img,
                                   parameter_object.out_img.replace('.tif', '.vrt'),
                                   bandList=[bd],
                                   cell_size=i_info.cellY,
                                   format='VRT',
                                   d_type='float32')

            new_image = parameter_object.out_img.replace('fea100', 'fea{:03d}'.format(bd))
            new_image = new_image.replace('bd{:d}'.format(parameter_object.band_position), 'bd-rgb')

            raster_tools.warp(parameter_object.out_img.replace('.tif', '.vrt'),
                              new_image,
                              cell_size=parameter_object.sfs_resample,
                              resampleAlg='average',
                              warpMemoryLimit=256,
                              multithread=True,
                              creationOptions=['COMPRESS=DEFLATE',
                                               'BIGTIFF=YES',
                                               'TILED=YES'])

            os.remove(parameter_object.out_img.replace('.tif', '.vrt'))

    i_info = None


def test_plot(bd, bdOrig, trigger, parameter_object):
    
    import matplotlib.cm as cm

    my_cmap = cm.autumn
    my_cmap.set_under('k', alpha=0)
    
    ax1 = plt.subplot(111)

    # subSt, subEnd = 0, -1
    
    # # bdOrig = bdOrig[subSt:subEnd, subSt:subEnd]
    # # bd = bd[subSt:subEnd, subSt:subEnd]	

    # ax1.set_xlim([0, bd.shape[1]])
    # ax1.set_ylim([0, bd.shape[0]])
    
    # ax1.imshow(bdOrig, cmap=cm.gray)#, vmin=bdOrig.min(), vmax=bdOrig.max())
        
    if trigger == 'lbp':

        ax1 = plt.subplot(111)
        # ax1 = plt.subplot(131)
        # ax2 = plt.subplot(132)
        # ax3 = plt.subplot(133)

        lbpBd, p_range = setLBP(bd)

        ax1.imshow(lbpBd[2], cmap=cm.gray, interpolation='nearest', clim=[0, lbpBd[2].max()])
        # ax2.imshow(lbpBd[1], cmap=cm.gray, interpolation='nearest', clim=[1, lbpBd[1].max()])
        # ax3.imshow(lbpBd[2], cmap=cm.gray, interpolation='nearest', clim=[1, lbpBd[2].max()])
        
    elif trigger == 'hough':

        from skimage.transform import probabilistic_hough_line as PHL
        
        # edge = np.asarray(bd, dtype=int)
        # bdCanny[(bdCanny == 1)] = 255
        
        # ax1.imshow(bdOrig, cmap=cm.gray, interpolation='nearest', clim=[bdOrig.min(), bdOrig.max()])
        ax1.imshow(bd, cmap=cm.gray, interpolation='nearest', clim=[bd.min(), bd.max()])

        # lines = cv2.HoughLinesP(bd, 1, np.radians(22.5), threshold, min_len, line_gap)[0]
        angles = [np.array([np.radians(22.5)]), np.array([np.radians(45)]), np.array([np.radians(67.5)]), \
                  np.array([np.radians(90)]), np.array([np.radians(112.5)]), np.array([np.radians(135)]), \
                  np.array([np.radians(157.5)]), np.array([np.radians(180)])]

        lines_list = []

        for angle in angles:

            lines_list.append(PHL(bd, threshold=parameter_object.hline_threshold, 
                                  line_length=parameter_object.hline_min, 
                                  line_gap=parameter_object.hline_gap, 
                                  theta=angle))

        for line_seg in lines_list:

            for line in line_seg:

                p0, p1 = line
                ax1.plot((p0[0], p1[0]), (p0[1], p1[1]))

        # [ plt.plot((line[0], line[2]), (line[1], line[3])) for line in lines ]	# opencv

    plt.show()

    sys.exit()


def get_slopes(X, y):
    slope, __, __, __, __ = linregress(X, y)
    return slope


def start_regress(X_y):
    return get_slopes(*X_y)


def wrapper(func, *args, **kwargs):

    def wrapped():
        return func(*args, **kwargs)

    return wrapped


def get_section_stats(bd, section_rows, section_cols, parameter_object, section_counter):

    """
    Split section into chunks and process features at each scale
    
    Args:
        bd (ndarray): The section array.
        section_rows (int)
        section_cols (int)
        parameter_object (class object)        
    
    Returns:
        List of computed features for each scale, for each statistic.
    """

    if parameter_object.trigger in ['pantex', 'lac']:
        out_d_range = (0, 31)
    else:
        out_d_range = (0, 255)

    # Scale the data to an 8-bit range.
    if (bd.dtype != 'uint8') and (parameter_object.trigger not in parameter_object.spectral_indices):

        bd = np.uint8(rescale_intensity(bd,
                                        in_range=(parameter_object.image_min,
                                                  parameter_object.image_max),
                                        out_range=out_d_range))

    # Apply histogram equalization.
    if parameter_object.trigger != 'dmp':

        if parameter_object.equalize:
            bd = equalize_hist(bd, nbins=256)

        elif parameter_object.equalize_adapt:
            
            bd = equalize_adapthist(bd,
                                    kernel_size=(int(section_rows / 128),
                                                 int(section_cols / 128)),
                                    clip_limit=.05,
                                    nbins=256)

        if parameter_object.equalize or parameter_object.equalize_adapt:

            bd = np.uint8(rescale_intensity(bd,
                                            in_range=(0., 1.0),
                                            out_range=(0, 255)))

        # Remove image noise.
        if parameter_object.smooth > 0:
            bd = np.uint8(cv2.bilateralFilter(bd, parameter_object.smooth, 0.1, 0.1))

    # elif parameter_object.trigger == 'lbp':
    #
    #     if parameter_object.visualize:
    #         bdOrig = bd.copy()
    #
    # elif parameter_object.trigger == 'hough':
    #
    #     # for display (testing) purposes only
    #     if parameter_object.visualize:
    #         bdOrig = bd.copy()
    #
    # # test canny and hough lines
    # if parameter_object.visualize:
    #
    #     # for display purposes only
    #     bdOrig = bd.copy()
    #
    #     test_plot(bd, bdOrig, parameter_object.trigger, parameter_object)

    # Get the row and column section chunk indices.
    # chunk_indices = get_chunk_indices(section_rows,
    #                                   section_cols,
    #                                   parameter_object.block,
    #                                   parameter_object.chunk_size,
    #                                   parameter_object.scales[-1])

    func_dict = dict(dmp=dict(name='Differential Morphological Profiles',
                              args=dict()),
                     evi2=dict(name='Two-band Enhanced Vegetation Index',
                               args=dict()),
                     fourier=dict(name='Fourier transfrom',
                                  args=dict()),
                     gabor=dict(name='Gabor filters',
                                args=dict()),
                     gndvi=dict(name='Green Normalized Difference Vegetation Index',
                                args=dict()),
                     grad=dict(name='Gradient magnitude',
                               args=dict()),
                     hog=dict(name='Histogram of Oriented Gradients',
                              args=dict()),
                     lac=dict(name='Lacunarity',
                              args=dict(lac_r=parameter_object.lac_r)),
                     lbp=dict(name='Local Binary Patterns',
                              args=dict()),
                     lbpm=dict(name='Local Binary Patterns moments',
                               args=dict()),
                     lsr=dict(name='Line support regions',
                              args=dict()),
                     mean=dict(name='Mean',
                               args=dict()),
                     ndvi=dict(name='Normalized Difference Vegetation Index',
                               args=dict()),
                     pantex=dict(name='PanTex',
                                 args=dict(weight=parameter_object.weight)),
                     orb=dict(name='Oriented FAST and Rotated BRIEF key points',
                              args=dict()),
                     saliency=dict(name='Image saliency',
                                   args=dict()),
                     seg=dict(name='Segmentation',
                              args=dict()),
                     sfs=dict(name='Structural Feature Sets',
                              args=dict(sfs_threshold=parameter_object.sfs_threshold,
                                        sfs_skip=parameter_object.sfs_skip)))

    for idx in parameter_object.spectral_indices:
        if idx not in func_dict:
            func_dict[idx] = {'name': idx, 'args': {}}

    logger.info('  Processing {} for section {:,d} of {:,d} ...'.format(func_dict[parameter_object.trigger]['name'],
                                                                        section_counter,
                                                                        parameter_object.n_sects))

    other_args = func_dict[parameter_object.trigger]['args']

    if parameter_object.trigger in parameter_object.spectral_indices:
        trigger = 'mean'
    else:
        trigger = parameter_object.trigger

    return call_func(bd,
                     parameter_object.block,
                     parameter_object.scales,
                     parameter_object.scales[-1],
                     trigger,
                     **other_args)

    # return Parallel(n_jobs=parameter_object.n_jobs_chunk,
    #                 max_nbytes=None)(delayed(call_func)(bd[chi[0]:chi[1],
    #                                                     chi[2]:chi[3]],
    #                                                     parameter_object.block,
    #                                                     parameter_object.scales,
    #                                                     parameter_object.scales[-1],
    #                                                     parameter_object.trigger,
    #                                                     **other_args) for chi in chunk_indices)
