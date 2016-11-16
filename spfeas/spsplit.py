"""
@author: Jordan Graesser
Date Created: 7/2/2013
"""

import os
import timeit
import subprocess
from joblib import Parallel, delayed

from spfunctions import *
from paths import get_path

SPFEAS_PATH = get_path()

try:
    from sphelpers import _stats
except:
    raise ImportError('The stats functions did not load')

try:
    from sphelpers import _chunk
except:
    raise ImportError('The chunk functions did not load')

try:
    from sphelpers.gabor_filter_bank import prep_gabor
except:
    print('\n!!!\nWarning: skimage.filter.gabor_kernel did not load\n \
    Cannot compute Gabor features\n Upgrade to latest scikit-image')

# Scikit-image
try:
    from skimage.exposure import equalize_hist, rescale_intensity, equalize_adapthist
    from skimage.filter import canny
except ImportError:
    raise ImportError('Scikits must be installed')

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


def call_gabor(block_array_, block_size_, scales_, end_scale_, kernels_):
    return _stats.feature_gabor(block_array_, block_size_, scales_, end_scale_, kernels_)


def call_fourier(block_array_, block_size_, scales_, end_scale_):
    return feature_fourier(block_array_, block_size_, scales_, end_scale_)


def call_hog(gradient_array_, orientation_array_, block_size_, scales_, end_scale_):
    return _stats.feature_hog(gradient_array_, orientation_array_, block_size_, scales_, end_scale_)


def call_hough(block_array_, block_size_, scales_, end_scale_, threshold_, min_len_, line_gap_):
    return _stats.feature_hough(block_array_, block_size_, scales_, end_scale_, threshold_, min_len_, line_gap_)


def call_lbp(block_array_, block_size_, scales_, end_scale_):
    return _stats.feature_lbp(block_array_, block_size_, scales_, end_scale_)


def call_lbpm(block_array_, block_size_, scales_, end_scale_):
    return _stats.feature_lbpm(block_array_, block_size_, scales_, end_scale_)


def call_lacunarity(block_array_, block_size_, scales_, end_scale_, lac_r_):
    return _stats.feature_lacunarity(block_array_, block_size_, scales_, end_scale_, lac_r_)


def call_lsr(block_array_, block_size_, scales_, end_scale_):
    return feature_lsr(block_array_, block_size_, scales_, end_scale_)


def call_mean(block_array_, block_size_, scales_, end_scale_):
    return _stats.feature_mean(block_array_, block_size_, scales_, end_scale_)


def call_pantex(block_array_, block_size_, scales_, end_scale_, weighted_):
    return _stats.feature_pantex(block_array_, block_size_, scales_, end_scale_, weighted_)


def call_sfs(block_array_, block_size_, scales_, end_scale_, cell_size_, sfs_thresh_, n_angles_):
    return _stats.feature_sfs(block_array_, block_size_, scales_, end_scale_, cell_size_, sfs_thresh_, n_angles_)


def startCtr(bd_blk_scs):
    return feaCtr(*bd_blk_scs)


def getOutRows(in_bd, blk, scs):
    
    rows, cols = in_bd.shape
    
    return len([i for i in xrange(0, rows-(scs[-1]-blk), blk)])


def getOutCols(in_bd, blk, scs):
    
    rows, cols = in_bd.shape

    return len([j for j in xrange(0, cols-(scs[-1]-blk), blk)])


def get_out_dims(bd, section_rows, section_cols, parameter_object):

    bd = _chunk.chunk_int(bd, section_rows, section_cols,
                          parameter_object.block, parameter_object.chunk_size, parameter_object.scales[-1])

    # get the number of output row and columns for each chunk
    oR = map(getOutRows, bd, [parameter_object.block]*len(bd), [parameter_object.scales]*len(bd))
    oC = map(getOutCols, bd, [parameter_object.block]*len(bd), [parameter_object.scales]*len(bd))

    # get the output section row and column size
    iR, jR = 0, 0
    colsR = True
    
    out_rows, out_cols = [], []
    
    for i in xrange(0, section_rows, parameter_object.chunk_size-(parameter_object.scales[-1]-parameter_object.block)):
    
        out_rows.append(oR[iR])
        
        for j in xrange(0, section_cols, parameter_object.chunk_size-(parameter_object.scales[-1]-parameter_object.block)):
        
            if colsR:
                out_cols.append(oC[jR])
                
            iR += 1
            jR += 1

        colsR = False

    out_rows = int(np.sum(out_rows))
    out_cols = int(np.sum(out_cols))

    return oR, oC, out_rows, out_cols


def get_dmp(bd, section_rows, section_cols):
    
    # if bd.dtype != np.uint8:
    #     bd = raster_tools.rescale_intensity(bd, 256, maxI=mx, minI=mn, dType='i').astype(np.uint8)

    # set the structuring element
    # se = cv2.getStructuringElement(cv2.MORPH_RECT, (se_size, se_size))
    # er = cv2.erode(sect_in, se, iterations=3)
    # di = cv2.dilate(sect_in, se, iterations=3)
    # # op = cv2.morphologyEx(sect_in, cv2.MORPH_OPEN, se, iterations=3)
    #
    # ## reconstruction
    # ## choose a random seed point and check for clouds
    # ## the seed point should be non-cloud
    # rws, cls = se.shape
    # no_seed = True
    # while no_seed:
    #
    #     row_seed = np.random.choice(range(rws), size=1, replace=False)[0]
    #     col_seed = np.random.choice(range(cls), size=1, replace=False)[0]
    #
    #     if se[row_seed, col_seed] != 0:
    #         no_seed = False
    #
    # sect_in_openrc_cv = np.pad(er, ((1, 1), (1, 1)), 'edge').astype(np.uint8)
    # cv2.floodFill(sect_in.astype(np.uint8), sect_in_openrc_cv, (col_seed, row_seed), (0, 0, 255), \
    #               flags=cv2.FLOODFILL_MASK_ONLY)

    ses = [3, 5, 7, 9, 11]
    X = np.float32(np.arange(len(ses) * 2))

    # holder for DMP
    # closings --> 1st len(ses) bands
    # openings --> last len(ses) bands
    dmp = np.uint8(np.empty((len(ses) * 2, section_rows, section_cols)))

    dmp_pos = 0
    for se_size in ses:

        se = cv2.getStructuringElement(cv2.MORPH_RECT, (se_size, se_size))

        # OpenCV morphological reconstruction

        # open by reconstruction

        # bd = mask
        # er = marker

        dmp_lyr = np.uint8(np.empty((bd.shape[0], section_rows, section_cols)))

        for lyr in xrange(0, bd.shape[0]):

            er_openrc = cv2.erode(bd[lyr], se, iterations=1)

            # changes = np.zeros((section_rows, section_cols)).astype(np.uint8)

            # stable = False
            max_iters = 5
            for iters in xrange(0, max_iters):
                # geodesic_er = np.logical_or(bd, er_openrc).astype(int)
                geodesic_er = np.where(np.abs(np.subtract(bd[lyr], er_openrc)) < 100, 1, 0)

                # changes = np.subtract(geodesic_er, changes)

                # if changes.max() == 0:
                #     stable = True
                # else:
                er_openrc = cv2.erode(np.where(geodesic_er == 1, bd[lyr], 0), se, iterations=1)

            dmp_lyr[lyr] = er_openrc

        dmp[dmp_pos + len(ses)] = -dmp_lyr.mean(axis=0)

        # closing by reconstruction

        for lyr in xrange(0, bd.shape[0]):

            di_closerc = cv2.dilate(bd[lyr], se, iterations=1)

            # changes = np.zeros((section_rows, section_cols)).astype(np.uint8)

            # stable = False
            for iters in xrange(0, max_iters):
                # geodesic_di = np.logical_and(bd, di_closerc).astype(int)
                geodesic_di = np.where(np.abs(np.subtract(bd[lyr], di_closerc)) < 100, 1, 0)

                # changes = np.subtract(geodesic_di, changes)

                # if changes.max() == 0:
                #     stable = True
                # else:
                di_closerc = cv2.dilate(np.where(geodesic_di == 1, bd[lyr], 0), se, iterations=1)

            dmp_lyr[lyr] = di_closerc

        dmp[dmp_pos] = dmp_lyr.mean(axis=0)

        dmp_pos += 1

    close_mean = dmp[:len(ses)].mean(axis=0)
    open_mean = dmp[len(ses):].mean(axis=0)

    return np.float32(np.multiply(close_mean, open_mean))

    # plt.subplot(221)
    # plt.imshow(bd[2])
    # plt.subplot(222)
    # plt.imshow(close_mean)
    # plt.subplot(223)
    # plt.imshow(open_mean)
    # plt.subplot(224)
    # plt.imshow(open_mean * close_mean)
    # plt.show()
    # sys.exit()

    # dmp = dmp.reshape(len(X), section_rows*section_cols).T
    # dmp = [dm for dm in dmp]
    #
    # X = np.indices((section_rows*section_cols, len(X)))[1]
    # X = [x for x in X]
    #
    # # create the multiprocessing pool object
    # if n_jobs == -1:
    #     pool = M.Pool(processes=M.cpu_count())
    # else:
    #     pool = M.Pool(processes=n_jobs)
    #
    # slopes = pool.map(start_regress, itertools.izip(X, dmp))
    # pool.close()
    #
    # del dmp, X
    #
    # bd = np.asarray(slopes).astype(np.float32).reshape(section_rows, section_cols)
    #
    # del slopes

    # else:
    #
    #     if bd.dtype != np.uint8:
    # bd = rescale_intensity(bd, in_range=(mn, mx), out_range=(0, 255))
    # bd = raster_tools.rescale_intensity(bd, 256, maxI=mx, minI=mn, dType='i').astype(np.uint8)
    # bd = rescale_intensity(bd, in_range=(mn, mx), out_range=(0, 255))
    

def get_mag_ang(img):

    """
    Get image gradient (magnitude) and orientation (angle) from a Sobel operator
    """

    gx = cv2.Sobel(np.float32(img), cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(np.float32(img), cv2.CV_32F, 0, 1)

    grad, ori = cv2.cartToPolar(gx, gy)

    return grad, ori


def getDist(line):
    return np.sqrt(((line[0][0] - line[1][0])**2.) + ((line[0][1] - line[1][1])**2.))


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


def get_sect_feas(bd, section_rows, section_cols, mn, mx, cell_size, trigger, parameter_object):

    """
    Split section into chunks and process features at each scale
    
    Args:
        bd (ndarray): Section array.
        blk_size (int)
        scs (list)
        trigger (str)
        chunk_size (int)
        mn (int)
        mx (int)
        min_len (int)
        weighted (bool)
        equalize (bool)
        blur (bool)
        vis (bool)
    
    Returns:
        List of computed features for each scale, for each statistic.
    """

    # if (bd.dtype != 'uint8') and (bd.dtype != 'uint16'):
    #     raise TypeError('\nThe input image should be stored in an unsigned 8-bit or 16-bit dynamic range.')

    # if image_max > 0:
    #     n_bins = image_max + 1
    # else:
    #     n_bins = 256

    # Scale the data to an 8-bit range.
    if trigger != 'dmp':

        bd = np.uint8(rescale_intensity(bd, in_range=(mn, mx), out_range=(0, 255)))

        # histogram equalization
        if parameter_object.equalize:
            bd = equalize_hist(bd, nbins=256)

        elif parameter_object.equalize_adapt:
            
            bd = equalize_adapthist(bd,
                                    ntiles_x=int(section_cols / 128),
                                    ntiles_y=int(section_rows / 128),
                                    clip_limit=.05,
                                    nbins=256)

        if parameter_object.equalize or parameter_object.equalize_adapt:
            bd = np.uint8(rescale_intensity(bd, in_range=(0., 1.), out_range=(0, 255)))

    # remove image noise
    if parameter_object.smooth > 0:

        # if bd.dtype != np.uint8:
            # bd = raster_tools.rescale_intensity(bd, 256, maxI=bd.max(), minI=0, dType='i').astype(np.uint8)
            # bd = rescale_intensity(bd, in_range=(mn, mx), out_range=(0, 255)).astype(np.uint8)

        # bd = cv2.GaussianBlur(bd, (3,3), 1)
        #bd = cv2.fastNlMeansDenoising(bd, h=smooth, templateWindowSize=7, searchWindowSize=21)
        # bd = cv2.medianBlur(bd, smooth)
        bd = np.uint8(cv2.bilateralFilter(bd, parameter_object.smooth, 5, 5))

    # get image gradient and orientation for HoG
    if trigger == 'hog':
        grad_img, ori_img = get_mag_ang(bd)
        
    # rescale for GLCM
    elif (trigger == 'pantex') or (trigger == 'lac'):
        
        # rescale to 32 grey scales for GLCM
        # bd = raster_tools.rescale_intensity(bd, 32, maxI=mx, minI=mn, dType='i').astype(np.uint8)
        bd = np.uint8(rescale_intensity(bd, in_range=(0, 255), out_range=(0, 31)))
        # bd = rescale_intensity(bd, in_range=(mn, mx), out_range=(0, 31))

    elif trigger == 'lbp':
        
        if parameter_object.visualize:
            bdOrig = bd.copy()

    elif trigger == 'hough':

        # rescale to byte for canny
        # if bd.dtype != np.uint8:

            # bd = raster_tools.rescale_intensity(bd, 256, maxI=bd.max(), minI=0, dType='i').astype(np.uint8)	# min./max. stretch
            # bd = rescale_intensity(bd, in_range=(mn, mx), out_range=(0, 255))

        # for display (testing) purposes only
        if parameter_object.visualize:
            bdOrig = bd.copy()

        # extract Canny edges
        # bd = cv2.Canny(bd.astype(np.uint8), .1*bd.max(), .2*bd.max(), apertureSize=3)		# 32, 16
        # bd = cv2.Canny(bd.astype(np.uint8), 32, 16, apertureSize=3)

        # bd, ori_img = get_mag_ang(bd)
        # del ori_img
        #
        # bd = np.where(bd > 250, 255, 0).astype(np.uint8)

        # plt.imshow(bd, cmap='gray')
        # plt.show()
        # sys.exit()

        # convert from boolean to integer if using OpenCV Canny?
        ##bd = np.asarray(bd).astype(np.uint8)
        ##bd[(bd == 1)] = 255
        
        # scikits
        # bd 		= canny(bd.astype(np.uint8), sigma=1)#, low_threshold=12, high_threshold=24)	# Scikits implementation of Canny	

    elif trigger == 'dmp':
        bd = get_dmp(bd, section_rows, section_cols)

    # test canny and hough lines
    if parameter_object.visualize:
        
        # if trigger != 'hough':
        
           # bd, _	= histEq(bd, numBins=256)	# histogram equalization
           # bd = equalize_hist(bd, nbins=256)

        # for display purposes only
        bdOrig = bd.copy()

        test_plot(bd, bdOrig, trigger, parameter_object)

    # split the image array into overlapping chunks where
    # the overlap is determined by the output pixel block size and
    # the maximum scale
    if trigger == 'hog':

        grad_img = _chunk.chunk_float(grad_img, section_rows, section_cols,
                                      parameter_object.block,
                                      parameter_object.chunk_size,
                                      parameter_object.scales[-1])

        ori_img = _chunk.chunk_float(ori_img, section_rows, section_cols,
                                     parameter_object.block,
                                     parameter_object.chunk_size,
                                     parameter_object.scales[-1])

    # elif trigger == 'ndvi':

        # bd_4 = bd[0].astype(np.float32)
        # bd_3 = bd[1].astype(np.float32)
        #
        # bd = ne.evaluate('(bd_4 - bd_3) / (bd_4 + bd_3)')

        # vie = VegIndicesEquations(bd)
        # bd = vie.NDVI()

        # bd = _chunk.chunk_int(bd, section_rows, section_cols, blk_size, chunk_size, scs[-1])

    elif trigger == 'dmp':

        bd = _chunk.chunk_float(bd, section_rows, section_cols,
                                parameter_object.block, parameter_object.chunk_size, parameter_object.scales[-1])

    else:

        bd = _chunk.chunk_int(bd, section_rows, section_cols,
                              parameter_object.block, parameter_object.chunk_size, parameter_object.scales[-1])

    if trigger in ['mean', 'ndvi', 'objects', 'dmp']:

        if trigger == 'mean':
            print '\n  Processing mean ...'
        else:
            print '\n  Processing mean {} ...'.format(trigger)

        # def mean_wrapper(block, blk_size, scs, n_jobs):
        #     return Parallel(n_jobs=8, max_nbytes=None)(delayed(call_mean)(bd_, blk_size, scs, scs[-1]) \
        #                                                    for bd_ in block)
        #
        # wrapped = wrapper(mean_wrapper, bd, blk_size, scs, n_jobs)
        # print 'Time: %f' % timeit.timeit(wrapped, number=1)

        return Parallel(n_jobs=parameter_object.n_jobs,
                        max_nbytes=None)(delayed(call_mean)(bd_, parameter_object.block,
                                                            parameter_object.scales,
                                                            parameter_object.scales[-1]) for bd_ in bd)

    elif trigger == 'lac':

        print '\n  Processing lacunarity ...'

        return Parallel(n_jobs=parameter_object.n_jobs,
                        max_nbytes=None)(delayed(call_lacunarity)(bd_, parameter_object.block,
                                                                  parameter_object.scales,
                                                                  parameter_object.scales[-1],
                                                                  parameter_object.lac_r) for bd_ in bd)

    # elif trigger == 'ctr':
    #
    #     print '\n  Copying band scales ...'
    #     return pool.map(startCtr, itertools.izip(bd, [blk_size]*len(bd), [scs]*len(bd)))

    elif trigger == 'gabor':

        print '\n  Processing Gabor ...'

        kernels = prep_gabor(n_orientations=32, sigmas=[1, 2, 4])

        return Parallel(n_jobs=parameter_object.n_jobs,
                        max_nbytes=None)(delayed(call_gabor)(bd_, parameter_object.block,
                                                             parameter_object.scales,
                                                             parameter_object.scales[-1], kernels) for bd_ in bd)

    elif trigger == 'fourier':

        print '\n  Processing Fourier ...'

        return Parallel(n_jobs=parameter_object.n_jobs,
                        max_nbytes=None)(delayed(call_fourier)(bd_, parameter_object.block,
                                                               parameter_object.scales,
                                                               parameter_object.scales[-1]) for bd_ in bd)

    elif trigger == 'hog':

        print '\n  Processing HoG ...'

        return Parallel(n_jobs=parameter_object.n_jobs,
                        max_nbytes=None)(delayed(call_hog)(gim, oim, parameter_object.block,
                                                           parameter_object.scales,
                                                           parameter_object.scales[-1])
                                         for gim, oim in itertools.izip(grad_img, ori_img))

    elif trigger == 'hough':

        print '\n  Processing Hough lines ...'

        return Parallel(n_jobs=parameter_object.n_jobs,
                        max_nbytes=None)(delayed(call_hough)(bd_, parameter_object.block,
                                                             parameter_object.scales,
                                                             parameter_object.scales[-1],
                                                             parameter_object.hline_threshold,
                                                             parameter_object.hline_min,
                                                             parameter_object.hline_gap) for bd_ in bd)

    elif trigger == 'lbp':

        print '\n  Processing LBP ...'

        return Parallel(n_jobs=parameter_object.n_jobs,
                        max_nbytes=None)(delayed(call_lbp)(bd_, parameter_object.block,
                                                           parameter_object.scales,
                                                           parameter_object.scales[-1]) for bd_ in bd)

    elif trigger == 'lbpm':

        print '\n  Processing LBP ...'

        return Parallel(n_jobs=parameter_object.n_jobs,
                        max_nbytes=None)(delayed(call_lbpm)(bd_, parameter_object.block,
                                                            parameter_object.scales,
                                                            parameter_object.scales[-1]) for bd_ in bd)

    elif trigger == 'lsr':

        print '\n  Processing LSR ...'

        return Parallel(n_jobs=parameter_object.n_jobs,
                        max_nbytes=None)(delayed(call_lsr)(bd_, parameter_object.block,
                                                           parameter_object.scales,
                                                           parameter_object.scales[-1]) for bd_ in bd)

    elif trigger == 'pantex':

        print '\n  Processing PanTex ...'

        # return pool.map(startPanTex, itertools.izip(bd, [blk_size]*len(bd), [scs]*len(bd), [weighted]*len(bd)))

        return Parallel(n_jobs=parameter_object.n_jobs,
                        max_nbytes=None)(delayed(call_pantex)(bd_, parameter_object.block,
                                                              parameter_object.scales,
                                                              parameter_object.scales[-1],
                                                              parameter_object.weight) for bd_ in bd)

        # tsk	= map(feaPanTex, bd, [blk_size]*len(bd), [scs]*len(bd), [weighted]*len(bd))

    elif trigger == 'sfs':

        print '\n  Processing SFS ...'

        return Parallel(n_jobs=parameter_object.n_jobs,
                        max_nbytes=None)(delayed(call_sfs)(bd_, parameter_object.block,
                                                           parameter_object.scales,
                                                           parameter_object.scales[-1], cell_size,
                                                           parameter_object.sfs_threshold,
                                                           parameter_object.sfs_angles) for bd_ in bd)

    else:
        raise NameError('\nThe trigger is not recognized\n')
