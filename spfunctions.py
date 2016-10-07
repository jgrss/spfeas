#!/usr/bin/env python

import sys
import itertools

from sphelpers import lsr
from sphelpers.spatial_pyramid_hist import pyramid_hist_sift

try:
    from skimage.feature import hog as HOG
    from skimage.feature import local_binary_pattern as LBP
    from skimage.feature import greycomatrix, greycoprops
    from skimage.exposure import histogram
except ImportError:
    raise ImportError('Scikits-image must be installed')
    
try:
    import numpy as np
except ImportError:
    raise ImportError('NumPy must be installed')
    
try:
    import cv2
except ImportError:
    raise ImportError('OpenCV must be installed')
    
try:
    from scipy.stats import moment, skew, kurtosis
    from scipy.ndimage import convolve as nd_convolve
except ImportError:
    raise ImportError('SciPy must be installed')

try:
    from bottleneck import nanvar, nanmean
except ImportError:
    raise ImportError('Bottleneck must be installed')

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError('Matplotlib must be installed')

global pi
pi = 3.14159265

global pi2
pi2 = pi / 2.


# list of descriptive statistics functions
desc_stats = [nanmean, nanvar, skew, kurtosis]
# desc_stats = [get_mean, get_var, skew, kurtosis]


def azimuthal_avg(image, center=None):

    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is
             None, which then uses the center of the image (including
             fracitonal pixels).

    """

    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max() - x.min()) / 2.0, (x.max() - x.min()) / 2.0])

    #   : get hypotenuse
    r = np.hypot(np.subtract(x, center[0]), np.subtract(y, center[1]))

    #   : get sorted radii indices
    ind = np.argsort(r.flat)
    rSorted = r.flat[ind]

    #   : get image values from index positions
    iSorted = image.flat[ind]

    #   : Get the integer part of the radii (bin size = 1)
    rInt = rSorted.astype(int)

    #   : Find all pixels that fall within each radial bin.
    deltar = np.subtract(rInt[1:], rInt[:-1])  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = np.subtract(rind[1:], rind[:-1])        # number of radius bin

    #   : Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(iSorted, dtype=float)
    tbin = np.subtract(csim[rind[1:]], csim[rind[:-1]])

    radialProf = np.divide(tbin, nr)

    return radialProf


def feature_fourier(chBd, blk, scs, end_scale):

    rows, cols = chBd.shape
    scales_half = end_scale / 2
    scales_blk = end_scale - blk
    out_len = 0
    pix_ctr = 0

    for i in xrange(0, rows-scales_blk, blk):
        for j in xrange(0, cols-scales_blk, blk):
            for k in scs:
                out_len += 2

    # set the output list
    out_list = np.zeros(out_len).astype(np.float32)

    for i in xrange(0, rows-scales_blk, blk):
        for j in xrange(0, cols-scales_blk, blk):
            for k in scs:

                ch_bd = chBd[i+scales_half-(k/2):i+scales_half-(k/2)+k, j+scales_half-(k/2):j+scales_half-(k/2)+k]

                # get the Fourier Transform
                dft = cv2.dft(np.float32(ch_bd), flags=cv2.DFT_COMPLEX_OUTPUT)
                dft_shift = np.fft.fftshift(dft)

                # get the Power Spectrum
                magnitude_spectrum = 20. * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

                psd1D = azimuthal_avg(magnitude_spectrum)

                sts = list(cv2.meanStdDev(psd1D))

                # plt.subplot(121)
                # plt.imshow(ch_bd, cmap='gray')
                # plt.subplot(122)
                # plt.imshow(magnitude_spectrum, interpolation='nearest')
                # plt.show()
                # print psd1D
                # sys.exit()

                for st in sts:

                    if np.isnan(st[0][0]):
                        out_list[pix_ctr] = 0.
                    else:
                        out_list[pix_ctr] = st[0][0]

                    pix_ctr += 1

    out_list[np.isnan(out_list) | np.isinf(out_list)] = 0.

    return out_list


def setLBP(chBd):

    """
    Get the Local Binary Patterns
    """

    rows, cols = chBd.shape

    # create LBP radius lookup dictionary
    Rdict = {4: 1, 8: 1, 16: 2, 32: 4, 64: 8, 128: 16}

    # build the P ranges
    p_range = [8, 16, 32]

    # create empty array for LBP
    lbpBd = np.zeros((len(p_range), rows, cols)).astype(np.uint8)

    # run lBP for each scale
    for scsC in xrange(0, len(p_range)):

        lbpBd[scsC] = LBP(chBd, p_range[scsC], Rdict[p_range[scsC]], 'uniform')

    return lbpBd, p_range


def feaLBP(chBd, blk, scs):

    """
    Returns at each scale
    ---------------------
    Concatenated histograms
    """

    rows, cols = chBd.shape
    scales_half = scs[-1] / 2
    scales_blk = scs[-1] - blk
    out_len = 0
    pix_ctr = 0

    # get the LBP images
    lbpBd, p_range = setLBP(chBd, rows, cols)

    # count of bins for all p,r LBP pairs
    pr_bin_count = np.sum([pr+2 for pr in p_range])

    for i in xrange(0, rows-scales_blk, blk):
        for j in xrange(0, cols-scales_blk, blk):
            for k in scs:
                out_len += pr_bin_count

    # set the output list
    out_list = np.zeros(out_len).astype(np.uint8)

    for i in xrange(0, rows-scales_blk, blk):
        for j in xrange(0, cols-scales_blk, blk):
            for k in scs:

                ch_bd = lbpBd[:, i+scales_half-(k/2):i+scales_half-(k/2)+k, j+scales_half-(k/2):j+scales_half-(k/2)+k]

                # get histograms and concatenate
                # sts = np.concatenate([ np.histogram(ch_bd[p_range.index(pc)], bins=pc+1, range=(0, pc))[0] for pc \
                #                        in p_range ]).astype(np.uint8)

                sts = np.concatenate([np.bincount(ch_bd[p_range.index(pc)].flat, minlength=pc+2) for pc in \
                                      p_range]).astype(np.uint8)

                #[out_list.append(st) for st in sts]
                for st in sts:
                    out_list[pix_ctr] = st

                    pix_ctr += 1

    # out_list[np.isnan(out_list) | np.isinf(out_list)] = 0.

    return list(out_list)


def feature_lsr(chBd, blk, scs, end_scale):

    rows, cols = chBd.shape
    out_list = []

    edge_ori, edge_mag, deriv_x, deriv_y = lsr.grad_mag(chBd)
    
    for i in xrange(0, rows-(end_scale-blk), blk):
        for j in xrange(0, cols-(end_scale-blk), blk):
            for k in scs:

                edoim_s = edge_ori[i+(end_scale/2)-(k/2):i+(end_scale/2)-(k/2)+k, j+(end_scale/2)-(k/2):j+(end_scale/2)-(k/2)+k]
                edmim_s = edge_mag[i+(end_scale/2)-(k/2):i+(end_scale/2)-(k/2)+k, j+(end_scale/2)-(k/2):j+(end_scale/2)-(k/2)+k]
                dx_s = deriv_x[i+(end_scale/2)-(k/2):i+(end_scale/2)-(k/2)+k, j+(end_scale/2)-(k/2):j+(end_scale/2)-(k/2)+k]
                dy_s = deriv_y[i+(end_scale/2)-(k/2):i+(end_scale/2)-(k/2)+k, j+(end_scale/2)-(k/2):j+(end_scale/2)-(k/2)+k]
    
                sts = lsr._feature_lsr(edoim_s, edmim_s, dx_s, dy_s)
                
                [out_list.append(st) for st in sts]
            
    return out_list


def panTexFuncWeighted(panTexArr, dists, dispVect):
    
    '''
    Get the local minimum contrast for all displacement vectors
    weighted by the DN value
    '''	
    
    if panTexArr.max() == 0:
        return 0.
    else:
        glcm = greycomatrix(panTexArr, dists, dispVect, levels=32, symmetric=True, normed=True)

        return np.array([greycoprops(glcm, 'contrast')[dist-1][dV] for dV in xrange(0, len(dispVect)) for dist in dists]).min() * panTexArr.mean()


def panTexFunc(panTexArr, dists, dispVect):
    
    """
    Get the local minimum contrast for all displacement vectors
    """
    
    if panTexArr.max() == 0:
        return 0.
    else:
        glcm_mat = greycomatrix(panTexArr, dists, dispVect, levels=32, symmetric=True, normed=True)

        return np.array([greycoprops(glcm_mat, 'contrast')[dist-1][dV] for dV in xrange(0, len(dispVect)) for dist in dists]).min()


def feaPanTex(chBd, blk, scs, weighted):
    
    """
    Get the Anisotropic Built-up Presence Index
    """

    rows, cols	= chBd.shape

    scales_half = scs[-1] / 2
    scales_blk = scs[-1] - blk
    out_len = 0
    pix_ctr = 0

    for i in xrange(0, rows-scales_blk, blk):
        for j in xrange(0, cols-scales_blk, blk):
            for k in scs:
                out_len += 1

    # set the output list
    out_list = np.zeros(out_len).astype(np.float64)

    # directions [E, NE, N, NW]
    dispVect = [0., pi/6., pi/4., pi/3., pi/2., (2.*pi)/3., (3.*pi)/4., (5.*pi)/6.]
    dists = [1, 2]

    if weighted:

        for i in xrange(0, rows-scales_blk, blk):
            for j in xrange(0, cols-scales_blk, blk):
                for k in scs:


                    out_list[pix_ctr] = panTexFuncWeighted(chBd[i+scales_half-(k/2):i+scales_half-(k/2)+k, \
                                                           j+scales_half-(k/2):j+scales_half-(k/2)+k], dists, dispVect)

                    pix_ctr += 1

    else:

        for i in xrange(0, rows-scales_blk, blk):
            for j in xrange(0, cols-scales_blk, blk):
                for k in scs:

                    out_list[pix_ctr] = panTexFunc(chBd[i+scales_half-(k/2):i+scales_half-(k/2)+k, \
                                                   j+scales_half-(k/2):j+scales_half-(k/2)+k], dists, dispVect)

                    pix_ctr += 1

    return out_list


    # if weighted:
    #     return [ panTexFuncWeighted(chBd[i+(scs[-1]/2)-(k/2):i+(scs[-1]/2)-(k/2)+k, j+(scs[-1]/2)-(k/2):j+(scs[-1]/2)-(k/2)+k], dists, dispVect) \
    #             for k in scs for (i,j) in itertools.product(xrange(0, rows-(scs[-1]-blk), blk), xrange(0, cols-(scs[-1]-blk), blk)) ]
    # else:
    #     return [ panTexFunc(chBd[i+(scs[-1]/2)-(k/2):i+(scs[-1]/2)-(k/2)+k, j+(scs[-1]/2)-(k/2):j+(scs[-1]/2)-(k/2)+k], dists, dispVect) \
    #             for k in scs for (i,j) in itertools.product(xrange(0, rows-(scs[-1]-blk), blk), xrange(0, cols-(scs[-1]-blk), blk)) ]


def surfFunc(surfArr, kPts, j, i, k, scs):
    
    """
    Get the moments
    """

    start_y = i+(scs[-1]/2)-(k/2)
    start_x = j+(scs[-1]/2)-(k/2)

    if surfArr.max() == 0:
        return 0.	# 21 length vector to match pyramid histogram length (1+4+16)
    else:
        if kPts:
            # return desc_stats[m](pyramid_hist_sift(surfArr, kPts, start_x, start_y).sp_hist)
            return get_moments(pyramid_hist_sift(surfArr, kPts, start_x, start_y).sp_hist)
        else:
            return [0., 0., 0., 0.]


def feaSURF(chBd, blk, scs):

    rows, cols	= chBd.shape

    # compute SURF features
    kPts, descrip = cv2.SURF(50).detectAndCompute(chBd, None)

    for i in xrange(0, rows-scales_blk, blk):
        for j in xrange(0, cols-scales_blk, blk):
            for k in scs:

                sts = surfFunc(chBd[i+scales_half-(k/2):i+scales_half-(k/2)+k, \
                               j+scales_half-(k/2):j+scales_half-(k/2)+k], kPts, j, i, k, scs)


    return [ surfFunc(chBd[i+(scs[-1]/2)-(k/2):i+(scs[-1]/2)-(k/2)+k, j+(scs[-1]/2)-(k/2):j+(scs[-1]/2)-(k/2)+k], kPts, j, i, k, scs, m) \
            for k in scs for m in xrange(0, 4) for (i,j) in itertools.product(xrange(0, rows-(scs[-1]-blk), blk), xrange(0, cols-(scs[-1]-blk), blk)) ]


def feaSFS_C(inImg, outImg, B, blk, spe, spa, nb, al, mxc, mem):

    '''
    Structural Feature Sets from the Orfeo OTB library
    '''

    dName, fName= os.path.split(outImg)
    fBase, fExt = os.path.splitext(fName)

    outImgSFStemp = '%s/%s_sfs_temp%s' % (dName, fBase, fExt)
    outImgSFS = '%s/%s_sfs%s' % (dName, fBase, fExt)

    com = 'otbcli_SFSTextureExtraction -progress true -in %s -channel %d -ram %d -parameters.spethre %f \
        -parameters.spathre %d -parameters.nbdir %d -parameters.alpha %f -parameters.maxcons %d -out %s' % \
          (inImg, B, mem, spe, spa, nb, al, mxc, outImgSFStemp)

    if os.path.isfile(outImgSFS):
        print '\n%s already exists ...\n' % outImgSFS
    else:

        try:
            subprocess.call(com, shell=True)
            print
        except:
            sys.exit('\nERROR!! The OTB Toolbox Python Application must be installed.\n\n \
                    Download Monteverdi at http://sourceforge.net/projects/orfeo-toolbox/files/\n \
                    Download the Python OTB Application at http://trac.osgeo.org/osgeo4w/\n')

    # or just resample here
    if blk > 1:

        iInfo = getRstInfo(outImgSFStemp)

        print '\nResampling SFS features ...\n'

        resamp = 'gdalwarp -tr %f %f -r cubic -co COMPRESS=LZW %s %s' % (float(blk)*iInfo.cellY, \
                                                                         float(blk)*iInfo.cellY, outImgSFStemp, \
                                                                         outImgSFS)

        subprocess.call(resamp, shell=True)

        os.remove(outImgSFStemp)
    else:
        os.rename(outImgSFStemp, outImgSFS)

    #for band in xrange(1, iInfo.bands):
    #
    #	for k in scs:
    #
    #		outImgMoms	= '%s/%s_f%d_k%d_moms%s' % (dName, fBase, band, k, fExt)
    #
    #		com			= 'otbcli_LocalStatisticExtraction -progress true -in %s -channel %d -ram %d -radius %d -out %s' % (outImgSFS, band, mem, k, outImgMoms)
    #
    #		if not os.path.isfile(outImgMoms):
    #
    #			subprocess.call(com, shell=True)


def sfs_feas(chunk, blk_size, cell_size, thresh_1):

    """
    Citation
    --------
    Zhang, Liangpei et al. 2006. "A Pixel Shape Index Coupled With Spectral Information for Classification of High
        Spatial Resolution Remotely Sensed Imagery." IEEE Transactions on Geoscience and Remote Sensing, V. 44, No. 10.

    Huang, Xin et al. 2007. "Classification and Extraction of Spatial Features in Urban Areas Using High-Resolution
        Multispectral Imagery." IEEE Transactions on Geoscience and Remote Sensing, V. 4, No. 2.

    Parameters
    ----------
    chunk -- 2d array
        : chunk array to extract features sets from
    blk_size -- int
        : block size of center pixels
    cell_size -- float
        : image cell size, in meters
    thresh_1 -- int
        : threshold for homogeneity

    Returns
    -------
    Directional lengths (length=8)
    PSI
    """

    if chunk.max() == 0:
        return [0., 0., 0., 0., 0.]
    else:

        # block adjustments
        blk_adjs = {1: 0, 2: 1, 4: 2}
        blk_adj = blk_adjs[blk_size]

        # get chunk size
        chunk_rws = chunk.shape[0]
        chunk_cls = chunk.shape[1]

        rows_half = chunk_rws / 2
        cols_half = chunk_cls / 2

        # get the center block average
        ctr_blk_mean = chunk[rows_half-1:rows_half+1, cols_half-1:cols_half+1].mean()

        # list for direction lengths
        dir_lengths = []

        x_pos = [np.add(np.array(range(chunk_cls-cols_half-blk_adj)), cols_half+1).astype(int), \
                    np.add(np.array(range(chunk_cls-cols_half-blk_adj)), cols_half+1).astype(int), \
                    np.array(range(cols_half-blk_adj, cols_half+blk_adj)).astype(int), \
                    np.array(range(chunk_cls-cols_half-blk_adj))[::-1].astype(int), \
                    np.array(range(chunk_cls-cols_half-blk_adj)).astype(int), \
                    np.array(range(chunk_cls-cols_half-blk_adj))[::-1].astype(int), \
                    np.array(range(cols_half-blk_adj, cols_half+blk_adj)).astype(int), \
                    np.add(np.array(range(chunk_cls-cols_half-blk_adj)), cols_half+1).astype(int)]

        y_pos = [np.array(range(rows_half-blk_adj, rows_half+blk_adj)).astype(int), \
                 np.array(range(chunk_rws-rows_half-blk_adj))[::-1].astype(int), \
                 np.array(range(chunk_rws-rows_half-blk_adj)).astype(int), \
                 np.array(range(chunk_rws-rows_half-blk_adj))[::-1].astype(int), \
                 np.array(range(rows_half-blk_adj, rows_half+blk_adj)).astype(int), \
                 np.add(np.array(range(chunk_rws-rows_half-blk_adj)), rows_half+1).astype(int), \
                 np.add(np.array(range(chunk_rws-rows_half-blk_adj)), rows_half+1).astype(int), \
                 np.add(np.array(range(chunk_rws-rows_half-blk_adj)), rows_half+1).astype(int)]

        dir_arrs = []

        ang_pos = 0
        dir_arrs.append(chunk[y_pos[ang_pos][0]:y_pos[ang_pos][-1]+1, x_pos[ang_pos][0]:x_pos[ang_pos][-1]+1].mean(axis=0).astype(np.float32))

        ang_pos = 1
        try:
            dir_arrs.append(chunk[y_pos[ang_pos], x_pos[ang_pos]].astype(np.float32))
        except:
            pos_min = np.minimum(len(y_pos[ang_pos]), len(x_pos[ang_pos]))
            dir_arrs.append(chunk[y_pos[ang_pos][:pos_min], x_pos[ang_pos][:pos_min]].astype(np.float32))

        ang_pos = 2
        dir_arrs.append(np.flipud(chunk[y_pos[ang_pos][0]:y_pos[ang_pos][-1]+1, x_pos[ang_pos][0]:x_pos[ang_pos][-1]+1]).mean(axis=1).astype(np.float32))

        ang_pos = 3
        try:
            dir_arrs.append(chunk[y_pos[ang_pos], x_pos[ang_pos]].astype(np.float32))
        except:
            pos_min = np.minimum(len(y_pos[ang_pos]), len(x_pos[ang_pos]))
            dir_arrs.append(chunk[y_pos[ang_pos][:pos_min], x_pos[ang_pos][:pos_min]].astype(np.float32))

        ang_pos = 4
        dir_arrs.append(np.fliplr(chunk[y_pos[ang_pos][0]:y_pos[ang_pos][-1]+1, x_pos[ang_pos][0]:x_pos[ang_pos][-1]+1]).mean(axis=0).astype(np.float32))

        ang_pos = 5
        try:
            dir_arrs.append(chunk[y_pos[ang_pos], x_pos[ang_pos]].astype(np.float32))
        except:
            pos_min = np.minimum(len(y_pos[ang_pos]), len(x_pos[ang_pos]))
            dir_arrs.append(chunk[y_pos[ang_pos][:pos_min], x_pos[ang_pos][:pos_min]].astype(np.float32))

        ang_pos = 6
        dir_arrs.append(chunk[y_pos[ang_pos][0]:y_pos[ang_pos][-1]+1, x_pos[ang_pos][0]:x_pos[ang_pos][-1]+1].mean(axis=1).astype(np.float32))

        ang_pos = 7
        try:
            dir_arrs.append(chunk[y_pos[ang_pos], x_pos[ang_pos]].astype(np.float32))
        except:
            pos_min = np.minimum(len(y_pos[ang_pos]), len(x_pos[ang_pos]))
            dir_arrs.append(chunk[y_pos[ang_pos][:pos_min], x_pos[ang_pos][:pos_min]].astype(np.float32))

        for dir_arr in dir_arrs:

            PH_i = 0

            for sur_ctr in xrange(1, len(dir_arr)+1):

                if PH_i >= thresh_1:
                    break
                else:

                    PH_i += np.abs(ctr_blk_mean - dir_arr[sur_ctr-1])

            dir_lengths.append(sur_ctr-1.)

        # row column directions
        # dir_lengths = []
        # for dir_idx in xrange(0, 8):
        #
        #     PH_i = 0
        #
        #     dir_iter = 1
        #     while PH_i < thresh_1:
        #
        #         if dir_idx == 0:
        #             # 0 degrees
        #             try:
        #                 sur = chunk[rows_half-blk_adj:rows_half+blk_adj, cols_half+dir_iter].mean()
        #             except:
        #                 break
        #         elif dir_idx == 1:
        #             # 45 degrees
        #             try:
        #                 sur = chunk[rows_half-blk_adj-dir_iter, cols_half+dir_iter]
        #             except:
        #                 break
        #         elif dir_idx == 2:
        #             # 90 degrees
        #             try:
        #                 sur = chunk[rows_half-blk_adj-dir_iter, cols_half-blk_adj:cols_half+blk_adj].mean()
        #             except:
        #                 break
        #         elif dir_idx == 3:
        #             # 135 degrees
        #             try:
        #                 sur = chunk[rows_half-blk_adj-dir_iter, cols_half-blk_adj-dir_iter]
        #             except:
        #                 break
        #         elif dir_idx == 4:
        #             # 180 degrees
        #             try:
        #                 sur = chunk[rows_half-blk_adj:rows_half+blk_adj, cols_half-blk_adj-dir_iter].mean()
        #             except:
        #                 break
        #             print chunk[rows_half-blk_adj:rows_half+blk_adj, cols_half-blk_adj-dir_iter]
        #         elif dir_idx == 5:
        #             # 225 degrees
        #             try:
        #                 sur = chunk[rows_half+dir_iter, cols_half-blk_adj-dir_iter]
        #             except:
        #                 break
        #         elif dir_idx == 6:
        #             # 270 degrees
        #             try:
        #                 sur = chunk[rows_half+dir_iter, cols_half-blk_adj:cols_half+blk_adj].mean()
        #             except:
        #                 break
        #         elif dir_idx == 7:
        #             # 315 degrees
        #             try:
        #                 sur = chunk[rows_half+dir_iter, cols_half+dir_iter]
        #             except:
        #                 break
        #
        #         # get the absolute difference in the surrounding pixel, at the current direction, and the center pixel
        #         PH_i += np.abs(ctr_blk_mean - sur)
        #
        #         dir_iter += 1
        #
        #     # get the length in the current direction
        #     dir_iter -= 1
        #
        #     dir_lengths.append(float(dir_iter))
        #
        # dir_lengths.append(np.sum(dir_lengths))

        sfs_max = np.max(dir_lengths)
        sfs_min = np.min(dir_lengths)
        sfs_psi, sfs_sd = cv2.meanStdDev(np.asarray(dir_lengths))
        # sfs_mag = ((sfs_psi**2.) + (ctr_blk_mean**2.)) ** .5

        return [sfs_max, sfs_min, sfs_psi, sfs_sd]#, sfs_mag]


def feaSFS(chBd, blk, scs, cell_size, thresh_1):

    rows, cols = chBd.shape
    scales_half = scs[-1] / 2
    scales_blk = scs[-1] - blk
    out_len = 0
    pix_ctr = 0

    for i in xrange(0, rows-scales_blk, blk):
        for j in xrange(0, cols-scales_blk, blk):
            for k in scs:
                out_len += 4

    # set the output list
    out_list = np.zeros(out_len).astype(np.float32)

    for i in xrange(0, rows-scales_blk, blk):
        for j in xrange(0, cols-scales_blk, blk):
            for k in scs:

                ch_bd = chBd[i+scales_half-(k/2):i+scales_half-(k/2)+k, j+scales_half-(k/2):j+scales_half-(k/2)+k]

                sts = sfs_feas(ch_bd, blk, cell_size, thresh_1)

                for st in sts:
                    out_list[pix_ctr] = st

                    pix_ctr += 1

    out_list[np.isnan(out_list) | np.isinf(out_list)] = 0.

    return out_list