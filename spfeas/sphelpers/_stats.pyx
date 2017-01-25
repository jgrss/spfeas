#!/usr/bin/env python

import cython
cimport cython
from cpython cimport array

import numpy as np
cimport numpy as np

from copy import copy

# from libc.stdlib cimport free
from libc.math cimport pow, atan, sqrt, sin, cos, round

from cython.parallel import parallel, prange
# from libc.math cimport isnan, isinf

from mpglue.stats import _lin_interp

# OpenCV
try:
    import cv2
except ImportError:
    raise ImportError('OpenCV did not load')

# Scikits-image
try:
    from skimage.feature import hog as HOG
    from skimage.feature import local_binary_pattern as LBP
    # from skimage.feature import greycomatrix, greycoprops
    from skimage.transform import probabilistic_hough_line as PHL
except:
    raise ImportError('Skimage.feature did not load')

# import pyximport
# pyximport.install(setup_args={'include_dirs': [np.get_include()]})

old_settings = np.seterr(all='ignore')

DTYPE_int = np.int
ctypedef np.int_t DTYPE_int_t

DTYPE_intp = np.intp
ctypedef np.intp_t DTYPE_intp_t

DTYPE_int64 = np.int64
ctypedef np.int64_t DTYPE_int64_t

DTYPE_uint8 = np.uint8
ctypedef np.uint8_t DTYPE_uint8_t

DTYPE_uint16 = np.uint16
ctypedef np.uint16_t DTYPE_uint16_t

DTYPE_uint32 = np.uint32
ctypedef np.uint32_t DTYPE_uint32_t

DTYPE_float32 = np.float32
ctypedef np.float32_t DTYPE_float32_t

DTYPE_float64 = np.float64
ctypedef np.float64_t DTYPE_float64_t

# cdef npceil = np.ceil

# cdef extern from 'numpy/npy_math.h':
#     DTYPE_float32_t npy_ceil(DTYPE_float32_t x)

cdef extern from 'numpy/npy_math.h':
    bint npy_isnan(DTYPE_float32_t x)

cdef extern from 'numpy/npy_math.h':
    bint npy_isinf(DTYPE_float32_t x)

cdef extern from 'math.h':
    DTYPE_float32_t ceil(DTYPE_float32_t x)


@cython.profile(False)
cdef inline int n_rows_cols(int pixel_index, int rows_cols, int block_size):
    return rows_cols if pixel_index + rows_cols < block_size else block_size - pixel_index


@cython.profile(False)
cdef inline int _get_min_sample_i(int s1, int s2):
    return s2 if s2 < s1 else s1


@cython.profile(False)
cdef inline DTYPE_float32_t _get_min_sample_f(DTYPE_float32_t s1, DTYPE_float32_t s2):
    return s2 if s2 < s1 else s1


@cython.profile(False)
cdef inline DTYPE_uint8_t _get_min_sample_int(DTYPE_uint8_t s1, DTYPE_uint8_t s2):
    return s2 if s2 < s1 else s1


@cython.profile(False)
cdef inline DTYPE_float32_t _get_max_sample(DTYPE_float32_t s1, DTYPE_float32_t s2):
    return s2 if s2 > s1 else s1


@cython.profile(False)
cdef inline DTYPE_uint8_t _get_max_sample_int(DTYPE_uint8_t s1, DTYPE_uint8_t s2):
    return s2 if s2 > s1 else s1


@cython.profile(False)
cdef inline DTYPE_float32_t _euclidean_distance(DTYPE_float32_t x1, DTYPE_float32_t y1, DTYPE_float32_t x2, DTYPE_float32_t y2):
    return (((x1 - x2)**2.) + ((y1 - y2)**2.))**.5


@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_float32_t _get_min_f(DTYPE_float32_t[:] in_row, int cols):

    cdef:
        Py_ssize_t a
        DTYPE_float32_t m = 100000000.

    for a in xrange(0, cols):
    # for a in prange(0, cols, nogil=True, num_threads=cols, schedule='static'):
        m = _get_min_sample_f(m, in_row[a])

    return m


@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_uint8_t _get_min(DTYPE_uint8_t[:, :] block, int rs, int cs):

    cdef:
        Py_ssize_t bi, bj
        DTYPE_uint8_t m = 255

    for bi in xrange(0, rs):
        for bj in xrange(0, cs):

            m = _get_min_sample_int(m, block[bi, bj])

    return m


@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_float32_t _get_max_f2d(DTYPE_float32_t[:, :] block, int rs, int cs):

    cdef:
        Py_ssize_t bi, bj
        DTYPE_float32_t m = -9999999.

    for bi in xrange(0, rs):
        for bj in xrange(0, cs):

            m = _get_max_sample(m, block[bi, bj])

    return m


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _get_max(DTYPE_uint8_t[:, :] block, Py_ssize_t rs, Py_ssize_t cs):

    cdef:
        Py_ssize_t bi, bj
        int m = -255

    for bi in xrange(0, rs):
        for bj in xrange(0, cs):

            m = _get_max_sample_int(m, block[bi, bj])

    return m


@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_float32_t _get_max_f(DTYPE_float32_t[:] in_row, int cols):

    cdef:
        Py_ssize_t a
        DTYPE_float32_t m = in_row[0]

    for a in xrange(1, cols):
        m = _get_max_sample(m, in_row[a])

    return m


@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_float32_t get_sum_1d(DTYPE_float32_t[:] block):

    """
    Calculate the sum of a 1d array
    """

    cdef:
        Py_ssize_t i
        int samps = block.shape[0]
        DTYPE_float32_t the_sum = block[0]

    # for i in prange(1, samps, nogil=True, num_threads=samps, schedule='static'):
    for i in xrange(1, samps):
        the_sum += block[i]

    return the_sum


# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
# cdef DTYPE_float32_t get_mean_1d(np.ndarray[DTYPE_float32_t, ndim=1] block):
#
#     """
#     Calculate the mean of a 1d array
#     """
#
#     cdef int samps = block.shape[0]
#
#     return get_sum_1d(block) / samps


# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef DTYPE_float32_t get_sum_2d(np.ndarray[DTYPE_float32_t, ndim=2] block):
#
#     """
#     Calculate the sum of a 2d array
#     """
#
#     cdef:
#         Py_ssize_t i
#         int rows = block.shape[0]
#         int cols = block.shape[1]
#         int samps = rows * cols
#         np.ndarray[DTYPE_float32_t, ndim=1] block_r = block.ravel()
#         DTYPE_float32_t the_sum = block_r[0]
#
#     with nogil:
#         for i in prange(1, samps):
#             the_sum += block_r[i]
#
#     return the_sum


@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_float64_t _get_sum64(DTYPE_float64_t[:, :] block, int rs, int cs):

    cdef:
        Py_ssize_t bi, bj
        DTYPE_float64_t block_sum = 0.

    # with nogil, parallel(num_threads=rs):

    for bi in xrange(0, rs):

        for bj in xrange(0, cs):
            block_sum += block[bi, bj]

    return block_sum


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef DTYPE_float64_t _get_mean64(DTYPE_float64_t[:, :] block, int rs, int cs):

    cdef:
        DTYPE_float64_t n_samps = float(rs*cs)

    return _get_sum64(block, rs, cs) / n_samps


@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_float32_t _get_sum_uint8(DTYPE_uint8_t[:, :] block, int rs, int cs):

    cdef:
        Py_ssize_t bi, bj
        DTYPE_float32_t block_sum = 0.

    for bi in xrange(0, rs):

        for bj in xrange(0, cs):

            block_sum += float(block[bi, bj])

    return block_sum


@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_float32_t _get_sum(DTYPE_float32_t[:, :] block, int rs, int cs):

    cdef:
        Py_ssize_t bi, bj
        DTYPE_float32_t block_sum = 0.

    # with nogil, parallel(num_threads=rs):

    for bi in xrange(0, rs):

        for bj in xrange(0, cs):

            block_sum += block[bi, bj]

    return block_sum


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef DTYPE_float32_t _get_mean(DTYPE_float32_t[:, :] block, int rs, int cs):

    cdef:
        DTYPE_float32_t n_samps = float(rs*cs)

    return _get_sum(block, rs, cs) / n_samps


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef DTYPE_float32_t _get_weighted_sum(DTYPE_float32_t[:, :] block, DTYPE_float32_t[:, :] weights, int rs, int cs):

    cdef:
        Py_ssize_t bi, bj
        DTYPE_float32_t block_sum = 0.
        DTYPE_float32_t dv

    for bi in xrange(0, rs):
        for bj in xrange(0, cs):

            dv = block[bi, bj] / weights[bi, bj]

            if not npy_isnan(dv) and not npy_isinf(dv):
                block_sum += dv

    return block_sum


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef DTYPE_float32_t _get_weighted_mean(DTYPE_float32_t[:, :] block, DTYPE_float32_t[:, :] weights, int rs, int cs):

    cdef:
        DTYPE_float32_t n_samps = float(rs*cs)

    return _get_weighted_sum(block, weights, rs, cs) / n_samps


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef DTYPE_float32_t _get_mean_uint8(DTYPE_uint8_t[:, :] block, int rs, int cs):

    cdef:
        DTYPE_float32_t n_samps = float(rs*cs)

    return _get_sum_uint8(block, rs, cs) / n_samps


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef DTYPE_float32_t _get_var(DTYPE_float32_t[:, :] block, int rs, int cs):

    cdef:
        Py_ssize_t bi, bj
        DTYPE_float32_t mu = _get_mean(block, rs, cs)
        DTYPE_float32_t block_var = 0.

    for bi in xrange(0, rs):
        for bj in xrange(0, cs):
            block_var += pow(float(block[bi, bj]) - mu, 2)

    return block_var / (rs*cs)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef DTYPE_float32_t _get_var_uint8(DTYPE_uint8_t[:, :] block, int rs, int cs):

    cdef:
        Py_ssize_t bi, bj
        DTYPE_float32_t mu = _get_mean_uint8(block, rs, cs)
        DTYPE_float32_t block_var = 0.

    for bi in xrange(0, rs):
        for bj in xrange(0, cs):
            block_var += pow(float(block[bi, bj]) - mu, 2)

    return block_var / (rs*cs)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_float32_t _get_std1d(DTYPE_float32_t[:] block, int cs, DTYPE_float32_t psi):

    cdef:
        Py_ssize_t bj
        DTYPE_float32_t[:] pow_array = np.zeros(cs, dtype='float32')

    # for bj in prange(0, cs, nogil=True, num_threads=cs, schedule='static'):
    for bj in xrange(0, cs):
        pow_array[bj] = pow((block[bj] - psi), 2)

    return pow(_get_sum1d(pow_array, cs), .5)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_float32_t _get_sum1d(DTYPE_float32_t[:] block, int cs):

    cdef:
        Py_ssize_t bj
        DTYPE_float32_t block_sum = block[0]

    # for bj in prange(0, cs, nogil=True, num_threads=cs, schedule='static'):
    for bj in xrange(1, cs):
        block_sum += block[bj]

    return block_sum


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef DTYPE_float32_t _get_mean1d(DTYPE_float32_t[:] block, int cs):

    return _get_sum1d(block, cs) / cs


# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
# cdef tuple get_stats_2d(np.ndarray[DTYPE_float32_t, ndim=2] block):
#
#     """
#     Calculate the mean and variance of a 2d array
#     """
#
#     cdef DTYPE_float32_t the_mean = get_mean_2d(block)
#     cdef int i, j
#     cdef int rows = block.shape[0]
#     cdef int cols = block.shape[1]
#     cdef int samps = rows * cols
#     cdef DTYPE_float32_t curr_val
#     cdef DTYPE_float32_t val_mean
#     cdef DTYPE_float32_t the_var = 0.
#     np.ndarray[DTYPE_float32_t, ndim=1] block_ravels = block.ravel()
#
#     with nogil, parallel():
#         for i in prange(1, samps):
#
#             curr_val = block_ravels[i]
#
#             val_mean = curr_val - the_mean
#
#             the_var += val_mean * val_mean
#
#         # for i in prange(0, rows):
#         #     for j in prange(0, cols):
#         #
#         #         curr_val = block[i, j]
#         #
#         #         val_mean = curr_val - the_mean
#         #
#         #         the_var += val_mean * val_mean
#
#     the_var /= samps
#
#     return the_mean, the_var


# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
# cdef DTYPE_float64_t get_mean(np.ndarray[DTYPE_float64_t, ndim=1] block):
#
#     """
#     Calculate the mean of a 1d array
#     """
#
#     cdef int idx
#     cdef int samps = len(block)
#     cdef DTYPE_float64_t the_sum = block[0]
#
#     for idx in xrange(1, samps):
#         the_sum += block[idx]
#
#     return the_sum / samps


@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple draw_line(Py_ssize_t y0, Py_ssize_t x0, Py_ssize_t y1, Py_ssize_t x1):

    """
    Graciously adapated from the Scikit-image team

    Generate line pixel coordinates.

    Parameters
    ----------
    y0, x0 : int
        Starting position (row, column).
    y1, x1 : int
        End position (row, column).

    Returns
    -------
    rr, cc : (N,) ndarray of int
        Indices of pixels that belong to the line.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.

    See Also
    --------
    line_aa : Anti-aliased line generator

    Examples
    --------
    >>> from skimage.draw import line
    >>> img = np.zeros((10, 10), dtype=np.uint8)
    >>> rr, cc = line(1, 1, 8, 8)
    >>> img[rr, cc] = 1
    >>> img
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    """

    cdef:
        char steep = 0
        Py_ssize_t x = x0
        Py_ssize_t y = y0
        Py_ssize_t dx = int(abs(float(x1) - float(x0)))
        Py_ssize_t dy = int(abs(float(y1) - float(y0)))
        Py_ssize_t sx, sy, d, i
        DTYPE_intp_t[:] rr, cc

    if (x1 - x) > 0:
        sx = 1
    else:
        sx = -1

    if (y1 - y) > 0:
        sy = 1
    else:
        sy = -1

    if dy > dx:

        steep = 1
        x, y = y, x
        dx, dy = dy, dx
        sx, sy = sy, sx

    d = (2 * dy) - dx

    rr = np.zeros(int(dx)+1, dtype='intp')
    cc = rr.copy()
    # rr = clone(template, int(dx)+1, True)
    # cc = clone(template, int(dx)+1, True)

    for i in xrange(0, dx):

        if steep:
            rr[i] = x
            cc[i] = y
        else:
            rr[i] = y
            cc[i] = x

        while d >= 0:

            y += sy
            d -= 2 * dx

        x += sx
        d += 2 * dy

    rr[dx] = y1
    cc[dx] = x1

    return rr, cc


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef DTYPE_float32_t[:] _get_stats(DTYPE_float32_t[:] block, int samps):

    """
    Calculate the mean, variance, skew, and kurtosis of a 1d array
    """

    cdef:
        DTYPE_float32_t the_mean = _get_mean1d(block, samps)    # cv2.mean(np.array(block).astype(np.float32))[0]
        Py_ssize_t idx
        DTYPE_float32_t val_mean = block[0] - the_mean
        DTYPE_float32_t the_var = val_mean * val_mean
        DTYPE_float32_t the_skew = val_mean * val_mean * val_mean
        DTYPE_float32_t the_kurtosis = val_mean * val_mean * val_mean * val_mean
        DTYPE_float32_t[:] output = np.empty(samps, dtype='float32')

    for idx in xrange(1, samps):

        val_mean = block[idx] - the_mean

        the_var += val_mean * val_mean
        the_skew += val_mean * val_mean * val_mean
        the_kurtosis += val_mean * val_mean * val_mean * val_mean

    the_var /= samps
    the_skew /= pow(the_var, 1.5)
    the_kurtosis /= pow(the_var, 2.)

    output[0] = the_mean
    output[1] = the_var
    output[2] = the_skew
    output[3] = the_kurtosis

    return output


@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_float32_t[:] get_moments(DTYPE_float32_t[:] img_arr):

    """
    Get the moments for 1d array
    """

    cdef:
        int img_arr_cols = img_arr.shape[0]
        DTYPE_float32_t[:] empty_array = np.zeros(img_arr_cols, dtype='float32')

    if _get_max_f(img_arr, img_arr_cols) == 0:
        return empty_array
    else:
        return _get_stats(img_arr, img_arr_cols)


# Gabor filter bank

# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef DTYPE_float32_t[:] gabor_mean_var(np.ndarray[unsigned char, ndim=2] gabor_array):
#
#     """
#     Get mean and variance of Gabor kernel convolution
#     """
#
#     cdef int img_arr_cols = gabor_array.shape[0]
#
#     if gabor_array.max() == 0:
#         return np.array([np.array([[0.]]), np.array([[0.]])], dtype='float32')
#     else:
#         return np.array(cv2.meanStdDev(gabor_array), dtype='float32')


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[DTYPE_float32_t, ndim=1] _feature_gabor(DTYPE_uint8_t[:, :] chBd, int blk,
                                                        DTYPE_uint16_t[:] scs, int out_len, int scales_half,
                                                        int scales_block, list kernels, int rows, int cols,
                                                        int scale_length):

    """
    Returns at each scale at each kernel

    1:	Mean
    2:	Variance
    """

    cdef:
        Py_ssize_t i, j, ki, kl
        DTYPE_uint16_t k, k_half
        DTYPE_uint8_t[:, :] ch_bd
        DTYPE_uint8_t[:, :] ch_bd_gabor
        DTYPE_float32_t[:] sts
        list st
        DTYPE_float32_t[:] out_list = np.zeros(out_len, dtype='float32')
        int pix_ctr = 0
        int n_kernels = np.asarray(kernels).shape[0]
        int bcr, bcc

    for i from 0 <= i < rows-scales_block by blk:
        for j from 0 <= j < cols-scales_block by blk:
            for ki in xrange(0, scale_length):

                k = scs[ki]

                k_half = k / 2

                ch_bd = chBd[i+scales_half-k_half:i+scales_half-k_half+k,
                             j+scales_half-k_half:j+scales_half-k_half+k]

                bcr = ch_bd.shape[0]
                bcc = ch_bd.shape[1]

                for kl in xrange(0, n_kernels):

                    # sts = gabor_mean_var(cv2.filter2D(np.array(ch_bd), -1, kernels[kl]))

                    ch_bd_gabor = cv2.filter2D(np.uint8(ch_bd), -1, kernels[kl])

                    out_list[pix_ctr] = _get_mean_uint8(ch_bd_gabor, bcr, bcc)
                    pix_ctr += 1
                    out_list[pix_ctr] = _get_var_uint8(ch_bd_gabor, bcr, bcc)
                    pix_ctr += 1

    return np.float32(out_list)


@cython.boundscheck(False)
@cython.wraparound(False)
def feature_gabor(np.ndarray[DTYPE_uint8_t, ndim=2] chBd, int blk, list scs, int end_scale, list kernels):

    cdef:
        Py_ssize_t i, j, ki, kl
        unsigned int scales_half = end_scale / 2
        unsigned int scales_block = end_scale - blk
        int out_len = 0
        int rows = chBd.shape[0]
        int cols = chBd.shape[1]
        DTYPE_uint16_t[:] scales_array = np.array(scs, dtype='uint16')
        int scale_length = scales_array.shape[0]
        int n_kernels = np.asarray(kernels).shape[0]

    for i from 0 <= i < rows-scales_block by blk:
        for j from 0 <= j < cols-scales_block by blk:
            for ki in xrange(0, scale_length):
                for kl in xrange(0, n_kernels):
                    out_len += 2

    return _feature_gabor(chBd, blk, scales_array, out_len, scales_half,
                          scales_block, kernels, rows, cols, scale_length)


# Histogram of Oriented Gradients


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[DTYPE_float32_t, ndim=1] calc_hog(np.ndarray[DTYPE_float32_t, ndim=2] mag_chunk,
                                                  np.ndarray[DTYPE_float32_t, ndim=2] ang_chunk,
                                                  DTYPE_float32_t pi2, int bin_n):

    # quantizing binvalues
    cdef np.ndarray[DTYPE_uint16_t, ndim=2] bins = (bin_n * ang_chunk / pi2).astype(np.uint16)

    return np.float32(np.bincount(np.array(bins).ravel(), weights=mag_chunk.ravel(), minlength=bin_n))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[DTYPE_float32_t, ndim=1] _feature_hog(DTYPE_float32_t[:, :] grad,
                                                      DTYPE_float32_t[:, :] ori,
                                                      int blk, DTYPE_uint16_t[:] scs, int end_scale, int scales_half,
                                                      int scales_block, int out_len, int rows, int cols,
                                                      int scale_length):

    """
    Computes the Histogram of Oriented Gradients

    At each scale, returns:
        1:	Mean
        2:	Variance
        3:	Skew
        4:	Kurtosis
    """

    cdef:
        Py_ssize_t i, j, ki, sti
        DTYPE_uint16_t k, k_half
        DTYPE_float32_t[:, :] ch_grad, ch_ori
        DTYPE_float32_t[:] sts
        DTYPE_float32_t[:] out_list = np.zeros(out_len, dtype='float32')
        int pix_ctr = 0
        DTYPE_float32_t pi2 = 3.14159 / 2.
        bin_n = 9

    for i from 0 <= i < rows-scales_block by blk:
        for j from 0 <= j < cols-scales_block by blk:
            for ki in xrange(0, scale_length):

                k = scs[ki]

                k_half = k / 2

                ch_grad = grad[i+scales_half-k_half:i+scales_half-k_half+k,
                               j+scales_half-k_half:j+scales_half-k_half+k]

                ch_ori = ori[i+scales_half-k_half:i+scales_half-k_half+k,
                             j+scales_half-k_half:j+scales_half-k_half+k]

                sts = get_moments(calc_hog(np.array(ch_grad), np.array(ch_ori), pi2, bin_n))

                for sti in xrange(0, 4):

                    out_list[pix_ctr] = sts[sti]

                    pix_ctr += 1

    return np.float32(out_list)


@cython.boundscheck(False)
@cython.wraparound(False)
def feature_hog(np.ndarray[DTYPE_float32_t, ndim=2] grad,
                np.ndarray[DTYPE_float32_t, ndim=2] ori,
                int blk, list scs, int end_scale):

    cdef:
        Py_ssize_t i, j, ki
        unsigned int scales_half = end_scale / 2
        unsigned int scales_block = end_scale - blk
        int out_len = 0
        int rows = grad.shape[0]
        int cols = grad.shape[1]
        DTYPE_uint16_t[:] scales_array = np.array(scs, dtype='uint16')
        int scale_length = scales_array.shape[0]

    for i from 0 <= i < rows-scales_block by blk:
        for j from 0 <= j < cols-scales_block by blk:
            for ki in xrange(0, scale_length):
                out_len += 4

    return _feature_hog(grad, ori, blk, scales_array, end_scale,
                        scales_half, scales_block, out_len, rows, cols, scale_length)


# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef np.ndarray[long, ndim=1] obscure_angle_indices(np.ndarray[long, ndim=1] arr, str do_m, fl):
#
#     cdef int arr_len = len(arr)
#     cdef int ctr = 2
#     cdef int s = 0
#     cdef int t
#     cdef np.ndarray[DTYPE_uint8_t, ndim=1] l = np.zeros(arr_len).astype(np.uint8)
#
#     for t in xrange(0, arr_len):
#
#         l[t] = s
#         if ctr == 2:
#             s += 1
#             ctr = 0
#
#         ctr += 1
#
#     if do_m == 'add':
#         if fl:
#             return np.add(arr, l)[::-1]
#         else:
#             return np.add(arr, l)
#     else:
#         if fl:
#             return np.subtract(arr, l)[::-1]
#         else:
#             return np.subtract(arr, l)


# @cython.boundscheck(False)
# cdef list get_directions(np.ndarray[DTYPE_uint8_t, ndim=2] chunk, int chunk_rws, int chunk_cls, int rows_half,
#                          int cols_half, int blk_adj, int n_angles=8):
#
#     cdef:
#         list dir_arrs = []
#         int ang_pos, pos_min
#         list x_pos, y_pos
#
#     if n_angles == 16:
#
#         x_pos = [np.add(np.array(range(chunk_cls-cols_half-blk_adj)), cols_half+1).astype(int),
#                  np.add(np.array(range(chunk_cls-cols_half-blk_adj)), cols_half+1).astype(int),
#                  np.add(np.array(range(chunk_cls-cols_half-blk_adj)), cols_half+1).astype(int),
#                  obscure_angle_indices(np.add(np.array(range(chunk_cls-cols_half-blk_adj)), cols_half+1).astype(int), 'sub', False),
#                  np.array(range(cols_half-blk_adj, cols_half+blk_adj)).astype(int),
#                  obscure_angle_indices(np.array(range(chunk_cls-cols_half-blk_adj)).astype(int), 'add', True),
#                  np.array(range(chunk_cls-cols_half-blk_adj))[::-1].astype(int),
#                  np.array(range(chunk_cls-cols_half-blk_adj))[::-1].astype(int),
#                  np.array(range(chunk_cls-cols_half-blk_adj))[::-1].astype(int),
#                  np.array(range(chunk_cls-cols_half-blk_adj))[::-1].astype(int),
#                  np.array(range(chunk_cls-cols_half-blk_adj))[::-1].astype(int),
#                  obscure_angle_indices(np.array(range(chunk_cls-cols_half-blk_adj)).astype(int), 'add', True),
#                  np.array(range(chunk_cls-cols_half-blk_adj))[::-1].astype(int),
#                  obscure_angle_indices(np.add(np.array(range(chunk_cls-cols_half-blk_adj)), cols_half+1).astype(int), 'sub', False),
#                  np.array(range(cols_half-blk_adj, cols_half+blk_adj)).astype(int),
#                  np.add(np.array(range(chunk_cls-cols_half-blk_adj)), cols_half+1).astype(int),
#                  np.add(np.array(range(chunk_cls-cols_half-blk_adj)), cols_half+1).astype(int)]
#
#         y_pos = [np.array(range(rows_half-blk_adj, rows_half+blk_adj)).astype(int),
#                     obscure_angle_indices(np.array(range(chunk_rws-rows_half-blk_adj))[::-1].astype(int), 'add', False),
#                     np.array(range(chunk_rws-rows_half-blk_adj))[::-1].astype(int),
#                     np.array(range(chunk_rws-rows_half-blk_adj))[::-1].astype(int),
#                     np.array(range(chunk_rws-rows_half-blk_adj))[::-1].astype(int),
#                     np.array(range(chunk_rws-rows_half-blk_adj))[::-1].astype(int),
#                     np.array(range(chunk_rws-rows_half-blk_adj))[::-1].astype(int),
#                     obscure_angle_indices(np.array(range(chunk_rws-rows_half-blk_adj))[::-1].astype(int), 'add', False),
#                     np.array(range(rows_half-blk_adj, rows_half+blk_adj)).astype(int),
#                     obscure_angle_indices(np.add(np.array(range(chunk_rws-rows_half-blk_adj)), rows_half+1).astype(int), 'sub', False),
#                     np.add(np.array(range(chunk_rws-rows_half-blk_adj)), rows_half+1).astype(int),
#                     np.add(np.array(range(chunk_rws-rows_half-blk_adj)), rows_half+1).astype(int),
#                     np.add(np.array(range(chunk_rws-rows_half-blk_adj)), rows_half+1).astype(int),
#                     np.add(np.array(range(chunk_rws-rows_half-blk_adj)), rows_half+1).astype(int),
#                     np.add(np.array(range(chunk_rws-rows_half-blk_adj)), rows_half+1).astype(int),
#                     obscure_angle_indices(np.add(np.array(range(chunk_rws-rows_half-blk_adj)), rows_half+1).astype(int), 'sub', False)]
#
#         ang_pos = 0
#         dir_arrs.append(chunk[y_pos[ang_pos][0]:y_pos[ang_pos][-1]+1, x_pos[ang_pos][0]:x_pos[ang_pos][-1]+1].mean(axis=0).astype(np.float32))
#
#         ang_pos = 1
#         try:
#             dir_arrs.append(chunk[y_pos[ang_pos], x_pos[ang_pos]].astype(np.float32))
#         except:
#             pos_min = np.minimum(len(y_pos[ang_pos]), len(x_pos[ang_pos]))
#             dir_arrs.append(chunk[y_pos[ang_pos][:pos_min], x_pos[ang_pos][:pos_min]].astype(np.float32))
#
#         ang_pos = 2
#         try:
#             dir_arrs.append(chunk[y_pos[ang_pos], x_pos[ang_pos]].astype(np.float32))
#         except:
#             pos_min = np.minimum(len(y_pos[ang_pos]), len(x_pos[ang_pos]))
#             dir_arrs.append(chunk[y_pos[ang_pos][:pos_min], x_pos[ang_pos][:pos_min]].astype(np.float32))
#
#         ang_pos = 3
#         try:
#             dir_arrs.append(chunk[y_pos[ang_pos], x_pos[ang_pos]].astype(np.float32))
#         except:
#             pos_min = np.minimum(len(y_pos[ang_pos]), len(x_pos[ang_pos]))
#             dir_arrs.append(chunk[y_pos[ang_pos][:pos_min], x_pos[ang_pos][:pos_min]].astype(np.float32))
#
#         ang_pos = 4
#         dir_arrs.append(np.flipud(chunk[y_pos[ang_pos][0]:y_pos[ang_pos][-1]+1, x_pos[ang_pos][0]:x_pos[ang_pos][-1]+1]).mean(axis=1).astype(np.float32))
#
#         ang_pos = 5
#         try:
#             dir_arrs.append(chunk[y_pos[ang_pos], x_pos[ang_pos]].astype(np.float32))
#         except:
#             pos_min = np.minimum(len(y_pos[ang_pos]), len(x_pos[ang_pos]))
#             dir_arrs.append(chunk[y_pos[ang_pos][:pos_min], x_pos[ang_pos][:pos_min]].astype(np.float32))
#
#         ang_pos = 6
#         try:
#             dir_arrs.append(chunk[y_pos[ang_pos], x_pos[ang_pos]].astype(np.float32))
#         except:
#             pos_min = np.minimum(len(y_pos[ang_pos]), len(x_pos[ang_pos]))
#             dir_arrs.append(chunk[y_pos[ang_pos][:pos_min], x_pos[ang_pos][:pos_min]].astype(np.float32))
#
#         ang_pos = 7
#         try:
#             dir_arrs.append(chunk[y_pos[ang_pos], x_pos[ang_pos]].astype(np.float32))
#         except:
#             pos_min = np.minimum(len(y_pos[ang_pos]), len(x_pos[ang_pos]))
#             dir_arrs.append(chunk[y_pos[ang_pos][:pos_min], x_pos[ang_pos][:pos_min]].astype(np.float32))
#
#         ang_pos = 8
#         dir_arrs.append(np.fliplr(chunk[y_pos[ang_pos][0]:y_pos[ang_pos][-1]+1, x_pos[ang_pos][0]:x_pos[ang_pos][-1]+1]).mean(axis=0).astype(np.float32))
#
#         ang_pos = 9
#         try:
#             dir_arrs.append(chunk[y_pos[ang_pos], x_pos[ang_pos]].astype(np.float32))
#         except:
#             pos_min = np.minimum(len(y_pos[ang_pos]), len(x_pos[ang_pos]))
#             dir_arrs.append(chunk[y_pos[ang_pos][:pos_min], x_pos[ang_pos][:pos_min]].astype(np.float32))
#
#         ang_pos = 10
#         try:
#             dir_arrs.append(chunk[y_pos[ang_pos], x_pos[ang_pos]].astype(np.float32))
#         except:
#             pos_min = np.minimum(len(y_pos[ang_pos]), len(x_pos[ang_pos]))
#             dir_arrs.append(chunk[y_pos[ang_pos][:pos_min], x_pos[ang_pos][:pos_min]].astype(np.float32))
#
#         ang_pos = 11
#         try:
#             dir_arrs.append(chunk[y_pos[ang_pos], x_pos[ang_pos]].astype(np.float32))
#         except:
#             pos_min = np.minimum(len(y_pos[ang_pos]), len(x_pos[ang_pos]))
#             dir_arrs.append(chunk[y_pos[ang_pos][:pos_min], x_pos[ang_pos][:pos_min]].astype(np.float32))
#
#         ang_pos = 12
#         dir_arrs.append(chunk[y_pos[ang_pos][0]:y_pos[ang_pos][-1]+1, x_pos[ang_pos][0]:x_pos[ang_pos][-1]+1].mean(axis=1).astype(np.float32))
#
#         ang_pos = 13
#         try:
#             dir_arrs.append(chunk[y_pos[ang_pos], x_pos[ang_pos]].astype(np.float32))
#         except:
#             pos_min = np.minimum(len(y_pos[ang_pos]), len(x_pos[ang_pos]))
#             dir_arrs.append(chunk[y_pos[ang_pos][:pos_min], x_pos[ang_pos][:pos_min]].astype(np.float32))
#
#         ang_pos = 14
#         try:
#             dir_arrs.append(chunk[y_pos[ang_pos], x_pos[ang_pos]].astype(np.float32))
#         except:
#             pos_min = np.minimum(len(y_pos[ang_pos]), len(x_pos[ang_pos]))
#             dir_arrs.append(chunk[y_pos[ang_pos][:pos_min], x_pos[ang_pos][:pos_min]].astype(np.float32))
#
#         ang_pos = 15
#         try:
#             dir_arrs.append(chunk[y_pos[ang_pos], x_pos[ang_pos]].astype(np.float32))
#         except:
#             pos_min = np.minimum(len(y_pos[ang_pos]), len(x_pos[ang_pos]))
#             dir_arrs.append(chunk[y_pos[ang_pos][:pos_min], x_pos[ang_pos][:pos_min]].astype(np.float32))
#
#     else:
#
#         x_pos = [np.add(np.array(range(chunk_cls-cols_half-blk_adj)), cols_half+1).astype(int),
#                     np.add(np.array(range(chunk_cls-cols_half-blk_adj)), cols_half+1).astype(int),
#                     np.array(range(cols_half-blk_adj, cols_half+blk_adj)).astype(int),
#                     np.array(range(chunk_cls-cols_half-blk_adj))[::-1].astype(int),
#                     np.array(range(chunk_cls-cols_half-blk_adj))[::-1].astype(int),
#                     np.array(range(chunk_cls-cols_half-blk_adj))[::-1].astype(int),
#                     np.array(range(cols_half-blk_adj, cols_half+blk_adj)).astype(int),
#                     np.add(np.array(range(chunk_cls-cols_half-blk_adj)), cols_half+1).astype(int)]
#
#         y_pos = [np.array(range(rows_half-blk_adj, rows_half+blk_adj)).astype(int),
#                  np.array(range(chunk_rws-rows_half-blk_adj))[::-1].astype(int),
#                  np.array(range(chunk_rws-rows_half-blk_adj))[::-1].astype(int),
#                  np.array(range(chunk_rws-rows_half-blk_adj))[::-1].astype(int),
#                  np.array(range(rows_half-blk_adj, rows_half+blk_adj)).astype(int),
#                  np.add(np.array(range(chunk_rws-rows_half-blk_adj)), rows_half+1).astype(int),
#                  np.add(np.array(range(chunk_rws-rows_half-blk_adj)), rows_half+1).astype(int),
#                  np.add(np.array(range(chunk_rws-rows_half-blk_adj)), rows_half+1).astype(int)]
#
#         ang_pos = 0
#         dir_arrs.append(chunk[y_pos[ang_pos][0]:y_pos[ang_pos][-1]+1, x_pos[ang_pos][0]:x_pos[ang_pos][-1]+1].mean(axis=0).astype(np.float32))
#
#         ang_pos = 1
#         try:
#             dir_arrs.append(chunk[y_pos[ang_pos], x_pos[ang_pos]].astype(np.float32))
#         except:
#             pos_min = np.minimum(len(y_pos[ang_pos]), len(x_pos[ang_pos]))
#             dir_arrs.append(chunk[y_pos[ang_pos][:pos_min], x_pos[ang_pos][:pos_min]].astype(np.float32))
#
#         ang_pos = 2
#         dir_arrs.append(np.flipud(chunk[y_pos[ang_pos][0]:y_pos[ang_pos][-1]+1, x_pos[ang_pos][0]:x_pos[ang_pos][-1]+1]).mean(axis=1).astype(np.float32))
#
#         ang_pos = 3
#         try:
#             dir_arrs.append(chunk[y_pos[ang_pos], x_pos[ang_pos]].astype(np.float32))
#         except:
#             pos_min = np.minimum(len(y_pos[ang_pos]), len(x_pos[ang_pos]))
#             dir_arrs.append(chunk[y_pos[ang_pos][:pos_min], x_pos[ang_pos][:pos_min]].astype(np.float32))
#
#         ang_pos = 4
#         dir_arrs.append(np.fliplr(chunk[y_pos[ang_pos][0]:y_pos[ang_pos][-1]+1, x_pos[ang_pos][0]:x_pos[ang_pos][-1]+1]).mean(axis=0).astype(np.float32))
#
#         ang_pos = 5
#         try:
#             dir_arrs.append(chunk[y_pos[ang_pos], x_pos[ang_pos]].astype(np.float32))
#         except:
#             pos_min = np.minimum(len(y_pos[ang_pos]), len(x_pos[ang_pos]))
#             dir_arrs.append(chunk[y_pos[ang_pos][:pos_min], x_pos[ang_pos][:pos_min]].astype(np.float32))
#
#         ang_pos = 6
#         dir_arrs.append(chunk[y_pos[ang_pos][0]:y_pos[ang_pos][-1]+1, x_pos[ang_pos][0]:x_pos[ang_pos][-1]+1].mean(axis=1).astype(np.float32))
#
#         ang_pos = 7
#         try:
#             dir_arrs.append(chunk[y_pos[ang_pos], x_pos[ang_pos]].astype(np.float32))
#         except:
#             pos_min = np.minimum(len(y_pos[ang_pos]), len(x_pos[ang_pos]))
#             dir_arrs.append(chunk[y_pos[ang_pos][:pos_min], x_pos[ang_pos][:pos_min]].astype(np.float32))
#
#     return dir_arrs


@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_uint8_t[:] extract_values(DTYPE_uint8_t[:, :] block, DTYPE_intp_t[:] rr_, DTYPE_intp_t[:] cc_, int fl):

    cdef:
        Py_ssize_t fi, fi_, fj_
        DTYPE_uint8_t[:] values = np.zeros(fl, dtype='uint8')

    for fi in xrange(0, fl):

        fi_ = rr_[fi]
        fj_ = cc_[fi]

        values[fi] = block[fi_, fj_]

    return values


@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_float32_t[:] _get_directions(DTYPE_uint8_t[:, :] chunk, int chunk_rws, int chunk_cls,
                                        int rows_half, int cols_half, DTYPE_float32_t center_mean,
                                        DTYPE_float32_t thresh_hom, DTYPE_float32_t[:] values):

    cdef:
        Py_ssize_t ia, ja, rr_shape, lni
        DTYPE_intp_t[:] rr, cc
        DTYPE_uint8_t[:] line_values
        DTYPE_float32_t ph_i
        DTYPE_float32_t sfs_max = 0.
        DTYPE_float32_t sfs_min = 999999.
        DTYPE_float32_t sfs_psi = 0.

    # Iterate over each angle, extract values
    ja = 0
    for ia in xrange(0, chunk_rws):

        ph_i = 0.

        # Draw a line between the two endpoints.
        rr, cc = draw_line(rows_half, cols_half, ia, ja)

        rr_shape = rr.shape[0]

        # Extract the values along the line.
        line_values = extract_values(chunk, rr, cc, rr_shape)

        # Iterate over line values.
        for lni in xrange(0, rr_shape):

            if ph_i < thresh_hom:
                ph_i += abs(center_mean - line_values[lni])
            else:
                break

        sfs_max = _get_max_sample(sfs_max, float(lni))
        sfs_min = _get_min_sample_f(sfs_min, float(lni))
        sfs_psi += float(lni)

    ja = chunk_cls - 1
    for ia in xrange(0, chunk_rws):

        ph_i = 0.

        # Draw a line between the two endpoints.
        rr, cc = draw_line(rows_half, cols_half, ia, ja)

        rr_shape = rr.shape[0]

        # Extract the values along the line.
        line_values = extract_values(chunk, rr, cc, rr_shape)

        # Iterate over line values.
        for lni in xrange(0, rr_shape):

            if ph_i < thresh_hom:
                ph_i += abs(center_mean - line_values[lni])
            else:
                break

        sfs_max = _get_max_sample(sfs_max, float(lni))
        sfs_min = _get_min_sample_f(sfs_min, float(lni))
        sfs_psi += float(lni)

    ia = 0
    for ja in xrange(0, chunk_cls):

        ph_i = 0.

        # Draw a line between the two endpoints.
        rr, cc = draw_line(rows_half, cols_half, ia, ja)

        rr_shape = rr.shape[0]

        # Extract the values along the line.
        line_values = extract_values(chunk, rr, cc, rr_shape)

        # Iterate over line values.
        for lni in xrange(0, rr_shape):

            if ph_i < thresh_hom:
                ph_i += abs(center_mean - line_values[lni])
            else:
                break

        sfs_max = _get_max_sample(sfs_max, float(lni))
        sfs_min = _get_min_sample_f(sfs_min, float(lni))
        sfs_psi += float(lni)

    ia = chunk_rws - 1
    for ja in xrange(0, chunk_cls):

        ph_i = 0.

        # Draw a line between the two endpoints.
        rr, cc = draw_line(rows_half, cols_half, ia, ja)

        rr_shape = rr.shape[0]

        # Extract the values along the line.
        line_values = extract_values(chunk, rr, cc, rr_shape)
    
        # Iterate over line values.
        for lni in xrange(0, rr_shape):

            if ph_i < thresh_hom:
                ph_i += abs(center_mean - line_values[lni])
            else:
                break

        sfs_max = _get_max_sample(sfs_max, float(lni))
        sfs_min = _get_min_sample_f(sfs_min, float(lni))
        sfs_psi += float(lni)

    if not npy_isnan(sfs_max) and not npy_isinf(sfs_max):
        values[0] = sfs_max

    if not npy_isnan(sfs_min) and not npy_isinf(sfs_min):
        values[1] = sfs_min

    if not npy_isnan(sfs_psi) and not npy_isinf(sfs_psi):
        values[2] = sfs_psi

    return values


@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_float32_t[:] _sfs_feas(DTYPE_uint8_t[:, :] chunk, int blk_size, DTYPE_float32_t thresh_hom):

    """
    Reference:
        Zhang, Liangpei et al. 2006. "A Pixel Shape Index Coupled With Spectral Information for Classification of High
            Spatial Resolution Remotely Sensed Imagery." IEEE Transactions on Geoscience and Remote Sensing, V. 44, No. 10.

        Huang, Xin et al. 2007. "Classification and Extraction of Spatial Features in Urban Areas Using High-Resolution
            Multispectral Imagery." IEEE Transactions on Geoscience and Remote Sensing, V. 4, No. 2.

    Returns:
        Directional lengths (length=8)
        PSI
    """

    cdef:
        # dict blk_adjs
        int chunk_rws, chunk_cls, rows_half, cols_half, blk_half
        DTYPE_float32_t ctr_blk_mean, sfs_value
        DTYPE_float32_t[:] sfs_values = np.zeros(3, dtype='float32')
        DTYPE_uint8_t[:, :] chunk_block
        Py_ssize_t cbr, cbc

    # block adjustments
    # blk_adjs = {1: 0, 2: 1, 4: 2, 6: 3, 8: 4, 10: 5, 12: 6, 14: 7, 16: 8}
    # blk_adj = blk_adjs[blk_size]

    # get chunk size
    chunk_rws = chunk.shape[0]
    chunk_cls = chunk.shape[1]

    rows_half = chunk_rws / 2
    cols_half = chunk_cls / 2

    blk_half = blk_size / 2

    # get the center block average
    chunk_block = chunk[rows_half-blk_half:rows_half+blk_half, cols_half-blk_half:cols_half+blk_half]
    cbr = chunk_block.shape[0]
    cbc = chunk_block.shape[1]

    ctr_blk_mean = _get_mean_uint8(chunk_block, cbr, cbc)

    # ctr_blk_mean = cv2.mean(chunk[rows_half-blk_half:rows_half+blk_half,
    #                         cols_half-blk_half:cols_half+blk_half].astype(np.float32))[0]

    # list for direction lengths
    # dir_lengths = []

    sfs_values = _get_directions(chunk, chunk_rws, chunk_cls, rows_half, cols_half,
                                 ctr_blk_mean, thresh_hom, sfs_values)

    # dir_arrs = get_directions(np.uint8(chunk), chunk_rws, chunk_cls, rows_half, cols_half, blk_adj, n_angles)

    # for dir_arr in dir_arrs:
    #
    #     PH_i = 0.
    #
    #     for sur_ctr in xrange(1, len(dir_arr)+1):
    #
    #         if PH_i >= thresh_1:
    #             break
    #         else:
    #
    #             PH_i += np.abs(ctr_blk_mean - dir_arr[sur_ctr-1])
    #
    #     dir_lengths.append(sur_ctr-1.)
    #
    # dir_lengths = [dir_l * cell_size for dir_l in dir_lengths]
    #
    # # sfs_max = np.max(dir_lengths)
    # sfs_max = _get_max_f(np.asarray(dir_lengths).astype(np.float32), len(dir_lengths))
    # # sfs_min = np.min(dir_lengths)
    # sfs_min = _get_min_f(np.asarray(dir_lengths).astype(np.float32), len(dir_lengths))
    # # sfs_psi = np.sum(dir_lengths)
    # sfs_psi = get_sum_1d(np.asarray(dir_lengths).astype(np.float32))
    # # sfs_mean, sfs_std = cv2.meanStdDev(np.asarray(dir_lengths))
    # # sfs_mean = np.mean(dir_lengths)
    # sfs_mean = cv2.mean(np.asarray(dir_lengths).astype(np.float32))[0]
    # sfs_sd = pow(get_sum_1d(np.asarray([pow((d_l - sfs_psi), 2) for d_l in dir_lengths]).astype(np.float32)), .5)
    #
    # sfs_list = [sfs_max, sfs_min, sfs_psi, sfs_mean, sfs_sd]

    # for sf_ in xrange(0, 5):
    #
    #     sfs_value = sfs_values[sf_]
    #
    #     if not npy_isnan(sfs_value) and not npy_isinf(sfs_value):
    #         values[sf_] = sfs_value

    return sfs_values


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[DTYPE_float32_t, ndim=1] _feature_sfs(DTYPE_uint8_t[:, :] ch_bd, int blk, list scs,
                                                      DTYPE_float32_t thresh_hom,
                                                      int scales_half, int scales_block, int out_len,
                                                      int rows, int cols):

    cdef:
        Py_ssize_t i, j, ki, k, k_half, st_
        DTYPE_uint8_t[:, :] ch_bd_
        DTYPE_float32_t[:] sts
        DTYPE_float32_t[:] out_list = np.empty(out_len, dtype='float32')
        int pix_ctr = 0
        int n_scales = np.array(scs).shape[0]

    for i from 0 <= i < rows-scales_block by blk:
        for j from 0 <= j < cols-scales_block by blk:
            for ki in xrange(0, n_scales):

                k = scs[ki]

                k_half = k / 2

                ch_bd_ = ch_bd[i+scales_half-k_half:i+scales_half-k_half+k,
                               j+scales_half-k_half:j+scales_half-k_half+k]

                sts = _sfs_feas(ch_bd_, blk, thresh_hom)

                for st_ in xrange(0, 3):

                    out_list[pix_ctr] = sts[st_]

                    pix_ctr += 1

    return np.float32(out_list)


@cython.boundscheck(False)
@cython.wraparound(False)
def feature_sfs(np.ndarray[DTYPE_uint8_t, ndim=2] chBd, int blk, list scs, int end_scale, DTYPE_float32_t thresh_hom):

    cdef:
        Py_ssize_t i, j, ki, k
        int scales_half = end_scale / 2
        int scales_block = end_scale - blk
        int out_len = 0
        int rows = chBd.shape[0]
        int cols = chBd.shape[1]
        int n_scales = np.array(scs).shape[0]

    for i from 0 <= i < rows-scales_block by blk:
        for j from 0 <= j < cols-scales_block by blk:
            for ki in xrange(0, n_scales):
                out_len += 3

    return _feature_sfs(chBd, blk, scs, thresh_hom, scales_half, scales_block, out_len, rows, cols)


# @cython.boundscheck(False)
# cdef list _feature_surf(np.ndarray[DTYPE_uint8_t, ndim=2] surf_arr, k_pts, int j, int i, int k, list scs):
#
#     """
#     Get the moments
#     """
#
#     cdef int start_y = i+(scs[-1]/2)-(k/2)
#     cdef int start_x = j+(scs[-1]/2)-(k/2)
#
#     if surf_arr.max() == 0:
#         return [0., 0., 0., 0.]
#     else:
#         if k_pts:
#             # return desc_stats[m](pyramid_hist_sift(surfArr, kPts, start_x, start_y).sp_hist)
#             return get_moments(pyramid_hist_sift(surf_arr, k_pts, start_x, start_y).sp_hist)
#         else:
#             return [0., 0., 0., 0.]
#
#
# @cython.boundscheck(False)
# def feature_surf(np.ndarray[DTYPE_uint8_t, ndim=2] chBd, int blk, list scs, int end_scale):
#
#     cdef:
#         Py_ssize_t i, j, ki, k, k_half
#         int rows = chBd.shape[0]
#         int cols = chBd.shape[1]
#         list sts
#         DTYPE_float64_t st
#         int scales_half = end_scale / 2
#         int scales_block = end_scale - blk
#         np.ndarray[DTYPE_float64_t, ndim=1] out_list
#         int out_len = 0
#         int pix_ctr = 0
#         int n_scales = np.array(scs).shape[0]
#
#     for i from 0 <= i < rows-scales_block by blk:
#         for j from 0 <= j < cols-scales_block by blk:
#             for ki in xrange(0, n_scales):
#                 out_len += 4
#
#     # set the output list
#     out_list = np.zeros(out_len).astype(np.float64)
#
#     # compute SURF features
#     kPts, descrip = cv2.SURF(50).detectAndCompute(chBd, None)
#
#     for i from 0 <= i < rows-scales_block by blk:
#         for j from 0 <= j < cols-scales_block by blk:
#             for ki in xrange(0, n_scales):
#
#                 sts = _feature_surf(chBd[i+scales_half-k_half:i+scales_half-k_half+k,
#                                     j+scales_half-k_half:j+scales_half-k_half+k], kPts, j, i, k, scs)
#
#                 for st in xrange(0, 4):
#
#                     out_list[pix_ctr] = sts[st]
#
#                     pix_ctr += 1
#
#     return out_list


# ORB keypoints
# Ethan Rublee, Vincent Rabaud, Kurt Konolige, Gary R. Bradski:
#   ORB: An efficient alternative to SIFT or SURF. ICCV 2011: 2564-2571.


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _check_points(DTYPE_float32_t key_x, DTYPE_float32_t key_y,
                        Py_ssize_t ki, Py_ssize_t kj,
                        Py_ssize_t i_, Py_ssize_t j_,
                        Py_ssize_t rr_rows, Py_ssize_t cc_cols,
                        DTYPE_float32_t[:] hist_, Py_ssize_t lv,
                        Py_ssize_t grid_counter) nogil:

    """
    pts = (x,y)
    """

    # Point within the current grid.
    if (i_+ki <= key_y < i_+ki+rr_rows) and (j_+kj <= key_x < j_+kj+cc_cols):
        hist_[lv+grid_counter] += 1


@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_float32_t[:, :] _fill_key_points(list key_point_list):

    cdef:
        Py_ssize_t n_key_points = len(key_point_list)
        DTYPE_float32_t[:, :] key_point_array = np.empty((n_key_points, 2), dtype='float32')

    for key_point_index in xrange(0, n_key_points):

        key_x, key_y = key_point_list[key_point_index].pt

        key_point_array[key_point_index, 0] = key_x
        key_point_array[key_point_index, 1] = key_y

    return key_point_array


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef DTYPE_float32_t[:] _pyramid_hist_sift(DTYPE_uint8_t[:, :] orb_array,
                                           DTYPE_float32_t[:, :] key_point_array,
                                           DTYPE_float32_t[:] hist,
                                           DTYPE_uint8_t[:] levels,
                                           int i, int j):

    cdef:
        Py_ssize_t n_key_points = key_point_array.shape[0]
        Py_ssize_t lv, y_tiles, x_tiles, ki, kj, key_point_index, grid_counter
        Py_ssize_t rr_rows, cc_cols
        Py_ssize_t orb_rows = orb_array.shape[0]
        Py_ssize_t orb_cols = orb_array.shape[1]

    # Fill the keypoints
    # key_point_array = _fill_key_points(k_pts)

    # hist[0] = n_key_points

    # Iterate over each level
    for lv in xrange(0, 3):

        y_tiles = orb_rows / levels[lv]
        x_tiles = orb_cols / levels[lv]

        grid_counter = 1

        for ki from 0 <= ki < orb_rows by y_tiles:

            rr_rows = n_rows_cols(ki, y_tiles, orb_rows)

            for kj from 0 <= kj < orb_cols by x_tiles:

                cc_cols = n_rows_cols(kj, x_tiles, orb_cols)

                # Iterate over each key point.
                # for key_point_index in xrange(0, n_key_points):
                for key_point_index in prange(0, n_key_points, nogil=True, num_threads=64, schedule='static'):

                    _check_points(key_point_array[key_point_index, 0],
                                  key_point_array[key_point_index, 1],
                                  ki, kj, i, j,
                                  rr_rows, cc_cols,
                                  hist, lv, grid_counter)

                grid_counter += 1

    return hist


@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_float32_t[:] _orb(DTYPE_uint8_t[:, :] orb_array,
                             DTYPE_float32_t[:, :] k_pts,
                             DTYPE_float32_t[:] nz,
                             DTYPE_uint8_t[:] levels,
                             int i, int j):

    """
    Get the moments
    """

    return get_moments(_pyramid_hist_sift(orb_array, k_pts, nz, levels, i, j))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[DTYPE_float32_t, ndim=1] _feature_orb(DTYPE_uint8_t[:, :] ch_bd, int blk,
                                                      DTYPE_uint16_t[:] scales_array,
                                                      int scales_half, int scales_block, int scale_length, int out_len,
                                                      int rows, int cols, int scales_length, int max_features):

    cdef:
        Py_ssize_t i, j, ki, st
        DTYPE_uint16_t k, k_half
        DTYPE_float32_t[:] sts
        DTYPE_float32_t[:] out_list
        DTYPE_float32_t[:] nz = np.zeros(84, dtype='float32')
        DTYPE_float32_t[:] nz_mom = np.zeros(4, dtype='float32')
        DTYPE_uint8_t[:] levels = np.array([2, 4, 8], dtype='uint8')
        DTYPE_uint8_t[:, :] ch_bd_block
        Py_ssize_t pix_ctr = 0
        list key_points
        DTYPE_float32_t[:, :] key_point_array

    # Set the output list
    out_list = np.zeros(out_len, dtype='float32')

    # Initiate ORB detector
    orb = cv2.ORB_create(nfeatures=max_features, edgeThreshold=31, patchSize=31, WTA_K=4)

    # Compute ORB keypoints
    key_points, __ = orb.detectAndCompute(np.uint8(ch_bd), None)

    # img = cv2.drawKeypoints(np.uint8(ch_bd), key_points, np.uint8(ch_bd).copy())

    key_point_array = _fill_key_points(key_points)

    for i from 0 <= i < rows-scales_block by blk:
        for j from 0 <= j < cols-scales_block by blk:
            for ki in xrange(0, scale_length):

                k = scales_array[ki]

                k_half = k / 2

                ch_bd_block = ch_bd[i+scales_half-k_half:i+scales_half-k_half+k,
                                    j+scales_half-k_half:j+scales_half-k_half+k]

                # # Compute ORB keypoints
                # key_points, __ = orb.detectAndCompute(np.uint8(ch_bd_block), None)

                if key_points:
                    sts = _orb(ch_bd_block, key_point_array, nz.copy(), levels, i, j)
                else:
                    sts = nz_mom.copy()

                for st in xrange(0, 4):

                    out_list[pix_ctr] = sts[st]

                    pix_ctr += 1

    return np.float32(out_list)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def feature_orb(DTYPE_uint8_t[:, :] ch_bd, int blk, list scs, int end_scale, int max_features=20000):

    cdef:
        Py_ssize_t i, j, ki
        int scales_half = end_scale / 2
        int scales_block = end_scale - blk
        int out_len = 0
        int rows = ch_bd.shape[0]
        int cols = ch_bd.shape[1]
        DTYPE_uint16_t[:] scales_array = np.array(scs, dtype='uint16')
        int scale_length = scales_array.shape[0]

    for i from 0 <= i < rows-scales_block by blk:
        for j from 0 <= j < cols-scales_block by blk:
            for ki in xrange(0, scale_length):
                out_len += 4

    return _feature_orb(ch_bd, blk, scales_array, scales_half, scales_block, scale_length,
                        out_len, rows, cols, scale_length, max_features)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _set_lbp(np.ndarray[DTYPE_uint8_t, ndim=2] chBd, int rows, int cols):

    """
    Get the Local Binary Patterns
    """

    # create LBP radius lookup dictionary
    cdef:
        Py_ssize_t scsc
        dict Rdict	= {4: 1, 8: 1, 16: 2, 32: 4, 64: 8, 128: 16}

        # build the P ranges
        list p_range = [8, 16, 32]
        np.ndarray[DTYPE_uint8_t, ndim=3] lbpBd = np.zeros((len(p_range), rows, cols), dtype='uint8')

    # run lBP for each scale
    for scsc in xrange(0, len(p_range)):
        lbpBd[scsc] = LBP(chBd, p_range[scsc], Rdict[p_range[scsc]], 'uniform')

    return lbpBd, p_range


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef list _feature_lbp(np.ndarray[DTYPE_uint8_t, ndim=2] chBd, int blk, list scs, int end_scale):

    cdef:
        Py_ssize_t i, j, ki, sti
        int pc, pr_bin_count
        int rows = chBd.shape[0]
        int cols = chBd.shape[1]
        np.ndarray[DTYPE_uint8_t, ndim=3] ch_bd
        np.ndarray[DTYPE_uint8_t, ndim=1] sts
        list p_range
        np.ndarray[DTYPE_uint8_t, ndim=3] lbpBd
        int scales_half = end_scale / 2
        int scales_block = end_scale - blk
        np.ndarray[DTYPE_uint8_t, ndim=1] out_list
        int out_len = 0
        int pix_ctr = 0
        DTYPE_uint16_t k, k_half
        DTYPE_uint16_t[:] scales_array = np.array(scs, dtype='uint16')
        int scale_length = scales_array.shape[0]

    # get the LBP images
    lbpBd, p_range = _set_lbp(chBd, rows, cols)

    # count of bins for all p,r LBP pairs
    pr_bin_count = np.sum([pr+2 for pr in p_range])

    for i from 0 <= i < rows-scales_block by blk:
        for j from 0 <= j < cols-scales_block by blk:
            for ki in xrange(0, scale_length):
                out_len += pr_bin_count

    # set the output list
    out_list = np.empty(out_len).astype(np.uint8)

    for i from 0 <= i < rows-scales_block by blk:
        for j from 0 <= j < cols-scales_block by blk:
            for ki in xrange(0, scale_length):

                k = scales_array[ki]

                k_half = k / 2

                ch_bd = lbpBd[:, i+scales_half-k_half:i+scales_half-k_half+k,
                              j+scales_half-k_half:j+scales_half-k_half+k]

                # get histograms and concatenate
                sts = np.concatenate([np.bincount(ch_bd[p_range.index(pc)].flat, minlength=pc+2)
                                      for pc in p_range]).astype(np.uint8)

                for sti in xrange(0, 4):

                    out_list[pix_ctr] = sts[sti]

                    pix_ctr += 1

    # out_list[isnan(out_list) | isinf(out_list)] = 0
    out_list[np.isnan(out_list) | np.isinf(out_list)] = 0

    return np.float32(out_list)


@cython.boundscheck(False)
@cython.wraparound(False)
def feature_lbp(np.ndarray[DTYPE_uint8_t, ndim=2] chBd, int blk, list scs, int end_scale):

    return _feature_lbp(chBd, blk, scs, end_scale)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef list _feature_lbpm(np.ndarray[DTYPE_uint8_t, ndim=2] chBd, int blk, list scs, int end_scale):

    """
    At each scale, returns:
        1: Mean
        2: Variance
        3: Skew
        4: Kurtosis
    """

    cdef:
        Py_ssize_t i, j, sti, ki
        int pc
        int rows = chBd.shape[0]
        int cols = chBd.shape[1]
        np.ndarray[DTYPE_uint8_t, ndim=3] ch_bd
        DTYPE_float32_t[:] sts
        list p_range
        np.ndarray[DTYPE_uint8_t, ndim=3] lbpBd
        int scales_half = end_scale / 2
        int scales_block = end_scale - blk
        np.ndarray[DTYPE_float64_t, ndim=1] out_list
        int out_len = 0
        int pix_ctr = 0
        DTYPE_uint16_t k, k_half
        DTYPE_uint16_t[:] scales_array = np.array(scs, dtype='uint16')
        int scale_length = scales_array.shape[0]

    # get the LBP images
    lbpBd, p_range = _set_lbp(chBd, rows, cols)

    for i from 0 <= i < rows-scales_block by blk:
        for j from 0 <= j < cols-scales_block by blk:
            for ki in xrange(0, scale_length):
                out_len += 4

    # set the output list
    out_list = np.zeros(out_len).astype(np.float64)

    for i from 0 <= i < rows-scales_block by blk:
        for j from 0 <= j < cols-scales_block by blk:
            for ki in xrange(0, scale_length):

                k = scales_array[ki]

                k_half = k / 2

                ch_bd = lbpBd[:, i+scales_half-k_half:i+scales_half-k_half+k,
                              j+scales_half-k_half:j+scales_half-k_half+k]

                # get histograms and concatenate
                sts = get_moments(np.concatenate([np.bincount(ch_bd[p_range.index(pc)].flat, minlength=pc+2)
                                                  for pc in p_range]).astype(np.float32))

                for sti in xrange(0, 4):

                    out_list[pix_ctr] = sts[sti]

                    pix_ctr += 1

    return list(out_list)


@cython.boundscheck(False)
@cython.wraparound(False)
def feature_lbpm(np.ndarray[DTYPE_uint8_t, ndim=2] chBd, int blk, list scs, int end_scale):

    return _feature_lbpm(chBd, blk, scs, end_scale)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline DTYPE_float32_t _get_distance(tuple line):

    return sqrt(pow((line[0][0] - line[1][0]), 2.) + pow((line[0][1] - line[1][1]), 2.))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline DTYPE_float64_t get_slope(tuple line):

    return np.degrees(atan(float((line[0][1] - line[1][1])) / float((line[1][0] - line[0][0]))))


@cython.boundscheck(False)
def houghFunc_1(np.ndarray[DTYPE_uint8_t, ndim=2] edgeArr, int houghIndex, int minLen, np.ndarray[long, ndim=2] checkList1_2, np.ndarray[long, ndim=2] checkList2_2,
                        list scs, int i, int j):

    cdef np.ndarray[int, ndim=1] line
    cdef DTYPE_float32_t pi = 3.14159265
    cdef DTYPE_float32_t pi2 = 3.14159265 / 2.

    if houghIndex == 0:

        if checkList1_2[i, j] == 1:
            return np.mean([ _get_distance(line) for line in cv2.HoughLinesP(edgeArr, 1, pi, minLen, minLineLength=minLen, maxLineGap=2)[0] ])	# average line length
        else:
            return 0.

    if houghIndex == 1:

        if checkList2_2[i, j] == 1:
            return np.mean([ _get_distance(line) for line in cv2.HoughLinesP(edgeArr, 1, pi2, minLen, minLineLength=minLen, maxLineGap=2)[0] ])	# average line length
        else:
            return 0.

    elif houghIndex == 2:

        if checkList1_2[i, j] == 1 and checkList2_2[i, j] == 1:
            return np.vstack((cv2.HoughLinesP(edgeArr, 1, pi, minLen, minLineLength=minLen, maxLineGap=2)[0],
                                    cv2.HoughLinesP(edgeArr, 1, pi2, minLen, minLineLength=minLen, maxLineGap=2)[0])).shape[0]	# number of lines
        elif checkList1_2[i, j] == 1 and checkList2_2[i, j] != 1:
            return cv2.HoughLinesP(edgeArr, 1, pi, minLen, minLineLength=minLen, maxLineGap=2)[0].shape[0]
        elif checkList1_2[i, j] != 1 and checkList2_2[i, j] == 1:
            return cv2.HoughLinesP(edgeArr, 1, pi2, minLen, minLineLength=minLen, maxLineGap=2)[0].shape[0]
        else:
            return 0.

    else:
        return (float(np.argwhere(edgeArr==255).shape[0]) / float(edgeArr.shape[0]*edgeArr.shape[1])) * 100.	# edge density

@cython.boundscheck(False)
def houghFunc_2(np.ndarray[DTYPE_uint8_t, ndim=2] edgeArr, int houghIndex, int minLen, np.ndarray[long, ndim=3] checkList1_3, np.ndarray[long, ndim=3] checkList2_3,
                        list scs, int i, int j, int k):

    cdef np.ndarray[int, ndim=1] line
    cdef DTYPE_float32_t pi = 3.14159265
    cdef DTYPE_float32_t pi2 = 3.14159265 / 2.

    if houghIndex == 0:

        if checkList1_3[scs.index(k), i, j] == 1:
            return np.mean([ _get_distance(line) for line in cv2.HoughLinesP(edgeArr, 1, pi, minLen, minLineLength=minLen, maxLineGap=2)[0] ])	# average vertical line length
        else:
            return 0.

    if houghIndex == 1:

        if checkList2_3[scs.index(k), i, j] == 1:
            return np.mean([ _get_distance(line) for line in cv2.HoughLinesP(edgeArr, 1, pi2, minLen, minLineLength=minLen, maxLineGap=2)[0] ])	# average horizontal line length
        else:
            return 0.

    elif houghIndex == 2:

        if checkList1_3[scs.index(k), i, j] == 1 and checkList2_3[scs.index(k), i, j] == 1:
            return np.vstack((cv2.HoughLinesP(edgeArr, 1, pi, minLen, minLineLength=minLen, maxLineGap=2)[0],
                                    cv2.HoughLinesP(edgeArr, 1, pi2, minLen, minLineLength=minLen, maxLineGap=2)[0])).shape[0]	# number of lines
        elif checkList1_3[scs.index(k), i, j] == 1 and checkList2_3[scs.index(k), i, j] != 1:
            return cv2.HoughLinesP(edgeArr, 1, pi, minLen, minLineLength=minLen, maxLineGap=2)[0].shape[0]
        elif checkList1_3[scs.index(k), i, j] != 1 and checkList2_3[scs.index(k), i, j] == 1:
            return cv2.HoughLinesP(edgeArr, 1, pi2, minLen, minLineLength=minLen, maxLineGap=2)[0].shape[0]
        else:
            return 0.

    else:
        return (float(np.argwhere(edgeArr==255).shape[0]) / float(edgeArr.shape[0]*edgeArr.shape[1])) * 100.	# edge density


@cython.boundscheck(False)
@cython.wraparound(False)
cdef list _hough_function(list lines_list, np.ndarray[DTYPE_uint8_t, ndim=2] edge_arr, int large_rws, int large_cls):

    cdef:
        int small_rws = edge_arr.shape[0]
        int small_cls = edge_arr.shape[1]
        list lines_list_chunk = []
        list line_seg
        DTYPE_float32_t rws_cls = float(small_rws * small_cls)
        tuple line
        DTYPE_float32_t mean_len, num_lines, edge_dens, std_slope
        int x_min = (large_cls - small_cls) / 2
        int x_max = x_min + small_cls
        int y_min = (large_rws - small_rws) / 2
        int y_max = y_min + small_rws

    for line_seg in lines_list:
        for line in line_seg:
            if (line[0][0] >= x_min) and (line[1][0] <= x_max) and (line[0][1] >= y_min) and (line[1][1] <= y_max):
                lines_list_chunk.append(line)

    # average line length
    # mean_len = np.mean([ _get_distance(line) if (len(line) > 0) else 0. for line in lines ])
    mean_len = np.mean([_get_distance(line) for line in lines_list_chunk])

    # number of lines
    # num_lines = float(len([ _get_distance(line) if (len(line) > 0) else 0. for line in lines ]))
    num_lines = float(len([_get_distance(line) for line in lines_list_chunk]))

    # edge density
    # edge_dens = (float(np.argwhere(edge_arr == 1).shape[0]) / rws_cls) * 100.
    edge_dens = (edge_arr.sum() / rws_cls) * 100.

    # standard deviation of line angles
    # slopes = [get_slope(line) if (len(line) > 0) else 0. for line in lines]
    std_slope = np.asarray([get_slope(line) for line in lines_list_chunk]).std()

    # angle bins
    # slope_hist, bins = np.histogram(np.searchsorted([0, 90, 180, 270, 360], slopes), bins=4, range=(1, 4))

    if edge_arr.max() == 0:
        return [0., 0., 0., 0.]
    else:
        # return list([mean_len, num_lines, edge_dens, mean_slope, slope_hist[0], slope_hist[1], slope_hist[2], slope_hist[3]])
        return list([mean_len, num_lines, edge_dens, std_slope])


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[DTYPE_float32_t, ndim=1] _feature_hough(np.ndarray[DTYPE_uint8_t, ndim=2] chBd, int blk,
                                                        DTYPE_uint16_t[:] scales_array,
                                                        int scales_half, int scales_block,
                                                        int scale_length,
                                                        int out_len, int rows, int cols, int threshold,
                                                        int min_len, int line_gap, int end_scale):

    cdef:
        Py_ssize_t i, j, ki
        DTYPE_uint16_t k, k_half
        int k_half_end = end_scale / 2
        DTYPE_float32_t pi = 3.14159
        np.ndarray[DTYPE_uint8_t, ndim=2] ch_bd, large_scale
        int pix_ctr = 0
        np.ndarray[DTYPE_float32_t, ndim=1] out_list = np.zeros(out_len, dtype='float32')
        int large_scale_rws, large_scale_cls
        list lines_list
        list angles = [np.array([np.radians(22.5)]), np.array([np.radians(45)]),
                                                       np.array([np.radians(67.5)]), np.array([np.radians(90)]),
                                                       np.array([np.radians(112.5)]), np.array([np.radians(135)]),
                                                       np.array([np.radians(157.5)]), np.array([np.radians(180)])]
        np.ndarray[DTYPE_float64_t, ndim=1] angle

    pix_ctr = 0

    for i from 0 <= i < rows-scales_block by blk:
        for j from 0 <= j < cols-scales_block by blk:

            # get the angles at the largest scale
            lines_list = []

            # get the largest scale array
            large_scale = chBd[i+scales_half-k_half_end:i+scales_half-k_half_end+end_scale,
                               j+scales_half-k_half_end:j+scales_half-k_half_end+end_scale]

            # get the dimensions for the largest scale
            large_scale_rws = large_scale.shape[0]
            large_scale_cls = large_scale.shape[1]

            # compute the PHL at various angles
            # and add to a list
            lines_list = [PHL(large_scale, threshold=threshold, line_length=min_len,
                              line_gap=line_gap, theta=angle) for angle in angles]

            # get the matching dimensions at each scale
            # and get line statistics
            for ki in xrange(0, scale_length):

                k = scales_array[ki]

                k_half = k / 2

                # get the current scale array
                ch_bd = chBd[i+scales_half - k_half:i+scales_half - k_half + k,
                             j+scales_half - k_half:j+scales_half - k_half + k]

                # get line statistics
                sts = _hough_function(lines_list, ch_bd, large_scale_rws, large_scale_cls)

                for st in sts:
                    out_list[pix_ctr] = st

                    pix_ctr += 1

    return out_list


@cython.boundscheck(False)
@cython.wraparound(False)
def feature_hough(np.ndarray[DTYPE_uint8_t, ndim=2] chBd, int blk, list scs, int end_scale, int threshold,
                  int min_len, int line_gap):

    cdef:
        Py_ssize_t i, j, k
        int rows = chBd.shape[0]
        int cols = chBd.shape[1]
        list sts
        DTYPE_float64_t st
        int out_len = 0
        int scales_half = end_scale / 2
        int scales_block = end_scale - blk
        DTYPE_uint16_t[:] scales_array = np.array(scs, dtype='uint16')
        int scale_length = scales_array.shape[0]

    for i from 0 <= i < rows-scales_block by blk:
        for j from 0 <= j < cols-scales_block by blk:
            for ki in xrange(0, scale_length):
                out_len += 4

    return _feature_hough(chBd, blk, scales_array, scales_half, scales_block, scale_length,
                          out_len, rows, cols, threshold, min_len, line_gap, end_scale)


# PanTex

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _glcm_loop(DTYPE_uint8_t[:, :] image, DTYPE_float32_t[:] distances,
                DTYPE_float32_t[:] angles, int levels,
                DTYPE_float32_t[:, :, :, ::1] out,
                DTYPE_float32_t[:, :] out_sums,
                Py_ssize_t rows, Py_ssize_t cols):

    cdef:
        Py_ssize_t a_idx, d_idx, r, c, row, col
        Py_ssize_t angles_, distances_
        DTYPE_uint8_t i, j
        DTYPE_float32_t angle, distance

    angles_ = angles.shape[0]
    distances_ = distances.shape[0]

    for a_idx in xrange(0, angles_):

        angle = angles[a_idx]

        for d_idx in xrange(0, distances_):

            distance = distances[d_idx]

            # Iterate over the image to get
            #   the grey-level pairs.
            for r in xrange(0, rows):

                for c in xrange(0, cols):

                    # Current row pixel value
                    i = image[r, c]

                    # compute the location of the offset pixel
                    row = r + <int>round(sin(angle) * distance)
                    col = c + <int>round(cos(angle) * distance)

                    # row = r + int(round(sin(angle) * distance))
                    # col = c + int(round(cos(angle) * distance))

                    # row = r + int(round(sin(angle) * distance))
                    # col = c + int(round(cos(angle) * distance))

                    # make sure the offset is within bounds
                    if (0 <= row < rows) and (0 <= col < cols):

                        # Current column pixel value
                        j = image[row, col]

                        if (0 <= i < levels) and (0 <= j < levels):

                            # Fill the co-occurrence matrix.
                            out[i, j, d_idx, a_idx] += 1

                            # Fill the co-occurrence matrix
                            #   for the symmetric pair.
                            out[j, i, d_idx, a_idx] += 1

                            # Add 2 for the symmetric sums
                            out_sums[d_idx, a_idx] += 2


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef _norm_glcm(DTYPE_float32_t[:, :, :, :] Pt,
                DTYPE_float32_t[:, :] Pt_sums, DTYPE_float32_t[:] distances,
                DTYPE_float32_t[:] angles, int levels,
                DTYPE_float32_t[:, :, :, :] glcm_normed_):

    cdef:
        Py_ssize_t a_idx, d_idx, r, c
        Py_ssize_t angles_, distances_
        DTYPE_float32_t angle_dist_sum

    angles_ = angles.shape[0]
    distances_ = distances.shape[0]

    # Get the sums
    for a_idx in xrange(0, angles_):

        for d_idx in xrange(0, distances_):

            # angle_dist_sum = 0.

            # Iterate over the co-occurrence array
            #   and normalize.
            for r in xrange(0, levels):

                for c in xrange(0, levels):
                    glcm_normed_[r, c, d_idx, a_idx] += (Pt[r, c, d_idx, a_idx] / Pt_sums[d_idx, a_idx])


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _check_nans(DTYPE_float32_t[:, :, :, :] glcm_mat_nan,
                                             DTYPE_float32_t[:] distances,
                                             DTYPE_float32_t[:] angles,
                                             int levels):

    cdef:
        Py_ssize_t a_idx, d_idx, r, c
        Py_ssize_t angles_, distances_
        DTYPE_float32_t value2check

    angles_ = angles.shape[0]
    distances_ = distances.shape[0]

    # Get the sums
    for a_idx in xrange(0, angles_):

        for d_idx in xrange(0, distances_):

            # angle_dist_sum = 0.

            # Iterate over the image
            for r in xrange(0, levels):

                for c in xrange(0, levels):

                    value2check = glcm_mat_nan[r, c, d_idx, a_idx]

                    if npy_isnan(value2check) or npy_isinf(value2check):
                        glcm_mat_nan[r, c, d_idx, a_idx] = 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_float32_t[:, :, :, ::1] _greycomatrix(DTYPE_uint8_t[:, :] image,
                                                 DTYPE_float32_t[:] distances,
                                                 DTYPE_float32_t[:] angles,
                                                 Py_ssize_t levels, Py_ssize_t rows, Py_ssize_t cols,
                                                 DTYPE_float32_t[:, :, :, ::1] P,
                                                 DTYPE_float32_t[:, :] angle_dist_sums,
                                                 DTYPE_float32_t[:, :, :, ::1] glcm_normed):

    # count co-occurences
    _glcm_loop(image, distances, angles, levels, P, angle_dist_sums, rows, cols)

    # Normalize the matrix
    _norm_glcm(P, angle_dist_sums, distances, angles, levels, glcm_normed)

    # glcm_normed = _check_nans(glcm_normed, distances, angles, levels)

    # return np.array(glcm_normed)
    return glcm_normed


@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_float32_t _glcm_contrast(DTYPE_float32_t[:, :, :, ::1] P,
                                    DTYPE_float32_t[:] distances,
                                    DTYPE_float32_t[:] angles, Py_ssize_t levels,
                                    DTYPE_float32_t[:, :] contrast_array):

    cdef:
        Py_ssize_t a_idx, d_idx, r, c
        Py_ssize_t angles_, distances_
        DTYPE_float32_t min_contrast = 1000000.
        DTYPE_float32_t contrast_sum

    angles_ = angles.shape[0]
    distances_ = distances.shape[0]

    for a_idx in xrange(0, angles_):

        for d_idx in xrange(0, distances_):

            # Sum the contrast for the current angle/distance pair.
            contrast_sum = 0.

            # Iterate over the co-occurrence matrix
            #   and get the contrast.
            for r in xrange(0, levels):

                for c in xrange(0, levels):
                    contrast_sum += contrast_array[r, c] * P[r, c, d_idx, a_idx]

            # Get the minimum contrast over all angle/distance pairs.
            min_contrast = _get_min_sample_f(min_contrast, contrast_sum)

    return min_contrast


# @cython.boundscheck(False)
# @cython.wraparound(False)
# def pantex_min(DTYPE_float32_t[:, :, :, ::1] glcm_mat,
#                                 DTYPE_float32_t[:] distances, DTYPE_float32_t[:] angles,
#                                 Py_ssize_t levels, DTYPE_float32_t[:, :] contrast_array):
#
#     """
#     Get the local minimum contrast for all displacement vectors
#     """
#
#     # cdef:
#     #     Py_ssize_t dV, dist
#     #     # np.ndarray[DTYPE_float32_t, ndim=1] gmat = np.asarray([greycoprops(glcm_mat, 'contrast')[dist-1][dV]
#     #     #                                                        for dV in xrange(0, len(dispVect))
#     #     #                                                        for dist in dists]).astype(np.float32)
#
#     return _glcm_contrast(glcm_mat, distances, angles, levels, contrast_array)

    # return _get_min_f(gmat, len(gmat))


@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_float32_t[:, :] _set_contrast_weights(int levels):

    cdef:
        Py_ssize_t li, lj
        DTYPE_float32_t[:, :] contrast_array = np.zeros((levels, levels), dtype='float32')

    for li in xrange(0, levels):
        for lj in xrange(0, levels):
            contrast_array[li, lj] = pow(li-lj, 2)

    return contrast_array


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[DTYPE_float32_t, ndim=1] _feature_pantex(DTYPE_uint8_t[:, :] chBd,
                                                         int blk, DTYPE_uint16_t[:] scs, int scales_half,
                                                         int scales_block, int out_len, bint weighted,
                                                         int rows, int cols, int scale_length,
                                                         int levels=32):

    """
    Get the Anisotropic Built-up Presence Index (PanTex)
    """

    cdef:
        Py_ssize_t i, j, ki, block_rows, block_cols
        DTYPE_uint16_t k, k_half
        DTYPE_uint8_t[:, :] ch_bd
        DTYPE_float32_t pi = 3.14159265
        DTYPE_float32_t[:, :, :, ::1] glcm_mat
        DTYPE_float32_t con_min
        Py_ssize_t pix_ctr = 0
        DTYPE_float32_t[:] out_list = np.zeros(out_len, dtype='float32')
        # directions [E, NE, N, NW]
        DTYPE_float32_t[:] disp_vect = np.array([0., pi / 6., pi / 4., pi / 3., pi / 2., (2. * pi) / 3.,
                                                 (3. * pi) / 4., (5. * pi) / 6.], dtype='float32')
        DTYPE_float32_t[:] dists = np.array([1, 2], dtype='float32')
        DTYPE_float32_t[:, :] contrast_weights = _set_contrast_weights(levels)

        DTYPE_float32_t[:, :, :, ::1] P_ = np.zeros((levels, levels, dists.shape[0], disp_vect.shape[0]),
                                                    dtype='float32')

        DTYPE_float32_t[:, :] angle_dist_sums_ = np.zeros((dists.shape[0], disp_vect.shape[0]),
                                                          dtype='float32')

        DTYPE_float32_t[:, :, :, ::1] glcm_normed_ = np.zeros((levels, levels, dists.shape[0], disp_vect.shape[0]),
                                                              dtype='float32')

    if weighted:

        for i from 0 <= i < rows-scales_block by blk:
            for j from 0 <= j < cols-scales_block by blk:
                for ki in xrange(0, scale_length):

                    k = scs[ki]

                    k_half = k / 2

                    ch_bd = chBd[i+scales_half-k_half:i+scales_half-k_half+k,
                                 j+scales_half-k_half:j+scales_half-k_half+k]

                    block_rows = ch_bd.shape[0]
                    block_cols = ch_bd.shape[1]

                    if _get_max(ch_bd, ch_bd.shape[0], ch_bd.shape[1]) == 0:
                        con_min = 0.
                    else:

                        # glcm_mat = greycomatrix(ch_bd, dists, disp_vect,
                        #                         levels=32, symmetric=True, normed=True).astype(np.float32)

                        glcm_mat = _greycomatrix(ch_bd, dists, disp_vect, levels, block_rows, block_cols,
                                                 P_.copy(), angle_dist_sums_.copy(), glcm_normed_.copy())

                        con_min = _glcm_contrast(glcm_mat, dists, disp_vect, levels, contrast_weights)
                        # con_min = pantex_min(glcm_mat, dists, disp_vect,
                        #                      levels, contrast_weights) * cv2.mean(ch_bd)[0]

                    out_list[pix_ctr] = con_min

                    pix_ctr += 1

    else:

        for i from 0 <= i < rows-scales_block by blk:
            for j from 0 <= j < cols-scales_block by blk:
                for ki in xrange(0, scale_length):

                    k = scs[ki]

                    k_half = k / 2

                    ch_bd = chBd[i+scales_half-k_half:i+scales_half-k_half+k,
                                 j+scales_half-k_half:j+scales_half-k_half+k]

                    block_rows = ch_bd.shape[0]
                    block_cols = ch_bd.shape[1]

                    if _get_max(ch_bd, block_rows, block_cols) == 0:
                        con_min = 0.
                    else:

                        # glcm_mat = greycomatrix(ch_bd, dists, disp_vect,
                        #                         levels=32, symmetric=True, normed=True).astype(np.float32)

                        glcm_mat = _greycomatrix(ch_bd, dists, disp_vect, levels, block_rows, block_cols,
                                                 P_.copy(), angle_dist_sums_.copy(), glcm_normed_.copy())

                        con_min = _glcm_contrast(glcm_mat, dists, disp_vect, levels, contrast_weights)

                        # print 'finished contrast'
                        # print con_min
                        # print
                        # con_min = pantex_min(glcm_mat, dists, disp_vect, levels, contrast_weights)
                    # print pix_ctr, out_list.shape[0]
                    out_list[pix_ctr] = con_min

                    pix_ctr += 1

    return np.float32(out_list)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def feature_pantex(np.ndarray[DTYPE_uint8_t, ndim=2] chBd, int blk, list scs, int end_scale, bint weighted):

    cdef:
        Py_ssize_t i, j, ki
        int scales_half = end_scale / 2
        int scales_block = end_scale - blk
        int out_len = 0
        int rows = chBd.shape[0]
        int cols = chBd.shape[1]
        DTYPE_uint16_t[:] scales_array = np.array(scs, dtype='uint16')
        int scale_length = scales_array.shape[0]

    for i from 0 <= i < rows-scales_block by blk:
        for j from 0 <= j < cols-scales_block by blk:
            for ki in xrange(0, scale_length):
                out_len += 1

    return _feature_pantex(chBd, blk, scales_array, scales_half, scales_block,
                           out_len, weighted, rows, cols, scale_length)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_float32_t[:, :] _create_weights(int rs, int cs):

    cdef:
        Py_ssize_t ri, rj
        DTYPE_float32_t[:, :] dist_weights = np.empty((rs, cs), dtype='float32')
        DTYPE_float32_t rm = rs / 2.
        DTYPE_float32_t cm = cs / 2.

    for ri in xrange(0, rs):
        for rj in xrange(0, cs):
            dist_weights[ri, rj] = _euclidean_distance(cm, rm, float(rj), float(ri))

    return dist_weights


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[DTYPE_float32_t, ndim=1] feature_mean_float32(DTYPE_float32_t[:, :] ch_bd, int blk,
                                                              DTYPE_uint16_t[:] scs,
                                                              int out_len, int scales_half, int scales_block,
                                                              int scale_length):

    cdef:
        Py_ssize_t i, j, ki, pix_ctr
        unsigned int bcr, bcc
        DTYPE_uint16_t k, k_half
        DTYPE_float32_t[:] out_list = np.zeros(out_len, dtype='float32')
        int rows = ch_bd.shape[0]
        int cols = ch_bd.shape[1]
        DTYPE_float32_t[:, :] block_chunk
        list weights_list = []

    for ki in xrange(0, scale_length):
        k = scs[ki]
        k_half = k / 2
        rs = (scales_half - k_half + k) - (scales_half - k_half)
        cs = (scales_half - k_half + k) - (scales_half - k_half)
        weights_list.append(_create_weights(rs, cs))

    pix_ctr = 0

    for i from 0 <= i < rows-scales_block by blk:
        for j from 0 <= j < cols-scales_block by blk:
            for ki in xrange(0, scale_length):

                k = scs[ki]

                k_half = k / 2

                block_chunk = ch_bd[i+scales_half-k_half:i+scales_half-k_half+k,
                                    j+scales_half-k_half:j+scales_half-k_half+k]

                bcr = block_chunk.shape[0]
                bcc = block_chunk.shape[1]

                out_list[pix_ctr] = _get_weighted_mean(block_chunk, weights_list[ki], bcr, bcc)

                pix_ctr += 1

    return np.float32(out_list)


# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
# cdef np.ndarray[DTYPE_float32_t, ndim=1] _feature_mean(DTYPE_uint8_t[:, :] chBd, int blk, DTYPE_uint16_t[:] scs,
#                                                        int out_len, int scales_half, int scales_block, int scale_length):
#
#     cdef:
#         int i, j, ki, pix_ctr
#         unsigned int bcr, bcc
#         DTYPE_uint16_t k, k_half
#         DTYPE_float32_t[:] out_list = np.zeros(out_len, dtype='float32')
#         int rows = chBd.shape[0]
#         int cols = chBd.shape[1]
#         DTYPE_uint8_t[:, :] block_chunk
#
#     pix_ctr = 0
#
#     for i from 0 <= i < rows-scales_block by blk:
#         for j from 0 <= j < cols-scales_block by blk:
#             for ki in xrange(0, scale_length):
#
#                 k = scs[ki]
#
#                 k_half = k / 2
#
#                 block_chunk = chBd[i+scales_half-k_half:i+scales_half-k_half+k,
#                                    j+scales_half-k_half:j+scales_half-k_half+k]
#
#                 bcr = block_chunk.shape[0]
#                 bcc = block_chunk.shape[1]
#
#                 out_list[pix_ctr] = _get_mean(np.float32(block_chunk), bcr, bcc)
#
#                 pix_ctr += 1
#
#     return np.float32(out_list)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def feature_mean(np.ndarray ch_bd, int blk, list scs, int end_scale):

    cdef:
        Py_ssize_t i, j, k
        int scales_half = end_scale / 2
        int scales_block = end_scale - blk
        int out_len = 0
        int rows = ch_bd.shape[0]
        int cols = ch_bd.shape[1]
        DTYPE_uint16_t[:] scales_array = np.array(scs, dtype='uint16')
        int scale_length = scales_array.shape[0]

    for i from 0 <= i < rows-scales_block by blk:
        for j from 0 <= j < cols-scales_block by blk:
            for ki in xrange(0, scale_length):
                out_len += 1

    return feature_mean_float32(np.float32(ch_bd), blk, scales_array, out_len, scales_half,
                                scales_block, scale_length)


@cython.boundscheck(False)
def feaCtrFloat64(np.ndarray[DTYPE_float64_t, ndim=2] chBd, int blk, list scs, int rows, int cols):

    cdef int i, j, k

    return [ chBd[i+scs[-1]/2, j+scs[-1]/2] for k in scs for i in xrange(0, rows-(scs[-1]-blk), blk) for j in xrange(0, cols-(scs[-1]-blk), blk) ]


@cython.boundscheck(False)
def feaCtrFloat32(np.ndarray[DTYPE_float32_t, ndim=2] chBd, int blk, list scs, int rows, int cols):

    cdef int i, j, k

    return [ chBd[i+scs[-1]/2, j+scs[-1]/2] for k in scs for i in xrange(0, rows-(scs[-1]-blk), blk) for j in xrange(0, cols-(scs[-1]-blk), blk) ]


@cython.boundscheck(False)
def feaCtr_uint16(np.ndarray[unsigned short, ndim=2] chBd, int blk, list scs, int rows, int cols):

    cdef int i, j, k

    return [ chBd[i+scs[-1]/2, j+scs[-1]/2] for k in scs for i in xrange(0, rows-(scs[-1]-blk), blk) for j in xrange(0, cols-(scs[-1]-blk), blk) ]


@cython.boundscheck(False)
def feaCtr_uint8(np.ndarray[unsigned char, ndim=2] chBd, int blk, list scs, int rows, int cols):

    cdef int i, j, k

    return [ chBd[i+scs[-1]/2, j+scs[-1]/2] for k in scs for i in xrange(0, rows-(scs[-1]-blk), blk) for j in xrange(0, cols-(scs[-1]-blk), blk) ]


@cython.boundscheck(False)
def feaCtr_uint(np.ndarray[int, ndim=2] chBd, int blk, list scs, int rows, int cols):

    cdef int i, j, k

    return [ chBd[i+scs[-1]/2, j+scs[-1]/2] for k in scs for i in xrange(0, rows-(scs[-1]-blk), blk) for j in xrange(0, cols-(scs[-1]-blk), blk) ]


@cython.boundscheck(False)
def feaCtr(np.ndarray chBd, int blk, list scs):

    cdef int rows = chBd.shape[0]
    cdef int cols = chBd.shape[1]

    if chBd.dtype == 'float64':
        return feaCtrFloat64(chBd, blk, scs, rows, cols)
    elif chBd.dtype == 'float32':
        return feaCtrFloat32(chBd, blk, scs, rows, cols)
    elif chBd.dtype == 'uint16':
        return feaCtr_uint16(chBd, blk, scs, rows, cols)
    elif chBd.dtype == 'uint8':
        try:
            return feaCtr_uint8(chBd, blk, scs, rows, cols)
        except:
            return feaCtr_uint(chBd, blk, scs, rows, cols)


# Lacunarity

@cython.cdivision(True)
cdef int max_box_number(DTYPE_uint8_t[:, :] w, int rr_rows, int rr_cols):

    cdef:
        int maxi = _get_max(w, rr_rows, rr_cols)
        int mini = _get_min(w, rr_rows, rr_cols)

        int boxes_max = <int>(ceil(float(maxi) / _get_min_sample_i(rr_rows, rr_cols)))
        int boxes_min = <int>(ceil(float(mini) / _get_min_sample_i(rr_rows, rr_cols)))

    return (boxes_max - boxes_min) + 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef DTYPE_float32_t[:] _div1d(DTYPE_float32_t[:] array1d, int cs, DTYPE_float32_t div_value) nogil:

    cdef:
        Py_ssize_t js

    for js in xrange(0, cs):
        array1d[js] /= div_value

    return array1d


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef DTYPE_float32_t _lacunarity(DTYPE_uint8_t[:, :] chunk_sub, int r):

    cdef:
        int rows_ = chunk_sub.shape[0]
        int cols_ = chunk_sub.shape[1]

        # Get max for probability.
        int maxw = (_get_max(chunk_sub, rows_, cols_) - _get_min(chunk_sub, rows_, cols_)) + 1

        # Get the maximum number of boxes in each block
        #   ceiling is needed for uneven rows or columns.
        int n_rows_ = int(ceil(float(rows_) / r))
        int n_cols_ = int(ceil(float(cols_) / r))

        int ns = int(float(n_rows_) * float(n_cols_))

        # Create array of zeros
        DTYPE_uint8_t[:] nsr = np.zeros(ns, dtype='uint8')

        int maxww = maxw + 1

        DTYPE_float32_t[:] nqr = np.zeros(maxww, dtype='float32')

        int nn = 0
        Py_ssize_t mm, n, dd
        int rr_rows, rr_cols
        DTYPE_uint8_t[:, :] w
        int m

        DTYPE_float32_t[:] nqr_sum
        DTYPE_float32_t[:] l1 = np.zeros(ns, dtype='float32')
        DTYPE_float32_t[:] l2 = l1.copy()
        int ns_rp

    for mm from 0 <= mm < rows_ by r:

        rr_rows = n_rows_cols(mm, r, rows_)

        for n from 0 <= n < cols_ by r:

            rr_cols = n_rows_cols(n, r, cols_)

            # Differential Box Counting
            #   Return max. box value minus min. box value for r x r window
            w = chunk_sub[mm:mm+rr_rows, n:n+rr_cols]
            m = max_box_number(w, rr_rows, rr_cols)

            # Append the mass to the temporary MASS array.
            # The length of array equals number of boxes in the block.
            nsr[nn] = m

            # Append MASS counts for probability.
            # The length of array equals max. number of MASS possibilites (+1).
            nqr[m] += 1

            nn += 1

    # Get probability for each MASS count.
    #   MASS counts divided by total number of boxes in k x k window.
    # nqr_sum = np.divide(np.asarray(nqr), ns).astype(np.float32)
    nqr_sum = _div1d(nqr, maxww, float(ns))

    # Calculate moments
    #   L1 = MASS squared times the probability
    #   (both are arrays of length equal to the number of boxes in k x k window).
    for dd in xrange(0, ns):

        ns_rp = nsr[dd]

        l1[dd] = pow(ns_rp, 2) * nqr_sum[ns_rp]
        l2[dd] = ns_rp * nqr_sum[ns_rp]

    # Sum the L2 array and square the result
    l2_sum = pow(_get_sum1d(l2, ns), 2)

    # Lacunarity for k x k block
    if l2_sum != 0:
        return _get_sum1d(l1, ns) / l2_sum
    else:
        return 0.


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[DTYPE_float32_t, ndim=1] _feature_lacunarity(DTYPE_uint8_t[:, :] chunk_block, int blk,
                                                             DTYPE_uint16_t[:] scales, int scales_half,
                                                             int scales_block, int rows, int cols, int r,
                                                             int out_len, int scale_length):

    cdef:
        Py_ssize_t i, j, ki, cr, cc
        DTYPE_uint16_t k, k_half
        Py_ssize_t pixel_counter = 0
        DTYPE_uint8_t[:, :] ch_bd
        DTYPE_float32_t[:] out_list = np.zeros(out_len, dtype='float32')

    for i from 0 <= i < rows-scales_block by blk:
        for j from 0 <= j < cols-scales_block by blk:
            for ki in xrange(0, scale_length):

                k = scales[ki]
                k_half = (k / 2)

                ch_bd = chunk_block[i+scales_half-k_half:i+scales_half-k_half+k,
                                    j+scales_half-k_half:j+scales_half-k_half+k]

                cr = ch_bd.shape[0]
                cc = ch_bd.shape[1]

                if _get_max(ch_bd, cr, cc) == 0:
                    out_list[pixel_counter] = 0
                else:
                    out_list[pixel_counter] = _lacunarity(ch_bd, r)

                pixel_counter += 1

    return np.float32(out_list)


@cython.boundscheck(False)
@cython.wraparound(False)
def feature_lacunarity(np.ndarray chunk_block, int blk, list scales, int end_scale, int r=2):

    cdef:
        Py_ssize_t i, j, ki
        int rows = chunk_block.shape[0]
        int cols = chunk_block.shape[1]
        int scales_half = end_scale / 2
        int scales_block = end_scale - blk
        int out_len = 0
        DTYPE_uint16_t[:] scale_array = np.array(scales, dtype='uint16')
        int scale_length = scale_array.shape[0]

    for i from 0 <= i < rows-scales_block by blk:
        for j from 0 <= j < cols-scales_block by blk:
            for ki in xrange(0, scale_length):
                out_len += 1

    return _feature_lacunarity(np.uint8(chunk_block), blk, scale_array, scales_half, scales_block,
                               rows, cols, r, out_len, scale_length)


@cython.boundscheck(False)
@cython.cdivision(True)
cdef azimuthal_avg(image, center=None):

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
        center = np.array([(x.max() - x.min()) / 2., (x.max() - x.min()) / 2.0])

    # get hypotenuse
    r = np.hypot(np.subtract(x, center[0]), np.subtract(y, center[1]))

    # get sorted radii indices
    ind = np.argsort(r.flat)
    rSorted = r.flat[ind]

    # get image values from index positions
    iSorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    rInt = rSorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = np.subtract(rInt[1:], rInt[:-1])  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = np.subtract(rind[1:], rind[:-1])        # number of radius bin

    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(iSorted, dtype=float)
    tbin = np.subtract(csim[rind[1:]], csim[rind[:-1]])

    radialProf = np.divide(tbin, nr)

    return radialProf


@cython.boundscheck(False)
cdef DTYPE_float32_t[:, :] _fourier_transform(DTYPE_uint8_t[:, :] chunk_block):

    cdef:
        DTYPE_float32_t[:, :] dft = cv2.dft(np.float32(chunk_block), flags=cv2.DFT_COMPLEX_OUTPUT)
        DTYPE_float32_t[:, :, :] dft_shift = np.fft.fftshift(dft)

    # get the Power Spectrum
    return np.float32(azimuthal_avg(20. * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[DTYPE_float32_t, ndim=1] _feature_fourier(DTYPE_uint8_t[:, :] chunk_block, int blk,
                                                          DTYPE_uint16_t[:] scales, int scales_half,
                                                          int scales_block, int rows, int cols,
                                                          int out_len, int scale_length):

    cdef:
        Py_ssize_t i, j, ki, cr, cc
        DTYPE_uint16_t k, k_half
        Py_ssize_t pixel_counter = 0
        DTYPE_uint8_t[:, :] ch_bd
        DTYPE_float32_t[:] out_list = np.zeros(out_len, dtype='float32')

    for i from 0 <= i < rows-scales_block by blk:
        for j from 0 <= j < cols-scales_block by blk:
            for ki in xrange(0, scale_length):

                k = scales[ki]
                k_half = k / 2

                ch_bd = chunk_block[i+scales_half-k_half:i+scales_half-k_half+k,
                                    j+scales_half-k_half:j+scales_half-k_half+k]

                bcr = ch_bd.shape[0]
                bcc = ch_bd.shape[1]

                ch_bd_transform = _fourier_transform(ch_bd)

                out_list[pixel_counter] = _get_mean(ch_bd_transform, bcr, bcc)
                pixel_counter += 1
                out_list[pixel_counter] = _get_mean(ch_bd_transform, bcr, bcc)
                pixel_counter += 1

    return np.float32(out_list)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def feature_fourier(np.ndarray chunk_block, int blk, list scales, int end_scale):

    cdef:
        Py_ssize_t i, j, ki
        int rows = chunk_block.shape[0]
        int cols = chunk_block.shape[1]
        int scales_half = end_scale / 2
        int scales_block = end_scale - blk
        int out_len = 0
        DTYPE_uint16_t[:] scale_array = np.array(scales, dtype='uint16')
        int scale_length = scale_array.shape[0]

    for i from 0 <= i < rows-scales_block by blk:
        for j from 0 <= j < cols-scales_block by blk:
            for ki in xrange(0, scale_length):
                out_len += 1

    return _feature_fourier(np.uint8(chunk_block), blk, scale_array, scales_half, scales_block,
                               rows, cols, out_len, scale_length)
