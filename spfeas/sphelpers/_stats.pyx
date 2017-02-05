#!/usr/bin/env python

import cython
cimport cython
from cpython cimport array

import numpy as np
cimport numpy as np

from copy import copy

# from libc.stdlib cimport free
from libc.math cimport atan, sqrt, sin, cos, floor

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

# cdef extern from 'numpy/npy_math.h':
#     DTYPE_float32_t npy_floor(DTYPE_float32_t x)


# @cython.profile(False)
cdef inline DTYPE_float32_t roundd(DTYPE_float32_t val) nogil:
     return floor(val + .5)


@cython.profile(False)
cdef inline DTYPE_float32_t sqrt_f(DTYPE_float32_t sx) nogil:
    return sx * .5


@cython.profile(False)
cdef inline DTYPE_float32_t abs_f(DTYPE_float32_t sx) nogil:
    return sx * -1. if sx < 0 else sx


@cython.profile(False)
cdef inline DTYPE_uint8_t abs_ui(DTYPE_uint8_t sx) nogil:
    return sx * -1 if sx < 0 else sx


@cython.profile(False)
cdef inline Py_ssize_t abs_s(Py_ssize_t sx) nogil:
    return sx * -1 if sx < 0 else sx


@cython.profile(False)
cdef inline int n_rows_cols(int pixel_index, int rows_cols, int block_size) nogil:
    return rows_cols if pixel_index + rows_cols < block_size else block_size - pixel_index


@cython.profile(False)
cdef inline DTYPE_float32_t pow2(DTYPE_float32_t sx) nogil:
    return sx * sx


@cython.profile(False)
cdef inline DTYPE_float32_t pow3(DTYPE_float32_t sx) nogil:
    return sx * sx * sx


@cython.profile(False)
cdef inline DTYPE_float32_t pow4(DTYPE_float32_t sx) nogil:
    return sx * sx * sx * sx


@cython.profile(False)
cdef inline int _get_min_sample_i(int s1, int s2):
    return s2 if s2 < s1 else s1


@cython.profile(False)
cdef inline DTYPE_float32_t _get_min_sample_f(DTYPE_float32_t s1, DTYPE_float32_t s2) nogil:
    return s2 if s2 < s1 else s1


@cython.profile(False)
cdef inline DTYPE_uint8_t _get_min_sample_int(DTYPE_uint8_t s1, DTYPE_uint8_t s2) nogil:
    return s2 if s2 < s1 else s1


@cython.profile(False)
cdef inline DTYPE_float32_t _get_max_sample(DTYPE_float32_t s1, DTYPE_float32_t s2) nogil:
    return s2 if s2 > s1 else s1


@cython.profile(False)
cdef inline DTYPE_uint8_t _get_max_sample_int(DTYPE_uint8_t s1, DTYPE_uint8_t s2) nogil:
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

    with nogil:

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

    with nogil:

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

    with nogil:

        for bi in xrange(0, rs):
            for bj in xrange(0, cs):

                m = _get_max_sample_int(m, block[bi, bj])

    return m


@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_float32_t _get_max_f(DTYPE_float32_t[:] in_row, int cols) nogil:

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

    with nogil:

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
cdef DTYPE_float32_t _get_sum_uint8(DTYPE_uint8_t[:, :] block, int rs, int cs) nogil:

    cdef:
        Py_ssize_t bi, bj
        DTYPE_float32_t block_sum = 0.

    for bi in range(0, rs):
        for bj in range(0, cs):
            block_sum += float(block[bi, bj])

    return block_sum


@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_float32_t _get_sum(DTYPE_float32_t[:, :] block, int rs, int cs) nogil:

    cdef:
        Py_ssize_t bi, bj
        DTYPE_float32_t block_sum = 0.

    for bi in range(0, rs):
        for bj in range(0, cs):
            block_sum += block[bi, bj]

    return block_sum


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef DTYPE_float32_t _get_mean(DTYPE_float32_t[:, :] block, int rs, int cs) nogil:

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

    for bi in range(0, rs):
        for bj in range(0, cs):

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
cdef DTYPE_float32_t[:] _get_weighted_mean_var(DTYPE_float32_t[:, :] block,
                                               DTYPE_float32_t[:, :] weights,
                                               int rs, int cs,
                                               DTYPE_float32_t[:] out_values_):

    cdef:
        Py_ssize_t bi, bj
        DTYPE_float32_t n_samps = float(rs*cs)
        DTYPE_float32_t mu = _get_weighted_mean(block, weights, rs, cs)
        DTYPE_float32_t block_var = 0.

    with nogil:

        for bi in range(0, rs):
            for bj in range(0, cs):
                block_var += pow2(float(block[bi, bj]) - mu)

    out_values_[0] = mu
    out_values_[1] = block_var / n_samps

    return out_values_


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef DTYPE_float32_t _get_mean_uint8(DTYPE_uint8_t[:, :] block, int rs, int cs) nogil:

    cdef:
        DTYPE_float32_t n_samps = float(rs*cs)

    return _get_sum_uint8(block, rs, cs) / n_samps


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef DTYPE_float32_t _get_std_1d(DTYPE_float32_t[:] block, int cs) nogil:

    cdef:
        Py_ssize_t bj
        DTYPE_float32_t mu = _get_mean_1d(block, cs)
        DTYPE_float32_t block_var = 0.

    with nogil:

        for bj in range(0, cs):
            block_var += pow2(float(block[bj]) - mu)

    return sqrt(block_var / cs)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef DTYPE_float32_t _get_std_1d_uint16(DTYPE_uint16_t[:] block, int cs) nogil:

    cdef:
        Py_ssize_t bj
        DTYPE_float32_t mu = _get_mean_1d_uint16(block, cs)
        DTYPE_float32_t block_var = 0.

    for bj in range(0, cs):
        block_var += pow2(float(block[bj]) - mu)

    return sqrt_f(block_var / cs)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef DTYPE_float32_t _get_var(DTYPE_float32_t[:, :] block, int rs, int cs) nogil:

    cdef:
        Py_ssize_t bi, bj
        DTYPE_float32_t mu = _get_mean(block, rs, cs)
        DTYPE_float32_t block_var = 0.

    for bi in range(0, rs):
        for bj in range(0, cs):
            block_var += pow2(float(block[bi, bj]) - mu)

    return block_var / (rs*cs)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef DTYPE_float32_t _get_var_uint8(DTYPE_uint8_t[:, :] block, int rs, int cs):

    cdef:
        Py_ssize_t bi, bj
        DTYPE_float32_t mu = _get_mean_uint8(block, rs, cs)
        DTYPE_float32_t block_var = 0.

    with nogil:

        for bi in range(0, rs):
            for bj in range(0, cs):
                block_var += pow2(float(block[bi, bj]) - mu)

    return block_var / (rs*cs)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_float32_t _get_std1d(DTYPE_float32_t[:] block, int cs, DTYPE_float32_t psi):

    cdef:
        Py_ssize_t bj
        DTYPE_float32_t[:] pow_array = np.zeros(cs, dtype='float32')

    # for bj in prange(0, cs, nogil=True, num_threads=cs, schedule='static'):
    for bj in range(0, cs):
        pow_array[bj] = pow2((block[bj] - psi))

    return sqrt(_get_sum1d(pow_array, cs))


@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_float32_t _get_sum1d(DTYPE_float32_t[:] block, int cs) nogil:

    cdef:
        Py_ssize_t bj
        DTYPE_float32_t block_sum = block[0]

    # for bj in prange(0, cs, nogil=True, num_threads=cs, schedule='static'):
    for bj in range(1, cs):
        block_sum += block[bj]

    return block_sum


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef DTYPE_float32_t _get_mean_1d(DTYPE_float32_t[:] block, int cs) nogil:
    return _get_sum1d(block, cs) / cs


@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_float32_t _get_sum1d_uint16(DTYPE_uint16_t[:] block, int cs) nogil:

    cdef:
        Py_ssize_t bj
        DTYPE_uint16_t block_sum = block[0]

    for bj in range(1, cs):
        block_sum += block[bj]

    return float(block_sum)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef DTYPE_float32_t _get_mean_1d_uint16(DTYPE_uint16_t[:] block, int cs) nogil:
    return _get_sum1d_uint16(block, cs) / cs


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
cdef DTYPE_uint16_t[:, :] draw_line(Py_ssize_t y0, Py_ssize_t x0, Py_ssize_t y1, Py_ssize_t x1):

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
        Py_ssize_t dx = abs_s(x1 - x0)
        Py_ssize_t dy = abs_s(y1 - y0)
        Py_ssize_t sx, sy, d, i
        DTYPE_uint16_t[:, :] rc

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

    rc = np.empty((2, dx+1), dtype='uint16')
    #rc = <double * >malloc((n ** 2) * sizeof(double))
    # cc = rr.copy()
    # rr = clone(template, int(dx)+1, True)
    # cc = clone(template, int(dx)+1, True)

    with nogil:

        for i in range(0, dx):

            if steep:
                rc[0, i] = x
                rc[1, i] = y
            else:
                rc[0, i] = y
                rc[1, i] = x

            while d >= 0:

                y += sy
                d -= 2 * dx

            x += sx
            d += 2 * dy

        rc[0, dx] = y1
        rc[1, dx] = x1

    return rc


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _get_stats(DTYPE_float32_t[:] block, int samps, DTYPE_float32_t[:] output_array) nogil:

    """
    Calculate the moments 1-4, skewness, and kurtosis
    """

    cdef:
        DTYPE_float32_t the_mean = _get_mean_1d(block, samps)
        DTYPE_float32_t the_max = _get_max_f(block, samps)
        Py_ssize_t idx
        DTYPE_float32_t bx = block[0]
        DTYPE_float32_t val_dev = bx - the_mean
        DTYPE_float32_t m1 = bx - the_mean      # 1st moment
        DTYPE_float32_t m2 = pow2(val_dev)      # 2nd moment
        DTYPE_float32_t m3 = pow3(val_dev)      # 3rd moment
        DTYPE_float32_t m4 = pow4(val_dev)      # 4th moment

    for idx in range(1, samps):

        bx = block[idx]
        val_dev = bx - the_mean

        m1 += val_dev
        m2 += pow2(val_dev)
        m3 += pow3(val_dev)
        m4 += pow4(val_dev)

    m1 /= samps
    m2 /= samps
    m3 /= samps
    m4 /= samps

    output_array[0] = the_max
    output_array[1] = m1
    output_array[2] = m2
    output_array[3] = m3
    output_array[4] = m4
    output_array[5] = m3 / pow3(sqrt(m2))    # skewness: ratio of 3rd moment and standard dev. cubed
    output_array[6] = m4 / pow2(m2)          # kurtosis


@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_float32_t[:] get_moments(DTYPE_float32_t[:] img_arr):

    """
    Get the moments for 1d array
    """

    cdef:
        int img_arr_cols = img_arr.shape[0]
        DTYPE_float32_t[:] output = np.zeros(7, dtype='float32')

    if _get_max_f(img_arr, img_arr_cols) != 0:
        _get_stats(img_arr, img_arr_cols, output)

    return output


# Gabor filter bank


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _convolution(DTYPE_float32_t[:, :] block2convolve,
                       DTYPE_float32_t[:, :] gkernel,
                       int br, int bc, int knr, int knc,
                       DTYPE_float32_t[:, :] out_convolved) nogil:

    """"
    2d convolution of a Gabor kernel over a local window
    """

    cdef:
        Py_ssize_t bi, bj, bki, bkj
        DTYPE_float32_t kernel_sum

    for bi in range(0, br-knr):
        for bj in range(0, bc-knc):

            kernel_sum = 0.

            for bki in range(0, knr):
                for bkj in range(0, knc):
                    kernel_sum += block2convolve[bi+bki, bj+bkj] * gkernel[bki, bkj]

            out_convolved[bi, bj] = kernel_sum


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[DTYPE_float32_t, ndim=1] _feature_gabor(DTYPE_float32_t[:, :] chBd, int blk,
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
        DTYPE_float32_t[:, :] ch_bd
        DTYPE_float32_t[:, :] ch_bd_gabor, ch_bd_gabor_c
        DTYPE_float32_t[:] sts
        list st
        DTYPE_float32_t[:] out_list = np.zeros(out_len, dtype='float32')
        int pix_ctr = 0
        int n_kernels = np.asarray(kernels).shape[0]
        int knr = kernels[0].shape[0]
        int knc = kernels[0].shape[1]
        DTYPE_float32_t[:, :] gkernel
        int bcr, bcc

    for i from 0 <= i < rows-scales_block by blk:
        for j from 0 <= j < cols-scales_block by blk:
            for ki in range(0, scale_length):

                k = scs[ki]

                k_half = k / 2

                ch_bd = chBd[i+scales_half-k_half:i+scales_half-k_half+k,
                             j+scales_half-k_half:j+scales_half-k_half+k]

                bcr = ch_bd.shape[0]
                bcc = ch_bd.shape[1]

                ch_bd_gabor = np.zeros((bcr, bcc), dtype='float32')

                for kl in range(0, n_kernels):

                    gkernel = kernels[kl]
                    ch_bd_gabor_c = ch_bd_gabor.copy()

                    with nogil:

                        _convolution(ch_bd, gkernel, bcr, bcc, knr, knc, ch_bd_gabor_c)

                        out_list[pix_ctr] = _get_mean(ch_bd_gabor_c, bcr, bcc)
                        pix_ctr += 1

                        out_list[pix_ctr] = _get_var(ch_bd_gabor_c, bcr, bcc)
                        pix_ctr += 1

    return np.float32(out_list)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def feature_gabor(np.ndarray chBd, int blk, list scs, int end_scale, list kernels):

    cdef:
        Py_ssize_t i, j, ki, kl
        int scales_half = int(end_scale / 2)
        int scales_block = end_scale - blk
        int out_len = 0
        int rows = chBd.shape[0]
        int cols = chBd.shape[1]
        DTYPE_uint16_t[:] scales_array
        int scale_length
        int n_kernels = len(kernels)

    if scs[0] < 16:
        scales_array = np.array(scs[1:], dtype='uint16')
    else:
        scales_array = np.array(scs, dtype='uint16')

    scale_length = scales_array.shape[0]

    for i from 0 <= i < rows-scales_block by blk:
        for j from 0 <= j < cols-scales_block by blk:
            for ki in range(0, scale_length):
                for kl in range(0, n_kernels):
                    out_len += 2

    return _feature_gabor(np.float32(chBd), blk, scales_array, out_len, scales_half,
                          scales_block, kernels, rows, cols, scale_length)


# Histogram of Oriented Gradients


# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
# cdef np.ndarray[DTYPE_float32_t, ndim=1] calc_hog(np.ndarray[DTYPE_float32_t, ndim=2] mag_chunk,
#                                                   np.ndarray[DTYPE_float32_t, ndim=2] ang_chunk,
#                                                   DTYPE_float32_t pi2, int bin_n,
#                                                   Py_ssize_t block_rows, Py_ssize_t block_cols):
#
#     # Quantizing bin values
#     cdef np.ndarray[DTYPE_uint16_t, ndim=2] bins = np.uint16(bin_n * ang_chunk / pi2)
#
#     return np.float32(np.bincount(np.array(bins).ravel(), weights=mag_chunk.ravel(), minlength=bin_n))
#
#
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[DTYPE_float32_t, ndim=1] _feature_hog(np.ndarray[DTYPE_float32_t, ndim=2] chbd,
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
        Py_ssize_t i, j, ki, sti, block_rows, block_cols
        DTYPE_uint16_t k, k_half
        np.ndarray[DTYPE_float32_t, ndim=2] ch_bd
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

                ch_bd = chbd[i+scales_half-k_half:i+scales_half-k_half+k,
                             j+scales_half-k_half:j+scales_half-k_half+k]

                block_rows = ch_bd.shape[0]
                block_cols = ch_bd.shape[1]

                if _get_max_f2d(ch_bd, block_rows, block_cols) > 0:

                    sts = get_moments(np.float32(HOG(ch_bd,
                                                     pixels_per_cell=(block_rows, block_cols),
                                                     cells_per_block=(1, 1))))

                    for sti in xrange(0, 7):

                        out_list[pix_ctr] = sts[sti]

                        pix_ctr += 1

                else:
                    pix_ctr += 7

    return np.float32(out_list)


@cython.boundscheck(False)
@cython.wraparound(False)
def feature_hog(np.ndarray chbd,
                int blk, list scs, int end_scale):

    cdef:
        Py_ssize_t i, j, ki
        int scales_half = int(end_scale / 2)
        int scales_block = end_scale - blk
        int out_len = 0
        int rows = chbd.shape[0]
        int cols = chbd.shape[1]
        DTYPE_uint16_t[:] scales_array = np.array(scs, dtype='uint16')
        int scale_length = scales_array.shape[0]

    for i from 0 <= i < rows-scales_block by blk:
        for j from 0 <= j < cols-scales_block by blk:
            for ki in xrange(0, scale_length):
                out_len += 7

    return _feature_hog(np.float32(chbd), blk, scales_array, end_scale,
                        scales_half, scales_block, out_len, rows, cols, scale_length)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _extract_values(DTYPE_uint8_t[:, :] block, DTYPE_uint16_t[:] values,
                          DTYPE_uint16_t[:, :] rc_, int fl) nogil:

    cdef:
        Py_ssize_t fi, fi_, fj_

    for fi in range(0, fl):

        fi_ = rc_[0, fi]
        fj_ = rc_[1, fi]

        values[fi] = block[fi_, fj_]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef Py_ssize_t _get_direction(DTYPE_uint8_t[:, :] chunk, int chunk_shape,
                               int rows_half, int cols_half, DTYPE_float32_t center_mean,
                               DTYPE_float32_t thresh_hom, DTYPE_float32_t[:] values_,
                               Py_ssize_t t_value, bint is_row,
                               DTYPE_float32_t[:] hist_, Py_ssize_t hist_counter,
                               int skip_factor):

    cdef:
        Py_ssize_t ija, rc_shape, lni
        DTYPE_uint16_t[:, :] rc
        DTYPE_float32_t ph_i, line_sd, lni_f
        DTYPE_float32_t alpha_ = .1
        DTYPE_float32_t sfs_max, sfs_min, sfs_psi, sfs_w_mean
        DTYPE_uint16_t[:] line_values

    # Iterate over every other angle
    for ija from 0 <= ija < chunk_shape by skip_factor:

        ph_i = 0.

        # Draw a line between the two endpoints.
        if is_row:
            rc = draw_line(rows_half, cols_half, ija, t_value)
        else:
            rc = draw_line(rows_half, cols_half, t_value, ija)

        rc_shape = rc.shape[1]
        line_values = rc[0].copy()

        with nogil:

            # Extract the values along the line.
            _extract_values(chunk, line_values, rc, rc_shape)

            # Get the standard deviation along the line.
            line_sd = _get_std_1d_uint16(line_values, rc_shape)

            # Iterate over line values.
            for lni in range(0, rc_shape):

                if ph_i < thresh_hom:
                    ph_i += abs_f(center_mean - float(line_values[lni]))
                else:
                    break

            # Get the line statistics
            lni_f = float(lni)

            sfs_max = _get_max_sample(values_[0], lni_f)
            sfs_min = _get_min_sample_f(values_[1], lni_f)
            sfs_psi = lni_f
            sfs_w_mean = (alpha_ * (lni_f - 1.)) / line_sd

            # Update the histogram with
            #   the line length.
            hist_[hist_counter] = lni_f

            hist_counter += 1

        # if not npy_isnan(sfs_max) and not npy_isinf(sfs_max):
            values_[0] = sfs_max

        # if not npy_isnan(sfs_min) and not npy_isinf(sfs_min):
            values_[1] = sfs_min

        # if not npy_isnan(sfs_psi) and not npy_isinf(sfs_psi):
            values_[2] += sfs_psi

        # if not npy_isnan(sfs_w_mean) and not npy_isinf(sfs_w_mean):
            values_[3] += sfs_w_mean

    return hist_counter


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _get_directions(DTYPE_uint8_t[:, :] chunk, int chunk_rws, int chunk_cls,
                          int rows_half, int cols_half, DTYPE_float32_t center_mean,
                          DTYPE_float32_t thresh_hom, DTYPE_float32_t[:] values,
                          int skip_factor):

    """
    Returns:
        1: Histogram maximum
        2: Histogram minimum
        3: PSI: Histogram mean
        4: Histogram w-mean
        5: Histogram standard deviation
        6: Maximum ratio of orthogonal angles
        7: Minimum ratio of orthogonal angles
    """

    cdef:
        Py_ssize_t i_, j_, iia_, ija_, ia, ja, rr_shape, lni
        DTYPE_intp_t[:] rr, cc
        DTYPE_uint8_t[:] line_values
        DTYPE_float32_t ph_i
        DTYPE_float32_t total_count
        Py_ssize_t hist_length = 0
        Py_ssize_t hist_counter = 0
        Py_ssize_t hist_counter_, ofc
        DTYPE_float32_t[:] hist
        DTYPE_float32_t max_diff, orthog_diff

    with nogil:

        # Get the histogram and row and column skip lengths.
        for i_ in range(0, 2):
            for iia_ from 0 <= iia_ < chunk_rws by skip_factor:
                hist_length += 1

        for j_ in range(0, 2):
            for ija_ from 0 <= ija_ < chunk_cls by skip_factor:
                hist_length += 1

    # Create the histogram
    hist = np.zeros(hist_length, dtype='float32')

    values[1] = 999999.

    # Fill the histogram

    # Rows, 1st column
    hist_counter = _get_direction(chunk, chunk_rws, rows_half, cols_half,
                                  center_mean, thresh_hom, values, 0, True,
                                  hist, hist_counter, skip_factor)

    # Rows, last column
    hist_counter = _get_direction(chunk, chunk_rws, rows_half, cols_half,
                                  center_mean, thresh_hom, values, chunk_cls-1, True,
                                  hist, hist_counter, skip_factor)

    # Columns, 1st row
    hist_counter = _get_direction(chunk, chunk_cls, rows_half, cols_half,
                                  center_mean, thresh_hom, values, 0, False,
                                  hist, hist_counter, skip_factor)

    # Columns, last row
    hist_counter = _get_direction(chunk, chunk_cls, rows_half, cols_half,
                                  center_mean, thresh_hom, values, chunk_rws-1, False,
                                  hist, hist_counter, skip_factor)

    total_count = _get_sum1d(hist, hist_length)

    values[3] /= total_count # H w-mean

    # Calculate the standard deviation
    #   of the histogram.
    values[4] = _get_std_1d(hist, hist_length)

    # Calculate the min orthogonal ratio.
    max_diff = 0.
    hist_counter_ = 0

    ofc = (chunk_rws * 2) - 1
    for iia_ from 0 <= iia_ < chunk_rws by skip_factor:
        # Ratio of orthogonal angles
        orthog_diff = abs_f(hist[iia_] - float((ofc - hist_counter_)))
        max_diff = _get_max_sample(max_diff, orthog_diff)
        hist_counter_ += 1

    ofc = (chunk_cls * 2) - 1
    for iia_ from 0 <= iia_ < chunk_cls by skip_factor:
        # Ratio of orthogonal angles
        orthog_diff = abs_f(hist[iia_] - float((ofc - hist_counter_)))
        max_diff = _get_max_sample(max_diff, orthog_diff)
        hist_counter_ += 1

    values[5] = max_diff


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef DTYPE_float32_t[:] _sfs_feas(DTYPE_uint8_t[:, :] chunk, int blk_size, DTYPE_float32_t thresh_hom,
                                  int skip_factor):

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
        int chunk_rws, chunk_cls, rows_half, cols_half, blk_half
        DTYPE_float32_t ctr_blk_mean, sfs_value
        DTYPE_float32_t[:] sfs_values = np.zeros(6, dtype='float32')
        DTYPE_uint8_t[:, :] chunk_block
        Py_ssize_t cbr, cbc

    # get chunk size
    chunk_rws = chunk.shape[0]
    chunk_cls = chunk.shape[1]

    rows_half = <int>(chunk_rws / 2)
    cols_half = <int>(chunk_cls / 2)

    blk_half = <int>(blk_size / 2)

    # get the center block average
    chunk_block = chunk[rows_half-blk_half:rows_half+blk_half, cols_half-blk_half:cols_half+blk_half]

    cbr = chunk_block.shape[0]
    cbc = chunk_block.shape[1]

    ctr_blk_mean = _get_mean_uint8(chunk_block, cbr, cbc)

    _get_directions(chunk, chunk_rws, chunk_cls, rows_half, cols_half,
                    ctr_blk_mean, thresh_hom, sfs_values, skip_factor)

    return sfs_values


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[DTYPE_float32_t, ndim=1] _feature_sfs(DTYPE_uint8_t[:, :] ch_bd, int blk,
                                                      DTYPE_uint16_t[:] scales_array, int n_scales,
                                                      DTYPE_float32_t thresh_hom,
                                                      int scales_half, int scales_block, int out_len,
                                                      int rows, int cols, int skip_factor):

    cdef:
        Py_ssize_t i, j, ki, k_half, st_
        DTYPE_uint16_t k
        DTYPE_uint8_t[:, :] ch_bd_
        DTYPE_float32_t[:] sts
        DTYPE_float32_t[:] out_list = np.empty(out_len, dtype='float32')
        int pix_ctr = 0

    for i from 0 <= i < rows-scales_block by blk:
        for j from 0 <= j < cols-scales_block by blk:
            for ki in range(0, n_scales):

                k = scales_array[ki]

                k_half = <int>(k / 2)

                ch_bd_ = ch_bd[i+scales_half-k_half:i+scales_half-k_half+k,
                               j+scales_half-k_half:j+scales_half-k_half+k]

                sts = _sfs_feas(ch_bd_, blk, thresh_hom, skip_factor)

                for st_ in range(0, 6):

                    out_list[pix_ctr] = sts[st_]

                    pix_ctr += 1

    return np.float32(out_list)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def feature_sfs(np.ndarray[DTYPE_uint8_t, ndim=2] chBd, int blk, list scs, int end_scale,
                DTYPE_float32_t thresh_hom, int skip_factor=4):

    cdef:
        Py_ssize_t i, j, ki, k
        int scales_half = int(end_scale / 2)
        int scales_block = end_scale - blk
        int out_len = 0
        int rows = chBd.shape[0]
        int cols = chBd.shape[1]
        DTYPE_uint16_t[:] scales_array = np.array(scs, dtype='uint16')
        int n_scales = scales_array.shape[0]

    with nogil:

        for i from 0 <= i < rows-scales_block by blk:
            for j from 0 <= j < cols-scales_block by blk:
                for ki in range(0, n_scales):
                    out_len += 6

    return _feature_sfs(chBd, blk, scales_array, n_scales, thresh_hom, scales_half,
                        scales_block, out_len, rows, cols, skip_factor)


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
def fill_key_points(DTYPE_float32_t[:, :] in_block, list key_point_list):

    cdef:
        Py_ssize_t key_point_index, key_y_idx, key_x_idx
        Py_ssize_t n_key_points = len(key_point_list)
        int brows = in_block.shape[0]
        int bcols = in_block.shape[1]
        DTYPE_uint8_t[:, :] key_point_array = np.empty((brows, bcols), dtype='uint8')

    for key_point_index in range(0, n_key_points):

        key_x, key_y = key_point_list[key_point_index].pt

        key_y_idx = int(floor(key_y))
        key_x_idx = int(floor(key_x))

        key_point_array[key_y_idx, key_x_idx] = 255

    return np.uint8(key_point_array)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef DTYPE_float32_t[:] _pyramid_hist_sift(DTYPE_uint8_t[:, :] key_point_array,
                                           DTYPE_float32_t[:] levels,
                                           int orb_rows, int orb_cols):

    cdef:
        Py_ssize_t lv, ki, kj, grid_counter
        int rr_rows, cc_cols, y_tiles, x_tiles
        DTYPE_float32_t[:] hist
        Py_ssize_t counter = 0
        DTYPE_uint8_t[:, :] kblock

    with nogil:

        # Iterate over each level
        for lv in range(0, 3):

            y_tiles = <int>(floor(orb_rows / levels[lv]))
            x_tiles = <int>(floor(orb_cols / levels[lv]))

            if (y_tiles > 1) and (x_tiles > 1):

                for ki from 0 <= ki < orb_rows-1 by y_tiles:
                    rr_rows = n_rows_cols(ki, y_tiles, orb_rows)
                    if rr_rows > 1:
                        for kj from 0 <= kj < orb_cols-1 by x_tiles:
                            cc_cols = n_rows_cols(kj, x_tiles, orb_cols)
                            if cc_cols > 1:
                                counter += 1

    hist = np.zeros(counter, dtype='float32')

    with nogil:

        grid_counter = 0

        # Iterate over each level
        for lv in range(0, 3):

            y_tiles = <int>(floor(orb_rows / levels[lv]))
            x_tiles = <int>(floor(orb_cols / levels[lv]))

            if (y_tiles > 1) and (x_tiles > 1):

                for ki from 0 <= ki < orb_rows-1 by y_tiles:

                    rr_rows = n_rows_cols(ki, y_tiles, orb_rows)

                    if rr_rows > 1:

                        for kj from 0 <= kj < orb_cols-1 by x_tiles:

                            cc_cols = n_rows_cols(kj, x_tiles, orb_cols)

                            if cc_cols > 1:

                                # Get the keypoint block
                                kblock = key_point_array[ki:ki+rr_rows, kj:kj+cc_cols]

                                # Enter the keypoint sum into the histogram
                                hist[grid_counter] += _get_sum_uint8(kblock, rr_rows, cc_cols)

                                grid_counter += 1

    return hist


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef DTYPE_float32_t[:] _feature_orb(DTYPE_uint8_t[:, :] ch_bd,
                                     int blk,
                                     DTYPE_uint16_t[:] scales_array,
                                     int scales_half, int scales_block,
                                     int scale_length, int out_len,
                                     int rows, int cols, int scales_length):

    cdef:
        Py_ssize_t i, j, ki, st
        DTYPE_uint16_t k
        int k_half
        DTYPE_float32_t[:] sts
        DTYPE_float32_t[:] out_list = np.zeros(out_len, dtype='float32')
        DTYPE_float32_t[:] levels = np.array([2, 4, 8], dtype='float32')
        Py_ssize_t pix_ctr = 0
        int block_rows, block_cols
        DTYPE_uint8_t[:, :] ch_bd_sub

    for i from 0 <= i < rows-scales_block by blk:
        for j from 0 <= j < cols-scales_block by blk:
            for ki in range(0, scale_length):

                k = scales_array[ki]

                k_half = <int>(k / 2)

                ch_bd_sub = ch_bd[i+scales_half-k_half:i+scales_half-k_half+k,
                                  j+scales_half-k_half:j+scales_half-k_half+k]

                block_rows = ch_bd_sub.shape[0]
                block_cols = ch_bd_sub.shape[1]

                if _get_max(ch_bd_sub, block_rows, block_cols) > 0:

                    sts = get_moments(_pyramid_hist_sift(ch_bd_sub, levels, block_rows, block_cols))

                    for st in range(0, 7):

                        out_list[pix_ctr] = sts[st]

                        pix_ctr += 1

                else:
                    pix_ctr += 7

    return out_list


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def feature_orb(DTYPE_uint8_t[:, :] ch_bd,
                int blk, list scs, int end_scale):

    cdef:
        Py_ssize_t i, j, ki
        int scales_half = end_scale / 2
        int scales_block = end_scale - blk
        int out_len = 0
        int rows = ch_bd.shape[0]
        int cols = ch_bd.shape[1]
        DTYPE_uint16_t[:] scales_array = np.array(scs, dtype='uint16')
        int scale_length = scales_array.shape[0]

    with nogil:

        for i from 0 <= i < rows-scales_block by blk:
            for j from 0 <= j < cols-scales_block by blk:
                for ki in range(0, scale_length):
                    out_len += 7

    return np.float32(_feature_orb(ch_bd, blk, scales_array,
                                   scales_half, scales_block, scale_length,
                                   out_len, rows, cols, scale_length))


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
cdef np.ndarray[DTYPE_float32_t, ndim=1] _feature_lbp(np.ndarray[DTYPE_uint8_t, ndim=2] chBd,
                                                      int blk, list scs, int end_scale):

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
        np.ndarray[DTYPE_float32_t, ndim=1] out_list_a

    # get the LBP images
    lbpBd, p_range = _set_lbp(chBd, rows, cols)

    # count of bins for all p,r LBP pairs
    pr_bin_count = np.sum([pr+2 for pr in p_range])

    for i from 0 <= i < rows-scales_block by blk:
        for j from 0 <= j < cols-scales_block by blk:
            for ki in xrange(0, scale_length):
                out_len += pr_bin_count

    # set the output list
    out_list = np.empty(out_len, dtype='uint8')

    for i from 0 <= i < rows-scales_block by blk:
        for j from 0 <= j < cols-scales_block by blk:
            for ki in range(0, scale_length):

                k = scales_array[ki]

                k_half = k / 2

                ch_bd = lbpBd[:, i+scales_half-k_half:i+scales_half-k_half+k,
                              j+scales_half-k_half:j+scales_half-k_half+k]

                # get histograms and concatenate
                sts = np.concatenate([np.bincount(ch_bd[p_range.index(pc)].flat, minlength=pc+2)
                                      for pc in p_range]).astype(np.uint8)

                for sti in range(0, 4):

                    out_list[pix_ctr] = sts[sti]

                    pix_ctr += 1

    out_list_a = np.float32(out_list)

    out_list_a[np.isnan(out_list_a) | np.isinf(out_list_a)] = 0

    return out_list_a


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
                out_len += 7

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

                for sti in xrange(0, 7):

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
cdef void _glcm_loop(DTYPE_uint8_t[:, :] image, DTYPE_float32_t[:] distances,
                     DTYPE_float32_t[:] angles, int levels,
                     DTYPE_float32_t[:, :, :, ::1] out,
                     DTYPE_float32_t[:, :] out_sums,
                     Py_ssize_t rows, Py_ssize_t cols) nogil:

    cdef:
        Py_ssize_t a_idx, d_idx, r, c, row, col
        Py_ssize_t angles_, distances_
        DTYPE_uint8_t i, j
        DTYPE_float32_t angle, distance

    angles_ = angles.shape[0]
    distances_ = distances.shape[0]

    for a_idx in range(0, angles_):

        angle = angles[a_idx]

        for d_idx in range(0, distances_):

            distance = distances[d_idx]

            # Iterate over the image to get
            #   the grey-level pairs.
            for r in range(0, rows):

                for c in range(0, cols):

                    # Current row pixel value
                    i = image[r, c]

                    # compute the location of the offset pixel
                    row = r + <int>(roundd(sin(angle) * distance))
                    col = c + <int>(roundd(cos(angle) * distance))

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
cdef void _norm_glcm(DTYPE_float32_t[:, :, :, :] Pt,
                     DTYPE_float32_t[:, :] Pt_sums, DTYPE_float32_t[:] distances,
                     DTYPE_float32_t[:] angles, int levels,
                     DTYPE_float32_t[:, :, :, :] glcm_normed_) nogil:

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
                                                 DTYPE_float32_t[:, :, :, ::1] glcm_normed) nogil:

    # count co-occurences
    _glcm_loop(image, distances, angles, levels, P, angle_dist_sums, rows, cols)

    # Normalize the matrix
    _norm_glcm(P, angle_dist_sums, distances, angles, levels, glcm_normed)

    return glcm_normed


@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_float32_t _glcm_contrast(DTYPE_float32_t[:, :, :, ::1] P,
                                    DTYPE_float32_t[:] distances,
                                    DTYPE_float32_t[:] angles, Py_ssize_t levels,
                                    DTYPE_float32_t[:, :] contrast_array) nogil:

    cdef:
        Py_ssize_t a_idx, d_idx, r, c
        Py_ssize_t angles_, distances_
        DTYPE_float32_t min_contrast = 1000000.
        DTYPE_float32_t contrast_sum

    angles_ = angles.shape[0]
    distances_ = distances.shape[0]

    for a_idx in range(0, angles_):

        for d_idx in range(0, distances_):

            # Sum the contrast for the current angle/distance pair.
            contrast_sum = 0.

            # Iterate over the co-occurrence matrix
            #   and get the contrast.
            for r in range(0, levels):

                for c in range(0, levels):
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
        DTYPE_uint16_t k
        int k_half
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
        DTYPE_float32_t[:, :, :, ::1] P_c
        DTYPE_float32_t[:, :] angle_dist_sums_c
        DTYPE_float32_t[:, :, :, ::1] glcm_normed_c
        DTYPE_float32_t[:] mean_var_values
        list weights_list = []
        DTYPE_float32_t[:, :] kernel_weight
        DTYPE_float32_t[:] in_zs = np.zeros(2, dtype='float32')
        DTYPE_float32_t[:] in_zs_c

    if weighted:

        for ki in range(0, scale_length):
            k = scs[ki]
            k_half = int(k / 2)
            rs = (scales_half - k_half + k) - (scales_half - k_half)
            cs = (scales_half - k_half + k) - (scales_half - k_half)
            weights_list.append(_create_weights(rs, cs))

        for i from 0 <= i < rows-scales_block by blk:
            for j from 0 <= j < cols-scales_block by blk:
                for ki in xrange(0, scale_length):

                    k = scs[ki]

                    k_half = int(k / 2)

                    ch_bd = chBd[i+scales_half-k_half:i+scales_half-k_half+k,
                                 j+scales_half-k_half:j+scales_half-k_half+k]

                    block_rows = ch_bd.shape[0]
                    block_cols = ch_bd.shape[1]

                    if _get_max(ch_bd, block_rows, block_cols) == 0:
                        con_min = 0.
                    else:

                        P_c = P_.copy()
                        angle_dist_sums_c = angle_dist_sums_.copy()
                        glcm_normed_c = glcm_normed_.copy()

                        with nogil:

                            glcm_mat = _greycomatrix(ch_bd, dists, disp_vect, levels, block_rows, block_cols,
                                                     P_c, angle_dist_sums_c, glcm_normed_c)

                            con_min = _glcm_contrast(glcm_mat, dists, disp_vect, levels, contrast_weights)

                    kernel_weight = weights_list[ki]
                    in_zs_c = in_zs.copy()

                    mean_var_values = _get_weighted_mean_var(np.float32(ch_bd), kernel_weight,
                                                             block_rows, block_cols, in_zs_c)

                    out_list[pix_ctr] = con_min * mean_var_values[0]

                    pix_ctr += 1

    else:

        for i from 0 <= i < rows-scales_block by blk:
            for j from 0 <= j < cols-scales_block by blk:
                for ki in xrange(0, scale_length):

                    k = scs[ki]

                    k_half = int(k / 2)

                    ch_bd = chBd[i+scales_half-k_half:i+scales_half-k_half+k,
                                 j+scales_half-k_half:j+scales_half-k_half+k]

                    block_rows = ch_bd.shape[0]
                    block_cols = ch_bd.shape[1]

                    if _get_max(ch_bd, block_rows, block_cols) == 0:
                        con_min = 0.
                    else:

                        P_c = P_.copy()
                        angle_dist_sums_c = angle_dist_sums_.copy()
                        glcm_normed_c = glcm_normed_.copy()

                        with nogil:

                            glcm_mat = _greycomatrix(ch_bd, dists, disp_vect, levels, block_rows, block_cols,
                                                     P_c, angle_dist_sums_c, glcm_normed_c)

                            con_min = _glcm_contrast(glcm_mat, dists, disp_vect, levels, contrast_weights)

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
        Py_ssize_t i, j, ki, pix_ctr, pi
        unsigned int bcr, bcc
        DTYPE_uint16_t k
        int k_half
        DTYPE_float32_t[:] out_list = np.zeros(out_len, dtype='float32')
        DTYPE_float32_t[:] in_zs = np.zeros(2, dtype='float32')
        DTYPE_float32_t[:] out_values
        int rows = ch_bd.shape[0]
        int cols = ch_bd.shape[1]
        DTYPE_float32_t[:, :] block_chunk
        list weights_list = []

    for ki in range(0, scale_length):
        k = scs[ki]
        k_half = <int>(k / 2)
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

                out_values = _get_weighted_mean_var(block_chunk, weights_list[ki], bcr, bcc, in_zs.copy())

                for pi in xrange(0, 2):

                    out_list[pix_ctr] = out_values[pi]

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
                out_len += 2

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

        int boxes_max = int(ceil(float(maxi) / _get_min_sample_i(rr_rows, rr_cols)))
        int boxes_min = int(ceil(float(mini) / _get_min_sample_i(rr_rows, rr_cols)))

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
