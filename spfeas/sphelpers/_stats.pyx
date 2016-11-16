#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

import sys

import cython
cimport cython

import numpy as np
cimport numpy as np

from libc.math cimport pow, atan, sqrt
# from libc.math cimport isnan, isinf
from cython.parallel import parallel, prange

from spatial_pyramid_hist import pyramid_hist_sift
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
    from skimage.feature import greycomatrix, greycoprops
    from skimage.transform import probabilistic_hough_line as PHL
except:
    raise ImportError('Skimage.feature did not load')

import pyximport
pyximport.install(setup_args={'include_dirs': [np.get_include()]})

old_settings = np.seterr(all='ignore')

DTYPE_int = np.int
ctypedef np.int_t DTYPE_int_t

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

cdef extern from "math.h":
    DTYPE_float32_t ceil(DTYPE_float32_t x)


cdef inline int n_rows_cols(int pixel_index, int rows_cols, int block_size):
    return rows_cols if pixel_index + rows_cols < block_size else block_size - pixel_index


cdef inline int _get_min_sample_i(int s1, int s2):
    return s2 if s2 < s1 else s1


cdef inline DTYPE_float32_t _get_min_sample_f(DTYPE_float32_t s1, DTYPE_float32_t s2):
    return s2 if s2 < s1 else s1


cdef inline DTYPE_float32_t _get_max_sample(DTYPE_float32_t s1, DTYPE_float32_t s2):
    return s2 if s2 > s1 else s1


cdef inline DTYPE_uint8_t _get_max_sample_int(DTYPE_uint8_t s1, DTYPE_uint8_t s2):
    return s2 if s2 > s1 else s1


cdef inline DTYPE_uint8_t _get_min_sample_int(DTYPE_uint8_t s1, DTYPE_uint8_t s2):
    return s2 if s2 < s1 else s1


@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_float32_t _get_min_f(DTYPE_float32_t[:] in_row, int cols):

    cdef:
        Py_ssize_t a
        DTYPE_float32_t m = 100000000.

    for a in xrange(0, cols):

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
cdef int _get_max(DTYPE_uint8_t[:, :] block, int rs, int cs):

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

    for i in prange(1, samps, nogil=True, num_threads=samps, schedule='static'):
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
cdef DTYPE_float32_t _get_mean_uint8(DTYPE_uint8_t[:, :] block, int rs, int cs):

    cdef:
        DTYPE_float32_t n_samps = float(rs*cs)

    return _get_sum_uint8(block, rs, cs) / n_samps


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
                                                        DTYPE_uint8_t[:] scs, int out_len, int scales_half,
                                                        int scales_block, list kernels, int rows, int cols,
                                                        int scale_length):

    """
    Returns at each scale at each kernel

    1:	Mean
    2:	Variance
    """

    cdef:
        Py_ssize_t i, j, ki, kl
        unsigned int k, k_half
        DTYPE_uint8_t[:, :] ch_bd
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

                    out_list[pix_ctr] = _get_mean_uint8(ch_bd, bcr, bcc)
                    pix_ctr += 1
                    out_list[pix_ctr] = _get_var_uint8(ch_bd, bcr, bcc)
                    pix_ctr += 1

    return np.asarray(out_list).astype(np.float32)


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
        DTYPE_uint8_t[:] scales_array = np.array(scs, dtype='uint8')
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


# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef np.ndarray[DTYPE_float32_t, ndim=1] feaHoG(np.ndarray[DTYPE_uint8_t, ndim=2] chBd, int blk, list scs):
#
#     """
#     Get the Histogram of Oriented Gradients
#
#     Returns at each scale
#     ---------------------
#     1:	Mean
#     2:	Variance
#     3:	Skew
#     4:	Kurtosis
#     """
#
#     cdef:
#         Py_ssize_t i, j
#         int k, the_min, k_half
#         int rows = chBd.shape[0]
#         int cols = chBd.shape[1]
#         int blk_rws, blk_cls
#         np.ndarray[DTYPE_uint8_t, ndim=2] ch_bd
#         np.ndarray[DTYPE_float64_t, ndim=1] hogArr
#         list sts
#         DTYPE_float64_t st
#         int scales_half = np.max(scs) / 2
#         int scales_blk = np.max(scs) - blk
#         np.ndarray[DTYPE_float32_t, ndim=1] out_list
#         int out_len = 0
#         int pix_ctr = 0
#
#     for i in xrange(0, rows-scales_blk, blk):
#         for j in xrange(0, cols-scales_blk, blk):
#             for k in scs:
#                 out_len += 4
#
#     # set the output list
#     out_list = np.empty(out_len).astype(np.float32)
#
#     for i in xrange(0, rows-scales_blk, blk):
#         for j in xrange(0, cols-scales_blk, blk):
#             for k in scs:
#
#                 k_half = k / 2
#
#                 ch_bd = chBd[i+scales_half-k_half:i+scales_half-k_half+k, j+scales_half-k_half:j+scales_half-k_half+k]
#
#                 blk_rws = ch_bd.shape[0]
#                 blk_cls = ch_bd.shape[1]
#
#                 the_min = _get_min_sample_i(blk_rws, blk_cls)
#
#                 hogArr = HOG(ch_bd, orientations=8, pixels_per_cell=(the_min, the_min), cells_per_block=(1, 1))
#
#                 sts = get_moments(hogArr)
#
#                 for st in sts:
#                     out_list[pix_ctr] = st
#
#                     pix_ctr += 1
#
#     return out_list


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[DTYPE_float32_t, ndim=1] calc_hog(np.ndarray[DTYPE_float32_t, ndim=2] mag_chunk,
                                                  np.ndarray[DTYPE_float32_t, ndim=2] ang_chunk,
                                                  DTYPE_float32_t pi2, int bin_n):

    # quantizing binvalues
    cdef np.ndarray[DTYPE_uint16_t, ndim=2] bins = (bin_n * ang_chunk / pi2).astype(np.uint16)

    return np.bincount(np.array(bins).ravel(), weights=np.array(mag_chunk).ravel(), minlength=bin_n).astype(np.float32)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[DTYPE_float32_t, ndim=1] _feature_hog(DTYPE_float32_t[:, :] grad,
                                                      DTYPE_float32_t[:, :] ori,
                                                      int blk, DTYPE_uint8_t[:] scs, int end_scale, int scales_half,
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
        unsigned int k, k_half
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

    return np.asarray(out_list).astype(np.float32)


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
        DTYPE_uint8_t[:] scales_array = np.array(scs, dtype='uint8')
        int scale_length = scales_array.shape[0]

    for i from 0 <= i < rows-scales_block by blk:
        for j from 0 <= j < cols-scales_block by blk:
            for ki in xrange(0, scale_length):
                out_len += 4

    return _feature_hog(grad, ori, blk, scales_array, end_scale,
                        scales_half, scales_block, out_len, rows, cols, scale_length)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[long, ndim=1] obscure_angle_indices(np.ndarray[long, ndim=1] arr, str do_m, fl):

    cdef int arr_len = len(arr)
    cdef int ctr = 2
    cdef int s = 0
    cdef int t
    cdef np.ndarray[DTYPE_uint8_t, ndim=1] l = np.zeros(arr_len).astype(np.uint8)

    for t in xrange(0, arr_len):

        l[t] = s
        if ctr == 2:
            s += 1
            ctr = 0

        ctr += 1

    if do_m == 'add':
        if fl:
            return np.add(arr, l)[::-1]
        else:
            return np.add(arr, l)
    else:
        if fl:
            return np.subtract(arr, l)[::-1]
        else:
            return np.subtract(arr, l)


@cython.boundscheck(False)
cdef list get_directions(np.ndarray[DTYPE_uint8_t, ndim=2] chunk, int chunk_rws, int chunk_cls, int rows_half, \
                         int cols_half, int blk_adj, int n_angles=8):

    cdef:
        list dir_arrs = []
        int ang_pos, pos_min
        list x_pos, y_pos

    if n_angles == 16:

        x_pos = [np.add(np.array(range(chunk_cls-cols_half-blk_adj)), cols_half+1).astype(int), \
                    np.add(np.array(range(chunk_cls-cols_half-blk_adj)), cols_half+1).astype(int), \
                    np.add(np.array(range(chunk_cls-cols_half-blk_adj)), cols_half+1).astype(int), \
                    obscure_angle_indices(np.add(np.array(range(chunk_cls-cols_half-blk_adj)), cols_half+1).astype(int), 'sub', False), \
                    np.array(range(cols_half-blk_adj, cols_half+blk_adj)).astype(int), \
                    obscure_angle_indices(np.array(range(chunk_cls-cols_half-blk_adj)).astype(int), 'add', True), \
                    np.array(range(chunk_cls-cols_half-blk_adj))[::-1].astype(int), \
                    np.array(range(chunk_cls-cols_half-blk_adj))[::-1].astype(int), \
                    np.array(range(chunk_cls-cols_half-blk_adj))[::-1].astype(int), \
                    np.array(range(chunk_cls-cols_half-blk_adj))[::-1].astype(int), \
                    np.array(range(chunk_cls-cols_half-blk_adj))[::-1].astype(int), \
                    obscure_angle_indices(np.array(range(chunk_cls-cols_half-blk_adj)).astype(int), 'add', True), \
                    np.array(range(chunk_cls-cols_half-blk_adj))[::-1].astype(int), \
                    obscure_angle_indices(np.add(np.array(range(chunk_cls-cols_half-blk_adj)), cols_half+1).astype(int), 'sub', False), \
                    np.array(range(cols_half-blk_adj, cols_half+blk_adj)).astype(int), \
                    np.add(np.array(range(chunk_cls-cols_half-blk_adj)), cols_half+1).astype(int), \
                    np.add(np.array(range(chunk_cls-cols_half-blk_adj)), cols_half+1).astype(int)]

        y_pos = [np.array(range(rows_half-blk_adj, rows_half+blk_adj)).astype(int), \
                    obscure_angle_indices(np.array(range(chunk_rws-rows_half-blk_adj))[::-1].astype(int), 'add', False), \
                    np.array(range(chunk_rws-rows_half-blk_adj))[::-1].astype(int), \
                    np.array(range(chunk_rws-rows_half-blk_adj))[::-1].astype(int), \
                    np.array(range(chunk_rws-rows_half-blk_adj))[::-1].astype(int), \
                    np.array(range(chunk_rws-rows_half-blk_adj))[::-1].astype(int), \
                    np.array(range(chunk_rws-rows_half-blk_adj))[::-1].astype(int), \
                    obscure_angle_indices(np.array(range(chunk_rws-rows_half-blk_adj))[::-1].astype(int), 'add', False), \
                    np.array(range(rows_half-blk_adj, rows_half+blk_adj)).astype(int), \
                    obscure_angle_indices(np.add(np.array(range(chunk_rws-rows_half-blk_adj)), rows_half+1).astype(int), 'sub', False), \
                    np.add(np.array(range(chunk_rws-rows_half-blk_adj)), rows_half+1).astype(int), \
                    np.add(np.array(range(chunk_rws-rows_half-blk_adj)), rows_half+1).astype(int), \
                    np.add(np.array(range(chunk_rws-rows_half-blk_adj)), rows_half+1).astype(int), \
                    np.add(np.array(range(chunk_rws-rows_half-blk_adj)), rows_half+1).astype(int), \
                    np.add(np.array(range(chunk_rws-rows_half-blk_adj)), rows_half+1).astype(int), \
                    obscure_angle_indices(np.add(np.array(range(chunk_rws-rows_half-blk_adj)), rows_half+1).astype(int), 'sub', False)]

        ang_pos = 0
        dir_arrs.append(chunk[y_pos[ang_pos][0]:y_pos[ang_pos][-1]+1, x_pos[ang_pos][0]:x_pos[ang_pos][-1]+1].mean(axis=0).astype(np.float32))

        ang_pos = 1
        try:
            dir_arrs.append(chunk[y_pos[ang_pos], x_pos[ang_pos]].astype(np.float32))
        except:
            pos_min = np.minimum(len(y_pos[ang_pos]), len(x_pos[ang_pos]))
            dir_arrs.append(chunk[y_pos[ang_pos][:pos_min], x_pos[ang_pos][:pos_min]].astype(np.float32))

        ang_pos = 2
        try:
            dir_arrs.append(chunk[y_pos[ang_pos], x_pos[ang_pos]].astype(np.float32))
        except:
            pos_min = np.minimum(len(y_pos[ang_pos]), len(x_pos[ang_pos]))
            dir_arrs.append(chunk[y_pos[ang_pos][:pos_min], x_pos[ang_pos][:pos_min]].astype(np.float32))

        ang_pos = 3
        try:
            dir_arrs.append(chunk[y_pos[ang_pos], x_pos[ang_pos]].astype(np.float32))
        except:
            pos_min = np.minimum(len(y_pos[ang_pos]), len(x_pos[ang_pos]))
            dir_arrs.append(chunk[y_pos[ang_pos][:pos_min], x_pos[ang_pos][:pos_min]].astype(np.float32))

        ang_pos = 4
        dir_arrs.append(np.flipud(chunk[y_pos[ang_pos][0]:y_pos[ang_pos][-1]+1, x_pos[ang_pos][0]:x_pos[ang_pos][-1]+1]).mean(axis=1).astype(np.float32))

        ang_pos = 5
        try:
            dir_arrs.append(chunk[y_pos[ang_pos], x_pos[ang_pos]].astype(np.float32))
        except:
            pos_min = np.minimum(len(y_pos[ang_pos]), len(x_pos[ang_pos]))
            dir_arrs.append(chunk[y_pos[ang_pos][:pos_min], x_pos[ang_pos][:pos_min]].astype(np.float32))

        ang_pos = 6
        try:
            dir_arrs.append(chunk[y_pos[ang_pos], x_pos[ang_pos]].astype(np.float32))
        except:
            pos_min = np.minimum(len(y_pos[ang_pos]), len(x_pos[ang_pos]))
            dir_arrs.append(chunk[y_pos[ang_pos][:pos_min], x_pos[ang_pos][:pos_min]].astype(np.float32))

        ang_pos = 7
        try:
            dir_arrs.append(chunk[y_pos[ang_pos], x_pos[ang_pos]].astype(np.float32))
        except:
            pos_min = np.minimum(len(y_pos[ang_pos]), len(x_pos[ang_pos]))
            dir_arrs.append(chunk[y_pos[ang_pos][:pos_min], x_pos[ang_pos][:pos_min]].astype(np.float32))

        ang_pos = 8
        dir_arrs.append(np.fliplr(chunk[y_pos[ang_pos][0]:y_pos[ang_pos][-1]+1, x_pos[ang_pos][0]:x_pos[ang_pos][-1]+1]).mean(axis=0).astype(np.float32))

        ang_pos = 9
        try:
            dir_arrs.append(chunk[y_pos[ang_pos], x_pos[ang_pos]].astype(np.float32))
        except:
            pos_min = np.minimum(len(y_pos[ang_pos]), len(x_pos[ang_pos]))
            dir_arrs.append(chunk[y_pos[ang_pos][:pos_min], x_pos[ang_pos][:pos_min]].astype(np.float32))

        ang_pos = 10
        try:
            dir_arrs.append(chunk[y_pos[ang_pos], x_pos[ang_pos]].astype(np.float32))
        except:
            pos_min = np.minimum(len(y_pos[ang_pos]), len(x_pos[ang_pos]))
            dir_arrs.append(chunk[y_pos[ang_pos][:pos_min], x_pos[ang_pos][:pos_min]].astype(np.float32))

        ang_pos = 11
        try:
            dir_arrs.append(chunk[y_pos[ang_pos], x_pos[ang_pos]].astype(np.float32))
        except:
            pos_min = np.minimum(len(y_pos[ang_pos]), len(x_pos[ang_pos]))
            dir_arrs.append(chunk[y_pos[ang_pos][:pos_min], x_pos[ang_pos][:pos_min]].astype(np.float32))

        ang_pos = 12
        dir_arrs.append(chunk[y_pos[ang_pos][0]:y_pos[ang_pos][-1]+1, x_pos[ang_pos][0]:x_pos[ang_pos][-1]+1].mean(axis=1).astype(np.float32))

        ang_pos = 13
        try:
            dir_arrs.append(chunk[y_pos[ang_pos], x_pos[ang_pos]].astype(np.float32))
        except:
            pos_min = np.minimum(len(y_pos[ang_pos]), len(x_pos[ang_pos]))
            dir_arrs.append(chunk[y_pos[ang_pos][:pos_min], x_pos[ang_pos][:pos_min]].astype(np.float32))

        ang_pos = 14
        try:
            dir_arrs.append(chunk[y_pos[ang_pos], x_pos[ang_pos]].astype(np.float32))
        except:
            pos_min = np.minimum(len(y_pos[ang_pos]), len(x_pos[ang_pos]))
            dir_arrs.append(chunk[y_pos[ang_pos][:pos_min], x_pos[ang_pos][:pos_min]].astype(np.float32))

        ang_pos = 15
        try:
            dir_arrs.append(chunk[y_pos[ang_pos], x_pos[ang_pos]].astype(np.float32))
        except:
            pos_min = np.minimum(len(y_pos[ang_pos]), len(x_pos[ang_pos]))
            dir_arrs.append(chunk[y_pos[ang_pos][:pos_min], x_pos[ang_pos][:pos_min]].astype(np.float32))

    else:

        x_pos = [np.add(np.array(range(chunk_cls-cols_half-blk_adj)), cols_half+1).astype(int), \
                    np.add(np.array(range(chunk_cls-cols_half-blk_adj)), cols_half+1).astype(int), \
                    np.array(range(cols_half-blk_adj, cols_half+blk_adj)).astype(int), \
                    np.array(range(chunk_cls-cols_half-blk_adj))[::-1].astype(int), \
                    np.array(range(chunk_cls-cols_half-blk_adj))[::-1].astype(int), \
                    np.array(range(chunk_cls-cols_half-blk_adj))[::-1].astype(int), \
                    np.array(range(cols_half-blk_adj, cols_half+blk_adj)).astype(int), \
                    np.add(np.array(range(chunk_cls-cols_half-blk_adj)), cols_half+1).astype(int)]

        y_pos = [np.array(range(rows_half-blk_adj, rows_half+blk_adj)).astype(int), \
                 np.array(range(chunk_rws-rows_half-blk_adj))[::-1].astype(int), \
                 np.array(range(chunk_rws-rows_half-blk_adj))[::-1].astype(int), \
                 np.array(range(chunk_rws-rows_half-blk_adj))[::-1].astype(int), \
                 np.array(range(rows_half-blk_adj, rows_half+blk_adj)).astype(int), \
                 np.add(np.array(range(chunk_rws-rows_half-blk_adj)), rows_half+1).astype(int), \
                 np.add(np.array(range(chunk_rws-rows_half-blk_adj)), rows_half+1).astype(int), \
                 np.add(np.array(range(chunk_rws-rows_half-blk_adj)), rows_half+1).astype(int)]

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

    return dir_arrs


@cython.boundscheck(False)
@cython.wraparound(False)
cdef list _sfs_feas(np.ndarray[DTYPE_uint8_t, ndim=2] chunk, int blk_size, DTYPE_float32_t cell_size, \
                    DTYPE_float32_t thresh_1, int n_angles):

    """
    Args:
        chunk (ndarray): chunk array to extract features sets from
        blk_size (int): block size of center pixels
        cell_size (float): image cell size, in meters
        thresh_1 (int): threshold for homogeneity

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
        dict blk_adjs
        int blk_adj, chunk_rws, chunk_cls, rows_half, cols_half, sur_ctr, blk_half, d_l
        DTYPE_float32_t ctr_blk_mean, PH_i
        list dir_lengths, dir_arrs, sfs_list
        np.ndarray[DTYPE_float32_t, ndim=1] dir_arr
        DTYPE_float32_t sfs_max, sfs_min, sfs_psi, sfs_mean, sfs_sd, sfs_value

    if chunk.max() == 0:
        return [0., 0., 0., 0., 0.]
    else:

        # block adjustments
        blk_adjs = {1: 0, 2: 1, 4: 2, 6: 3, 8: 4, 10: 5, 12: 6, 14: 7, 16: 8}
        blk_adj = blk_adjs[blk_size]

        # get chunk size
        chunk_rws = chunk.shape[0]
        chunk_cls = chunk.shape[1]

        rows_half = chunk_rws / 2
        cols_half = chunk_cls / 2

        blk_half = blk_size / 2

        # get the center block average
        ctr_blk_mean = cv2.mean(chunk[rows_half-blk_half:rows_half+blk_half, \
                                cols_half-blk_half:cols_half+blk_half].astype(np.float32))[0]

        # list for direction lengths
        dir_lengths = []

        dir_arrs = get_directions(chunk, chunk_rws, chunk_cls, rows_half, cols_half, blk_adj, n_angles)

        for dir_arr in dir_arrs:

            PH_i = 0.

            for sur_ctr in xrange(1, len(dir_arr)+1):

                if PH_i >= thresh_1:
                    break
                else:

                    PH_i += np.abs(ctr_blk_mean - dir_arr[sur_ctr-1])

            dir_lengths.append(sur_ctr-1.)

        dir_lengths = [dir_l * cell_size for dir_l in dir_lengths]

        # sfs_max = np.max(dir_lengths)
        sfs_max = _get_max_f(np.asarray(dir_lengths).astype(np.float32), len(dir_lengths))
        # sfs_min = np.min(dir_lengths)
        sfs_min = _get_min_f(np.asarray(dir_lengths).astype(np.float32), len(dir_lengths))
        # sfs_psi = np.sum(dir_lengths)
        sfs_psi = get_sum_1d(np.asarray(dir_lengths).astype(np.float32))
        # sfs_mean, sfs_std = cv2.meanStdDev(np.asarray(dir_lengths))
        # sfs_mean = np.mean(dir_lengths)
        sfs_mean = cv2.mean(np.asarray(dir_lengths).astype(np.float32))[0]
        sfs_sd = pow(get_sum_1d(np.asarray([pow((d_l - sfs_psi), 2) for d_l in dir_lengths]).astype(np.float32)), .5)

        sfs_list = [sfs_max, sfs_min, sfs_psi, sfs_mean, sfs_sd]

        return [sfs_value if not npy_isnan(sfs_value) else 0. for sfs_value in sfs_list]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[DTYPE_float32_t, ndim=1] _feature_sfs(np.ndarray[DTYPE_uint8_t, ndim=2] chBd, int blk, list scs,
                                                      DTYPE_float32_t cell_size, DTYPE_float32_t thresh_1,
                                                      int n_angles, int scales_half, int scales_block, int out_len,
                                                      int rows, int cols):

    cdef:
        Py_ssize_t i, j, ki, k, k_half
        np.ndarray[DTYPE_uint8_t, ndim = 2] ch_bd
        list sts
        DTYPE_float64_t st
        np.ndarray[DTYPE_float32_t, ndim=1] out_list = np.empty(out_len).astype(np.float32)
        int pix_ctr = 0
        int n_scales = np.array(scs).shape[0]

    for i from 0 <= i < rows-scales_block by blk:
        for j from 0 <= j < cols-scales_block by blk:
            for ki in xrange(0, n_scales):

                k = scs[ki]

                k_half = k / 2

                ch_bd = chBd[i+scales_half-k_half:i+scales_half-k_half+k, j+scales_half-k_half:j+scales_half-k_half+k]

                sts = _sfs_feas(ch_bd, blk, cell_size, thresh_1, n_angles)

                for st in sts:
                    out_list[pix_ctr] = st

                    pix_ctr += 1

    return out_list


@cython.boundscheck(False)
@cython.wraparound(False)
def feature_sfs(np.ndarray[DTYPE_uint8_t, ndim=2] chBd, int blk, list scs, int end_scale, \
                DTYPE_float32_t cell_size, DTYPE_float32_t thresh_1, int n_angles):

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
                out_len += 5

    return _feature_sfs(chBd, blk, scs, cell_size, thresh_1, n_angles, scales_half, scales_block, out_len, rows, cols)


@cython.boundscheck(False)
cdef list surfFunc(np.ndarray[DTYPE_uint8_t, ndim=2] surf_arr, k_pts, int j, int i, int k, list scs):

    """
    Get the moments
    """

    cdef int start_y = i+(scs[-1]/2)-(k/2)
    cdef int start_x = j+(scs[-1]/2)-(k/2)

    if surf_arr.max() == 0:
        return [0., 0., 0., 0.]
    else:
        if k_pts:
            # return desc_stats[m](pyramid_hist_sift(surfArr, kPts, start_x, start_y).sp_hist)
            return get_moments(pyramid_hist_sift(surf_arr, k_pts, start_x, start_y).sp_hist)
        else:
            return [0., 0., 0., 0.]


@cython.boundscheck(False)
def feaSURF(np.ndarray[DTYPE_uint8_t, ndim=2] chBd, int blk, list scs):

    cdef int i, j, k
    cdef int rows = chBd.shape[0]
    cdef int cols = chBd.shape[1]
    cdef list sts
    cdef DTYPE_float64_t st
    cdef int scales_half = scs[-1] / 2
    cdef int scales_blk = scs[-1] - blk
    cdef np.ndarray[DTYPE_float64_t, ndim=1] out_list
    cdef int out_len = 0
    cdef int pix_ctr = 0

    for i in xrange(0, rows-scales_blk, blk):
        for j in xrange(0, cols-scales_blk, blk):
            for k in scs:
                out_len += 4

    # set the output list
    out_list = np.zeros(out_len).astype(np.float64)

    # compute SURF features
    kPts, descrip = cv2.SURF(50).detectAndCompute(chBd, None)

    for i in xrange(0, rows-scales_blk, blk):
        for j in xrange(0, cols-scales_blk, blk):
            for k in scs:

                sts = surfFunc(chBd[i+scales_half-(k/2):i+scales_half-(k/2)+k, \
                               j+scales_half-(k/2):j+scales_half-(k/2)+k], kPts, j, i, k, scs)

                for st in sts:
                    out_list[pix_ctr] = st

                    pix_ctr += 1

    return out_list


# Local binary patterns


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
cdef list _feature_lbp(np.ndarray[DTYPE_uint8_t, ndim=2] chBd, int blk, list scs, int end_scale):

    cdef:
        Py_ssize_t i, j, ki, sti
        int k, pc, pr_bin_count, k_half
        int rows = chBd.shape[0]
        int cols = chBd.shape[1]
        np.ndarray[DTYPE_uint8_t, ndim=3] ch_bd
        np.ndarray[DTYPE_uint8_t, ndim=1] sts
        list p_range
        np.ndarray[DTYPE_uint8_t, ndim=3] lbpBd
        int scales_half = end_scale / 2
        cdef int scales_block = end_scale - blk
        np.ndarray[DTYPE_uint8_t, ndim=1] out_list
        int out_len = 0
        int pix_ctr = 0
        DTYPE_uint8_t[:] scales_array = np.array(scs, dtype='uint8')
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

    return list(out_list)


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
        int k, pc, k_half
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
        DTYPE_uint8_t[:] scales_array = np.array(scs, dtype='uint8')
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
def houghFunc_1(np.ndarray[DTYPE_uint8_t, ndim=2] edgeArr, int houghIndex, int minLen, np.ndarray[long, ndim=2] checkList1_2, np.ndarray[long, ndim=2] checkList2_2, \
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
            return np.vstack((cv2.HoughLinesP(edgeArr, 1, pi, minLen, minLineLength=minLen, maxLineGap=2)[0], \
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
def houghFunc_2(np.ndarray[DTYPE_uint8_t, ndim=2] edgeArr, int houghIndex, int minLen, np.ndarray[long, ndim=3] checkList1_3, np.ndarray[long, ndim=3] checkList2_3, \
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
            return np.vstack((cv2.HoughLinesP(edgeArr, 1, pi, minLen, minLineLength=minLen, maxLineGap=2)[0], \
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
                                                        list scs, int scales_half, int scales_block,
                                                        int out_len, int rows, int cols, int threshold,
                                                        int min_len, int line_gap, int end_scale):

    cdef:
        Py_ssize_t i, j, k
        int k_half
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
    for i in xrange(0, rows-scales_block, blk):
        for j in xrange(0, cols-scales_block, blk):

            # get the angles at the largest scale
            lines_list = []

            # get the largest scale array
            large_scale = chBd[i+scales_half-(end_scale/2):i+scales_half-(end_scale/2)+end_scale,
                          j+scales_half-(end_scale/2):j+scales_half-(end_scale/2)+end_scale]

            # get the dimensions for the largest scale
            large_scale_rws = large_scale.shape[0]
            large_scale_cls = large_scale.shape[1]

            # compute the PHL at various angles
            # and add to a list
            lines_list = [PHL(large_scale, threshold=threshold, line_length=min_len, \
                              line_gap=line_gap, theta=angle) for angle in angles]

            # get the matching dimensions at each scale
            # and get line statistics
            for k in scs:

                k_half = k / 2

                # get the current scale array
                ch_bd = chBd[i+scales_half - k_half:i+scales_half - k_half + k, \
                             j+scales_half - k_half:j+scales_half - k_half + k]

                # get line statistics
                sts = _hough_function(lines_list, ch_bd, large_scale_rws, large_scale_cls)

                for st in sts:
                    out_list[pix_ctr] = st

                    pix_ctr += 1

    return out_list

        
@cython.boundscheck(False)
@cython.wraparound(False)
def feature_hough(np.ndarray[DTYPE_uint8_t, ndim=2] chBd, int blk, list scs, int end_scale, int threshold, \
                  int min_len, int line_gap):

    cdef:
        Py_ssize_t i, j, k, pix_ctr
        int rows = chBd.shape[0]
        int cols = chBd.shape[1]
        list sts
        DTYPE_float64_t st
        int out_len = 0
        int scales_half = end_scale / 2
        int scales_block = end_scale - blk


    for i in xrange(0, rows-scales_block, blk):
        for j in xrange(0, cols-scales_block, blk):
            for k in scs:
                out_len += 4

    return _feature_hough(chBd, blk, scs, scales_half, scales_block, out_len, rows, cols,
                          threshold, min_len, line_gap, end_scale)


# PanTex

@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_float32_t pantex_min(np.ndarray[DTYPE_float32_t, ndim=4] glcm_mat, \
                                list dists, list dispVect, \
                                np.ndarray[DTYPE_uint8_t, ndim=2] ch_bd, weighted):
    
    """
    Get the local minimum contrast for all displacement vectors
    """

    cdef:
        int dV, dist
        np.ndarray[DTYPE_float32_t, ndim=1] gmat = np.asarray([greycoprops(glcm_mat, 'contrast')[dist-1][dV]
                                                               for dV in xrange(0, len(dispVect))
                                                               for dist in dists]).astype(np.float32)

    return _get_min_f(gmat, len(gmat))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[DTYPE_float32_t, ndim=1] _feature_pantex(np.ndarray[DTYPE_uint8_t, ndim=2] chBd,
                                                         int blk, DTYPE_uint8_t[:] scs, int scales_half,
                                                         int scales_block, int out_len, bint weighted,
                                                         int rows, int cols, int scale_length):
    
    """
    Get the Anisotropic Built-up Presence Index
    """

    cdef:
        Py_ssize_t i, j, ki
        unsigned int k, k_half
        np.ndarray[DTYPE_uint8_t, ndim=2] ch_bd
        DTYPE_float32_t pi = 3.14159265
        np.ndarray[DTYPE_float32_t, ndim=4] glcm_mat
        DTYPE_float64_t con_min
        int pix_ctr = 0
        np.ndarray[DTYPE_float32_t, ndim=1] out_list = np.zeros(out_len, dtype='float32')
        # directions [E, NE, N, NW]
        list dispVect = [0., pi / 6., pi / 4., pi / 3., pi / 2., (2. * pi) / 3., (3. * pi) / 4., (5. * pi) / 6.]
        list dists = [1, 2]

    if weighted:

        for i from 0 <= i < rows-scales_block by blk:
            for j from 0 <= j < cols-scales_block by blk:
                for ki in xrange(0, scale_length):

                    k = scs[ki]

                    k_half = k / 2

                    ch_bd = chBd[i+scales_half-k_half:i+scales_half-k_half+k,
                                 j+scales_half-k_half:j+scales_half-k_half+k]

                    if ch_bd.max() == 0:
                        con_min = 0.
                    else:

                        glcm_mat = greycomatrix(ch_bd, dists, dispVect,
                                                levels=32, symmetric=True, normed=True).astype(np.float32)

                        con_min = pantex_min(glcm_mat, dists, dispVect, ch_bd, weighted) * cv2.mean(ch_bd)[0]
                    
                    out_list[pix_ctr] = con_min

                    pix_ctr += 1

    else:

        for i from 0 <= i < rows-scales_block by blk:
            for j from 0 <= j < cols-scales_block by blk:
                for ki in xrange(0, scale_length):

                    k = scs[ki]

                    k_half = k / 2

                    ch_bd = np.asarray(chBd[i+scales_half-k_half:i+scales_half-k_half+k,
                                       j+scales_half-k_half:j+scales_half-k_half+k])

                    if ch_bd.max() == 0:
                        con_min = 0.
                    else:

                        glcm_mat = greycomatrix(ch_bd, dists, dispVect,
                                                levels=32, symmetric=True, normed=True).astype(np.float32)

                        con_min = pantex_min(glcm_mat, dists, dispVect, ch_bd, weighted)

                    out_list[pix_ctr] = con_min

                    pix_ctr += 1
            
    return out_list


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def feature_pantex(np.ndarray[DTYPE_uint8_t, ndim=2] chBd, int blk, list scs, int end_scale, bint weighted):

    cdef:
        Py_ssize_t i, j, k
        int scales_half = end_scale / 2
        int scales_block = end_scale - blk
        int out_len = 0
        int rows = chBd.shape[0]
        int cols = chBd.shape[1]
        DTYPE_uint8_t[:] scales_array = np.array(scs, dtype='uint8')
        int scale_length = scales_array.shape[0]

    for i from 0 <= i < rows-scales_block by blk:
        for j from 0 <= j < cols-scales_block by blk:
            for ki in xrange(0, scale_length):
                out_len += 1

    return _feature_pantex(chBd, blk, scales_array, scales_half, scales_block,
                           out_len, weighted, rows, cols, scale_length)


@cython.boundscheck(False)     
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[DTYPE_float64_t, ndim=1] feature_mean_float64(DTYPE_float64_t[:, :] chBd, int blk, DTYPE_uint8_t[:] scs,
                                                              int out_len, int scales_half, int scales_block,
                                                              int scale_length):

    cdef:
        int i, j, ki, pix_ctr
        unsigned int k, bcr, bcc
        DTYPE_float64_t[:] out_list = np.zeros(out_len, dtype='float64')
        int rows = chBd.shape[0]
        int cols = chBd.shape[1]
        DTYPE_float64_t[:, :] block_chunk

    pix_ctr = 0

    for i from 0 <= i < rows-scales_block by blk:
        for j from 0 <= j < cols-scales_block by blk:
            for ki in xrange(0, scale_length):

                k = scs[ki]

                block_chunk = chBd[i+scales_half-(k/2):i+scales_half-(k/2)+k, j+scales_half-(k/2):j+scales_half-(k/2)+k]
                bcr = block_chunk.shape[0]
                bcc = block_chunk.shape[1]

                out_list[pix_ctr] = _get_mean64(block_chunk, bcr, bcc)

                pix_ctr += 1

    return np.asarray(out_list).astype(np.float64)


@cython.boundscheck(False)       
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[DTYPE_float32_t, ndim=1] feature_mean_float32(DTYPE_float32_t[:, :] chBd, int blk, DTYPE_uint8_t[:] scs,
                                                              int out_len, int scales_half, int scales_block,
                                                              int scale_length):

    cdef:
        Py_ssize_t i, j, ki, pix_ctr
        unsigned int k, bcr, bcc
        DTYPE_float32_t[:] out_list = np.zeros(out_len, dtype='float32')
        int rows = chBd.shape[0]
        int cols = chBd.shape[1]
        DTYPE_float32_t[:, :] block_chunk

    pix_ctr = 0

    for i from 0 <= i < rows-scales_block by blk:
        for j from 0 <= j < cols-scales_block by blk:
            for ki in xrange(0, scale_length):

                k = scs[ki]

                block_chunk = chBd[i+scales_half-(k/2):i+scales_half-(k/2)+k, j+scales_half-(k/2):j+scales_half-(k/2)+k]
                bcr = block_chunk.shape[0]
                bcc = block_chunk.shape[1]

                out_list[pix_ctr] = _get_mean(block_chunk, bcr, bcc)

                pix_ctr += 1

    return np.asarray(out_list).astype(np.float32)


@cython.boundscheck(False)      
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[DTYPE_float32_t, ndim=1] feature_mean_uint16(DTYPE_uint16_t[:, :] chBd, int blk, DTYPE_uint8_t[:] scs,
                                                             int out_len, int scales_half, int scales_block,
                                                             int scale_length):

    cdef:
        int i, j, ki, pix_ctr
        unsigned int k, bcr, bcc
        DTYPE_float32_t[:] out_list = np.zeros(out_len, dtype='float32')
        int rows = chBd.shape[0]
        int cols = chBd.shape[1]
        DTYPE_uint16_t[:, :] block_chunk

    pix_ctr = 0

    for i from 0 <= i < rows-scales_block by blk:
        for j from 0 <= j < cols-scales_block by blk:
            for ki in xrange(0, scale_length):

                k = scs[ki]

                block_chunk = chBd[i+scales_half-(k/2):i+scales_half-(k/2)+k, j+scales_half-(k/2):j+scales_half-(k/2)+k]
                bcr = block_chunk.shape[0]
                bcc = block_chunk.shape[1]

                out_list[pix_ctr] = _get_mean(np.float32(block_chunk), bcr, bcc)

                pix_ctr += 1

    return np.asarray(out_list).astype(np.float32)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[DTYPE_float32_t, ndim=1] _feature_mean(DTYPE_uint8_t[:, :] chBd, int blk, DTYPE_uint8_t[:] scs,
                                                       int out_len, int scales_half, int scales_block, int scale_length):

    cdef:
        int i, j, ki, pix_ctr
        unsigned int k, bcr, bcc
        DTYPE_float32_t[:] out_list = np.zeros(out_len, dtype='float32')
        int rows = chBd.shape[0]
        int cols = chBd.shape[1]
        DTYPE_uint8_t[:, :] block_chunk

    pix_ctr = 0

    for i from 0 <= i < rows-scales_block by blk:
        for j from 0 <= j < cols-scales_block by blk:
            for ki in xrange(0, scale_length):

                k = scs[ki]

                block_chunk = chBd[i+scales_half-(k/2):i+scales_half-(k/2)+k, j+scales_half-(k/2):j+scales_half-(k/2)+k]
                bcr = block_chunk.shape[0]
                bcc = block_chunk.shape[1]

                out_list[pix_ctr] = _get_mean(np.float32(block_chunk), bcr, bcc)

                pix_ctr += 1

    return np.asarray(out_list).astype(np.float32)

            
@cython.boundscheck(False)  
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[DTYPE_float32_t, ndim=1] feature_mean_uint(DTYPE_int_t[:, :] chBd, int blk, DTYPE_uint8_t[:] scs,
                                                           int out_len, int scales_half, int scales_block,
                                                           int scale_length):

    cdef:
        int i, j, ki, pix_ctr
        unsigned int k, bcr, bcc
        DTYPE_float32_t[:] out_list = np.zeros(out_len, dtype='float32')
        int rows = chBd.shape[0]
        int cols = chBd.shape[1]
        DTYPE_int_t[:, :] block_chunk

    pix_ctr = 0

    for i from 0 <= i < rows-scales_block by blk:
        for j from 0 <= j < cols-scales_block by blk:
            for ki in xrange(0, scale_length):

                k = scs[ki]

                block_chunk = chBd[i+scales_half-(k/2):i+scales_half-(k/2)+k, j+scales_half-(k/2):j+scales_half-(k/2)+k]
                bcr = block_chunk.shape[0]
                bcc = block_chunk.shape[1]

                out_list[pix_ctr] = _get_mean(np.float32(block_chunk), bcr, bcc)

                pix_ctr += 1
    
    return np.asarray(out_list).astype(np.float32)

            
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def feature_mean(np.ndarray chBd, int blk, list scs, int end_scale):

    cdef:
        Py_ssize_t i, j, k
        int scales_half = end_scale / 2
        int scales_block = end_scale - blk
        int out_len = 0
        int rows = chBd.shape[0]
        int cols = chBd.shape[1]
        DTYPE_uint8_t[:] scales_array = np.array(scs, dtype='uint8')
        int scale_length = scales_array.shape[0]

    for i from 0 <= i < rows-scales_block by blk:
        for j from 0 <= j < cols-scales_block by blk:
            for ki in xrange(0, scale_length):
                out_len += 1

    if chBd.dtype == 'float64':
        return feature_mean_float64(chBd, blk, scales_array, out_len, scales_half, scales_block, scale_length)
    elif chBd.dtype == 'float32':
        return feature_mean_float32(chBd, blk, scales_array, out_len, scales_half, scales_block, scale_length)
    elif chBd.dtype == 'uint16':

        try:

            return feature_mean_float32(chBd.astype(np.float32), blk, scales_array, out_len, scales_half,
                                        scales_block, scale_length)

        except:
            return feature_mean_uint16(chBd, blk, scales_array, out_len, scales_half, scales_block, scale_length)

    elif chBd.dtype == 'uint8':

        try:

            return feature_mean_float32(chBd.astype(np.float32), blk, scales_array, out_len, scales_half,
                                        scales_block, scale_length)

        except:
            return feature_mean_uint(chBd, blk, scales_array, out_len, scales_half, scales_block, scale_length)


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

        int boxes_max = int(ceil(float(maxi) / min(rr_rows, rr_cols)))
        int boxes_min = int(ceil(float(mini) / min(rr_rows, rr_cols)))

    return (boxes_max - boxes_min) + 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef DTYPE_float32_t[:] _div1d(DTYPE_float32_t[:] array1d, int cs, DTYPE_float32_t div_value):

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
cdef list _feature_lacunarity(DTYPE_uint8_t[:, :] chunk_block, int blk, list scales, int scales_half,
                              int scales_block, int rows, int cols, int r, int out_len, int scale_length):

    cdef:
        Py_ssize_t i, j, k, ki, cr, cc
        Py_ssize_t pixel_counter = 0
        DTYPE_uint8_t[:, :] ch_bd
        DTYPE_float32_t[:] out_list = np.zeros(out_len, dtype='float32')
        int k_half

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

    return list(np.asarray(out_list))


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
        int scale_length = len(scales)

    for i from 0 <= i < rows-scales_block by blk:
        for j from 0 <= j < cols-scales_block by blk:
            for ki in xrange(0, scale_length):
                out_len += 1

    return _feature_lacunarity(np.uint8(chunk_block), blk, scales, scales_half, scales_block,
                               rows, cols, r, out_len, scale_length)
