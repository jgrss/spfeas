# cython: profile=False
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import cython
cimport cython
from cpython cimport array

import numpy as np
cimport numpy as np

# from libc.stdlib cimport free
from libc.math cimport atan, sqrt, sin, cos, floor, ceil
# from libc.math cimport isnan as npy_isnan
# from libc.math cimport isinf as npy_isinf

# from cython.parallel import parallel, prange
# from libc.math cimport isnan, isinf

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

DTYPE_uint64 = np.uint64
ctypedef np.uint64_t DTYPE_uint64_t

DTYPE_float32 = np.float32
ctypedef np.float32_t DTYPE_float32_t

DTYPE_float64 = np.float64
ctypedef np.float64_t DTYPE_float64_t

# cdef npceil = np.ceil

# cdef extern from 'numpy/npy_math.h':
#     DTYPE_float32_t npy_ceil(DTYPE_float32_t x) nogil

cdef extern from 'numpy/npy_math.h':
    bint npy_isnan(DTYPE_float32_t x) nogil

cdef extern from 'numpy/npy_math.h':
    bint npy_isinf(DTYPE_float32_t x) nogil

# cdef extern from 'math.h':
#     DTYPE_float32_t ceil(DTYPE_float32_t x)

# cdef extern from 'numpy/npy_math.h':
#     DTYPE_float32_t npy_floor(DTYPE_float32_t x)


# Define a function pointer to a metric.
ctypedef DTYPE_float32_t[:, :, :, ::1] (*metric_ptr)(DTYPE_float32_t[:, :, :, ::1], DTYPE_float32_t[:, :], DTYPE_float32_t[:], DTYPE_float32_t[:], int, DTYPE_float32_t[:, :, :, ::1]) nogil


cdef inline DTYPE_float32_t roundd(DTYPE_float32_t val) nogil:
     return floor(val + 0.5)


cdef inline DTYPE_float32_t sqrt_f(DTYPE_float32_t sx) nogil:
    return sx * 0.5


cdef inline DTYPE_float32_t abs_f(DTYPE_float32_t sx) nogil:
    return sx * -1.0 if sx < 0 else sx


cdef inline DTYPE_uint8_t abs_ui(DTYPE_uint8_t sx) nogil:
    return sx * -1 if sx < 0 else sx


cdef inline Py_ssize_t abs_s(Py_ssize_t sx) nogil:
    return sx * -1 if sx < 0 else sx


cdef inline int n_rows_cols(int pixel_index, int rows_cols, int block_size) nogil:
    return rows_cols if pixel_index + rows_cols < block_size else block_size - pixel_index


cdef inline DTYPE_float32_t pow2(DTYPE_float32_t sx) nogil:
    return sx * sx


cdef inline DTYPE_float32_t pow3(DTYPE_float32_t sx) nogil:
    return sx * sx * sx


cdef inline DTYPE_float32_t pow4(DTYPE_float32_t sx) nogil:
    return sx * sx * sx * sx


cdef inline int _get_min_sample_i(int s1, int s2) nogil:
    return s2 if s2 < s1 else s1


cdef inline DTYPE_float32_t _get_min_sample(DTYPE_float32_t s1, DTYPE_float32_t s2) nogil:
    return s2 if s2 < s1 else s1


cdef inline DTYPE_uint8_t _get_min_sample_int(DTYPE_uint8_t s1, DTYPE_uint8_t s2) nogil:
    return s2 if s2 < s1 else s1


cdef inline DTYPE_float32_t _get_max_sample(DTYPE_float32_t s1, DTYPE_float32_t s2) nogil:
    return s2 if s2 > s1 else s1


cdef inline DTYPE_uint8_t _get_max_sample_int(DTYPE_uint8_t s1, DTYPE_uint8_t s2) nogil:
    return s2 if s2 > s1 else s1


cdef inline DTYPE_float32_t _euclidean_distance(DTYPE_float32_t x1, DTYPE_float32_t y1, DTYPE_float32_t x2, DTYPE_float32_t y2) nogil:
    return (((x1 - x2)**2.) + ((y1 - y2)**2.))**0.5


cdef inline DTYPE_float32_t _get_line_length(DTYPE_float32_t y1, DTYPE_float32_t x1, DTYPE_float32_t y2, DTYPE_float32_t x2) nogil:
    return ((y1 - x1)**2 + (y2 - x2)**2)**0.5


cdef unsigned int _get_output_length(int rows,
                                     int cols,
                                     int scales_block,
                                     int block_size,
                                     int scale_length,
                                     int n_features):

    cdef:
        Py_ssize_t i, j, ki
        unsigned int out_len = 0

    for i from 0 <= i < rows-scales_block by block_size:

        for j from 0 <= j < cols-scales_block by block_size:

            for ki in range(0, scale_length):
                out_len += n_features

    return out_len


cdef DTYPE_uint8_t _get_min(DTYPE_uint8_t[:, :] block, int rs, int cs) nogil:

    cdef:
        Py_ssize_t bi, bj
        DTYPE_uint8_t m = 255

    for bi in range(0, rs):
        for bj in range(0, cs):

            m = _get_min_sample_int(m, block[bi, bj])

    return m


cdef DTYPE_float32_t _get_max_f2d(DTYPE_float32_t[:, :] block, int rs, int cs) nogil:

    cdef:
        Py_ssize_t bi, bj
        DTYPE_float32_t m = -9999999.

    for bi in range(0, rs):
        for bj in range(0, cs):

            m = _get_max_sample(m, block[bi, bj])

    return m


cdef int _get_max(DTYPE_uint8_t[:, :] block, Py_ssize_t rs, Py_ssize_t cs) nogil:

    cdef:
        Py_ssize_t bi, bj
        int m = -255

    for bi in range(0, rs):
        for bj in range(0, cs):

            m = _get_max_sample_int(m, block[bi, bj])

    return m


cdef DTYPE_float32_t _get_max_f(DTYPE_float32_t[:] in_row, int cols) nogil:

    cdef:
        Py_ssize_t a
        DTYPE_float32_t m = in_row[0]

    for a in range(1, cols):
        m = _get_max_sample(m, in_row[a])

    return m


cdef DTYPE_float32_t _get_sum_uint8(DTYPE_uint8_t[:, :] block, int rs, int cs) nogil:

    cdef:
        Py_ssize_t bi, bj
        DTYPE_float32_t block_sum = 0.

    for bi in range(0, rs):
        for bj in range(0, cs):
            block_sum += float(block[bi, bj])

    return block_sum


cdef DTYPE_float32_t _get_sum(DTYPE_float32_t[:, :] block, int rs, int cs) nogil:

    cdef:
        Py_ssize_t bi, bj
        DTYPE_float32_t block_sum = 0.

    for bi in range(0, rs):
        for bj in range(0, cs):
            block_sum += block[bi, bj]

    return block_sum


cdef DTYPE_float32_t _get_mean(DTYPE_float32_t[:, :] block, int rs, int cs) nogil:

    cdef:
        DTYPE_float32_t n_samps = float(rs*cs)

    return _get_sum(block, rs, cs) / n_samps


cdef void _get_mean_var(DTYPE_float32_t[:, :] block,
                        int rs,
                        int cs,
                        DTYPE_float32_t[:] out_values_) nogil:

    cdef:
        Py_ssize_t bi, bj
        DTYPE_float32_t n_samps = float(rs*cs)
        DTYPE_float32_t mu = _get_mean(block, rs, cs)
        DTYPE_float32_t block_var = 0.

    for bi in range(0, rs):
        for bj in range(0, cs):
            block_var += pow2(float(block[bi, bj]) - mu)

    out_values_[0] = mu
    out_values_[1] = block_var / n_samps


cdef DTYPE_float32_t _get_weighted_sum(DTYPE_float32_t[:, ::1] block,
                                       DTYPE_float32_t[:, ::1] weights,
                                       int rs,
                                       int cs) nogil:

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


cdef DTYPE_float32_t _get_weighted_sum_byte(DTYPE_uint8_t[:, :] block, DTYPE_float32_t[:, :] weights, int rs, int cs) nogil:

    cdef:
        Py_ssize_t bi, bj
        DTYPE_float32_t block_sum = 0.
        DTYPE_float32_t dv

    for bi in range(0, rs):
        for bj in range(0, cs):

            dv = float(block[bi, bj]) / weights[bi, bj]

            if not npy_isnan(dv) and not npy_isinf(dv):
                block_sum += dv

    return block_sum


cdef DTYPE_float32_t _get_weighted_mean(DTYPE_float32_t[:, ::1] block,
                                        DTYPE_float32_t[:, ::1] weights,
                                        int rs,
                                        int cs) nogil:

    cdef:
        DTYPE_float32_t n_samps = float(rs*cs)

    return _get_weighted_sum(block, weights, rs, cs) / n_samps


cdef DTYPE_float32_t _get_weighted_mean_byte(DTYPE_uint8_t[:, :] block, DTYPE_float32_t[:, :] weights, int rs, int cs) nogil:

    cdef:
        DTYPE_float32_t n_samps = float(rs*cs)

    return _get_weighted_sum_byte(block, weights, rs, cs) / n_samps


cdef void _get_weighted_mean_var(DTYPE_float32_t[:, ::1] block,
                                 DTYPE_float32_t[:, ::1] weights,
                                 int rs,
                                 int cs,
                                 DTYPE_float32_t[::1] out_values_) nogil:

    cdef:
        Py_ssize_t bi, bj
        DTYPE_float32_t n_samps = float(rs*cs)
        DTYPE_float32_t mu = _get_weighted_mean(block, weights, rs, cs)
        DTYPE_float32_t block_var = 0.

    for bi in range(0, rs):
        for bj in range(0, cs):
            block_var += pow2(float(block[bi, bj]) - mu)

    out_values_[0] = mu
    out_values_[1] = block_var / n_samps


cdef void _get_weighted_mean_var_byte(DTYPE_uint8_t[:, ::1] block,
                                      DTYPE_float32_t[:, ::1] weights,
                                      int rs,
                                      int cs,
                                      DTYPE_float32_t[::1] out_values_) nogil:

    cdef:
        Py_ssize_t bi, bj
        DTYPE_float32_t n_samps = float(rs*cs)
        DTYPE_float32_t block_mu = _get_weighted_mean_byte(block, weights, rs, cs)
        DTYPE_float32_t block_var = 0.

    for bi in range(0, rs):
        for bj in range(0, cs):
            block_var += pow2(float(block[bi, bj]) - block_mu)

    out_values_[0] = block_mu
    out_values_[1] = block_var / n_samps


# cdef void _get_directional_weighted_mean_var(DTYPE_float32_t[:, :] block,
#                                              int rs,
#                                              int cs,
#                                              DTYPE_float32_t[:] out_values_) nogil:
#
#     cdef:
#         int rs_half = <int>(rs / 2.)
#         int cs_half = <int>(cs / 2.)
#         int rs_qu = <int>(rs_half / 2.)
#         int cs_qu = <int>(cs_half / 2.)
#
#     # Upper left box
#     out_values_[0] = _get_mean(block[:rs_half, :cs_half], rs_half, cs_half)
#
#     # Upper center box
#     out_values_[1] = _get_mean(block[:rs_half, cs_qu:cs_qu+cs_half], rs_half, cs_half)
#
#     # Upper right box
#     out_values_[2] = _get_mean(block[:rs_half, cs_half:], rs_half, cs_half)
#
#     # Left box
#     out_values_[3] = _get_mean(block[rs_qu:rs_qu+rs_half, :cs_half], rs_half, cs_half)
#
#     # Center box
#     out_values_[4] = _get_mean(block[rs_qu:rs_qu+rs_half, cs_qu:cs_qu+cs_half], rs_half, cs_half)
#
#     # Right box
#     out_values_[5] = _get_mean(block[rs_qu:rs_qu+rs_half, cs_half:], rs_half, cs_half)
#
#     # Lower left box
#     out_values_[6] = _get_mean(block[rs_half:, :cs_half], rs_half, cs_half)
#
#     # Lower center box
#     out_values_[7] = _get_mean(block[rs_half:, cs_qu:cs_qu+cs_half], rs_half, cs_half)
#
#     # Lower right box
#     out_values_[8] = _get_mean(block[rs_half:, cs_half:], rs_half, cs_half)


# cdef void _get_angle_stats(DTYPE_float32_t[:, :] block,
#                            int rs,
#                            int cs,
#                            DTYPE_float32_t[:] out_values_) nogil:
#
#     cdef:
#         Py_ssize_t i, j, jc
#         int rsh = <int>(rs / 2.)
#         int csh = <int>(cs / 2.)
#         DTYPE_float32_t line_sum
#
#     # Center horizontal
#     line_sum = 0.
#     for j in range(0, cs):
#         line_sum += block[rsh, j]
#
#     out_values_[0] = line_sum / cs
#
#     # Center vertical
#     line_sum = 0.
#     for i in range(0, rs):
#         line_sum += block[i, csh]
#
#     out_values_[1] = line_sum / rs
#
#     # Top left to bottom right
#     line_sum = 0.
#     for i in range(0, rs):
#         line_sum += block[i, i]
#
#     out_values_[2] = line_sum / rs
#
#     # Top right to bottom left
#     line_sum = 0.
#     jc = cs - 1
#     for i in range(0, rs):
#         line_sum += block[i, jc]
#         jc -= 1
#
#     out_values_[3] = line_sum / rs


cdef DTYPE_float32_t _get_mean_uint8(DTYPE_uint8_t[:, :] block, int rs, int cs) nogil:

    cdef:
        DTYPE_float32_t n_samps = float(rs*cs)

    return _get_sum_uint8(block, rs, cs) / n_samps


cdef DTYPE_float32_t _get_std_1d(DTYPE_float32_t[:] block_line, int cs, DTYPE_float32_t psi) nogil:

    cdef:
        Py_ssize_t bj
        DTYPE_float32_t block_std = 0.

    for bj in range(0, cs):
        block_std += pow2(block_line[bj] - psi)

    return sqrt(block_std / cs)


cdef DTYPE_float32_t _get_std_1d_uint16(DTYPE_uint16_t[:] block, int cs) nogil:

    cdef:
        Py_ssize_t bj
        DTYPE_float32_t mu = _get_mean_1d_uint16(block, cs)
        DTYPE_float32_t block_var = 0.

    for bj in range(0, cs):
        block_var += pow2(float(block[bj]) - mu)

    return sqrt_f(block_var / cs)


cdef DTYPE_float32_t _get_var(DTYPE_float32_t[:, :] block, int rs, int cs, DTYPE_float32_t ddof=1.) nogil:

    cdef:
        Py_ssize_t bi, bj
        DTYPE_float32_t mu = _get_mean(block, rs, cs)
        DTYPE_float32_t block_var = 0.

    for bi in range(0, rs):
        for bj in range(0, cs):
            block_var += pow2(float(block[bi, bj]) - mu)

    return block_var / ((rs*cs) - ddof)


cdef DTYPE_float32_t _get_var_uint8(DTYPE_uint8_t[:, :] block, int rs, int cs, DTYPE_float32_t ddof=1.) nogil:

    cdef:
        Py_ssize_t bi, bj
        DTYPE_float32_t mu = _get_mean_uint8(block, rs, cs)
        DTYPE_float32_t block_var = 0.

    for bi in range(0, rs):
        for bj in range(0, cs):
            block_var += pow2(float(block[bi, bj]) - mu)

    return block_var / ((rs*cs) - ddof)


cdef DTYPE_float32_t _get_sum1d(DTYPE_float32_t[:] block, int cs) nogil:

    cdef:
        Py_ssize_t bj
        DTYPE_float32_t block_sum = block[0]

    # for bj in prange(0, cs, nogil=True, num_threads=cs, schedule='static'):
    for bj in range(1, cs):
        block_sum += block[bj]

    return block_sum


cdef DTYPE_float32_t _get_mean_1d(DTYPE_float32_t[:] block, int cs) nogil:
    return _get_sum1d(block, cs) / cs


cdef DTYPE_float32_t _get_var_1d(DTYPE_float32_t[:] block, int cs, DTYPE_float32_t mu, DTYPE_float32_t ddof=1.) nogil:

    cdef:
        Py_ssize_t bj
        DTYPE_float32_t block_var = 0.

    for bj in range(0, cs):
        block_var += pow2(float(block[bj]) - mu)

    return block_var / (cs - ddof)


cdef DTYPE_float32_t _get_sum1d_uint16(DTYPE_uint16_t[:] block, int cs) nogil:

    cdef:
        Py_ssize_t bj
        DTYPE_uint16_t block_sum = block[0]

    for bj in range(1, cs):
        block_sum += block[bj]

    return float(block_sum)


cdef DTYPE_float32_t _get_mean_1d_uint16(DTYPE_uint16_t[:] block, int cs) nogil:
    return _get_sum1d_uint16(block, cs) / cs


cdef void draw_line(Py_ssize_t y0, Py_ssize_t x0, Py_ssize_t y1, Py_ssize_t x1, DTYPE_uint16_t[:, :] rc_) nogil:

    """
    *Graciously adapted from the Scikit-image team @ https://github.com/scikit-image/scikit-image/blob/master/skimage/draw/_draw.pyx

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

    # rc = np.empty((2, dx+1), dtype='uint16')
    #rc = <double * >malloc((n ** 2) * sizeof(double))
    # cc = rr.copy()
    # rr = clone(template, int(dx)+1, True)
    # cc = clone(template, int(dx)+1, True)

    for i in range(0, dx):

        if steep:
            rc_[0, i] = x
            rc_[1, i] = y
        else:
            rc_[0, i] = y
            rc_[1, i] = x

        while d >= 0:

            y += sy
            d -= 2 * dx

        x += sx
        d += 2 * dy

    rc_[0, dx] = y1
    rc_[1, dx] = x1

    # Store the real line length
    rc_[2, 0] = dx + 1


cdef void _get_stats(DTYPE_float32_t[:] block, int samps, DTYPE_float32_t[:] output_array) nogil:

    """Calculate the central moments 1-4"""

    cdef:
        Py_ssize_t idx
        DTYPE_float32_t the_max = _get_max_f(block, samps)
        DTYPE_float32_t m1 = _get_mean_1d(block, samps)         # 1st moment (mean)
        DTYPE_float32_t m2 = _get_var_1d(block, samps, m1)      # 2nd moment (variance)
        DTYPE_float32_t stdev = sqrt_f(m2)                      # standard deviation
        DTYPE_float32_t bx = block[0]
        DTYPE_float32_t val_dev = bx - m1
        DTYPE_float32_t m3 = pow3(val_dev)                      # 3rd moment (standard deviation)
        DTYPE_float32_t m4 = pow4(val_dev)                      # 4th moment (kurtosis)

    for idx in range(1, samps):

        bx = block[idx]
        val_dev = bx - m1

        m3 += pow3(val_dev)
        m4 += pow4(val_dev)

    m3 /= samps
    m4 /= samps

    output_array[0] = the_max               # max
    output_array[1] = m1                    # mean
    output_array[2] = m2                    # variance
    output_array[3] = m3 / pow3(stdev)      # skewness: ratio of 3rd moment and standard dev. cubed
    output_array[4] = m4 / pow4(stdev)      # kurtosis


cdef void _get_moments(DTYPE_float32_t[::1] img_arr, DTYPE_float32_t[::1] output) nogil:

    """Get the moments for 1d array"""

    cdef:
        int img_arr_cols = img_arr.shape[0]

    if _get_max_f(img_arr, img_arr_cols) != 0:
        _get_stats(img_arr, img_arr_cols, output)


# Gabor filter bank

cdef void _convolution(DTYPE_float32_t[:, :] block2convolve,
                       DTYPE_float32_t[:, :] gkernel,
                       int br, int bc,
                       int knr, int knc,
                       int knrh, int knch,
                       DTYPE_float32_t[:, :] out_convolved) nogil:

    """"2d convolution of a Gabor kernel over a local window"""

    cdef:
        Py_ssize_t bi, bj, bki, bkj
        DTYPE_float32_t kernel_sum

    # Move the kernel
    for bi in range(0, br-knr):

        for bj in range(0, bc-knc):

            kernel_sum = 0.

            # Process the kernel
            for bki in range(0, knr):

                for bkj in range(0, knc):

                    kernel_sum += block2convolve[bi+bki, bj+bkj] * gkernel[bki, bkj]

            out_convolved[bi+knrh, bj+knch] = kernel_sum


cdef void _feature_gabor(DTYPE_float32_t[:, :, ::1] ch_bdka,
                         int blk,
                         DTYPE_uint16_t[::1] scs,
                         int out_len,
                         int scales_half,
                         int scales_block,
                         int n_kernels,
                         int rows,
                         int cols,
                         int scale_length,
                         int end_scale,
                         DTYPE_float32_t[::1] out_list_):

    """
    Returns at each scale at each kernel

    1:	Mean
    2:	Variance
    """

    cdef:
        Py_ssize_t i, j, ki, kl, pi, scale_kernel
        DTYPE_uint16_t k
        unsigned int rs, cs, k_half
        DTYPE_float32_t[:, ::1] ch_bd

        # list ch_bd_k = []
        # np.ndarray[DTYPE_float32_t, ndim=3] ch_bdka_array = np.zeros((n_kernels, rows, cols), dtype='float32')
        # DTYPE_float32_t[:, :, :] ch_bdka = np.zeros((n_kernels, rows, cols), dtype='float32')
        # DTYPE_float32_t[:, :] ch_bd_gabor
        # DTYPE_float32_t[:] sts
        # list st
        int pix_ctr = 0
        # int knr = kernels[0].shape[0]
        # int knc = kernels[0].shape[1]
        # int knrh = <int>(knr / 2.)
        # int knch = <int>(knc / 2.)
        # DTYPE_float32_t[:, :] gkernel
        # DTYPE_float32_t[:] out_values
        DTYPE_float32_t[:, :, ::1] dist_weights_stack = np.zeros((scale_length*n_kernels,
                                                                  end_scale*2,
                                                                  end_scale*2), dtype='float32')


        int bcr, bcc
        DTYPE_float32_t[::1] in_zs = np.zeros(2, dtype='float32')
        DTYPE_float32_t[:, ::1] dist_weights, dw

    scale_kernel = 0

    for ki in range(0, scale_length):

        k = scs[ki]
        k_half = <int>(k / 2.)

        rs = (scales_half - k_half + k) - (scales_half - k_half)
        cs = (scales_half - k_half + k) - (scales_half - k_half)

        dist_weights = np.zeros((rs, cs), dtype='float32')

        for kl in range(0, n_kernels):

            dist_weights_stack[scale_kernel, :rs, :cs] = _create_weights(dist_weights, rs, cs)

            scale_kernel += 1

    with nogil:

        for i from 0 <= i < rows-scales_block by blk:

            for j from 0 <= j < cols-scales_block by blk:

                scale_kernel = 0

                for ki in range(0, scale_length):

                    k = scs[ki]
                    k_half = <int>(k / 2.)

                    rs = (scales_half - k_half + k) - (scales_half - k_half)
                    cs = (scales_half - k_half + k) - (scales_half - k_half)

                    for kl in range(0, n_kernels):

                        ch_bd = ch_bdka[scale_kernel,
                                        i+scales_half-k_half:i+scales_half-k_half+k,
                                        j+scales_half-k_half:j+scales_half-k_half+k]

                        bcr = ch_bd.shape[0]
                        bcc = ch_bd.shape[1]

                        dw = dist_weights_stack[scale_kernel, :rs, :cs]

                        # _convolution(ch_bd, gkernel, bcr, bcc, knr, knc, knrh, knch, ch_bd_gabor)

                        _get_weighted_mean_var(ch_bd, dw, bcr, bcc, in_zs)
                        # _get_angle_stats(ch_bd_gabor, bcr, bcc, in_zs)

                        # _get_directional_weighted_mean_var(ch_bd_gabor, bcr, bcc, in_zs)

                        # _get_moments(in_zs, sts)

                        for pi in range(0, 2):

                            out_list_[pix_ctr] = in_zs[pi]
                            pix_ctr += 1

                        scale_kernel += 1


def feature_gabor(DTYPE_float32_t[:, :, ::1] chbd, int blk, list scs, int end_scale, int n_kernels=8):

    cdef:
        Py_ssize_t i, j, ki, kl
        int scales_half = <int>(end_scale / 2.)
        int scales_block = end_scale - blk
        int rows = chbd.shape[1]
        int cols = chbd.shape[2]
        DTYPE_uint16_t[::1] scales_array = np.array(scs, dtype='uint16')
        int scale_length = scales_array.shape[0]
        DTYPE_float32_t[::1] out_list
        unsigned int out_len = 0

    for i from 0 <= i < rows-scales_block by blk:
        for j from 0 <= j < cols-scales_block by blk:
            for ki in range(0, scale_length):
                for kl in range(0, n_kernels):
                    out_len += 2

    out_list = np.zeros(out_len, dtype='float32')

    _feature_gabor(chbd,
                   blk,
                   scales_array,
                   out_len,
                   scales_half,
                   scales_block,
                   n_kernels,
                   rows,
                   cols,
                   scale_length,
                   end_scale,
                   out_list)

    return np.float32(out_list)


# Histogram of Oriented Gradients

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


cdef void _feature_hog(DTYPE_float32_t[:, ::1] chbd,
                       int blk,
                       DTYPE_uint16_t[:] scs,
                       int end_scale,
                       int scales_half,
                       int scales_block,
                       int out_len,
                       int rows,
                       int cols,
                       int scale_length,
                       DTYPE_float32_t[::1] out_list_):

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
        DTYPE_float32_t[:, ::1] ch_bd
        int pix_ctr = 0
        bin_n = 9
        DTYPE_float32_t[::1] sts = np.zeros(5, dtype='float32')
        DTYPE_float32_t[::1] sts_ = sts.copy()
        DTYPE_float32_t[::1] hog_results

    for i from 0 <= i < rows-scales_block by blk:

        for j from 0 <= j < cols-scales_block by blk:

            for ki in range(0, scale_length):

                k = scs[ki]

                k_half = <int>(k / 2.)

                ch_bd = chbd[i+scales_half-k_half:i+scales_half-k_half+k,
                             j+scales_half-k_half:j+scales_half-k_half+k]

                block_rows = ch_bd.shape[0]
                block_cols = ch_bd.shape[1]

                if _get_max_f2d(ch_bd, block_rows, block_cols) > 0:

                    hog_results = np.float32(np.ascontiguousarray(HOG(np.float32(ch_bd),
                                                                      pixels_per_cell=(block_rows, block_cols),
                                                                      cells_per_block=(1, 1))))

                    sts_[...] = sts

                    _get_moments(hog_results, sts_)

                    for sti in range(0, 5):

                        out_list_[pix_ctr] = sts_[sti]

                        pix_ctr += 1

                else:
                    pix_ctr += 5


def feature_hog(DTYPE_float32_t[:, ::1] chbd, int blk, list scs, int end_scale):

    cdef:
        Py_ssize_t i, j, ki
        int scales_half = <int>(end_scale / 2.0)
        int scales_block = end_scale - blk
        int rows = chbd.shape[0]
        int cols = chbd.shape[1]
        DTYPE_uint16_t[:] scales_array = np.array(scs, dtype='uint16')
        int scale_length = scales_array.shape[0]
        unsigned int out_len = _get_output_length(rows, cols, scales_block, blk, scale_length, 5)
        DTYPE_float32_t[::1] out_list = np.zeros(out_len, dtype='float32')

    _feature_hog(chbd,
                 blk,
                 scales_array,
                 end_scale,
                 scales_half,
                 scales_block,
                 out_len,
                 rows,
                 cols,
                 scale_length,
                 out_list)

    return np.float32(out_list)


cdef void _add_dmps(DTYPE_float32_t[:, ::1] ch_bd_array,
                    int block_rows,
                    int block_cols,
                    DTYPE_float32_t[::1] dmp_vector_array) nogil:

    cdef:
        Py_ssize_t ri, cj

    for ri in range(0, block_rows):
        for cj in range(0, block_cols):
            dmp_vector_array[0] += ch_bd_array[ri, cj]


cdef void _feature_dmp(DTYPE_float32_t[:, ::1] chbd,
                       int blk,
                       DTYPE_uint16_t[::1] scs,
                       int end_scale,
                       int scales_half,
                       int scales_block,
                       int out_len,
                       int rows,
                       int cols,
                       int scale_length,
                       DTYPE_float32_t[::1] out_list_):

    cdef:
        Py_ssize_t i, j, ki, sti, block_rows, block_cols
        DTYPE_uint16_t k, k_half
        unsigned int rc_start, rc_end, rc
        DTYPE_float32_t[:, ::1] ch_bd
        int pix_ctr = 0
        #DTYPE_float32_t[::1] sts = np.zeros(5, dtype='float32')
        #DTYPE_float32_t[::1] sts_ = sts.copy()
        #DTYPE_float32_t[::1] dmp_vector = np.zeros(1, dtype='float32')
        #DTYPE_float32_t[::1] dmp_vector_ = dmp_vector.copy()
        DTYPE_float32_t[:, ::1] dist_weights, dw
        DTYPE_float32_t[:, :, ::1] dist_weights_stack = np.zeros((scale_length, end_scale*2, end_scale*2), dtype='float32')
        DTYPE_float32_t[::1] in_zs = np.zeros(2, dtype='float32')

    for ki in range(0, scale_length):

        k = scs[ki]
        k_half = <int>(k / 2.0)

        rc_start = scales_half - k_half
        rc_end = scales_half - k_half + k

        rc = rc_end - rc_start

        dist_weights = np.zeros((rc, rc), dtype='float32')

        dist_weights_stack[ki, :rc, :rc] = _create_weights(dist_weights, rc, rc)

    with nogil:

        for i from 0 <= i < rows-scales_block by blk:

            for j from 0 <= j < cols-scales_block by blk:

                for ki in range(0, scale_length):

                    k = scs[ki]

                    k_half = <int>(k / 2.0)

                    # Get the DMPS.
                    ch_bd = chbd[i+scales_half-k_half:i+scales_half-k_half+k,
                                 j+scales_half-k_half:j+scales_half-k_half+k]

                    block_rows = ch_bd.shape[0]
                    block_cols = ch_bd.shape[1]

                    # Add the DMPs for the local scale.
                    #dmp_vector_[...] = dmp_vector
                    #_add_dmps(ch_bd, block_rows, block_cols, dmp_vector_)

                    dw = dist_weights_stack[ki, :block_rows, :block_cols]
                    _get_weighted_mean_var(ch_bd, dw, block_rows, block_cols, in_zs)

                    #sts_[...] = sts

                    # Get the DMP vector
                    #   central moments.
                    #_get_moments(dmp_vector_, sts_)

                    # Fill the output.
                    for sti in range(0, 2):

                        out_list_[pix_ctr] = in_zs[sti]

                        pix_ctr += 1


def feature_dmp(DTYPE_float32_t[:, ::1] chbd, int blk, list scs, int end_scale):

    cdef:
        int scales_half = <int>(end_scale / 2.0)
        int scales_block = end_scale - blk
        int rows = chbd.shape[0]
        int cols = chbd.shape[1]
        DTYPE_uint16_t[::1] scales_array = np.array(scs, dtype='uint16')
        int scale_length = scales_array.shape[0]
        unsigned int out_len = _get_output_length(rows, cols, scales_block, blk, scale_length, 2)
        DTYPE_float32_t[::1] out_list = np.zeros(out_len, dtype='float32')

    _feature_dmp(chbd,
                 blk,
                 scales_array,
                 end_scale,
                 scales_half,
                 scales_block,
                 out_len,
                 rows,
                 cols,
                 scale_length,
                 out_list)

    return np.float32(out_list)


cdef void _extract_values(DTYPE_uint8_t[:, :] block,
                          DTYPE_uint16_t[:] values,
                          DTYPE_uint16_t[:, :] rc_,
                          int fl) nogil:

    cdef:
        Py_ssize_t fi, fi_, fj_

    for fi in range(0, fl):

        fi_ = rc_[0, fi]
        fj_ = rc_[1, fi]

        values[fi] = block[fi_, fj_]


cdef void _get_direction(DTYPE_uint8_t[:, ::1] chunk,
                         int chunk_shape,
                         int rows_half,
                         int cols_half,
                         DTYPE_float32_t center_mean,
                         DTYPE_float32_t thresh_hom,
                         DTYPE_float32_t[::1] values_,
                         Py_ssize_t t_value,
                         bint is_row,
                         DTYPE_float32_t[::1] hist_,
                         Py_ssize_t hist_counter,
                         int skip_factor,
                         DTYPE_uint16_t[:, ::1] rc) nogil:

    cdef:
        Py_ssize_t ija, lni, lni_f, rc_shape
        DTYPE_float32_t ph_i, line_sd
        DTYPE_float32_t alpha_ = .1
        DTYPE_float32_t sfs_max, sfs_min, d_i, sfs_w_mean
        DTYPE_uint16_t[::1] line_values

    # Iterate over every other angle
    for ija from 0 <= ija < chunk_shape by skip_factor:

        ph_i = 0.

        # Draw a line between the two endpoints.
        if is_row:
            draw_line(rows_half, cols_half, ija, t_value, rc)
        else:
            draw_line(rows_half, cols_half, t_value, ija, rc)

        # rc_shape = rc.shape[1]
        rc_shape = rc[2, 0]  # the real line length

        # line_values = rc[0].copy()
        line_values = rc[3, :rc_shape]  # row of zeros, up to the line length

        # Extract the values along the line.
        _extract_values(chunk, line_values, rc, rc_shape)

        # Iterate over line values.
        lni_f = 0
        for lni in range(0, rc_shape):

            if ph_i < thresh_hom:

                # Pixel homogeneity
                ph_i += abs_f(center_mean - float(line_values[lni]))
                lni_f += 1

            else:
                break

        # Get the line length
        d_i = _get_line_length(float(rows_half), float(cols_half), float(rc[0, lni_f]), float(rc[1, lni_f]))

        # Get the standard deviation along the line.
        line_sd = _get_std_1d_uint16(line_values[:lni_f], lni_f)

        # Get the line statistics
        sfs_max = _get_max_sample(values_[0], d_i)
        sfs_min = _get_min_sample(values_[1], d_i)
        sfs_w_mean = (alpha_ * (d_i - 1.)) / line_sd

        # Update the histogram with
        #   the line length.
        if not npy_isnan(d_i) and not npy_isinf(d_i):
            hist_[hist_counter] = d_i

        hist_counter += 1

        if not npy_isnan(sfs_max) and not npy_isinf(sfs_max):
            values_[0] = sfs_max

        if (sfs_min != 0) and not npy_isnan(sfs_min) and not npy_isinf(sfs_min):
            values_[1] = sfs_min

        if not npy_isnan(d_i) and not npy_isinf(d_i):
            values_[2] += d_i

        if not npy_isnan(sfs_w_mean) and not npy_isinf(sfs_w_mean):
            values_[3] += sfs_w_mean


cdef void _get_directions(DTYPE_uint8_t[:, ::1] chunk,
                          int chunk_rws,
                          int chunk_cls,
                          int rows_half,
                          int cols_half,
                          DTYPE_float32_t center_mean,
                          DTYPE_float32_t thresh_hom,
                          DTYPE_float32_t[::1] values,
                          int skip_factor,
                          DTYPE_uint16_t[:, ::1] rcc_,
                          DTYPE_float32_t[::1] hist) nogil:

    """
    Returns:
        1: Length (maximum line length)
        2: Width (minimum line length)
        3: Mean
        4: w-mean
        5: Standard deviation
        6: Maximum ratio of orthogonal angles

        Not currently implemented:
            7: Minimum ratio of orthogonal angles
    """

    cdef:
        Py_ssize_t i_, j_, iia_, ija_, ia, ja, rr_shape, lni
        DTYPE_float32_t ph_i
        DTYPE_float32_t total_count
        Py_ssize_t hist_length = 0
        Py_ssize_t hist_counter = 0
        Py_ssize_t hist_counter_, ofc
        DTYPE_float32_t max_diff, orthog_diff

    # Get the histogram and row and column skip lengths.
    for i_ in range(0, 2):
        for iia_ from 0 <= iia_ < chunk_rws by skip_factor:
            hist_length += 1

    for j_ in range(0, 2):
        for ija_ from 0 <= ija_ < chunk_cls by skip_factor:
            hist_length += 1

    values[1] = 999999.

    # Fill the histogram

    # Rows, 1st column
    _get_direction(chunk, chunk_rws, rows_half, cols_half,
                   center_mean, thresh_hom, values, 0, True,
                   hist, hist_counter, skip_factor, rcc_)

    # Rows, last column
    _get_direction(chunk, chunk_rws, rows_half, cols_half,
                   center_mean, thresh_hom, values, chunk_cls-1, True,
                   hist, hist_counter, skip_factor, rcc_)

    # Columns, 1st row
    _get_direction(chunk, chunk_cls, rows_half, cols_half,
                   center_mean, thresh_hom, values, 0, False,
                   hist, hist_counter, skip_factor, rcc_)

    # Columns, last row
    _get_direction(chunk, chunk_cls, rows_half, cols_half,
                   center_mean, thresh_hom, values, chunk_rws-1, False,
                   hist, hist_counter, skip_factor, rcc_)

    values[2] /= float(hist_length)  # mean
    values[3] /= float(hist_length)  # w-mean

    # Calculate the standard deviation
    #   of the histogram.
    values[4] = _get_std_1d(hist, hist_length, values[2])

    # total_count = _get_sum1d(hist, hist_length)

    # values[3] /= total_count # H w-mean

    # Calculate the standard deviation
    #   of the histogram.
    # values[4] = _get_std_1d(hist, hist_length)

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
    for ija_ from 0 <= ija_ < chunk_cls by skip_factor:

        # Ratio of orthogonal angles
        orthog_diff = abs_f(hist[ija_] - float((ofc - hist_counter_)))
        max_diff = _get_max_sample(max_diff, orthog_diff)
        hist_counter_ += 1

    values[5] = max_diff


cdef void _sfs_feas(DTYPE_uint8_t[:, ::1] chunk,
                    unsigned int block_size,
                    DTYPE_float32_t thresh_hom,
                    unsigned int skip_factor,
                    DTYPE_uint16_t[:, ::1] rcc_,
                    DTYPE_float32_t[::1] hist_,
                    DTYPE_float32_t[::1] sfs_values) nogil:

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
        unsigned int chunk_rws, chunk_cls, rows_half, cols_half, block_half
        DTYPE_float32_t ctr_blk_mean, sfs_value
        DTYPE_uint8_t[:, ::1] chunk_block
        Py_ssize_t cbr, cbc

    # get chunk size
    chunk_rws = chunk.shape[0]
    chunk_cls = chunk.shape[1]

    rows_half = <int>(chunk_rws / 2.0)
    cols_half = <int>(chunk_cls / 2.0)

    block_half = <int>(block_size / 2.0)

    # Get the current window chunk
    chunk_block = chunk[rows_half-block_half:rows_half+block_half,
                        cols_half-block_half:cols_half+block_half]

    # Get the center block average
    if block_size > 1:

        cbr = chunk_block.shape[0]
        cbc = chunk_block.shape[1]

        ctr_blk_mean = _get_mean_uint8(chunk_block, cbr, cbc)

    else:
        ctr_blk_mean = chunk_block[rows_half, cols_half]

    _get_directions(chunk,
                    chunk_rws,
                    chunk_cls,
                    rows_half,
                    cols_half,
                    ctr_blk_mean,
                    thresh_hom,
                    sfs_values,
                    skip_factor,
                    rcc_,
                    hist_)


cdef void _feature_sfs(DTYPE_uint8_t[:, ::1] ch_bd,
                       unsigned int block_size,
                       DTYPE_uint16_t[::1] scales_array,
                       int n_scales,
                       DTYPE_float32_t thresh_hom,
                       int scales_half,
                       int scales_block,
                       int out_len,
                       int rows,
                       int cols,
                       int skip_factor,
                       DTYPE_uint16_t[:, ::1] rcc_,
                       DTYPE_float32_t[::1] hist_,
                       DTYPE_float32_t[::1] out_list_):

    cdef:
        Py_ssize_t i, j, ki, k_half, st_
        DTYPE_uint16_t k
        DTYPE_uint8_t[:, ::1] ch_bd_
        DTYPE_float32_t[::1] sts = np.zeros(6, dtype='float32')
        DTYPE_float32_t[::1] sts_ = sts.copy()
        int pix_ctr = 0

    with nogil:

        if block_size > 1:

            for i from 0 <= i < rows-scales_block by block_size:

                for j from 0 <= j < cols-scales_block by block_size:

                    for ki in range(0, n_scales):

                        k = scales_array[ki]

                        k_half = <int>(k / 2.0)

                        ch_bd_ = ch_bd[i+scales_half-k_half:i+scales_half-k_half+k,
                                       j+scales_half-k_half:j+scales_half-k_half+k]

                        sts_[...] = sts

                        _sfs_feas(ch_bd_, block_size, thresh_hom, skip_factor, rcc_, hist_, sts_)

                        for st_ in range(0, 6):

                            out_list_[pix_ctr] = sts_[st_]

                            pix_ctr += 1

        else:

            for i in range(0, rows-scales_block):

                for j in range(0, cols-scales_block):

                    for ki in range(0, n_scales):

                        k = scales_array[ki]

                        k_half = <int>(k / 2.0)

                        ch_bd_ = ch_bd[i+scales_half-k_half:i+scales_half-k_half+k,
                                       j+scales_half-k_half:j+scales_half-k_half+k]

                        sts_[...] = sts

                        _sfs_feas(ch_bd_, block_size, thresh_hom, skip_factor, rcc_, hist_, sts_)

                        for st_ in range(0, 6):

                            out_list_[pix_ctr] = sts_[st_]

                            pix_ctr += 1


def feature_sfs(DTYPE_uint8_t[:, ::1] chbd,
                unsigned int block_size,
                list scales,
                unsigned int end_scale,
                DTYPE_float32_t thresh_hom,
                unsigned int skip_factor=4):

    cdef:
        Py_ssize_t i, j, ki, k
        int scales_half = <int>(end_scale / 2.0)
        int scales_block = end_scale - block_size
        int rows = chbd.shape[0]
        int cols = chbd.shape[1]
        DTYPE_uint16_t[::1] scales_array = np.array(scales, dtype='uint16')
        int scale_length = scales_array.shape[0]
        DTYPE_uint16_t[:, ::1] rcc = np.zeros((4, end_scale), dtype='uint16')
        DTYPE_float32_t[::1] histogram = np.zeros(end_scale, dtype='float32')
        DTYPE_float32_t[::1] out_list
        unsigned int out_len = _get_output_length(rows, cols, scales_block, block_size, scale_length, 6)

    out_list = np.zeros(out_len, dtype='float32')

    _feature_sfs(chbd,
                 block_size,
                 scales_array,
                 scale_length,
                 thresh_hom,
                 scales_half,
                 scales_block,
                 out_len,
                 rows,
                 cols,
                 skip_factor,
                 rcc,
                 histogram,
                 out_list)

    return np.float32(out_list)


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
#             for ki in range(0, n_scales):
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
#             for ki in range(0, n_scales):
#
#                 sts = _feature_surf(chBd[i+scales_half-k_half:i+scales_half-k_half+k,
#                                     j+scales_half-k_half:j+scales_half-k_half+k], kPts, j, i, k, scs)
#
#                 for st in range(0, 4):
#
#                     out_list[pix_ctr] = sts[st]
#
#                     pix_ctr += 1
#
#     return out_list


# ORB keypoints
# Ethan Rublee, Vincent Rabaud, Kurt Konolige, Gary R. Bradski:
#   ORB: An efficient alternative to SIFT or SURF. ICCV 2011: 2564-2571.

def fill_key_points(DTYPE_float32_t[:, ::1] in_block, list key_point_list):

    cdef:
        Py_ssize_t key_point_index
        int n_key_points = len(key_point_list)
        int brows = in_block.shape[0]
        int bcols = in_block.shape[1]
        double key_y, key_x
        int key_y_idx, key_x_idx
        DTYPE_uint8_t[:, ::1] key_point_array = np.zeros((brows, bcols), dtype='uint8')

    for key_point_index in range(0, n_key_points):

        key_x, key_y = key_point_list[key_point_index].pt

        key_y_idx = <int>(floor(key_y))
        key_x_idx = <int>(floor(key_x))

        key_point_array[key_y_idx, key_x_idx] = 1

    return np.uint8(key_point_array)


cdef DTYPE_float32_t[::1] _pyramid_hist_sift(DTYPE_uint8_t[:, ::1] key_point_array,
                                             DTYPE_float32_t[::1] levels,
                                             int orb_rows,
                                             int orb_cols,
                                             DTYPE_float32_t[::1] hist_) nogil:

    cdef:
        Py_ssize_t lv, ki, kj, grid_counter
        int rr_rows, cc_cols, y_tiles, x_tiles
        DTYPE_uint8_t[:, ::1] kblock

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
                            hist_[grid_counter] += _get_sum_uint8(kblock, rr_rows, cc_cols)

                            grid_counter += 1

    return hist_[:grid_counter]


cdef void _feature_orb(DTYPE_uint8_t[:, ::1] ch_bd,
                       int blk,
                       DTYPE_uint16_t[::1] scales_array,
                       int scales_half,
                       int scales_block,
                       int scale_length,
                       int out_len,
                       int rows,
                       int cols,
                       int scales_length,
                       int end_scale,
                       DTYPE_float32_t[::1] out_list_):

    cdef:
        Py_ssize_t i, j, ki, st
        DTYPE_uint16_t k
        int k_half
        DTYPE_float32_t[::1] levels = np.array([2, 4, 8], dtype='float32')
        Py_ssize_t pix_ctr = 0
        int block_rows, block_cols
        DTYPE_uint8_t[:, ::1] ch_bd_sub
        DTYPE_float32_t[::1] sts = np.zeros(5, dtype='float32')
        DTYPE_float32_t[::1] sts_ = sts.copy()
        DTYPE_float32_t[::1] hist = np.zeros(end_scale*end_scale*4, dtype='float32')
        DTYPE_float32_t[::1] hist_ = hist.copy()

    with nogil:
    
        for i from 0 <= i < rows-scales_block by blk:

            for j from 0 <= j < cols-scales_block by blk:

                for ki in range(0, scale_length):

                    k = scales_array[ki]

                    k_half = <int>(k / 2.)

                    ch_bd_sub = ch_bd[i+scales_half-k_half:i+scales_half-k_half+k,
                                      j+scales_half-k_half:j+scales_half-k_half+k]

                    block_rows = ch_bd_sub.shape[0]
                    block_cols = ch_bd_sub.shape[1]

                    if _get_max(ch_bd_sub, block_rows, block_cols) > 0:

                        sts_[...] = sts
                        hist_[...] = hist

                        _get_moments(_pyramid_hist_sift(ch_bd_sub, levels, block_rows, block_cols, hist_), sts_)

                        for st in range(0, 5):

                            out_list_[pix_ctr] = sts_[st]

                            pix_ctr += 1

                    else:
                        pix_ctr += 5


def feature_orb(DTYPE_uint8_t[:, ::1] chbd,
                int blk,
                list scs,
                int end_scale):

    cdef:
        Py_ssize_t i, j, ki
        int scales_half = <int>(end_scale / 2.)
        int scales_block = end_scale - blk
        int rows = chbd.shape[0]
        int cols = chbd.shape[1]
        DTYPE_uint16_t[::1] scales_array = np.array(scs, dtype='uint16')
        int scale_length = scales_array.shape[0]
        unsigned int out_len = _get_output_length(rows, cols, scales_block, blk, scale_length, 5)
        DTYPE_float32_t[::1] out_list = np.zeros(out_len, dtype='float32')

    _feature_orb(chbd,
                 blk,
                 scales_array,
                 scales_half,
                 scales_block,
                 scale_length,
                 out_len,
                 rows,
                 cols,
                 scale_length,
                 end_scale,
                 out_list)

    return np.float32(out_list)


cdef DTYPE_uint8_t[:, :, :] _set_lbp(DTYPE_uint8_t[:, ::1] chbd,
                                     int rows,
                                     int cols,
                                     DTYPE_uint8_t[::1] p_range,
                                     dict rdict):

    """
    Get the Local Binary Patterns
    """

    cdef:
        Py_ssize_t scsc
        unsigned int p_len = 3
        DTYPE_uint8_t[:, :, :] lbp_bd = np.zeros((p_len, rows, cols), dtype='uint8')
        DTYPE_uint8_t[:, :] lbp_array

    # Run LBP for each scale
    for scsc in range(0, p_len):

        lbp_array = np.uint8(LBP(np.uint8(chbd),
                                 int(p_range[scsc]),
                                 int(rdict[int(p_range[scsc])]),
                                 method='uniform'))

        lbp_bd[scsc, :, :] = lbp_array[...]

    return lbp_bd


cdef np.ndarray[DTYPE_float32_t, ndim=1] _feature_lbp(DTYPE_uint8_t[:, ::1] chbd,
                                                      int blk,
                                                      list scs,
                                                      int end_scale):

    cdef:
        Py_ssize_t i, j, ki, sti
        unsigned int pc, pr_bin_count, k_half
        int rows = chbd.shape[0]
        int cols = chbd.shape[1]
        DTYPE_uint8_t[:, :, ::1] lbp_bd
        DTYPE_uint8_t[:, :, ::1] ch_bd
        unsigned int scales_half = <int>(end_scale / 2.)
        unsigned int scales_block = end_scale - blk
        DTYPE_float32_t[::1] sts, out_list
        unsigned int pix_ctr = 0
        DTYPE_uint16_t k
        DTYPE_uint16_t[:] scales_array = np.array(scs, dtype='uint16')
        int scale_length = scales_array.shape[0]
        np.ndarray[DTYPE_float32_t, ndim=1] out_list_a
        unsigned int out_len
        DTYPE_uint8_t[::1] p_range = np.array([8, 16, 32], dtype='uint8')
        dict rdict	= {4: 1, 8: 1, 16: 2, 32: 4, 64: 8, 128: 16}

    # get the LBP images
    lbp_bd = np.ascontiguousarray(_set_lbp(chbd, rows, cols, p_range, rdict))

    # count of bins for all p,r LBP pairs
    pr_bin_count = np.sum([pr+2 for pr in p_range])

    out_len = _get_output_length(rows, cols, scales_block, blk, scale_length, pr_bin_count)

    # set the output list
    out_list = np.empty(out_len, dtype='float32')

    for i from 0 <= i < rows-scales_block by blk:

        for j from 0 <= j < cols-scales_block by blk:

            for ki in range(0, scale_length):

                k = scales_array[ki]

                k_half = <int>(k / 2.0)

                ch_bd = np.ascontiguousarray(np.uint8(lbp_bd[:,
                                                             i+scales_half-k_half:i+scales_half-k_half+k,
                                                             j+scales_half-k_half:j+scales_half-k_half+k]))

                # get histograms and concatenate
                sts = np.float32(np.ascontiguousarray(np.concatenate([np.bincount(np.uint8(ch_bd[pc]).flat,
                                                                                  minlength=pc+2) for pc in range(0, 3)])))

                for sti in range(0, 4):

                    out_list[pix_ctr] = sts[sti]

                    pix_ctr += 1

    out_list_a = np.float32(out_list)

    out_list_a[np.isnan(out_list_a) | np.isinf(out_list_a)] = 0

    return out_list_a


def feature_lbp(np.ndarray[DTYPE_uint8_t, ndim=2] chbd, int blk, list scs, int end_scale):
    return _feature_lbp(chbd, blk, scs, end_scale)


cdef void _feature_lbpm(DTYPE_uint8_t[:, ::1] chBd,
                        int blk,
                        DTYPE_uint16_t[:] scs,
                        int end_scale,
                        int scales_half,
                        int scales_block,
                        int out_len,
                        int rows,
                        int cols,
                        int scale_length,
                        DTYPE_float32_t[::1] out_list_):

    """
    At each scale, returns:
        1: Mean
        2: Variance
        3: Skew
        4: Kurtosis
    """

    cdef:
        Py_ssize_t i, j, sti, ki
        unsigned int pc, k_half, k
        DTYPE_uint8_t[:, :, ::1] lbp_bd
        DTYPE_uint8_t[:, :, ::1] ch_bd
        int pix_ctr = 0
        DTYPE_float32_t[::1] sts = np.zeros(5, dtype='float32')
        DTYPE_float32_t[::1] sts_ = sts.copy()
        DTYPE_float32_t[::1] lbp_results
        DTYPE_uint8_t[::1] p_range = np.array([8, 16, 32], dtype='uint8')
        dict rdict	= {4: 1, 8: 1, 16: 2, 32: 4, 64: 8, 128: 16}

    # get the LBP images
    lbp_bd = np.ascontiguousarray(_set_lbp(chBd, rows, cols, p_range, rdict))

    for i from 0 <= i < rows-scales_block by blk:

        for j from 0 <= j < cols-scales_block by blk:

            for ki in range(0, scale_length):

                k = scs[ki]

                k_half = <int>(k / 2.0)

                ch_bd = np.ascontiguousarray(np.uint8(lbp_bd[:,
                                                             i+scales_half-k_half:i+scales_half-k_half+k,
                                                             j+scales_half-k_half:j+scales_half-k_half+k]))

                # get histograms and concatenate
                lbp_results = np.float32(np.ascontiguousarray(np.concatenate([np.bincount(np.uint8(ch_bd[pc]).flat,
                                                                                          minlength=pc+2)
                                                                              for pc in range(0, 3)])))

                sts_[...] = sts

                _get_moments(lbp_results, sts_)

                for sti in range(0, 5):

                    out_list_[pix_ctr] = sts_[sti]

                    pix_ctr += 1


def feature_lbpm(np.ndarray[DTYPE_uint8_t, ndim=2] chbd, int blk, list scs, int end_scale):

    cdef:
        Py_ssize_t i, j, ki
        int scales_half = <int>(end_scale / 2.)
        int scales_block = end_scale - blk
        int rows = chbd.shape[0]
        int cols = chbd.shape[1]
        DTYPE_uint16_t[:] scales_array = np.array(scs, dtype='uint16')
        int scale_length = scales_array.shape[0]
        unsigned int out_len = _get_output_length(rows, cols, scales_block, blk, scale_length, 5)
        DTYPE_float32_t[::1] out_list = np.zeros(out_len, dtype='float32')

    _feature_lbpm(chbd,
                  blk,
                  scales_array,
                  end_scale,
                  scales_half,
                  scales_block,
                  out_len,
                  rows,
                  cols,
                  scale_length,
                  out_list)

    return np.float32(out_list)


cdef inline DTYPE_float32_t _get_distance(tuple line):
    return sqrt(pow((line[0][0] - line[1][0]), 2.) + pow((line[0][1] - line[1][1]), 2.))


cdef inline DTYPE_float64_t get_slope(tuple line):
    return np.degrees(atan(float((line[0][1] - line[1][1])) / float((line[1][0] - line[0][0]))))


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
            for ki in range(0, scale_length):

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


def feature_hough(np.ndarray[DTYPE_uint8_t, ndim=2] chBd, int blk, list scs, int end_scale, int threshold,
                  int min_len, int line_gap):

    cdef:
        Py_ssize_t i, j, k
        int rows = chBd.shape[0]
        int cols = chBd.shape[1]
        list sts
        DTYPE_float64_t st
        int scales_half = end_scale / 2
        int scales_block = end_scale - blk
        DTYPE_uint16_t[:] scales_array = np.array(scs, dtype='uint16')
        int scale_length = scales_array.shape[0]
        unsigned int out_len = _get_output_length(rows, cols, scales_block, blk, scale_length, 4)

    return _feature_hough(chBd, blk, scales_array, scales_half, scales_block, scale_length,
                          out_len, rows, cols, threshold, min_len, line_gap, end_scale)


# PanTex

cdef void _glcm_loop(DTYPE_uint8_t[:, :] image,
                     DTYPE_float32_t[:] distances,
                     DTYPE_float32_t[:] angles,
                     int levels,
                     DTYPE_float32_t[:, :, :, ::1] out,
                     DTYPE_float32_t[:, :] out_sums,
                     Py_ssize_t rows,
                     Py_ssize_t cols) nogil:

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


cdef DTYPE_float32_t[:, :, :, ::1] _norm_glcm(DTYPE_float32_t[:, :, :, ::1] Pt,
                                              DTYPE_float32_t[:, :] Pt_sums,
                                              DTYPE_float32_t[:] distances,
                                              DTYPE_float32_t[:] angles, int levels,
                                              DTYPE_float32_t[:, :, :, ::1] glcm_normed_) nogil:

    cdef:
        Py_ssize_t a_idx, d_idx, r, c
        Py_ssize_t angles_, distances_
        DTYPE_float32_t angle_dist_sum

    angles_ = angles.shape[0]
    distances_ = distances.shape[0]

    # Get the sums
    for a_idx in range(0, angles_):

        for d_idx in range(0, distances_):

            # angle_dist_sum = 0.

            # Iterate over the co-occurrence array
            #   and normalize.
            for r in range(0, levels):

                for c in range(0, levels):
                    glcm_normed_[r, c, d_idx, a_idx] += (Pt[r, c, d_idx, a_idx] / Pt_sums[d_idx, a_idx])

    return glcm_normed_


# cdef _check_nans(DTYPE_float32_t[:, :, :, :] glcm_mat_nan,
#                  DTYPE_float32_t[:] distances,
#                  DTYPE_float32_t[:] angles,
#                  int levels) nogil:
#
#     cdef:
#         Py_ssize_t a_idx, d_idx, r, c
#         Py_ssize_t angles_, distances_
#         DTYPE_float32_t value2check
#
#     angles_ = angles.shape[0]
#     distances_ = distances.shape[0]
#
#     # Get the sums
#     for a_idx in range(0, angles_):
#
#         for d_idx in range(0, distances_):
#
#             # angle_dist_sum = 0.
#
#             # Iterate over the image
#             for r in range(0, levels):
#
#                 for c in range(0, levels):
#
#                     value2check = glcm_mat_nan[r, c, d_idx, a_idx]
#
#                     if npy_isnan(value2check) or npy_isinf(value2check):
#                         glcm_mat_nan[r, c, d_idx, a_idx] = 0


cdef DTYPE_float32_t[:, :, :, ::1] _greycomatrix(DTYPE_uint8_t[:, :] image,
                                                 DTYPE_float32_t[:] distances,
                                                 DTYPE_float32_t[:] angles,
                                                 Py_ssize_t levels, Py_ssize_t rows, Py_ssize_t cols,
                                                 DTYPE_float32_t[:, :, :, ::1] P,
                                                 DTYPE_float32_t[:, :] angle_dist_sums,
                                                 DTYPE_float32_t[:, :, :, ::1] glcm_normed) nogil:

    cdef:
        metric_ptr metric_norm

    metric_norm = &_norm_glcm

    # count co-occurences
    _glcm_loop(image, distances, angles, levels, P, angle_dist_sums, rows, cols)

    # Normalize the matrix
    return metric_norm(P, angle_dist_sums, distances, angles, levels, glcm_normed)


cdef DTYPE_float32_t _glcm_contrast(DTYPE_float32_t[:, :, :, ::1] P,
                                    DTYPE_float32_t[:] distances,
                                    DTYPE_float32_t[:] angles,
                                    Py_ssize_t levels,
                                    DTYPE_float32_t[:, ::1] contrast_array) nogil:

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
            min_contrast = _get_min_sample(min_contrast, contrast_sum)

    return min_contrast


cdef DTYPE_float32_t[:, ::1] _set_contrast_weights(int levels):

    cdef:
        Py_ssize_t li, lj
        DTYPE_float32_t[:, ::1] contrast_array = np.zeros((levels, levels), dtype='float32')

    for li in range(0, levels):
        for lj in range(0, levels):
            contrast_array[li, lj] = pow2(float(li-lj))

    return contrast_array


cdef void _feature_pantex(DTYPE_uint8_t[:, ::1] chBd,
                          int blk,
                          DTYPE_uint16_t[::1] scs,
                          int scales_half,
                          int scales_block,
                          int out_len,
                          bint weighted,
                          int rows,
                          int cols,
                          int scale_length,
                          int levels,
                          DTYPE_float32_t[::1] out_list_):

    """
    Calculates the Anisotropic Built-up Presence Index (PanTex)

    The GLCM code was adapted from the Scikit-image team
    @ https://github.com/scikit-image/scikit-image/blob/master/skimage/feature/_texture.pyx
    """

    cdef:
        Py_ssize_t i, j, ki, block_rows, block_cols
        DTYPE_uint16_t k
        int k_half
        DTYPE_uint8_t[:, ::1] ch_bd
        DTYPE_float32_t pi = 3.14159265
        DTYPE_float32_t[:, :, :, ::1] glcm_mat
        DTYPE_float32_t con_min
        Py_ssize_t pix_ctr = 0

        # directions [E, NE, N, NW]
        DTYPE_float32_t[:] disp_vect = np.array([0., pi / 6., pi / 4., pi / 3., pi / 2., (2. * pi) / 3.,
                                                 (3. * pi) / 4., (5. * pi) / 6.], dtype='float32')

        DTYPE_float32_t[:] dists = np.array([1, 2], dtype='float32')
        DTYPE_float32_t[:, ::1] contrast_weights = _set_contrast_weights(levels)

        DTYPE_float32_t[:, :, :, ::1] P_ = np.zeros((levels, levels, dists.shape[0], disp_vect.shape[0]),
                                                    dtype='float32')

        DTYPE_float32_t[:, :] angle_dist_sums_ = np.zeros((dists.shape[0], disp_vect.shape[0]),
                                                          dtype='float32')

        DTYPE_float32_t[:, :, :, ::1] glcm_normed_ = P_.copy()
        DTYPE_float32_t[:, :, :, ::1] P_c = P_.copy()
        DTYPE_float32_t[:, :] angle_dist_sums_c = angle_dist_sums_.copy()
        DTYPE_float32_t[:] mean_var_values
        DTYPE_float32_t[:, ::1] kernel_weight
        DTYPE_float32_t[::1] in_zs = np.zeros(2, dtype='float32')
        DTYPE_float32_t[:, ::1] dist_weights
        list dist_weights_m = []

    if weighted:

        for ki in range(0, scale_length):

            k = scs[ki]
            k_half = <int>(k / 2.)
            rs = (scales_half - k_half + k) - (scales_half - k_half)
            cs = (scales_half - k_half + k) - (scales_half - k_half)

            dist_weights = np.empty((rs, cs), dtype='float32')
            dist_weights_m.append(_create_weights(dist_weights, rs, cs))

        with nogil:

            for i from 0 <= i < rows-scales_block by blk:

                for j from 0 <= j < cols-scales_block by blk:

                    for ki in range(0, scale_length):

                        k = scs[ki]

                        k_half = <int>(k / 2.)

                        ch_bd = chBd[i+scales_half-k_half:i+scales_half-k_half+k,
                                     j+scales_half-k_half:j+scales_half-k_half+k]

                        block_rows = ch_bd.shape[0]
                        block_cols = ch_bd.shape[1]

                        if _get_max(ch_bd, block_rows, block_cols) == 0:
                            con_min = 0.
                        else:

                            P_c[...] = P_
                            angle_dist_sums_c[...] = angle_dist_sums_
                            glcm_normed_[...] = P_

                            glcm_mat = _greycomatrix(ch_bd,
                                                     dists,
                                                     disp_vect,
                                                     levels,
                                                     block_rows,
                                                     block_cols,
                                                     P_c,
                                                     angle_dist_sums_c,
                                                     glcm_normed_)

                            con_min = _glcm_contrast(glcm_mat,
                                                     dists,
                                                     disp_vect,
                                                     levels,
                                                     contrast_weights)

                        with gil:
                            kernel_weight = dist_weights_m[ki]

                        _get_weighted_mean_var_byte(ch_bd, kernel_weight, block_rows, block_cols, in_zs)

                        if not npy_isnan(con_min) and not npy_isinf(con_min):
                            out_list_[pix_ctr] = con_min * in_zs[0]

                        pix_ctr += 1

    else:

        with nogil:

            for i from 0 <= i < rows-scales_block by blk:

                for j from 0 <= j < cols-scales_block by blk:

                    for ki in range(0, scale_length):

                        k = scs[ki]

                        k_half = <int>(k / 2.)

                        ch_bd = chBd[i+scales_half-k_half:i+scales_half-k_half+k,
                                     j+scales_half-k_half:j+scales_half-k_half+k]

                        block_rows = ch_bd.shape[0]
                        block_cols = ch_bd.shape[1]

                        if _get_max(ch_bd, block_rows, block_cols) == 0:
                            con_min = 0.
                        else:

                            P_c[...] = P_
                            angle_dist_sums_c[...] = angle_dist_sums_
                            glcm_normed_[...] = P_

                            glcm_mat = _greycomatrix(ch_bd,
                                                     dists,
                                                     disp_vect,
                                                     levels,
                                                     block_rows,
                                                     block_cols,
                                                     P_c,
                                                     angle_dist_sums_c,
                                                     glcm_normed_)

                            con_min = _glcm_contrast(glcm_mat, dists, disp_vect, levels, contrast_weights)

                        if not npy_isnan(con_min) and not npy_isinf(con_min):
                            out_list_[pix_ctr] = con_min

                        pix_ctr += 1


def feature_pantex(DTYPE_uint8_t[:, ::1] chbd, int blk, list scs, int end_scale, bint weighted, int levels=32):

    cdef:
        Py_ssize_t i, j, ki
        int scales_half = <int>(end_scale / 2.)
        int scales_block = end_scale - blk
        int rows = chbd.shape[0]
        int cols = chbd.shape[1]
        DTYPE_uint16_t[::1] scales_array = np.array(scs, dtype='uint16')
        int scale_length = scales_array.shape[0]
        unsigned int out_len = _get_output_length(rows, cols, scales_block, blk, scale_length, 1)
        DTYPE_float32_t[::1] out_list = np.zeros(out_len, dtype='float32')

    _feature_pantex(chbd,
                    blk,
                    scales_array,
                    scales_half,
                    scales_block,
                    out_len,
                    weighted,
                    rows,
                    cols,
                    scale_length,
                    levels,
                    out_list)

    return np.float32(out_list)


cdef DTYPE_float32_t[:, ::1] _create_weights(DTYPE_float32_t[:, ::1] dist_weights, int rs, int cs) nogil:

    cdef:
        Py_ssize_t ri, rj
        DTYPE_float32_t rm = rs / 2.
        DTYPE_float32_t cm = cs / 2.

    for ri in range(0, rs):
        for rj in range(0, cs):
            dist_weights[ri, rj] = _euclidean_distance(cm, rm, float(rj), float(ri))

    return dist_weights


cdef void feature_mean_float32(DTYPE_float32_t[:, ::1] ch_bd,
                               unsigned int blk,
                               DTYPE_uint16_t[::1] scs,
                               unsigned int scales_half,
                               unsigned int scales_block,
                               unsigned int scale_length,
                               DTYPE_float32_t[:, :, ::1] dist_weights_stack,
                               DTYPE_float32_t[::1] in_zs,
                               DTYPE_float32_t[::1] out_list_):

    cdef:
        Py_ssize_t i, j, ki, pix_ctr, pi
        DTYPE_uint16_t k
        unsigned int k_half, r_size, c_size
        unsigned int rows = ch_bd.shape[0]
        unsigned int cols = ch_bd.shape[1]
        DTYPE_float32_t[:, ::1] block_chunk, dw

    pix_ctr = 0

    with nogil:

        for i from 0 <= i < rows-scales_block by blk:

            for j from 0 <= j < cols-scales_block by blk:

                for ki in range(0, scale_length):

                    k = scs[ki]

                    k_half = <int>(k / 2.)

                    #rc_start = scales_half - k_half
                    #rc_end = scales_half - k_half + k

                    #r_size = (i + rc_end) - (i + rc_start) if (i + rc_end) - (i + rc_start) <= rows else rows - (i + rc_start)
                    #c_size = (j + rc_end) - (j + rc_start) if (j + rc_end) - (j + rc_start) <= cols else cols - (j + rc_start)

                    block_chunk = ch_bd[i+scales_half-k_half:i+scales_half-k_half+k,
                                        j+scales_half-k_half:j+scales_half-k_half+k]

                    r_size = block_chunk.shape[0]
                    c_size = block_chunk.shape[1]

                    dw = dist_weights_stack[ki, :r_size, :c_size]

                    _get_weighted_mean_var(block_chunk, dw, r_size, c_size, in_zs)

                    for pi in range(0, 2):

                        out_list_[pix_ctr] = in_zs[pi]

                        pix_ctr += 1


def feature_mean(DTYPE_float32_t[:, ::1] ch_bd, int blk, list scs, int end_scale):

    cdef:
        Py_ssize_t i, j, ki
        unsigned int scales_half = <int>(end_scale / 2.)
        unsigned int scales_block = end_scale - blk
        unsigned int rows = ch_bd.shape[0]
        unsigned int cols = ch_bd.shape[1]
        DTYPE_uint16_t[::1] scales_array = np.array(scs, dtype='uint16')
        unsigned int scale_length = scales_array.shape[0]
        unsigned int k, k_half, rc_start, rc_end, rc
        DTYPE_float32_t[:, :, ::1] dist_weights_stack = np.zeros((scale_length, end_scale*2, end_scale*2), dtype='float32')
        DTYPE_float32_t[:, ::1] dist_weights
        DTYPE_float32_t[::1] in_zs = np.zeros(2, dtype='float32')
        unsigned int out_len = _get_output_length(rows, cols, scales_block, blk, scale_length, 2)
        DTYPE_float32_t[::1] out_list = np.zeros(out_len, dtype='float32')

    for ki in range(0, scale_length):

        k = scales_array[ki]
        k_half = <int>(k / 2.)

        rc_start = scales_half - k_half
        rc_end = scales_half - k_half + k

        rc = rc_end - rc_start

        dist_weights = np.zeros((rc, rc), dtype='float32')

        dist_weights_stack[ki, :rc, :rc] = _create_weights(dist_weights, rc, rc)

    feature_mean_float32(ch_bd,
                         blk,
                         scales_array,
                         scales_half,
                         scales_block,
                         scale_length,
                         dist_weights_stack,
                         in_zs,
                         out_list)

    return np.float32(out_list)


# def feaCtrFloat64(np.ndarray[DTYPE_float64_t, ndim=2] chBd, int blk, list scs, int rows, int cols):
#
#     cdef int i, j, k
#
#     return [ chBd[i+scs[-1]/2, j+scs[-1]/2] for k in scs for i in range(0, rows-(scs[-1]-blk), blk) for j in range(0, cols-(scs[-1]-blk), blk) ]
#
#
# def feaCtrFloat32(np.ndarray[DTYPE_float32_t, ndim=2] chBd, int blk, list scs, int rows, int cols):
#
#     cdef int i, j, k
#
#     return [ chBd[i+scs[-1]/2, j+scs[-1]/2] for k in scs for i in range(0, rows-(scs[-1]-blk), blk) for j in range(0, cols-(scs[-1]-blk), blk) ]
#
#
# def feaCtr_uint16(np.ndarray[unsigned short, ndim=2] chBd, int blk, list scs, int rows, int cols):
#
#     cdef int i, j, k
#
#     return [ chBd[i+scs[-1]/2, j+scs[-1]/2] for k in scs for i in range(0, rows-(scs[-1]-blk), blk) for j in range(0, cols-(scs[-1]-blk), blk) ]
#
#
# def feaCtr_uint8(np.ndarray[unsigned char, ndim=2] chBd, int blk, list scs, int rows, int cols):
#
#     cdef int i, j, k
#
#     return [ chBd[i+scs[-1]/2, j+scs[-1]/2] for k in scs for i in range(0, rows-(scs[-1]-blk), blk) for j in range(0, cols-(scs[-1]-blk), blk) ]
#
#
# def feaCtr_uint(np.ndarray[int, ndim=2] chBd, int blk, list scs, int rows, int cols):
#
#     cdef int i, j, k
#
#     return [ chBd[i+scs[-1]/2, j+scs[-1]/2] for k in scs for i in range(0, rows-(scs[-1]-blk), blk) for j in range(0, cols-(scs[-1]-blk), blk) ]
#
#
# def feaCtr(np.ndarray chBd, int blk, list scs):
#
#     cdef int rows = chBd.shape[0]
#     cdef int cols = chBd.shape[1]
#
#     if chBd.dtype == 'float64':
#         return feaCtrFloat64(chBd, blk, scs, rows, cols)
#     elif chBd.dtype == 'float32':
#         return feaCtrFloat32(chBd, blk, scs, rows, cols)
#     elif chBd.dtype == 'uint16':
#         return feaCtr_uint16(chBd, blk, scs, rows, cols)
#     elif chBd.dtype == 'uint8':
#         try:
#             return feaCtr_uint8(chBd, blk, scs, rows, cols)
#         except:
#             return feaCtr_uint(chBd, blk, scs, rows, cols)


# Lacunarity

cdef int max_box_number(DTYPE_uint8_t[:, :] w, int rr_rows, int rr_cols) nogil:

    cdef:
        int maxi = _get_max(w, rr_rows, rr_cols)
        int mini = _get_min(w, rr_rows, rr_cols)

        int boxes_max = <int>(ceil(float(maxi) / _get_min_sample_i(rr_rows, rr_cols)))
        int boxes_min = <int>(ceil(float(mini) / _get_min_sample_i(rr_rows, rr_cols)))

    return <int>((boxes_max - boxes_min) + 1)


cdef void _div1d(DTYPE_float32_t[:] array1d, int cs, DTYPE_float32_t div_value) nogil:

    cdef:
        Py_ssize_t js

    for js in range(0, cs):
        array1d[js] /= div_value


cdef DTYPE_float32_t _lacunarity(DTYPE_uint8_t[:, ::1] chunk_sub,
                                 int r,
                                 DTYPE_float32_t[::1] zs) nogil:

    cdef:
        int rows_ = chunk_sub.shape[0]
        int cols_ = chunk_sub.shape[1]

        # Get max for probability.
        int maxw = (_get_max(chunk_sub, rows_, cols_) - _get_min(chunk_sub, rows_, cols_)) + 1

        # Get the maximum number of boxes in each block
        #   ceiling is needed for uneven rows or columns.
        int n_rows_ = <int>(ceil(float(rows_) / r))
        int n_cols_ = <int>(ceil(float(cols_) / r))

        int ns = <int>(float(n_rows_) * float(n_cols_))

        # Create array of zeros
        DTYPE_float32_t[::1] nsr = zs[:ns]

        int maxww = maxw + 1

        DTYPE_float32_t[::1] nqr = zs[:maxww]

        int nn = 0
        Py_ssize_t mm, n, dd
        int rr_rows, rr_cols
        DTYPE_uint8_t[:, ::1] w
        int m

        # DTYPE_float32_t[:] nqr_sum
        DTYPE_float32_t smn, l2_sum
        DTYPE_float32_t[::1] l1 = zs[:ns]
        DTYPE_float32_t[::1] l2 = zs[:ns]
        DTYPE_float32_t ns_rp

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
    _div1d(nqr, maxww, float(ns))

    # Calculate moments
    #   L1 = MASS squared times the probability
    #   (both are arrays of length equal to the number of boxes in k x k window).
    for dd in range(0, ns):

        ns_rp = nsr[dd]

        l1[dd] = pow2(ns_rp) * nqr[<int>ns_rp]
        l2[dd] = ns_rp * nqr[<int>ns_rp]

    # Sum the L2 array and square the result
    smn = _get_sum1d(l2, ns)
    l2_sum = pow2(smn)

    # Lacunarity for k x k block
    if l2_sum != 0:
        return _get_sum1d(l1, ns) / l2_sum
    else:
        return 0.


cdef void _feature_lacunarity(DTYPE_uint8_t[:, ::1] chunk_block,
                              int blk,
                              DTYPE_uint16_t[::1] scales,
                              int scales_half,
                              int scales_block,
                              int rows,
                              int cols,
                              int r,
                              int out_len,
                              int scale_length,
                              DTYPE_float32_t[::1] zs,
                              DTYPE_float32_t[::1] out_list_):

    cdef:
        Py_ssize_t i, j, ki, cr, cc
        unsigned int k, k_half
        Py_ssize_t pixel_counter = 0
        DTYPE_uint8_t[:, ::1] ch_bd

    with nogil:

        for i from 0 <= i < rows-scales_block by blk:

            for j from 0 <= j < cols-scales_block by blk:

                for ki in range(0, scale_length):

                    k = scales[ki]
                    k_half = <int>(k / 2.)

                    ch_bd = chunk_block[i+scales_half-k_half:i+scales_half-k_half+k,
                                        j+scales_half-k_half:j+scales_half-k_half+k]

                    cr = ch_bd.shape[0]
                    cc = ch_bd.shape[1]

                    if _get_max(ch_bd, cr, cc) == 0:
                        out_list_[pixel_counter] = 0
                    else:
                        out_list_[pixel_counter] = _lacunarity(ch_bd, r, zs)

                    pixel_counter += 1


def feature_lacunarity(DTYPE_uint8_t[:, ::1] chunk_block, int blk, list scales, int end_scale, int r=2):

    cdef:
        Py_ssize_t i, j, ki
        int rows = chunk_block.shape[0]
        int cols = chunk_block.shape[1]
        int scales_half = end_scale / 2
        int scales_block = end_scale - blk
        DTYPE_uint16_t[::1] scale_array = np.array(scales, dtype='uint16')
        int scale_length = scale_array.shape[0]
        DTYPE_float32_t[::1] zs = np.zeros((end_scale*2)*(end_scale*2), dtype='float32')
        unsigned int out_len = _get_output_length(rows, cols, scales_block, blk, scale_length, 1)
        DTYPE_float32_t[::1] out_list = np.zeros(out_len, dtype='float32')

    _feature_lacunarity(chunk_block,
                        blk,
                        scale_array,
                        scales_half,
                        scales_block,
                        rows,
                        cols,
                        r,
                        out_len,
                        scale_length,
                        zs,
                        out_list)

    return np.float32(out_list)


# cdef azimuthal_avg(image, center=None):
#
#     """
#     Calculate the azimuthally averaged radial profile.
#
#     image - The 2D image
#     center - The [x,y] pixel coordinates used as the center. The default is
#              None, which then uses the center of the image (including
#              fracitonal pixels).
#     """
#
#     cdef:
#         np.ndarray[DTYPE_int64_t, ndim=2] y, x
#
#     # Calculate the indices from the image
#     y, x = np.indices(image.shape)
#
#     if not center:
#         center = np.array([(x.max() - x.min()) / 2., (x.max() - x.min()) / 2.0])
#
#     # get hypotenuse
#     r = np.hypot(np.subtract(x, center[0]), np.subtract(y, center[1]))
#
#     # get sorted radii indices
#     ind = np.argsort(r.flat)
#     rSorted = r.flat[ind]
#
#     # get image values from index positions
#     iSorted = image.flat[ind]
#
#     # Get the integer part of the radii (bin size = 1)
#     rInt = rSorted.astype(int)
#
#     # Find all pixels that fall within each radial bin.
#     deltar = np.subtract(rInt[1:], rInt[:-1])  # Assumes all radii represented
#     rind = np.where(deltar)[0]       # location of changed radius
#     nr = np.subtract(rind[1:], rind[:-1])        # number of radius bin
#
#     # Cumulative sum to figure out sums for each radius bin
#     csim = np.cumsum(iSorted, dtype=float)
#     tbin = np.subtract(csim[rind[1:]], csim[rind[:-1]])
#
#     radialProf = np.divide(tbin, nr)
#
#     return radialProf
#
#
# cdef DTYPE_float32_t[:, ::1] _fourier_transform(DTYPE_uint8_t[:, ::1] chunk_block):
#
#     cdef:
#         DTYPE_float32_t[:, :] dft = cv2.dft(np.float32(chunk_block), flags=cv2.DFT_COMPLEX_OUTPUT)
#         DTYPE_float32_t[:, :, :] dft_shift = np.fft.fftshift(dft)
#
#     # get the Power Spectrum
#     return np.float32(azimuthal_avg(20. * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))))
#
#
# cdef np.ndarray[DTYPE_float32_t, ndim=1] _feature_fourier(DTYPE_uint8_t[:, ::1] chunk_block,
#                                                           int blk,
#                                                           DTYPE_uint16_t[::1] scales,
#                                                           int scales_half,
#                                                           int scales_block,
#                                                           int rows,
#                                                           int cols,
#                                                           int out_len,
#                                                           int scale_length):
#
#     cdef:
#         Py_ssize_t i, j, ki, cr, cc
#         int k, k_half
#         Py_ssize_t pixel_counter = 0
#         DTYPE_uint8_t[:, ::1] ch_bd
#         DTYPE_float32_t[::1] out_list = np.zeros(out_len, dtype='float32')
#
#     for i from 0 <= i < rows-scales_block by blk:
#         for j from 0 <= j < cols-scales_block by blk:
#             for ki in range(0, scale_length):
#
#                 k = scales[ki]
#                 k_half = <int>(k / 2.)
#
#                 ch_bd = chunk_block[i+scales_half-k_half:i+scales_half-k_half+k,
#                                     j+scales_half-k_half:j+scales_half-k_half+k]
#
#                 bcr = ch_bd.shape[0]
#                 bcc = ch_bd.shape[1]
#
#                 ch_bd_transform = _fourier_transform(ch_bd)
#
#                 out_list[pixel_counter] = _get_mean(ch_bd_transform, bcr, bcc)
#                 pixel_counter += 1
#
#                 out_list[pixel_counter] = _get_var(ch_bd_transform, bcr, bcc)
#                 pixel_counter += 1
#
#     return np.float32(out_list)
#
#
# def feature_fourier(np.ndarray chunk_block, int blk, list scales, int end_scale):
#
#     cdef:
#         Py_ssize_t i, j, ki
#         int rows = chunk_block.shape[0]
#         int cols = chunk_block.shape[1]
#         int scales_half = <int>(end_scale / 2.)
#         int scales_block = end_scale - blk
#         DTYPE_uint16_t[::1] scale_array = np.array(scales, dtype='uint16')
#         int scale_length = scale_array.shape[0]
#         int out_len = _get_output_length(rows, cols, scales_block, blk, scale_length, 1)
#
#     return _feature_fourier(np.uint8(chunk_block),
#                             blk,
#                             scale_array,
#                             scales_half,
#                             scales_block,
#                             rows,
#                             cols,
#                             out_len,
#                             scale_length)


cdef void _fill_labels(DTYPE_uint64_t[:, ::1] im,
                       DTYPE_uint64_t[:, ::1] area_im,
                       DTYPE_uint64_t[::1] props,
                       DTYPE_uint64_t[::1] unique_labels,
                       unsigned int n_unique,
                       unsigned int rows,
                       unsigned int cols):

    cdef:
        Py_ssize_t ui, i, j
        int unq, uli_area

    with nogil:

        for uli in range(0, n_unique):

            unq = unique_labels[uli]

            if unq > 0:

                uli_area = props[uli]

                for i in range(0, rows):

                    for j in range(0, cols):

                        if im[i, j] == unq:
                            area_im[i, j] = uli_area


def fill_labels(DTYPE_uint64_t[:, ::1] im, DTYPE_uint64_t[::1] props):

    cdef:
        unsigned int rows = im.shape[0]
        unsigned int cols = im.shape[1]
        DTYPE_uint64_t[:, ::1] area_im = np.zeros((rows, cols), dtype='uint64')
        DTYPE_uint64_t[::1] unique_labels = np.uint64(np.unique(im))
        unsigned int n_unique = unique_labels.shape[0]

    _fill_labels(im, area_im, props, unique_labels, n_unique, rows, cols)

    return np.uint64(area_im)
