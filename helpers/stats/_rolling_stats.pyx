#!/usr/bin/env python

"""
@author: Jordan Graesser
"""

import cython
cimport cython

from cython.parallel import prange

import numpy as np
cimport numpy as cnp

from libc.math cimport pow

DTYPE_uint64 = np.uint64
ctypedef cnp.uint64_t DTYPE_uint64_t

DTYPE_float32 = np.float32
ctypedef cnp.float32_t DTYPE_float32_t

DTYPE_float64 = np.float64
ctypedef cnp.float64_t DTYPE_float64_t

cdef extern from 'numpy/npy_math.h':
    bint npy_isnan(DTYPE_float32_t x)


# cdef inline DTYPE_float32_t get_mean(np.ndarray[DTYPE_float32_t, ndim=1] row_arr, int window_size):
#
#     cdef int v
#     cdef DTYPE_float32_t the_sum = 0.
#
#     for v in xrange(0, window_size):
#
#         the_sum += row_arr[v]
#
#     return the_sum / float(window_size)


# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef int _get_max_val(np.ndarray[DTYPE_float32_t, ndim=1] arr, int n_cols):
#
#     cdef:
#         int cl, mx_idx
#         DTYPE_float32_t mx = 0.
#
#     for cl in xrange(0, n_cols):
#
#         if arr[cl] > mx:
#
#             mx = arr[cl]
#             mx_idx = cl
#
#     return mx_idx
#
#
# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef int _get_min_val(np.ndarray[DTYPE_float32_t, ndim=1] arr, int n_cols):
#
#     cdef:
#         int cl, mn_idx
#         DTYPE_float32_t mn = 10000.
#
#     for cl in xrange(0, n_cols):
#
#         if arr[cl] < mn:
#
#             mn = arr[cl]
#             mn_idx = cl
#
#     return mn_idx


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef DTYPE_float32_t _get_mean(DTYPE_float32_t[:] array_1d, int array_length):

    cdef:
        Py_ssize_t jj
        DTYPE_float32_t array_sum = array_1d[0]

    for jj in xrange(1, array_length):
        array_sum += array_1d[jj]

    return array_sum / array_length


@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_float32_t[:] _get_min(DTYPE_float32_t[:] val1, DTYPE_float32_t[:] val2, int cols):

    cdef:
        Py_ssize_t c

    for c in xrange(0, cols):

        if val2[c] < val1[c]:

            val1[c] = val2[c]

    return val1


@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_float32_t[:] _get_max(DTYPE_float32_t[:] val1, DTYPE_float32_t[:] val2, int cols):

    cdef:
        Py_ssize_t c

    for c in xrange(0, cols):

        if val2[c] > val1[c]:

            val1[c] = val2[c]

    return val1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef DTYPE_float32_t[:] _least_squares(DTYPE_float32_t[:] x, DTYPE_float32_t[:, :] y, int n, int cols):

    cdef:
        DTYPE_float32_t temp
        Py_ssize_t ii, jj
        DTYPE_float32_t x_avg = 0.
        DTYPE_float32_t[:] y_avg = np.zeros(cols, dtype='float32')
        DTYPE_float32_t var_x = 0.
        DTYPE_float32_t[:] cov_xy = np.zeros(cols, dtype='float32')

    for ii in xrange(0, n):

        x_avg += x[ii]

        for jj in prange(0, cols, nogil=True, num_threads=cols, schedule='static'):
            y_avg[jj] += y[ii, jj]

    x_avg /= n

    for jj in prange(0, cols, nogil=True, num_threads=cols, schedule='static'):
        y_avg[jj] /= n

    for ii in xrange(0, n):

        temp = x[ii] - x_avg
        var_x += pow(temp, 2.)

        for jj in prange(0, cols, nogil=True, num_threads=cols, schedule='static'):
            cov_xy[jj] += temp * (y[ii, jj] - y_avg[jj])

    for jj in prange(0, cols, nogil=True, num_threads=cols, schedule='static'):
        cov_xy[jj] /= var_x

    return cov_xy


# cdef class Vector(object):
#
#     cdef float *data
#     cdef public int n_ax0
#
#     def __init__(Vector self, int n_ax0):
#
#         self.data = <float*> malloc (sizeof(float) * n_ax0)
#         self.n_ax0 = n_ax0
#
#     def __dealloc__(Vector self):
#         free(self.data)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_float32_t[:] _replace_nans(DTYPE_float32_t[:] nan_array, int replace_value, int cols):

    cdef:
        Py_ssize_t j

    # for j in prange(0, cols, nogil=True, num_threads=cols, schedule='static'):
    for j in xrange(0, cols):

        if npy_isnan(nan_array[j]):
            nan_array[j] = 0

    return nan_array


@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple _rolling_least_squares(DTYPE_float32_t[:, :] image_array, int window_size):

    cdef:
        Py_ssize_t w
        unsigned int rows = image_array.shape[0]
        unsigned int cols = image_array.shape[1]
        # np.ndarray[DTYPE_float32_t, ndim=1, mode='c'] X = np.arange(window_size, 'float32')
        # np.ndarray[DTYPE_float32_t, ndim=1, mode='c'] slopes_max = np.zeros(cols, dtype='float32')
        # np.ndarray[DTYPE_float32_t, ndim=1, mode='c'] slopes_min = np.multiply(np.ones(cols, dtype='float32'), 1000)
        # np.ndarray[DTYPE_float32_t, ndim=1, mode='c'] slopes

        # DTYPE_float32_t[:] X = cython.view.array(shape=(10, 2), itemsize=sizeof(int), format="i")
        # cdef Vector X = Vector(cols)

        DTYPE_float32_t[:] X = np.arange(window_size, dtype='float32')
        DTYPE_float32_t[:] slopes_max = np.zeros(cols, dtype='float32')
        DTYPE_float32_t[:] slopes_min = np.ones(cols, dtype='float32') * 1000.
        DTYPE_float32_t[:] slopes

    for w in xrange(0, rows-window_size):

        slopes = _least_squares(X, image_array[w:w+window_size, :], window_size, cols)

        slopes_min = _get_min(slopes_min, slopes, cols)
        slopes_max = _get_max(slopes_max, slopes, cols)

    slopes_min = _replace_nans(slopes_min, 0, cols)
    slopes_max = _replace_nans(slopes_max, 0, cols)

    # slopes_min[np.isnan(slopes_min) | np.isinf(slopes_min)] = 0
    # slopes_max[np.isnan(slopes_max) | np.isinf(slopes_max)] = 0

    return np.asarray(slopes_min), np.asarray(slopes_max)


# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef DTYPE_float32_t get_median(np.ndarray[DTYPE_float32_t, ndim=1] the_row, DTYPE_float32_t mxv, DTYPE_float32_t mnv, int window_size):
#
#     cdef:
#         unsigned int v
#         DTYPE_float32_t med_val
#
#     for v in xrange(0, window_size):
#
#         if (v != mxv) and (v != mnv):
#
#             med_val = the_row[v]
#
#             break
#
#     return med_val


@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_float32_t[:, :] _rolling_median(DTYPE_float32_t[:, :] arr, int window_size):

    cdef:
        Py_ssize_t i, j
        unsigned int rows = arr.shape[0]
        unsigned int cols = arr.shape[1]
        unsigned int window_half1 = window_size / 2
        unsigned int window_half2 = (window_size / 2) + 1
        DTYPE_float32_t[:] the_row
        DTYPE_float32_t[:, :] results = np.zeros((rows, cols), dtype='float32')

    ## iterate over the array by rows
    for i in xrange(0, rows):

        the_row = arr[i]

        for j in xrange(0, cols-window_half1):

            the_row[j+window_size-window_half2] = sorted(the_row[j:j+window_size])[window_half1]

        results[i] = the_row

    return results


@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.ndarray[DTYPE_float32_t, ndim=3, mode='c'] _rolling_window(cnp.ndarray[DTYPE_float32_t, ndim=2, mode='c'] a,
                                                                    int window):

    cdef:
        tuple shape, strides

    shape = (<object> a).shape[:1] + ((<object> a).shape[1] - window + 1, window)
    strides = (<object> a).strides + ((<object> a).strides[1],)

    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.ndarray[DTYPE_float32_t, ndim=2, mode='c'] _rolling_median_v(cnp.ndarray[DTYPE_float32_t, ndim=2, mode='c'] arr, int window_size):

    cdef:
        int cols = arr.shape[1]
        int window_half1 = window_size / 2
        cnp.ndarray[DTYPE_float32_t, ndim=3] arr_strides, arr_strides_sorted
        cnp.ndarray[DTYPE_float32_t, ndim=2] results

    # get window strides
    arr_strides = _rolling_window(arr, window_size)

    # sort on the last axis
    arr_strides_sorted = np.sort(arr_strides)

    # get the center value
    results = np.sort(arr_strides_sorted)[:, :, window_half1]

    # put back into original array
    arr[:, 1:cols-window_half1] = results

    return arr


@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_float32_t[:, :] _rolling_mean(DTYPE_float32_t[:, :] arr, int window_size):

    cdef:
        Py_ssize_t i, j
        unsigned int rows = arr.shape[0]
        unsigned int cols = arr.shape[1]
        unsigned int window_half = window_size / 2
        DTYPE_float32_t[:] the_row
        DTYPE_float32_t[:, :] results = np.zeros((rows, cols), dtype='float32')

    # Iterate over the array by rows.
    for i in xrange(0, rows):

        # Get the current row
        the_row = arr[i]

        # Get the rolling mean.
        for j in xrange(0, cols-window_half):

            the_row[j+window_half] = _get_mean(the_row[j:j+window_size], window_size)

        results[i] = the_row

    return results


@cython.boundscheck(False)
@cython.wraparound(False)
def rolling_stats(cnp.ndarray[DTYPE_float32_t, ndim=2] image_array, str stat='median', int window_size=3):

    """
    Computes rolling statistics
    
    Args:
        image_array (ndarray): The expected shape is (image dimensions x samples), or (image dimensions x 
            (rows x columns)). To reshape the array to expected format, follow:
            If image shape = (dimensions, rows, columns):
                image.reshape(dimensions, rows*columns)
                image.T.reshape(rows*columns, dimensions) -->
                    image.reshape(columns, rows, dimensions).T.reshape(dimensions, rows*columns)
        stat (Optional[str]): The statistic to compute. Default is 'median'. Choices are ['mean', 'median', 'slope'].
        window_size (Optional[int]): The window size. Default is 3.
    """

    if stat == 'mean':

        return _rolling_mean(np.ascontiguousarray(image_array), window_size)

    elif stat == 'median':

        return _rolling_median(np.ascontiguousarray(image_array), window_size)

    elif stat == 'slope':

        return _rolling_least_squares(np.ascontiguousarray(image_array), window_size)
