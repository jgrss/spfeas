# cython: profile=False
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import cython
cimport cython

import numpy as np
cimport numpy as np
#from cython.parallel import prange

from libc.math cimport atan2, sqrt, floor

DTYPE_uint8 = np.uint8
ctypedef np.uint8_t DTYPE_uint8_t

DTYPE_uint16 = np.uint16
ctypedef np.uint16_t DTYPE_uint16_t

DTYPE_float32 = np.float32
ctypedef np.float32_t DTYPE_float32_t

cdef extern from 'numpy/npy_math.h':
    bint npy_isnan(DTYPE_float32_t x)

cdef extern from 'numpy/npy_math.h':
    bint npy_isinf(DTYPE_float32_t x)


"""
Based on _hog.py from https://github.com/scikit-image/scikit-image

Copyright (C) 2011, the scikit-image team
(C) 2013 Tim Sheerman-Chase
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

 1. Redistributions of source code must retain the above copyright
	notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
	notice, this list of conditions and the following disclaimer in
	the documentation and/or other materials provided with the
	distribution.
 3. Neither the name of skimage nor the names of its contributors may be
	used to endorse or promote products derived from this software without
	specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""


cdef inline DTYPE_uint8_t _get_max_sample_int(DTYPE_uint8_t s1, DTYPE_uint8_t s2):
    return s2 if s2 > s1 else s1


cdef inline DTYPE_float32_t _get_max_sample(DTYPE_float32_t s1, DTYPE_float32_t s2) nogil:
    return s2 if s2 > s1 else s1


cdef inline DTYPE_float32_t pow2(DTYPE_float32_t sx) nogil:
    return sx * sx


cdef inline DTYPE_float32_t pow3(DTYPE_float32_t sx) nogil:
    return sx * sx * sx


cdef inline DTYPE_float32_t pow4(DTYPE_float32_t sx) nogil:
    return sx * sx * sx * sx


cdef DTYPE_float32_t _get_block_sum3d(DTYPE_float32_t[:, :, :] block_, int ds, int rs, int cs):

    cdef:
        Py_ssize_t a, b, c
        DTYPE_float32_t block_sum = 0.

    for a in range(0, ds):
        for b in range(0, rs):
            for c in range(0, cs):
                block_sum += block_[a, b, c]

    return block_sum


cdef DTYPE_float32_t cell_hog(DTYPE_float32_t[:, :] magnitude,
                              DTYPE_float32_t[:, :] orientation,
                              DTYPE_float32_t ori1, DTYPE_float32_t ori2,
                              int cx, int cy,
                              int xi, int yi,
                              int sx, int sy) nogil:

    cdef:
        Py_ssize_t cx1, cy1
        DTYPE_float32_t total = 0.

    for cy1 in range(-cy / 2, cy / 2):

        for cx1 in range(-cx / 2, cx / 2):

            if yi + cy1 < 0:
                continue

            if yi + cy1 >= sy:
                continue

            if xi + cx1 < 0:
                continue
            if xi + cx1 >= sx:
                continue

            if orientation[yi + cy1, xi + cx1] >= ori1:
                continue

            if orientation[yi + cy1, xi + cx1] < ori2:
                continue

            total += magnitude[yi+cy1, xi+cx1]

    return total


cdef hog_third_stage(int cx, int cy, int bx, int by_,
                     int sx, int sy,
                     int n_cellsx, int n_cellsy,
                     int orientations,
                     DTYPE_float32_t[:, :, :] orientation_histogram,
                     DTYPE_float32_t[:, :] empty_block,
                     DTYPE_float32_t[:, :] magnitude_,
                     DTYPE_float32_t[:, :] orientation_):

    """
    The third stage aims to produce an encoding that is sensitive to
    local image content while remaining resistant to small changes in
    pose or appearance. The adopted method pools gradient orientation
    information locally in the same way as the SIFT [Lowe 2004]
    feature. The image window is divided into small spatial regions,
    called "cells". For each cell we accumulate a local 1-D histogram
    of gradient or edge orientations over all the pixels in the
    cell. This combined cell-level 1-D histogram forms the basic
    "orientation histogram" representation. Each orientation histogram
    divides the gradient angle range into a fixed number of
    predetermined bins. The gradient magnitudes of the pixels in the
    cell are used to vote into the orientation histogram.
    """

    cdef:
        Py_ssize_t i
        int x, y, o, yi, xi, cy1, cy2, cx1, cx2
        DTYPE_float32_t ori1, ori2
        DTYPE_float32_t number_of_orientations_per_180 = 180. / orientations

    # compute orientations integral images

    with nogil:

        for i in range(0, orientations):

            # isolate orientations in this range
            #ori1 = 180. / orientations * (i + 1)
            #ori2 = 180. / orientations * i
            ori1 = number_of_orientations_per_180 * (i + 1)
            ori2 = number_of_orientations_per_180 * i

            y = cy / 2
            cy2 = cy * n_cellsy
            x = cx / 2
            cx2 = cx * n_cellsx
            yi = 0
            xi = 0

            while y < cy2:
                xi = 0
                x = cx / 2

                while x < cx2:

                    orientation_histogram[yi, xi, i] = cell_hog(magnitude_, orientation_, ori1, ori2,
                                                                cx, cy, x, y, sx, sy)

                    xi += 1
                    x += cx

                yi += 1
                y += cy


cdef DTYPE_float32_t _get_sum1d(DTYPE_float32_t[:] block, int cs):

    cdef:
        Py_ssize_t bj
        DTYPE_float32_t block_sum = block[0]

    with nogil:

        for bj in xrange(1, cs):
            block_sum += block[bj]

    return block_sum


cdef DTYPE_float32_t _get_mean1d(DTYPE_float32_t[:] block, int cs):
    return _get_sum1d(block, cs) / cs


cdef DTYPE_float32_t _get_max_f(DTYPE_float32_t[:] in_row, int cols):

    cdef:
        Py_ssize_t a
        DTYPE_float32_t m = in_row[0]

    for a in xrange(1, cols):
        m = _get_max_sample(m, in_row[a])

    return m


cdef DTYPE_float32_t[:] _get_stats(DTYPE_float32_t[:] block, int samps):

    """
    Calculate the moments 1-4, skewness, and kurtosis
    """

    cdef:
        DTYPE_float32_t the_mean = _get_mean1d(block, samps)
        DTYPE_float32_t the_max = _get_max_f(block, samps)
        Py_ssize_t idx
        DTYPE_float32_t bx = block[0]
        DTYPE_float32_t val_dev = bx - the_mean
        DTYPE_float32_t m1 = bx - the_mean      # 1st moment
        DTYPE_float32_t m2 = pow2(val_dev)      # 2nd moment
        DTYPE_float32_t m3 = pow3(val_dev)      # 3rd moment
        DTYPE_float32_t m4 = pow4(val_dev)      # 4th moment
        DTYPE_float32_t[:] output = np.empty(7, dtype='float32')

    with nogil:

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

    output[0] = the_max
    output[1] = m1
    output[2] = m2
    output[3] = m3
    output[4] = m4
    output[5] = m3 / pow3(sqrt(m2))    # skewness: ratio of 3rd moment and standard dev. cubed
    output[6] = m4 / pow2(m2)          # kurtosis

    return output


cdef DTYPE_float32_t[:] get_moments(DTYPE_float32_t[:] img_arr):

    """
    Get the moments for 1d array
    """

    cdef:
        int img_arr_size = img_arr.shape[0]

    if _get_max_f(img_arr, img_arr_size) == 0:
        return np.zeros(img_arr_size, dtype='float32')
    else:
        return _get_stats(img_arr, img_arr_size)


cdef np.ndarray[DTYPE_float32_t, ndim=1] calc_hog(DTYPE_float32_t[:, :] magnitude,
                                                  DTYPE_float32_t[:, :] orientation,
                                                  int sy, int sx,
                                                  int orientations):

    """Extract Histogram of Oriented Gradients (HOG) for a given image.

    Compute a Histogram of Oriented Gradients (HOG) by

        1. (optional) global image normalisation
        2. computing the gradient image in x and y
        3. computing gradient histograms
        4. normalising across blocks
        5. flattening into a feature vector

    Parameters
    ----------
    image : (M, N) ndarray
        Input image (greyscale).
    orientations : int
        Number of orientation bins.
    pixels_per_cell : 2 tuple (int, int)
        Size (in pixels) of a cell.
    cells_per_block  : 2 tuple (int,int)
        Number of cells in each block.
    visualise : bool, optional
        Also return an image of the HOG.
    normalize : bool, optional
        Apply power law compression to normalize the image before
        processing.

    Returns
    -------
    newarr : ndarray
        HOG for the image as a 1D (flattened) array.
    hog_image : ndarray (if visualise=True)
        A visualisation of the HOG image.

    References
    ----------
    * http://en.wikipedia.org/wiki/Histogram_of_oriented_gradients

    * Dalal, N and Triggs, B, Histograms of Oriented Gradients for
      Human Detection, IEEE Computer Society Conference on Computer
      Vision and Pattern Recognition 2005 San Diego, CA, USA

    The first stage applies an optional global image normalisation
    equalisation that is designed to reduce the influence of illumination
    effects. In practice we use gamma (power law) compression, either
    computing the square root or the log of each colour channel.
    Image texture strength is typically proportional to the local surface
    illumination so this compression helps to reduce the effects of local
    shadowing and illumination variations.

    The second stage computes first order image gradients. These capture
    contour, silhouette and some texture information, while providing
    further resistance to illumination variations. The locally dominant
    colour channel is used, which provides colour invariance to a large
    extent. Variant methods may also include second order image derivatives,
    which act as primitive bar detectors - a useful feature for capturing,
    e.g. bar like structures in bicycles and limbs in humans.

    The fourth stage computes normalisation, which takes local groups of
    cells and contrast normalizes their overall responses before passing
    to next stage. Normalisation introduces better invariance to illumination,
    shadowing, and edge contrast. It is performed by accumulating a measure
    of local histogram "energy" over local groups of cells that we call
    "blocks". The result is used to normalize each cell in the block.
    Typically each individual cell is shared between several blocks, but
    its normalisations are block dependent and thus different. The cell
    thus appears several times in the final output vector with different
    normalisations. This may seem redundant but it improves the performance.
    We refer to the normalized block descriptors as Histogram of Oriented
    Gradient (HOG) descriptors.

    The final step collects the HOG descriptors from all blocks of a dense
    overlapping grid of blocks covering the detection window into a combined
    feature vector for use in the window classifier.
    """

    cdef:
        Py_ssize_t x, y
        #np.ndarray[DTYPE_float32_t, ndim=2] gx = np.zeros((sy, sx), dtype='float32')
        #np.ndarray[DTYPE_float32_t, ndim=2] gy = np.zeros((sy, sx), dtype='float32')
        DTYPE_float32_t[:, :] empty_block = np.zeros((sy, sx), dtype='float32')
        int cx = sx
        int cy = sy
        int bx = 1
        int by_ = 1
        int n_cellsx = int(floor(sx / cx))  # number of cells in x
        int n_cellsy = int(floor(sy / cy))  # number of cells in y
        DTYPE_float32_t[:, :, :] orientation_histogram = np.zeros((n_cellsy, n_cellsx, orientations), dtype='float32')
        np.ndarray[np.float32_t, ndim=3] block
        DTYPE_float32_t[:, :] hog_image
        np.ndarray[DTYPE_float32_t, ndim=5] normalized_blocks
        int n_blocksx
        int n_blocksy
        DTYPE_float32_t eps = 1e-5
        int br_shape, bc_shape

    hog_third_stage(cx, cy, bx, by_, sx, sy, n_cellsx, n_cellsy,
                    orientations, orientation_histogram, empty_block,
                    magnitude, orientation)

    n_blocksx = (n_cellsx - bx) + 1
    n_blocksy = (n_cellsy - by_) + 1

    normalized_blocks = np.zeros((n_blocksy, n_blocksx, by_, bx, orientations), dtype='float32')

    for y in range(0, n_blocksy):
        for x in range(0, n_blocksx):

            block = np.float32(orientation_histogram[y:y+by_, x:x+bx, :])

            normalized_blocks[y, x, :] = block / np.sqrt(block.sum()**2 + eps)

            #br_shape = y+by_ - y
            #bc_shape = x+bx - x

            #normalized_blocks[y, x, :] = _3d_block_division(normalized_blocks, block, eps,
            #                                                y, x, br_shape, bc_shape, by_, bx, orientations)

    return np.float32(normalized_blocks).ravel()


cdef DTYPE_float32_t _get_max(DTYPE_float32_t[:, :] block, int rs, int cs):

    cdef:
        Py_ssize_t bi, bj
        DTYPE_float32_t m = -999999.

    with nogil:

        for bi in xrange(0, rs):
            for bj in xrange(0, cs):
                m = _get_max_sample(m, block[bi, bj])

    return m


cdef np.ndarray[DTYPE_float32_t, ndim=1] _feature_hog(DTYPE_float32_t[:, :] grad,
                                                      DTYPE_float32_t[:, :] ori,
                                                      int blk, DTYPE_uint16_t[:] scs,
                                                      int end_scale, int scales_half,
                                                      int scales_block, int out_len,
                                                      int rows, int cols,
                                                      int scale_length,
                                                      int orientations):

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
        DTYPE_float32_t[:, :] ch_grad, ch_ori
        DTYPE_float32_t[:] sts
        DTYPE_float32_t[:] out_list = np.zeros(out_len, dtype='float32')
        int pix_ctr = 0

    for i from 0 <= i < rows-scales_block by blk:
        for j from 0 <= j < cols-scales_block by blk:
            for ki in xrange(0, scale_length):

                k = scs[ki]

                k_half = k / 2

                ch_grad = grad[i+scales_half-k_half:i+scales_half-k_half+k,
                               j+scales_half-k_half:j+scales_half-k_half+k]

                ch_ori = ori[i+scales_half-k_half:i+scales_half-k_half+k,
                             j+scales_half-k_half:j+scales_half-k_half+k]

                block_rows = ch_grad.shape[0]
                block_cols = ch_grad.shape[1]

                if _get_max(ch_grad, block_rows, block_cols) > 0:

                    sts = get_moments(calc_hog(ch_grad,
                                               ch_ori,
                                               block_rows,
                                               block_cols,
                                               orientations))

                    for sti in range(0, 7):

                        out_list[pix_ctr] = sts[sti]

                        pix_ctr += 1

                else:
                    pix_ctr += 7

    return np.float32(out_list)


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
        int orientations = 9

    for i from 0 <= i < rows-scales_block by blk:
        for j from 0 <= j < cols-scales_block by blk:
            for ki in xrange(0, scale_length):
                out_len += 7

    return _feature_hog(grad, ori, blk, scales_array, end_scale,
                        scales_half, scales_block, out_len,
                        rows, cols, scale_length,
                        orientations)
