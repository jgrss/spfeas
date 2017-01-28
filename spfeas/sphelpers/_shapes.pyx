# cython: profile=False
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import cython
cimport cython

import numpy as np
cimport numpy as np

DTYPE_uint8 = np.uint8
ctypedef np.uint8_t DTYPE_uint8_t

DTYPE_uint32 = np.uint32
ctypedef np.uint32_t DTYPE_uint32_t

DTYPE_int = np.int
ctypedef np.int_t DTYPE_int_t

DTYPE_float32 = np.float32
ctypedef np.float32_t DTYPE_float32_t


#cdef inline DTYPE_float32_t _get_min(DTYPE_float32_t s1, DTYPE_float32_t s2, Py_ssize_t iy, Py_ssize_t jx):
#    return s2 if s2 < s1 else s1


cdef inline DTYPE_uint8_t abss(DTYPE_int_t sx) nogil:
    return sx * -1 if sx < 0 else sx


cdef np.ndarray[DTYPE_uint32_t, ndim=2] _feature_shape(DTYPE_uint8_t[:, :] image,
                                                       int stop_thresh, int merge_thresh,
                                                       int rows, int cols):

    cdef:
        Py_ssize_t i, j, i_, j_, ii, jj, ic, jc, cum_value
        Py_ssize_t label = 1
        DTYPE_uint8_t center_value, min_dev
        DTYPE_int_t[:] idx = np.array([-1, 0, 1], dtype='int')
        DTYPE_uint32_t[:, :] out_image = np.zeros((rows, cols), dtype='uint32')

    ic = -1
    jc = -1

    with nogil:

        # Iterate over the entire image
        for i in range(0, rows-2):

            i_ = i

            for j in range(0, cols-2):

                j_ = j

                # 1) Check if the pixel is already marked.
                if out_image[i, j] == 0:

                    # Label the pixel
                    out_image[i, j] = label

                    # Used to determine the stopping point.
                    cum_value = 0

                    # Continue until the threshold has been reached.
                    while cum_value < stop_thresh:

                        # Get the current value.
                        if (i_ < 0) or (i_ >= rows) or (j_ < 0) or (j_ >= cols):
                            cum_value = stop_thresh + 1
                        else:

                            center_value = image[i_, j_]

                            min_dev = 255

                            # Find the surrounding pixel with the
                            #   smallest deviation from the center.
                            for ii in range(0, 3):

                                # Bounds checking
                                if -1 < i_ + idx[ii] < rows:

                                    for jj in range(0, 3):

                                        # Bounds checking
                                        if -1 < j_ + idx[jj] < cols:

                                            # Center value
                                            if (ii != 1) and (jj != 1):

                                                # Get the absolute deviation from the center value.
                                                dev = abss(center_value - image[i_+idx[ii], j_+idx[jj]])

                                                # Get the smallest deviation
                                                if dev < min_dev:

                                                    min_dev = dev

                                                    # Update the new route.
                                                    ic = i_ + idx[ii]   # Current row position + route offset
                                                    jc = j_ + idx[jj]

                            if (ic < 0) or (ic >= rows) or (jc < 0) or (jc >= cols):
                                cum_value = stop_thresh + 1
                            else:

                                # Update the route total deviation.
                                if min_dev == 0:
                                    cum_value += 1
                                else:
                                    cum_value += min_dev

                                # Mark the current label.

                                # 1) Check if the pixel is already marked.
                                if out_image[ic, jc] == 0:
                                    out_image[ic, jc] = label
                                else:

                                    # 2) Check if the label can be merged.
                                    if min_dev < merge_thresh:
                                        out_image[ic, jc] = out_image[i, j]

                                # Continue along the route of
                                #   smallest deviation.
                                i_ = ic
                                j_ = jc

                    # Update the label.
                    label += 1

    return np.uint32(out_image)


def feature_shape(np.ndarray[DTYPE_uint8_t, ndim=2] image, int stop_thresh, int merge_thresh):

    cdef:
        int rows = image.shape[0]
        int cols = image.shape[1]

    return _feature_shape(image, stop_thresh, merge_thresh, rows, cols)
