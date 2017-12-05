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


cdef inline DTYPE_float32_t _color_distance(DTYPE_float32_t[:] r, DTYPE_float32_t[:] g, DTYPE_float32_t[:] b, DTYPE_float32_t sq_diff) nogil:
    return ((r[1] - r[0])**2 + (g[1] - g[0])**2 + (b[1] - b[0])**2)**.5 / sq_diff


cdef np.ndarray[DTYPE_uint32_t, ndim=2] _feature_shape(DTYPE_float32_t[:, :, :] image,
                                                       DTYPE_float32_t stop_thresh,
                                                       DTYPE_float32_t merge_thresh,
                                                       DTYPE_float32_t min_break_thresh1,
                                                       DTYPE_float32_t min_break_thresh2,
                                                       DTYPE_float32_t min_break_thresh3,
                                                       int rows, int cols):

    cdef:
        Py_ssize_t i, j, i_, j_, ii, jj, ic, jc
        Py_ssize_t label = 2
        DTYPE_float32_t min_dev, dev, cum_value
        DTYPE_float32_t p_value_r, p_value_g, p_value_b, p_value_r_o, p_value_g_o, p_value_b_o
        DTYPE_int_t[:] idx = np.array([-1, 0, 1], dtype='int')
        DTYPE_float32_t[:, :] out_image = np.ones((rows, cols), dtype='float32')
        DTYPE_float32_t sq_diff = (255.**2. + 255.**2. + 255.**2.) ** .5
        DTYPE_float32_t[:] r_tuple = np.array([0, 0], dtype='float32')
        DTYPE_float32_t[:] g_tuple = r_tuple.copy()
        DTYPE_float32_t[:] b_tuple = r_tuple.copy()

    ic = -1
    jc = -1

    with nogil:

        # Iterate over the entire image
        for i in range(0, rows-2):

            for j in range(0, cols-2):

                i_ = i
                j_ = j

                # 1) Check if the pixel is already marked.
                if out_image[i, j] == 1:

                    p_value_r = image[0, i_, j_]
                    p_value_g = image[1, i_, j_]
                    p_value_b = image[2, i_, j_]

                    if (p_value_r == 0) and (p_value_g == 0) and (p_value_b == 0):
                        out_image[i, j] = 0
                        continue

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

                            p_value_r = image[0, i_, j_]
                            p_value_g = image[1, i_, j_]
                            p_value_b = image[2, i_, j_]

                            min_dev = 999999.

                            # Find the surrounding pixel with the
                            #   smallest deviation from the center.
                            for ii in range(0, 3):

                                # Bounds checking
                                if -1 < i_ + idx[ii] < rows:

                                    for jj in range(0, 3):

                                        # Bounds checking
                                        if -1 < j_ + idx[jj] < cols:

                                            # Center value
                                            if (ii == 1) and (jj == 1):
                                                continue
                                            else:

                                                # Get the absolute deviation from the center value.
                                                p_value_r_o = image[0, i_+idx[ii], j_+idx[jj]]
                                                p_value_g_o = image[1, i_+idx[ii], j_+idx[jj]]
                                                p_value_b_o = image[2, i_+idx[ii], j_+idx[jj]]

                                                r_tuple[0] = p_value_r
                                                r_tuple[1] = p_value_r_o

                                                g_tuple[0] = p_value_g
                                                g_tuple[1] = p_value_g_o

                                                b_tuple[0] = p_value_b
                                                b_tuple[1] = p_value_b_o

                                                dev = _color_distance(r_tuple,
                                                                      g_tuple,
                                                                      b_tuple,
                                                                      sq_diff)

                                                # Get the smallest deviation
                                                if dev < min_dev:

                                                    # Update the new route.
                                                    ic = i_ + idx[ii]   # Current row position + route offset
                                                    jc = j_ + idx[jj]

                                                    if out_image[ic, jc] == 1:

                                                        min_dev = dev

                                                        if min_dev == 0:
                                                            break

                            if (ic < 0) or (ic >= rows) or (jc < 0) or (jc >= cols) or (min_dev == 999999.):
                                cum_value = stop_thresh + 1
                            else:

                                # Update the route total deviation.
                                if min_break_thresh1 <= min_dev < min_break_thresh2:
                                    cum_value += (min_dev**.5)
                                elif min_break_thresh2 <= min_dev < min_break_thresh3:
                                    cum_value = (min_dev**.5) * 2
                                elif min_dev >= min_break_thresh3:
                                    cum_value = stop_thresh + 1.

                                # Mark the current label.

                                # 1) Check if the pixel is already marked.
                                if out_image[ic, jc] == 1:
                                    out_image[ic, jc] = label
                                else:

                                    # 2) Check if the label can be merged.
                                    if (min_dev < merge_thresh) and (out_image[i_, j_] > 1):
                                        out_image[ic, jc] = out_image[i_, j_]

                                    # Ensure movement
                                    cum_value += (min_dev + .01)

                                # Continue along the route of
                                #   smallest deviation.
                                i_ = ic
                                j_ = jc

                    # Update the label.
                    label += 1

    return np.uint32(out_image)


def feature_shape(np.ndarray image,
                  DTYPE_float32_t stop_thresh,
                  DTYPE_float32_t merge_thresh,
                  DTYPE_float32_t min_break_thresh1,
                  DTYPE_float32_t min_break_thresh2,
                  DTYPE_float32_t min_break_thresh3):

    cdef:
        int rows = image.shape[1]
        int cols = image.shape[2]

    return _feature_shape(np.float32(image),
                          stop_thresh,
                          merge_thresh,
                          min_break_thresh1,
                          min_break_thresh2,
                          min_break_thresh3,
                          rows, cols)
