# cython: profile=False
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import cython
cimport cython

import numpy as np
cimport numpy as np

DTYPE_float32 = np.float32
ctypedef np.float32_t DTYPE_float32_t


def get_features(DTYPE_float32_t[:, :] lsfarr,
                 DTYPE_float32_t[:, :, :] lsfim1,
                 DTYPE_float32_t[:, :, :] lsfim2,
                 Py_ssize_t rows, Py_ssize_t cols):

    cdef:
        Py_ssize_t i, j
        DTYPE_float32_t lsfim1_1, lsfim2_1, lsfim1_2, lsfim2_2

    with nogil:

        for i in range(0, rows):

            for j in range(0, cols):

                lsfim1_1 = lsfim1[0, i, j]	    # max should equal lsfarr rows
                lsfim2_1 = lsfim2[0, i, j]

                if lsfim1_1 > 0:
                    lsfim1_1 -= 1

                if lsfim2_1 > 0:
                    lsfim2_1 -= 1

                if (lsfim1_1 == 0) and (lsfim2_1 == 0):
                    continue

                lsfim1_2 = lsfim1[1, i, j]
                lsfim2_2 = lsfim2[1, i, j]

                if lsfim1_2 > lsfim2_2:
                    lsfarr[<int>lsfim1_1, 5] += 1
                    lsfarr[<int>lsfim2_1, 5] -= 1
                else:
                    lsfarr[<int>lsfim1_1, 5] -= 1
                    lsfarr[<int>lsfim2_1, 5] += 1

    return np.float32(lsfarr)
