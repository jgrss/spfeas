#!/usr/bin/env python

import cython
cimport cython

import numpy as np
cimport numpy as np

DTYPE_uint8 = np.uint8
ctypedef np.uint8_t DTYPE_uint8_t
DTYPE_float32 = np.float32
ctypedef np.float32_t DTYPE_float32_t


@cython.boundscheck(False)
@cython.wraparound(False)
def chunk_int(np.ndarray[DTYPE_uint8_t, ndim=2] bd, int rows, int cols, int block_size, int chunk_size, int scale):

    cdef int i, j

    return [bd[i:i+chunk_size, j:j+chunk_size] for i in xrange(0, rows, chunk_size-(scale-block_size)) \
            for j in xrange(0, cols, chunk_size-(scale-block_size))]


@cython.boundscheck(False)
@cython.wraparound(False)
def chunk_float(np.ndarray[DTYPE_float32_t, ndim=2] bd, int rows, int cols, int block_size, int chunk_size, int scale):

    cdef int i, j

    return [bd[i:i+chunk_size, j:j+chunk_size] for i in xrange(0, rows, chunk_size-(scale-block_size)) \
            for j in xrange(0, cols, chunk_size-(scale-block_size))]