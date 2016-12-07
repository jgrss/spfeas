#!/usr/bin/env python

import sys
import math
import numpy as np


def _glcm_loop(image, distances, angles, levels,
               out, out_sums, rows, cols):

    angles_ = angles.shape[0]
    distances_ = distances.shape[0]

    for a_idx in xrange(0, angles_):

        angle = angles[a_idx]

        for d_idx in xrange(0, distances_):

            distance = distances[d_idx]

            # Iterate over the image
            for r in xrange(0, rows):

                for c in xrange(0, cols):

                    # Current pixel value
                    i = image[r, c]

                    # compute the location of the offset pixel
                    # row = r + <int>round(sin(angle) * distance)
                    # col = c + <int>round(cos(angle) * distance)

                    row = r + int(round(math.sin(angle) * distance))
                    col = c + int(round(math.cos(angle) * distance))

                    # make sure the offset is within bounds
                    if (0 <= row < rows) and (0 <= col < cols):
                    # if row >= 0 and row < rows and col >= 0 and col < cols:
                        j = image[row, col]

                        if (0 <= i < levels) and (0 <= j < levels):
                        # if i >= 0 and i < levels and j >= 0 and j < levels:

                            out[i, j, d_idx, a_idx] += 1

                            # symmetric
                            out[levels-1-i, levels-1-j, d_idx, a_idx] += 1

                            out_sums[d_idx, a_idx] += 2


def _norm_glcm(Pt, Pt_sums, distances, angles, levels, glcm_normed_):

    angles_ = angles.shape[0]
    distances_ = distances.shape[0]

    # Get the sums
    for a_idx in xrange(0, angles_):

        for d_idx in xrange(0, distances_):

            # Iterate over the image
            for r in xrange(0, levels):

                for c in xrange(0, levels):
                    glcm_normed_[r, c, d_idx, a_idx] += (Pt[r, c, d_idx, a_idx] / Pt_sums[d_idx, a_idx])


def _greycomatrix(image, distances, angles, levels, rows, cols):

    P = np.zeros((levels, levels, distances.shape[0], angles.shape[0]), dtype='float32')
    angle_dist_sums = np.zeros((distances.shape[0], angles.shape[0]), dtype='float32')
    glcm_normed = np.zeros((levels, levels, distances.shape[0], angles.shape[0]), dtype='float32')

    # count co-occurences
    _glcm_loop(image, distances, angles, levels, P, angle_dist_sums, rows, cols)

    # Normalize the matrix
    _norm_glcm(P, angle_dist_sums, distances, angles, levels, glcm_normed)

    # glcm_normed = _check_nans(glcm_normed, distances, angles, rows, cols)

    # return np.array(glcm_normed)
    return glcm_normed


def _glcm_contrast(P, distances, angles, levels, contrast_array):

    angles_ = angles.shape[0]
    distances_ = distances.shape[0]

    min_contrast = 1000000.

    for a_idx in xrange(0, angles_):

        for d_idx in xrange(0, distances_):

            # Sum the contrast for the current angle/distance pair.
            contrast_sum = 0.

            # Iterate over the image
            for r in xrange(0, levels):

                for c in xrange(0, levels):
                    # print a_idx, d_idx, r, c, np.array(contrast_array).shape, np.array(P).shape
                    contrast_sum += contrast_array[r, c] * P[r, c, d_idx, a_idx]

            # Get the minimum contrast over all angle/distance pairs
            min_contrast = min(min_contrast, contrast_sum)

    return min_contrast


def _set_contrast_weights(levels):

    contrast_array = np.zeros((levels, levels), dtype='float32')

    for li in xrange(0, levels):
        for lj in xrange(0, levels):
            contrast_array[li, lj] = math.pow(li-lj, 2)

    return contrast_array


def main():

    ch_bd = np.random.randn(100*100).reshape(100,100).astype(np.uint8)
    block_rows, block_cols = ch_bd.shape

    pi = 3.14159265
    levels = 32

    disp_vect = np.array([0., pi / 6., pi / 4., pi / 3., pi / 2., (2. * pi) / 3., (3. * pi) / 4., (5. * pi) / 6.], dtype='float32')
    dists = np.array([1, 2], dtype='float32')
    contrast_weights = _set_contrast_weights(levels)

    # print disp_vect.dtype
    # print disp_vect.shape
    # print dists.dtype
    # print dists.shape
    # print contrast_weights.dtype
    # print contrast_weights.shape

    glcm_mat = _greycomatrix(ch_bd, dists, disp_vect, levels, block_rows, block_cols)

    con_min = _glcm_contrast(np.array(glcm_mat), dists, disp_vect, levels, contrast_weights)

    print con_min


if __name__ == '__main__':
    main()
