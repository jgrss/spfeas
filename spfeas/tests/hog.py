#!/usr/bin/env python

import unittest

from mpglue import raster_tools
from spfeas.sphelpers import _hog
from spfeas.spsplit import get_mag_ang

import numpy as np
from skimage.exposure import rescale_intensity

import cv2

# class TestHoG(unittest.TestCase):


def n_rows_cols(pixel_index, rows_cols, block_size):
    return rows_cols if pixel_index + rows_cols < block_size else block_size - pixel_index


def _check_points(key_x, key_y, ki, kj, i_, j_, rr_rows, cc_cols,
                  hist_, grid_counter):

    """
    pts = (x,y)
    """

    # Point within the current grid.
    if (i_+ki <= key_y < i_+ki+rr_rows) and (j_+kj <= key_x < j_+kj+cc_cols):
        hist_[grid_counter] += 1


def _pyramid_hist_sift(orb_array, key_point_array, levels, i, j):

    n_key_points = key_point_array.shape[0]
    orb_rows = orb_array.shape[0]
    orb_cols = orb_array.shape[1]
    counter = 0

    # Iterate over each level
    for lv in range(0, 3):

        if (orb_rows < levels[lv]) or (orb_cols < levels[lv]):
            continue

        y_tiles = orb_rows / levels[lv]
        x_tiles = orb_cols / levels[lv]

        for ki in xrange(0, orb_rows, y_tiles):
            for kj in xrange(0, orb_cols, x_tiles):
                counter += 1

    hist = np.zeros(counter, dtype='float32')

    grid_counter = 0

    # Iterate over each level
    for lv in range(0, 3):

        if (orb_rows < levels[lv]) or (orb_cols < levels[lv]):
            continue

        y_tiles = orb_rows / levels[lv]
        x_tiles = orb_cols / levels[lv]

        for ki in xrange(0, orb_rows, y_tiles):

            rr_rows = n_rows_cols(ki, y_tiles, orb_rows)

            for kj in xrange(0, orb_cols, x_tiles):

                cc_cols = n_rows_cols(kj, x_tiles, orb_cols)

                # Iterate over each key point.
                for key_point_index in range(0, n_key_points):

                    _check_points(key_point_array[key_point_index, 0],
                                  key_point_array[key_point_index, 1],
                                  ki, kj, i, j,
                                  rr_rows, cc_cols,
                                  hist, grid_counter)

                grid_counter += 1

    return hist


def _fill_key_points(key_point_list):

    n_key_points = len(key_point_list)
    key_point_array = np.empty((n_key_points, 2), dtype='float32')

    for key_point_index in range(0, n_key_points):
        key_x, key_y = key_point_list[key_point_index].pt

        key_point_array[key_point_index, 0] = key_x
        key_point_array[key_point_index, 1] = key_y

    return key_point_array


def _orb(orb_array, k_pts, levels, i, j):

    """
    Get the moments
    """

    return _pyramid_hist_sift(orb_array, k_pts, levels, i, j)


def _feature_orb(ch_bd, blk, scales_array, scales_half, scales_block,
                 scale_length, out_len, rows, cols, scales_length, max_features):

    levels = np.array([2, 4, 8], dtype='uint8')
    pix_ctr = 0

    # Set the output list
    out_list = np.zeros(out_len, dtype='float32')

    # Initiate ORB detector
    orb = cv2.ORB_create(nfeatures=max_features, edgeThreshold=31, patchSize=31, WTA_K=4)

    # Compute ORB keypoints
    key_points, __ = orb.detectAndCompute(np.uint8(ch_bd), None)

    # img = cv2.drawKeypoints(np.uint8(ch_bd), key_points, np.uint8(ch_bd).copy())

    key_point_array = _fill_key_points(key_points)

    for i in xrange(0, rows-scales_block, blk):
        for j in xrange(0, cols-scales_block, blk):
            for ki in range(0, scale_length):

                k = scales_array[ki]

                k_half = k / 2

                ch_bd_block = ch_bd[i+scales_half-k_half:i+scales_half-k_half+k,
                                    j+scales_half-k_half:j+scales_half-k_half+k]

                if key_points:

                    sts = _orb(ch_bd_block, key_point_array, levels, i, j)

                    for st in xrange(0, 7):

                        out_list[pix_ctr] = sts[st]

                        pix_ctr += 1

                else:
                    pix_ctr += 7

    return np.float32(out_list)


def feature_orb(ch_bd, blk, scs, end_scale, max_features=20000):

    scales_half = end_scale / 2
    scales_block = end_scale - blk
    out_len = 0
    rows = ch_bd.shape[0]
    cols = ch_bd.shape[1]
    scales_array = np.array(scs, dtype='uint16')
    scale_length = scales_array.shape[0]

    for i in xrange(0, rows-scales_block, blk):
        for j in xrange(0, cols-scales_block, blk):
            for ki in range(0, scale_length):
                out_len += 7

    return _feature_orb(ch_bd, blk, scales_array, scales_half, scales_block, scale_length,
                        out_len, rows, cols, scale_length, max_features)


def main():

    with raster_tools.ropen('data/test.tif') as i_info:

        a = i_info.read(bands2open=1,
                        d_type='float32')

        a = np.uint8(rescale_intensity(a, out_range=(0, 255)))

        # orb = cv2.ORB_create(nfeatures=20000, edgeThreshold=31, patchSize=31, WTA_K=4)
        # key_points, __ = orb.detectAndCompute(np.uint8(a), None)
        # key_point_array = _fill_key_points(key_points)

        feature_orb(a, 8, [8], 8, max_features=20000)

        # a = np.sqrt(a)
        #
        # grad_img, ori_img = get_mag_ang(a)
        #
        # b = _hog.feature_hog(grad_img, ori_img, 8, [16], 16,
        #                      pixels_per_cell=[4, 4],
        #                      cells_per_block=[3, 3])
        #
        # print b
        # print b.shape


if __name__ == '__main__':
    # unittest.main()
    main()
