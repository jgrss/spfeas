#!/usr/bin/env python

import sys

# NumPy
try:
    import numpy as np
except ImportError:
    raise ImportError('NumPy did not load')

# OpenCV
try:
    import cv2
except ImportError:
    raise ImportError('OpenCV did not load')


class pyramid_hist_sift(object):

    def __init__(self, img, key_pts, start_x, start_y, levels=3):

        """
        Parameters
        ----------
        img -- ndarray
        key_pts -- tuple of key points returned from SIFT or SURF
        """

        self.key_pts = key_pts
        self.levels = levels

        self.img_rows = img.shape[0]
        self.img_cols = img.shape[1]

        self.start_x = start_x
        self.start_y = start_y

        self.count_keypoints(img)

    def count_keypoints(self, img):

        # level 1
        # count keypoints in the image
        self.l1_count = [len(self.key_pts)]

        scs = [2, 4, 8, 16, 32]

        self.l2_count = np.zeros(4).astype(int)
        self.l3_count = np.zeros(16).astype(int)

        # loop through each level
        for sc in xrange(0, self.levels-1):

            y_tiles = (self.img_rows / scs[sc]) + 1

            # loop through each key point
            for kp in xrange(0, len(self.key_pts)):

                grid_count = 0

                # count features at each level
                for ki in xrange(0, self.img_rows, y_tiles):

                    x_tiles = (self.img_cols / scs[sc]) + 1

                    for kj in xrange(0, self.img_cols, x_tiles):

                        self.check_pts(self.key_pts[kp].pt, ki, kj, y_tiles, x_tiles, grid_count, sc)

                        grid_count += 1

        self.sp_hist = np.concatenate((self.l1_count, self.l2_count, self.l3_count)).astype(np.float64)

    def check_pts(self, pts, ki, kj, y_tiles, x_tiles, grid_count, sc):

        """
        pts = (x,y)
        """

        if (pts[1] >= ki+self.start_y) and (pts[1] < ki+y_tiles+self.start_y) and (pts[0] >= kj+self.start_x) \
            and (pts[0] < kj+x_tiles+self.start_x):

            if sc == 0:

                self.l2_count[grid_count] += 1

            elif sc == 1:

                self.l3_count[grid_count] += 1


class pyramid_hist_stats(object):

    def __init__(self, img, levels=2):

        """
        Purpose
        -------
        Get pyramid mean and standard dev. from an image array

        Parameters
        ----------
        img -- ndarray

        Returns
        -------
        Histogram

        levels = 1
            hist length = 2
        levels = 2
            hist length = 10
        levels = 3
            hist length = 42
        """

        self.levels = levels

        self.img_rows = img.shape[0]
        self.img_cols = img.shape[1]

        self.get_stats(img)

    def get_stats(self, img):

        self.sp_hist = []

        # level 1
        # count keypoints in the image
        l_mean, l_std = cv2.meanStdDev(img)

        self.sp_hist.append(l_mean[0][0])
        self.sp_hist.append(l_std[0][0])

        scs = [2, 4, 8, 16, 32]

        # loop through each level
        for sc in xrange(0, self.levels-1):

            y_tiles = self.img_rows / scs[sc]

            grid_count = 0

            # count features at each level
            for ki in xrange(0, self.img_rows, y_tiles):

                x_tiles = self.img_cols / scs[sc]

                for kj in xrange(0, self.img_cols, x_tiles):

                    l_mean, l_std = cv2.meanStdDev(img[ki:ki+y_tiles, kj:kj+x_tiles])

                    self.sp_hist.append(l_mean[0][0])
                    self.sp_hist.append(l_std[0][0])
