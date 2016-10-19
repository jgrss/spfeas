#!/usr/bin/env python

"""
@author: Jordan Graesser
Date Created: 9/24/2011
"""

import sys
import time
import logging
from copy import copy
from six import string_types

from mpglue import raster_tools

try:
    import numpy as np
except ImportError:
    raise ImportError('NumPy must be installed.')

# Scikit-learn
try:
    from sklearn import metrics
except ImportError:
    raise ImportError('Scikits-learn must be installed.')

# Ndimage
try:
    from scipy.ndimage.measurements import label as lab_img
except ImportError:
    raise ImportError('Ndimage must be installed')

# Scikit-image
try:
    from skimage.measure import regionprops
except ImportError:
    raise ImportError('Scikit-image must be installed')

# Matplotlib
try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError('Matplotlib must be installed')

import warnings
warnings.filterwarnings('ignore')


class error_matrix(object):

    """
    Computes accuracy statistics

    Args:
        po_text (str): Predicted and observed labels as a text file, where (predicted, observed)
            are the last two columns.
        po_array (ndarray): Predicted and observed labels as an array, where (predicted, observed)
            are the last two columns.
        header (Optional[bool]): Whether ``file`` or ``predicted_observed`` contains a header. Default is False.
        class_list (Optional[list])
        discrete (Optional[bool])

    Attributes:
        n_classes (int): Number of unique classes.
        class_list (list): List of unique classes.
        e_matrix (ndarray): Error matrix.
        accuracy (float): Overall accuracy.
        report
        f_scores (float)
        f_beta (float)
        hamming (float)
        kappa_score (float)
        mae (float)
        mse (float)
        rmse (float)
        r_squared (float)

    Examples:
        >>> from mappy.sample import error_matrix
        >>>
        >>> emat = error_matrix()
        >>>
        >>> # Get an accuracy report from an array
        >>> emat.get_stats(po_array=test_array)
        >>> print emat.accuracy
        >>>
        >>> # Get an accuracy report from a text file
        >>> emat.get_stats(po_text='/test_samples.txt')
        >>>
        >>> # Write statistics to file
        >>> emat.write_stats('/accuracy_report.txt')

    Returns:
        None, writes to ``files`` if given.

    Reference:
        Overall accuracy:
            where,
                Oacc = D / (N * 100)
                    where,
                        D = total number of correctly labeled samples (on the diagonal)
                        N = total number of samples in the matrix
        Kappa:
            : Measure of agreement
        F1-score:
            where,
                F-measure = 2 * ((Producer's * User's) / (Producer's + User's))
        Producer's accuracy:
            Omission error
                -- "excluding an area from the category in which it truly belongs"
                    - Congalton and Green (1999)

              # correctly classified observed samples for class N
            = ----------------------------------------------------- x 100
              total # of observed (column-wise) samples for class N

        User's accuracy:
            Commission error
                -- "including an area into a category when it does not belong to that category"
                    - Congalton and Green (1999)

              # correctly classified predicted samples for class N
            = ---------------------------------------------------- x 100
              total # of predicted (row-wise) samples for class N

        RMSE:
            where,
                (Square root of (Sum of (x - y)^2) / N)

        |========================================================================|
        |                ********************                                    |
        |                * Confusion Matrix *                                    |
        |                ********************                                    |
        |                                                                        |
        |                   Observed/                                            |
        |                   Reference                                            |
        |                  ---------------                                       |
        |                  C0   C1   C2   ..   Cn  | Column totals | User's (%)  |
        |                  ----------------------- | ------------- | ----------- |
        |  Predicted/| C0 |(#)                     | sum(C0 row)   | %           |
        |  Labeled   | C1 |     (#)                | sum(C1 row)   | %           |
        |            | C2 |          (#)           | sum(C2 row)   | %           |
        |            | .. |               ..       | ..            | %           |
        |            | Cn |                   (#)  | sum(Cn row)   | %           |
        |       Row totals|                        | (TOTAL)       |             |
        |   Producer's (%)| %    %    %   ..   %   |               | (Overall %) |
        |========================================================================|
    """

    def __init__(self):

        self.time_stamp = time.asctime(time.localtime(time.time()))

    def get_stats(self, po_text=None, po_array=None, header=False, class_list=[], discrete=True):

        self.discrete = discrete

        if isinstance(po_text, str):

            samples = np.genfromtxt(po_text, delimiter=',').astype(int)

        else:

            try:
                samples = po_array.copy()
            except ValueError:
                raise ValueError('Observed and predicted labels must be passed.')

        if header:
            hdr_idx = 1
        else:
            hdr_idx = 0

        # observed (true)
        y = np.asarray(samples[hdr_idx:, -1].astype(np.float32)).astype(int)

        # predicted
        X = np.asarray(samples[hdr_idx:, -2].astype(np.float32)).astype(int)

        self.n_samps = len(y)

        if not class_list:
            # get unique class values
            class_list1 = reduce(lambda X, y: X + y if y[0] not in X else X, map(lambda X: [X], y))
            class_list2 = reduce(lambda y, X: y + X if X[0] not in y else y, map(lambda y: [y], X))

            self.merge_lists(class_list1, class_list2)
        else:
            self.class_list = class_list

        self.n_classes = len(self.class_list)
        self.class_list = sorted(self.class_list)
        self.n_samples = y.shape[0]

        if self.discrete:

            # create the error matrix
            self.e_matrix = np.zeros((self.n_classes, self.n_classes)).astype(int)

            # add to error matrix
            for predicted, observed in zip(X, y):
                self.e_matrix[self.class_list.index(predicted), self.class_list.index(observed)] += 1

            # Producer's and User's accuracy
            self.producers_accuracy()
            self.users_accuracy()

            # overall accuracy
            self.accuracy = metrics.accuracy_score(y, X) * 100.

            # statistics report
            self.report = metrics.classification_report(y, X)

            # get f scores for each class
            self.f_scores = metrics.f1_score(y, X, average=None)

            # get the weighted f beta score
            self.f_beta = metrics.fbeta_score(y, X, beta=.5, labels=self.class_list, pos_label=self.class_list[1])

            # get the hamming loss score
            self.hamming = metrics.hamming_loss(y, X)

            # get the Kappa score
            self.kappa(y, X)

        else:

            # get the mean absolute error
            self.mae = metrics.mean_absolute_error(y, X)

            # get the mean square error
            self.mse = metrics.mean_squared_error(y, X)

            # get the median absolute error
            self.medae = metrics.median_absolute_error(y, X)

            # get the root mean squared error
            self.rmse = np.sqrt(self.mse)

            # get the r squared
            self.r_squared = metrics.r2_score(y, X)

    def producers_accuracy(self):

        self.producers = np.zeros(self.n_classes, dtype='float32')

        producer_sums = self.e_matrix.sum(axis=0)

        for pr_j in xrange(0, self.n_classes):
            self.producers[pr_j] = (self.e_matrix[pr_j, pr_j] / float(producer_sums[pr_j])) * 100.

        self.producers[np.isnan(self.producers) | np.isinf(self.producers)] = 0.

    def users_accuracy(self):

        self.users = np.zeros(self.n_classes, dtype='float32')

        user_sums = self.e_matrix.sum(axis=1)

        for pr_i in xrange(0, self.n_classes):
            self.users[pr_i] = (self.e_matrix[pr_i, pr_i] / float(user_sums[pr_i])) * 100.

        self.users[np.isnan(self.users) | np.isinf(self.users)] = 0.

    def merge_lists(self, list1, list2):

        self.class_list = copy(list1)

        for value2 in list2:

            if value2 not in list1:
                self.class_list.append(value2)

    def kappa(self, y_true, y_pred, weights=None, allow_off_by_one=False):

        """
        Calculates the kappa inter-rater agreement between two the gold standard
        and the predicted ratings. Potential values range from -1 (representing
        complete disagreement) to 1 (representing complete agreement).  A kappa
        value of 0 is expected if all agreement is due to chance.

        In the course of calculating kappa, all items in ``y_true`` and ``y_pred`` will
        first be converted to floats and then rounded to integers.

        It is assumed that ``y_true`` and ``y_pred`` contain the complete range of possible
        ratings.

        This function contains a combination of code from yorchopolis's kappa-stats
        and Ben Hamner's Metrics projects on Github.

        Args:
            y_true
            y_pred
            weights (Optional[str or numpy array]): Specifies the weight matrix for the calculation. Choices
                are [None :: unweighted-kappa, 'quadratic' :: quadratic-weighted kappa, 'linear' ::
                linear-weighted kappa, two-dimensional numpy array :: a custom matrix of weights. Each weight
                corresponds to the :math:`w_{ij}` values in the wikipedia description of how to calculate
                weighted Cohen's kappa.]
            allow_off_by_one (Optional[bool]): If true, ratings that are off by one are counted as equal, and
                all other differences are reduced by one. For example, 1 and 2 will be considered to be
                equal, whereas 1 and 3 will have a difference of 1 for when building the weights matrix.

        Reference:
            Authors: SciKit-Learn Laboratory
            https://skll.readthedocs.org/en/latest/_modules/skll/metrics.html
        """

        logger = logging.getLogger(__name__)

        # Ensure that the lists are both the same length
        assert(len(y_true) == len(y_pred))

        # This rather crazy looking typecast is intended to work as follows:
        # If an input is an int, the operations will have no effect.
        # If it is a float, it will be rounded and then converted to an int
        # because the ml_metrics package requires ints.
        # If it is a str like "1", then it will be converted to a (rounded) int.
        # If it is a str that can't be typecast, then the user is
        # given a hopefully useful error message.
        try:
            y_true = [int(round(float(y))) for y in y_true]
            y_pred = [int(round(float(y))) for y in y_pred]
        except ValueError as e:
            logger.error("For kappa, the labels should be integers or strings " +
                         "that can be converted to ints (E.g., '4.0' or '3').")
            raise e

        # Figure out normalized expected values
        min_rating = np.minimum(np.min(y_true), np.min(y_pred))
        max_rating = np.maximum(np.max(y_true), np.max(y_pred))

        # shift the values so that the lowest value is 0
        # (to support scales that include negative values)
        y_true = [y - min_rating for y in y_true]
        y_pred = [y - min_rating for y in y_pred]

        # Build the observed/confusion matrix
        num_ratings = max_rating - min_rating + 1
        obsv = metrics.confusion_matrix(y_true, y_pred, labels=list(range(num_ratings)))
        num_scored_items = float(len(y_true))

        # Build weight array if weren't passed one
        if isinstance(weights, string_types):
            wt_scheme = weights
            weights = None
        else:
            wt_scheme = ''
        if weights is None:
            weights = np.empty((num_ratings, num_ratings))

            for i in range(num_ratings):
                for j in range(num_ratings):
                    diff = np.abs(i - j)
                    if allow_off_by_one and diff:
                        diff -= 1
                    if wt_scheme == 'linear':
                        weights[i, j] = diff
                    elif wt_scheme == 'quadratic':
                        weights[i, j] = diff ** 2
                    elif not wt_scheme:  # unweighted
                        weights[i, j] = bool(diff)
                    else:
                        raise ValueError(('Invalid weight scheme specified for ' +
                                          'kappa: {}').format(wt_scheme))

        hist_true = np.bincount(y_true, minlength=num_ratings)
        hist_true = hist_true[: num_ratings] / num_scored_items
        hist_pred = np.bincount(y_pred, minlength=num_ratings)
        hist_pred = hist_pred[: num_ratings] / num_scored_items
        expected = np.outer(hist_true, hist_pred)

        # Normalize observed array
        obsv = obsv / num_scored_items

        # If all weights are zero, that means no disagreements matter.
        self.kappa_score = 1.

        if np.count_nonzero(weights):
            self.kappa_score -= (np.sum(np.sum(weights * obsv)) / np.sum(np.sum(weights * expected)))

    def write_stats(self, out_report):

        """
        Args:
            out_report (str): The file to write to.
        """

        with open(out_report, 'a') as write_txt:

            if self.discrete:

                # write statistics to text file
                write_txt.write('============\n')
                write_txt.write('Error Matrix\n')
                write_txt.write('============\n\n')

                write_txt.write('               Observed\n')
                write_txt.write('               --------\n')
                write_txt.write('               Class %d   ' % self.class_list[0])

                for c in xrange(1, self.n_classes):

                    write_txt.write('Class %d   ' % self.class_list[c])

                write_txt.write('Total   User(%)\n')

                write_txt.write('               -------   ')

                for c in xrange(0, self.n_classes-1):

                    write_txt.write('-------   ')

                write_txt.write('-----   -------\n')

                for i in xrange(0, self.n_classes):

                    if i == 0:
                        write_txt.write('Predicted| C%d| (' % self.class_list[0])
                    elif self.class_list[i] >= 10:
                        write_txt.write('          C%d| ' % self.class_list[i])
                    else:
                        write_txt.write('           C%d| ' % self.class_list[i])

                    for j in xrange(0, self.n_classes):

                        spacer = len(str(int(self.e_matrix[i, j])))

                        if spacer == 1: spacer = 9
                        elif spacer == 2: spacer = 8
                        elif spacer == 3: spacer = 7
                        elif spacer == 4: spacer = 6
                        elif spacer == 5: spacer = 5
                        elif spacer == 6: spacer = 4
                        elif spacer == 7: spacer = 3
                        elif spacer == 8: spacer = 2
                        elif spacer == 9: spacer = 1

                        if i == j and i != 0:
                            write_txt.write('(')
                        if i == j:
                            write_txt.write(str(int(self.e_matrix[i, j])) + ')')

                            for s in xrange(0, spacer-2):
                                write_txt.write(' ')
                        else:
                            write_txt.write(str(int(self.e_matrix[i, j])))

                            for s in xrange(0, spacer):
                                write_txt.write(' ')

                    write_txt.write('{:d}'.format(self.e_matrix[i, :].sum()))

                    spacer = len(str(self.e_matrix[i, :].sum()))

                    if spacer == 1: spacer = 7
                    elif spacer == 2: spacer = 6
                    elif spacer == 3: spacer = 5
                    elif spacer == 4: spacer = 4
                    elif spacer == 5: spacer = 3
                    elif spacer == 6: spacer = 2
                    elif spacer == 7: spacer = 1

                    for s in xrange(0, spacer):
                        write_txt.write(' ')

                    # User's accuracy
                    write_txt.write('{:.2f}\n'.format(self.users[i]))

                write_txt.write('        Total| ')

                for j in xrange(0, self.n_classes):

                    spacer = len(str(int(self.e_matrix[:, j].sum())))

                    if spacer == 1: spacer = 9
                    elif spacer == 2: spacer = 8
                    elif spacer == 3: spacer = 7
                    elif spacer == 4: spacer = 6
                    elif spacer == 5: spacer = 5
                    elif spacer == 6: spacer = 4
                    elif spacer == 7: spacer = 3
                    elif spacer == 8: spacer = 2
                    elif spacer == 9: spacer = 1

                    write_txt.write('{:d}'.format(self.e_matrix[:, j].sum()))

                    for s in xrange(0, spacer):
                        write_txt.write(' ')

                write_txt.write('({:d})\n'.format(self.e_matrix.sum(axis=0).sum()))

                # Producer's accuracy
                write_txt.write('  Producer(%)| ')

                for pr_j in self.producers:

                    pr_jf = '{:.2f}'.format(pr_j)

                    spacer = len(pr_jf)

                    if spacer == 1: spacer = 9
                    elif spacer == 2: spacer = 8
                    elif spacer == 3: spacer = 7
                    elif spacer == 4: spacer = 6
                    elif spacer == 5: spacer = 5
                    elif spacer == 6: spacer = 4
                    elif spacer == 7: spacer = 3
                    elif spacer == 8: spacer = 2
                    elif spacer == 9: spacer = 1

                    write_txt.write(pr_jf)

                    for s in xrange(0, spacer):
                        write_txt.write(' ')

                if len(pr_jf) == 1:
                    sp = 17
                elif len(pr_jf) == 2:
                    sp = 16
                elif len(pr_jf) == 3:
                    sp = 15
                elif len(pr_jf) == 4:
                    sp = 14
                elif len(pr_jf) == 5:
                    sp = 13
                elif len(pr_jf) == 6:
                    sp = 12
                elif len(pr_jf) == 7:
                    sp = 11
                elif len(pr_jf) == 8:
                    sp = 10
                elif len(pr_jf) == 9:
                    sp = 9

                for s in xrange(0, sp-spacer):
                    write_txt.write(' ')

                write_txt.write('({:.2f}%)\n'.format(self.accuracy))

                write_txt.write('\nSamples: {:d}\n'.format(self.n_samps))
                write_txt.write('\n==========\n')
                write_txt.write('Statistics\n')
                write_txt.write('==========\n')
                write_txt.write('\nOverall Accuracy (%): {:.2f}\n'.format(self.accuracy))
                write_txt.write('Kappa: {:.2f}\n'.format(self.kappa_score))
                write_txt.write('F-beta: {:.2f}\n'.format(self.f_beta))
                write_txt.write('Hamming loss: {:.2f}\n'.format(self.hamming))

                write_txt.write('\n============\n')
                write_txt.write('Class report\n')
                write_txt.write('============\n')
                write_txt.write('\n{}'.format(self.report))

            else:

                write_txt.write('=====================\n')
                write_txt.write('Regression statistics\n')
                write_txt.write('=====================\n\n')
                write_txt.write('Mean Absolute Error: {:.4f}\n'.format(self.mae))
                write_txt.write('Median Absolute Error: {:.4f}\n'.format(self.medae))
                write_txt.write('Mean Squared Error: {:.4f}\n'.format(self.mse))
                write_txt.write('Root Mean Squared Error: {:.4f}\n'.format(self.rmse))
                write_txt.write('R squared: {:.4f}\n'.format(self.r_squared))


class object_accuracy(object):

    """
    Assesses object accuracy measures.

    Args:
        reference_array (ndarray)
        predicted_array (ndarray)

    Methods:
        error_array, which is a (5 x rows x columns) array, where the error layers are ...
            1: over-segmentation
            2: under-segmentation
            3: fragmentation
            4: shape error
            5: offset (Euclidean distance (in map units) of object centroids, not found in Persello et al. (2010))

    Reference:
        Persello C and Bruzzone L (2010) A Novel Protocol for Accuracy Assessment in Classification of Very
            High Resolution Images. IEEE Transactions on Geoscience and Remote Sensing, 48(3).

    Examples:
        >>> from mappy.sample import object_accuracy
        >>>
        >>> oi = object_accuracy(reference_array, predicted_array)
        >>> oi.label_objects()
        >>> oi.iterate_objects()
        >>>
        >>> # write to file
        >>> o_info = copy(i_info)
        >>> o_info.storage = 'float32'
        >>> o_info.bands = 3
        >>> oi.write_stats('/out_object_accuracy.tif', o_info)
    """

    def __init__(self, reference_array, predicted_array, image_id=None):

        # predicted_array[predicted_array > 0] = 1
        # reference_array[reference_array > 0] = 1

        self.image_id = image_id

        self.unique_object_ids = np.unique(reference_array)

        self.rows = predicted_array.shape[0]
        self.cols = predicted_array.shape[1]

        self.reference_array = reference_array
        self.predicted_array = predicted_array

        # convert the predictions to binary
        self.predicted_array[self.predicted_array > 0] = 1

        # over_segmentation = band 1
        # under_segmentation = band 2
        # fragmentation = band 3
        # shape error = band 4
        # offset error = band 5
        self.error_array = np.zeros((5, self.rows, self.cols), dtype='float32') + 999

    def label_objects(self):

        # Label the objects of the reference array.
        # self.reference_objects, num_objs = lab_img(self.reference_array)
        self.predicted_objects, num_objs = lab_img(self.predicted_array)

        # get the object properties of the labeled reference array
        # self.props = regionprops(self.reference_objects)

    def iterate_ids(self):

        """
        Iterate over objects, where each object has a unique id
        """

        self.ids = []
        self.over = []
        self.under = []
        self.frag = []
        self.shape = []
        self.dist = []
        self.area = []

        # iterate over each object
        for uoi in self.unique_object_ids:

            # get the current object (binary)
            if uoi == 0:
                continue

            self.reference_sub = np.where(self.reference_array == uoi, 1, 0)

            # This is the reference object sum, which is
            #   O_i in Persello et al. (2010)
            self.reference_object_area = self.reference_sub.sum()

            # Get the object properties of the current object.
            self.props = regionprops(self.reference_sub)

            # Bounding box of the current object (min row, min col, max row, max col)
            object_properties = [(prop.bbox, prop.eccentricity, prop.area) for prop in self.props][0]

            bbox = object_properties[0]
            self.reference_eccentricity = object_properties[1]
            self.reference_area = object_properties[2]

            # subset the object (50 is for padding)
            self.get_padding(bbox, 50)

            self.reference_sub = self.reference_sub[bbox[0]-self.row_min:bbox[2]+50, bbox[1]-self.col_min:bbox[3]+50]
            self.predicted_sub = self.predicted_objects[bbox[0]-self.row_min:bbox[2]+50, bbox[1]-self.col_min:bbox[3]+50]

            # Get the object properties of the current object
            self.props_ = regionprops(self.reference_sub)

            # bounding box of the current object (min row, min col, max row, max col)
            self.reference_centroid = [prop_.centroid for prop_ in self.props_][0]

            # Get predicted object with the maximum number
            #   of overlapping points in the reference object.
            unique_labels = np.unique(self.predicted_sub)

            self.fragments = np.where(self.reference_sub == 1, self.predicted_sub, 0)

            self.max_sum = 0

            # Iterate over each fragment in the predicted array. Here
            #   we iterate over each fragment rather than just doing
            #   an intersection because we are interested in the full
            #   fragment that touches the reference object.
            for unique_label in unique_labels:

                if unique_label == 0:
                    continue

                # Get the pixel count where predicted and the reference object overlap.
                self.overlap_sum = np.where((self.predicted_sub == unique_label) &
                                            (self.reference_sub == 1), 1, 0).sum()

                if self.overlap_sum > self.max_sum:

                    # This is the predicted object sum, which is
                    #   M_i in Persello et al. (2010).
                    self.predicted_sub_ = np.where(self.predicted_sub == unique_label, 1, 0)
                    self.predicted_object_area = self.predicted_sub_.sum()

                    # Get the object properties of the current predicted object
                    pprops = regionprops(self.predicted_sub_)

                    predicted_object_properties = [(pprop.eccentricity, pprop.centroid) for pprop in pprops][0]

                    # Get the eccentricity of the predicted object.
                    self.predicted_eccentricity = predicted_object_properties[0]

                    # Get the centroid of the predicted object.
                    self.predicted_centroid = predicted_object_properties[1]

                    # The object label of the highest overlapping object.
                    self.max_label = copy(unique_label)

                    # This is the union of O_i and M_i in Persello et al. (2010).
                    self.max_sum = copy(self.overlap_sum)

            if self.max_sum == 0:
                continue

            # now, <max_label> has the most overlapping pixels with
            # the reference object, so we can get statistics for it.
            self.error_array[0][self.reference_array == uoi] = self.over_segmentation()
            self.error_array[1][self.reference_array == uoi] = self.under_segmentation()
            self.error_array[2][self.reference_array == uoi] = self.fragmentation()
            self.error_array[3][self.reference_array == uoi] = self.shape_error()
            self.error_array[4][self.reference_array == uoi] = self.offset_error()

            self.ids.append(uoi)
            self.over.append(self.over_segmentation())
            self.under.append(self.under_segmentation())
            self.frag.append(self.fragmentation())
            self.shape.append(self.shape_error())
            self.dist.append(self.offset_error())
            self.area.append(self.reference_area)

        self.error_array[self.error_array == 999] = 0

    def iterate_objects(self):

        """
        Iterate over objects, where each object equals 1 are clearly separated
        """

        # iterate over each object
        for prop in self.props:

            # bounding box of the current object (min row, min col, max row, max col)
            bbox = prop.bbox

            # get the current object (binary)
            self.reference_sub = np.where(self.reference_objects == prop.label, 1, 0)

            # This is the reference object sum, which is
            # O_i in Persello et al. (2010)
            self.reference_object_area = self.reference_sub.sum()

            # subset the object (50 is for padding)
            self.get_padding(bbox, 50)

            self.reference_sub = self.reference_sub[bbox[0]-self.row_min:bbox[2]+50, bbox[1]-self.col_min:bbox[3]+50]
            self.predicted_sub = self.predicted_objects[bbox[0]-self.row_min:bbox[2]+50, bbox[1]-self.col_min:bbox[3]+50]

            # get predicted object with the maximum number of overlapping points in the reference object
            unique_labels = np.unique(self.predicted_sub)

            self.max_sum = 0
            for unique_label in unique_labels:

                # Get the pixel count where predicted and the reference object overlap.
                self.overlap_sum = np.where((self.predicted_sub == unique_label) &
                                            (self.reference_sub == 1), 1, 0).sum()

                if self.overlap_sum >= self.max_sum:

                    # This is the predicted object sum, which is
                    # M_i in Persello et al. (2010)
                    self.predicted_object_area = np.where(self.predicted_sub == unique_label, 1, 0).sum()

                    # The object label of the highest overlapping object.
                    self.max_label = copy(unique_label)

                    # This is the union of O_i and M_i in Persello et al. (2010)
                    self.max_sum = copy(self.overlap_sum)

            # now, <max_label> has the most overlapping pixels with the reference object, so we can get
            # statistics for it
            self.error_array[0][self.reference_objects == prop.label] = self.over_segmentation()
            self.error_array[1][self.reference_objects == prop.label] = self.under_segmentation()
            self.error_array[2][self.reference_objects == prop.label] = self.fragmentation()
            self.error_array[3][self.reference_objects == prop.label] = self.shape_error()
            self.error_array[4][self.reference_objects == prop.label] = self.offset_error()

    def over_segmentation(self):

        """
        0-1 range
            0 = perfect agreement
            1 = high amount of oversegmentation
        """

        # Get the union of the reference object and
        #   the highest overlapping object.
        # union(reference object, highest overlap) / reference object sum (object of interest is 1, so we can sum)
        return 1. - (float(self.max_sum) / self.reference_object_area)

    def under_segmentation(self):

        """
        0-1 range
            0 = perfect agreement
            1 = high amount of undersegmentation
        """

        # union(reference object, highest overlap) / sum of object with highest overlap
        return 1. - (float(self.max_sum) / self.predicted_object_area)

    def fragmentation(self):

        """
        0-1 range
            0 being the optimum case (i.e., only one region is overlapping with the reference object)
            1 is where all the pixels belong to different regions
        """

        # number of regions - 1 / reference total - 1
        n_fragments = np.unique(self.fragments)

        if 0 in n_fragments:
            r_i = len(n_fragments) - 1
        else:
            r_i = len(n_fragments)

        return (r_i - 1.) / (self.reference_object_area - 1.)

    def shape_error(self):

        return abs(self.reference_eccentricity - self.predicted_eccentricity)

    def offset_error(self):

        """
        Returns the euclidean distance of two centroids
        """

        return np.sqrt((self.predicted_centroid[0] - self.reference_centroid[0])**2. +
                       (self.predicted_centroid[1] - self.reference_centroid[1])**2.)

    def get_padding(self, bbox, pad):

        if (bbox[0] - pad) < 0:
            self.row_min = 0
        else:
            self.row_min = bbox[0] - pad

        if (bbox[1] - pad) < 0:
            self.col_min = 0
        else:
            self.col_min = bbox[1] - pad

    def write_report(self, out_report):

        with open(out_report, 'w') as ro:

            ro.write('UNQ,ID,AREA,OVER,UNDER,FRAG,SHAPE,OFFSET\n')

            for unq, ar, ov, un, fr, sh, di in zip(self.ids, self.area,
                                                   self.over, self.under,
                                                   self.frag, self.shape, self.dist):

                ro.write('{:d},{},{:f},{:f},{:f},{:f},{:f},{:f}\n'.format(int(unq), self.image_id,
                                                                          ar, ov, un, fr, sh, di))

    def write_stats(self, out_image, o_info):

        raster_tools.write2raster(self.error_array, out_image, o_info=o_info,
                                  compress='none', tile=False, flush_final=True)