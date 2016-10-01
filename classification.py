#!/usr/bin/env python

"""
@author: Jordan Graesser
Date created: 12/29/2013
"""

import os
import sys
import time
import subprocess
import ast
import platform
import shutil
from copy import copy, deepcopy
# from multiprocessing import Pool
# import multiprocessing as mm
from joblib import Parallel, delayed
import itertools
from collections import OrderedDict
# from operator import itemgetter
# import pathos.multiprocessing as M
# import xml.etree.ElementTree as ET

# MapPy
import raster_tools
import vector_tools
import error_matrix
from paths import gdal_path
# from mappy.helpers.other.progress_iter import _iteration_parameters
from spfeas.paths import get_mappy_path

MAPPY_PATH = get_mappy_path()

# helpers
try:
    from sphelpers.stats import _lin_interp
except:

    os.chdir('{}/helpers/stats'.format(MAPPY_PATH))

    com = 'python setup_lin_interp.py build_ext --inplace'
    subprocess.call(com, shell=True)

    from sphelpers.stats import _lin_interp

try:
    from sphelpers.stats import _rolling_stats
except:

    os.chdir('{}/helpers/stats'.format(MAPPY_PATH))

    com = 'python setup_rolling_stats.py build_ext --inplace'
    subprocess.call(com, shell=True)

    from sphelpers.stats import _rolling_stats

# Pickle
try:
    import cPickle as pickle
except:
    from six.moves import cPickle as pickle
else:
   import pickle

# NumPy
try:
    import numpy as np
except ImportError:
    raise ImportError('NumPy must be installed')

# SciPy
try:
    from scipy import stats
    from scipy.ndimage.interpolation import zoom
    from scipy.interpolate import interp1d
    from scipy.spatial import distance as sci_dist
    # from scipy.signal import savgol_filter
except ImportError:
    raise ImportError('SciPy must be installed')

# GDAL
try:
    from osgeo import gdal
    from osgeo.gdalconst import *
except ImportError:
    raise ImportError('GDAL must be installed')

# OpenCV
try:
    import cv2
except ImportError:
    raise ImportError('OpenCV must be installed')

# Scikit-learn
try:
    from sklearn.feature_selection import chi2
    from sklearn.preprocessing import StandardScaler
    from sklearn import ensemble, tree
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    # from cudatree import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
    from sklearn.naive_bayes import GaussianNB
    from sklearn.covariance import EllipticEnvelope
    from sklearn.cluster import KMeans
    from sklearn.semi_supervised import label_propagation
    from sklearn.grid_search import GridSearchCV
    from sklearn import cross_validation
    from sklearn import metrics
    from sklearn.decomposition import PCA as skPCA
    from sklearn.decomposition import IncrementalPCA
    from sklearn import manifold
    from sklearn import calibration
    from sklearn.feature_selection import VarianceThreshold
except ImportError:
    raise ImportError('Scikit-learn must be installed')

# Matplotlib
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.colors import ListedColormap
    import matplotlib.ticker as ticker
except ImportError:
    raise ImportError('Matplotlib must be installed')

# pd
try:
    import pandas as pd
    # import pd.rpy.common as com
except ImportError:
    raise ImportError('pd must be installed')

# retry
try:
    from retrying import retry
except:
    try:
        print('retrying not found. Attempting to install ...')
        subprocess.call('pip install retrying', shell=True)

        from retrying import retry
    except ImportError:
        raise ImportError('retrying must be installed')

# Rtree
try:
    import rtree
except:
    print('Rtree must be installed to use spatial indexing')

# from numba import jit as numba_jit
# from parakeet import jit as para_jit

import warnings
warnings.filterwarnings('ignore')


# @numba_jit
# def predict_opencv_numba3(feas, dims, input_model):
#
#     n_feas = dims[0]
#     rs = dims[1]
#     cs = dims[2]
#
#     # reshape the features
#     n_samps = rs * cs
#
#     out_feas = np.empty(n_samps).astype(np.uint8)
#
#     feas = feas.T.reshape(n_samps, n_feas)
#
#     for fea_idx in xrange(0, n_samps):
#
#         out_feas[fea_idx] = int(input_model(feas[fea_idx]))
#
#     return out_feas.reshape(cs, rs).T


# @para_jit
# def predict_opencv_parakeet(feas, dims, scaler, input_model):
#
#     n_feas = dims[0]
#     rs = dims[1]
#     cs = dims[2]
#
#     # reshape the features
#     n_samps = rs * cs
#
#     if isinstance(scaler, str) or scaler:
#         feas = scaler.transform(feas.T.reshape(n_samps, n_feas))
#     else:
#         feas = feas.T.reshape(n_samps, n_feas)
#
#     return np.asarray([int(input_model.predict(p_row)) for p_row in feas]).reshape(cs, rs).T
#
#
# @para_jit
# def predict_opencv_parakeet2(feas, dims, scaler, input_model):
#
#     n_feas = dims[0]
#     rs = dims[1]
#     cs = dims[2]
#
#     # reshape the features
#     n_samps = rs * cs
#
#     if isinstance(scaler, str) or scaler:
#         feas = scaler.transform(feas.T.reshape(n_samps, n_feas))
#     else:
#         feas = feas.T.reshape(n_samps, n_feas)
#
#     out_array = np.zeros(n_samps).astype(np.uint8)
#
#     for p_row in xrange(0, n_samps):
#
#         out_array[p_row] = int(input_model.predict(feas[p_row]))
#
#     return out_array
#
#
# @para_jit
# def predict_opencv_parakeet3(feas, dims, input_model):
#
#     n_feas = dims[0]
#     rs = dims[1]
#     cs = dims[2]
#
#     # reshape the features
#     n_samps = rs * cs
#
#     out_feas = np.empty(n_samps).astype(np.uint8)
#
#     feas = feas.T.reshape(n_samps, n_feas)
#
#     for fea_idx in xrange(0, n_samps):
#
#         out_feas[fea_idx] = int(input_model(feas[fea_idx]))
#
#     return out_feas.reshape(cs, rs).T


# def _predict_opencv01(features_1d, model_name, predictor):
#
#     """
#     loads the model for each sample
#
#     features -- (, n_features) ndarray
#     model_name -- str
#         : Full path to the .xml model
#     predictor -- str
#         : Name of model ('CVRF', 'CVEX_RF')
#
#     Returns
#     -------
#     Predicted class label as an integer
#     """
#
#     predictors = {'CVRF': cv2.RTrees(), 'CVEX_RF': cv2.ERTrees()}
#
#     model = predictors[predictor]
#     model.load(model_name)
#
#     return int(model.predict(features_1d))


def predict_pp(ci, cs):
    predicted[ci:ci+cs] = model_pp.predict(features[ci:ci+cs])


# def predict_cv(arg, **kwarg):
#     classification.predict_cv(*arg, **kwarg)


def _do_c5_cubist_predict(c5_cubist_model, classifier_name, predict_samps, rows_i=None):

    """
    A C5/Cubist prediction function

    Args:
        c5_cubist_model (object):
        classifier_name (str):
        predict_samps (rpy2 array): An array of features to make predictions on.
        rows_i (Optional[rpy2 object]): A R/rpy2 model instance of feature rows to make predictions on. If not passed,
            predictions are made on all rows. Default is None.

    Returns:
        NumPy 1d array of class predictions
    """

    if classifier_name == 'C5':

        if not rows_i:
            return np.array(C50.predict_C5_0(c5_cubist_model, newdata=predict_samps, type='class'), dtype='uint8')
        else:
            return np.array(C50.predict_C5_0(c5_cubist_model, newdata=predict_samps.rx(rows_i, True),
                                             type='class'), dtype='uint8')

    elif classifier_name == 'Cubist':

        if not rows_i:
            return np.array(Cubist.predict_cubist(c5_cubist_model, newdata=predict_samps), dtype='float32')
        else:
            return np.array(Cubist.predict_cubist(c5_cubist_model, newdata=predict_samps.rx(rows_i, True)),
                            dtype='float32')


def predict_c5_cubist(input_model, ip):

    """
    A C5/Cubist prediction function for parallel predictions

    Args:
        input_model (str): The model file to load.
        ip (list): A indice list of rows to extract from ``predict_samps``.
    """

    ci, m, h = pickle.load(file(input_model, 'rb'))

    rows_i = ro.IntVector(range(ip[0], ip[0]+ip[1]))

    if ci['classifier'] == 'C5':
        # TODO: type='prob'
        return np.array(C50.predict_C5_0(m, newdata=predict_samps.rx(rows_i, True), type='class'), dtype='uint8')
    else:
        return np.array(Cubist.predict_cubist(m, newdata=predict_samps.rx(rows_i, True)), dtype='float32')


def predict_scikit(input_model, ip):

    """
    A Scikit-learn prediction function for parallel predictions
    """

    if isinstance(input_model, str):
        __, m = pickle.load(file(input_model, 'rb'))
    else:
        m = input_model

    return m.predict(features[ip[0]:ip[0]+ip[1]])


def predict_cv(ci, cs, fn, pc, cr, ig, xy, cinfo, wc):

    """
    This is an ugly (and hopefully temporary) hack to get around the missing OpenCV model ``load`` method.
    """

    cl = classification()
    cl.split_samples(fn, perc_samp=pc, classes2remove=cr, ignore_feas=ig, use_xy=xy)
    cl.construct_model(classifier_info=cinfo, weight_classes=wc, be_quiet=True)

    return cl.model.predict(features[ci:ci+cs])[1]


def get_available_models():

    """
    Get a list of available models
    """

    return ['AB_DT', 'AB_EX_DT', 'AB_RF', 'AB_EX_RF', 'AB_Bag', 'AB_DTR', 'AB_EX_DTR',
            'AB_RFR', 'AB_EX_RFR', 'AB_BagR',
            'Bag', 'BagR', 'Bayes', 'DT', 'DTR',
            'EX_DT', 'EX_DTR', 'GB', 'GBR', 'C5', 'Cubist',
            'EX_RF', 'CVEX_RF', 'EX_RFR',
            'Logistic', 'NN',
            'RF', 'CVGBoost', 'CVRF', 'RFR', 'CVMLP',
            'SVM', 'SVMR', 'CVSVM', 'CVSVMA', 'CVSVR', 'CVSVRA', 'QDA']


class ParameterHandler(object):

    def __init__(self, classifier):

        self.equal_params = {'trees': 'n_estimators',
                             'min_samps': 'min_samples_split'}

        self.forests = ['RF', 'EX_RF']
        self.forests_regressed = ['RFR', 'EX_RFR']

        self.bagged = ['Bag', 'BagR']

        self.trees = ['DT', 'EX_DT']
        self.trees_regressed = ['DTR', 'EX_DTR']

        self.boosted = ['AB_DT', 'AB_EX_DT', 'AB_RF', 'AB_EX_RF', 'AB_Bag']

        self.boosted_g = ['GB']
        self.boosted_g_regressed = ['GBR']

        if classifier in self.forests:

            self.valid_params = ['n_estimators', 'criterion', 'max_depth', 'min_samples_split',
                                 'min_samples_leaf', 'min_weight_fraction_leaf', 'max_features',
                                 'max_leaf_nodes', 'bootstrap', 'oob_score', 'n_jobs',
                                 'random_state', 'verbose', 'warm_start', 'class_weight']

        elif classifier in self.forests_regressed:

            self.valid_params = ['n_estimators', 'criterion', 'max_depth', 'min_samples_split',
                                 'min_samples_leaf', 'min_weight_fraction_leaf', 'max_features',
                                 'max_leaf_nodes', 'bootstrap', 'oob_score', 'n_jobs',
                                 'random_state', 'verbose', 'warm_start']

        elif classifier in self.bagged:

            self.valid_params = ['base_estimator', 'n_estimators', 'max_samples', 'max_features',
                                 'bootstrap', 'bootstrap_features', 'oob_score', 'warm_start',
                                 'n_jobs', 'random_state', 'verbose']

        elif classifier in self.trees:

            self.valid_params = ['criterion', 'splitter', 'max_depth', 'min_samples_split',
                                 'min_samples_leaf', 'min_weight_fraction_leaf', 'max_features',
                                 'random_state', 'max_leaf_nodes', 'class_weight', 'presort']

        elif classifier in self.trees_regressed:

            self.valid_params = ['criterion', 'splitter', 'max_depth', 'min_samples_split', 'min_samples_leaf',
                                 'min_weight_fraction_leaf', 'max_features', 'random_state', 'max_leaf_nodes',
                                 'presort']

        elif classifier in self.boosted_g:

            self.valid_params = ['loss', 'learning_rate', 'n_estimators', 'subsample', 'min_samples_split',
                                 'min_samples_leaf', 'min_weight_fraction_leaf', 'max_depth', 'init',
                                 'random_state', 'max_features', 'verbose', 'max_leaf_nodes', 'warm_start',
                                 'presort']

        elif classifier in self.boosted_g_regressed:

            self.valid_params = ['loss', 'learning_rate', 'n_estimators', 'subsample', 'min_samples_split',
                                 'min_samples_leaf', 'min_weight_fraction_leaf', 'max_depth', 'init',
                                 'random_state', 'max_features', 'alpha', 'verbose', 'max_leaf_nodes',
                                 'warm_start', 'presort']

        elif classifier in self.boosted:
            self.valid_params = ['base_estimator', 'n_estimators', 'learning_rate', 'algorithm', 'random_state']

        elif classifier == 'NN':

            self.valid_params = ['n_neighbors', 'weights', 'algorithm', 'leaf_size', 'p', 'metric',
                                 'metric_params', 'n_jobs']

        elif classifier == 'Logistic':

            self.valid_params = ['penalty', 'dual', 'tol', 'C', 'fit_intercept', 'intercept_scaling',
                                 'class_weight', 'random_state', 'solver', 'max_iter', 'multi_class',
                                 'verbose', 'warm_start', 'n_jobs']

        else:
            raise NameError('The classifier is not supported.')

    def check_parameters(self, cinfo, default_params, trials_set=False):

        # Set defaults
        for k, v in default_params.iteritems():

            if k not in cinfo and k in self.valid_params:
                cinfo[k] = v

        for param_key, param_value in cinfo.copy().iteritems():

            if param_key in self.equal_params:

                if param_key == 'trials':

                    if not trials_set:

                        cinfo[self.equal_params[param_key]] = param_value
                        del cinfo[param_key]

                else:
                    if self.equal_params[param_key] in cinfo:
                        param_key_ = copy(param_key)
                        param_key = self.equal_params[param_key]
                        del cinfo[param_key_]

            if param_key not in self.valid_params and param_key in cinfo:
                del cinfo[param_key]

        return cinfo


class Samples(object):

    """
    A class to handle data samples

    Args:
        file_name (str): Input .txt file with samples and labels.
        perc_samp (Optional[float]): Percent to sample from all samples. Default is .9. *This parameter
            samples from the entire set of samples, regardless of which class they are in.
        perc_samp_each (Optional[float]): Percent to sample from each class. Default is 0. *This parameter
            overrides ``perc_samp`` and forces a percentage of samples from each class.
        scale_data (Optional[bool]): Whether to scale (normalize) data. Default is False.
        class_subs (Optional[dict]): Dictionary of class percentages or number to sample. Default is empty, or None.
            Example:
                Sample by percentage = {1:.9, 2:.9, 3:.5}
                Sample by integer = {1:300, 2:300, 3:150}
        header (Optional[bool]): Whether the samples contain a header. Default is True.
        norm_struct (Optional[bool]): Whether the structure of the data is normals Default is True. In MapPy's
            case, normal is (X,Y,Var1,Var2,Var3,Var4,...,VarN,Labels), whereas the alternative (i.e., False) is
            (Labels,Var1,Var2,Var3,Var4,...,VarN)
        labs_type (Optional[str]): Read class labels as integer ('int') or float ('float'). Default is 'int'.
        recode_dict (Optional[dict]): Dictionary of classes to recode. Default is {}, or empty dictionary.
        classes2remove (Optional[list]): List of classes to remove from samples. Default is [], or keep
            all classes.
        sample_weight (Optional[list or 1d array]): Sample weights. Default is None.
        ignore_feas (Optional[list]): A list of feature (image layer) indexes to ignore. Default is [], or use all
            features. *The features are sorted.
        use_xy (Optional[bool]): Whether to use the x, y coordinates as predictive variables. Default is False.
        stratified (Optional[bool]): Whether to stratify the samples. Default is False.
        spacing (Optional[float]): The grid spacing (meters) to use for stratification (in ``stratified``).
            Default is 1000.

    Attributes:
        file_name (str)
        p_vars (ndarray)
        p_vars_test (ndarray)
        labels (list)
        labels_test (list)
        use_xy (bool)
        headers (list)
        all_samps (ndarray)
        XY (ndarray)
        n_samps (int)
        n_feas (int)
        classes (list)
        class_counts (dict)
    """

    def __init__(self):

        self.time_stamp = time.asctime(time.localtime(time.time()))

    def split_samples(self, file_name, perc_samp=.9, perc_samp_each=0, scale_data=False, class_subs={},
                      header=True, norm_struct=True, labs_type='int', recode_dict={}, classes2remove=[],
                      sample_weight=None, ignore_feas=[], use_xy=False, stratified=False, spacing=1000.,
                      x_label='X', y_label='Y', response_label='response'):

        if platform.system() == 'Windows':
            self.file_name = file_name.replace('\\', '/')
        else:
            self.file_name = file_name

        self.labels_test = None
        self.p_vars = None
        self.p_vars_test = None
        self.labels = None
        self.use_xy = use_xy
        self.perc_samp = perc_samp
        self.perc_samp_each = perc_samp_each
        self.classes2remove = classes2remove
        self.sample_weight = sample_weight

        if isinstance(self.sample_weight, list):
            self.sample_weight = np.array(self.sample_weight, dtype='float32')

        # Open the data samples.
        df = pd.read_csv(self.file_name, sep=',')

        # Parse the headers.
        self.headers = df.columns.values.tolist()

        if norm_struct:

            data_position = 2

            self.headers = self.headers[self.headers.index(x_label):]

            # The response index position.
            self.label_idx = -1

        else:

            self.headers = self.headers[self.headers.index(response_label):]

            # The response index position.
            self.label_idx = 0

            data_position = 0

        # Parse the x variables.
        self.all_samps = df.loc[:, self.headers[data_position:]].values

        # Parse the x, y coordinates.
        self.XY = df[[x_label, y_label]].values

        # Spatial stratified sampling.
        if stratified:

            self.use_xy = True

            min_x = self.XY[:, 0].min()
            max_x = self.XY[:, 0].max()

            min_y = self.XY[:, 1].min()
            max_y = self.XY[:, 1].max()

            x_grids = np.arange(min_x, max_x+spacing, spacing)
            y_grids = np.arange(min_y, max_y+spacing, spacing)

        # Remove specified x variables.
        if ignore_feas:

            ignore_feas_ = [f-1 for f in ignore_feas]
            ignore_feas = sorted([int(f-1) for f in ignore_feas])
            self.all_samps = np.delete(self.all_samps, ignore_feas, axis=1)

            temp_headers = self.headers[2:-1]

            for offset, index in enumerate(ignore_feas_):
                index -= offset
                del temp_headers[index]

            self.headers = self.headers[:2] + temp_headers + [self.headers[-1]]

        # Append the x, y coordinates to the x variables.
        if self.use_xy:
            self.all_samps = np.c_[self.all_samps[:, :-1], self.XY, self.all_samps[:, -1]]
            self.headers = self.headers[2:-1] + self.headers[:2] + [self.headers[-1]]
        else:
            # Remove x, y
            self.headers = self.headers[2:]

        # Remove unwanted classes.
        if self.classes2remove:
            self._remove_classes(self.classes2remove)

        # Get the number of samples and x variables.
        #   rows = number of samples
        #   cols = number of features
        self.n_samps = self.all_samps.shape[0]
        self.n_feas = self.all_samps.shape[1] - 1

        if isinstance(self.sample_weight, np.ndarray):
            assert len(self.sample_weight) == self.n_samps

        # Recode specified classes.
        if recode_dict:

            new_samps = np.zeros(self.all_samps.shape[0], dtype='int16')
            temp_labels = self.all_samps[:, -1]

            for recode_key, cl in sorted(recode_dict.iteritems()):
                new_samps[temp_labels == recode_key] = cl

            self.all_samps[:, -1] = new_samps

        # Sample a specified number per class.
        if class_subs:

            counter = 1

            for class_key, cl in sorted(class_subs.iteritems()):

                # Get all the samples that match the current class.
                curr_cl = self.all_samps[np.where(self.all_samps[:, self.label_idx] == class_key)]

                # Shuffle rows (samples) for randomness.
                np.random.shuffle(curr_cl)

                # Check for float or integer.
                if isinstance(cl, float):
                    ran = np.random.choice(range(curr_cl.shape[0]), size=int(cl * curr_cl.shape[0]), replace=False)
                elif isinstance(cl, int):
                    ran = np.random.choice(range(curr_cl.shape[0]), size=cl, replace=False)

                # Create the test samples.
                test_samps_temp = np.delete(curr_cl, ran, axis=0)

                # Get the current samples.
                curr_cl = curr_cl[ran]

                if counter == 1:
                    train_stk = np.copy(curr_cl)
                    test_stk = np.copy(test_samps_temp)
                else:
                    train_stk = np.vstack((train_stk, curr_cl))
                    test_stk = np.vstack((test_stk, test_samps_temp))

                counter += 1

            self.all_samps = np.copy(train_stk)
            test_samps = np.copy(test_stk)

        elif 0 < perc_samp_each < 1:

            # Get unique class values.
            if labs_type == 'int':
                self.labels = np.asarray([int(l) for l in self.all_samps[:, self.label_idx]])
            elif labs_type == 'float':
                self.labels = np.asarray([float(l) for l in self.all_samps[:, self.label_idx]]).astype(np.float32)
            else:
                raise TypeError('\n``labs_type`` should be int or float\n')

            self.classes = list(np.unique(self.labels))

            class_subs = {}

            for clp in self.classes:
                class_subs[clp] = perc_samp_each

            counter = 1

            if isinstance(self.sample_weight, np.ndarray):
                self.all_samps = np.c_[self.sample_weight, self.all_samps]

            for class_key, cl in sorted(class_subs.iteritems()):

                # Get all the samples that match the current class.
                curr_cl = self.all_samps[np.where(self.all_samps[:, self.label_idx] == class_key)]

                # Shuffle rows (samples).
                np.random.shuffle(curr_cl)

                # Check for float or integer.
                if isinstance(cl, float):
                    ran = np.random.choice(range(curr_cl.shape[0]), size=int(cl * curr_cl.shape[0]), replace=False)
                elif isinstance(cl, int):
                    ran = np.random.choice(range(curr_cl.shape[0]), size=cl, replace=False)

                # Create the test samples.
                test_samps_temp = np.delete(curr_cl, ran, axis=0)

                # Get the current samples.
                curr_cl = curr_cl[ran]

                if counter == 1:
                    train_stk = np.copy(curr_cl)
                    test_stk = np.copy(test_samps_temp)
                else:
                    train_stk = np.vstack((train_stk, curr_cl))
                    test_stk = np.vstack((test_stk, test_samps_temp))

                counter += 1

            if isinstance(self.sample_weight, np.ndarray):

                self.all_samps = np.copy(train_stk[:, 1:])
                test_samps = np.copy(test_stk[:, 1:])

                self.sample_weight = train_stk[:, 0]

            else:

                self.all_samps = np.copy(train_stk)
                test_samps = np.copy(test_stk)

        elif (perc_samp < 1) and (perc_samp_each == 0):

            if stratified:

                print 'Stratifying ...'

                n_total_samps = int(perc_samp * self.n_samps)
                n_match_samps = 0

                # We need x, y coordinates, so force it.
                if not self.use_xy:
                    self.all_samps = np.c_[self.all_samps[:, :-1], self.XY, self.all_samps[:, -1]]

                while n_match_samps < n_total_samps:
                    n_match_samps = self._stratify(y_grids, x_grids, n_match_samps, n_total_samps)

                test_samps = copy(self.all_samps)
                self.all_samps = copy(self.stratified_samps)

            else:

                # Shuffle rows (samples).
                np.random.shuffle(self.all_samps)

                # Get random indices.
                ran = np.random.choice(range(self.n_samps), size=int(perc_samp * self.n_samps), replace=False)

                # Extract test samples.
                test_samps = np.delete(self.all_samps, ran, axis=0)

                # Extract training samples.
                self.all_samps = self.all_samps[ran]

                if isinstance(self.sample_weight, np.ndarray):
                    self.sample_weight = self.sample_weight[ran]

        self.n_samps = self.all_samps.shape[0]

        # Get unique class values.
        if labs_type == 'int':
            self.labels = np.asarray([int(l) for l in self.all_samps[:, self.label_idx]])
        elif labs_type == 'float':
            self.labels = np.float32(np.asarray([float(l) for l in self.all_samps[:, self.label_idx]]))
        else:
            raise ValueError('\n``labs_type`` should be int or float\n')

        self.classes = list(np.unique(self.labels))
        self.n_classes = len(self.classes)

        if norm_struct:
            self.p_vars = self.all_samps[:, :self.label_idx].astype(np.float32)
        else:
            self.p_vars = self.all_samps[:, 1:].astype(np.float32)

        self.p_vars[np.isnan(self.p_vars) | np.isinf(self.p_vars)] = 0.

        if ((perc_samp < 1) and (perc_samp_each == 0)) or class_subs or (0 < perc_samp_each < 1):

            if labs_type == 'int':
                self.labels_test = np.asarray([int(l) for l in test_samps[:, self.label_idx]])
            elif labs_type == 'float':
                self.labels_test = np.asarray([float(l) for l in test_samps[:, self.label_idx]]).astype(np.float32)

            if norm_struct:
                self.p_vars_test = test_samps[:, :self.label_idx].astype(np.float32)
            else:
                self.p_vars_test = test_samps[:, 1:].astype(np.float32)

            self.p_vars_test[np.isnan(self.p_vars_test) | np.isinf(self.p_vars_test)] = 0.

            self.p_vars_test_rows = self.p_vars_test.shape[0]
            self.p_vars_test_cols = self.p_vars_test.shape[1]

        # Get individual class counts.
        self._update_class_counts()

        if scale_data:

            d_name, f_name = os.path.split(self.file_name)
            f_base, f_ext = os.path.splitext(f_name)

            scaler_file = '{}/{}_scaler.txt'.format(d_name, f_base)

            self.scaler = StandardScaler().fit(self.p_vars)

            # pickle scaler to file for later use
            pickle.dump(self.scaler, file(scaler_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

            # Save the unscaled samples.
            self.p_vars_original = self.p_vars.copy()

            # Scale the data.
            self.p_vars = self.scaler.transform(self.p_vars)

            self.scaled = True

        else:
            self.scaled = False

    def _update_class_counts(self):

        self.classes = map(int, list(np.unique(self.labels)))
        self.n_classes = len(self.classes)

        self.class_counts = {}

        for indv_class in self.classes:
            self.class_counts[indv_class] = len(np.where(self.labels == indv_class)[0])

    def _stratify(self, y_grids, x_grids, n_match_samps, n_total_samps):

        """
        Grid stratification
        """

        for ygi, xgj in itertools.product(range(0, len(y_grids)-1), range(0, len(x_grids)-1)):

            # Get all of the samples in the current grid.
            gi = np.where((self.all_samps[:, -2] >= y_grids[ygi]) &
                          (self.all_samps[:, -2] < y_grids[ygi+1]) &
                          (self.all_samps[:, -3] >= x_grids[xgj]) &
                          (self.all_samps[:, -3] < x_grids[xgj+1]))[0]

            if len(gi) > 0:

                # Randomly sample from the grid samples.
                ran = np.random.choice(range(len(gi)), size=1, replace=False)

                gi_i = gi[ran[0]]

                # Remove the samples.
                if n_match_samps == 0:

                    # Reshape (add 1 for the labels)
                    self.stratified_samps = self.all_samps[gi_i].reshape(1, self.n_feas+1)
                    self.all_samps = np.delete(self.all_samps, gi_i, axis=0)

                else:

                    self.stratified_samps = np.r_[self.stratified_samps, self.all_samps[gi_i].reshape(1, self.n_feas+1)]
                    self.all_samps = np.delete(self.all_samps, gi_i, axis=0)

                n_match_samps += 1

                if n_match_samps >= n_total_samps:
                    return n_match_samps

        return n_match_samps

    def _remove_classes(self, classes2remove):

        """
        Remove specific classes from the data
        """

        for class2remove in classes2remove:

            class2remove_idx = np.where(self.all_samps[:, self.label_idx] == class2remove)

            self.all_samps = np.delete(self.all_samps, class2remove_idx, axis=0)

            if isinstance(self.p_vars, np.ndarray):

                self.p_vars = np.float32(np.delete(self.p_vars, class2remove_idx, axis=0))
                self.labels = np.float32(np.delete(self.labels, class2remove_idx, axis=0))

            if isinstance(self.sample_weight, np.ndarray):
                self.sample_weight = np.float32(np.delete(self.sample_weight, class2remove_idx, axis=0))

    def remove_values(self, value2remove, fea_check):

        """
        Removes values from the sample data

        Args:
            value2remove (int): The value to remove.
            fea_check (int): The feature position to use for checking.

        Attributes:
            p_vars (ndarray)
            labels (ndarray)
        """

        idx = np.where(self.p_vars[:, fea_check-1] < value2remove)

        self.p_vars = np.float32(np.delete(self.p_vars, idx, axis=0))
        self.labels = np.float32(np.delete(self.labels, idx, axis=0))


class EndMembers(object):

    """
    A class for endmember extraction
    """

    def __init__(self):

        self.time_stamp = time.asctime(time.localtime(time.time()))

    def extract_endmembers(self, n_members=2, method='nfindr'):

        """
        Extracts land cover endmembers from training samples. Be sure to remove all classes
        except the one of interest.

        Args:
            n_members (Optional[int]): The number of endmembers to extract. Default is 2.
            method (Optional[str]): The method to use. Default is 'ndfindr'.
                Choices are ['fippi', 'nfinder'].

        Reference:
            http://pysptools.sourceforge.net/nbex_methanol_burner.html

        Examples:
            >>> from mappy.classifiers import classification
            >>> cl = classification()
            >>>
            >>> # remove all classes except the one of interest (here, class 2)
            >>> cl.split_samples('/land_cover_samples.txt', perc_samp=1.,
            >>>                  classes2remove=[1, 3, 4, 5, 6],
            >>>                  ignore_feas=[37, 38])
            >>>
            >>> cl.extract_endmembers()
            >>> print cl.endmembers
        """

        try:
            import pysptools.eea as eea
        except ImportError:
            raise ImportError('Pysptools needs to be installed to extract endmembers')

        map_methods = {'fippi': eea.FIPPI(), 'nfinder': eea.NFINDR()}

        try:
            end_finder = map_methods[method]
        except NameError:
            raise NameError('The {} method is not an option.'.format(method))

        # for an image
        # image.T.reshape(i_info.rows, i_info.cols, i_info.bands)

        self.endmembers = end_finder.extract(self.p_vars.reshape(1, self.n_samps, self.n_feas), n_members, maxit=5)

    def get_abundance(self, input_image, out_image, mask=None, method='nnls', class2keep=1, ignore_feas=[]):

        """
        Gets the land cover abundance based on extracted endmembers.

        Args:
            input_image (str): The image to extract endmembers from.
            out_image (str): The output abundance image.
            mask (Optional[str]): The thematic image with land cover classes. Default is None.
            method (Optional[str]): The mapping method to use. Default is 'nnls'.
                Choices are ['fcls', 'nnls', 'ucls'].
            class2keep (Optional[int]): The land cover class to get abundance from. Default is 1.
            ignore_feas (Optional[int list]): A list of image features to ignore. Default is [].

        Returns:
            None, writes abundance to ``out_image``.

        Examples:
            >>> # An example of automatic endmember extraction for
            >>> #   a chosen class, followed by abundance estimation.
            >>> from mappy.classifiers import classification
            >>> cl = classification()
            >>>
            >>> # Here we load the samples and remove each class, except 2
            >>> cl.split_samples('/08N_points_merged_02.txt', perc_samp=1.,
            >>>                  classes2remove=[1, 3, 4, 5, 6], ignore_feas=[37, 38])
            >>>
            >>> # The endmembers for class 2 are automatically extracted.
            >>> cl.extract_endmembers()
            >>>
            >>> # The endmembers are then used to estimate abundance
            >>> #   and the <class2keep> parameter is only for masking.
            >>> cl.get_abundance('/subs/08N_all_bands.tif',
            >>>                  '/subs/08N_all_bands_pasture_abundance.tif',
            >>>                  mask='mask.tif', class2keep=2)
        """

        try:
            import pysptools.abundance_maps as amp
        except ImportError:
            raise ImportError('Pysptools needs to be installed to extract endmembers')

        map_methods = {'fcls': amp.FCLS(), 'nnls': amp.NNLS(), 'ucls': amp.UCLS()}

        try:
            mapper = map_methods[method]
        except NameError:
            raise NameError('The {} method is not an option.'.format(method))

        # open the input image
        i_info = raster_tools.rinfo(input_image)
        arr = i_info.mparray(bands2open=-1)

        if ignore_feas:

            ignore_feas = map(int, ignore_feas)
            arr = np.delete(arr, ignore_feas, axis=0)

        arr_dims, arr_rows, arr_cols = arr.shape

        self.abundance_maps = mapper.map(arr.T.reshape(arr_rows, arr_cols, arr_dims), self.endmembers, normalize=True)

        r, c, b = self.abundance_maps.shape

        self.abundance_maps = self.abundance_maps.reshape(c, r, b).T

        o_info = i_info.copy()
        o_info.update_info(storage='float32', bands=1)

        if isinstance(mask, str):

            d_name, f_name = os.path.split(out_image)
            f_base, f_ext = os.path.splitext(f_name)

            out_img_not_masked = '{}/{}_not_masked{}'.format(d_name, f_base, f_ext)

            raster_tools.write2raster(self.abundance_maps[1], out_img_not_masked, o_info=o_info, flush_final=True)

        else:

            raster_tools.write2raster(self.abundance_maps[1], out_image, o_info=o_info, flush_final=True)

        if isinstance(mask, str):

            a_info = raster_tools.rinfo(out_img_not_masked)
            m_info = raster_tools.rinfo(mask)

            a_arr = a_info.mparray()
            m_arr = m_info.mparray()

            a_arr[m_arr != class2keep] = 0

            raster_tools.write2raster(a_arr, out_image, o_info=o_info, flush_final=True)

        # plt.imshow(amaps.reshape(c, r, b).T[1])
        # plt.show()

        i_info.close()


class Visualization(object):

    """
    A class for data visualization
    """

    def __init__(self):

        self.time_stamp = time.asctime(time.localtime(time.time()))

    def vis_parallel_coordinates(self):

        """
        Visualize time series data in parallel coordinates style

        Examples:
            >>> from mappy.classifiers import classification
            >>> cl = classification()
            >>>
            >>> cl.split_samples('/samples.txt', perc_samp_each=.5)
            >>> cl.vis_parallel_coordinates()
        """

        ax = plt.figure().add_subplot(111)

        x = range(self.p_vars.shape[1])

        colors = {1: 'black', 2: 'cyan', 3: 'yellow', 4: 'red', 5: 'orange', 6: 'green',
                  7: 'purple', 8: 'magenta', 9: '#5F4C0B', 10: '#21610B', 11: '#210B61'}

        leg_items = []
        leg_names = []

        for class_label in self.classes:

            idx = np.where(self.labels == class_label)
            current_class_array = self.p_vars[idx]

            for current_class in current_class_array:

                p = ax.plot(x, current_class, c=colors[class_label], label=class_label)

                leg_items.append(p)
                leg_names.append(str(class_label))

        plt.legend(tuple(leg_items), tuple(leg_names),
                   scatterpoints=1,
                   loc='upper left',
                   ncol=3,
                   fontsize=12)

        plt.show()

        plt.close()

    def vis_dimensionality_reduction(self, method='pca', n_components=3, class_list=[], class_names={}, labels=None):

        """
        Visualize dimensionality reduction

        Args:
            method (Optional[str]): Reduction method. Choices are ['pca' (default) :: Principal Components Analysis,
                'spe' :: Spectral Embedding (also known as Laplacian Eigenmaps),
                'tsne' :: t-distributed Stochastic Neighbor Embedding].
            n_components (Optional[int]): The number of components to return. Default is 3.
            class_list (Optional[list]): A list of classes to compare. The default is an empty list, or all classes.
            class_names (Optional[dict]): A dictionary of class names. The default is an empty dictionary, so the
                labels are the class values.

        Examples:
            >>> from mappy.classifiers import classification
            >>> cl = classification()
            >>>
            >>> cl.split_samples('/samples.txt')
            >>> cl.vis_dimensionality_reduction(n_components=3)
        """

        if method == 'spe':

            embedder = manifold.SpectralEmbedding(n_components=n_components, random_state=0, eigen_solver='arpack')

            # transform the variables
            self.p_vars_reduced = embedder.fit_transform(self.p_vars)

        elif method == 'pca':

            skPCA_ = skPCA(n_components=n_components)
            skPCA_.fit(self.p_vars)
            self.p_vars_reduced = skPCA_.transform(self.p_vars)

            # mn, eigen_values = cv2.PCACompute(self.p_vars.T, self.p_vars.T.mean(axis=0).reshape(1, -1),
            #                                   maxComponents=n_components)

            # self.p_vars_reduced = eigen_values.T

        elif method == 'tsne':

            tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)

            self.p_vars_reduced = tsne.fit_transform(self.p_vars)

        if n_components > 2:
            ax = plt.figure().add_subplot(111, projection='3d')
        else:
            ax = plt.figure().add_subplot(111)

        colors = ['black', 'cyan', 'yellow', 'red', 'orange', 'green', 'purple', 'magenta',
                  '#5F4C0B', '#21610B', '#210B61']

        if class_list:
            n_classes = len(class_list)
        else:
            n_classes = self.n_classes
            class_list = self.classes

        leg_items = []
        leg_names = []

        for n_class in xrange(0, n_classes):

            if class_list:

                if class_names:
                    leg_names.append(str(class_names[class_list[n_class]]))
                else:
                    leg_names.append(str(class_list[n_class]))

            else:
                leg_names.append(str(class_list[n_class]))

            cl_idx = np.where(self.labels == self.classes[n_class])

            if n_components > 2:

                curr_pl = ax.scatter(self.p_vars_reduced[:, 0][cl_idx], self.p_vars_reduced[:, 1][cl_idx],
                                     self.p_vars_reduced[:, 2][cl_idx], c=colors[n_class],
                                     edgecolor=colors[n_class], alpha=.5, label=leg_names[n_class])

            else:

                curr_pl = ax.scatter(self.p_vars_reduced[:, 0][cl_idx], self.p_vars_reduced[:, 1][cl_idx],
                                     c=colors[n_class], edgecolor=colors[n_class], alpha=.5)

            leg_items.append(curr_pl)

            ax.set_xlabel('1st component')
            ax.set_ylabel('2nd component')

        ax.set_xlim3d(self.p_vars_reduced[:, 0].min(), self.p_vars_reduced[:, 0].max())
        ax.set_ylim3d(self.p_vars_reduced[:, 1].min(), self.p_vars_reduced[:, 1].max())

        if n_components > 2:

            ax.set_zlim3d(self.p_vars_reduced[:, 2].min(), self.p_vars_reduced[:, 2].max())

            ax.set_zlabel('3rd component')
            ax.legend()

        else:

            plt.legend(tuple(leg_items), tuple(leg_names),
                       scatterpoints=1,
                       loc='upper left',
                       ncol=3,
                       fontsize=12)

        if labels:

            # plot x, y coordinates as labels
            x, y = self.XY[:, 0], self.XY[:, 1]

            x = x[labels]
            y = y[labels]
            pv = self.p_vars[labels]
            # l = self.labels[labels]

            for i in xrange(0, len(x)):

                ax.annotate('%d, %d' % (int(x[i]), int(y[i])), xy=(pv[i, 0], pv[i, 1]), size=6, color='#1C1C1C',
                            xytext=(-10, 10), bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=.5),
                            arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=0'),
                            textcoords='offset points', ha='right', va='bottom')

        plt.show()

        plt.close()

    def vis_data(self, fea_1, fea_2, fea_3=None, class_list=[], class_names={}, labels=None):

        """
        Visualize classes in feature space

        Args:
            fea_1 (int): The first feature to plot.
            fea_2 (int): The second feature to plot.
            fea_3 (Optional[int]): The optional, third feature to plot. Default is None.
            class_list (Optional[list]): A list of classes to compare. The default is an empty list, or all classes.
            class_names (Optional[dict]): A dictionary of class names. The default is an empty dictionary, so the
                labels are the class values.

        Examples:
            >>> from mappy.classifiers import classification
            >>> cl = classification()
            >>>
            >>> cl.split_samples('/samples.txt', classes2remove=[1, 4],
            >>>                  class_subs={2:.1, 5:.01, 8:.1, 9:.9})
            >>>
            >>> cl.vis_data(1, 2)
            >>> # or
            >>> cl.vis_data(1, 2, fea_3=5, class_list=[3, 5, 8],
            >>>             class_names={3: 'forest', 5: 'agriculture', 8: 'water'})
        """

        if isinstance(fea_3, int):
            ax = plt.figure().add_subplot(111, projection='3d')
        else:
            ax = plt.figure().add_subplot(111)

        colors = ['black', 'cyan', 'yellow', 'red', 'orange', 'green', 'purple', 'magenta',
                  '#5F4C0B', '#21610B', '#210B61']

        if class_list:
            n_classes = len(class_list)
        else:
            n_classes = self.n_classes
            class_list = self.classes

        leg_items = []
        leg_names = []

        for n_class in xrange(0, n_classes):

            if class_list:

                if class_names:
                    leg_names.append(str(class_names[class_list[n_class]]))
                else:
                    leg_names.append(str(class_list[n_class]))

            else:
                leg_names.append(str(class_list[n_class]))

            cl_idx = np.where(self.labels == self.classes[n_class])

            if fea_3:

                curr_pl = ax.scatter(self.p_vars[:, fea_1-1][cl_idx], self.p_vars[:, fea_2-1][cl_idx],
                                     self.p_vars[:, fea_3-1][cl_idx], c=colors[n_class], edgecolor=colors[n_class],
                                     alpha=.5, label=leg_names[n_class])

            else:

                curr_pl = ax.scatter(self.p_vars[:, fea_1-1][cl_idx], self.p_vars[:, fea_2-1][cl_idx],
                                     c=colors[n_class], edgecolor=colors[n_class], alpha=.5)

            leg_items.append(curr_pl)

            # plt.xlabel('Feature: %d' % fea_1)
            # plt.ylabel('Feature: %d' % fea_2)

            ax.set_xlabel('Feature: %d' % fea_1)
            ax.set_ylabel('Feature: %d' % fea_2)

        limits = False

        if limits:

            ax.set_xlim(-1, np.max(self.p_vars[:, fea_1-1]))
            ax.set_ylim(-1, np.max(self.p_vars[:, fea_2-1]))

        if fea_3:

            ax.set_zlabel('Feature: %d' % fea_3)
            ax.legend()

            # if limits:
            #     ax.set_zlim(int(np.percentile(self.p_vars[:, fea_2-1], 1)),
            #                 int(np.percentile(self.p_vars[:, fea_2-1], 100)))

        else:

            plt.legend(tuple(leg_items), tuple(leg_names),
                       scatterpoints=1,
                       loc='upper left',
                       ncol=3,
                       fontsize=12)

        if labels:

            # plot x, y coordinates as labels
            x, y = self.XY[:, 0], self.XY[:, 1]

            x = x[labels]
            y = y[labels]
            pv = self.p_vars[labels]
            # l = self.labels[labels]

            for i in xrange(0, len(x)):

                ax.annotate('%d, %d' % (int(x[i]), int(y[i])), xy=(pv[i, fea_1-1], pv[i, fea_2-1]), size=6,
                            color='#1C1C1C', xytext=(-10, 10), bbox=dict(boxstyle='round,pad=0.5',
                                                                         fc='white', alpha=.5),
                            arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=0'),
                            textcoords='offset points', ha='right', va='bottom')

        plt.show()

        plt.close()

    def vis_decision(self, fea_1, fea_2, classifier_info={'classifier': 'RF'}, class2check=1,
                     compare=1, locate_outliers=False):

        """
        Visualize a model decision function

        Args:
            classifier_info (dict): Parameters for Random Forest, SVM, and Bayes.
            fea_1 (int): The first feature to compare.
            fea_1 (int): The second feature to compare.
            class2check (int): The class value to visualize.
            compare (int): Compare one classifier against itself using different parameters (1), or compare
                several classifiers (2).

        Examples:
            >>> from mappy.classifiers import classification
            >>> cl = classification()
            >>>
            >>> # load 100% of the samples and scale the data
            >>> cl.split_samples('/samples.txt', scale_data=True, perc_samp=1.)
            >>>
            >>> # semi supervised learning
            >>> cl.semi_supervised()
            >>>
            >>> # or train a model
            >>> cl.construct_model()
            >>>
            >>> # remove outliers in the data
            >>> cl.remove_outliers()
            >>>
            >>> # plot the decision
            >>> cl.vis_decision(1, 2)
            >>>
            >>> # Command line
            >>> > ./classification.py -s /samples.txt --scale yes -p 1 --semi yes --outliers yes --decision 1,2,1,2
        """

        self.classifier_info = classifier_info

        self._default_parameters()

        # take only two features
        self.p_vars = self.p_vars[:, [fea_1-1, fea_2-1]]

        # max_depth_2 = classifier_info['max_depth'] + 50
        # C2 = classifier_info['C'] + 5

        colors = ['black', 'cyan', 'yellow', 'red', 'orange', 'green', 'purple', 'magenta', '#5F4C0B', '#21610B',
                  '#210B61']

        cm = plt.cm.gist_stern # plt.cm.RdBu    # for the decision boundaries
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])

        x_min, x_max = self.p_vars[:, 0].min() - .5, self.p_vars[:, 0].max() + .5
        y_min, y_max = self.p_vars[:, 1].min() - .5, self.p_vars[:, 1].max() + .5

        xx, yy = np.meshgrid(np.arange(x_min, x_max, .05), np.arange(y_min, y_max, .05))

        if compare == 1:

            if 'RF' in classifier_info['classifier']:

                clf1 = RandomForestClassifier(**self.classifier_info_rf)

                clf2 = ExtraTreesClassifier(**self.classifier_info_rf)

            elif classifier_info['classifier'] == 'SVM':

                clf1 = SVC(gamma=classifier_info['gamma'], C=classifier_info['C'])
                clf2 = SVC(gamma=classifier_info['gamma'], C=C2)

            elif classifier_info['classifier'] == 'Bayes':

                clf1 = GaussianNB()
                clf2 = GaussianNB()

            clf1.fit(self.p_vars, self.labels)
            clf2.fit(self.p_vars, self.labels)

            ## plot the dataset first
            ax1 = plt.subplot(121)
            ax2 = plt.subplot(122)

            ax1.set_xlim(xx.min(), xx.max())
            ax1.set_ylim(yy.min(), yy.max())
            ax1.set_xticks(())
            ax1.set_yticks(())

            ax2.set_xlim(xx.min(), xx.max())
            ax2.set_ylim(yy.min(), yy.max())
            ax2.set_xticks(())
            ax2.set_yticks(())

            # plot the decision boundary
            if hasattr(clf1, 'decision_function'):
                Z1 = clf1.decision_function(np.c_[xx.ravel(), yy.ravel()])[:, class2check-1]
                Z2 = clf2.decision_function(np.c_[xx.ravel(), yy.ravel()])[:, class2check-1]
            else:
                Z1 = clf1.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, class2check-1]
                Z2 = clf2.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, class2check-1]

            # Put the result into a color plot
            Z1 = Z1.reshape(xx.shape)
            ax1.contourf(xx, yy, Z1, cmap=cm, alpha=.8)

            Z2 = Z2.reshape(xx.shape)
            ax2.contourf(xx, yy, Z2, cmap=cm, alpha=.8)

        elif compare == 2:

            clf1 = RandomForestClassifier(max_depth=classifier_info['max_depth'],
                                          n_estimators=classifier_info['trees'],
                                          max_features=classifier_info['rand_vars'],
                                          min_samples_split=classifier_info['min_samps'],
                                          n_jobs=-1)

            clf2 = ExtraTreesClassifier(max_depth=classifier_info['max_depth'],
                                        n_estimators=classifier_info['trees'],
                                        max_features=classifier_info['rand_vars'],
                                        min_samples_split=classifier_info['min_samps'],
                                        n_jobs=-1)

            clf3 = SVC(gamma=classifier_info['gamma'], C=classifier_info['C'])

            clf4 = GaussianNB()

            clf1.fit(self.p_vars, self.labels)
            clf2.fit(self.p_vars, self.labels)

            if locate_outliers:

                weights = np.ones(len(self.labels))
                for c, curr_c_idx in self.class_outliers.iteritems():

                    class_idx = np.where(self.labels == c)

                    weights[class_idx][curr_c_idx] *= 10

            else:
                weights = None

            clf3.fit(self.p_vars, self.labels, sample_weight=weights)

            clf4.fit(self.p_vars, self.labels)

            ## plot the dataset first
            ax1 = plt.subplot(221)
            ax2 = plt.subplot(222)
            ax3 = plt.subplot(223)
            ax4 = plt.subplot(224)

            ax1.set_xlim(xx.min(), xx.max())
            ax1.set_ylim(yy.min(), yy.max())
            ax1.set_xticks(())
            ax1.set_yticks(())

            ax2.set_xlim(xx.min(), xx.max())
            ax2.set_ylim(yy.min(), yy.max())
            ax2.set_xticks(())
            ax2.set_yticks(())

            ax3.set_xlim(xx.min(), xx.max())
            ax3.set_ylim(yy.min(), yy.max())
            ax3.set_xticks(())
            ax3.set_yticks(())

            ax4.set_xlim(xx.min(), xx.max())
            ax4.set_ylim(yy.min(), yy.max())
            ax4.set_xticks(())
            ax4.set_yticks(())

            ## plot the decision boundary
            if hasattr(clf1, 'decision_function'):
                Z1 = clf1.decision_function(np.c_[xx.ravel(), yy.ravel()])[:, class2check-1]
            else:
                Z1 = clf1.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, class2check-1]

            if hasattr(clf2, 'decision_function'):
                Z2 = clf2.decision_function(np.c_[xx.ravel(), yy.ravel()])[:, class2check-1]
            else:
                Z2 = clf2.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, class2check-1]

            if hasattr(clf3, 'decision_function'):
                Z3 = clf3.decision_function(np.c_[xx.ravel(), yy.ravel()])[:, class2check-1]
            else:
                Z3 = clf3.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, class2check-1]

            if hasattr(clf4, 'decision_function'):
                Z4 = clf4.decision_function(np.c_[xx.ravel(), yy.ravel()])[:, class2check-1]
            else:
                Z4 = clf4.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, class2check-1]

            # Put the result into a color plot
            Z1 = Z1.reshape(xx.shape)
            ax1.contourf(xx, yy, Z1, cmap=cm, alpha=.8)

            Z2 = Z2.reshape(xx.shape)
            ax2.contourf(xx, yy, Z2, cmap=cm, alpha=.8)

            Z3 = Z3.reshape(xx.shape)
            ax3.contourf(xx, yy, Z3, cmap=cm, alpha=.8)

            Z4 = Z4.reshape(xx.shape)
            ax4.contourf(xx, yy, Z4, cmap=cm, alpha=.8)

        leg_items = []
        leg_names = []

        for n_class in xrange(0, self.n_classes):

            cl_idx = np.where(self.labels == self.classes[n_class])

            # plot the training points
            curr_pl = ax1.scatter(self.p_vars[:, 0][cl_idx], self.p_vars[:, 1][cl_idx],
                                 c=colors[n_class], alpha=.7)#, cmap=cm_bright)

            ax2.scatter(self.p_vars[:, 0][cl_idx], self.p_vars[:, 1][cl_idx],
                                 c=colors[n_class], alpha=.7)#, cmap=cm_bright)

            if compare == 2:

                ax3.scatter(self.p_vars[:, 0][cl_idx], self.p_vars[:, 1][cl_idx],
                            c=colors[n_class], alpha=.7)#, cmap=cm_bright)

                ax4.scatter(self.p_vars[:, 0][cl_idx], self.p_vars[:, 1][cl_idx],
                            c=colors[n_class], alpha=.7)#, cmap=cm_bright)

            leg_items.append(curr_pl)
            leg_names.append(str(self.classes[n_class]))

        if compare == 1:

            if 'RF' in classifier_info['classifier']:

                ax1.set_xlabel('RF, Max. depth: %d' % classifier_info['max_depth'])
                ax2.set_xlabel('Extreme RF, Max. depth: %d' % classifier_info['max_depth'])

            elif classifier_info['classifier'] == 'SVM':

                ax1.set_xlabel('C: %d' % classifier_info['C'])
                ax2.set_xlabel('C: %d' % C2)

        else:

            ax1.set_xlabel('Random Forest')
            ax2.set_xlabel('Extremely Random Forest')
            ax3.set_xlabel('SVM')
            ax4.set_xlabel('Naives Bayes')

        plt.show()

        plt.close()

    def vis_series(self, class_list=[], class_names={}, smooth=True, window_size=3, xaxis_labs=[],
                   show_intervals=True, show_raw=False):

        """
        Visualize classes in a time series

        Args:
            class_list (Optional[list]): A list of classes to compare. Default is [], or all classes.
            class_names (Optional[dict]): A dictionary of class names. Default is {}, so the labels
                are the class values.
            smooth (Optional[bool]): Whether to smooth the time series. Default is True.
            window_size (Optional[int]): The window size to use for smoothing. Default is 3.
            xaxis_labs (Optional[str list]): A list of labels for the x-axis. Default is [].
            show_intervals (Optional[bool]): Whether to fill axis intervals. Default is True.
            show_raw (Optional[bool]): Whether to plot the raw data points. Default is False.

        Examples:
            >>> from mappy.classifiers import classification
            >>> cl = classification()
            >>>
            >>> cl.split_samples('/samples.txt', classes2remove=[1, 4],
            >>>                  class_subs={2:.1, 5:.01, 8:.1, 9:.9})
            >>> cl.vis_series(1, class_list=[3, 5, 8],
            >>>               class_names={3: 'forest', 5: 'agriculture', 8: 'water'})
        """

        fig = plt.figure(facecolor='white')

        ax = fig.add_subplot(111, axisbg='white')

        mpl.rcParams['font.size'] = 12.
        mpl.rcParams['font.family'] = 'Verdana'
        mpl.rcParams['axes.labelsize'] = 8.
        mpl.rcParams['xtick.labelsize'] = 8.
        mpl.rcParams['ytick.labelsize'] = 8.

        # new x values
        xn_ax = np.linspace(0, self.n_feas-1, (self.n_feas-1)*10)

        colors = ['black', 'cyan', 'yellow', 'red', 'orange', 'green', 'purple', 'magenta', '#5F4C0B', '#21610B',
                  '#210B61']

        if not class_list:
            class_list = self.classes

        ## setup the class names
        ## we might not have all the classes in the current samples
        class_names_list = [class_names[cl] for cl in class_list]

        # self.df = pd.DataFrame(np.zeros((0, self.n_feas)))

        leg_items = []
        for n_class, class_name in enumerate(class_names_list):

            # get the current class array indices
            cl_idx = np.where(self.labels == self.classes[n_class])

            idx2del = []
            for ri, r in enumerate(self.p_vars[cl_idx]):
                if r.max() == 0:
                    idx2del.append(ri)

            if idx2del:
                vis_p_vars = np.delete(self.p_vars[cl_idx], idx2del, axis=0)
            else:
                vis_p_vars = np.copy(self.p_vars[cl_idx])

            # df_sm = self.pd_interpolate(vis_p_vars.astype(np.float32), window_size)
            # the_nans, x = self.nan_helper(vis_p_vars)
            # df_sm = self.lin_interp2(x, y, the_nans)

            # idx = np.arange(vis_p_vars.shape[1])
            # df_sm = np.apply_along_axis(self.lin_interp, 1, vis_p_vars.astype(np.float32), idx)

            # vis_p_vars[vis_p_vars == 0] = np.nan
            df_sm = _lin_interp.lin_interp(vis_p_vars.astype(np.float32))

            df_sm = _rolling_stats.rolling_stats(df_sm, stat='median', window_size=window_size)

            df_sm_std = df_sm.std(axis=1)
            df_sm_u = df_sm.mean(axis=1)
            df_sm_up = df_sm_u + (1.5 * df_sm_std)
            df_sm_um = df_sm_u - (1.5 * df_sm_std)

            for idx_check in xrange(0, 2):

                idx = np.where((df_sm[:, idx_check] > df_sm_up) | (df_sm[:, idx_check] < df_sm_um))

                if len(idx[0]) > 0:

                    df_sm[:, idx_check][idx] = np.median(df_sm[:, :3][idx], axis=1)

            for idx_check in xrange(self.n_feas-1, self.n_feas-3, -1):

                idx = np.where((df_sm[:, idx_check] > df_sm_up) | (df_sm[:, idx_check] < df_sm_um))

                if len(idx[0]) > 0:

                    df_sm[:, idx_check][idx] = np.median(df_sm[:, :3][idx], axis=1)

            df_sm_u = np.nanmean(df_sm, axis=0)
            df_sm_std = np.nanstd(df_sm, axis=0)

            if smooth:

                df_sm_u_int = interp1d(range(self.n_feas), df_sm_u, kind='cubic')
                df_sm_std_int = interp1d(range(self.n_feas), df_sm_std, kind='cubic')

            # add the class index
            # df_sm.index = [class_name]*df_sm.shape[0]

            # self.df = self.df.append(df_sm)
            marker_size = .1
            line_width = 1.5
            alpha = .5

            for r in xrange(0, df_sm.shape[0]):

                if show_raw:

                    # raw data
                    ax.scatter(range(len(vis_p_vars[r])), vis_p_vars[r], marker='o', edgecolor='none', s=40,
                               facecolor=colors[n_class], c=colors[n_class])

                # new y values
                if smooth:
                    yn_cor = interp1d(range(self.n_feas), df_sm[r, :], kind='cubic')

                ## Savitsky Golay filtered
                # ax.plot(range(len(df_sm_sav[r])), df_sm_sav[r], marker='o', markeredgecolor='none', markersize=5,
                #          markerfacecolor=colors[-1], c=colors[-1], alpha=.7, lw=2)

                    ## Cubic interpolation
                    ax.plot(xn_ax, yn_cor(xn_ax), marker='o', markeredgecolor='none', markersize=marker_size,
                             markerfacecolor=colors[n_class], linestyle='-', c=colors[n_class], alpha=alpha,
                             lw=line_width)

                else:

                    ## raw data
                    ax.plot(range(len(df_sm[r])), df_sm[r], marker='o', markeredgecolor='none',
                            markersize=marker_size, markerfacecolor=colors[-2], linestyle='-', c=colors[n_class],
                            alpha=alpha, lw=line_width)

            if smooth:

                yn_cor = interp1d(range(self.n_feas), df_sm[-1, :], kind='cubic')

                dummy = ax.scatter(xn_ax, yn_cor(xn_ax), marker='o', edgecolor='none', s=marker_size,
                                 facecolor=colors[n_class], c=colors[n_class], alpha=alpha, lw=line_width,
                                 label=class_name)

                if show_intervals:

                    ax.fill_between(xn_ax, df_sm_u_int(xn_ax)-(2*df_sm_std_int(xn_ax)),
                                     df_sm_u_int(xn_ax)+(2*df_sm_std_int(xn_ax)), color=colors[n_class], alpha=.1)

            else:

                dummy = ax.scatter(range(len(df_sm[r])), df_sm[r], marker='o', edgecolor='none', s=marker_size,
                                 facecolor=colors[n_class], c=colors[n_class], alpha=alpha, lw=line_width,
                                 label=class_name)

                if show_intervals:

                    ax.fill_between(range(len(df_sm_u)), df_sm_u-(2*df_sm_std), df_sm_u+(2*df_sm_std),
                                    color=colors[n_class], alpha=.1)

            leg_items.append(dummy)

            plt.ylabel('Value')
            # plt.ylabel('Feature: %d' % fea_2)

            # ax.set_xlabel('Feature: %d')
            # ax.set_ylabel('Feature: %d' % fea)

        limits = False

        leg = plt.legend(tuple(leg_items), tuple(class_names_list), scatterpoints=1, loc='lower left',
                         markerscale=marker_size*200)

        leg.get_frame().set_edgecolor('#D8D8D8')
        leg.get_frame().set_linewidth(.5)

        if xaxis_labs:

            ax.set_xticks(range(self.n_feas))
            ax.set_xticklabels(xaxis_labs)

        plt.xlim(0, self.n_feas)
        plt.ylim(50, 250)

        plt.setp(plt.xticks()[1], rotation=30)

        plt.tight_layout()

        plt.show()

        plt.close(fig)

    # def lin_interp(self, in_block, indices):
    #
    #     in_block[in_block == 0] = np.nan
    #
    #     not_nan = np.logical_not(np.isnan(in_block))
    #
    #     return np.interp(indices, indices[not_nan], in_block[not_nan]).astype(np.float32)

    # def pd_interpolate(self, in_block, window_size):
    #
    #     in_block[in_block == 0] = np.nan
    #
    #     df = pd.DataFrame(in_block)
    #
        # linear interpolation along the x axis (layers)
        # df = df.apply(pd.Series.interpolate, axis=1).values.astype(np.float32)
        # df = df.apply(pd.Series.interpolate, axis=1)

        # rolling mean along the x axis and converted to ndarray
        # df = pd.rolling_median(df, window=window_size, axis=1).values
        # df = mp.rolling_stats(df, stat='median', window_size=window_size)

        # # fill the first two columns
        # if window_size == 3:
        #
        #     # df[:, 0] = np.median(df[:, :window_size-1], axis=1)
        #     # df[:, 1] = np.median(df[:, :window_size], axis=1)
        #     # df[:, -1] = np.median(df[:, -window_size:], axis=1)
        #     # df[:, -2] = np.median(df[:, -window_size-1:], axis=1)
        #
        #     df[:, 0] = np.median(df[:, :window_size-1+(window_size/2)], axis=1)
        #     df[:, 1] = np.median(df[:, :window_size+(window_size/2)], axis=1)
        #     df[:, -1] = np.median(df[:, -window_size-(window_size/2):], axis=1)
        #     df[:, -2] = np.median(df[:, -window_size-1-(window_size/2):], axis=1)
        #
        # elif window_size == 5:
        #
        #     df[:, 0] = np.median(df[:, :window_size-3+(window_size/2)], axis=1)
        #     df[:, 1] = np.median(df[:, :window_size-2+(window_size/2)], axis=1)
        #     df[:, 2] = np.median(df[:, :window_size-1+(window_size/2)], axis=1)
        #     df[:, 3] = np.median(df[:, :window_size+(window_size/2)], axis=1)
        #
        #     df[:, -1] = np.median(df[:, -window_size-(window_size/2):], axis=1)
        #     df[:, -2] = np.median(df[:, -window_size-1-(window_size/2):], axis=1)
        #     df[:, -3] = np.median(df[:, -window_size-2-(window_size/2):], axis=1)
        #     df[:, -4] = np.median(df[:, -window_size-3-(window_size/2):], axis=1)

        # df[np.isnan(df)] = 0

        # return np.apply_along_axis(savgol_filter, 1, df, 5, 3)
        # return df

    def vis_k_means(self, image, bands2vis=[1, 2, 3], clusters=3):

        """
        Use k-means clustering to visualize data in image

        Args:
            image (str): The image to visualize.
            bands2vis (Optional[int list]): A list of bands to visualize. Default is [1, 2, 3].
            clusters (Optional[int]): The number of clusters. Default is 3.
        """

        # open the image
        i_info = raster_tools.rinfo(image)

        band_arrays = [zoom(i_info.mparray(bands2open=[bd], d_type='float32'), .5) for bd in bands2vis]

        rws, cls = band_arrays[0].shape[0], band_arrays[1].shape[1]

        ## reshape the arrays
        multi_d = np.empty((len(bands2vis), rws, cls)).astype(np.float32)

        ctr = 0
        for n in xrange(len(bands2vis)):

            multi_d[ctr] = band_arrays[n]

            ctr += 1

        multi_d = multi_d.reshape((len(bands2vis), rws*cls)).astype(np.float32).T

        # run k means clustering
        clt = KMeans(max_iter=300, n_jobs=-1, n_clusters=clusters)
        clt.fit(multi_d)

        hst = self._centroid_histogram(clt)

        bar = self._plot_colors(hst, clt.cluster_centers_)

        plt.figure()
        plt.axis('off')
        plt.imshow(bar)
        plt.show()

        plt.close()

    def _centroid_histogram(self, clt):

        # grab the number of different clusters and create a histogram
        # based on the number of pixels assigned to each cluster
        n_labels = np.arange(0, len(np.unique(clt.labels_) + 1))
        hist, _ = np.histogram(clt.labels_, bins=n_labels)

        # normalize the histogram, such that it sums to one
        hist = hist.astype('float')
        hist /= hist.sum()

        return hist

    def _plot_colors(self, hist, centroids):

        # initialize the bar chart representing the relative frequency of each of the colors
        bar = np.zeros((50, 300, 3), dtype='uint8')
        start_x = 0

        # iterate over the percentage of each cluster and the color of each cluster
        for (percent, color) in zip(hist, centroids):

            # plot the relative percentage of each cluster
            end_x = start_x + (percent * 300)

            cv2.rectangle(bar, (int(start_x), 0), (int(end_x), 50), color.astype('uint8').tolist(), -1)

            start_x = end_x

        return bar


class Preprocessing(object):

    def __init__(self):

        self.time_stamp = time.asctime(time.localtime(time.time()))

    def compare_features(self, f1, f2, method='mahalanobis'):

        """
        Compares features (within samples) using distance-based methods

        Args:
            f1 (int): The first feature position to compare.
            f2 (int): The second feature position to compare.
            method (Optional[str]): The distance method to use. Default is 'mahalanobis'.
        """

        dist_methods = {'mahalanobis': sci_dist.mahalanobis,
                        'correlation': sci_dist.correlation,
                        'euclidean': sci_dist.euclidean}

        if method == 'mahalanobis':

            return dist_methods[method](self.p_vars[f1-1], self.p_vars[f2-1], np.linalg.cov(self.p_vars[f1-1],
                                                                                            self.p_vars[f2-1],
                                                                                            rowvar=0))

        else:
            return dist_methods[method](self.p_vars[f1-1], self.p_vars[f2-1])

    def compare_samples(self, base_samples, compare_samples, output, id_label='Id', y_label='Y',
                        response_label='response', dist_threshold=500, pct_threshold=.75,
                        replaced_weight=2, semi_supervised=False, spatial_weights=False, add2base=False):

        """
        Compares features (between samples) and removes samples

        Args:
            base_samples (str): The baseline samples.
            compare_samples (str): The samples to compare to the baseline, ``base_samples``.
            output (str): The output (potentially reduced) samples.
            id_label (Optional[str]): The id label. Default is 'Id'.
            y_label (Optional[str]): The Y label. Default is 'Y'.
            response_label (Optional[str]): The response (or class outcome) label. Default is 'response'.
            dist_threshold (Optional[int]): The euclidean distance threshold, where samples with distance
                values above `dist_threshold` are removed. Default is 300.
            pct_threshold (Optional[float]): The proportional number of image variables required
                above 'pct_threshold`. Default is 0.75.
            replaced_weight (Optional[int or float]): The weight value to add to new samples. Default is 2.
            semi_supervised (Optional[bool]): Whether to apply semi-supervised learning to the
                unselected samples. Default is False.
            spatial_weights (Optional[bool]): Whether to apply inverse spatial weights. Default is False.
            add2base (Optional[bool]): Whether to add the samples to the baseline set. Default is False.

        Example:
            >>> compare_samples('/2000.txt', '/2014.txt', '/2014_mod.csv')

        Explained:
            1) Get the euclidean distance between image variables.

        Returns:
            None, writes to ``output``.
        """

        weights = None

        df_base = pd.read_csv(base_samples, sep=',')
        df_compare = pd.read_csv(compare_samples, sep=',')

        # Load sample weights.
        if os.path.isfile(base_samples.replace('.txt', '_w.txt')):

            weights = pickle.load(file(base_samples.replace('.txt', '_w.txt'), 'rb'))

            if isinstance(weights, list):
                weights = np.array(weights, dtype='float32')

        # Reset the ids in case of stacked samples.
        df_base[id_label] = range(1, df_base.shape[0] + 1)
        df_compare[id_label] = range(1, df_compare.shape[0] + 1)

        all_headers = df_base.columns.values.tolist()

        leaders = all_headers[all_headers.index(id_label):all_headers.index(y_label) + 1]

        headers = all_headers[all_headers.index(y_label) + 1:all_headers.index(response_label)]

        added_headers = ('_y,'.format(headers[0]).join(headers) + '_y').split(',')

        df_base.rename(columns=dict(zip(headers, added_headers)), inplace=True)

        if isinstance(weights, np.ndarray):

            df_base['WEIGHT'] = weights

            df = pd.merge(df_compare, df_base[[id_label] + added_headers + ['WEIGHT']], on=id_label, how='inner')

        else:
            # Merge column-wise, with 'compare samples' first, then 'base samples'
            df = pd.merge(df_compare, df_base[[id_label] + added_headers], on=id_label, how='inner')

        if spatial_weights:

            self.index_samples(df.query('WEIGHT == 1'))

            dist_weights = self.weight_samples(df.query('WEIGHT == 1'), df.query('WEIGHT != 1'))

            # Calculate the inverse distance
            df.loc[df['WEIGHT'] != 1, 'WEIGHT'] = 1. - (dist_weights['SP_DIST'] / dist_weights['SP_DIST'].max())

        def e_dist(d, h1, h2):
            return (d[h1] - d[h2]) ** 2.

        df['COUNT'] = 0

        # Iterate over each image variable and
        #   calculate the euclidean distance.
        for compare_header, base_header in zip(headers, added_headers):

            df['DIST'] = e_dist(df, compare_header, base_header)

            df.loc[df['DIST'] < dist_threshold, 'COUNT'] += 1

        if semi_supervised:

            # Add unlabeled values to samples with high distance values.
            df.loc[df['COUNT'] < int(pct_threshold * len(headers)), 'response'] = -1

            # Semi-supervised learning.
            label_spread = label_propagation.LabelSpreading(kernel='rbf')
            label_spread.fit(df[headers], df['response'])

            # Replace the high distance samples' unlabeled responses.
            df.loc[df['COUNT'] < int(pct_threshold * len(headers)), 'response'] = label_spread.transduction_

        else:
            df = df.query('COUNT >= {:d}'.format(int(pct_threshold * len(headers))))

        # Copy the 'base samples' weights and add new weights
        if isinstance(weights, np.ndarray):

            weights_out = df['WEIGHT'].values
            weights_out = np.where(weights_out == 1, replaced_weight, weights_out)

        # Get the 'compare sample' image variables.
        df = df[leaders + headers + ['response']]

        if add2base:

            # Add the original column names back.
            df_base = df_base.rename(columns=dict(zip(added_headers, headers)))[leaders + headers + ['response']]

            # Concatenate the base samples with the new samples.
            df = pd.concat([df_base, df], axis=0)

            if isinstance(weights, np.ndarray):

                # Concatenate the base weights with the new weights.
                weights_out = np.concatenate([weights, weights_out], axis=0)

                assert df.shape[0] == len(weights_out)

        print 'Base samples: {:,d}'.format(df_base.shape[0])
        print 'New samples: {:,d}'.format(df.shape[0])

        if os.path.isfile(output):
            os.remove(output)

        df.to_csv(output, sep=',', index=False)

        if os.path.isfile(base_samples.replace('.txt', '_w.txt')):

            if os.path.isfile(compare_samples.replace('.txt', '_w.txt')):
                os.remove(compare_samples.replace('.txt', '_w.txt'))

            pickle.dump(weights_out, open(compare_samples.replace('.txt', '_w.txt'), 'wb'),
                        protocol=pickle.HIGHEST_PROTOCOL)

    def index_samples(self, base_samples, id_label='Id', x_label='X', y_label='Y'):

        """
        Args:
            base_samples (DataFrame): It should only contain 'good' points.
        """

        base_samples[id_label] = base_samples[id_label].astype(int)

        self.rtree_index = rtree.index.Index(interleaved=False)

        # Iterate over each sample.
        for di, df_row in base_samples.iterrows():

            x = float(df_row[x_label])
            y = float(df_row[y_label])

            self.rtree_index.insert(int(df_row[id_label]), (x, y))

    def weight_samples(self, base_samples, compare_samples, id_label='Id', x_label='X', y_label='Y', n_nearest=1):

        """
        Args:
            compare_samples (DataFrame): It should only contain points to compare against 'good' points.
        """

        from scipy.spatial.distance import euclidean

        base_samples[id_label] = base_samples[id_label].astype(int)
        compare_samples[id_label] = compare_samples[id_label].astype(int)

        sp_dists = []

        # Iterate over each sample.
        for di, df_row in compare_samples.iterrows():

            x1 = float(df_row[x_label])
            y1 = float(df_row[y_label])

            id = list(self.rtree_index.nearest((x1, y1), n_nearest))

            x2 = float(base_samples.loc[base_samples[id_label] == id[0], x_label])
            y2 = float(base_samples.loc[base_samples[id_label] == id[0], y_label])

            # Calculate the distance.
            sp_dists.append(euclidean([x1, y1], [x2, y2]))

        compare_samples['SP_DIST'] = sp_dists

        return compare_samples

    def remove_outliers(self, outliers_fraction=.25, locate_only=False):

        """
        Removes outliers from each class by fitting an Elliptic Envelope

        Args:
            outliers_fraction (Optional[float]): The proportion of outliers. Default is .25.
            locate_only (Optional[bool]): Whether to locate and do not remove outliers. Default is False.

        Examples:
            >>> from mappy.classifiers import classification
            >>> cl = classification()
            >>>
            >>> # Get predictive variables and class labels data.
            >>>
            >>> # The data should be scaled, as the the Elliptic Envelope
            >>> #   assumes a Gaussian distribution
            >>> cl.split_samples('/samples.txt', perc_samp=1., scale_data=True)
            >>>
            >>> # Search for outliers in the sample data
            >>> #   the new p_vars are stored in the <cl> instance.
            >>> cl.remove_outliers()
            >>>
            >>> # Check the outlier locations
            >>> print cl.class_outliers
        """

        if not self.scaled:
            raise NameError('The data should be scaled prior to outlier removal.')

        self.outliers_fraction = outliers_fraction

        # xx, yy = np.meshgrid(np.linspace(-7, 7, self.n_samps*self.n_feas),
        #                      np.linspace(-7, 7, self.n_samps*self.n_feas))

        new_p_vars = np.empty((0, self.n_feas), dtype='float32')
        new_labels = np.array([], dtype='int16')

        self.class_outliers = {}

        for check_class in self.classes:

            print 'Class {:d} ...'.format(check_class)

            try:
                new_p_vars, new_labels = self._remove_outliers(check_class, new_p_vars, new_labels)
            except RuntimeError:
                raise RuntimeError('Could not fit the data for class {:d}'.format(check_class))

        if not locate_only:

            self.p_vars = new_p_vars
            self.labels = new_labels

            self._update_class_counts()

    @retry(wait_random_min=500, wait_random_max=1000, stop_max_attempt_number=5)
    def _remove_outliers(self, check_class, new_p_vars, new_labels):

        # row indices for current class
        class_idx = np.where(self.labels == check_class)

        temp_p_vars = self.p_vars[class_idx]
        temp_labels = self.labels[class_idx]

        # outlier detection
        outlier_clf = EllipticEnvelope(contamination=.1)

        try:
            outlier_clf.fit(temp_p_vars)
        except:

            new_p_vars = np.vstack((new_p_vars, self.p_vars_original[class_idx]))
            new_labels = np.concatenate((new_labels, temp_labels))

            return new_p_vars, new_labels

        y_pred = outlier_clf.decision_function(temp_p_vars).ravel()

        threshold = stats.scoreatpercentile(y_pred, 100. * self.outliers_fraction)

        inlier_idx = np.where(y_pred >= threshold)
        outlier_idx = np.where(y_pred < threshold)

        self.class_outliers[check_class] = outlier_idx

        n_outliers = len(y_pred) - len(inlier_idx[0])

        print '  {:d} outliers in class {:d}'.format(n_outliers, check_class)

        # temp_p_vars = temp_p_vars[inlier_idx]

        temp_labels = temp_labels[inlier_idx]

        # update the features
        new_p_vars = np.vstack((new_p_vars, self.p_vars_original[class_idx][inlier_idx]))

        # update the labels
        new_labels = np.concatenate((new_labels, temp_labels))

        return new_p_vars, new_labels

    def semi_supervised(self, classifier_info={'classifier': 'RF'}, kernel='knn'):

        """
        Predict class values of unlabeled samples

        Args:
            classifier_info (Optional[dict]): The model parameters. Default is {'classifier': 'RF'}.
            kernel (str): The kernel to use (rbf or knn). Default is 'knn'.

        Examples:
            >>> # create the classifier object
            >>> from mappy.classifiers import classification
            >>> cl = classification()
            >>>
            >>> # get predictive variables and class labels data, sampling 100%
            >>> # the unknown samples should have a class value of -1
            >>> cl.split_samples('/samples.txt', perc_samp=1.)
            >>>
            >>> # run semi-supervised learning to predict unknowns
            >>> # the instances <labels>, <classes>, <n_classes>,
            >>> # and <class_counts> are updated
            >>> cl.semi_supervised()
        """

        self.classifier_info = classifier_info

        # label_spread = label_propagation.LabelSpreading(kernel=kernel, n_neighbors=10, alpha=20, gamma=100,
        #                                                 max_iter=100, tol=.001)
        # label_spread.fit(self.p_vars, self.labels)
        #
        # self.labels = label_spread.transduction_

        # the model parameters
        self._default_parameters()

        if classifier_info['classifier'] == 'RF':

            label_spread = RandomForestClassifier(max_depth=classifier_info['max_depth'],
                                                  n_estimators=classifier_info['trees'],
                                                  max_features=classifier_info['rand_vars'],
                                                  min_samples_split=classifier_info['min_samps'],
                                                  n_jobs=-1)

        elif classifier_info['classifier'] == 'EX_RF':

            label_spread = ExtraTreesClassifier(max_depth=classifier_info['max_depth'],
                                                n_estimators=classifier_info['trees'],
                                                max_features=classifier_info['rand_vars'],
                                                min_samples_split=classifier_info['min_samps'],
                                                n_jobs=-1)

        labeled_vars_idx = np.where(self.labels != -1)
        labeled_vars = self.p_vars[labeled_vars_idx]
        labels = self.labels[labeled_vars_idx]

        label_spread.fit(labeled_vars, labels)

        # keep the good labels
        unknown_labels_idx = np.where(self.labels == -1)

        # predict the unlabeled
        temp_labels = label_spread.predict(self.p_vars)

        # save the predictions of the unknowns
        self.labels[unknown_labels_idx] = temp_labels[unknown_labels_idx]

        # update the individual class counts
        self.classes = list(np.delete(self.classes, 0))
        self.class_counts = {}
        for indv_class in self.classes:

            self.class_counts[indv_class] = len(np.where(self.labels == indv_class)[0])

        self.n_classes = len(self.classes)


class classification(Samples, EndMembers, Visualization, Preprocessing):

    """
    A class for image sampling and classification

    Example:
        >>> from mappy.classifiers import classification
        >>>
        >>> # Create the classification object.
        >>> cl = classification()
        >>>
        >>> # Open land cover samples and split
        >>> #   into train and test datasets.
        >>> cl.split_samples('/samples.txt')
        >>>
        >>> # Train a Random Forest classification model.
        >>> # *Note that the model is NOT saved to file in
        >>> #   this example. However, the model IS passed
        >>> #   to the ``cl`` instance. To use the same model
        >>> #   after Python cleanup, save the model to file
        >>> #   with the ``output_model`` keyword. See the
        >>> #   ``construct_model`` function more details.
        >>> cl.construct_model(classifier_info={'classifier': 'RF',
        >>>                                     'trees': 1000,
        >>>                                     'max_depth': 25})
        >>>
        >>> # Apply the model to predict an entire image.
        >>> cl.predict('/image_variables.tif', '/image_labels.tif')
    """

    def __init__(self):

        self.time_stamp = time.asctime(time.localtime(time.time()))

    def model_options(self):

        return """\

        Supported models

        ===========================
        Parameter name -- Long name
              {Classifier defaults}
              *Scikit-learn parameter names and defaults
        ===========================

        AB_DT -- AdaBoost with CART (classification problems)
              *Scikit-learn
        AB_EX_DT-- AdaBoost with extremely random trees (classification problems)
              *Scikit-learn
        AB_RF-- AdaBoost with Random Forest (classification problems)
              *Scikit-learn
        AB_EX_RF-- AdaBoost with Extremely Random Forest (classification problems)
              *Scikit-learn
        AB_DTR-- AdaBoost with CART (regression problems)
              *Scikit-learn
        AB_EX_DTR-- AdaBoost with extremely random trees (regression problems)
              *Scikit-learn
        Bag   -- Bagging (classification problems)
              *Scikit-learn
        BagR  -- Bagging (regression problems)
              *Scikit-learn
        Bag_EX_DT-- Bagging with extra trees (classification problems)
              *Scikit-learn
        Bayes -- Naives Bayes (classification problems)
        DT    -- Decision Trees based on CART algorithm (classification problems)
              *Scikit-learn
        DTR   -- Decision Trees Regression based on CART algorithm (regression problems)
              *Scikit-learn
        EX_DT -- Extra Decision Trees based on CART algorithm (classification problems)
              *Scikit-learn
        EX_DTR-- Extra Decision Trees Regression based on CART algorithm (regression problems)
              *Scikit-learn
        GB    -- Gradient Boosted Trees (classification problems)
              *Scikit-learn
        GBR   -- Gradient Boosted Trees (regression problems)
              *Scikit-learn
        C5    -- C5 decision trees (classification problems)
              {classifier:C5,trials:10,CF:.25,min_cases:2,winnow:False,no_prune:False,fuzzy:False}
        Cubist-- Cubist regression trees (regression problems)
              {classifier:Cubist,committees:5,unbiased:False,rules:100,extrapolation:10}
        EX_RF -- Extremely Random Forests (classification problems)
              *Scikit-learn
        CVEX_RF -- Extremely Random Forests in OpenCV (classification problems)
              *NOT CURRENTLY SUPPORTED IN OPENCV 3.0*
              {classifier:CVEX_RF,trees:1000,min_samps:0,rand_vars:sqrt(feas),max_depth:25}
        EX_RFR-- Extremely Random Forests (regression problems)
              *Scikit-learn
        Logistic-- Logistic Regression (classification problems)
              *Scikit-learn
        NN    -- K Nearest Neighbor (classification problems)
              *Scikit-learn
        RF    -- Random Forests (classification problems)
              *Scikit-learn
        CVRF  -- Random Forests in OpenCV (classification problems)
              {classifier:CVRF,trees:1000,min_samps:0,rand_vars:0,max_depth:25,weight_classes:None,truncate:False}
        RFR   -- Random Forests (regression problems)
              *Scikit-learn
        CVMLP -- Feed-forward, artificial neural network, multi-layer perceptrons in OpenCV (classification problems)
              {classifier:CVMLP}
        SVM   -- Support Vector Machine (classification problems)
              {classifier:SVM,C:1,g:1.}
        SVMR  -- Support Vector Machine (regression problems)
              {classifier:SVMR,C:1,g:1.}
        CVSVM -- Support Vector Machine in OpenCV (classification problems)
              {classifier:CVSVM,C:1,g:1.}
        CVSVMA-- Support Vector Machine, auto-tuned in OpenCV (classification problems)
              {classifier:CVSVMA}
        CVSVMR-- Support Vector Machine in OpenCV (regression problems)
              {classifier:CVSVMR,C:1,g:1.}
        CVSVMRA-- Support Vector Machine, auto-tuned in OpenCV (regression problems)
              {classifier:CVSVMRA}
        QDA   -- Quadratic Discriminant Analysis (classification problems)
              *Scikit-learn

        """

    def construct_model(self, input_model=None, output_model=None, classifier_info=None,
                        class_weight=None, var_imp=True, rank_method=None, top_feas=.5,
                        get_probs=False, input_image=None, in_shapefile=None, out_stats=None,
                        stats_from_image=False, calibrate_proba=False, be_quiet=False, n_jobs=-1):

        """
        Loads, trains, and saves a predictive model.

        Args:
            input_model (Optional[str]): The input model name with .xml extension.
            output_model (Optional[str]): The output model name with .xml extension.
            classifier_info (Optional[dict]): A dictionary of classifier information. Default is {'classifier': 'RF'}.
            class_weight (Optional[bool]): How to weight classes for priors. Default is None. Choices are
                [None, 'percent', 'inverse'].
                *Example when class_weight=True:
                    IF
                        labels = [1, 1, 1, 2, 1, 2, 3, 2, 3]
                    THEN
                        class_weight = {1: .22, 2: .33, 3: .44}
            var_imp (Optional[bool]): Whether to return feature importance. Default is True.
            rank_method (Optional[str]): The rank method to use. 'chi2' or 'RF'. Default is None.
            top_feas (Optional[int or float]): The number or percentage of top ranked features to return.
                Default is .5, or 50%.
            get_probs (Optional[bool]): Whether to return class probabilities. Default is False.
            input_image (Optional[str]): An input image for Orfeo models. Default is None.
            in_shapefile (Optional[str]): An input shapefile for Orfeo models. Default is None.
            output_stats (Optional[str]): A statistics file for Orfeo models. Default is None.
            stats_from_image (Optional[bool]): Whether to collect statistics from the image for Orfeo models. Default
                is False.
            calibrate_proba (Optional[bool]): Whether to calibrate posterior probabilities with a sigmoid
                calibration. Default is False.

        Examples:
            >>> # create the classifier object
            >>> from mappy.classifiers import classification
            >>> cl = classification()
            >>>
            >>> # get predictive variables and class labels data
            >>> cl.split_samples('/samples.txt')
            >>> # or
            >>> cl.split_samples('/samples.txt', classes2remove=[1, 4],
            >>>                  class_subs={2:.1, 5:.01, 8:.1, 9:.9})
            >>>
            >>> # train a Random Forest model
            >>> cl.construct_model(output_model='/test_model.txt',
            >>>                    classifier_info={'classifier': 'RF',
            >>>                                     'trees': 1000,
            >>>                                     'max_depth': 25})
            >>>
            >>> # or load a previously trained RF model
            >>> cl.construct_model(input_model='/test_model.txt')
            >>>
            >>> # use Orfeo to train a model
            >>> cl.construct_model(classifier_info={'classifier': 'OR_RF', 'trees': 1000,
            >>>                    'max_depth': 25, 'min_samps': 5, 'rand_vars': 10},
            >>>                    input_image='/image.tif', in_shapefile='/shapefile.shp',
            >>>                    out_stats='/stats.xml', output_model='/rf_model.xml')
            >>>
            >>> # or collect statistics from samples rather than the entire image
            >>> cl.construct_model(classifier_info={'classifier': 'OR_RF', 'trees': 1000,
            >>>                    'max_depth': 25, 'min_samps': 5, 'rand_vars': 10},
            >>>                    input_image='/image.tif', in_shapefile='/shapefile.shp',
            >>>                    out_stats='/stats.xml', output_model='/rf_model.xml',
            >>>                    stats_from_image=False)
        """

        self.input_model = input_model
        self.output_model = output_model
        self.var_imp = var_imp
        self.rank_method = rank_method
        self.top_feas = top_feas
        self.get_probs = get_probs
        self.compute_importances = None
        self.in_shapefile = in_shapefile
        self.out_stats = out_stats
        self.stats_from_image = stats_from_image
        self.input_image = input_image
        self.classifier_info = classifier_info
        self.calibrate_proba = calibrate_proba
        self.class_weight = class_weight
        self.be_quiet = be_quiet
        self.n_jobs = n_jobs

        if isinstance(self.input_model, str):

            if not os.path.isfile(self.input_model):
                raise OSError('\n{} does not exist.\n'.format(self.input_model))

        if not isinstance(self.input_model, str):

            # check that the model is valid
            try:
                __ = self.classifier_info['classifier']
            except ValueError:
                raise ValueError('\nThe model must be declared.\n')

            if not isinstance(self.classifier_info['classifier'], list):
                if self.classifier_info['classifier'] not in get_available_models():
                    raise NameError('\n{} is not an option.\n'.format(self.classifier_info['classifier']))

        if isinstance(self.output_model, str):

            d_name, f_name = os.path.split(self.output_model)
            f_base, f_ext = os.path.splitext(f_name)

            self.out_acc = '{}/{}_acc.txt'.format(d_name, f_base)

            if os.path.isfile(self.out_acc):
                os.remove(self.out_acc)

            if 'CV' in self.classifier_info['classifier']:

                if 'xml' not in f_ext.lower():
                    raise OSError('\nThe output model for OpenCV models must be XML.\n')

            if not os.path.isdir(d_name):
                os.makedirs(d_name)

        if isinstance(self.rank_method, str):
            self.compute_importances = True

        if isinstance(self.class_weight, str):

            class_proportions = OrderedDict()
            class_counts_ordered = OrderedDict(self.class_counts)

            # Get the proportion of samples for each class.
            for class_value in self.classes:
                class_proportions[class_value] = class_counts_ordered[class_value] / float(self.n_samps)

                # len(np.array(self.classes)[np.where(np.array(self.classes) == class_value)]) / float(len(self.classes))

            if self.class_weight == 'inverse':

                # rank self.class_counts from smallest to largest
                class_counts_ordered = OrderedDict(sorted(class_counts_ordered.items(), key=lambda t: t[1]))

                # rank class_proportions from largest to smallest
                class_proportions = OrderedDict(sorted(class_proportions.items(), key=lambda t: t[1], reverse=True))

                # swap the proportions of the largest class counts to the smallest

                self.class_weight = {}

                for (k1, v1), (k2, v2) in zip(class_counts_ordered.items(), class_proportions.items()):
                    self.class_weight[k1] = v2
                    # self.class_weight.append(v2)

                if 'CV' in self.classifier_info['classifier']:
                    self.class_weight = np.array(self.class_weight.values(), dtype='float32')

            elif self.class_weight == 'percent':

                if 'CV' in self.classifier_info['classifier']:
                    self.class_weight = np.array(class_proportions.values(), dtype='float32')
                else:
                    self.class_weight = class_proportions

            else:
                raise ValueError('The weight method is not supported.')

        if isinstance(self.input_model, str):

            # Load the classifier parameters and the model.
            self._load_model()

        else:

            # Set the model parameters.
            self._default_parameters()

            # the model instance
            self._set_model()

            if self.classifier_info['classifier'] != 'ORRF':

                # get model parameters
                if not self.get_probs:
                    self._set_parameters()

                # train the model
                self._train_model()

    def _load_model(self):

        """
        Loads a previously saved model
        """

        if '.xml' in self.input_model:

            # first load the parameters
            try:
                self.classifier_info, __ = pickle.load(file(self.input_model.replace('.xml', '.txt'), 'rb'))
            except OSError:
                raise OSError('\nCould not load {}\n'.format(self.input_model))

            # load the correct model
            self._set_model()

            # now load the model
            try:
                self.model.load(self.input_model)
            except OSError:
                raise OSError('\nCould not load {}\n'.format(self.input_model))

        else:

            # Scikit-learn models
            try:
                self.classifier_info, self.model = pickle.load(file(self.input_model, 'rb'))
            except OSError:
                raise OSError('\nCould not load {}\n'.format(self.input_model))

    def _default_parameters(self):
        
        """
        Sets model parameters
        """

        defaults_ = {'n_estimators': 500,
                     'trials': 10,
                     'max_depth': 25,
                     'min_samples_split': 2,
                     'learning_rate': .1,
                     'n_jobs': -1}

        # Check if model parameters are set,
        #   otherwise, set defaults.

        if 'classifier' not in self.classifier_info:
            self.classifier_info['classifier'] = 'RF'

        if self.classifier_info['classifier'].startswith('AB_'):

            class_base = copy(self.classifier_info['classifier'])

            self.classifier_info['classifier'] = \
                self.classifier_info['classifier'][self.classifier_info['classifier'].find('_')+1:]

        else:
            class_base = 'none'

        vp = ParameterHandler(self.classifier_info['classifier'])

        # Check the parameters.
        self.classifier_info_ = copy(self.classifier_info)
        self.classifier_info_ = vp.check_parameters(self.classifier_info_, defaults_)

        # Create a separate instance for AdaBoost base classifiers.
        if class_base.startswith('AB_'):

            self.classifier_info_base = copy(self.classifier_info)
            self.classifier_info_base['classifier'] = class_base

            if 'trials' in self.classifier_info_base:
                self.classifier_info_base['n_estimators'] = self.classifier_info_base['trials']
                del self.classifier_info_base['trials']
            else:
                self.classifier_info_base['n_estimators'] = defaults_['trials']

            vp_base = ParameterHandler(self.classifier_info_base['classifier'])

            self.classifier_info_base = vp_base.check_parameters(self.classifier_info_base, defaults_, trials_set=True)

            if 'base_estimator' in self.classifier_info_base:
                del self.classifier_info_base['base_estimator']

            self.classifier_info['classifier'] = class_base

        # Random Forest in OpenCV
        if self.classifier_info['classifier'] in ['CVRF', 'CVEX_RF', 'CVRFR', 'CVEX_RFR']:

            if not self.input_model:

                # trees
                if 'trees' in self.classifier_info:
                    self.classifier_info['term_crit'] = (cv2.TERM_CRITERIA_MAX_ITER,
                                                         self.classifier_info['trees'], .1)
                else:
                    if 'term_crit' not in self.classifier_info:
                        self.classifier_info['term_crit'] = (cv2.TERM_CRITERIA_MAX_ITER, self.DEFAULT_TREES, .1)

                # minimum node samples
                if 'min_samps' not in self.classifier_info:
                    self.classifier_info['min_samps'] = int(np.ceil(.01 * self.n_samps))

                # random features
                if 'rand_vars' not in self.classifier_info:
                    # sqrt of feature count
                    self.classifier_info['rand_vars'] = 0

                # maximum node depth
                if 'max_depth' not in self.classifier_info:
                    self.classifier_info['max_depth'] = self.DEFAULT_MAX_DEPTH

                if 'calc_var_importance' not in self.classifier_info:
                    self.classifier_info['calc_var_importance'] = 0

                if 'truncate' not in self.classifier_info:
                    self.classifier_info['truncate'] = False

                if 'priors' not in self.classifier_info:
                    if isinstance(self.class_weight, np.ndarray):
                        self.classifier_info['priors'] = self.class_weight
                    else:
                        self.classifier_info['priors'] = np.ones(self.n_classes, dtype='float32')

            # MLP
            elif self.classifier_info['classifier'] == 'CVMLP':

                if not self.input_model:

                    # hidden nodes
                    try:
                        __ = self.classifier_info['n_hidden']
                    except:
                        try:
                            self.classifier_info['n_hidden'] = (self.n_feas + self.n_classes) / 2
                        except ValueError:
                            raise ValueError('\nCannot infer number of hidden nodes.\n')

    def _set_model(self):

        """
        Sets the model object
        """

        if self.classifier_info['classifier'] in ['ABR', 'GBR', 'EX_RFR', 'RFR', 'EX_RFR', 'SVR', 'SVRA']:
            self.discrete = False
        else:
            self.discrete = True

        # Create the model object.
        if isinstance(self.classifier_info['classifier'], list):

            classifier_list = []

            for ci, classifier in enumerate(self.classifier_info['classifier']):

                if classifier == 'Bayes':
                    classifier_list.append(('Bayes', GaussianNB()))

                elif classifier == 'NN':
                    classifier_list.append(('NN', KNeighborsClassifier(**self.classifier_info_)))

                elif classifier == 'Logistic':
                    classifier_list.append(('Logistic', LogisticRegression(**self.classifier_info_)))

                elif classifier == 'RF':
                    classifier_list.append(('RF', ensemble.RandomForestClassifier(**self.classifier_info_)))

                elif classifier == 'EX_RF':
                    classifier_list.append(('EX_RF', ensemble.ExtraTreesClassifier(**self.classifier_info_)))

                elif classifier == 'AB':
                    classifier_list.append(('AB',
                                            ensemble.AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(**self.classifier_info_),
                                                                        **self.classifier_info_base)))

                elif classifier == 'AB_RF':
                    classifier_list.append(('AB_RF',
                                            ensemble.AdaBoostClassifier(base_estimator=ensemble.RandomForestClassifier(**self.classifier_info_),
                                                                        **self.classifier_info_base)))

                elif classifier == 'AB_EX_RF':
                    classifier_list.append(('AB_EX_RF',
                                            ensemble.AdaBoostClassifier(base_estimator=ensemble.ExtraTreesClassifier(**self.classifier_info_),
                                                                        **self.classifier_info_base)))

                elif classifier == 'AB_DT':
                    classifier_list.append(('AB_DT', ensemble.AdaBoostClassifier(base_estimator=tree.ExtraTreeClassifier(**self.classifier_info_),
                                                                                 **self.classifier_info_base)))

                elif classifier == 'AB_EX_DT':
                    classifier_list.append(('AB_EX_DT',
                                            ensemble.AdaBoostClassifier(base_estimator=tree.ExtraTreeClassifier(**self.classifier_info_),
                                                                        **self.classifier_info_base)))

                elif classifier == 'Bag':
                    classifier_list.append(('Bag',
                                            ensemble.BaggingClassifier(base_estimator=tree.DecisionTreeClassifier(**self.classifier_info_),
                                                                       **self.classifier_info_base)))

                elif classifier == 'EX_Bag':
                    classifier_list.append(('EX_Bag',
                                            ensemble.BaggingClassifier(base_estimator=tree.ExtraTreeClassifier(**self.classifier_info_),
                                                                       **self.classifier_info_base)))

                elif classifier == 'GB':
                    classifier_list.append(('GB', ensemble.GradientBoostingClassifier(**self.classifier_info_)))

                if self.calibrate_proba:

                    temp_model = classifier_list[ci][1]

                    if isinstance(self.sample_weight, np.ndarray):
                        temp_model.fit(self.p_vars, self.labels, sample_weight=self.sample_weight)
                    else:
                        temp_model.fit(self.p_vars, self.labels)

                    if self.n_samps >= 1000:
                        cal_model = calibration.CalibratedClassifierCV(temp_model,
                                                                       method='isotonic')
                    else:
                        cal_model = calibration.CalibratedClassifierCV(temp_model,
                                                                       method='sigmoid')

                    cal_model.fit(self.p_vars_test, self.labels_test)

                    classifier_list[ci] = (classifier, cal_model)

            self.model = ensemble.VotingClassifier(estimators=classifier_list, voting='soft')

        else:

            if self.classifier_info['classifier'] == 'Bayes':

                self.model = GaussianNB()

                # self.model = cv2.ml.NormalBayesClassifier_create()

            elif self.classifier_info['classifier'] == 'CART':

                self.model = cv2.ml.DTrees_create()

            # elif self.classifier_info['classifier'] in ['CVEX_RF', 'CVEX_RFR']:
            #
            #     self.model = cv2.ERTrees()

            elif self.classifier_info['classifier'] in ['CVRF', 'CVRFR']:

                if not self.get_probs:
                    self.model = cv2.ml.RTrees_create()

            # elif self.classifier_info['classifier'] == 'CVMLP':
            #
            #     if self.input_model:
            #     self.model = cv2.ml.ANN_MLP_create()
            #     # else:
            #     #     self.model = cv2.ml.ANN_MLP_create(np.array([self.n_feas, self.classifier_info['n_hidden'],
            #     #                                                  self.n_classes]))

            elif self.classifier_info['classifier'] in ['CVSVM', 'CVSVMA', 'CVSVMR', 'CVSVMRA']:

                self.model = cv2.ml.SVM_create()

            elif self.classifier_info['classifier'] == 'DT':

                self.model = tree.DecisionTreeClassifier(**self.classifier_info_)

            elif self.classifier_info['classifier'] == 'DTR':

                self.model = tree.DecisionTreeRegressor(**self.classifier_info_)

            elif self.classifier_info['classifier'] == 'EX_DT':

                self.model = tree.ExtraTreeClassifier(**self.classifier_info_)

            elif self.classifier_info['classifier'] == 'EX_DTR':

                self.model = tree.ExtraTreeRegressor(**self.classifier_info_)

            elif self.classifier_info['classifier'] == 'Logistic':

                self.model = LogisticRegression(**self.classifier_info_)

            elif self.classifier_info['classifier'] == 'NN':

                self.model = KNeighborsClassifier(**self.classifier_info_)

            elif self.classifier_info['classifier'] == 'RF':

                self.model = ensemble.RandomForestClassifier(**self.classifier_info_)

            elif self.classifier_info['classifier'] == 'EX_RF':

                self.model = ensemble.ExtraTreesClassifier(**self.classifier_info_)

            elif self.classifier_info['classifier'] == 'RFR':

                self.model = ensemble.RandomForestRegressor(**self.classifier_info_)

            elif self.classifier_info['classifier'] == 'EX_RFR':

                self.model = ensemble.ExtraTreesRegressor(**self.classifier_info_)

            elif self.classifier_info['classifier'] == 'AB':

                self.model = ensemble.AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(**self.classifier_info_),
                                                         **self.classifier_info_base)

            elif self.classifier_info['classifier'] == 'AB_RF':

                self.model = ensemble.AdaBoostClassifier(base_estimator=ensemble.RandomForestClassifier(**self.classifier_info_),
                                                         **self.classifier_info_base)

            elif self.classifier_info['classifier'] == 'AB_EX_RF':

                self.model = ensemble.AdaBoostClassifier(base_estimator=ensemble.ExtraTreesClassifier(**self.classifier_info_),
                                                         **self.classifier_info_base)

            elif self.classifier_info['classifier'] == 'AB_EX_DT':

                self.model = ensemble.AdaBoostClassifier(base_estimator=tree.ExtraTreeClassifier(**self.classifier_info_),
                                                         **self.classifier_info_base)

            elif self.classifier_info['classifier'] == 'ABR':

                self.model = ensemble.AdaBoostRegressor(base_estimator=tree.DecisionTreeRegressor(**self.classifier_info_),
                                                        **self.classifier_info_base)

            elif self.classifier_info['classifier'] == 'ABR_EX_DTR':

                self.model = ensemble.AdaBoostRegressor(base_estimator=tree.ExtraTreeRegressor(**self.classifier_info_),
                                                        **self.classifier_info_base)

            elif self.classifier_info['classifier'] == 'Bag':

                self.model = ensemble.BaggingClassifier(base_estimator=tree.DecisionTreeClassifier(**self.classifier_info_),
                                                        **self.classifier_info_base)

            elif self.classifier_info['classifier'] == 'BagR':

                self.model = ensemble.BaggingRegressor(base_estimator=tree.DecisionTreeRegressor(**self.classifier_info_),
                                                       **self.classifier_info_base)

            elif self.classifier_info['classifier'] == 'EX_Bag':

                self.model = ensemble.BaggingClassifier(base_estimator=tree.ExtraTreeClassifier(**self.classifier_info_),
                                                        **self.classifier_info_base)

            elif self.classifier_info['classifier'] == 'GB':

                self.model = ensemble.GradientBoostingClassifier(**self.classifier_info_)

            elif self.classifier_info['classifier'] == 'GBR':

                self.model = ensemble.GradientBoostingRegressor(**self.classifier_info_)

            elif self.classifier_info['classifier'] == 'SVM':

                self.model = SVC(**self.classifier_info_)

            elif self.classifier_info['classifier'] == 'QDA':

                self.model = QDA()

            elif self.classifier_info['classifier'] == 'ORRF':

                # try:
                #     import otbApplication as otb
                # except ImportError:
                #     raise ImportError('Orfeo tooblox needs to be installed')

                v_info = vector_tools.vinfo(self.in_shapefile)

                if v_info.shp_geom_name.lower() == 'point':
                    sys.exit('\nThe input shapefile must be a polygon.\n')

                if os.path.isfile(self.out_stats):

                    print '\nThe statistics already exist'

                else:

                    if self.stats_from_image:

                        # image statistics
                        com = 'otbcli_ComputeImagesStatistics -il %s -out %s' % (self.input_image, self.out_stats)

                        subprocess.call(com, shell=True)

                    else:

                        gap_1 = '    '
                        gap_2 = '        '

                        xml_string = '<?xml version="1.0" ?>\n<FeatureStatistics>\n{}<Statistic name="mean">\n{}'.format(gap_1, gap_2)

                        # gather stats from samples
                        for fea_pos in xrange(0, self.n_feas):

                            stat_line = '<StatisticVector value="%f" />' % self.p_vars[:, fea_pos].mean()

                            # add the line to the xml string
                            if (fea_pos + 1) == self.n_feas:
                                xml_string = '%s%s\n%s' % (xml_string, stat_line, gap_1)
                            else:
                                xml_string = '%s%s\n%s' % (xml_string, stat_line, gap_2)

                        xml_string = '%s</Statistic>\n%s<Statistic name="stddev">\n%s' % (xml_string, gap_1, gap_2)

                        # gather stats from samples
                        for fea_pos in xrange(0, self.n_feas):

                            stat_line = '<StatisticVector value="%f" />' % self.p_vars[:, fea_pos].std()

                            # add the line to the xml string
                            if (fea_pos + 1) == self.n_feas:
                                xml_string = '%s%s\n%s' % (xml_string, stat_line, gap_1)
                            else:
                                xml_string = '%s%s\n%s' % (xml_string, stat_line, gap_2)

                        xml_string = '%s</Statistic>\n</FeatureStatistics>\n' % xml_string

                        with open(self.out_stats, 'w') as xml_wr:
                            xml_wr.writelines(xml_string)

                # app = otb.Registry.CreateApplication('ComputeImagesStatistics')
                # app.SetParameterString('il', input_image)
                # app.SetParameterString('out', output_stats)
                # app.ExecuteAndWriteOutput()

                if os.path.isfile(self.output_model):
                    os.remove(self.output_model)

                # train the model
                com = 'otbcli_TrainImagesClassifier -io.il {} -io.vd {} -io.imstat {} -classifier rf \
                -classifier.rf.max {:d} -classifier.rf.nbtrees {:d} \
                -classifier.rf.min {:d} -classifier.rf.var {:d} -io.out {}'.format(self.input_image,
                                                                                   self.in_shapefile,
                                                                                   self.out_stats,
                                                                                   self.classifier_info['max_depth'],
                                                                                   self.classifier_info['trees'],
                                                                                   self.classifier_info['min_samps'],
                                                                                   self.classifier_info['rand_vars'],
                                                                                   self.output_model)

                subprocess.call(com, shell=True)

                # app = otb.Registry.CreateApplication('TrainImagesClassifier')
                # app.SetParameterString('io.il', input_image)
                # app.SetParameterString('io.vd', input_shapefile)
                # app.SetParameterString('io.imstat', output_stats)
                # app.SetParameterString('classifier', 'rf')
                # app.SetParameterString('classifier.rf.max', str(classifier_info['max_depth']))
                # app.SetParameterString('classifier.rf.nbtrees', str(classifier_info['trees']))
                # app.SetParameterString('classifier.rf.min', str(classifier_info['min_samps']))
                # app.SetParameterString('classifier.rf.var', str(classifier_info['rand_vars']))
                # app.SetParameterString('io.out', output_model)
                # app.ExecuteAndWriteOutput()

            else:
                raise NameError('\nThe model {} is not supported'.format(self.classifier_info['classifier']))

    def _set_parameters(self):

        """
        Sets model parameters for OpenCV
        """

        #############################################
        # Set algorithm parameters for OpenCV models.
        #############################################

        # if self.classifier_info['classifier'] == 'Boost':
        #
        #     # GBTREES_SQUARED_LOSS, GBTREES_ABSOLUTE_LOSS, GBTREES_HUBER_LOSS, GBTREES_DEVIANCE_LOSS
        #     self.parameters = dict(loss_function_type=cv2.GBTREES_HUBER_LOSS,
        #                            subsample_portion=.05,
        #                            weak_count=self.classifier_info['trees'],
        #                            max_depth=self.classifier_info['max_depth'])

        if self.classifier_info['classifier'] == 'CART':

            self.parameters = dict(max_depth=self.classifier_info['max_depth'],
                                   min_sample_count=self.classifier_info['min_samps'],
                                   use_surrogates=self.var_imp,
                                   term_crit=(cv2.TERM_CRITERIA_MAX_ITER, self.classifier_info['trees'], .1))

        elif self.classifier_info['classifier'] in ['CVRF', 'CVEX_RF']:

            self.model.setMaxDepth(self.classifier_info['max_depth'])
            self.model.setMinSampleCount(self.classifier_info['min_samps'])
            self.model.setCalculateVarImportance(self.classifier_info['calc_var_importance'])
            self.model.setActiveVarCount(self.classifier_info['rand_vars'])
            self.model.setTermCriteria(self.classifier_info['term_crit'])

            if self.classifier_info['priors'].min() < 1:
                self.model.setPriors(self.classifier_info['priors'])
            
            self.model.setTruncatePrunedTree(self.classifier_info['truncate'])

            # self.parameters = dict(max_depth=self.classifier_info['max_depth'],
            #                        min_sample_count=self.classifier_info['min_sample_count'],
            #                        calc_var_importance=self.classifier_info['calc_var_importance'],
            #                        nactive_vars=self.classifier_info['nactive_vars'],
            #                        term_crit=self.classifier_info['term_crit'])

            # termcrit_type=cv2.TERM_CRITERIA_MAX_ITER)    # cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER

        elif self.classifier_info['classifier'] == 'CVMLP':

            n_steps = 1000
            max_err = .0001
            step_size = .3
            momentum = .2

            # cv2.TERM_CRITERIA_EPS
            self.parameters = dict(term_crit=(cv2.TERM_CRITERIA_COUNT, n_steps, max_err),
                                   train_method=cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP,
                                   bp_dw_scale=step_size,
                                   bp_moment_scale=momentum)

        elif self.classifier_info['classifier'] == 'CVSVM':

            self.model.setC(self.classifier_info_svm['C'])
            self.model.setGamma(self.classifier_info_svm['gamma'])
            self.model.setKernel(cv2.ml.SVM_RBF)
            self.model.setType(cv2.ml.SVM_C_SVC)

            # self.parameters = dict(kernel_type=cv2.ml.SVM_RBF,
            #                        svm_type=cv2.ml.SVM_C_SVC,
            #                        C=self.classifier_info_svm['C'],
            #                        gamma=self.classifier_info_svm['gamma'])

        elif self.classifier_info['classifier'] == 'CVSVMA':

            # SVM, parameters optimized
            self.parameters = dict(kernel_type=cv2.ml.SVM_RBF, svm_type=cv2.ml.SVM_C_SVC)

        elif self.classifier_info['classifier'] == 'CVSVMR':

            # SVM regression
            self.parameters = dict(kernel_type=cv2.ml.SVM_RBF,
                                   svm_type=cv2.ml.SVM_NU_SVR,
                                   C=self.classifier_info_svm['C'],
                                   gamma=self.classifier_info_svm['gamma'],
                                   nu=self.classifier_info_svm['nu'],
                                   p=self.classifier_info_svm['p'])

        elif self.classifier_info['classifier'] == 'CVSVMRA':

            # SVM regression, parameters optimized
            self.parameters = dict(kernel_type=cv2.ml.SVM_RBF, svm_type=cv2.ml.SVM_NU_SVR,
                                   nu=self.classifier_info['nu'])

        else:

            self.parameters = None

    def _train_model(self):

        """
        Trains a model and saves to file if prompted
        """

        if not self.be_quiet:

            if self.classifier_info['classifier'][0].lower() in 'aeiou':
                a_or_an = 'an'
            else:
                a_or_an = 'a'

            if isinstance(self.classifier_info['classifier'], list):
                print '\nTraining a voting model with {} ...\n'.format(','.join(self.classifier_info['classifier']))
            else:
                print '\nTraining {} {} model with {:,d} samples and {:,d} variables ...\n'.format(a_or_an,
                                                                                                   self.classifier_info['classifier'],
                                                                                                   self.n_samps,
                                                                                                   self.n_feas)

        # OpenCV tree-based models
        if self.classifier_info['classifier'] in ['CART', 'CVRF', 'CVEX_RF']:

            ## first, run the model with all features to get the importance
            ## then, re-train the model with the desired feature subset
            # if isinstance(self.rank_method, str):

                ## it is necessary to train a RF model to get the feature importance
                ## however, this can be skipped if the features are ranked with 'chi2'
                # if 'RF' in self.rank_method:
                    # self.model.train(self.p_vars, cv2.CV_ROW_SAMPLE, self.labels, params=self.parameters)

                # rank the features
                # self.rank_feas(rank_method=self.rank_method, top_feas=self.top_feas)

                # self.model.train(self.p_vars, cv2.CV_ROW_SAMPLE, self.labels, varIdx=self.ranked_feas-1,
                #                  params=self.parameters)

            # if self.get_probs:
            #
            #     self.model = RandomForestClassifier(n_estimators=self.classifier_info['trees'],
            #                                         min_samples_split=self.classifier_info['min_samps'],
            #                                         max_depth=self.classifier_info['max_depth'],
            #                                         n_jobs=-1).fit(self.p_vars, self.labels)

            self.model.train(self.p_vars, 0, self.labels)
            # self.model.train(self.p_vars, cv2.CV_ROW_SAMPLE, self.labels, params=self.parameters)

        # OpenCV tree-based regression
        elif self.classifier_info['classifier'] in ['CVRFR', 'CVEX_RFR']:

            self.model.train(self.p_vars, cv2.CV_ROW_SAMPLE, self.labels,
                             varType=np.zeros(self.n_feas+1).astype(np.uint8), params=self.parameters)

        elif self.classifier_info['classifier'] == 'CVMLP':

            # Convert input strings to binary zeros and ones, and set the output
            # array to all -1's with ones along the diagonal.
            targets = -1 * np.ones((self.n_samps, self.n_classes), 'float')

            for i in xrange(0, self.n_samps):

                lab_idx = sorted(self.classes).index(self.labels[i])

                targets[i, lab_idx] = 1

            self.model.train(self.p_vars, targets, None, params=self.parameters)

        elif self.classifier_info['classifier'] in ['CVSVM', 'CVSVMR']:

            if isinstance(self.rank_method, str):
                self.rank_feas(rank_method=self.rank_method, top_feas=self.top_feas)
                # self.model.train(self.p_vars, self.labels, varIdx=self.ranked_feas-1, params=self.parameters)
                self.model.train(self.p_vars, self.labels, varIdx=self.ranked_feas-1)
            else:
                # self.model.train(self.p_vars, self.labels, params=self.parameters)
                self.model.train(self.p_vars, 0, self.labels)

        elif self.classifier_info['classifier'] in ['CVSVMA', 'CVSVRA']:

            print '  Be patient. Auto tuning can take a while.\n'

            self.model.train_auto(self.p_vars, self.labels, None, None, params=self.parameters, k_fold=10)

        # Scikit-learn models
        else:

            if isinstance(self.sample_weight, np.ndarray):
                self.model.fit(self.p_vars, self.labels, sample_weight=self.sample_weight)
            else:
                self.model.fit(self.p_vars, self.labels)

            if self.calibrate_proba:

                # clf_probs = self.model.predict_proba(self.p_vars_test)

                if self.n_samps >= 1000:
                    self.model = calibration.CalibratedClassifierCV(self.model, method='isotonic', cv='prefit')
                else:
                    self.model = calibration.CalibratedClassifierCV(self.model, method='sigmoid', cv='prefit')

                self.model.fit(self.p_vars_test, self.labels_test)

                # clf_probs_cal = self.model_.predict_proba(self.p_vars_test)
                #
                # import matplotlib.pyplot as plt
                #
                # plt.figure(0)
                # colors = ['r', 'g', 'b', 'orange', 'purple', 'green', 'yellow', 'black', 'cyan', 'magenta', 'gray']
                # for i in xrange(clf_probs.shape[0]):
                #     plt.arrow(clf_probs[i, 0], clf_probs[i, 1],
                #               clf_probs_cal[i, 0] - clf_probs[i, 0],
                #               clf_probs_cal[i, 1] - clf_probs[i, 1],
                #               color=colors[self.labels_test[i]], head_width=1e-2)
                #
                # plt.show()
                # sys.exit()

        if isinstance(self.output_model, str):

            if 'CV' in self.classifier_info['classifier']:

                if '.xml' not in self.output_model.lower():
                    raise NameError('\nAn OpenCV model should be .xml.\n')

                try:

                    self.model.save(self.output_model)

                    # dump the parameters to a text file
                    pickle.dump([self.classifier_info, self.model],
                                file(self.output_model.replace('.xml', '.txt'), 'wb'),
                                protocol=pickle.HIGHEST_PROTOCOL)

                except OSError:
                    raise OSError('\nCould not save {} to file.\n'.format(self.output_model))

            else:

                if '.txt' not in self.output_model.lower():
                    raise NameError('\nA Scikit-learn model should be .txt.\n')

                try:

                    pickle.dump([self.classifier_info, self.model],
                                file(self.output_model, 'wb'),
                                protocol=pickle.HIGHEST_PROTOCOL)

                except OSError:
                    raise OSError('\nCould not save {} to file.\n'.format(self.output_model))

            if isinstance(self.p_vars_test, np.ndarray):
                self.test_accuracy(out_acc=self.out_acc, discrete=self.discrete)

    def predict_array(self, array2predict):

        if self.classifier_info['classifier'] in ['C5', 'Cubist']:

            features = ro.r.matrix(array2predict, nrow=array2predict.shape[0],
                                   ncol=array2predict.shape[1])

            features.colnames = StrVector(self.headers[:-1])

            return _do_c5_cubist_predict(self.model, self.classifier_info['classifier'], features)

        else:
            return self.model.predict(array2predict)

    def predict(self, input_image, out_image, additional_layers=[], scale_data=False,
                band_check=-1, ignore_feas=[], use_xy=False, in_stats=None,
                in_model=None, mask_background=None, background_band=2,
                background_value=0, minimum_observations=0, observation_band=0,
                row_block_size=1024, col_block_size=1024, n_jobs=-1, gdal_cache=256):

        """
        Applies a model to predict class labels

        Args:
            input_image (str): The input image to classify.
            out_image (str): The output image.
            additional_layers (Optional[list]): A list of additional images (layers) that are not part
                of ``input_image``.
            scale_data (Optional[bool]): Whether to scale data. Default is False.
            band_check (Optional[int]): The band to check for 'no data'. Default is -1, or do not perform check. 
            ignore_feas (Optional[list]): A list of features (band layers) to ignore. Default is an empty list, 
                or use all features.
            use_xy (Optional[bool]): Whether to use x, y coordinates as predictive variables. Default is False.
            in_stats (Optional[str]): A XML statistics file. Default is None. *Only applicable to Orfeo models.
            in_model (Optional[str]): A model file to load. Default is None. *Only applicable to Orfeo
                and C5/Cubist models.
            mask_background (Optional[str]): An image to use as a background mask, applied post-classification.
                Default is None.
            background_band (int): The band from ``mask_background`` to use for null background value. Default is 2.
            background_value (Optional[int]): The background value in ``mask_background``. Default is 0.
            minimum_observations (Optional[int]): A minimum number of observations in ``mask_background`` to be
                recoded to 0. Default is 0, or no minimum observation filter.
            observation_band (Optional[int]): The band position in ``mask_background`` of the ``minimum_observations``
                counts. Default is 0.
            row_block_size (Optional[int]): The row block size (pixels). Default is 1024.
            col_block_size (Optional[int]): The column block size (pixels). Default is 1024.
            n_jobs (Optional[int]): The number of processors to use for parallel mapping. Default is -1, or all
                available processors.
            gdal_cache (Optional[int]). The GDAL cache (MB). Default is 256.
            
        Returns:
            None, writes to ``out_image``.

        Examples:
            >>> from mappy.classifiers import classification
            >>> cl = classification()
            >>>
            >>> # You can use x, y coordinates, but note that these must be
            >>> #   supplied to ``predict`` also.
            >>> cl.split_samples('/samples.txt', perc_samp_each=.7, use_xy=True)
            >>>
            >>> # Random Forest with Scikit-learn
            >>> cl.construct_model(classifier_info={'classifier': 'RF', 'trees': 100, 'max_depth': 25})
            >>>
            >>> # Random Forest with OpenCV
            >>> cl.construct_model(classifier_info={'classifier': 'CVRF', 'trees': 100,
            >>>                    'max_depth': 25, 'truncate': True})
            >>>
            >>> # Apply the classification model to map image class labels.
            >>> cl.predict('/image_feas.tif', '/image_labels.tif', ignore_feas=[1, 6])
            >>>
            >>> # or use Orfeo to predict class labels
            >>> cl.construct_model(classifier_info={'classifier': 'OR_RF', 'trees': 1000,
            >>>                    'max_depth': 25, 'min_samps': 5, 'rand_vars': 10},
            >>>                    input_image='/image.tif', in_shapefile='/shapefile.shp',
            >>>                    out_stats='/stats.xml', output_model='/rf_model.xml')
            >>>
            >>> cl.predict('/image_feas.tif', '/image_labels.tif',
            >>>            in_stats='/stats.xml', in_model='/rf_model.xml')
        """

        self.input_image = input_image
        self.out_image = out_image        
        self.additional_layers = additional_layers
        self.ignore_feas = ignore_feas
        self.scale_data = scale_data
        self.band_check = band_check
        self.row_block_size = row_block_size
        self.col_block_size = col_block_size
        self.use_xy = use_xy
        self.mask_background = mask_background
        self.background_band = background_band
        self.background_value = background_value
        self.minimum_observations = minimum_observations
        self.observation_band = observation_band
        self.n_jobs = n_jobs
        self.chunk_size = (self.row_block_size * self.col_block_size) / 100

        if not hasattr(self, 'classifier_info'):

            raise NameError("""\
            There is no `classifier_info` object. Be sure to run `construct_model`
            or `construct_r_model` before running `predict`.
            """)

        d_name, f_name = os.path.split(self.out_image)
        f_base, f_ext = os.path.splitext(f_name)

        self.out_image_temp = '{}/{}_temp.tif'.format(d_name, f_base)
        self.temp_model_file = '{}/temp_model_file.txt'.format(d_name)

        if not os.path.isdir(d_name):
            os.makedirs(d_name)
        
        # Orfeo Toolbox application
        if 'OR' in self.classifier_info['classifier']:

            # make predictions
            if isinstance(in_stats, str):

                if isinstance(self.mask_background, str):

                    com = 'otbcli_ImageClassifier -in {} -imstat {} \
                    -model {} -out {} -ram {:d}'.format(input_image, in_stats, in_model,
                                                        self.out_image_temp, gdal_cache)

                else:
                    com = 'otbcli_ImageClassifier -in {} -imstat {} \
                    -model {} -out {} -ram {:d}'.format(input_image, in_stats, in_model, self.out_image, gdal_cache)

            else:

                if isinstance(self.mask_background, str):

                    com = 'otbcli_ImageClassifier -in {} -model {} -out {} -ram {:d}'.format(input_image, in_model,
                                                                                             self.out_image_temp,
                                                                                             gdal_cache)

                else:

                    com = 'otbcli_ImageClassifier -in {} -model {} -out {} -ram {:d}'.format(input_image, in_model,
                                                                                             self.out_image, gdal_cache)

            try:
                subprocess.call(com, shell=True)
            except OSError:
                raise OSError('Are you sure the Orfeo Toolbox is installed?')

            if isinstance(self.mask_background, str):

                self._mask_background(self.out_image_temp, self.out_image, self.mask_background,
                                      self.background_band, self.background_value,
                                      self.minimum_observations, self.observation_band)

            # app = otb.Registry.CreateApplication('ImageClassifier')
            # app.SetParameterString('in', input_image)
            # app.SetParameterString('imstat', output_stats)
            # app.SetParameterString('model', output_model)
            # app.SetParameterString('out', output_map)
            # app.ExecuteAndWriteOutput()

        else:

            self.open_image = False

            self.i_info = raster_tools.rinfo(self.input_image)

            # Block record keeping.
            self.record_keeping = '{}/{}_record.txt'.format(d_name, f_base)

            if os.path.isfile(self.record_keeping):
                self.record_list = pickle.load(file(self.record_keeping, 'rb'))
            else:
                self.record_list = []

            # Output image information.
            self.o_info = self.i_info.copy()

            # Set the number of output bands.
            # if hasattr(self, 'get_probs'):
            if self.get_probs:
                self.o_info.bands = self.n_classes
            else:
                self.o_info.bands = 1

            if self.classifier_info['classifier'] in ['ABR', 'ABR_EX_DTR', 'BGR', 'BagR', 'RFR', 'EX_RFR', 'CV_RFR',
                                                      'CVEX_RFR', 'SVR', 'SVRA', 'Cubist', 'DTR']:
                self.o_info.storage = 'float32'
            else:
                self.o_info.storage = 'byte'

            print '\nMapping labels ...\n'

            self._predict()

            if self.open_image:
                self.i_info.close()

            self.o_info.close()

    def _predict(self):

        # Global variables for parallel processing.
        global features, model_pp, predict_samps

        if 'CV' not in self.classifier_info['classifier']:
            model_pp = deepcopy(self.model)

        rows, cols = self.i_info.rows, self.i_info.cols

        if self.ignore_feas:
            bands2open = sorted([bd for bd in xrange(1, self.i_info.bands+1) if bd not in self.ignore_feas])
        else:
            bands2open = range(1, self.i_info.bands+1)

        if isinstance(self.mask_background, str):
            out_raster_object = raster_tools.create_raster(self.out_image_temp, self.o_info,
                                                           compress='none', tile=False, bigtiff='yes')
        else:
            out_raster_object = raster_tools.create_raster(self.out_image, self.o_info, tile=False)

        # if hasattr(self, 'get_probs'):

        if self.get_probs:
            out_bands = [out_raster_object.datasource.GetRasterBand(bd) for bd in xrange(1, self.o_info.bands+1)]
        else:
            out_raster_object.get_band(1)

        out_raster_object.fill(0)

        if isinstance(self.scale_data, str):
            self.scaler = pickle.load(file(self.scale_data, 'rb'))
        elif not self.scale_data:
            self.scaler = False

        block_rows, block_cols = raster_tools.block_dimensions(rows, cols,
                                                               row_block_size=self.row_block_size,
                                                               col_block_size=self.col_block_size)

        # set widget and pbar
        #ttl_blks_ct, pbar = _iteration_parameters(rows, cols, block_rows, block_cols)

        n_blocks = 0
        for i in xrange(0, rows, block_rows):
            for j in xrange(0, cols, block_cols):
                n_blocks += 1

        n_block = 1
        for i in xrange(0, rows, block_rows):

            n_rows = self._num_rows_cols(i, block_rows, rows)

            for j in xrange(0, cols, block_cols):

                print 'Block {:d} of {:d} ...'.format(n_block, n_blocks)
                n_block += 1

                if n_block in self.record_list:

                    print '  Skipping current block ...'

                    continue
                    
                n_cols = self._num_rows_cols(j, block_cols, cols)

                # Check for zeros in the block.
                if self.band_check != -1:

                    if self.open_image:
                        self.i_info = raster_tools.rinfo(self.input_image)
                        self.open_image = False

                    max_check = self.i_info.mparray(bands2open=self.band_check,
                                                    i=i,
                                                    j=j,
                                                    rows=n_rows,
                                                    cols=n_cols).max()

                    if max_check == 0:
                        continue

                if not self.open_image:

                    # Close the image information object because it
                    #   needs to be reopened for parallel ``mparray``.
                    self.i_info.close()
                    self.open_image = True

                if 'CV' in self.classifier_info['classifier']:

                    if len(bands2open) != self.model.getVarCount():
                        raise ValueError('\nThe number of predictive layers does not match the number of model estimators.\n')

                elif self.classifier_info['classifier'] not in ['C5', 'Cubist']:

                    if ('CV' not in self.classifier_info['classifier']) and (self.classifier_info['classifier'] != 'QDA'):

                        try:

                            if len(bands2open) != self.model.n_features_:
                                raise ValueError('\nThe number of predictive layers does not match the number of model estimators.\n')

                        except:

                            try:
                                if len(bands2open) != self.model.base_estimator.n_features_:
                                    raise ValueError('\nThe number of predictive layers does not match the number of model estimators.\n')
                            except:
                                pass

                # Get all the bands for the tile. The shape
                #   of the features is (features x rows x columns).
                features = raster_tools.mparray(image2open=self.input_image,
                                                bands2open=bands2open,
                                                i=i, j=j,
                                                rows=n_rows, cols=n_cols,
                                                predictions=True,
                                                d_type='float32',
                                                n_jobs=-1)

                n_samples = n_rows * n_cols

                if isinstance(self.scale_data, str) or self.scale_data:
                    features = self.scaler.transform(features)

                # MLP and SVM give the option to predict all samples at once
                #   otherwise, pass it to compiled C to predict samples one by one
                # if hasattr(self, 'get_probs'):
                if self.get_probs:

                    predicted = self.model.predict_proba(features)

                    for cl in xrange(0, self.n_classes):
                        out_bands[cl].WriteArray(predicted[:, cl].reshape(n_cols, n_rows).T, j, i)

                else:

                    if 'CV' in self.classifier_info['classifier']:

                        if self.classifier_info['classifier'] == 'CVMLP':

                            self.model.predict(features, predicted)

                            predicted = np.argmax(predicted, axis=1)

                        else:

                            # Setup the global array to write to. This avoids
                            #   passing it to the joblib workers.
                            # predicted = np.empty((n_samples, 1), dtype='uint8')

                            predicted = Parallel(n_jobs=self.n_jobs,
                                                 max_nbytes=None)(delayed(predict_cv)(chunk, self.chunk_size,
                                                                                      self.file_name, self.perc_samp,
                                                                                      self.classes2remove,
                                                                                      self.ignore_feas, self.use_xy,
                                                                                      self.classifier_info,
                                                                                      self.weight_classes)
                                                                  for chunk in xrange(0, n_samples, self.chunk_size))

                        # transpose and reshape the predicted labels to (rows x columns)
                        out_raster_object.write_array(np.array(list(itertools.chain.from_iterable(predicted))).reshape(n_cols, n_rows).T, i, j)

                    elif self.classifier_info['classifier'] in ['C5', 'Cubist']:

                        # Load the predictor variables.
                        # predict_samps = pandas2ri.py2ri(pd.DataFrame(features))

                        predict_samps = ro.r.matrix(features, nrow=n_samples, ncol=len(bands2open))
                        predict_samps.colnames = StrVector(self.headers[:-1])

                        # Get chunks for parallel processing.
                        indice_pairs = []
                        for i_ in xrange(1, n_samples+1, self.chunk_size):
                            n_rows_ = self._num_rows_cols(i_, self.chunk_size, n_samples)
                            indice_pairs.append([i_, n_rows_])
                        indice_pairs[-1][1] += 1

                        # Make the predictions and convert to a NumPy array.
                        if isinstance(self.input_model, str):

                            predicted = Parallel(n_jobs=self.n_jobs,
                                                 max_nbytes=None)(delayed(predict_c5_cubist)(self.input_model, ip)
                                                                  for ip in indice_pairs)

                            # Write the predictions to file.
                            out_raster_object.write_array(np.array(list(itertools.chain.from_iterable(predicted))).reshape(n_cols, n_rows).T, i, j)

                        else:

                            out_raster_object.write_array(_do_c5_cubist_predict(self.model,
                                                                                self.classifier_info['classifier'],
                                                                                predict_samps).reshape(n_cols,
                                                                                                       n_rows).T, i, j)

                    else:

                        if (self.n_jobs != 0) and (self.n_jobs != 1):

                            # Get chunks for parallel processing.
                            indice_pairs = []
                            for i_ in xrange(0, n_samples, self.chunk_size):
                                n_rows_ = self._num_rows_cols(i_, self.chunk_size, n_samples)
                                indice_pairs.append([i_, n_rows_])

                            # Make the predictions and convert to a NumPy array.
                            if isinstance(self.input_model, str):

                                predicted = Parallel(n_jobs=self.n_jobs,
                                                     max_nbytes=None)(delayed(predict_scikit)(self.input_model,
                                                                                              ip)
                                                                      for ip in indice_pairs)

                                # Write the predictions to file.
                                out_raster_object.write_array(np.array(list(itertools.chain.from_iterable(predicted))).reshape(n_cols, n_rows).T, i, j)

                        else:

                            # Get chunks for parallel processing.
                            indice_pairs = []
                            for i_ in xrange(0, n_samples, self.chunk_size):
                                n_rows_ = self._num_rows_cols(i_, self.chunk_size, n_samples)
                                indice_pairs.append([i_, n_rows_])

                            if isinstance(self.input_model, str):
                                __, m = pickle.load(file(self.input_model, 'rb'))
                            else:
                                m = self.model

                            # Make the predictions and convert to a NumPy array.
                            if isinstance(self.input_model, str):

                                predicted = [predict_scikit(m, ip) for ip in indice_pairs]

                                # Write the predictions to file.
                                out_raster_object.write_array(np.array(list(itertools.chain.from_iterable(predicted))).reshape(n_cols, n_rows).T, i, j)

                self.record_list.append(n_block)

                if os.path.isfile(self.record_keeping):
                    os.remove(self.record_keeping)

                pickle.dump(self.record_list, file(self.record_keeping, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

                #pbar.update(ttl_blks_ct)
                #ttl_blks_ct += 1

        # os.remove(self.temp_model_file)

        #pbar.finish()

        # if hasattr(self, 'get_probs'):
        if self.get_probs:

            for cl in xrange(0, self.n_classes):

                out_bands[cl].GetStatistics(0, 1)
                out_bands[cl].FlushCache()

        out_raster_object.close_all()

        out_raster_object = None

        if isinstance(self.mask_background, str):

            self._mask_background(self.out_image_temp, self.out_image, self.mask_background,
                                  self.background_band, self.background_value, self.minimum_observations,
                                  self.observation_band)

    def _mask_background(self, image2mask, masked_image, background_image, background_band,
                         background_value, minimum_observations, observation_band):

        """
        Recodes background values to zeros

        Args:
            image2mask (str): The image to mask.
            masked_image (str): The output masked image.
            background_image (str): The image with null background values.
            background_band (int): The band from ``background_image`` to use for null background value.
            background_value (Optional[int]): The null background value. Default is 0.
            minimum_observations (Optional[int]): A minimum number of observations to be recoded to 0.
                Default is 0, or no minimum observation filter.
            observation_band (Optional[int]): The band position of the observation counts, which is
                expected to be in ``background_image``. Default is 0.

        Returns:
            None, writes to ``masked_image``.
        """

        m_info = raster_tools.rinfo(image2mask)
        b_info = raster_tools.rinfo(background_image)

        m_info.get_band(1)
        m_info.storage = 'byte'

        out_rst_object = raster_tools.create_raster(masked_image, m_info, compress='none', tile=False)

        out_rst_object.get_band(1)

        b_rows, b_cols = m_info.rows, m_info.cols

        block_rows, block_cols = raster_tools.block_dimensions(b_rows, b_cols,
                                                               row_block_size=self.row_block_size,
                                                               col_block_size=self.col_block_size)

        for i in xrange(0, b_rows, block_rows):

            n_rows = self._num_rows_cols(i, block_rows, b_rows)

            for j in xrange(0, b_cols, block_cols):

                n_cols = self._num_rows_cols(j, block_cols, b_cols)

                m_array = m_info.mparray(i=i, j=j,
                                         rows=n_rows, cols=n_cols,
                                         d_type='byte')

                # Get the background array.
                b_array = raster_tools.mparray(i_info=b_info,
                                               bands2open=background_band,
                                               i=i, j=j,
                                               rows=n_rows, cols=n_cols,
                                               d_type='byte')

                m_array[b_array == background_value] = 0

                if minimum_observations > 0:

                    # Get the observation counts array.
                    observation_array = raster_tools.mparray(i_info=b_info,
                                                             bands2open=observation_band,
                                                             i=i, j=j,
                                                             rows=n_rows, cols=n_cols,
                                                             d_type='byte')

                    m_array[observation_array < minimum_observations] = 0

                out_rst_object.write_array(m_array, i, j)

        m_info.close()
        b_info.close()

        out_rst_object.close_all()

        out_rst_object = None

        os.remove(image2mask)

    def _num_rows_cols(self, pix, rows_cols, samp_in):

        return rows_cols if (pix + rows_cols < samp_in) else samp_in - pix

    def _get_feas(self, img_obj_list, i, j, n_rows, n_cols):

        if self.use_xy:

            self._create_indices(i, j, n_rows, n_cols)

            feature_arrays = [self.x_coordinates, self.y_coordinates]

        else:

            feature_arrays = []

        # for bd in xrange(0, self.i_info.bands):
        for iol in img_obj_list:

            if iol[-1]:

                __, __, start_j, start_i = vector_tools.get_xy_offsets(image_info=self.i_info, xy_info=iol[1])

            else:

                start_j, start_i = 0, 0

            # print start_j, start_i
            # sys.exit()

            # if iol[3] > self.i_info.cellY:
            #
            #     n_cols_coarse = int((n_cols * self.i_info.cellY) / iol[3])
            #     n_rows_coarse = int((n_rows * self.i_info.cellY) / iol[3])
            #
            #     coarse_array = iol[0].ReadAsArray(iol[1]+j, [2]+i, n_cols_coarse, n_rows_coarse).astype(np.float32)
            #
            #     row_zoom_factor = n_rows / float(n_rows_coarse)
            #     col_zoom_factor = n_cols / float(n_cols_coarse)
            #
            #     feature_array = zoom(coarse_array, (row_zoom_factor, col_zoom_factor), order=2)
            #
            # else:

            feature_arrays.append(iol[0].ReadAsArray(start_j+j, start_i+i, n_cols, n_rows).astype(np.float32))

        return np.vstack(feature_arrays).reshape(self.i_info.bands, n_rows, n_cols)

        # return np.vstack([img_obj_list[bd][0].ReadAsArray(img_obj_list[bd][1]+j, img_obj_list[bd][2]+i,
        #                                                   n_cols, n_rows).astype(np.float32) for bd in
        #                   xrange(0, self.i_info.bands)]).reshape(self.i_info.bands, n_rows, n_cols)

    def _create_indices(self, i, j, n_rows, n_cols):

        left = self.i_info.left + (j * self.i_info.cellY)
        top = self.i_info.top - (i * self.i_info.cellY)

        # create the longitudes
        lons_line = np.arange(left, left + (n_cols*self.i_info.cellY), self.i_info.cellY)

        if lons_line.shape[0] > n_cols:
            lons_line = lons_line[:n_cols]

        lons_line = lons_line.reshape(1, n_cols)

        self.x_coordinates = np.repeat(lons_line, n_rows, axis=0).astype(np.float32)

        # create latitudes
        lats_line = np.arange(top - (n_rows*self.i_info.cellY), top, self.i_info.cellY)[::-1]

        if lats_line.shape[0] > n_rows:
            lats_line = lats_line[:n_rows]

        lats_line = lats_line.reshape(n_rows, 1)

        self.y_coordinates = np.repeat(lats_line, n_cols, axis=1).astype(np.float32)

    def _get_slope(self, elevation_array, pad=50):

        elevation_array = cv2.copyMakeBorder(elevation_array, pad, pad, pad, pad, cv2.BORDER_REFLECT)

        x_grad, y_grad = np.gradient(elevation_array)

        return (np.pi / 2.) - np.arctan(np.sqrt((x_grad * x_grad) + (y_grad * y_grad)))

    def test_accuracy(self, out_acc=None, discrete=True, be_quiet=False):

        """
        Tests the accuracy of a model (a model must be trained or loaded).

        Args:
            out_acc (str): The output name of the accuracy text file.
            discrete (Optional[bool]): Whether the accuracy should assume discrete data.
                Otherwise, assumes continuous. Default is True.
            be_quiet (Optional[bool]): Whether to be quiet and do not print to screen. Default is False.

        Returns:
            None, writes to ``out_acc`` if given, and prints results to screen.

        Examples:
            >>> # get test accuracy
            >>> cl.test_accuracy('/out_accuracy.txt')
            >>> print cl.emat.accuracy
        """

        if self.classifier_info['classifier'] == 'CVMLP':

            test_labs_pred = np.empty((self.p_vars_test_rows, self.n_classes), dtype='uint8')
            self.model.predict(self.p_vars_test, test_labs_pred)
            test_labs_pred = np.argmax(test_labs_pred, axis=1)

        elif 'CV' in self.classifier_info['classifier']:

            if (0 < self.perc_samp_each < 1) or ((self.perc_samp_each == 0) and (0 < self.perc_samp < 1)):
                __, test_labs_pred = self.model.predict(self.p_vars_test)
            else:
                __, test_labs_pred = self.model.predict(self.p_vars)

        elif self.classifier_info['classifier'] in ['C5', 'Cubist']:

            if (0 < self.perc_samp_each < 1) or ((self.perc_samp_each == 0) and (0 < self.perc_samp < 1)):

                features = ro.r.matrix(self.p_vars_test, nrow=self.p_vars_test.shape[0],
                                       ncol=self.p_vars_test.shape[1])
            else:

                features = ro.r.matrix(self.p_vars, nrow=self.p_vars.shape[0],
                                       ncol=self.p_vars.shape[1])

            features.colnames = StrVector(self.headers[:-1])

            test_labs_pred = _do_c5_cubist_predict(self.model, self.classifier_info['classifier'],
                                                   features)

        else:

            if (0 < self.perc_samp_each < 1) or ((self.perc_samp_each == 0) and (0 < self.perc_samp < 1)):
                test_labs_pred = self.model.predict(self.p_vars_test)
            else:

                # Test the train variables if no test variables exist.
                test_labs_pred = self.model.predict(self.p_vars)

        if (0 < self.perc_samp_each < 1) or ((self.perc_samp_each == 0) and (0 < self.perc_samp < 1)):

            if discrete:
                self.test_array = np.uint8(np.c_[test_labs_pred, self.labels_test])
            else:
                self.test_array = np.float32(np.c_[test_labs_pred, self.labels_test])

        else:

            if discrete:
                self.test_array = np.uint8(np.c_[test_labs_pred, self.labels])
            else:
                self.test_array = np.float32(np.c_[test_labs_pred, self.labels])

        if not be_quiet:
            print '\nGetting test accuracy ...\n'

        self.emat = error_matrix()
        self.emat.get_stats(po_array=self.test_array, discrete=discrete)

        if out_acc:
            self.emat.write_stats(out_acc)

    def recursive_elimination(self, method='RF', perc_samp_each=.5):

        """
        Recursively eliminates features.

        Args:
            method (Optional[str]): The method to use. Default is 'RF'. Choices are ['RF', 'chi2'].
            perc_samp_each (Optional[float]): The percentage to sample at each iteration. Default is .5. 

        Returns:
            None, plots results.

        Examples:
            >>> from mappy.classifiers import classification
            >>> cl = classification()
            >>>
            >>> cl.split_samples('/samples.txt', perc_samp=1.)
            >>> cl.construct_model(classifier_info={'classifier': 'RF', 'trees': 500})
            >>> cl.recursive_elimination()
        """

        if method == 'chi2':

            p_vars = np.copy(self.p_vars)

            # Scale negative values to positive (for Chi Squared)
            for var_col_pos in xrange(0, p_vars.shape[1]):

                col_min = p_vars[:, var_col_pos].min()

                if col_min < 0:
                    p_vars[:, var_col_pos] = np.add(p_vars[:, var_col_pos], abs(col_min))

            try:
                feas_ranked, p_val = chi2(p_vars, self.labels)
            except NameError:
                sys.exit('\nERROR!! Be sure to run split_samples() to create the predictors and predictees.\n')

        elif method == 'RF':

            try:
                feas_ranked = self.model.feature_importances_
            except NameError:
                sys.exit('\nERROR!! A RF model must be trained to use RF feature importance.\n')

        loop_len = len(feas_ranked) + 1

        feas_ranked[np.isnan(feas_ranked)] = 0.

        self.fea_rank = {}

        for i in xrange(1, loop_len):

            self.fea_rank[i] = feas_ranked[i-1]

        indices = []
        indice_counts = []
        accuracy_scores = []

        for i, s in enumerate(sorted(self.fea_rank, key=self.fea_rank.get)):

            if (len(self.fea_rank) - i) <= (len(self.classes) * 2):

                break

            else:

                indices.append(s)

                self.split_samples(self.file_name, perc_samp_each=perc_samp_each, ignore_feas=indices,
                                   use_xy=self.use_xy)

                print '{:d} features ...'.format(self.n_feas)

                self.construct_model(classifier_info=self.classifier_info)

                self.test_accuracy()

                # print 'Overall accuracy: {:.2f}'.format(self.emat.accuracy)
                # print 'Kappa score: {:.2f}\n'.format(self.emat.kappa_score)

                indice_counts.append(len(indices))
                accuracy_scores.append(self.emat.accuracy)

        accuracy_scores_sm = [sum(accuracy_scores[r:r+3]) / 3. for r in xrange(0, len(accuracy_scores)-2)]
        accuracy_scores_sm.insert(0, sum(accuracy_scores_sm[:2]) / 2.)
        accuracy_scores_sm.append(sum(accuracy_scores_sm[-2:]) / 2.)

        plt.plot(indice_counts, accuracy_scores_sm)
        plt.xlabel('Number of features removed')
        plt.ylabel('Overall accuracy')
        plt.show()

        plt.close()

    def rank_feas(self, rank_text=None, rank_method='chi2', top_feas=1., be_quiet=False):

        """
        Ranks image features by importance.

        Args:
            rank_text (Optional[str]): A text file to write ranked features to. Default is None.
            rank_method (Optional[str]): The method to use for feature ranking. Default is 'chi2' (Chi^2). Choices are 
                ['chi2', 'RF'].
            top_feas (Optional[float or int]): The percentage or total number of features to reduce to. 
                Default is 1., or no reduction.
            be_quiet (Optional[bool]): Whether to be quiet and do not print to screen. Default is False.

        Returns:
            None, writes to ``rank_text`` if given and prints results to screen.

        Examples:
            >>> # rank image features
            >>> cl.split_samples('/samples.txt', scale_data=True)
            >>> cl.rank_feas(rank_text='/ranked_feas.txt',
            >>>              rank_method='chi2', top_feas=.2)
            >>> print cl.fea_rank
            >>>
            >>> # a RF model must be trained before feature ranking
            >>> cl.construct_model()
            >>> cl.rank_feas(rank_method='RF', top_feas=.5)
        """

        if isinstance(rank_text, str):

            d_name, f_name = os.path.split(rank_text)

            if not os.path.isdir(d_name):
                os.makedirs(d_name)

            rank_txt_wr = open(rank_text, 'wb')

        if rank_method == 'chi2':

            p_vars = np.copy(self.p_vars)

            # Scale negative values to positive (for Chi Squared)
            for var_col_pos in xrange(0, p_vars.shape[1]):

                col_min = p_vars[:, var_col_pos].min()

                if col_min < 0:
                    p_vars[:, var_col_pos] = np.add(p_vars[:, var_col_pos], abs(col_min))

            try:
                feas_ranked, p_val = chi2(p_vars, self.labels)
            except NameError:
                sys.exit('\nERROR!! Be sure to run split_samples() to create the predictors and predictees.\n')

            loop_len = len(feas_ranked) + 1

        elif rank_method == 'RF':

            try:
                feas_ranked = self.model.feature_importances_
            except NameError:
                sys.exit('\nERROR!! A RF model must be trained to use RF feature importance.\n')

            loop_len = len(feas_ranked) + 1

        feas_ranked[np.isnan(feas_ranked)] = 0.

        self.fea_rank = {}
        for i in xrange(1, loop_len):

            self.fea_rank[i] = feas_ranked[i-1]

        if rank_method == 'chi2':
            title = '**********************\n*                    *\n* Chi^2 Feature Rank *\n*                    *\n**********************\n\nRank      Variable      Value\n----      --------      -----'
        elif rank_method == 'RF':
            title = '************************************\n*                                  *\n* Random Forest Feature Importance *\n*                                  *\n************************************\n\nRank      Variable      Value\n----      --------      -----'

        if not be_quiet:
            print title

        if isinstance(rank_text, str):
            rank_txt_wr.write('%s\n' % title)

        if isinstance(top_feas, float):
            n_best_feas = int(top_feas * len(self.fea_rank))
        elif isinstance(top_feas, int):
            n_best_feas = copy(top_feas)

        r = 1
        self.bad_features = []

        for s in sorted(self.fea_rank, key=self.fea_rank.get, reverse=True):

            if r <= n_best_feas:

                if r < 10 and s < 10:
                    ranks = '%d         %d             %s' % (r, s, str(self.fea_rank[s]))
                elif r >= 10 and s < 10:
                    ranks = '%d        %d             %s' % (r, s, str(self.fea_rank[s]))
                elif r < 10 and s >= 10:
                    ranks = '%d         %d            %s' % (r, s, str(self.fea_rank[s]))
                else:
                    ranks = '%d        %d            %s' % (r, s, str(self.fea_rank[s]))

                if not be_quiet:
                    print ranks

                if isinstance(rank_text, str):
                    rank_txt_wr.write('%s\n' % ranks)

            else:
                # append excluded variables and remove from the "good" variables
                self.bad_features.append(s)

                del self.fea_rank[s]

            r += 1

        self.ranked_feas = np.array(sorted(self.fea_rank, key=self.fea_rank.get, reverse=True))

        if not be_quiet:

            print '\nMean score:  %.2f' % np.average([v for k, v in self.fea_rank.iteritems()])

            print '\n=================='
            print 'Excluded variables'
            print '=================='
            print ','.join(map(str, sorted(self.bad_features)))
            print

        if isinstance(rank_text, str):

            rank_txt_wr.write('\n==================\n')
            rank_txt_wr.write('Excluded variables\n')
            rank_txt_wr.write('==================\n')
            rank_txt_wr.write(','.join([str(bf) for bf in sorted(self.bad_features)]))
            rank_txt_wr.close()

    def add_variable_names(self, layer_names, stat_names, additional_features=[]):

        """
        Adds band-stat name pairs.

        Args:
            layer_names (list): A list of layer names.
            stat_names (list): A list of statistics names.
            additional_features (Optional[list]): Additional features. Default is [].

        Examples:
            >>> from mappy.classifiers import classification
            >>> cl = classification()
            >>>
            >>> cl.add_variable_names(['NDVI', 'EVI2', 'GNDVI', 'NDWI', 'NDBaI'],
            >>>                       ['min', 'max', 'median', 'cv', 'jd', 'slopemx', 'slopemn'],
            >>>                       additional_features=['x', 'y'])
            >>>
            >>> # get the 10th variable
            >>> cl.variable_names[10]
        """

        counter = 1
        self.variable_names = {}

        for layer_name in layer_names:

            for stat_name in stat_names:

                self.variable_names[counter] = '{} {}'.format(layer_name, stat_name)

                counter += 1

        if additional_features:

            for additional_feature in additional_features:

                self.variable_names[counter] = additional_feature

                counter += 1

        for k, v in self.variable_names.iteritems():

            print k, v

    def sub_feas(self, input_image, out_img, band_list=[]):

        """
        Subsets features. 

        Args:
            input_image (str): Full path, name, and extension of a single image.
            out_img (str): The output image.
            band_list (Optional[list]): A list of bands to subset. Default is []. If empty, ``sub_feas`` subsets 
                the top n best features returned by ``rank_feas``.

        Returns:
            None, writes to ``out_img``.

        Examples:
            >>> from mappy.classifiers import classification
            >>> cl = classification()
            >>>
            >>> cl.split_samples('/samples.txt', scale_data=True)
            >>> cl.rank_feas(rank_method='chi2', top_feas=.2)
            >>>
            >>> # apply feature rank to subset a feature image using cl.ranked_feas
            >>> cl.sub_feas('/in_image.vrt', '/ranked_feas.vrt')
        """

        if not hasattr(self, 'ranked_feas'):
            sys.exit('\nERROR!! The features need to be ranked first. See <rank_feas> method.\n')

        if not isinstance(input_image, str):
            sys.exit('\nERROR!! The input image needs to be specified in order set the extent.\n')

        if not os.path.isfile(input_image):
            sys.exit('\nERROR!! %s does not exist.\n' % input_image)

        d_name, f_name = os.path.split(out_img)
        f_base, f_ext = os.path.splitext(f_name)

        if not os.path.isdir(d_name):
            os.makedirs(d_name)

        if 'vrt' not in f_ext.lower():
            out_img_orig = copy(out_img)
            out_img = '%s/%s.vrt' % (d_name, f_base)

        if not band_list:
            # create the band list
            band_list = ''
            for fea_idx in self.ranked_feas:
                band_list = '%s-b %d ' % (band_list, fea_idx)

        print '\nSubsetting ranked features ...\n'

        com = 'gdalbuildvrt %s %s %s' % (band_list, out_img, input_image)

        try:
            subprocess.call(com, shell=True)
        except:

            com = r'%s/helpers/%s/apps/%s' % (os.path.realpath('..'), gdal_path, com)
            subprocess.call(com, shell=True)

        if 'tif' in f_ext.lower():

            com = 'gdal_translate --config GDAL_CACHEMAX 256 -of GTiff -co TILED=YES -co COMPRESS=LZW %s %s' % \
                  (out_img, out_img_orig)

            try:
                subprocess.call(com, shell=True)
            except:

                com = r'%s/helpers/%s/apps/%s' % (os.path.realpath('..'), gdal_path, com)
                subprocess.call(com, shell=True)

        elif 'img' in f_ext.lower():

            com = 'gdal_translate --config GDAL_CACHEMAX 256 -of HFA -co COMPRESS=YES %s %s' % \
                  (out_img, out_img_orig)

            try:
                subprocess.call(com, shell=True)
            except:

                com = r'%s/helpers/%s/apps/%s' % (os.path.realpath('..'), gdal_path, com)
                subprocess.call(com, shell=True)

    def grid_search(self, classifier_name, classifier_parameters, file_name, k_folds=3,
                    perc_samp=.5, ignore_feas=[], use_xy=False, classes2remove=[],
                    method='overall', metric='accuracy', f1_class=0, stratified=False, spacing=1000.,
                    output_file=None, calibrate_proba=False):

        """
        Classifier parameter grid search

        Args:
            classifier_name (str): The classifier to optimize.
            classifier_parameters (dict): The classifier parameters.
            file_name (str): The sample file name.
            k_folds (Optional[int]): The number of cross-validation folds. Default is 3.
            perc_samp (Optional[float]): The percentage of samples to take at each fold. Default is .5.
            ignore_feas (Optional[int list]): A list of features to ignore. Default is [].
            use_xy (Optional[bool]): Whether to use x, y coordinates. Default is False.
            classes2remove (Optional[int list]): A list of classes to remove. Default is [].
            method (Optional[str]): The score method to use, 'overall' (default) or 'f1'. Choices are ['overall, 'f1'].
            metric (Optinoal[str]): The scoring metric to use. Default is 'accuracy'.
                Choices are ['accuracy', 'r_squared', 'rmse', 'mae', 'medae', 'mse'].
            f1_class (Optional[int]): The class position to evaluate when ``method`` is equal to 'f1'. Default is 0,
                or first index position.
            stratified (Optional[bool]):
            spacing (Optional[float]):
            output_file (Optional[str]):

        Returns:
            DataFrame with scores.
        """

        regressors = ['Cubist', 'RFR', 'ABR', 'BagR', 'EX_RFR', 'EX_DTR', 'DTR']

        if metric not in ['accuracy', 'r_squared', 'rmse', 'mae', 'medae', 'mse']:
            raise NameError('The metric is not supported.')

        if classifier_name in regressors and metric == 'accuracy':
            raise NameError('Overall accuracy is not supported with regression classifiers.')

        if classifier_name not in regressors and metric in ['r_squared', 'rmse', 'mae', 'medae', 'mse']:
            raise NameError('Overall accuracy is the only option with discrete classifiers.')

        if classifier_name in ['C5', 'Cubist']:

            if 'R_installed' not in globals():
                raise NameError('You must use `classification_r` to use C5 and Cubist.')

            if not R_installed:
                raise ImportError('R and rpy2 must be installed to use C5 or Cubist.')

        if classifier_name in regressors:
            discrete = False
        else:
            discrete = True

        score_label = metric.upper()

        param_order = classifier_parameters.keys()

        df_param_headers = '-'.join(param_order)
        df_fold_headers = ('F' + '-F'.join(map(str, range(1, k_folds + 1)))).split('-')

        # Setup the output scores table.
        df = pd.DataFrame(columns=df_fold_headers)
        df[df_param_headers] = list(itertools.product(*classifier_parameters.values()))

        # Open the weights file.
        lc_weights = file_name.replace('.txt', '_w.txt')

        if os.path.isfile(lc_weights):
            weights = pickle.load(open(lc_weights, 'r'))
        else:
            weights = None

        for k_fold in xrange(1, k_folds + 1):

            print 'Fold {:d} of {:d} ...'.format(k_fold, k_folds)

            self.split_samples(file_name, perc_samp_each=perc_samp, ignore_feas=ignore_feas,
                               use_xy=use_xy, classes2remove=classes2remove, stratified=stratified,
                               spacing=spacing, sample_weight=weights)

            if classifier_name in ['C5', 'Cubist']:

                predict_samps = ro.r.matrix(self.p_vars, nrow=self.n_samps, ncol=self.n_feas)
                predict_samps.colnames = StrVector(self.headers[:-1])

            # Iteratve over all possible combinations.
            for param_combo in list(itertools.product(*classifier_parameters.values())):

                # Set the current parameters.
                current_combo = dict(zip(param_order, param_combo))

                # Add the classifier name to the dictionary.
                current_combo['classifier'] = classifier_name

                if classifier_name in ['C5', 'Cubist']:
                    self.construct_r_model(classifier_info=current_combo)
                else:
                    self.construct_model(classifier_info=current_combo,
                                         calibrate_proba=calibrate_proba)

                # Get the accuracy
                self.test_accuracy(discrete=discrete)

                if method == 'overall':
                    df.loc[df[df_param_headers] == param_combo, 'F{:d}'.format(k_fold)] = getattr(self.emat, metric)

                elif method == 'f1':
                    df.loc[df[df_param_headers] == param_combo, 'F{:d}'.format(k_fold)] = self.emat.f_scores[f1_class]

        df[score_label] = df[df_fold_headers].mean(axis=1)

        if metric in ['accuracy', 'r_squared']:
            best_score_index = np.argmax(df[score_label].values)
        else:
            best_score_index = np.argmin(df[score_label].values)

        print '\nBest {} score: {:f}'.format(metric, df[score_label].values[best_score_index])

        print '\nBest parameters:\n'
        print ''.join(['='] * len(df_param_headers))
        print df_param_headers
        print ''.join(['=']*len(df_param_headers))
        print df[df_param_headers].values[best_score_index]

        if isinstance(output_file, str):
            df.to_csv(output_file, sep=',', index=False)

        return df

    def optimize_parameters(self, file_name, classifier_info={'classifier': 'RF'},
                            n_trees_list=[500, 1000, 1500, 2000], trials_list=[2, 5, 10],
                            max_depth_list=[25, 30, 35, 40, 45, 50], min_samps_list=[2, 5, 10],
                            criterion_list=['gini'], rand_vars_list=['sqrt'],
                            cf_list=[.25, .5, .75], committees_list=[1, 2, 5, 10],
                            rules_list=[25, 50, 100, 500], extrapolation_list=[0, 1, 5, 10],
                            class_weight_list=[None, 'balanced', 'balanced_subsample'],
                            learn_rate_list=[.1, .2, .4, .6, .8, 1.],
                            bool_list=[True, False], c_list=[1., 10., 20., 100.],
                            gamma_list=[.001, .001, .01, .1, 1., 5.],
                            k_folds=3, perc_samp=.5, ignore_feas=[], use_xy=False, classes2remove=[],
                            method='overall', f1_class=0, stratified=False, spacing=1000.,
                            calibrate_proba=False, output_file=None):

        """
        Finds the optimal parameters for a classifier by training and testing a range of classifier parameters
        by n-folds cross-validation.

        Args:
            file_name (str): The file name of the samples.
            classifier_info (Optional[dict]): The model parameters dictionary. Default is {'classifier': 'RF'}.
            n_trees_list (Optional[int list]): A list of trees. Default is [500, 1000].
            trials_list (Optional[int list]): A list of boosting trials. Default is [5, 10, 20].
            max_depth_list (Optional[int list]): A list of maximum depths. Default is [5, 10, 20, 25, 30, 50].
            min_samps_list (Optional[int list]): A list of minimum samples. Default is [2, 5, 10].
            criterion_list (Optional[str list]): A list of RF criterion. Default is ['gini', 'entropy'].
            rand_vars_list (Optional[str list]): A list of random variables. Default is ['sqrt'].
            class_weight_list (Optional[bool]): A list of class weights.
                Default is [None, 'balanced', 'balanced_subsample'].
            c_list (Optional[float list]): A list of SVM C parameters. Default is [1., 10., 20., 100.].
            gamma_list (Optional[float list]): A list of SVM gamma parameters. Default is [.001, .001, .01, .1, 1., 5.].
            k_folds (Optional[int]): The number of N cross-validation folds. Default is 3.
            ignore_feas (Optional[int list]): A list of features to ignore. Default is [].
            use_xy (Optional[bool]): Whether to use x, y coordinates. Default is False.
            classes2remove (Optional[int list]): A list of classes to remove. Default is [].
            method (Optional[str]): The score method to use, 'overall' (default) or 'f1'.1
            f1_class (Optional[int]): The class position to evaluate when ``method`` is equal to 'f1'. Default is 0,
                or first index position.
            stratified (Optional[bool]):
            spacing (Optional[float]):
            output_file (Optional[str]):

        Returns:
            `Pandas DataFrame` when classifier_info['classifier'] == 'C5',
                otherwise None, prints results to screen.

        Examples:
            >>> # find the optimal parameters (max depth, min samps, trees)
            >>> # randomly sampling 50% (with replacement) and testing on the 50% set aside
            >>> # repeat 5 (k_folds) times and average the results
            >>> from mappy.classifiers import classification
            >>> cl = classification()
            >>>
            >>> # Find the optimum parameters for an Extremely Randomized Forest.
            >>> cl.optimize_parameters('/samples.txt',
            >>>                        classifier_info={'classifier': 'EX_RF'},
            >>>                        use_xy=True)
            >>>
            >>> # Find the optimum parameters for a Random Forest, but assess
            >>> #   only one class (1st class position) of interest.
            >>> cl.optimize_parameters('/samples.txt',
            >>>                        classifier_info={'classifier': 'RF'},
            >>>                        use_xy=True, method='f1', f1_class=0)
            >>>
            >>> # Optimizing C5 parameters
            >>> from mappy.classifiers import classification_r
            >>> cl = classification_r()
            >>>
            >>> df = cl.optimize_parameters('/samples.txt', classifier_info={'classifier': 'C5'},
            >>>                             trials_list=[2, 5, 10], cf_list=[.25, .5, .75],
            >>>                             min_samps_list=[2, 5, 10], bool_list=[True, False],
            >>>                             k_folds=5, stratified=True, spacing=50000.)
            >>>
            >>> print df
        """

        if classifier_info['classifier'] not in ['C5', 'Cubist']:

            self.split_samples(file_name, perc_samp=1., ignore_feas=ignore_feas,
                               use_xy=use_xy, classes2remove=classes2remove)

            prediction_models = {'RF': ensemble.RandomForestClassifier(n_jobs=-1),
                                 'RFR': ensemble.RandomForestRegressor(n_jobs=-1),
                                 'EX_RF': ensemble.ExtraTreesClassifier(n_jobs=-1),
                                 'EX_RRF': ensemble.ExtraTreesRegressor(n_jobs=-1),
                                 'Bag': ensemble.BaggingClassifier(base_estimator=tree.DecisionTreeClassifier(),
                                                                   n_jobs=-1),
                                 'AB': ensemble.AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier()),
                                 'GB': ensemble.GradientBoostingClassifier(),
                                 'DT': tree.DecisionTreeClassifier(),
                                 'SVM': SVC(kernel='rbf')}

        if classifier_info['classifier'] in ['RF', 'EX_RF']:

            parameters = {'criterion': criterion_list,
                          'n_estimators': n_trees_list,
                          'max_depth': max_depth_list,
                          'max_features': rand_vars_list,
                          'min_samples_split': min_samps_list,
                          'class_weight': class_weight_list}

        elif classifier_info['classifier'] in ['RFR', 'EX_RFR', 'DTR']:

            parameters = {'trees': n_trees_list,
                          'max_depth': max_depth_list,
                          'rand_vars': rand_vars_list,
                          'min_samps': min_samps_list}

        elif classifier_info['classifier'] in ['AB', 'AB_DT', 'AB_EX_DT', 'AB_RF', 'AB_EX_RF']:

            parameters = {'n_estimators': n_trees_list,
                          'trials': trials_list,
                          'learning_rate': learn_rate_list,
                          'max_depth': max_depth_list,
                          'min_samps': min_samps_list,
                          'class_weight': class_weight_list}

        elif classifier_info['classifier'] in ['ABR', 'ABR_EX_DTR']:

            parameters = {'trees': n_trees_list,
                          'rate': learn_rate_list}

        elif classifier_info['classifier'] == 'Bag':

            parameters = {'n_estimators': n_trees_list,
                          'warm_start': bool_list,
                          'bootstrap': bool_list,                          
                          'bootstrap_features': bool_list}

        elif classifier_info['classifier'] == 'BagR':

            parameters = {'trees': n_trees_list,
                          'warm_start': bool_list,
                          'bootstrap': bool_list,
                          'bootstrap_features': bool_list}

        elif classifier_info['classifier'] == 'GB':

            parameters = {'n_estimators': n_trees_list,
                          'max_depth': max_depth_list,
                          'max_features': rand_vars_list,
                          'min_samples_split': min_samps_list,
                          'learning_rate': learn_rate_list}

        elif classifier_info['classifier'] == 'GBR':

            parameters = {'trees': n_trees_list,
                          'max_depth': max_depth_list,
                          'rand_vars': rand_vars_list,
                          'min_samps': min_samps_list,
                          'learning_rate': learn_rate_list}

        elif classifier_info['classifier'] == 'DT':

            parameters = {'n_estimators': n_trees_list,
                          'max_depth': max_depth_list,
                          'max_features': rand_vars_list,
                          'min_samples_split': min_samps_list}

        elif classifier_info['classifier'] == 'SVM':

            parameters = {'C': c_list,
                          'gamma': gamma_list}

        elif classifier_info['classifier'] == 'C5':

            parameters = {'trials': trials_list,
                          'min_cases': min_samps_list,
                          'CF': cf_list,
                          'fuzzy': bool_list}

        elif classifier_info['classifier'] == 'Cubist':

            parameters = {'committees': committees_list,
                          'rules': rules_list,
                          'extrapolation': extrapolation_list,
                          'unbiased': bool_list}

        else:
            raise NameError('\nThe model cannot be optimized.\n')

        print '\nFinding the best paramaters for a {} model ...\n'.format(classifier_info['classifier'])

        core_classifiers = ['C5', 'Cubist', 'RF', 'RFR',
                            'AB', 'AB_RF', 'AB_EX_RF', 'AB_DT', 'AB_EX_DT',
                            'ABR', 'BagR', 'EX_RF', 'EX_RFR', 'EX_DTR', 'DTR']

        if classifier_info['classifier'] in core_classifiers:

            return self.grid_search(classifier_info['classifier'], parameters, file_name,
                                    k_folds=k_folds, perc_samp=perc_samp, ignore_feas=ignore_feas,
                                    use_xy=use_xy, classes2remove=classes2remove,
                                    method=method, f1_class=f1_class, stratified=stratified,
                                    spacing=spacing, output_file=output_file,
                                    calibrate_proba=calibrate_proba)

        elif (method == 'overall') and (classifier_info['classifier'] not in core_classifiers):

            clf = prediction_models[classifier_info['classifier']]

            grid_search = GridSearchCV(clf, param_grid=parameters, n_jobs=-1,
                                       cv=k_folds, verbose=1)

            grid_search.fit(self.p_vars, self.labels)

            print grid_search.best_estimator_
            print '\nBest score: {:f}'.format(grid_search.best_score_)
            print '\nBest parameters: {}\n'.format(grid_search.best_params_)

        else:
            raise NameError('The score method {} is not supported.'.format(method))

    def stack_majority(self, img, output_model, out_img, classifier_info, scale_data=False, ignore_feas=[]):

        """
        A majority vote filter.

        Args:
            img (str): The input image.
            output_model (str): The output model.
            out_img (str): The output map.
            classifier_info (dict): The model parameters dictionary.
            scale_data (Optional[bool or str]): Whether to scale the data prior to classification. 
                Default is False. *If ``scale_data`` is a string, the scaler will be loaded from the string text file.
            ignore_features (Optional[int list]): A list of features to ignore. Default is [].

        Returns:
            None, writes results to ``out_img``.

        Examples:
            >>> from mappy.classifiers import classification
            >>> cl = classification()
            >>>
            >>> cl.split_samples('/samples.txt', scale_data=True)
            >>>
            >>> # setup three classifiers
            >>> classifier_info = {'classifiers': ['RF', 'SVM', 'Bayes'], 'trees': 100, 'C': 1}
            >>> cl.stack_majority('/in_image.tif', '/out_model.xml', '/out_image.tif',
            >>>                   classifier_info, scale_data=True)
            >>>
            >>> # Command line
            >>> > ./classification.py -s /samples.txt -i /in_image.tif -mo /out_model.xml -o /out_image.tif --parameters ...
            >>>     classifiers:RF-SVM-Bayes,trees:100,C:1 --scale yes
        """

        d_name_mdl, f_name_mdl = os.path.split(output_model)
        f_base_mdl, f_ext_mdl = os.path.splitext(f_name_mdl)

        d_name, f_name = os.path.split(out_img)
        f_base, f_ext = os.path.splitext(f_name)

        map_list = []

        for classifier in classifier_info['classifiers']:

            output_model = '%s/%s_%s%s' % (d_name_mdl, f_base_mdl, classifier, f_ext_mdl)

            out_image_temp = '%s/%s_%s%s' % (d_name, f_base, classifier, f_ext)
            map_list.append(out_image_temp)

            classifier_info['classifier'] = classifier

            self.construct_model(output_model=output_model, classifier_info=classifier_info)

            # load the model for multiproccessing
            self.construct_model(input_model=output_model, classifier_info=classifier_info)

            self.predict(img, out_image_temp, scale_data=scale_data, ignore_feas=ignore_feas)

        i_info = raster_tools.rinfo(map_list[0])
        rows, cols = i_info.rows, i_info.cols

        i_info.bands = 1

        out_rst = raster_tools.create_raster(out_img, i_info, bigtiff='yes')

        out_rst.get_band(1)

        rst_objs = [raster_tools.rinfo(img).datasource.GetRasterBand(1) for img in map_list]

        if rows >= 512:
            blk_size_rows = 512
        else:
            blk_size_rows = copy(rows)

        if cols >= 1024:
            block_size_cls = 1024
        else:
            block_size_cls = copy(cols)

        for i in xrange(0, rows, blk_size_rows):

            n_rows = self._num_rows_cols(i, blk_size_rows, rows)

            for j in xrange(0, cols, block_size_cls):

                n_cols = self._num_rows_cols(j, block_size_cls, cols)

                mode_img = np.vstack(([obj.ReadAsArray(j, i, n_cols, n_rows)
                                       for obj in rst_objs])).reshape(len(map_list), n_rows, n_cols)

                out_mode = stats.mode(mode_img)[0]

                out_rst.write_array(out_mode, i=i, j=j)

        out_rst.close_all()


class classification_r(classification):

    """
    Class interface to R C5/Cubist

    Examples:
        >>> from mappy.classifiers import classification_r
        >>> cl = classification_r()
        >>>
        >>> # load the samples
        >>> # *Note that the sample instances are stored in cl.classification, 
        >>> # structurally different than using the base
        >>> # classification() which inherits the properties directly
        >>> cl.split_samples('/samples.txt', classes2remove=[4, 9],
        >>>                  class_subs={2:.9, 5:.1, 8:.9})
        >>>
        >>> # Train a Cubist model.
        >>> cl.construct_r_model(output_model='/models/cubist_model', classifier_info={'classifier': 'Cubist',
        >>>                        'committees': 5, 'rules': 100, 'extrap': 10})
        >>>
        >>> # Predict labels with the Cubist model.
        >>> cl.predict('/feas/image_feas.vrt', '/maps/out_labels.tif',
        >>>            input_model='/models/cubist_model', in_samps='/samples.txt')
        >>>
        >>> # Train a C5 model.
        >>> cl.construct_r_model(output_model='/models/c5_model', classifier_info={'classifier': 'C5',
        >>>                        'trials': 10, 'C5': .25, 'min': 2})
        >>>
        >>> # Predict labels with the C5 model. There is no need
        >>> #   to load a model if the prediction is applied within
        >>> #   the same session.
        >>> cl.predict('/feas/image_feas.vrt', '/maps/out_labels.tif')
        >>>
        >>> # However, you must provide the model
        >>> #   file to predict in parallel.
        >>> # First, load the model
        >>> cl.construct_r_model(input_model='/models/c5_model.tree')
        >>>
        >>> # Then apply the predictions.
        >>> cl.predict('/feas/image_feas.vrt', '/maps/out_labels.tif')
    """

    global R_installed, ro, Cubist, C50, pandas2ri, StrVector

    # rpy2
    try:
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
        import rpy2.robjects.packages as rpackages

        from rpy2.robjects.numpy2ri import numpy2ri
        ro.numpy2ri.activate()

        # R vector of strings
        from rpy2.robjects.vectors import StrVector

        from rpy2.robjects import pandas2ri
        pandas2ri.activate()

        # import R's utility package
        utils = rpackages.importr('utils')

        R_installed = True

    except:
        R_installed = False

    if R_installed:

        # select a mirror for R packages
        utils.chooseCRANmirror(ind=1)  # select the first mirror in the list

        # R package names
        package_names = ('Cubist', 'C50', 'raster', 'rgdal')#, 'foreach', 'doSNOW')

        # Selectively install what needs to be install.
        # We are fancy, just because we can.
        names_to_install = [x for x in package_names if not rpackages.isinstalled(x)]

        # Install necessary libraries.
        if len(names_to_install) > 0:

            print('Installing R packages--{} ...'.format(', '.join(names_to_install)))

            utils.install_packages(StrVector(names_to_install))

        # print('Importing R packages--{} ...'.format(', '.join(package_names)))

        # Cubist
        Cubist = importr('Cubist', suppress_messages=True)

        # C50
        C50 = importr('C50', suppress_messages=True)

        # raster
        raster = importr('raster', suppress_messages=True)

        # rgdal
        rgdal = importr('rgdal', suppress_messages=True)

        # # foreach
        # foreach = importr('foreach', suppress_messages=True)
        #
        # # doSNOW
        # doSNOW = importr('doSNOW', suppress_messages=True)

    def __init__(self):

        self.time_stamp = time.asctime(time.localtime(time.time()))

        self.OS_SYSTEM = platform.system()

    def construct_r_model(self, input_model=None, output_model=None, write_summary=False,
                          get_probs=False, cost_array=None, case_weights=None,
                          classifier_info={'classifier': 'Cubist'}):

        """
        Trains a Cubist model.

        Args:
            input_model (Optional[str]): The input model to load. Default is None.
            output_model (Optional[str]): The output model to write to file. Default is None.
                *No extension should be added. This is added automatically.
            write_summary (Optional[bool]): Whether to write the model summary to file. Default is False.
            get_probs (Optional[bool]): Whether to return class probabilities. Default is False.
            cost_array (Optional[2d array]): A cost matrix, where rows are the predicted costs and columns are
                the true costs. Default is None.

                In the example below, the cost of predicting R as G is 3x more costly as the reverse, predicting
                    R as B ix 7x more costly as the reverse, predicting G as R is 2x more costly as the reverse,
                    and so on.

                        R  G  B
                    R [[0, 2, 4],
                    G  [3, 0, 5],
                    B  [7, 1, 0]]

            case_weights (Optional[list or 1d array]): A list of case weights. Default is None.
            classifier_info (dict): The model parameter dictionary: Default is {'classifier': 'Cubist', 
                'committees': 5, 'rules': 100, 'extrap': 10})
        """

        if not R_installed:
            raise ImportError('R and rpy2 must be installed to use C5 or Cubist.')

        self.get_probs = get_probs

        # replace forward slashes for Windows
        if self.OS_SYSTEM == 'Windows':
            output_model = output_model.replace('\\', '/')

        if isinstance(input_model, str):
            self.classifier_info, self.model, self.headers = pickle.load(file(input_model, 'rb'))
            self.input_model = input_model
            return
        else:
            self.classifier_info = classifier_info

        # Check if model parameters are set
        #   otherwise, set defaults.
        if self.classifier_info['classifier'] == 'Cubist':

            # The number of committees.
            if 'committees' not in self.classifier_info:
                self.classifier_info['committees'] = 5

            # The number of rules.
            if 'rules' not in self.classifier_info:
                self.classifier_info['rules'] = 100

            # Whether to use unbiased rules.
            if 'unbiased' not in self.classifier_info:
                self.classifier_info['unbiased'] = False

            # The extrapolation percentage, between 0-100.
            if 'extrapolation' not in self.classifier_info:
                self.classifier_info['extrapolation'] = 10

        elif self.classifier_info['classifier'] == 'C5':

            # The number of boosted trials.
            if 'trials' not in self.classifier_info:
                self.classifier_info['trials'] = 10

            # The minimum number of cases and node level.
            if 'min_cases' not in self.classifier_info:
                self.classifier_info['min_cases'] = 2

            # Whether to apply winnowing (i.e., feature selection)
            if 'winnow' not in self.classifier_info:
                self.classifier_info['winnow'] = False

            # Whether to turn off global pruning
            if 'no_prune' not in self.classifier_info:
                self.classifier_info['no_prune'] = False

            # The confidence factor for pruning. Low values result
            #   in more pruning.]
            if 'CF' not in self.classifier_info:
                self.classifier_info['CF'] = .25

            # Whether to apply a fuzzy threshold of probabilities.
            if 'fuzzy' not in self.classifier_info:
                self.classifier_info['fuzzy'] = False

        else:
            raise NameError('\nThe classifier must be C5 or Cubist.\n')

        if isinstance(output_model, str):

            self.model_dir, self.model_base = os.path.split(output_model)
            self.output_model = '{}/{}.tree'.format(self.model_dir, self.model_base)

            if os.path.isfile(self.output_model):
                os.remove(self.output_model)

            if not os.path.isdir(self.model_dir):
                os.makedirs(self.model_dir)

            os.chdir(self.model_dir)

        ## prepare the predictive samples and labels
        # R('samps = read.csv(file="%s", head=TRUE, sep=",")' % file_name)
        # samps = R['read.csv'](file_name)

        # samps = com.convert_to_r_dataframe(pd.DataFrame(self.p_vars))
        # samps = pandas2ri.py2ri(pd.DataFrame(self.p_vars))
        # samps.colnames = self.headers[:-1]

        samps = ro.r.matrix(self.p_vars, nrow=self.n_samps, ncol=self.n_feas)
        samps.colnames = StrVector(self.headers[:-1])

        if 'Cubist' in self.classifier_info['classifier']:
            labels = ro.FloatVector(self.labels)
        elif 'C5' in self.classifier_info['classifier']:
            labels = ro.FactorVector(pd.Categorical(self.labels))

        if isinstance(case_weights, list) or isinstance(case_weights, np.ndarray):
            case_weights = ro.FloatVector(case_weights)

        if isinstance(cost_array, np.ndarray):

            cost_array = ro.r.matrix(cost_array, nrow=cost_array.shape[0], ncol=cost_array.shape[1])
            cost_array.rownames = StrVector(sorted(self.classes))
            cost_array.colnames = StrVector(sorted(self.classes))

        # samps = DataFrame.from_csvfile(self.file_name, header=True, sep=',')
        # R('labels = samps[,c("%s")]' % self.classification.hdrs[-1])
        # labels = samps.rx(True, ro.StrVector(tuple(self.headers[-1:])))
        # R('samps = samps[,1:%d+2]' % self.classification.n_feas)
        # samps = samps.rx(True, ro.IntVector(tuple(range(3, self.n_feas+3))))

        # Train a Cubist model.
        if 'Cubist' in self.classifier_info['classifier']:

            print '\nTraining a Cubist model with {:d} committees, {:d} rules, {:d}% extrapolation, and {:,d} samples ...\n'.format(self.classifier_info['committees'],
                                                                                                                                    self.classifier_info['rules'],
                                                                                                                                    self.classifier_info['extrapolation'],
                                                                                                                                    self.n_samps)

            # train the Cubist model
            # R('model = Cubist::cubist(x=samps, y=labels, committees=%d, control=cubistControl(rules=%d, extrapolation=%d))' % \
            #   (self.classifier_info['committees'], self.classifier_info['rules'], self.classifier_info['extrap']))

            self.model = Cubist.cubist(x=samps, y=labels, committees=self.classifier_info['committees'],
                                       control=Cubist.cubistControl(rules=self.classifier_info['rules'],
                                                                    extrapolation=self.classifier_info['extrapolation'],
                                                                    unbiased=self.classifier_info['unbiased']))

            if isinstance(output_model, str):

                print 'Writing the model to file ...'

                # Write the Cubist model and .names to file.
                if self.OS_SYSTEM == 'Windows':

                    # R('Cubist::exportCubistFiles(model, prefix="%s")' % self.model_base)
                    Cubist.exportCubistFiles(self.model, prefix=self.model_base)

                else:

                    pickle.dump([self.classifier_info, self.model, self.headers],
                                file(self.output_model, 'wb'),
                                protocol=pickle.HIGHEST_PROTOCOL)

                if write_summary:

                    print 'Writing the model summary to file ...'

                    # Write the Cubist model summary to file.
                    with open('{}/{}_summary.txt'.format(self.model_dir, self.model_base), 'wb') as out_tree:
                        out_tree.write(str(Cubist.print_summary_cubist(self.model)))

        elif 'C5' in self.classifier_info['classifier']:

            print '\nTraining a C5 model with {:d} trials, {:.2f} CF, {:d} minimum cases, and {:,d} samples ...\n'.format(self.classifier_info['trials'],
                                                                                                                          self.classifier_info['CF'],
                                                                                                                          self.classifier_info['min_cases'],
                                                                                                                          self.n_samps)

            # train the C5 model
            # R('model = C50::C5.0(x=samps, y=factor(labels), trials=%d, control=C5.0Control(CF=%f, minCases=%d))' % \
            #   (self.classifier_info['trials'], self.classifier_info['CF'], self.classifier_info['min']))

            # weights = case_weights,
            # costs = cost_array,

            self.model = C50.C5_0(x=samps, y=labels,
                                  trials=self.classifier_info['trials'],
                                  control=C50.C5_0Control(CF=self.classifier_info['CF'],
                                                          minCases=self.classifier_info['min_cases'],
                                                          winnow=self.classifier_info['winnow'],
                                                          noGlobalPruning=self.classifier_info['no_prune'],
                                                          fuzzyThreshold=self.classifier_info['fuzzy'],
                                                          label='response'))

            if isinstance(output_model, str):

                print 'Writing the model to file ...'

                # Write the C5 tree to file.
                if self.OS_SYSTEM == 'Windows':

                    with open(self.output_model, 'w') as out_tree:
                        ro.globalenv['model'] = self.model
                        out_tree.write(str(ro.r('model$tree')))

                else:

                    pickle.dump([self.classifier_info, self.model, self.headers],
                                file(self.output_model, 'wb'),
                                protocol=pickle.HIGHEST_PROTOCOL)

                if write_summary:

                    print 'Writing the model summary to file (this may take a few minutes with large trees) ...'

                    # Write the C5 model summary to file.
                    with open('{}/{}_summary.txt'.format(self.model_dir, self.model_base), 'wb') as out_imp:
                        out_imp.write(str(C50.print_summary_C5_0(self.model)))

    def predict_c5_cubist(self, input_image, out_image, input_model=None, in_samps=None,
                          ignore_feas=[], row_block_size=1024, col_block_size=1024,
                          mask_background=None, background_band=0, background_value=0,
                          minimum_observations=0, observation_band=0, n_jobs=-1, chunk_size=1024):

        """
        Predicts class labels from C5 or Cubist model.

        Args:
            input_image (str): The image features with the same number of layers as used to train the model.
            out_image (str): The output image.
            input_model (Optional[str]): The full directory and base name of the model to use.
            in_samps (Optional[str]): The image samples used to build the model.
                *This is necessary to match the header names with Windows.
            tree_model (str): The decision tree model to use. Default is 'Cubist'. Choices are ['C5' or 'Cubist'].
        """

        global predict_samps

        self.ignore_feas = ignore_feas
        self.row_block_size = row_block_size
        self.col_block_size = col_block_size
        self.mask_background = mask_background
        self.background_band = background_band
        self.background_value = background_value
        self.minimum_observations = minimum_observations
        self.observation_band = observation_band
        self.out_image = out_image
        self.n_jobs = n_jobs
        self.chunk_size = chunk_size

        # Block record keeping.
        d_name, f_name = os.path.split(self.out_image)
        f_base, f_ext = os.path.splitext(f_name)

        self.out_image_temp = '{}/{}_temp.tif'.format(d_name, f_base)
        self.record_keeping = '{}/{}_record.txt'.format(d_name, f_base)

        if os.path.isfile(self.record_keeping):
            self.record_list = pickle.load(file(self.record_keeping, 'rb'))
        else:
            self.record_list = []

        if isinstance(input_model, str):
            self.classifier_info, self.model, self.headers = pickle.load(file(input_model, 'rb'))

        if self.OS_SYSTEM == 'Windows':

            input_model = input_model.replace('\\', '/')
            in_samps = in_samps.replace('\\', '/')
            input_image = input_image.replace('\\', '/')
            out_image = out_image.replace('\\', '/')

            out_image_dir, f_name = os.path.split(out_image)
            out_image_base, f_ext = os.path.splitext(f_name)

            if not os.path.isdir(out_image_dir):
                os.makedirs(out_image_dir)

            if 'img' in f_ext.lower():
                out_type = 'HFA'
            elif 'tif' in f_ext.lower():
                out_type = 'GTiff'
            else:
                sys.exit('\nERROR!! The file extension is not supported.\n')

            self.model_dir, self.model_base = os.path.split(input_model)

            self.input_image = input_image

            # get the number of features
            self.i_info = raster_tools.rinfo(input_image)
            self.n_feas = self.i_info.bands

            # build the icases file
            self._build_icases(in_samps, tree_model)

            if 'C5' in tree_model:

                # write the .names and .data files to text
                self._build_C5_names(in_samps)

            # self.mapC5_dir = os.path.realpath('../helpers/mapC5')
            # python_home = 'C:/Python27/ArcGIS10.1/Lib/site-packages'
            self.mapC5_dir = '{}/helpers/mapC5'.format(MAPPY_PATH)

            # copy the mapC5 files to the model directory
            self._copy_mapC5(tree_model)

            # change to the map_C5 model directory
            os.chdir(self.model_dir)

            # execute mapC5
            if tree_model == 'Cubist':

                com = '{}\mapCubist_v202.exe {} {} {}\{}'.format(self.model_dir, self.model_base,
                                                                 out_type, out_image_dir, out_image_base)

            elif tree_model == 'C5':

                com = '{}\mapC5_v202.exe {} {} {}\{} {}\{}_error'.format(self.model_dir, self.model_base, out_type,
                                                                         out_image_dir, out_image_base, out_image_dir,
                                                                         out_image_base)

            try:
                subprocess.call(com, shell=True)
            except:

                com = r'{}/helpers/{}/apps/{}'.format(MAPPY_PATH, gdal_path, com)
                subprocess.call(com, shell=True)

            self._clean_mapC5(tree_model)

        else:

            # Open the image.
            self.i_info = raster_tools.rinfo(input_image)

            if self.ignore_feas:
                bands2open = sorted([bd for bd in xrange(1, self.i_info.bands + 1) if bd not in self.ignore_feas])
            else:
                bands2open = range(1, self.i_info.bands + 1)

            # Output image information.
            self.o_info = self.i_info.copy()

            # Set the number of output bands.
            self.o_info.bands = 1

            if self.classifier_info['classifier'] == 'Cubist':
                self.o_info.storage = 'float32'
            else:
                self.o_info.storage = 'byte'

            self.o_info.close()

            # Create the output image
            if isinstance(self.mask_background, str):
                out_raster_object = raster_tools.create_raster(self.out_image_temp, self.o_info,
                                                               compress='none', tile=False, bigtiff='yes')
            else:
                out_raster_object = raster_tools.create_raster(self.out_image, self.o_info, tile=False)

            out_raster_object.get_band(1)
            out_raster_object.fill(0)

            rows = self.i_info.rows
            cols = self.i_info.cols

            block_rows, block_cols = raster_tools.block_dimensions(rows, cols,
                                                                   row_block_size=self.row_block_size,
                                                                   col_block_size=self.col_block_size)

            n_blocks = 0
            for i in xrange(0, rows, block_rows):
                for j in xrange(0, cols, block_cols):
                    n_blocks += 1

            n_block = 1

            print '\nMapping labels ...\n'

            for i in xrange(0, rows, block_rows):

                n_rows = self._num_rows_cols(i, block_rows, rows)

                for j in xrange(0, cols, block_cols):

                    print 'Block {:d} of {:d} ...'.format(n_block, n_blocks)
                    n_block += 1

                    if n_block in self.record_list:
                        print '  Skipping current block ...'

                        continue

                    n_cols = self._num_rows_cols(j, block_cols, cols)

                    features = raster_tools.mparray(image2open=input_image,
                                                    bands2open=bands2open,
                                                    i=i, j=j,
                                                    rows=n_rows, cols=n_cols,
                                                    predictions=True,
                                                    d_type='float32',
                                                    n_jobs=-1)

                    # Load
                    predict_samps = pandas2ri.py2ri(pd.DataFrame(features))
                    predict_samps.colnames = self.headers

                    # Make the predictions and convert to
                    #   a Pandas Categorical object, followed
                    #   by a conversion to a NumPy array.

                    # Get chunks for parallel processing.
                    samp_rows = predict_samps.shape[0]
                    indice_pairs = []
                    for i_ in xrange(0, samp_rows, self.chunk_size):
                        n_rows_ = self._num_rows_cols(i_, self.chunk_size, samp_rows)
                        indice_pairs.append([i_, n_rows_])

                    if isinstance(self.input_model, str):

                        predicted = Parallel(n_jobs=self.n_jobs,
                                             max_nbytes=None)(delayed(self.c5_predict_parallel)(input_model, ip)
                                                              for ip in indice_pairs)

                        # Write the predictions to file.
                        out_raster_object.write_array(np.array(list(itertools.chain.from_iterable(predicted))).reshape(n_cols, n_rows).T, i, j)

                    else:

                        # Write the predictions to file.
                        out_raster_object.write_array(np.uint8(pandas2ri.ri2py(C50.predict_C5_0(self.model,
                                                                                                newdata=predict_samps))).reshape(n_cols, n_rows).T, i, j)

                    self.record_list.append(n_block)

            out_raster_object.close_all()

            out_raster_object = None

            if isinstance(self.mask_background, str):

                self._mask_background(self.out_image_temp, self.out_image, self.mask_background,
                                      self.background_band, self.background_value, self.minimum_observations,
                                      self.observation_band)

            # ro.r('x = new("GDALReadOnlyDataset", "{}")'.format(input_image))

            # TODO: R predict functionality
            # print(R('names(samps)'))

            # R('x = new("GDALReadOnlyDataset", "%s")' % input_image)
            # R('feas = data.frame(getRasterTable(x))')
            # R('names(feas) = c("x", "y", names(samps))')
            # R('feas = feas[1:%d+2]' % n_feas)
            # R('feas = stack("%s")' % input_image)
            # R('predict(feas, model, filename="%s", format="GTiff", datetype="INT1U", progress="window", package="raster")' % out_img)

            # print(R('names(feas)'))

            # R('predict(feas, fit, filename="%s", format="GTiff", datetype="INT1U", progress="window")' % out_img)

    # def c5_predict_parallel(self, input_model, ip):
    #
    #     ci, m, h = pickle.load(file(input_model, 'rb'))
    #
    #     return np.uint8(pandas2ri.ri2py(C50.predict_C5_0(m, newdata=predict_samps[ip[0]:ip[0]+ip[1]])))

    def _build_icases(self, in_samps, tree_model):

        """
        Creates the icases file needed to run mapC5

        Args:
            in_samps (str): The samples used to train the model.
            tree_model (str): 'C5' or 'Cubist'
        """

        icases_txt = '{}/{}.icases'.format(self.model_dir, self.model_base)

        # the output icases file
        if os.path.isfile(icases_txt):
            os.remove(icases_txt)

        icases = open(icases_txt, 'w')

        if 'Cubist' in tree_model:
            icases.write('{} ignore 1\n'.format(self.headers[-1]))
        elif 'C5' in tree_model:
            icases.write('X ignore 1\n')
            icases.write('Y ignore 1\n')

        bd = 1
        for hdr in self.headers[2:-1]:

            icases.write('{} {} {:d}\n'.format(hdr, self.input_image, bd))

            bd += 1

        icases.close()

    def _build_C5_names(self, in_samps):

        """
        Builds the C5 .names file.

        Args:
            in_samps (str): The samples used to train the model.
        """

        names_txt = '{}/{}.names'.format(self.model_dir, self.model_base)
        data_txt = '{}/{}.data'.format(self.model_dir, self.model_base)

        # the output .names file
        if os.path.isfile(names_txt):
            os.remove(names_txt)

        if os.path.isfile(data_txt):
            os.remove(data_txt)

        # create the .data file
        shutil.copy2(in_samps, data_txt)

        names = open(names_txt, 'w')

        names.write('{}.\n\n'.format(self.headers[-1]))
        names.write('X: ignore.\n')
        names.write('Y: ignore.\n')

        for hdr in self.headers[2:-1]:

            names.write('{}: continuous.\n'.format(hdr))

        # write the classes
        class_str_list = ','.join(map(str, sorted(self.classes)))
        names.write('{}: {}'.format(self.headers[-1], class_str_list))

        names.close()

    def _copy_mapC5(self, tree_model):

        """
        Copies files needed to run mapC5.

        Args:
            tree_model (str): The decision tree model to use ('C5' or 'Cubist').
        """

        if tree_model == 'Cubist':

            mapC5_list = ['gdal13.dll', 'gdal15.dll', 'install.bat', 'mapCubist_v202.exe', 'msvcp71.dll', 'msvcr71.dll']

        elif tree_model == 'C5':

            mapC5_list = ['gdal13.dll', 'gdal15.dll', 'install.bat', 'mapC5_v202.exe', 'msvcp71.dll', 'msvcr71.dll']

        for mapC5_item in mapC5_list:

            full_item = '{}/{}'.format(self.mapC5_dir, mapC5_item)
            out_item = '{}/{}'.format(self.model_dir, mapC5_item)

            if not os.path.isfile(out_item):
                shutil.copy2(full_item, out_item)

    def _clean_mapC5(self, tree_model):

        """
        Cleans the C5 directories
        """

        if tree_model == 'Cubist':

            mapC5_list = ['gdal13.dll', 'gdal15.dll', 'install.bat', 'mapCubist_v202.exe', 'msvcp71.dll', 'msvcr71.dll']

        elif tree_model == 'C5':

            mapC5_list = ['gdal13.dll', 'gdal15.dll', 'install.bat', 'mapC5_v202.exe', 'msvcp71.dll', 'msvcr71.dll']

        for mapC5_item in mapC5_list:

            full_item = '{}/{}'.format(self.model_dir, mapC5_item)

            if os.path.isfile(full_item):
                os.remove(full_item)


def _examples():

    sys.exit("""\

    --Find the optimum RF maximum depth--
    classification.py -s /samples.txt -p .5 --optimize 10
    ... would train and test (50/50) a range of depths over 10 folds cross-validation

    ===============
    Training models
    ===============

    --Train & save a Random Forest model--
    classification.py -s /samples.txt -mo /model_rf.xml
    ... would train and save a Random Forest model to model_rf.xml

    --Train & save a Random Forest model--
    classification.py -s /samples.txt --parameters classifier:RF,trees:2000 -mo /model_rf.xml
    ... would train and save a Random Forest model with 2000 trees to model_rf.xml

    --Train & save a Random Forest model--
    classification.py -s /samples.txt --parameters classifier:RF,trees:2000 -ig 5,10,15 -mo /model_rf.xml
    ... would train and save a Random Forest model with 2000 trees to model_rf.xml. The 5th, 10th, and 15th feature would
    not be used in the model

    --Train & save a Gradient Boosted Tree model--
    classification.py -s /samples.txt -labst float --parameters classifier:Boost -mo /model_boost.xml
    ... would train and save a Gradient Boosted Tree model with 1000 trees to model_boost.xml

    --Train & save a Support Vector Machine model--
    classification.py -s /samples.txt --parameters classifier:SVMA -mo /model_svm.xml --scale yes
    ... would train and save an auto-tuned Support Vector Machine model to model_svm.xml

    --Train & save a Cubist regression model--
    classification.py -s /samples.txt --parameters classifier:Cubist,committees:10,extrap:20 -mo /models/Cubist
    ... would train and save a Cubist model

    --Train & save a C5 model--
    classification.py -s /samples.txt --parameters classifier:C5,trials:10,CF:.4 -mo /models/C5
    ... would train and save a C5 model

    =======
    Mapping
    =======

    --Load model & map image--
    classification.py -mi /model_rf.xml -i /image_feas.tif -o /mapped_image.tif
    ... would load a Random Forest model and map image image_feas.tif

    --Load model & map image--
    classification.py -mi /model_rf.xml -ig 5,10,15 -i /image_feas.tif -o /mapped_image.tif
    ... would load a Random Forest model and map image image_feas.tif, ignore the 5th, 10th, and 15th image layer during
    the classification process.

    --Load model & map image--
    classification.py -mi /model_rf.xml -i /image_feas.tif -o /mapped_image.tif --rank RF -rankt /ranked_feas.txt --accuracy /accuracy.txt
    ... would load a Random Forest model, map image image_feas.tif, write ranked RF features to text, and write accuracy report to text

    --Load model & map image--
    classification.py -mi /model_Cubist -s /samples.txt --parameters classifier:Cubist -i /image_feas.tif -o /mapped_image.tif
    ... would load a Cubist model model and map image image_feas.tif

    ==================
    Ranking & Accuracy
    ==================

    --Rank and subset features--
    classification.py -s /samples.txt -i /image_feas.tif -or /image_feas_ranked.vrt --rank chi2
    ... would rank image features with Chi^2 and write to image_feas_ranked.vrt

    --Test model accuracy--
    classification.py -s /samples.txt --accuracy /accuracy.txt -mi /model_rf.xml
    ... would test the accuracy of model_rf.xml on 10% of data randomly sampled

    --Test model accuracy--
    classification.py -s /samples.txt --accuracy /accuracy.txt -mi /model_rf.xml -p .5
    ... would test the accuracy of model_rf.xml on 50% of data randomly sampled

    --Test model accuracy--
    classification.py -s /samples.txt --accuracy /accuracy.txt
    ... would test the accuracy of a new RF model on 10% of data withheld from training
    """)


def _usage():

    sys.exit("""\

    classification.py ...

    PRE-PROCESSING
    ==============
        [-s <Training samples (str) :: Default=None>]
        [-labst <Labels type (int or float) (str) :: Default=int>]
            *For reading samples (-s)
        [-p <Percent to sample from all samples (float) :: Default=.9>]
        [-pe <Percent to sample from each class (float) :: Default=None>]
            *Overrides -p
        [--subs <Dictionary of samples per class (dict) in place of -p :: Default={}>]
        [--recode <Dictionary recoded class values (dict) :: Default={}>]
        [-clrm <Classes to remove from data (int list) :: Default=[]>]
        [-valrm <Values, based on feature, to remove from data (val,fea) :: Default=[]>]
        [-ig <Features to ignore (int list) :: Default=[]>]
        [-xy <Use x, y coordinates as predictive variables? :: Default=no>]
        [--outliers <Remove outliers (str) :: Default=no>]
        [--loc_outliers <Locate outliers and do not remove (str) :: Default=no>]
        [--scale <Scale data (str) :: Default=no>]
        [--semi <Semi supervised labeling (str) :: Default=no>]
        [--visualize <Visualize data in feature space on two or three features (fea1,fea2,OPTfea3) :: Default=[]>]
        [--decision <Visualize the decision function on two features (fea1,fea2,class,compare--1 or 2) :: Default=[]>]
    MODEL
    =====
        [--parameters <Classifier & parameters (str) :: Default=classifier:RF>]
            *Use --parameters key1:parameter,key2:parameter except with --majority, where classifiers:RF-SVM-EX_RF-Bayes, e.g.
        [-mi <Input model (str) :: Default=None>]
        [-mo <Output model (str), .txt for Scikit models, .xml for OpenCV models :: Default=None>]
        [--accuracy <Accuracy of test samples withheld from training (str) :: Default=Automatic with -mo>]
        [-probs <Get class probability layers instead of labels (str) :: Default=no>]
        [--rank <Rank method (str) :: Default=None>]
        [-topf <Number of top features to subset (int -total- or float -percentage-) :: Default=1.>]
        [-or <Output ranked image (str) :: Default=None>]
        [-rankt <Write ranked features to rankt text (str) :: Default=None>]
        [--optimize <Optimize parameters (str) :: Default=no>]
    MAPS
    ====
        [--majority <Rank the majority classification with --parameters classifiers:cl1-cl2-cl3-etc (str) :: Default=no>]
        [-i <Input image to classify (str) :: Default=None>]
        [-o <Output classified image (str) :: Default=None>]
        [-addl <Additional images to use in the model (str list) :: Default=[]>]
        [--jobs <Number of jobs for parallel mapping, with --multi (int) :: Default=-1>]
        [-c <Chunk size for parallel mapping (int) :: Default=8000>]
        [-bc <Band to check for zeros (int) :: Default=-1]
    [-h <Prints this dialogue>
    [--options <Print list of classifier options>]
    [-e <Prints examples>

    """)


def main():

    argv = sys.argv

    if argv is None:
        sys.exit(0)

    samples = None
    img = None
    out_img = None
    out_img_rank = None
    input_model = None
    output_model = None
    perc_samp = .9
    perc_samp_each = 0
    scale_data = False
    labs_type = 'int'
    class_subs = {}
    recode_dict = {}
    classes2remove = []
    valrm_fea = []
    ignore_feas = []
    use_xy = False
    outrm = False
    locate_outliers = False
    semi = False
    semi_kernel = 'knn'
    feature_space = []
    decision_function = []
    header = True
    norm_struct = True
    classifier_info = {'classifier': 'RF'}
    var_imp = True
    rank_method = None
    top_feas = 1.
    out_acc = None
    get_majority = False
    optimize = False
    rank_txt = None
    get_probs = False
    additional_layers = []
    n_jobs = -1
    band_check = -1
    chunk_size = 8000

    i = 1
    while i < len(argv):

        arg = argv[i]

        if arg == '-i':
            i += 1
            img = argv[i]

        elif arg == '-o':
            i += 1
            out_img = argv[i]

        elif arg == '-or':
            i += 1
            out_img_rank = argv[i]

        elif arg == '-s':
            i += 1
            samples = argv[i]

        elif arg == '--scale':
            i += 1
            scale_data = argv[i]

            if scale_data == 'yes':
                scale_data = True

        elif arg == '--parameters':
            i += 1

            classifier_info = argv[i]
            classifier_info = classifier_info.split(',')

            info_dict = '{'
            cli_ctr = 1
            for cli in classifier_info:

                cli_split = cli.split(':')

                if 'classifiers' in cli:
                    if cli_ctr == len(classifier_info):
                        info_dict = "%s'%s':%s" % (info_dict, cli_split[0], cli_split[1].split('-'))
                    else:
                        info_dict = "%s'%s':%s," % (info_dict, cli_split[0], cli_split[1].split('-'))
                elif cli_ctr == len(classifier_info):
                    info_dict = "%s'%s':'%s'" % (info_dict, cli_split[0], cli_split[1])
                else:
                    info_dict = "%s'%s':'%s'," % (info_dict, cli_split[0], cli_split[1])

                cli_ctr += 1

            info_dict = '%s}' % info_dict

            classifier_info = ast.literal_eval(info_dict)

            # convert values to integers
            for key in classifier_info:
                is_int = False
                try:
                    classifier_info[key] = int(classifier_info[key])
                    is_int = True
                except:
                    pass

                if not is_int:
                    try:
                        classifier_info[key] = float(classifier_info[key])
                    except:
                        pass

        elif arg == '-p':
            i += 1
            perc_samp = float(argv[i])

        elif arg == '-pe':
            i += 1
            perc_samp_each = float(argv[i])

        elif arg == '--subs':
            i += 1

            class_subs = ''.join(argv[i])
            class_subs = '{%s}' % class_subs

            class_subs = ast.literal_eval(class_subs)

        elif arg == '--recode':
            i += 1

            recode_dict = ''.join(argv[i])
            recode_dict = '{%s}' % recode_dict

            recode_dict = ast.literal_eval(recode_dict)

        elif arg == '-clrm':
            i += 1
            classes2remove = argv[i].split(',')
            classes2remove = map(int, classes2remove)

        elif arg == '-valrm':
            i += 1
            valrm_fea = argv[i].split(',')
            valrm_fea = map(int, valrm_fea)

        elif arg == '-ig':
            i += 1
            ignore_feas = argv[i].split(',')
            ignore_feas = map(int, ignore_feas)

        elif arg == '-xy':
            i += 1
            use_xy = argv[i]
            if use_xy == 'yes':
                use_xy = True

        elif arg == '--outliers':
            i += 1
            outrm = argv[i]
            if outrm == 'yes':
                outrm = True

        elif arg == '--loc_outliers':
            i += 1
            locate_outliers = argv[i]
            if locate_outliers == 'yes':
                locate_outliers = True

        elif arg == '--semi':
            i += 1
            semi = argv[i]
            if semi == 'yes':
                semi = True

        elif arg == '-semik':
            i += 1
            semi_kernel = argv[i]

        elif arg == '--visualize':
            i += 1
            feature_space = argv[i].split(',')
            feature_space = map(int, feature_space)

        elif arg == '--decision':
            i += 1
            decision_function = argv[i].split(',')
            decision_function = map(int, decision_function)

        elif arg == '--optimize':
            i += 1
            if argv[i] == 'yes':
                optimize = True

        elif arg == '-mi':
            i += 1
            input_model = argv[i]

        elif arg == '-mo':
            i += 1
            output_model = argv[i]

        elif arg == '--rank':
            i += 1
            rank_method = argv[i]

        elif arg == '-rankt':
            i += 1
            rank_txt = argv[i]

        elif arg == '-labst':
            i += 1
            labs_type = argv[i]

        elif arg == '-topf':
            i += 1
            top_feas = argv[i]

            if '.' in top_feas:
                top_feas = float(top_feas)
            else:
                top_feas = int(top_feas)

        elif arg == '--accuracy':
            i += 1
            out_acc = argv[i]

        elif arg == '--majority':
            i += 1
            get_majority = argv[i]
            if get_majority == 'yes':
                get_majority = True

        elif arg == '-probs':
            i += 1
            get_probs = argv[i]

            if get_probs == 'yes':
                get_probs = True

        elif arg == '-addl':
            i += 1
            additional_layers = argv[i].split(',')

        elif arg == '--jobs':
            i += 1
            n_jobs = int(argv[i])

        elif arg == '-bc':
            i += 1
            band_check = int(argv[i])

        elif arg == '-c':
            i += 1
            chunk_size = int(argv[i])

        elif arg == '-h':
            _usage()

        elif arg == '-e':
            _examples()

        elif arg == '--options':
            _options()

        elif arg[:1] == ':':
            print('Unrecognized command option: %s' % arg)
            _usage()

        i += 1

    print('\nStart date & time --- (%s)\n' % time.asctime(time.localtime(time.time())))

    start_time = time.time()

    try:
        dummy = classifier_info['classifier']
    except:
        classifier_info['classifier'] = 'RF'

    if 'Cubist' in classifier_info['classifier'] or 'C5' in classifier_info['classifier']:

        # create the C5/Cubist object
        cl = c5_cubist()

        if samples:

            if rank_method:
                scale_data = True

            cl.split_samples(samples, perc_samp=perc_samp, perc_samp_each=perc_samp_each, scale_data=scale_data, \
                           class_subs=class_subs, header=header, norm_struct=norm_struct, labs_type=labs_type, \
                           recode_dict=recode_dict, classes2remove=classes2remove, ignore_feas=ignore_feas)

            if valrm_fea:
                cl.remove_values(valrm_fea[0], valrm_fea[1])

            if outrm:
                cl.remove_outliers(locate_only=locate_outliers)

        # train the model
        if output_model:

            # train the C5/Cubist model
            cl.train_c5_cubist(samples, output_model, classifier_info=classifier_info)

            # cl = classification()

            # cl.split_samples(samples, perc_samp=perc_samp, header=header, norm_struct=norm_struct, labs_type='float')

            # out_acc = '%s/%s_acc.txt' % (c5_cubist.model_dir, c5_cubist.model_base)

            # cl.test_accuracy(out_acc=out_acc, discrete=False)

        # predict labels
        if input_model and out_img:

            cl.map_labels_c5_cubist(input_model, samples, img, out_img, tree_model=classifier_info['classifier'])

    else:

        # create the classifier object
        cl = classification()

        # get predictive variables and class labels data
        if optimize:

            cl.optimize_parameters(samples, classifier_info, perc_samp, max_depth_range=(1, 100), k_folds=optimize)

            print '\nThe optimum depth was %d' % cl.opt_depth
            print 'The maximum accuracy was %f\n' % cl.max_acc

        if samples:

            if rank_method:
                scale_data = True

            cl.split_samples(samples, perc_samp=perc_samp, perc_samp_each=perc_samp_each, scale_data=scale_data, \
                           class_subs=class_subs, header=header, norm_struct=norm_struct, labs_type=labs_type, \
                           recode_dict=recode_dict, classes2remove=classes2remove, ignore_feas=ignore_feas, \
                           use_xy=use_xy)

            if feature_space:

                if len(feature_space) == 3:
                    fea_z = feature_space[2]
                else:
                    fea_z = None

                if semi:
                    # classified_labels = np.where(cl.labels != -1)
                    classified_labels = None
                else:
                    classified_labels = None

                cl.vis_data(feature_space[0], feature_space[1], fea_3=fea_z, labels=classified_labels)

            if valrm_fea:

                cl.remove_values(valrm_fea[0], valrm_fea[1])

            if semi:

                cl.semi_supervised(classifier_info, kernel=semi_kernel)

                if feature_space:

                    if len(feature_space) == 3:
                        fea_z = feature_space[2]
                    else:
                        fea_z = None

                    cl.vis_data(feature_space[0], feature_space[1], fea_3=fea_z, labels=classified_labels)

            if outrm:

                cl.remove_outliers(locate_only=locate_outliers)

                if feature_space:

                    if len(feature_space) == 3:
                        fea_z = feature_space[2]
                    else:
                        fea_z = None

                    cl.vis_data(feature_space[0], feature_space[1], fea_3=fea_z, labels=classified_labels)

            if decision_function:

                cl.vis_decision(decision_function[0], decision_function[1], classifier_info=classifier_info,
                                class2check=decision_function[2], compare=decision_function[3],
                                locate_outliers=locate_outliers)

        if get_majority:

            cl.stack_majority(img, output_model, out_img, classifier_info, scale_data, ignore_feas=ignore_feas)

        if input_model or output_model or img or (rank_method == 'RF') or out_acc and not get_majority and \
                (rank_method != 'chi2'):

            cl.construct_model(input_model=input_model, output_model=output_model, classifier_info=classifier_info,
                               var_imp=var_imp, rank_method=rank_method, top_feas=top_feas, get_probs=get_probs)

            if out_acc:
                cl.test_accuracy(out_acc=out_acc)

        if rank_method:

            cl.rank_feas(rank_text=rank_txt, rank_method=rank_method, top_feas=top_feas)

        if out_img and not get_majority:

            # apply classification model to map image class labels
            cl.predict(img, out_img, additional_layers=additional_layers, n_jobs=n_jobs, band_check=band_check,
                       scale_data=scale_data, ignore_feas=ignore_feas, chunk_size=chunk_size, use_xy=use_xy)

        if out_img_rank:

            cl.sub_feas(img, out_img_rank)

    print('\nEnd data & time -- (%s)\nTotal processing time -- (%.2gs)\n' %
          (time.asctime(time.localtime(time.time())), (time.time()-start_time)))

if __name__ == '__main__':
    main()
