#!/usr/bin/env python

"""
@author: Jordan Graesser
Date Created: 3/26/2014
"""

import os
import sys
import time
import subprocess
import fnmatch
import argparse
import ast
import itertools

# Pickle
try:
    import cPickle as pickle
except:
    from six.moves import cPickle as pickle
else:
   import pickle

from .helpers import moving_window
from .helpers.plot import _handle_axis_spines, _add_axis_commas, _handle_axis_ticks, \
    _handle_axis_frame, _handle_tick_labels

from mpglue import raster_tools, vector_tools, classification

# Scikit-learn
try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import brier_score_loss, precision_score, recall_score, f1_score
    from sklearn.calibration import CalibratedClassifierCV, calibration_curve
except ImportError:
    raise ImportError('Scikit-learn must be installed')

# NumPy
try:
    import numpy as np
except ImportError:
    raise ImportError('NumPy must be installed')

# Pandas
try:
    import pandas as pd
except ImportError:
    raise ImportError('Pandas must be installed')

# GDAL
try:
    from osgeo import gdal
except ImportError:
    raise ImportError('GDAL must be installed')

# SciPy Ndimage
try:
    from scipy.ndimage import zoom
except ImportError:
    raise ImportError('Ndimage must be installed')

# Matplotlib
try:
    import matplotlib as mpl
    mpl.use('TkAgg')
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError('Matplotlib must be installed')

# Seaborn
# try:
#     import seaborn as sns
# except ImportError:
#     raise ImportError('Seaborn must be installed')

# YAML
try:
    import yaml
except ImportError:
    raise ImportError('YAML must be installed')


dpi = 200

mpl.rcParams['font.family'] = 'Calibri'
mpl.rcParams['figure.dpi'] = dpi
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['figure.edgecolor'] = 'white'
# mpl.rcParams['axes.facecolor'] = 'white'
# mpl.rcParams['axes.edgecolor'] = '#2E2E2E'
mpl.rcParams['axes.labelcolor'] = 'black'
mpl.rcParams['ytick.color'] = 'black'
mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['axes.grid'] = False
mpl.rcParams['savefig.facecolor'] = 'white'
mpl.rcParams['savefig.edgecolor'] = 'white'
mpl.rcParams['savefig.transparent'] = False
mpl.rcParams['savefig.dpi'] = dpi
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = .1
mpl.rcParams['font.size'] = 10.
mpl.rcParams['axes.labelsize'] = 10.
mpl.rcParams['xtick.labelsize'] = 7.
mpl.rcParams['ytick.labelsize'] = 7.


def samples2file(samples2write, bands, out_samples):

    # Prepend dummy x, y coordinates.
    samples2write = np.c_[np.zeros((samples2write.shape[0], 2), dtype='float32'), samples2write]

    hdr = 'X,Y'
    bd_list = np.arange(1, bands + 1)
    bd_list = [str(bd) for bd in bd_list]

    bd_list = ',bd_'.join(bd_list)

    headers = '{},bd_{},response'.format(hdr, bd_list).split(',')

    df = pd.DataFrame(data=samples2write, columns=headers, index=range(1, samples2write.shape[0]+1))

    df.index.name = 'Id'

    # save to text
    if os.path.isfile(out_samples):
        os.remove(out_samples)

    # np.savetxt(out_samples, samples2write, delimiter=',', newline='\n', header=headers, comments='')
    df.to_csv(out_samples, sep=',')


def prep_building_training():

    """
    Prepares building training samples
    """

    out_dir = PARAMETERS['out dir']
    window_size = PARAMETERS['window size meters']
    building_image_directory = os.path.join(out_dir, 'buildings2density')
    feature_image_directory = os.path.join(out_dir, 'patch_features')

    out_training_dir = os.path.join(out_dir, 'training')

    if not os.path.isdir(out_training_dir):
        os.makedirs(out_training_dir)

    out_training_text = os.path.join(out_dir, 'training_master.txt')

    half_window = window_size / 2

    # List all of the building density rasters.
    bldg_img_list = fnmatch.filter(os.listdir(building_image_directory), '*.tif')

    # Iterate over each building density raster.
    for bi, bldg_img in enumerate(bldg_img_list):

        bldg_density = os.path.join(building_image_directory, bldg_img)
        features = os.path.join(feature_image_directory, bldg_img)

        if not os.path.isfile(features):
            continue

        # Open the features image.
        f_info = raster_tools.rinfo(features)

        bands = f_info.bands

        # Open the feature arrays.
        fea_arr = f_info.mparray(bands2open=-1, d_type='float32')

        f_info.close()

        # Subset the arrays to avoid the grid edges.
        fea_arr = fea_arr[:, half_window:-half_window, half_window:-half_window]

        # Get the new array shape.
        dims, rs, cs = fea_arr.shape
        samples = rs * cs

        # Reshape the features for predictions.
        fea_arr = fea_arr.reshape(dims, samples).T.reshape(samples, dims)

        # Open the building density image.
        i_info = raster_tools.rinfo(bldg_density)

        bldg_arr = i_info.mparray(bands2open=1, d_type='float32')

        i_info.close()

        bldg_arr = bldg_arr[half_window:-half_window, half_window:-half_window]

        # Append the building density samples as the independent variable.
        fea_arr = np.c_[fea_arr, bldg_arr.ravel()]

        # Save the patch samples to file.
        out_training_text_patch = os.path.join(out_training_dir,
                                               'training_{}.txt'.format(bldg_img.replace('.tif', '')))

        samples2file(fea_arr, bands, out_training_text_patch)

        # Add samples.
        if bi == 0:
            fea_arr_master = fea_arr.copy()
        else:
            fea_arr_master = np.r_[fea_arr_master, fea_arr]

    samples2file(fea_arr_master, bands, out_training_text)


def rank_patch_features(model_parameters, top_feas):

    """
    Ranks patch image features
    """

    out_dir = PARAMETERS['out dir']

    training_dir = os.path.join(out_dir, 'training')
    bad_dir = os.path.join(out_dir, 'bad_feas')

    if not os.path.isdir(bad_dir):
        os.makedirs(bad_dir)

    if model_parameters['classifier'] == 'Cubist':
        cl = classification_r()
    else:
        cl = classification()

    # Train a model on each patch, plus the master
    for training_file in fnmatch.filter(os.listdir(training_dir), '*.txt'):

        input_training = os.path.join(training_dir, training_file)
        output_bad_features = os.path.join(bad_dir, training_file)

        cl.split_samples(input_training, perc_samp=1., scale_data=True)

        cl.rank_feas(rank_method='chi2', top_feas=top_feas)

        if os.path.isfile(output_bad_features):
            os.remove(output_bad_features)

        pickle.dump(sorted(cl.bad_features),
                    file(output_bad_features, 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)


def buildings2density(overwrite):

    """
    Converts building points to raster, then densifies
    """

    vector_buildings = PARAMETERS['buildings']
    vector_patches = PARAMETERS['patches']
    out_dir = PARAMETERS['out dir']
    in_cell = PARAMETERS['sample size']
    patch_id = PARAMETERS['patch id']
    point_patch = PARAMETERS['point patch id']
    density_m = PARAMETERS['window size meters']
    fea_cell_size = PARAMETERS['feature cell size']

    patch_features_dir = os.path.join(out_dir, 'patch_features')
    out_dir = os.path.join(out_dir, 'buildings2density')

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # get patch vector information
    vct_info = vector_tools.vinfo(vector_patches)

    # get the patch count in the vector file
    n_patches = vct_info.n_feas

    # iterate over each patch
    for n_patch in xrange(0, n_patches):

        # get the current patch layer
        vct_patch = vct_info.lyr.GetFeature(n_patch)

        # get the feature geometry
        vct_patch_geom = vct_patch.GetGeometryRef()

        lon_left, lon_right, lat_bottom, lat_top = vct_patch_geom.GetEnvelope()

        # set the feature extent
        n_patch_extent = (lon_left, lat_top, lon_right, lat_bottom)

        # get the patch layer id number
        vct_patch_num = vct_patch.GetField(patch_id)

        # temp_csv = '%s/master.csv' % out_dir
        # temp_vrt = '%s/master.vrt' % out_dir

        # name the output feature patch by the patch id number
        if vct_patch_num < 10:
            count_str = '00{:d}'.format(vct_patch_num)
        elif 10 <= vct_patch_num < 100:
            count_str = '0{:d}'.format(vct_patch_num)
        else:
            count_str = str(vct_patch_num)

        temp_rst = os.path.join(out_dir, 'temp_{}.tif'.format(count_str))
        fea_patch = os.path.join(patch_features_dir, '{}.tif'.format(count_str))
        bldg_density = os.path.join(out_dir, '{}.tif'.format(count_str))

        if overwrite:

            if os.path.isfile(bldg_density):
                os.remove(bldg_density)

        else:

            if os.path.isfile(bldg_density):
                continue

        # rasterize current patch where patch_id=n_patch, with lab_id value
        com = 'gdal_rasterize -init 0 -a {} -where {}="{:d}" -te {:f} {:f} {:f} {:f} \
        -tr {:f} {:f} -ot Byte {} {}'.format(patch_id, point_patch, vct_patch_num,
                                             n_patch_extent[0], n_patch_extent[3],
                                             n_patch_extent[2], n_patch_extent[1],
                                             in_cell, in_cell, vector_buildings, temp_rst)

        subprocess.call(com, shell=True)

        o_info = raster_tools.rinfo(fea_patch)
        o_info.bands = 1

        t_info = raster_tools.rinfo(temp_rst)
        t_array = t_info.mparray(d_type='float32')

        t_info.close()

        # First, sum the buildings within ``density_m`` area.
        density_array = moving_window(t_array, statistic='sum', window_size=density_m)

        # Next, sum the buildings within feature-sized cells.
        density_array = moving_window(density_array, statistic='sum', skip_block=int(fea_cell_size))

        # plt.subplot(121)
        # plt.imshow(density_array)
        # plt.subplot(122)
        # plt.imshow(density_array2)
        # plt.show()
        # sys.exit()

        # Resample if necessary
        row_zoom_factor = o_info.rows / float(density_array.shape[0])
        col_zoom_factor = o_info.cols / float(density_array.shape[1])

        if row_zoom_factor != 1 or col_zoom_factor != 1:
            density_array = zoom(density_array, (row_zoom_factor, col_zoom_factor), order=1)

        raster_tools.write2raster(density_array, bldg_density, o_info, flush_final=True)

        o_info.close()

        # Resample the cell size.
        # com = 'gdalwarp -tr {:f} {:f} -r average -co COMPRESS=LZW {} {}'.format(fea_cell_size, fea_cell_size,
        #                                                                         temp_rst2, bldg_density)
        #
        # if os.path.isfile(bldg_density):
        #     os.remove(bldg_density)
        #
        # subprocess.call(com, shell=True)
        #
        os.remove(temp_rst)
        # os.remove(temp_rst2)


def prep_image_features(overwrite, resample2features=0.):

    """
    Extracts image feature patches as defined by patch vector boundaries
    """

    vector_patches = PARAMETERS['patches']
    image_features = PARAMETERS['features']
    out_dir = PARAMETERS['out dir']
    patch_id = PARAMETERS['patch id']

    out_dir = os.path.join(out_dir, 'patch_features')

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Get vector information.
    vct_info = vector_tools.vinfo(vector_patches)

    # Get the patch count in the vector file.
    n_patches = vct_info.n_feas

    # Get the image feature info.
    f_info = raster_tools.rinfo(image_features)

    com_orig = 'gdalwarp --config GDAL_CACHEMAX 256'

    # iterate over each patch
    for n_patch in xrange(0, n_patches):

        # Get the current patch layer.
        vct_patch = vct_info.lyr.GetFeature(n_patch)

        # Get the feature geometry.
        vct_patch_geom = vct_patch.GetGeometryRef()

        lon_left, lon_right, lat_bottom, lat_top = vct_patch_geom.GetEnvelope()

        # Set the feature extent.
        n_patch_extent = (lon_left, lat_top, lon_right, lat_bottom)

        # Get the patch layer id number.
        vct_patch_num = vct_patch.GetField(patch_id)

        # Name the output feature patch by the patch id number
        if vct_patch_num < 10:
            out_patch = os.path.join(out_dir, '00{:d}.tif'.format(vct_patch_num))
        elif 10 <= vct_patch_num < 100:
            out_patch = os.path.join(out_dir, '0{:d}.tif'.format(vct_patch_num))
        else:
            out_patch = os.path.join(out_dir, '{:d}.tif'.format(vct_patch_num))

        com = '{} -te {:f} {:f} {:f} {:f} -tr {:f} {:f} {} {}'.format(com_orig,
                                                                      n_patch_extent[0], n_patch_extent[3],
                                                                      n_patch_extent[2], n_patch_extent[1],
                                                                      f_info.cellY, f_info.cellY,
                                                                      image_features, out_patch)

        if overwrite:

            if os.path.isfile(out_patch):
                os.remove(out_patch)

            subprocess.call(com, shell=True)

        else:

            if not os.path.isfile(out_patch):
                subprocess.call(com, shell=True)

        if resample2features > 0:

            d_name, f_name = os.path.split(out_patch)
            f_base, f_ext = os.path.splitext(f_name)

            out_patch_temp = os.path.join(d_name, '{}_temp{}'.format(f_base, f_ext))

            com = 'gdalwarp --config GDAL_CACHEMAX 256 -tr {:f} {:f} \
            -r cubic -co COMPRESS=LZW {} {}'.format(resample2features, resample2features,
                                                    out_patch, out_patch_temp)

            if not os.path.isfile(out_patch_temp):
                subprocess.call(com, shell=True)

            os.remove(out_patch)
            os.rename(out_patch_temp, out_patch)

    f_info.close()


def train(model_parameters, overwrite, ignore_feas):

    """
    Trains a regression model

    Args:
        model_parameters (str): The model parameter dictionary.
    """

    out_dir = PARAMETERS['out dir']
    training_dir = os.path.join(out_dir, 'training')

    model_dir = os.path.join(out_dir, 'models')
    report_dir = os.path.join(out_dir, 'reports')
    bad_dir = os.path.join(out_dir, 'bad_feas')

    for dir2create in [model_dir, report_dir]:

        if not os.path.isdir(dir2create):
            os.makedirs(dir2create)

    if model_parameters['classifier'] == 'Cubist':
        cl = classification_r()
    else:
        cl = classification()

    # Train a model on each patch, plus the master
    for training_file in fnmatch.filter(os.listdir(training_dir), '*.txt'):

        if '_scaler.txt' in training_file:
            continue

        input_training = os.path.join(training_dir, training_file)
        output_accuracy = os.path.join(report_dir, training_file)

        if model_parameters['classifier'] == 'Cubist':

            output_model = os.path.join(model_dir, '{}_{}'.format(model_parameters['classifier'],
                                                                  training_file.replace('.txt', '')))

        else:

            output_model = os.path.join(model_dir, '{}_{}.txt'.format(model_parameters['classifier'],
                                                                      training_file.replace('.txt', '')))

        if overwrite:

            for file2create in [output_model, output_accuracy]:

                if os.path.isfile(file2create):
                    os.remove(file2create)

        # cl.split_samples(input_training, perc_samp=.5, labs_type='float')

        # Train a regression model with 50% of samples.
        # cl.construct_model(classifier_info=model_parameters)

        # Get test accuracy on 50% of the samples.
        # cl.test_accuracy(out_acc=output_accuracy, discrete=False)

        if ignore_feas:

            output_bad_features = os.path.join(bad_dir, training_file)
            ignore_feas = pickle.load(file(output_bad_features, 'rb'))

        else:
            ignore_feas = []

        cl.split_samples(input_training, perc_samp=1, labs_type='float', ignore_feas=ignore_feas)

        # Train a model with all samples and write to file.
        if model_parameters['classifier'] == 'Cubist':
            cl.construct_r_model(classifier_info=model_parameters, output_model=output_model, write_summary=False)
        else:
            cl.construct_model(classifier_info=model_parameters, output_model=output_model)

    # Write the classifier parameters to file.
    parameter_file = os.path.join(out_dir, 'model_parameters_{}.yaml'.format(model_parameters['classifier']))

    if os.path.isfile(parameter_file):

        with open(parameter_file, 'r') as pf:
            model_parameters = yaml.load(pf)

        for k, v in cl.classifier_info.iteritems():
            model_parameters[k] = v

    else:
        model_parameters = cl.classifier_info

    with open(parameter_file, 'w') as pf:
        pf.write(yaml.dump(model_parameters, default_flow_style=False))


def test(model_parameters, ignore_feas):

    """
    Tests predictions of the master file on all patches

    Args:
        model_parameters (str): The model parameter dictionary.
    """

    out_dir = PARAMETERS['out dir']

    training_dir = os.path.join(out_dir, 'training')
    model_dir = os.path.join(out_dir, 'models')
    report_dir = os.path.join(out_dir, 'reports')
    bad_dir = os.path.join(out_dir, 'bad_feas')

    assert os.path.isdir(training_dir) and os.path.isdir(model_dir) and os.path.isdir(report_dir)

    # Load the master model.
    if model_parameters['classifier'] == 'Cubist':

        input_model = os.path.join(model_dir, '{}_training_master.tree'.format(model_parameters['classifier']))

        cl = classification_r()
        cl.construct_r_model(input_model=input_model)

    else:

        input_model = os.path.join(model_dir, '{}_training_master.txt'.format(model_parameters['classifier']))

        cl = classification()
        cl.construct_model(input_model=input_model)

    # Test the master model against each patch.
    for training_file in fnmatch.filter(os.listdir(training_dir), '*.txt'):

        if '_scaler.txt' in training_file:
            continue

        input_training = os.path.join(training_dir, training_file)
        output_accuracy = os.path.join(report_dir, '{}_{}'.format(model_parameters['classifier'], training_file))

        if os.path.isfile(output_accuracy):
            os.remove(output_accuracy)

        if ignore_feas:

            output_bad_features = os.path.join(bad_dir, training_file)
            ignore_feas = pickle.load(file(output_bad_features, 'rb'))

        else:
            ignore_feas = []

        # Load the samples to test.
        cl.split_samples(input_training, perc_samp=1, labs_type='float', ignore_feas=ignore_feas)

        # Test the accuracy.
        cl.test_accuracy(out_acc=output_accuracy, discrete=False)


def plot(model_parameters, what2plot='scatter'):

    """
    Plots predictions vs. actual

    Args:
        model_parameters (str): The model parameter dictionary.
        what2plot (Optional[str]): Choices are ['patches', 'scatter'].
    """

    # sns.set_style('white')

    # fig = plt.figure()
    # fig, axes = plt.subplots(4, 4, figsize=(9, 9), sharex=True, sharey=True)

    out_dir = PARAMETERS['out dir']

    training_dir = os.path.join(out_dir, 'training')
    model_dir = os.path.join(out_dir, 'models')
    figure_dir = os.path.join(out_dir, 'figures')
    patch_dir = os.path.join(out_dir, 'patch_features')
    buildings_dir = os.path.join(out_dir, 'buildings2density')

    output_figure = os.path.join(figure_dir, '{}_{}.png'.format(model_parameters['classifier'], what2plot))

    if not os.path.isdir(figure_dir):
        os.makedirs(figure_dir)

    assert os.path.isdir(training_dir) and os.path.isdir(model_dir) and os.path.isdir(figure_dir)

    # Load the master model.
    if model_parameters['classifier'] == 'Cubist':

        input_model = os.path.join(model_dir, '{}_training_master.tree'.format(model_parameters['classifier']))

        cl = classification_r()
        cl.construct_r_model(input_model=input_model)

    else:

        input_model = os.path.join(model_dir, '{}_training_master.txt'.format(model_parameters['classifier']))

        cl = classification()
        cl.construct_model(input_model=input_model)

    # fig = plt.figure()
    if what2plot == 'scatter':

        axes_rows = 4
        axes_cols = 4
        w = 7
        h = 7

        axes_iters = list(itertools.product(*[range(axes_rows), range(axes_cols)]))

    elif what2plot == 'patches':

        axes_rows = 7+8-1
        axes_cols = 2
        w = 3
        h = 15

    fig, axes = plt.subplots(axes_rows, axes_cols, sharex=True, sharey=True)

    fig.set_size_inches(w, h)

    # Test the master model against each patch.
    for ti, training_file in enumerate(fnmatch.filter(os.listdir(training_dir), '*.txt')):

        input_training = os.path.join(training_dir, training_file)

        # Load the samples to test.
        cl.split_samples(input_training, perc_samp=1, labs_type='float')

        # fig = plt.figure()
        # ax = fig.add_subplot(2, 2, 1, axisbg='white')

        # if what2plot == 'scatter':
        #     fig, ax = plt.subplots(7, 8, ti+1, axisbg='white', sharex='col', sharey='row')
        #     # ax = fig.add_subplot(7, 8, ti+1, axisbg='white', sharex='col', sharey='row')
        # elif what2plot == 'patches':
        #     fig, ax = plt.subplots(14, 2, ti+1, axisbg='white', sharex='col', sharey='row')

        # ax = plt.subplot2grid((8, 7), (0, ti))
        # ax = plt.subplot(4, 4, ti+1)

        # ax.hexbin(cl.test_array[:, 0], cl.test_array[:, 1], cmap='Blues')

        # res, r_sq_poly, p_val_poly, fitted_data, sig_poly = least_squares.fit(x_variables=cl.test_array[:, 0],
        #                                                                       y_variables=cl.test_array[:, 1],
        #                                                                       polynomial_order=1,
        #                                                                       y_label='Actual',
        #                                                                       x_label='Predicted')

        # g = sns.jointplot(cl.test_array[:, 0], cl.test_array[:, 1], kind='reg', color='#4CB391')

        # ax.plot(cl.test_array[:, 0], fitted_data, c='#8A4B08', lw=.7)

        if what2plot == 'scatter':

            ax = axes[axes_iters[ti][0], axes_iters[ti][1]]

            cl.test_accuracy(discrete=False, be_quiet=True)
            predictions = cl.test_array

            ax.plot([0, predictions[:, 1].max()+10], [0, predictions[:, 1].max()+10], c='#8A4B08', lw=.7)
            ax.scatter(predictions[:, 0], predictions[:, 1], c='#A4A4A4', edgecolor='#A4A4A4', s=5, alpha=.6)

            # ax.set_xlim(0, predictions[:, 1].max()+10)
            # ax.set_ylim(0, predictions[:, 1].max()+10)

            ax.set_xlim(0, 2000)
            ax.set_ylim(0, 2000)

            # ax.set_xlabel('Predicted buildings/ha')
            # ax.set_ylabel('Actual buildings/ha')

            ax = _add_axis_commas(ax)

            ax = _handle_axis_spines(ax)
            ax = _handle_axis_ticks(ax)

        elif what2plot == 'patches':

            if 'master' in training_file:
                continue

            ax1 = axes[ti, 0]
            ax2 = axes[ti, 1]

            # Load the buildings image.
            patch_image = os.path.join(patch_dir, training_file.replace('.txt', '.tif'))
            building_image = os.path.join(buildings_dir, training_file.replace('.txt', '.tif'))

            # Open the patch image.
            i_info = raster_tools.rinfo(patch_image.replace('training_', ''))

            p_array = i_info.mparray(bands2open=-1, predictions=True)

            # Open the buildings images.
            b_info = raster_tools.rinfo(building_image.replace('training_', ''))

            b_array = b_info.mparray()

            # Get predictions.
            i_info.array = cl.predict_array(p_array)
            i_info.rrows = b_array.shape[0]
            i_info.ccols = b_array.shape[1]
            i_info.predictions2norm()

            im1 = ax1.imshow(b_array, interpolation='nearest')
            im2 = ax2.imshow(i_info.array, interpolation='nearest')

            im1.set_cmap('magma')
            im1.set_clim(0, 500)
            im2.set_cmap('magma')
            im2.set_clim(0, 500)

            im1.axes.get_xaxis().set_visible(False)
            im2.axes.get_yaxis().set_visible(False)

            plt.axis('off')

            for ax in [ax1, ax2]:
                ax = _handle_tick_labels(ax)
                # ax = _handle_axis_spines(ax)
                ax = _handle_axis_frame(ax)
                ax = _handle_axis_ticks(ax)

        # print dir(g.ax_joint)
        # g.ax_joint.set_ylim(0, cl.test_array[:, 1].max()+10)
        # g.ax_joint.set_xlim(0, cl.test_array[:, 1].max()+10)

        # g.ax_joint.set_xlabel('Predicted')
        # g.ax_joint.set_ylabel('Actual')

        #, ax=axes.flat[ti])
        # joint_kws={'gridsize': 40}, ax=ax)

        # sns.despine()

        # g.set(ylim=(0, 2000), xlim=(0, 2000), xlabel='Predicted', ylabel='Actual')

        # sns.jointplot(cl.test_array[:, 0], cl.test_array[:, 1], kind='reg', ax=ax, scatter=False)

        # ax.set_xticks([])
        # ax.set_xticklabels([''])
        # ax.set_yticks([])
        # ax.set_yticklabels([''])

        # for pos in ['top', 'bottom', 'right', 'left']:
        #     ax.spines[pos].set_edgecolor('#71777F')

    # Remove the last axis
    if what2plot == 'scatter':

        ax = axes[axes_iters[-1][0], axes_iters[-1][1]]

        ax = _handle_axis_ticks(ax)
        ax = _handle_axis_frame(ax)

        ax.set_xticks([])
        ax.set_yticks([])

        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')

        plt.xlabel('Predicted buildings/ha', size=16)
        plt.ylabel('Actual buildings/ha', size=16)

        plt.title(model_parameters['classifier'], size=16)

    if what2plot == 'patches':

        fig.text(.3, .91, 'Actual', ha='center', size=16)
        fig.text(.72, .91, 'Predicted', ha='center', size=16)

    if os.path.isfile(output_figure):
        os.remove(output_figure)

    # plt.tight_layout()

    # fig.text(.45, -.02, 'Predicted', size=16)
    # fig.text(-.02, .45, 'Actual', size=16, rotation='vertical')

    plt.savefig(output_figure, facecolor=fig.get_facecolor(), edgecolor='none')

    plt.close(fig)


def predict(overwrite, model_parameters, ignore_feas):

    """
    Predicts building density
    """

    out_dir = PARAMETERS['out dir']
    image2classify = PARAMETERS['features']

    model_dir = os.path.join(out_dir, 'models')
    bad_dir = os.path.join(out_dir, 'bad_feas')
    map_dir = os.path.join(out_dir, 'maps')

    if not os.path.isdir(map_dir):
        os.makedirs(map_dir)

    if model_parameters['classifier'] == 'Cubist':

        input_model = os.path.join(model_dir, '{}_training_master.tree'.format(model_parameters['classifier']))

        cl = classification_r()
        cl.construct_r_model(input_model=input_model)

    else:

        input_model = os.path.join(model_dir, '{}_training_master.txt'.format(model_parameters['classifier']))

        cl = classification()
        cl.construct_model(input_model=input_model)

    output_image = os.path.join(map_dir, 'building_density_{}.tif'.format(cl.classifier_info['classifier']))

    if overwrite:

        if os.path.isfile(output_image):
            os.remove(output_image)

    else:

        if os.path.isfile(output_image):
            return

    if ignore_feas:

        output_bad_features = os.path.join(bad_dir, 'training_master.txt')
        ignore_feas = pickle.load(file(output_bad_features, 'rb'))

    else:
        ignore_feas = []

    cl.predict(image2classify, output_image, in_model=input_model, ignore_feas=ignore_feas, n_jobs=-1)


def detect(setup=False, out_dir=None, patches=None, buildings=None, features=None, sample_size=1.,
           window_size=100, patch_id='Id', point_patch='PATCH', model_parameters=None,
           prep_features=False, prep_buildings=False, prep_train=False, rank_feas=False,
           construct_model=False, test_model=False, plot_predictions=False, what2plot='scatter',
           predict_buildings=False, overwrite=False, ignore_feas=False, top_feas=.5):

    """
    The main function

    Args:
        setup
        out_dir
        patches
        buildings
        features
        sample_size
        window_size
        patch_id
        point_patch
        model_parameters
        prep_features
        prep_buildings
        prep_train
        rank_feas
        construct_model
        test_model
        plot_predictions
        what2plot
        predict_buildings
        overwrite
        ignore_feas
        top_feas
    """

    global PARAMETERS

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    parameter_file = os.path.join(out_dir, 'parameters.yaml')

    if setup:

        i_info = raster_tools.rinfo(features)

        density_m = int(window_size / i_info.cellY)

        # Ensure odd sized window
        if density_m % 2 == 0:
            density_m -= 1

        parameters = {'out dir': out_dir,
                      'patches': patches,
                      'buildings': buildings,
                      'features': features,
                      'feature cell size': i_info.cellY,
                      'sample size': sample_size,
                      'window size': window_size,
                      'window size meters': density_m,
                      'patch id': patch_id,
                      'point patch id': point_patch}

        i_info.close()

        if os.path.isfile(parameter_file):
            os.remove(parameter_file)

        with open(parameter_file, 'w') as pf:
            pf.write(yaml.dump(parameters, default_flow_style=False))

    else:

        # Load the parameters
        with open(parameter_file, 'r') as pf:
            PARAMETERS = yaml.load(pf)

        if prep_features:
            prep_image_features(overwrite)

        if prep_buildings:
            buildings2density(overwrite)

        if prep_train:
            prep_building_training()

        if rank_feas:
            rank_patch_features(model_parameters, top_feas)

        if construct_model:
            train(model_parameters, overwrite, ignore_feas)

        if test_model:
            test(model_parameters, ignore_feas)

        if plot_predictions:
            plot(model_parameters, what2plot=what2plot)

        if predict_buildings:
            predict(overwrite, model_parameters, ignore_feas)


def _examples():

    sys.exit("""\

    # (1) Setup the parameters
    density --setup -o /building_density --patches /patches.shp --buildings /buildings.shp --features /features.vrt -s 1. -w 100

    # (2) Prepare image features for each patch
    density --prep-features -o /building_density

    # (3) Prepare sampled buildings for patch density
    density --prep-buildings -o /building_density

    # (4) Prepare building density samples from image features
    density --prep-train -o /building_density

    # Rank feature variable importance
    density --rank-feas -o /building_density

    # (5) Train a regression model
    density --train -o /building_density --parameters "{'classifier': 'RFR', 'trees': 500}"

    # (6) Test a regression model
    density --test -o /building_density --parameters "{'classifier': 'RFR'}"

    # (7) Plot the predictions of a regression model
    density --plot --what2plot scatter -o /building_density --parameters "{'classifier': 'RFR'}"

    # (8) Predict building density (choose the model to use)
    density --predict -o /building_density --parameters "{'classifier': 'RFR'}"

    """)


def main():

    parser = argparse.ArgumentParser(description='Building density estimates',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-e', '--examples', dest='examples', action='store_true', help='Show usage examples and exit')
    parser.add_argument('--setup', dest='setup', help='Whether to setup parameters', action='store_true')
    parser.add_argument('-o', '--out-dir', dest='out_dir', help='The output base directory', default=None)
    parser.add_argument('--patches', dest='patches', help='The patches shapefile', default=None)
    parser.add_argument('--buildings', dest='buildings', help='The buildings shapefile', default=None)
    parser.add_argument('--features', dest='features', help='The features mosaic', default=None)
    parser.add_argument('-s', dest='sample_size', help='The sample size at which buildings were collected',
                        default=1., type=float)
    parser.add_argument('-w', '--window-size', dest='window_size', help='The window size (in meters)',
                        default=100, type=int)
    parser.add_argument('-id', '--patch-id', dest='patch_id', help='The patch unique id', default='Id')
    parser.add_argument('-pid', '--point-patch', dest='point_patch',
                        help='The patch Id field name in the point shapefile.', default='PATCH')
    parser.add_argument('--prep-features', dest='prep_features', help='Whether to prep the image feature patches',
                        action='store_true')
    parser.add_argument('--prep-buildings', dest='prep_buildings', help='Whether to prep the building points',
                        action='store_true')
    parser.add_argument('--prep-train', dest='prep_train', help='Whether to prep the building densities for models',
                        action='store_true')
    parser.add_argument('--rank-feas', dest='rank_feas', help='Whether to rank feature importance',
                        action='store_true')
    parser.add_argument('--train', dest='construct_model', help='Whether to train the regression model',
                        action='store_true')
    parser.add_argument('--test', dest='test_model', help='Whether to test the regression model',
                        action='store_true')
    parser.add_argument('--plot', dest='plot_predictions', help='Whether to plot the predictions for each patch',
                        action='store_true')
    parser.add_argument('--what2plot', dest='what2plot', help='The variable to plot', default='scatter',
                        choices=['scatter', 'patches'])
    parser.add_argument('--parameters', dest='model_parameters', help='The regression model parameters',
                        default="{'classifier': 'ABR_EX_DTR', 'trials': 50, 'max_depth': 25, 'min_samps': 2}")
    parser.add_argument('--predict', dest='predict_buildings', help='Whether to predict building density',
                        action='store_true')
    parser.add_argument('--overwrite', dest='overwrite', help='Whether to overwrite outputs', action='store_true')
    parser.add_argument('--ignore-feas', dest='ignore_feas', help='Whether to ignore bad features', action='store_true')
    parser.add_argument('--top-feas', dest='top_feas', help='The percentage of top features to include',
                        default=.5, type=float)

    args = parser.parse_args()

    if args.examples:
        _examples()

    start_time = time.time()

    detect(setup=args.setup, out_dir=args.out_dir, patches=args.patches,
           buildings=args.buildings, features=args.features, sample_size=args.sample_size,
           window_size=args.window_size, patch_id=args.patch_id, point_patch=args.point_patch,
           prep_features=args.prep_features, prep_buildings=args.prep_buildings, prep_train=args.prep_train,
           rank_feas=args.rank_feas, construct_model=args.construct_model,
           model_parameters=ast.literal_eval(args.model_parameters),
           test_model=args.test_model, plot_predictions=args.plot_predictions, what2plot=args.what2plot,
           predict_buildings=args.predict_buildings, overwrite=args.overwrite,
           ignore_feas=args.ignore_feas, top_feas=args.top_feas)

    print('\nEnd data & time -- (%s)\nTotal processing time -- (%.2gs)\n' %
          (time.asctime(time.localtime(time.time())), (time.time() - start_time)))


if __name__ == '__main__':
    main()
