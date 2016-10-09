#!/usr/bin/env python

import os
import sys
import time
import platform
import copy
import psutil
import itertools

import sputilities
import spsplit
import spreshape
import raster_tools
from veg_indices import VegIndicesEquations

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


def run(parameter_object):

    """
    Args:
        input_image, output_dir, band_positions=[1], rgb2gray=None, block=2, scales=[8], triggers=['mean'],
        threshold=20, min_len=10, line_gap=2, weighted=False, sfs_thresh=80, resamp_sfs=0., n_angles=8,
        equalize=False, equalize_adapt=False, smooth=0, visualize=False, convert_stk=False, gdal_cache=256,
        do_pca=False, stack_feas=True, stack_only=False, band_red=3, band_nir=4, neighbors=False, n_jobs=-1,
        reset_sects=False, image_max=0, lac_r=2, section_size=8000, chunk_size=512
    """

    sputilities.parameter_checks(parameter_object)

    # Setup the log file.
    if os.path.isfile(parameter_object.log_txt):

        with open(parameter_object.log_txt, 'rb') as log_txt_wr:
            log_hist = log_txt_wr.readlines()

        log_txt_wr = open(parameter_object.log_txt, 'wb')
        log_txt_wr.writelines(log_hist)

    else:
        log_txt_wr = open(parameter_object.log_txt, 'wb')

    log_txt_wr.writelines("""\
    =================================================
    Start date & time --- ({})
    =================================================
    Input image: {}
    Output directory: {}
    Bands: {}
    Smoothing: {:d}
    Block size: {:d}
    Scales: {}
    Feature triggers: {}
    SFS stopping threshold: {:d}
    SFS angles: {:d}
    Red band position: {:d}
    NIR band position: {:d}
    {} compute features as neighbors
    {} perform histogram equalization
    {} perform adaptive histogram equalization
    """.format(time.asctime(time.localtime(time.time())), parameter_object.input_image, parameter_object.output_dir,
               parameter_object.rgb2write, parameter_object.smooth, parameter_object.block,
               ','.join([str(bpos) for bpos in parameter_object.scales]), ','.join(parameter_object.triggers),
               parameter_object.sfs_thresh, parameter_object.n_angles, parameter_object.band_red,
               parameter_object.band_nir, parameter_object.write_neighbors, parameter_object.write_equalize,
               parameter_object.write_equalize_adapt))

    log_txt_wr.close()

    if platform.system() == 'Windows':

        new_feas_list = '{}/{}_win_feas_list.txt'.format(parameter_object.output_dir, parameter_object.f_base)

        win_feas_list_o = open(new_feas_list, 'w')

    else:
        new_feas_list = []

    if parameter_object.stack_only:

        for trigger in parameter_object.triggers:

            # Set the output features folder.
            feas_dir, parameter_object = sputilities.set_feas_dir(parameter_object, trigger)

            for band_p in parameter_object.band_positions:

                # get feature names
                obds = 1
                for scale in parameter_object.scales:

                    for feature in xrange(1, parameter_object.features_dict[trigger]+1):

                        out_img, out_img_base = sputilities.scale_fea_check(trigger, feas_dir, band_p, scale, feature,
                                                                            parameter_object)

                        # skip the feature if it doesn't exist
                        if not os.path.isfile(out_img):
                            continue

                        # append new features to a list to stack
                        if platform.system() == 'Windows':
                            win_feas_list_o.write('%s\n' % out_img)
                        else:
                            new_feas_list.append(out_img)

                        obds += 1

        if platform.system() == 'Windows':
            win_feas_list_o.close()

        # If prompted, stack features only.
        out_vrt = sputilities.stack_features(parameter_object, new_feas_list)

    else:

        trigger_orig_seg = False

        # Iterate over each feature trigger.
        for trigger in parameter_object.triggers:

            # output features folder# Set the output features folder.
            feas_dir, parameter_object = sputilities.set_feas_dir(parameter_object, trigger)

            for band_p in parameter_object.band_positions:

                # get image information
                i_info = raster_tools.rinfo(parameter_object.input_image)

                # check available memory
                avilable_space = psutil.virtual_memory().available * 9.53674e-7
                # free = psutil.virtual_memory().free * 9.53674e-7

                # There seems to be bugs with GDAL
                #   get statistics, so we will try
                #   to avoid it if possible.
                #
                # 8,000 row x 8,000 column image
                # 8 GB memory
                rows_and_cols = i_info.rows * i_info.cols

                if (rows_and_cols < 25000000) or (rows_and_cols < 64000000) and (avilable_space > 8000):

                    if isinstance(parameter_object.rgb2gray, str):

                        __, mn, mx = sputilities.convert_rgb2gray(i_info, 0, 0, 0, 0,
                                                                  rgb=parameter_object.rgb2gray, stats=True)

                    else:

                        if parameter_object.image_max > 0:
                            mx = parameter_object.image_max
                            mn = 0
                        else:
                            mx = i_info.mparray(bands2open=band_p).max()
                            mn = i_info.mparray(bands2open=band_p).min()

                        # mx = i_info.datasource.GetRasterBand(1).ReadAsArray(0, 0).max()
                        # mn = i_info.datasource.GetRasterBand(1).ReadAsArray(0, 0).min()

                else:

                    if isinstance(parameter_object.rgb2gray, str):

                        __, mn, mx = sputilities.convert_rgb2gray(i_info, 0, 0, 0, 0,
                                                                  rgb=parameter_object.rgb2gray, stats=True)

                    else:

                        if parameter_object.image_max > 0:
                            mx = parameter_object.image_max
                            mn = 0
                        else:
                            mn, mx, mnn, stdev = i_info.datasource.GetRasterBand(band_p).GetStatistics(1, 1)

                # Get section and chunk size.
                sect_row_size, sect_col_size = sputilities.get_sect_chunk_size(i_info, parameter_object.section_size)

                # Create the output feature bands.
                obds = 1
                for scale in parameter_object.scales:

                    for feature in xrange(1, parameter_object.features_dict[trigger] + 1):

                        in_trig_name = copy.copy(trigger)

                        out_img, out_img_base = sputilities.scale_fea_check(trigger, feas_dir, band_p, scale, feature,
                                                                            parameter_object)

                        # status dictionary
                        if os.path.isfile(parameter_object.status_dict_txt):

                            # open the status dictionary
                            with open(parameter_object.status_dict_txt, 'rb') as status_dict_txt_o:

                                # pickle the status dictionary
                                status_dict = pickle.load(status_dict_txt_o)

                            # get the feature status
                            try:
                                feature_status = status_dict[out_img_base]
                            except:
                                status_dict[out_img_base] = -999
                                feature_status = -999

                            status_dict_txt_o = open(parameter_object.status_dict_txt, 'wb')

                        else:

                            # create the status dictionary
                            status_dict_txt_o = open(parameter_object.status_dict_txt, 'wb')

                            status_dict = dict()

                            # set the layer feature status as non-existent
                            status_dict[out_img_base] = -999

                            feature_status = -999

                        if parameter_object.reset_sects:
                            feature_status = -999

                        # append new features to a list to stack
                        if platform.system() == 'Windows':
                            win_feas_list_o.write('{}\n'.format(out_img))
                        else:
                            new_feas_list.append(out_img)

                        # only create a new feature if the file does not exist
                        if feature_status == -999:

                            sputilities.create_band(i_info, out_img, parameter_object, 1)

                            # set the status as created
                            status_dict[out_img_base] = 0

                        # pickle the status dictionary
                        pickle.dump(status_dict, status_dict_txt_o)

                        status_dict_txt_o.close()

                        obds += 1

                # Get the number of sections in
                #   the image (only used as a counter).
                n_row_sects = len([i_sect for i_sect in xrange(0, i_info.rows,
                                                               sect_row_size - (parameter_object.scales[-1] -
                                                                                parameter_object.block))])

                n_col_sects = len([j_sect for j_sect in xrange(0, i_info.cols,
                                                               sect_col_size - (parameter_object.scales[-1] -
                                                                                parameter_object.block))])

                n_sects = len([j_sect for (i_sect, j_sect) in
                               itertools.product(xrange(0, i_info.rows,
                                                        sect_row_size - (parameter_object.scales[-1] -
                                                                         parameter_object.block)),
                                                 xrange(0, i_info.cols,
                                                        sect_col_size - (parameter_object.scales[-1] -
                                                                         parameter_object.block)))])

                # Here we loop through the
                #   image by sections.
                n_sect = 1
                i_sect_ctr = 0
                i_sect_blk_ctr = 1

                for i_sect in xrange(0, i_info.rows, sect_row_size -
                        (parameter_object.scales[-1] - parameter_object.block)):

                    numRws = raster_tools.n_rows_cols(i_sect, sect_row_size, i_info.rows)

                    j_sect_ctr = 0
                    j_sect_blk_ctr = 1

                    for j_sect in xrange(0, i_info.cols, sect_col_size -
                            (parameter_object.scales[-1] - parameter_object.block)):

                        print '\nSection {:d} of {:d} ...'.format(n_sect, n_sects)

                        numCols = raster_tools.n_rows_cols(j_sect, sect_col_size, i_info.cols)

                        #####################################
                        # Check if the section has been
                        # processed for all feature scales.
                        #####################################

                        sects_good = True

                        obds = 1
                        for scale in parameter_object.scales:

                            for feature in xrange(1, parameter_object.features_dict[trigger]+1):

                                if trigger_orig_seg:
                                    in_trig_name = 'seg'
                                else:
                                    in_trig_name = copy.copy(trigger)

                                out_img, out_img_base = sputilities.scale_fea_check(trigger, feas_dir, band_p, scale,
                                                                                    feature, parameter_object)

                                # Open the status dictionary.
                                with open(parameter_object.status_dict_txt, 'rb') as status_dict_txt_o:

                                    # Pickle the status dictionary.
                                    status_dict = pickle.load(status_dict_txt_o)

                                sect_status = status_dict[out_img_base]

                                obds += 1

                                # If any of the sections are not current,
                                #   continue with feature extraction.
                                if sect_status < n_sect:
                                    sects_good = False

                        # Open the image array.
                        if trigger == 'ndvi':

                            sect_in = i_info.mparray(bands2open=[parameter_object.band_red, parameter_object.band_nir],
                                                     i=i_sect, j=j_sect, rows=numRws, cols=numCols, d_type='float32')

                            vie = VegIndicesEquations(sect_in, chunk_size=-1)
                            sect_in = vie.compute('NDVI', out_type=2)

                            mn = 0
                            mx = 255

                        elif trigger == 'dmp':

                            sect_in = np.asarray([i_info.mparray(bands2open=dmp_bd, i=i_sect, j=j_sect,
                                                                 rows=numRws, cols=numCols, d_type='float32')
                                                  for dmp_bd in xrange(1, i_info.bands+1)]).reshape(i_info.bands,
                                                                                                    numRws, numCols)

                        elif isinstance(parameter_object.rgb2gray, str):

                            sect_in, __, __ = sputilities.convert_rgb2gray(i_info, j_sect, i_sect, numCols, numRws,
                                                                           rgb=parameter_object.rgb2gray)

                        else:

                            sect_in = i_info.mparray(bands2open=band_p, i=i_sect, j=j_sect,
                                                     rows=numRws, cols=numCols)

                        # pad array here
                        # (top, bottom), (left, right)

                        # pad left and top
                        if parameter_object.scales[-1] != parameter_object.block:

                            pad_len = (parameter_object.scales[-1] / 2) - (parameter_object.block / 2)

                            if (i_sect_blk_ctr == 1) and (j_sect_blk_ctr == 1):

                                if trigger == 'dmp':

                                    sect_in = np.asarray([np.pad(sect_in[pos], ((pad_len, 0), (pad_len, 0)), 'wrap')
                                                          for pos in xrange(0, sect_in.shape[0])]).reshape(sect_in.shape[0],
                                                                                                           numRws+pad_len,
                                                                                                           numCols+pad_len)

                                else:
                                    sect_in = np.pad(sect_in, ((pad_len, 0), (pad_len, 0)), 'wrap')

                            # pad top only
                            elif (i_sect_blk_ctr == 1) and (j_sect_blk_ctr > 1) and (j_sect_blk_ctr < n_col_sects):

                                if trigger == 'dmp':

                                    sect_in = np.asarray([np.pad(sect_in[pos], ((pad_len, 0), (0, 0)), 'wrap')
                                                          for pos in xrange(0, sect_in.shape[0])]).reshape(sect_in.shape[0],
                                                                                                           numRws+pad_len, numCols)

                                else:
                                    sect_in = np.pad(sect_in, ((pad_len, 0), (0, 0)), 'wrap')

                            # pad top and right
                            elif (i_sect_blk_ctr == 1) and (j_sect_blk_ctr == n_col_sects):

                                if trigger == 'dmp':

                                    sect_in = np.asarray([np.pad(sect_in[pos], ((pad_len, 0), (0, pad_len)), 'wrap')
                                                          for pos in xrange(0, sect_in.shape[0])]).reshape(sect_in.shape[0],
                                                                                                           numRws+pad_len, numCols+pad_len)

                                else:
                                    sect_in = np.pad(sect_in, ((pad_len, 0), (0, pad_len)), 'wrap')

                            # pad left only
                            elif (i_sect_blk_ctr > 1) and (i_sect_blk_ctr < n_row_sects) and (j_sect_blk_ctr == 1):

                                if trigger == 'dmp':

                                    sect_in = np.asarray([np.pad(sect_in[pos], ((pad_len, 0), (pad_len, 0)), 'wrap')
                                                          for pos in xrange(0, sect_in.shape[0])]).reshape(sect_in.shape[0],
                                                                                                           numRws+pad_len, numCols+pad_len)

                                else:
                                    sect_in = np.pad(sect_in, ((0, 0), (pad_len, 0)), 'wrap')

                            # pad right only
                            elif (i_sect_blk_ctr > 1) and (i_sect_blk_ctr < n_row_sects) and (j_sect_blk_ctr == n_col_sects):

                                if trigger == 'dmp':

                                    sect_in = np.asarray([np.pad(sect_in[pos], ((0, 0), (0, pad_len)), 'wrap')
                                                          for pos in xrange(0, sect_in.shape[0])]).reshape(sect_in.shape[0],
                                                                                                           numRws, numCols+pad_len)

                                else:
                                    sect_in = np.pad(sect_in, ((0, 0), (0, pad_len)), 'wrap')

                            # pad left and bottom
                            elif (i_sect_blk_ctr == n_row_sects) and (j_sect_blk_ctr == 1):

                                if trigger == 'dmp':

                                    sect_in = np.asarray([np.pad(sect_in[pos], ((0, pad_len), (pad_len, 0)), 'wrap')
                                                          for pos in xrange(0, sect_in.shape[0])]).reshape(sect_in.shape[0],
                                                                                                           numRws+pad_len, numCols+pad_len)

                                else:
                                    sect_in = np.pad(sect_in, ((0, pad_len), (pad_len, 0)), 'wrap')

                            # pad bottom only
                            elif (i_sect_blk_ctr == n_row_sects) and (j_sect_blk_ctr > 1) and (j_sect_blk_ctr < n_col_sects):

                                if trigger == 'dmp':

                                    sect_in = np.asarray([np.pad(sect_in[pos], ((0, pad_len), (0, 0)), 'wrap')
                                                          for pos in xrange(0, sect_in.shape[0])]).reshape(sect_in.shape[0],
                                                                                                           numRws+pad_len, numCols)

                                else:
                                    sect_in = np.pad(sect_in, ((0, pad_len), (0, 0)), 'wrap')

                            # pad right and bottom
                            elif (i_sect_blk_ctr == n_row_sects) and (j_sect_blk_ctr == n_col_sects):

                                if trigger == 'dmp':

                                    sect_in = np.asarray([np.pad(sect_in[pos], ((0, pad_len), (0, pad_len)), 'wrap')
                                                          for pos in xrange(0,
                                                                            sect_in.shape[0])]).reshape(sect_in.shape[0],
                                                                                                        numRws+pad_len,
                                                                                                        numCols+pad_len)

                                else:
                                    sect_in = np.pad(sect_in, ((0, pad_len), (0, pad_len)), 'wrap')

                        if trigger == 'dmp':

                            lRows, lCols = sect_in[0].shape

                            oR, oC, out_rows, out_cols = spsplit.get_out_dims(np.uint8(sect_in[0]), lRows, lCols,
                                                                              parameter_object)
                        else:
                            lRows, lCols = sect_in.shape

                            oR, oC, out_rows, out_cols = spsplit.get_out_dims(np.uint8(sect_in), lRows, lCols,
                                                                              parameter_object)

                        # Only extract features if the section hasn't
                        #   been completed or if the section does not
                        #   contain all zeros.
                        if sects_good or sect_in.max() == 0:
                            pass
                        else:

                            # Here we split the current section into
                            #   chunks and process the features.

                            # Split image and compute features.
                            tk = spsplit.get_sect_feas(sect_in, lRows, lCols, mn, mx, i_info.cellY, trigger,
                                                       parameter_object)

                            # Reshape list of features into
                            #   <features x rows x columns> array.
                            out_sect_arr = spreshape.reshape_feas(trigger, tk, oR, oC, lRows,
                                                                  lCols, out_rows, out_cols, parameter_object)

                            print '  Writing features to file ...'

                            obds = 1
                            for scale in parameter_object.scales:

                                for feature in xrange(1, parameter_object.features_dict[trigger]+1):

                                    if trigger_orig_seg:
                                        in_trig_name = 'seg'
                                    else:
                                        in_trig_name = copy.copy(trigger)

                                    out_img, out_img_base = sputilities.scale_fea_check(trigger, feas_dir, band_p,
                                                                                        scale, feature,
                                                                                        parameter_object)

                                    o_info = raster_tools.rinfo(out_img, open2read=False)
                                    out_band_obj = o_info.datasource.GetRasterBand(1)

                                    # write array to file
                                    if (i_sect_blk_ctr < n_row_sects) and (j_sect_blk_ctr == n_col_sects):

                                        try:
                                            out_band_obj.WriteArray(out_sect_arr[obds-1], j_sect_ctr, i_sect_ctr)
                                        except:
                                            out_band_obj.WriteArray(out_sect_arr[obds-1], j_sect_ctr-1, i_sect_ctr)

                                    elif (i_sect_blk_ctr == n_row_sects) and (j_sect_blk_ctr < n_col_sects):

                                        try:
                                            out_band_obj.WriteArray(out_sect_arr[obds-1], j_sect_ctr, i_sect_ctr)
                                        except:
                                            out_band_obj.WriteArray(out_sect_arr[obds-1], j_sect_ctr, i_sect_ctr-1)

                                    elif (i_sect_blk_ctr == n_row_sects) and (j_sect_blk_ctr == n_col_sects):

                                        try:
                                            out_band_obj.WriteArray(out_sect_arr[obds-1], j_sect_ctr, i_sect_ctr)
                                        except:
                                            out_band_obj.WriteArray(out_sect_arr[obds-1], j_sect_ctr-1, i_sect_ctr-1)

                                    else:
                                        out_band_obj.WriteArray(out_sect_arr[obds-1], j_sect_ctr, i_sect_ctr)

                                    out_band_obj = None

                                    o_info.close()

                                    o_info = None

                                    obds += 1

                            ###################################################
                            # Update status dictionary with the section number.
                            ###################################################

                            obds_t = 1
                            for scale in parameter_object.scales:

                                for feature in xrange(1, parameter_object.features_dict[trigger]+1):

                                    if trigger_orig_seg:
                                        in_trig_name = 'seg'
                                    else:
                                        in_trig_name = copy.copy(trigger)

                                    out_img, out_img_base = sputilities.scale_fea_check(trigger, feas_dir, band_p,
                                                                                        scale, feature,
                                                                                        parameter_object)

                                    # open the status dictionary
                                    with open(parameter_object.status_dict_txt, 'rb') as status_dict_txt_o:

                                        # pickle the status dictionary
                                        status_dict = pickle.load(status_dict_txt_o)

                                    status_dict[out_img_base] = n_sect

                                    # open the status dictionary
                                    with open(parameter_object.status_dict_txt, 'wb') as status_dict_txt_o:

                                        # pickle the status dictionary
                                        pickle.dump(status_dict, status_dict_txt_o)

                                    obds_t += 1

                            del tk, oR, oC

                        j_sect_ctr += out_cols
                        j_sect_blk_ctr += 1

                        n_sect += 1

                    i_sect_ctr += out_rows
                    i_sect_blk_ctr += 1

                obds = 1
                for scale in parameter_object.scales:

                    for feature in xrange(1, parameter_object.features_dict[trigger]+1):

                        if trigger_orig_seg:
                            in_trig_name = 'seg'
                        else:
                            in_trig_name = copy.copy(trigger)

                        out_img, out_img_base = sputilities.scale_fea_check(trigger, feas_dir, band_p,
                                                                            scale, feature,
                                                                            parameter_object)

                        # Resample the SFS image.
                        #
                        # SFS radiates from a center pixel, so is
                        #   more useful when computed with a
                        #   smaller block size.
                        if hasattr(parameter_object, 'sfs_resample'):

                            if parameter_object.sfs_resample > 0:

                                out_img_d_name, out_img_f_name = os.path.split(out_img)

                                out_img_resamp = '{}/{}_resamp{}'.format(out_img_d_name, out_img_base,
                                                                         parameter_object.f_ext)

                                # Replace the block size.
                                out_img_resamp = out_img_resamp.replace('blk{:d}'.format(parameter_object.block),
                                                                        'blk{:d}'.format(int(parameter_object.sfs_resample)))

                                print '\nResampling SFS to {:.1f}m x {:.1f}m cell size ...\n'.format(parameter_object.sfs_resample,
                                                                                                     parameter_object.sfs_resample)

                                if 'img' in parameter_object.f_ext.lower():

                                    raster_tools.warp(out_img, out_img_resamp,
                                                      cell_size=parameter_object.sfs_resample,
                                                      resampleAlg='average',
                                                      warpMemoryLimit=256,
                                                      format='HFA',
                                                      multithread=True,
                                                      creationOptions=['COMPRESS=YES'])

                                    # sfs_resamp_com = 'gdalwarp -multi -wo NUM_THREADS=ALL_CPUS \
                                    # --config GDAL_CACHEMAX {:d} -co COMPRESS=YES \
                                    # -of HFA -tr {:f} {:f} -r average {} {}'.format(parameter_object.gdal_cache,
                                    #                                                parameter_object.sfs_resample,
                                    #                                                parameter_object.sfs_resample,
                                    #                                                out_img, out_img_resamp)

                                else:

                                    raster_tools.warp(out_img, out_img_resamp,
                                                      cell_size=parameter_object.sfs_resample,
                                                      resampleAlg='average',
                                                      warpMemoryLimit=256,
                                                      format='HFA',
                                                      multithread=True,
                                                      creationOptions=['COMPRESS=DEFLATE', 'BIGTIFF=YES', 'TILED=YES'])

                                    # sfs_resamp_com = 'gdalwarp -multi -wo NUM_THREADS=ALL_CPUS \
                                    #                                 --config GDAL_CACHEMAX {:d} -co COMPRESS=DEFLATE \
                                    #                                 -co TILED=YES -co BIGTIFF=YES -tr {:f} {:f} \
                                    #                                 -r average {} {}'.format(parameter_object.gdal_cache,
                                    #                                                          parameter_object.sfs_resample,
                                    #                                                          parameter_object.sfs_resample,
                                    #                                                          out_img, out_img_resamp)

                                # subprocess.call(sfs_resamp_com, shell=True)

                                out_img_new = out_img_resamp.replace('_resamp', '')

                                # Replace the block size.
                                out_img_new = out_img_new.replace('blk{:d}'.format(int(parameter_object.sfs_resample)),
                                                                  'blk{:d}'.format(parameter_object.block))

                                os.remove(out_img)

                                os.rename(out_img_resamp, out_img_new)

                        obds += 1

        if platform.system() == 'Windows':
            win_feas_list_o.close()

        # Stack the features
        if hasattr(parameter_object, 'stack'):
            if parameter_object.stack:
                out_vrt = sputilities.stack_features(parameter_object, new_feas_list)

        # Optional conversion to GeoTiff.
        if hasattr(parameter_object, 'convert'):
                
            if parameter_object.convert:

                scales_str = [str(sc) for sc in parameter_object.scales]
                band_pos_str = [str(bp) for bp in parameter_object.band_positions]

                out_gtiff = '{}/{}.{}.stk.bd{}.block{}.scales{}.tif'.format(parameter_object.output_dir,
                                                                            parameter_object.f_base,
                                                                            '-'.join(parameter_object.triggers),
                                                                            '-'.join(band_pos_str),
                                                                            parameter_object.block,
                                                                            '-'.join(scales_str))

                raster_tools.translate(out_vrt, out_gtiff,
                                       format='GTiff',
                                       creationOptions=['TILED=YES', 'COMPRESS=LZW'])

                # com = 'gdal_translate --config GDAL_CACHEMAX {:d} \
                # -of GTiff -co TILED=YES -co COMPRESS=LZW {} {}'.format(parameter_object.gdal_cache, out_vrt, out_gtiff)

                # subprocess.call(com, shell=True)

        # Run PCA on features.
        # if parameter_object.pca:
        #
        #     out_pca = '{}/{}.{}.stk.bd{}.block{}.scales{}_pca.tif'.format(parameter_object.output_dir,
        #                                                                   parameter_object.f_base,
        #                                                                   '-'.join(parameter_object.triggers),
        #                                                                   '-'.join(band_pos_str),
        #                                                                   parameter_object.block,
        #                                                                   '-'.join(scales_str))
        #
        #     # Execute PCA.
        #     pca(out_vrt, out_pca, do_pca)