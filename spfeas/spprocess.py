#!/usr/bin/env python

import os
import sys
import time
import platform
import copy
import itertools

from .sphelpers import sputilities
from . import spsplit
from .sphelpers import spreshape

from mpglue import raster_tools, VegIndicesEquations

# YAML
try:
    import yaml
except ImportError:
    raise ImportError('YAML must be installed')

# NumPy
try:
    import numpy as np
except ImportError:
    raise ImportError('NumPy must be installed')


def run(parameter_object):

    """
    Args:
        input_image, output_dir, band_positions=[1], use_rgb=False, block=2, scales=[8], triggers=['mean'],
        threshold=20, min_len=10, line_gap=2, weighted=False, sfs_thresh=80, resamp_sfs=0.,
        equalize=False, equalize_adapt=False, smooth=0, visualize=False, convert_stk=False, gdal_cache=256,
        do_pca=False, stack_feas=True, stack_only=False, band_red=3, band_nir=4, neighbors=False, n_jobs=-1,
        reset_sects=False, image_max=0, lac_r=2, section_size=8000, chunk_size=512
    """

    sputilities.parameter_checks(parameter_object)

    # Write the parameters to file.
    sputilities.write_log(parameter_object)

    new_feas_list = []

    n_jobs_orig = parameter_object.n_jobs

    if parameter_object.stack_only:

        # If prompted, stack features without processing.
        parameter_object = sputilities.stack_features(parameter_object, new_feas_list)

    else:

        trigger_orig_seg = False

        # Iterate over each feature trigger.
        for trigger in parameter_object.triggers:

            if trigger in []:
                parameter_object.n_jobs = 1
            else:
                parameter_object.n_jobs = copy.copy(n_jobs_orig)

            parameter_object.update_info(trigger=trigger)

            # Set the output features folder.
            parameter_object = sputilities.set_feas_dir(parameter_object)

            # Iterate over each band
            for band_position in parameter_object.band_positions:

                parameter_object.update_info(band_position=band_position)

                # Get image information
                i_info = raster_tools.ropen(parameter_object.input_image)

                # Check if any of the bands are corrupted.
                i_info.check_corrupted_bands()

                if i_info.corrupted_bands:

                    print
                    print('The following bands appear to be corrupted:')
                    print ', '.join(i_info.corrupted_bands)

                    return

                # Get image statistics.
                parameter_object = sputilities.get_stats(i_info, parameter_object)

                # Get section and chunk size.
                parameter_object = sputilities.get_sect_chunk_size(i_info, parameter_object)

                if parameter_object.trigger == 'sfsorf':
                    parameter_object.update_info(scale=parameter_object.scales[-1],
                                                 feature=100)
                    parameter_object = sputilities.scale_fea_check(parameter_object)
                    spsplit.sfs_orfeo(parameter_object)
                    if os.path.isfile(parameter_object.out_img):
                        os.remove(parameter_object.out_img)
                    continue

                # Create the output feature bands.
                obds = 1
                for scale in parameter_object.scales:

                    # TODO: Temporary hack for gabor kernel size above scale size
                    if (trigger in ['gabor']) and scale < 16:
                        continue

                    parameter_object.update_info(scale=scale)

                    for feature in xrange(1, parameter_object.features_dict[parameter_object.trigger]+1):

                        parameter_object.update_info(feature=feature)

                        parameter_object = sputilities.scale_fea_check(parameter_object)

                        # Set the status dictionary.
                        parameter_object = sputilities.set_status(parameter_object)

                        new_feas_list.append(parameter_object.out_img)

                        # only create a new feature if the file does not exist
                        if parameter_object.feature_status == -999:

                            sputilities.create_band(i_info, parameter_object, 1)

                            # set the status as created
                            parameter_object.status_dict[parameter_object.out_img_base] = 0

                        # Store the status dictionary
                        with open(parameter_object.status_dict_txt, 'w') as pf:

                            pf.write(yaml.dump(parameter_object.status_dict,
                                               default_flow_style=False))

                        obds += 1

                # Get the number of sections in
                #   the image (only used as a counter).
                parameter_object = sputilities.get_n_sects(i_info, parameter_object)

                # Here we iterate over the image by sections.
                n_sect = 1
                i_sect_ctr = 0

                parameter_object.update_info(i_sect_blk_ctr=1)

                for i_sect in xrange(0, i_info.rows, parameter_object.sect_row_size -
                        (parameter_object.scales[-1] - parameter_object.block)):

                    n_rows = raster_tools.n_rows_cols(i_sect, parameter_object.sect_row_size, i_info.rows)

                    j_sect_ctr = 0

                    parameter_object.update_info(j_sect_blk_ctr=1)

                    for j_sect in xrange(0, i_info.cols, parameter_object.sect_col_size -
                            (parameter_object.scales[-1] - parameter_object.block)):

                        print('\nSection {:d} of {:d} ...'.format(n_sect, parameter_object.n_sects))

                        n_cols = raster_tools.n_rows_cols(j_sect, parameter_object.sect_col_size, i_info.cols)

                        ###################################
                        # Check if the section has been
                        # processed for all feature scales.
                        ###################################

                        sects_good = True

                        obds = 1
                        for scale in parameter_object.scales:

                            # TODO: Temporary hack for gabor kernel size above scale size
                            if (trigger in ['gabor']) and scale < 16:
                                continue

                            parameter_object.update_info(scale=scale)

                            for feature in xrange(1, parameter_object.features_dict[parameter_object.trigger]+1):

                                parameter_object.update_info(feature=feature)

                                parameter_object = sputilities.scale_fea_check(parameter_object)

                                # Open the status dictionary.
                                with open(parameter_object.status_dict_txt, 'r') as pf:
                                    parameter_object.status_dict = yaml.load(pf)

                                sect_status = parameter_object.status_dict[parameter_object.out_img_base]

                                obds += 1

                                # If any of the sections are not current,
                                #   continue with feature extraction.
                                if sect_status < n_sect:
                                    sects_good = False

                        # Open the image array.
                        # TODO: add other indices
                        if parameter_object.trigger in ['ndvi', 'evi2']:

                            sect_in = i_info.read(bands2open=[parameter_object.band_red,
                                                              parameter_object.band_nir],
                                                  i=i_sect, j=j_sect,
                                                  rows=n_rows, cols=n_cols,
                                                  d_type='float32')

                            vie = VegIndicesEquations(sect_in, chunk_size=-1)
                            sect_in = vie.compute(parameter_object.trigger.upper(), out_type=2)

                            parameter_object.min = 0
                            parameter_object.max = 255

                        elif parameter_object.trigger == 'dmp':

                            sect_in = np.asarray([i_info.read(bands2open=dmp_bd,
                                                              i=i_sect, j=j_sect,
                                                              rows=n_rows, cols=n_cols,
                                                              d_type='float32')
                                                  for dmp_bd in xrange(1, i_info.bands+1)]).reshape(i_info.bands,
                                                                                                    n_rows, n_cols)

                        elif parameter_object.trigger == 'saliency':

                            # parameter_object.update_info(min=0, max=255)
                            sect_in = spsplit.saliency(i_info, parameter_object,
                                                       i_sect, j_sect,
                                                       n_rows, n_cols)

                        elif parameter_object.use_rgb and trigger not in ['ndvi', 'evi2', 'dmp', 'saliency']:

                            sect_in, __, __ = sputilities.convert_rgb2gray(i_info,
                                                                           i_sect, j_sect,
                                                                           n_rows, n_cols)

                        else:

                            sect_in = i_info.read(bands2open=parameter_object.band_position,
                                                  i=i_sect, j=j_sect,
                                                  rows=n_rows, cols=n_cols)

                        if parameter_object.trigger == 'orb':
                            sect_in = spsplit.get_orb_keypoints(sect_in, parameter_object)
                            parameter_object.update_info(min=0, max=255)

                        # pad array here
                        # (top, bottom), (left, right)
                        sect_in = sputilities.pad_array(parameter_object, sect_in, n_rows, n_cols)

                        if parameter_object.trigger == 'dmp':

                            l_rows, l_cols = sect_in[0].shape

                            oR, oC, out_rows, out_cols = spsplit.get_out_dims(np.uint8(sect_in[0]), l_rows, l_cols,
                                                                              parameter_object)
                        else:
                            l_rows, l_cols = sect_in.shape

                            oR, oC, out_rows, out_cols = spsplit.get_out_dims(np.uint8(sect_in), l_rows, l_cols,
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
                            tk = spsplit.get_sect_feas(sect_in,
                                                       l_rows, l_cols,
                                                       parameter_object)

                            # Reshape list of features into
                            #   <features x rows x columns> array.
                            out_sect_arr = spreshape.reshape_feas(parameter_object.trigger,
                                                                  tk, oR, oC,
                                                                  l_rows, l_cols,
                                                                  out_rows, out_cols,
                                                                  parameter_object)

                            print '  Writing features to file ...'

                            obds = 1
                            for scale in parameter_object.scales:

                                # TODO: Temporary hack for gabor kernel size above scale size
                                if (trigger in ['gabor']) and scale < 16:
                                    continue

                                parameter_object.update_info(scale=scale)

                                for feature in xrange(1, parameter_object.features_dict[parameter_object.trigger]+1):

                                    parameter_object.update_info(feature=feature)

                                    parameter_object = sputilities.scale_fea_check(parameter_object)

                                    o_info = raster_tools.ropen(parameter_object.out_img, open2read=False)
                                    out_band_obj = o_info.datasource.GetRasterBand(1)

                                    # write array to file
                                    if (parameter_object.i_sect_blk_ctr < parameter_object.n_row_sects) and \
                                            (parameter_object.j_sect_blk_ctr == parameter_object.n_col_sects):

                                        try:
                                            out_band_obj.WriteArray(out_sect_arr[obds-1], j_sect_ctr, i_sect_ctr)
                                        except:
                                            out_band_obj.WriteArray(out_sect_arr[obds-1], j_sect_ctr-1, i_sect_ctr)

                                    elif (parameter_object.i_sect_blk_ctr == parameter_object.n_row_sects) and \
                                            (parameter_object.j_sect_blk_ctr < parameter_object.n_col_sects):

                                        try:
                                            out_band_obj.WriteArray(out_sect_arr[obds-1], j_sect_ctr, i_sect_ctr)
                                        except:
                                            out_band_obj.WriteArray(out_sect_arr[obds-1], j_sect_ctr, i_sect_ctr-1)

                                    elif (parameter_object.i_sect_blk_ctr == parameter_object.n_row_sects) and \
                                            (parameter_object.j_sect_blk_ctr == parameter_object.n_col_sects):

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

                                # TODO: Temporary hack for gabor kernel size above scale size
                                if (trigger in ['gabor']) and scale < 16:
                                    continue

                                parameter_object.update_info(scale=scale)

                                for feature in xrange(1, parameter_object.features_dict[parameter_object.trigger]+1):

                                    parameter_object.update_info(feature=feature)

                                    parameter_object = sputilities.scale_fea_check(parameter_object)

                                    # open the status dictionary
                                    with open(parameter_object.status_dict_txt, 'r') as pf:
                                        parameter_object.status_dict = yaml.load(pf)

                                    parameter_object.status_dict[parameter_object.out_img_base] = n_sect

                                    # store the status dictionary
                                    with open(parameter_object.status_dict_txt, 'w') as pf:
                                        pf.write(yaml.dump(parameter_object.status_dict, default_flow_style=False))

                                    obds_t += 1

                            del tk, oR, oC

                        j_sect_ctr += out_cols
                        parameter_object.j_sect_blk_ctr += 1

                        n_sect += 1

                    i_sect_ctr += out_rows
                    parameter_object.i_sect_blk_ctr += 1

                i_info.close()
                i_info = None

                obds = 1
                for scale in parameter_object.scales:

                    # TODO: Temporary hack for gabor kernel size above scale size
                    if (trigger in ['gabor']) and scale < 16:
                        continue

                    parameter_object.update_info(scale=scale)

                    for feature in xrange(1, parameter_object.features_dict[parameter_object.trigger]+1):

                        parameter_object.update_info(feature=feature)

                        parameter_object = sputilities.scale_fea_check(parameter_object)

                        # Resample the SFS image.
                        #
                        # SFS radiates from a center pixel, so is
                        #   more useful when computed with a
                        #   smaller block size.
                        if hasattr(parameter_object, 'sfs_resample'):

                            if parameter_object.sfs_resample > 0:

                                out_img_d_name, out_img_f_name = os.path.split(parameter_object.out_img)

                                out_img_resamp = os.path.join(out_img_d_name,
                                                              '{}_resamp{}'.format(parameter_object.out_img_base,
                                                                                   parameter_object.f_ext))

                                # Replace the block size.
                                out_img_resamp = out_img_resamp.replace('blk{:d}'.format(parameter_object.block),
                                                                        'blk{:d}'.format(int(parameter_object.sfs_resample)))

                                print '\nResampling SFS to {:.1f}m x {:.1f}m cell size ...\n'.format(parameter_object.sfs_resample,
                                                                                                     parameter_object.sfs_resample)

                                if 'img' in parameter_object.f_ext.lower():

                                    raster_tools.warp(parameter_object.out_img, out_img_resamp,
                                                      cell_size=parameter_object.sfs_resample,
                                                      resampleAlg='average',
                                                      warpMemoryLimit=256,
                                                      format='HFA',
                                                      multithread=True,
                                                      creationOptions=['COMPRESS=YES'])

                                else:

                                    raster_tools.warp(parameter_object.out_img, out_img_resamp,
                                                      cell_size=parameter_object.sfs_resample,
                                                      resampleAlg='average',
                                                      warpMemoryLimit=256,
                                                      multithread=True,
                                                      creationOptions=['COMPRESS=DEFLATE',
                                                                       'BIGTIFF=YES',
                                                                       'TILED=YES'])

                                out_img_new = out_img_resamp.replace('_resamp', '')

                                # Replace the block size.
                                out_img_new = out_img_new.replace('blk{:d}'.format(int(parameter_object.sfs_resample)),
                                                                  'blk{:d}'.format(int(parameter_object.sfs_resample /
                                                                                       parameter_object.block)))

                                os.remove(parameter_object.out_img)

                                os.rename(out_img_resamp, out_img_new)

                        obds += 1

        # Stack the features
        if hasattr(parameter_object, 'stack'):

            if parameter_object.stack:
                parameter_object = sputilities.stack_features(parameter_object, new_feas_list)

    # Optional conversion to GeoTiff.
    if hasattr(parameter_object, 'convert'):

        if parameter_object.convert:

            scales_str = [str(sc) for sc in parameter_object.scales]
            band_pos_str = [str(bp) for bp in parameter_object.band_positions]

            out_gtiff = os.path.join(parameter_object.output_dir,
                                     '{}.{}.stk.bd{}.block{}.scales{}.tif'.format(parameter_object.f_base,
                                                                                  '-'.join(parameter_object.triggers),
                                                                                  '-'.join(band_pos_str),
                                                                                  parameter_object.block,
                                                                                  '-'.join(scales_str)))

            raster_tools.translate(parameter_object.out_vrt, out_gtiff,
                                   format='GTiff',
                                   creationOptions=['TILED=YES', 'COMPRESS=LZW'])
