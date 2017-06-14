#!/usr/bin/env python

import os
import sys
# import time
# import platform
import copy
# import itertools
from joblib import Parallel, delayed

from .sphelpers import sputilities
from . import spsplit
from .sphelpers import spreshape
from .spfunctions import get_mag_avg
from . import errors

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


def _write_section2file(this_parameter_object__, meta_info, section2write, 
                        i_sect, j_sect, section_counter):
    
    print('  Writing section {:d} to file ...'.format(section_counter))

    o_info = meta_info.copy()

    section_shape = section2write.shape

    o_info = sputilities.get_output_info_tile(meta_info, 
                                              o_info, 
                                              this_parameter_object__,
                                              i_sect, 
                                              j_sect,
                                              section_shape)

    if not isinstance(section2write, np.ndarray):

        section2write = np.zeros((o_info.bands,
                                  o_info.rows,
                                  o_info.cols), dtype='uint8')

    # Create the output raster.
    with raster_tools.create_raster(this_parameter_object__.out_img, o_info) as out_raster:

        # Write each scale and feature.
        for feature_band in range(1, this_parameter_object__.out_bands_dict[this_parameter_object__.trigger]+1):
            out_raster.write_array(section2write[feature_band-1, 1:, 1:], band=feature_band)
            
    out_raster = None

    # Check if any of the bands are corrupted.
    with raster_tools.ropen(this_parameter_object__.out_img) as ob_info:

        ob_info.check_corrupted_bands()

        # Open the status YAML file.
        mts = sputilities.ManageStatus()

        # Load the status dictionary
        mts.status_file = this_parameter_object__.status_file
        mts.load_status()

        if ob_info.corrupted_bands:
            mts.status_dict[this_parameter_object__.out_img] = 'corrupt'
        else:
            mts.status_dict[this_parameter_object__.out_img] = 'complete'

        mts.dump_status()

    ob_info = None
            

def _section_read_write(section_counter, section_pair):
    
    this_parameter_object_ = this_parameter_object.copy()

    this_parameter_object_.update_info(section_counter=section_counter)

    # Set the output name.
    this_parameter_object_ = sputilities.scale_fea_check(this_parameter_object_)

    # Open the status YAML file.
    mts = sputilities.ManageStatus()

    # Load the status dictionary
    mts.status_file = this_parameter_object_.status_file
    mts.load_status()

    # Check file status.
    if os.path.isfile(this_parameter_object_.out_img):

        # The file has been processed.
        if this_parameter_object_.out_img in mts.status_dict:

            if mts.status_dict[this_parameter_object_.out_img] == 'complete':

                if this_parameter_object_.overwrite:

                    os.remove(this_parameter_object_.out_img)
                    mts.status_dict[this_parameter_object_.out_img] = 'incomplete'

                else:
                    return

            elif mts.status_dict[this_parameter_object_.out_img] == 'corrupt':

                os.remove(this_parameter_object_.out_img)
                mts.status_dict[this_parameter_object_.out_img] = 'incomplete'

        else:

            os.remove(this_parameter_object_.out_img)
            mts.status_dict[this_parameter_object_.out_img] = 'incomplete'

    i_sect = section_pair[0]
    j_sect = section_pair[1]

    n_rows = raster_tools.n_rows_cols(i_sect, this_parameter_object_.sect_row_size, this_image_info.rows)
    n_cols = raster_tools.n_rows_cols(j_sect, this_parameter_object_.sect_col_size, this_image_info.cols)

    # Open the image array.
    # TODO: add other indices
    if this_parameter_object_.trigger in ['ndvi', 'evi2']:

        sect_in = this_image_info.read(bands2open=[this_parameter_object_.band_red,
                                                   this_parameter_object_.band_nir],
                                       i=i_sect,
                                       j=j_sect,
                                       rows=n_rows,
                                       cols=n_cols,
                                       d_type='float32')

        vie = VegIndicesEquations(sect_in, chunk_size=-1)
        sect_in = vie.compute(this_parameter_object_.trigger.upper(), out_type=2)

        this_parameter_object_.min = 0
        this_parameter_object_.max = 255

    elif this_parameter_object_.trigger == 'dmp':

        sect_in = np.asarray([this_image_info.read(bands2open=dmp_bd,
                                                   i=i_sect,
                                                   j=j_sect,
                                                   rows=n_rows,
                                                   cols=n_cols,
                                                   d_type='float32')
                              for dmp_bd in range(1, this_image_info.bands+1)]).reshape(this_image_info.bands,
                                                                                        n_rows,
                                                                                        n_cols)

    elif this_parameter_object_.trigger == 'saliency':

        sect_in = spsplit.saliency(this_image_info,
                                   this_parameter_object_,
                                   i_sect,
                                   j_sect,
                                   n_rows,
                                   n_cols)

    elif this_parameter_object_.trigger == 'grad':

        sect_in, __, __ = sputilities.convert_rgb2gray(this_image_info,
                                                       i_sect,
                                                       j_sect,
                                                       n_rows,
                                                       n_cols)

        sect_in = get_mag_avg(sect_in)
        this_parameter_object_.update_info(min=0, max=255)

    elif this_parameter_object_.use_rgb and this_parameter_object_.trigger not in ['grad', 'ndvi', 'evi2', 'dmp', 'saliency']:

        sect_in, __, __ = sputilities.convert_rgb2gray(this_image_info,
                                                       i_sect,
                                                       j_sect,
                                                       n_rows,
                                                       n_cols)

    else:

        sect_in = this_image_info.read(bands2open=this_parameter_object_.band_position,
                                       i=i_sect,
                                       j=j_sect,
                                       rows=n_rows,
                                       cols=n_cols)

    # if this_parameter_object_.trigger == 'orb':
    #     sect_in = spsplit.get_orb_keypoints(sect_in, this_parameter_object_)
    #     this_parameter_object_.update_info(min=0, max=255)

    # pad array here
    # (top, bottom), (left, right)
    this_parameter_object_.update_info(i_sect_blk_ctr=1,
                                       j_sect_blk_ctr=1)

    sect_in = sputilities.pad_array(this_parameter_object_, sect_in, n_rows, n_cols)

    if this_parameter_object_.trigger == 'dmp':

        l_rows, l_cols = sect_in[0].shape
        oR, oC, out_rows, out_cols = spsplit.get_out_dims(l_rows,
                                                          l_cols,
                                                          this_parameter_object_)
    else:

        l_rows, l_cols = sect_in.shape
        oR, oC, out_rows, out_cols = spsplit.get_out_dims(l_rows,
                                                          l_cols,
                                                          this_parameter_object_)

    # out_section_array = None

    # Only extract features if the section hasn't
    #   been completed or if the section does not
    #   contain all zeros.
    # if sect_in.max() > 0:

    # Here we split the current section into
    #   chunks and process the features.

    # Split image and compute features.
    section_stats_array = spsplit.get_section_stats(sect_in,
                                                    l_rows,
                                                    l_cols,
                                                    this_parameter_object_)

    # Reshape list of features into
    #   <features x rows x columns> array.
    out_section_array = spreshape.chunks2section(this_parameter_object_.trigger,
                                                 section_stats_array,
                                                 oR,
                                                 oC,
                                                 l_rows,
                                                 l_cols,
                                                 out_rows,
                                                 out_cols,
                                                 this_parameter_object_)

    _write_section2file(this_parameter_object_,
                        this_image_info,
                        out_section_array,
                        i_sect,
                        j_sect,
                        section_counter)

    # else:
    #
    #     _write_section2file(this_parameter_object_,
    #                         this_image_info,
    #                         None,
    #                         i_sect,
    #                         j_sect)

    this_parameter_object_ = None
    this_image_info_ = None


def run(parameter_object):

    """
    Args:
        input_image, output_dir, band_positions=[1], use_rgb=False, block=2, scales=[8], triggers=['mean'],
        threshold=20, min_len=10, line_gap=2, weighted=False, sfs_thresh=80, resamp_sfs=0.,
        equalize=False, equalize_adapt=False, smooth=0, visualize=False, convert_stk=False, gdal_cache=256,
        do_pca=False, stack_feas=True, stack_only=False, band_red=3, band_nir=4, neighbors=False, n_jobs=-1,
        reset_sects=False, image_max=0, lac_r=2, section_size=8000, chunk_size=512
    """

    global this_parameter_object, this_image_info

    sputilities.parameter_checks(parameter_object)

    # Write the parameters to file.
    sputilities.write_log(parameter_object)

    new_feas_list = []

    if parameter_object.stack_only:

        # If prompted, stack features without processing.
        parameter_object = sputilities.stack_features(parameter_object, new_feas_list)

    else:

        trigger_orig_seg = False

        # Iterate over each feature trigger.
        for trigger in parameter_object.triggers:

            parameter_object.update_info(trigger=trigger)

            # Set the output features folder.
            parameter_object = sputilities.set_feas_dir(parameter_object)

            # Iterate over each band
            for band_position in parameter_object.band_positions:

                parameter_object.update_info(band_position=band_position)

                # Get input image information.
                i_info = raster_tools.ropen(parameter_object.input_image)

                # Check if any of the input
                #   bands are corrupted.
                i_info.check_corrupted_bands()

                if i_info.corrupted_bands:

                    errors.logger.error('\nThe following bands appear to be corrupted:\n{}'.format(', '.join(i_info.corrupted_bands)))
                    raise errors.CorruptedBandsError('\nThe following bands appear to be corrupted:\n{}'.format(', '.join(i_info.corrupted_bands)))

                # Get image statistics.
                parameter_object = sputilities.get_stats(i_info, parameter_object)

                # Get section and chunk size.
                parameter_object = sputilities.get_sect_chunk_size(i_info, parameter_object)

                # if parameter_object.trigger == 'sfsorf':
                #     parameter_object.update_info(scale=parameter_object.scales[-1],
                #                                  feature=100)
                #     parameter_object = sputilities.scale_fea_check(parameter_object)
                #     spsplit.sfs_orfeo(parameter_object)
                #     if os.path.isfile(parameter_object.out_img):
                #         os.remove(parameter_object.out_img)
                #     continue

                # Create the output feature bands.
                # parameter_object = sputilities.create_outputs(parameter_object,
                #                                               new_feas_list,
                #                                               i_info)

                # Get the number of sections in
                #   the image (only used as a counter).
                parameter_object = sputilities.get_n_sects(i_info, parameter_object)

                this_parameter_object = parameter_object.copy()
                this_image_info = i_info.copy()

                # Create the status dictionary.
                mts = sputilities.ManageStatus()
                mts.status_file = parameter_object.status_file

                # Load the status dictionary, or start from scratch
                if not os.path.isfile(parameter_object.status_file):

                    mts.status_dict = dict()
                    mts.status_dict['all'] = 'incomplete'
                    mts.dump_status()

                Parallel(n_jobs=parameter_object.n_jobs_section,
                         max_nbytes=None)(delayed(_section_read_write)(idx_pair,
                                                                       parameter_object.section_idx_pairs[idx_pair-1])
                                          for idx_pair in range(1, parameter_object.n_sects+1))





    #             # Here we iterate over the image by sections.
    #             n_sect = 1
    #             i_sect_ctr = 0
    #
    #             parameter_object.update_info(i_sect_blk_ctr=1)
    #
    #             for i_sect in xrange(0, i_info.rows, parameter_object.sect_row_size -
    #                     (parameter_object.scales[-1] - parameter_object.block)):
    #
    #                 n_rows = raster_tools.n_rows_cols(i_sect, parameter_object.sect_row_size, i_info.rows)
    #
    #                 j_sect_ctr = 0
    #
    #                 parameter_object.update_info(j_sect_blk_ctr=1)
    #
    #                 for j_sect in xrange(0, i_info.cols, parameter_object.sect_col_size -
    #                         (parameter_object.scales[-1] - parameter_object.block)):
    #
    #                     print('\nSection {:d} of {:d} ...'.format(n_sect, parameter_object.n_sects))
    #
    #                     n_cols = raster_tools.n_rows_cols(j_sect, parameter_object.sect_col_size, i_info.cols)
    #
    #                     ###################################
    #                     # Check if the section has been
    #                     # processed for all feature scales.
    #                     ###################################
    #
    #                     # sects_good = True
    #
    #                     obds = 1
    #                     for scale in parameter_object.scales:
    #
    #                         parameter_object.update_info(scale=scale)
    #
    #                         for feature in xrange(1, parameter_object.features_dict[parameter_object.trigger]+1):
    #
    #                             parameter_object.update_info(feature=feature)
    #
    #                             parameter_object = sputilities.scale_fea_check(parameter_object)
    #
    #                             # Open the status dictionary.
    #                             # with open(parameter_object.status_dict_txt, 'r') as pf:
    #                             #     parameter_object.status_dict = yaml.load(pf)
    #                             #
    #                             # sect_status = parameter_object.status_dict[parameter_object.out_img_base]
    #
    #                             obds += 1
    #
    #                             # If any of the sections are not current,
    #                             #   continue with feature extraction.
    #                             # if sect_status < n_sect:
    #                             #     sects_good = False
    #
    #                     # Open the image array.
    #                     # TODO: add other indices
    #                     if parameter_object.trigger in ['ndvi', 'evi2']:
    #
    #                         sect_in = i_info.read(bands2open=[parameter_object.band_red,
    #                                                           parameter_object.band_nir],
    #                                               i=i_sect, j=j_sect,
    #                                               rows=n_rows, cols=n_cols,
    #                                               d_type='float32')
    #
    #                         vie = VegIndicesEquations(sect_in, chunk_size=-1)
    #                         sect_in = vie.compute(parameter_object.trigger.upper(), out_type=2)
    #
    #                         parameter_object.min = 0
    #                         parameter_object.max = 255
    #
    #                     elif parameter_object.trigger == 'dmp':
    #
    #                         sect_in = np.asarray([i_info.read(bands2open=dmp_bd,
    #                                                           i=i_sect, j=j_sect,
    #                                                           rows=n_rows, cols=n_cols,
    #                                                           d_type='float32')
    #                                               for dmp_bd in xrange(1, i_info.bands+1)]).reshape(i_info.bands,
    #                                                                                                 n_rows, n_cols)
    #
    #                     elif parameter_object.trigger == 'saliency':
    #
    #                         # parameter_object.update_info(min=0, max=255)
    #                         sect_in = spsplit.saliency(i_info, parameter_object,
    #                                                    i_sect, j_sect,
    #                                                    n_rows, n_cols)
    #
    #                     elif parameter_object.trigger == 'grad':
    #
    #                         sect_in, __, __ = sputilities.convert_rgb2gray(i_info,
    #                                                                        i_sect, j_sect,
    #                                                                        n_rows, n_cols)
    #
    #                         sect_in = get_mag_avg(sect_in)
    #                         parameter_object.update_info(min=0, max=255)
    #
    #                     elif parameter_object.use_rgb and trigger not in ['grad', 'ndvi', 'evi2', 'dmp', 'saliency']:
    #
    #                         sect_in, __, __ = sputilities.convert_rgb2gray(i_info,
    #                                                                        i_sect, j_sect,
    #                                                                        n_rows, n_cols)
    #
    #                     else:
    #
    #                         sect_in = i_info.read(bands2open=parameter_object.band_position,
    #                                               i=i_sect, j=j_sect,
    #                                               rows=n_rows, cols=n_cols)
    #
    #                     if parameter_object.trigger == 'orb':
    #                         sect_in = spsplit.get_orb_keypoints(sect_in, parameter_object)
    #                         parameter_object.update_info(min=0, max=255)
    #
    #                     # pad array here
    #                     # (top, bottom), (left, right)
    #                     sect_in = sputilities.pad_array(parameter_object, sect_in, n_rows, n_cols)
    #
    #                     if parameter_object.trigger == 'dmp':
    #
    #                         l_rows, l_cols = sect_in[0].shape
    #                         oR, oC, out_rows, out_cols = spsplit.get_out_dims(l_rows, l_cols,
    #                                                                           parameter_object)
    #                     else:
    #
    #                         l_rows, l_cols = sect_in.shape
    #                         oR, oC, out_rows, out_cols = spsplit.get_out_dims(l_rows, l_cols,
    #                                                                           parameter_object)
    #
    #                     # Only extract features if the section hasn't
    #                     #   been completed or if the section does not
    #                     #   contain all zeros.
    #                     if sect_in.max() > 0:
    #
    #                         # Here we split the current section into
    #                         #   chunks and process the features.
    #
    #                         # Split image and compute features.
    #                         tk = spsplit.get_section_stats(sect_in,
    #                                                    l_rows, l_cols,
    #                                                    parameter_object)
    #
    #                         # Reshape list of features into
    #                         #   <features x rows x columns> array.
    #                         out_sect_arr = spreshape.chunks2section(parameter_object.trigger,
    #                                                               tk, oR, oC,
    #                                                               l_rows, l_cols,
    #                                                               out_rows, out_cols,
    #                                                               parameter_object)
    #
    #                         print('  Writing features to file ...')
    #
    #                         obds = 1
    #                         for scale in parameter_object.scales:
    #
    #                             parameter_object.update_info(scale=scale)
    #
    #                             for feature in xrange(1, parameter_object.features_dict[parameter_object.trigger]+1):
    #
    #                                 parameter_object.update_info(feature=feature)
    #
    #                                 parameter_object = sputilities.scale_fea_check(parameter_object)
    #
    #                                 o_info = raster_tools.ropen(parameter_object.out_img, open2read=False)
    #                                 out_band_obj = o_info.datasource.GetRasterBand(1)
    #
    #                                 # write array to file
    #                                 if (parameter_object.i_sect_blk_ctr < parameter_object.n_row_sects) and \
    #                                         (parameter_object.j_sect_blk_ctr == parameter_object.n_col_sects):
    #
    #                                     try:
    #                                         out_band_obj.WriteArray(out_sect_arr[obds-1], j_sect_ctr, i_sect_ctr)
    #                                     except:
    #                                         out_band_obj.WriteArray(out_sect_arr[obds-1], j_sect_ctr-1, i_sect_ctr)
    #
    #                                 elif (parameter_object.i_sect_blk_ctr == parameter_object.n_row_sects) and \
    #                                         (parameter_object.j_sect_blk_ctr < parameter_object.n_col_sects):
    #
    #                                     try:
    #                                         out_band_obj.WriteArray(out_sect_arr[obds-1], j_sect_ctr, i_sect_ctr)
    #                                     except:
    #                                         out_band_obj.WriteArray(out_sect_arr[obds-1], j_sect_ctr, i_sect_ctr-1)
    #
    #                                 elif (parameter_object.i_sect_blk_ctr == parameter_object.n_row_sects) and \
    #                                         (parameter_object.j_sect_blk_ctr == parameter_object.n_col_sects):
    #
    #                                     try:
    #                                         out_band_obj.WriteArray(out_sect_arr[obds-1], j_sect_ctr, i_sect_ctr)
    #                                     except:
    #                                         out_band_obj.WriteArray(out_sect_arr[obds-1], j_sect_ctr-1, i_sect_ctr-1)
    #
    #                                 else:
    #                                     out_band_obj.WriteArray(out_sect_arr[obds-1], j_sect_ctr, i_sect_ctr)
    #
    #                                 out_band_obj.FlushCache()
    #                                 out_band_obj = None
    #
    #                                 o_info.close()
    #                                 o_info = None
    #
    #                                 obds += 1
    #
    #                         ###################################################
    #                         # Update status dictionary with the section number.
    #                         ###################################################
    #
    #                         obds_t = 1
    #                         for scale in parameter_object.scales:
    #
    #                             parameter_object.update_info(scale=scale)
    #
    #                             for feature in xrange(1, parameter_object.features_dict[parameter_object.trigger]+1):
    #
    #                                 parameter_object.update_info(feature=feature)
    #
    #                                 parameter_object = sputilities.scale_fea_check(parameter_object)
    #
    #                                 # open the status dictionary
    #                                 # with open(parameter_object.status_dict_txt, 'r') as pf:
    #                                 #     parameter_object.status_dict = yaml.load(pf)
    #
    #                                 # parameter_object.status_dict[parameter_object.out_img_base] = n_sect
    #
    #                                 # store the status dictionary
    #                                 # with open(parameter_object.status_dict_txt, 'w') as pf:
    #                                 #     pf.write(yaml.dump(parameter_object.status_dict, default_flow_style=False))
    #
    #                                 obds_t += 1
    #
    #                         del tk, oR, oC
    #
    #                     j_sect_ctr += out_cols
    #                     parameter_object.j_sect_blk_ctr += 1
    #
    #                     n_sect += 1
    #
    #                 i_sect_ctr += out_rows
    #                 parameter_object.i_sect_blk_ctr += 1
    #
    #             i_info.close()
    #             i_info = None
    #
    #             obds = 1
    #             for scale in parameter_object.scales:
    #
    #                 parameter_object.update_info(scale=scale)
    #
    #                 for feature in xrange(1, parameter_object.features_dict[parameter_object.trigger]+1):
    #
    #                     parameter_object.update_info(feature=feature)
    #
    #                     parameter_object = sputilities.scale_fea_check(parameter_object)
    #
    #                     # Resample the SFS image.
    #                     #
    #                     # SFS radiates from a center pixel, so is
    #                     #   more useful when computed with a
    #                     #   smaller block size.
    #                     if hasattr(parameter_object, 'sfs_resample'):
    #
    #                         if parameter_object.sfs_resample > 0:
    #
    #                             out_img_d_name, out_img_f_name = os.path.split(parameter_object.out_img)
    #
    #                             out_img_resamp = os.path.join(out_img_d_name,
    #                                                           '{}_resamp{}'.format(parameter_object.out_img_base,
    #                                                                                parameter_object.f_ext))
    #
    #                             # Replace the block size.
    #                             out_img_resamp = out_img_resamp.replace('blk{:d}'.format(parameter_object.block),
    #                                                                     'blk{:d}'.format(int(parameter_object.sfs_resample)))
    #
    #                             print('\nResampling SFS to {:.1f}m x {:.1f}m cell size ...\n'.format(parameter_object.sfs_resample,
    #                                                                                                  parameter_object.sfs_resample))
    #
    #                             if 'img' in parameter_object.f_ext.lower():
    #
    #                                 raster_tools.warp(parameter_object.out_img, out_img_resamp,
    #                                                   cell_size=parameter_object.sfs_resample,
    #                                                   resampleAlg='average',
    #                                                   warpMemoryLimit=256,
    #                                                   format='HFA',
    #                                                   multithread=True,
    #                                                   creationOptions=['COMPRESS=YES'])
    #
    #                             else:
    #
    #                                 raster_tools.warp(parameter_object.out_img, out_img_resamp,
    #                                                   cell_size=parameter_object.sfs_resample,
    #                                                   resampleAlg='average',
    #                                                   warpMemoryLimit=256,
    #                                                   multithread=True,
    #                                                   creationOptions=['COMPRESS=DEFLATE',
    #                                                                    'BIGTIFF=YES',
    #                                                                    'TILED=YES'])
    #
    #                             out_img_new = out_img_resamp.replace('_resamp', '')
    #
    #                             # Replace the block size.
    #                             out_img_new = out_img_new.replace('blk{:d}'.format(int(parameter_object.sfs_resample)),
    #                                                               'blk{:d}'.format(int(parameter_object.sfs_resample /
    #                                                                                    parameter_object.block)))
    #
    #                             os.remove(parameter_object.out_img)
    #
    #                             os.rename(out_img_resamp, out_img_new)
    #
    #                     obds += 1
    #
    #     # Stack the features
    #     if hasattr(parameter_object, 'stack'):
    #
    #         if parameter_object.stack:
    #             parameter_object = sputilities.stack_features(parameter_object, new_feas_list)
    #
    # # Optional conversion to GeoTiff.
    # if hasattr(parameter_object, 'convert'):
    #
    #     if parameter_object.convert:
    #
    #         scales_str = [str(sc) for sc in parameter_object.scales]
    #         band_pos_str = [str(bp) for bp in parameter_object.band_positions]
    #
    #         out_gtiff = os.path.join(parameter_object.output_dir,
    #                                  '{}.{}.stk.bd{}.block{}.scales{}.tif'.format(parameter_object.f_base,
    #                                                                               '-'.join(parameter_object.triggers),
    #                                                                               '-'.join(band_pos_str),
    #                                                                               parameter_object.block,
    #                                                                               '-'.join(scales_str)))
    #
    #         raster_tools.translate(parameter_object.out_vrt, out_gtiff,
    #                                format='GTiff',
    #                                creationOptions=['TILED=YES', 'COMPRESS=LZW'])
