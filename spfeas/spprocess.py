from __future__ import division
from future.utils import viewitems
from builtins import map

import os
import copy
import fnmatch
import multiprocessing as multi

from .errors import logger, CorruptedBandsError
from .sphelpers import sputilities
from . import spsplit
from .sphelpers import spreshape
from .spfunctions import get_mag_avg, get_saliency_tile_mean, saliency, segment_image, get_dmp, get_orb_keypoints, convolve_gabor

# MpGlue
try:
    from mpglue import raster_tools, VegIndicesEquations, vrt_builder
    from mpglue import utils
except:
    logger.error('MpGlue must be installed')
    raise ImportError

# YAML
try:
    import yaml
except:
    logger.error('YAML must be installed')
    raise ImportError

# NumPy
try:
    import numpy as np
except:
    logger.error('NumPy must be installed')
    raise ImportError

# Scikit-image
try:
    from skimage.exposure import rescale_intensity
except ImportError:
    raise ImportError('Scikit-learn must be installed')


def _write_section2file(this_parameter_object__,
                        meta_info,
                        section2write,
                        i_sect,
                        j_sect,
                        out_rows,
                        out_cols,
                        section_counter):

    """
    Writes the section array to disk

    Args:
        this_parameter_object__ (class)
        meta_info (`rinfo` object)
        section2write (list of 1d arrays)
        i_sect (int)
        j_sect (int)
        section_counter (int)
    """
    
    logger.info('  Writing section {:d} of {:d} to file ...'.format(section_counter,
                                                                    this_parameter_object__.n_sects))

    o_info = meta_info.copy()

    o_info = sputilities.get_output_info_tile(meta_info, 
                                              o_info, 
                                              this_parameter_object__,
                                              i_sect, 
                                              j_sect,
                                              out_rows,
                                              out_cols)

    if not isinstance(section2write, np.ndarray):

        section2write = np.zeros((o_info.bands,
                                  o_info.rows,
                                  o_info.cols), dtype='uint8')

    start_band = this_parameter_object__.band_info[this_parameter_object__.trigger] + this_parameter_object__.band_counter + 1
    n_bands = this_parameter_object__.out_bands_dict[this_parameter_object__.trigger]

    if section2write[0].shape[0] == 0 or section2write[0].shape[1] == 0:
        pass
    else:

        if os.path.isfile(this_parameter_object__.out_img):

            # Open the file and write the new bands.
            with raster_tools.ropen(this_parameter_object__.out_img, open2read=False) as out_raster:

                array_layer_counter = 0

                # Write each scale and feature.
                for feature_band in range(start_band, start_band+n_bands):

                    out_raster.write_array(section2write[array_layer_counter], band=feature_band)
                    out_raster.close_band()

                    array_layer_counter += 1

        else:

            # Create the output raster.
            with raster_tools.create_raster(this_parameter_object__.out_img,
                                            o_info,
                                            bigtiff='yes') as out_raster:

                array_layer_counter = 0

                # Write each scale and feature.
                for feature_band in range(start_band, start_band+n_bands):

                    out_raster.write_array(section2write[array_layer_counter], band=feature_band)
                    out_raster.close_band()

                    array_layer_counter += 1

        del out_raster

    is_corrupt = False

    # The tile won't be written to file
    #   in the case of zero-length sections.
    if os.path.isfile(this_parameter_object__.out_img):

        # Check if any of the bands are corrupted.
        with raster_tools.ropen(this_parameter_object__.out_img) as ob_info:

            ob_info.check_corrupted_bands()

            if ob_info.corrupted_bands:
                is_corrupt = True
            else:
                is_corrupt = False

        del ob_info

    return is_corrupt


def _section_read_write(section_counter):

    """
    Handles the section reading and writing

    Args:
        section_counter (int)
    """

    section_pair = potsi[section_counter-1]

    # this_parameter_object_ = this_parameter_object.copy()
    this_parameter_object_ = copy.copy(param_dict)
    this_parameter_object_ = sputilities.dict2class(this_parameter_object_)

    # Get the input image information.
    with raster_tools.ropen(this_parameter_object_.input_image) as this_image_info:

        this_parameter_object_.update_info(section_counter=section_counter)

        # Set the output name.
        this_parameter_object_ = sputilities.scale_fea_check(this_parameter_object_)

        # Open the status YAML file.
        mts_ = sputilities.ManageStatus()

        # Load the status dictionary
        mts_.load_status(this_parameter_object_.status_file)

        # Check file status.
        if os.path.isfile(this_parameter_object_.out_img):

            if this_parameter_object_.out_img_base in mts_.status_dict:

                if this_parameter_object_.trigger in mts_.status_dict[this_parameter_object_.out_img_base]:

                    # Check every trigger because the
                    #   entire file needs to be removed.
                    status_list = [mts_.status_dict[this_parameter_object_.out_img_base]['{TR}-{BD}'.format(TR=tr,
                                                                                                            BD=this_parameter_object_.band_position)]
                                   for tr in this_parameter_object_.triggers]

                    if 'corrupt' in status_list:

                        logger.info('Re-running {} ...'.format(this_parameter_object_.out_img))

                        # Remove the file on the first trigger
                        #   if the file is corrupt.
                        if this_parameter_object_.trigger == this_parameter_object_.triggers[0]:
                            os.remove(this_parameter_object_.out_img)

                        mts_.status_dict[this_parameter_object_.out_img_base]['{TR}-{BD}'.format(TR=this_parameter_object_.trigger,
                                                                                                 BD=this_parameter_object_.band_position)] = 'incomplete'
                        mts_.dump_status(this_parameter_object_.status_file)

                    elif ('corrupt' not in status_list) and ('incomplete' in status_list):

                        logger.info('Re-running {} ...'.format(this_parameter_object_.out_img))

                    else:

                        if this_parameter_object_.overwrite:

                            logger.info('Re-running {} ...'.format(this_parameter_object_.out_img))

                            # Remove the file on the first trigger.
                            if this_parameter_object_.trigger == this_parameter_object_.triggers[0]:
                                os.remove(this_parameter_object_.out_img)

                            mts_.status_dict[this_parameter_object_.out_img_base]['{TR}-{BD}'.format(TR=this_parameter_object_.trigger,
                                                                                                     BD=this_parameter_object_.band_position)] = 'incomplete'
                            mts_.dump_status(this_parameter_object_.status_file)

                        else:

                            logger.info('{} is already finished ...'.format(this_parameter_object_.out_img))
                            return

            else:

                # Remove the file on the first trigger.
                if this_parameter_object_.trigger == this_parameter_object_.triggers[0]:
                    os.remove(this_parameter_object_.out_img)

                logger.info('Re-running {} ...'.format(this_parameter_object_.out_img))

        i_sect = section_pair[0]
        j_sect = section_pair[1]

        # Row and column section bounds checking
        n_rows = raster_tools.n_rows_cols(i_sect,
                                          this_parameter_object_.sect_row_size,
                                          this_image_info.rows)

        n_cols = raster_tools.n_rows_cols(j_sect,
                                          this_parameter_object_.sect_col_size,
                                          this_image_info.cols)

        # Open the image array.
        if this_parameter_object_.trigger in this_parameter_object_.spectral_indices:

            wavelengths = utils.VI_WAVELENGTHS[this_parameter_object_.trigger.upper()]

            # Check if the sensor supports the spectral index
            utils.sensor_wavelength_check(this_parameter_object_.sat_sensor,
                                          wavelengths)

            # Get the band positions needed
            #   to process the spectral index.
            spectral_bands = utils.get_index_bands(this_parameter_object_.trigger.upper(),
                                                   this_parameter_object_.sat_sensor)

            sect_in = this_image_info.read(bands2open=spectral_bands,
                                           i=i_sect,
                                           j=j_sect,
                                           rows=n_rows,
                                           cols=n_cols,
                                           d_type='float32')

            sect_in[sect_in >= this_parameter_object_.image_max] = this_parameter_object_.image_max
            sect_in /= this_parameter_object_.image_max

            vie = VegIndicesEquations(sect_in, chunk_size=-1)
            sect_in = vie.compute(this_parameter_object_.trigger.upper(), out_type=1)

            this_parameter_object_.update_info(image_min=0,
                                               image_max=1)

        elif this_parameter_object_.trigger == 'saliency':

            sect_in = saliency(this_image_info,
                               this_parameter_object_,
                               i_sect,
                               j_sect,
                               n_rows,
                               n_cols)

            this_parameter_object_.update_info(image_min=0,
                                               image_max=255)

        elif this_parameter_object_.trigger == 'seg':

            sect_in = this_image_info.read(bands2open=[1, 2, 3],
                                           i=i_sect,
                                           j=j_sect,
                                           rows=n_rows,
                                           cols=n_cols)

            sect_in = segment_image(sect_in, this_parameter_object_)

        elif this_parameter_object_.trigger == 'grad':

            if this_image_info.bands >= 3:

                sect_in = sputilities.convert_rgb2gray(this_image_info,
                                                       i_sect,
                                                       j_sect,
                                                       n_rows,
                                                       n_cols,
                                                       this_parameter_object_.sat_sensor)[0]

            else:

                sect_in = this_image_info.read(bands2open=this_parameter_object_.band_position,
                                               i=i_sect,
                                               j=j_sect,
                                               rows=n_rows,
                                               cols=n_cols)

            sect_in = np.uint8(rescale_intensity(sect_in,
                                                 in_range=(this_parameter_object_.image_min,
                                                           this_parameter_object_.image_max),
                                                 out_range=(0, 255)))

            sect_in = get_mag_avg(sect_in)

        elif this_parameter_object_.use_rgb and this_parameter_object_.trigger \
                not in this_parameter_object_.spectral_indices + ['grad', 'saliency', 'seg']:

            sect_in = sputilities.convert_rgb2gray(this_image_info,
                                                   i_sect,
                                                   j_sect,
                                                   n_rows,
                                                   n_cols,
                                                   this_parameter_object_.sat_sensor)[0]

        else:

            sect_in = this_image_info.read(bands2open=this_parameter_object_.band_position,
                                           i=i_sect,
                                           j=j_sect,
                                           rows=n_rows,
                                           cols=n_cols)

        if this_parameter_object_.trigger == 'dmp':

            # The Differential Morphological Profile
            #   is a [D x M x N] array
            # where,
            #   D = the opening/closing derivative.
            sect_in = get_dmp(sect_in,
                              this_parameter_object_.image_min,
                              this_parameter_object_.image_max)

        if this_parameter_object_.trigger == 'gabor':

            sect_in = convolve_gabor(sect_in,
                                     this_parameter_object_.image_min,
                                     this_parameter_object_.image_max,
                                     this_parameter_object_.scales)

        if this_parameter_object_.trigger == 'orb':

            sect_in = get_orb_keypoints(sect_in,
                                        this_parameter_object_.image_min,
                                        this_parameter_object_.image_max)

        this_parameter_object_.update_info(i_sect_blk_ctr=1,
                                           j_sect_blk_ctr=1)

        if this_parameter_object_.trigger == 'gabor':
            l_rows, l_cols = sect_in[0].shape
        else:
            l_rows, l_cols = sect_in.shape

        # Compute section statistics.
        section_stats_array = spsplit.get_section_stats(sect_in,
                                                        l_rows,
                                                        l_cols,
                                                        this_parameter_object_,
                                                        section_counter)

        # Get the section output rows and columns.
        out_rows, out_cols = spsplit.get_out_dims(l_rows,
                                                  l_cols,
                                                  this_parameter_object_)

        # Reshape the list of features into
        #   <features x rows x columns> array.
        out_section_array = spreshape.reshape_feature_list(section_stats_array,
                                                           out_rows,
                                                           out_cols,
                                                           this_parameter_object_)

        is_corrupt = _write_section2file(this_parameter_object_,
                                         this_image_info,
                                         out_section_array,
                                         i_sect,
                                         j_sect,
                                         out_rows,
                                         out_cols,
                                         section_counter)

    this_parameter_object_ = None
    this_image_info_ = None

    return is_corrupt


def run(parameter_object):

    """
    Args:
        input_image, output_dir, band_positions=[1], use_rgb=False, block=2, scales=[8], triggers=['mean'],
        threshold=20, min_len=10, line_gap=2, weighted=False, sfs_thresh=80, resamp_sfs=0.,
        equalize=False, equalize_adapt=False, smooth=0, visualize=False, convert_stk=False, gdal_cache=256,
        do_pca=False, stack_feas=True, stack_only=False, neighbors=False, n_jobs=-1,
        reset_sects=False, image_max=0, lac_r=2, section_size=8000, chunk_size=512
    """

    global potsi, param_dict

    if parameter_object.n_jobs == 0:
        parameter_object.n_jobs = 1
    elif parameter_object.n_jobs < 0:
        parameter_object.n_jobs = multi.cpu_count()
    elif parameter_object.n_jobs > multi.cpu_count():
        parameter_object.n_jobs = multi.cpu_count()

    # It is assumed in various places that the scales are sorted
    parameter_object.scales.sort()

    sputilities.parameter_checks(parameter_object)

    # Write the parameters to file.
    sputilities.write_log(parameter_object)

    if parameter_object.stack_only:

        with raster_tools.ropen(parameter_object.input_image) as i_info:

            # Get image statistics.
            parameter_object = sputilities.get_stats(i_info, parameter_object)

            # Get the section size.
            parameter_object = sputilities.get_section_size(i_info, parameter_object)

            # Get the number of sections in
            #   the image (only used as a counter).
            parameter_object = sputilities.get_n_sects(i_info, parameter_object)

        i_info = None

        new_feas_list = list()

        # If prompted, stack features without processing.
        parameter_object = sputilities.stack_features(parameter_object,
                                                      new_feas_list)

    else:

        # Create the status object.
        mts = sputilities.ManageStatus()

        parameter_object.remove_files = False

        # Setup the status dictionary.
        if os.path.isfile(parameter_object.status_file):

            mts.load_status(parameter_object.status_file)

            if parameter_object.section_size != mts.status_dict['SECTION_SIZE']:

                logger.warning('The section size was changed, so all existing tiled images will be removed.')

                parameter_object.remove_files = True

            if not isinstance(mts.status_dict, dict):

                logger.error('The YAML file already existed, but was not properly stored and saved.\nPlease remove and re-run.')
                raise AttributeError

        else:

            mts.status_dict = dict()

            mts.status_dict['ALL_FINISHED'] = 'no'
            mts.status_dict['BAND_ORDER'] = dict()

            # Save the band order.
            for trigger in parameter_object.triggers:

                mts.status_dict['BAND_ORDER']['{}'.format(trigger)] = '{:d}-{:d}'.format(parameter_object.band_info[trigger]+1,
                                                                                         parameter_object.band_info[trigger]+parameter_object.out_bands_dict[trigger]*parameter_object.n_bands)

            mts.status_dict['SECTION_SIZE'] = parameter_object.section_size

            mts.dump_status(parameter_object.status_file)

        process_image = True

        if 'ALL_FINISHED' in mts.status_dict:

            if mts.status_dict['ALL_FINISHED'] == 'yes':
                process_image = False

        # Set the output features folder.
        parameter_object = sputilities.set_feas_dir(parameter_object)

        if parameter_object.remove_files:

            image_list = fnmatch.filter(os.listdir(parameter_object.feas_dir), '*.tif')

            if image_list:

                image_list = [os.path.join(parameter_object.feas_dir, im_) for im_ in image_list]

                for full_image in image_list:
                    os.remove(full_image)

        if not process_image:
            logger.warning('The input image, {}, is set as finished processing.'.format(parameter_object.input_image))
        else:

            original_band_positions = copy.copy(parameter_object.band_positions)

            # Iterate over each feature trigger.
            for trigger in parameter_object.triggers:

                parameter_object.update_info(trigger=trigger,
                                             band_positions=original_band_positions,
                                             band_counter=0)

                # Iterate over each band
                for band_position in parameter_object.band_positions:

                    parameter_object.update_info(band_position=band_position)

                    # Get the input image information.
                    with raster_tools.ropen(parameter_object.input_image) as i_info:

                        # Check if any of the input
                        #   bands are corrupted.
                        i_info.check_corrupted_bands()

                        if i_info.corrupted_bands:

                            logger.error('\nThe following bands appear to be corrupted:\n{}'.format(', '.join(i_info.corrupted_bands)))
                            raise CorruptedBandsError

                        # Get image statistics.
                        parameter_object = sputilities.get_stats(i_info, parameter_object)

                        # Get the section size.
                        parameter_object = sputilities.get_section_size(i_info, parameter_object)

                        # Get the number of sections in
                        #   the image (only used as a counter).
                        parameter_object = sputilities.get_n_sects(i_info, parameter_object)

                        if parameter_object.trigger == 'saliency':

                            bp = raster_tools.BlockFunc(get_saliency_tile_mean,
                                                        [i_info],
                                                        None,
                                                        None,
                                                        band_list=[[1, 2, 3]],
                                                        d_types=['float32'],
                                                        write_array=False,
                                                        close_files=False,
                                                        be_quiet=True,
                                                        print_statement='\nGetting tile lab means for saliency',
                                                        out_attributes=['lab_means'],
                                                        block_rows=parameter_object.sect_row_size,
                                                        block_cols=parameter_object.sect_col_size,
                                                        min_max=[(parameter_object.image_min,
                                                                  parameter_object.image_max)]*3,
                                                        vis_order=parameter_object.vis_order)

                            bp.run()

                            parameter_object.update_info(lab_means=np.array(bp.lab_means,
                                                                            dtype='float32').mean(axis=0))

                    del i_info

                    mts = sputilities.ManageStatus()
                    mts.load_status(parameter_object.status_file)

                    for sect_counter in range(1, parameter_object.n_sects+1):

                        parameter_object.update_info(section_counter=sect_counter)
                        parameter_object = sputilities.scale_fea_check(parameter_object)

                        if trigger == parameter_object.triggers[0]:
                            mts.status_dict[parameter_object.out_img_base] = dict()

                        mts.status_dict[parameter_object.out_img_base]['{TR}-{BD}'.format(TR=parameter_object.trigger,
                                                                                          BD=parameter_object.band_position)] = 'unprocessed'

                    mts.dump_status(parameter_object.status_file)

                    param_dict = sputilities.class2dict(parameter_object)
                    potsi = parameter_object.section_idx_pairs

                    # PROCESS IN PARALLEL CHUNKS

                    parallel_chunk_counter = 1

                    for parallel_chunk in range(1,
                                                parameter_object.n_sects+parameter_object.n_jobs,
                                                parameter_object.n_jobs):

                        if parallel_chunk + parameter_object.n_jobs < parameter_object.n_sects:
                            parallel_chunk_end = parallel_chunk + parameter_object.n_jobs
                        else:
                            parallel_chunk_end = parallel_chunk + (parameter_object.n_sects - parallel_chunk) + 1

                        # Testing
                        # results = list(map(_section_read_write, range(parallel_chunk, parallel_chunk_end)))

                        pool = multi.Pool(processes=parameter_object.n_jobs)

                        results = pool.map(_section_read_write,
                                           range(parallel_chunk,
                                                 parallel_chunk_end))

                        pool.close()
                        pool.join()
                        pool = None

                        logger.info('  Updating status ...')

                        for result in results:

                            parameter_object.update_info(section_counter=parallel_chunk_counter)
                            parameter_object = sputilities.scale_fea_check(parameter_object)

                            # Open the status YAML file.
                            mts = sputilities.ManageStatus()

                            # Load the status dictionary
                            mts.load_status(parameter_object.status_file)

                            if parameter_object.out_img_base in mts.status_dict:

                                if result:

                                    mts.status_dict[parameter_object.out_img_base]['{TR}-{BD}'.format(TR=parameter_object.trigger,
                                                                                                      BD=parameter_object.band_position)] = 'corrupt'

                                else:

                                    mts.status_dict[parameter_object.out_img_base]['{TR}-{BD}'.format(TR=parameter_object.trigger,
                                                                                                      BD=parameter_object.band_position)] = 'complete'

                            mts.dump_status(parameter_object.status_file)

                            parallel_chunk_counter += 1

                    # Parallel(n_jobs=parameter_object.n_jobs,
                    #          batch_size=1,
                    #          max_nbytes=None)(delayed(_section_read_write)(idx_pair,
                    #                                                        parameter_object.section_idx_pairs[idx_pair-1],
                    #                                                        param_dict)
                    #                           for idx_pair in range(1, parameter_object.n_sects+1))

                    parameter_object.band_counter += parameter_object.out_bands_dict[parameter_object.trigger]

        # Check the corruption status.
        mts.load_status(parameter_object.status_file)

        n_corrupt = 0
        for k, v in viewitems(mts.status_dict):

            if isinstance(v, dict):

                for ksub, vsub in viewitems(v):

                    if vsub in ['corrupt', 'incomplete']:
                        n_corrupt += 1

        if n_corrupt == 0:

            mts.status_dict['ALL_FINISHED'] = 'yes'
            mts.dump_status(parameter_object.status_file)

            # Finally, mosaic the image tiles.

            logger.info('  Creating the VRT mosaic ...')

            comp_dict = dict()

            # Get the image list.
            parameter_object = sputilities.scale_fea_check(parameter_object, is_image=False)

            image_list = fnmatch.filter(os.listdir(parameter_object.feas_dir), parameter_object.search_wildcard)
            image_list = [os.path.join(parameter_object.feas_dir, im) for im in image_list]

            comp_dict['001'] = image_list

            vrt_mosaic = parameter_object.status_file.replace('.yaml', '.vrt')

            vrt_builder(comp_dict,
                        vrt_mosaic,
                        force_type='float32',
                        be_quiet=True,
                        overwrite=True,
                        relative_path=parameter_object.relative_path)

            if parameter_object.overviews:

                logger.info('\nBuilding VRT overviews ...')

                with raster_tools.ropen(vrt_mosaic, open2read=False) as vrt_info:

                    vrt_info.remove_overviews()
                    vrt_info.build_overviews(levels=[2, 4, 8, 16])

                del vrt_info

        else:

            if n_corrupt == 1:
                logger.warning('\nThere was {:d} corrupt or incomplete tile.\nRe-run the command with the same parameters.'.format(n_corrupt))
            else:
                logger.warning('\nThere were {:d} corrupt or incomplete tiles.\nRe-run the command with the same parameters.'.format(n_corrupt))
