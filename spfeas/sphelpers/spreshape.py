"""
@author: Jordan Graesser
Date Created: 7/2/2013
"""

from __future__ import division

from ..errors import logger

try:
    import numpy as np
except ImportError:
    raise ImportError('NumPy must be installed')


def reshape_feature_list(features2reshape, out_rows, out_cols, parameter_object):

    """
    Reshapes a feature statistic list

    Args:
        features2reshape (1d array)
        out_rows (int)
        out_cols (int)
        parameter_object (class object)
    """

    logger.info('  Reshaping features ...')

    # The number of dimensions
    #   for the current feature.
    out_dims = parameter_object.out_bands_dict[parameter_object.trigger]

    # The output section array.
    out_sect_array = np.empty((out_dims, out_rows, out_cols), dtype='float32')

    # Reshape each feature vector.
    for bd2wr in range(0, out_dims):
        out_sect_array[bd2wr, :, :] = np.asarray(features2reshape[bd2wr::out_dims]).reshape(out_rows, out_cols)

    return out_sect_array


def chunks2section(trigger, tk, o_r, o_c, l_rows, l_cols, out_rows, out_cols, parameter_object):

    """
    Reshapes section chunks to the full section
    
    trigger (str)
    tk (list of 1d arrays): List of returned features.
    o_c (list): List of chunk columns.
    l_rows (int): Input section rows used in feature processing. This is probably overkill, and we could just use out_rows.
    l_cols (int): Same as above for columns.
    out_rows (int): Number of output section rows.
    out_cols (int): Number of output section columns.
    parameter_object (class object)

    Returns:
        Contextual features as 3d array (features x rows x columns)
    """

    out_dims = parameter_object.out_bands_dict[trigger]

    # The output section array.
    out_sect_arr = np.empty((out_dims, out_rows, out_cols), dtype='float32')

    # The index writer for
    #   the feature vector.
    tk_ctr = 0

    # The index writer for
    #   the chunk rows.
    row_chunk_ctr = 0

    # The current section
    #   row index.
    i_sect_idx = 0

    scales_blk = parameter_object.scales[-1] - parameter_object.block

    chunk_min_blk = parameter_object.chunk_size - scales_blk

    for i in range(0, l_rows, chunk_min_blk):

        col_chunk_ctr = 0     # the index writer for the chunk columns
        j_sect_idx = 0		# the current section column index

        for j in range(0, l_cols, chunk_min_blk):

            # row and column counter taken from the feature processing
            rw = o_r[row_chunk_ctr]
            cw = o_c[col_chunk_ctr]

            # get the feature vector for the current chunk
            ts = tk[tk_ctr]

            # Here we reshape the feature vector

            bd2wr = 0

            # ignore the chunk features if there isn't data
            if len(ts) > 0:

                if trigger == 'ctr':

                    # divide the feature vector by the number of scales
                    for sc in range(0, len(ts), len(ts)/len(parameter_object.scales)):

                        # get the current feature vector and reshape into 2D
                        ts_arr = np.array(ts[sc:len(ts)/len(parameter_object.scales)+sc]).reshape(rw, cw)

                        out_sect_arr[bd2wr, i_sect_idx:i_sect_idx+rw, j_sect_idx:j_sect_idx+cw] = ts_arr

                        bd2wr += 1

                else:

                    # reshape each feature vector
                    for rs in range(0, out_dims):

                        out_sect_arr[bd2wr,
                                     i_sect_idx:i_sect_idx+rw,
                                     j_sect_idx:j_sect_idx+cw] = np.asarray(ts[rs::out_dims]).reshape(rw, cw)

                        bd2wr += 1

            tk_ctr += 1

            j_sect_idx += cw

            row_chunk_ctr += 1
            col_chunk_ctr += 1

        i_sect_idx += rw

    # get feature neighbors
    # this adds the four direct neighbors to the same feature vector
    if parameter_object.neighbors:

        # create the new output array
        # the dimensions are multiplied x5 (1 for each direct neighbor plus the center)
        out_sect_arr_temp = np.empty((out_dims*5, out_rows, out_cols), dtype='float32')

        out_dim_ctr = 0
        for dim in range(0, out_dims):

            curr_dim = out_sect_arr[dim]

            # the first feature is the original feature
            out_sect_arr_temp[out_dim_ctr] = curr_dim

            # pad by one pixel on each edge
            curr_dim = np.pad(curr_dim, ((1, 1), (1, 1)), 'edge')

            ## get the neighbors
            # top neighbor
            out_sect_arr_temp[out_dim_ctr+1] = np.roll(curr_dim, 1, axis=0)[1:out_rows+1, 1:out_cols+1]

            # left neighbor
            out_sect_arr_temp[out_dim_ctr+2] = np.roll(curr_dim, 1, axis=1)[1:out_rows+1, 1:out_cols+1]

            # right neighbor
            out_sect_arr_temp[out_dim_ctr+3] = np.roll(curr_dim, -1, axis=1)[1:out_rows+1, 1:out_cols+1]

            # bottom neighbor
            out_sect_arr_temp[out_dim_ctr+4] = np.roll(curr_dim, -1, axis=0)[1:out_rows+1, 1:out_cols+1]

            out_dim_ctr += 5

        out_sect_arr = np.copy(out_sect_arr_temp)
        del out_sect_arr_temp

    out_sect_arr[np.isnan(out_sect_arr) | np.isinf(out_sect_arr)] = 0.

    return out_sect_arr
