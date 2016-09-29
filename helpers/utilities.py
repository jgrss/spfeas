#!/usr/bin/env python

"""
@author: Jordan Graesser
Date Created: 1/26/2016
"""

import os
import shutil
import fnmatch
import random
import datetime
from collections import OrderedDict

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

# PySAL
try:
    import pysal
except:
    print('PySAL is not installed')


def random_float(low, high):
    return random.random() * (high - low) + low


def get_time_stamp(the_file):

    """
    Get the time stamp of a file

    Args:
        the_file (str): The file to get the time information from.
    """

    t = os.path.getmtime(the_file)

    the_date = datetime.datetime.fromtimestamp(t).strftime('%D')
    the_hour = datetime.datetime.fromtimestamp(t).strftime('%H')
    the_minute = datetime.datetime.fromtimestamp(t).strftime('%M')

    return the_date, the_hour, the_minute


def cleanup(items2delete):

    """
    Cleans directories of all files in the input list

    Args:
        items2delete (str list): A list of files to remove.
    """

    if isinstance(items2delete, list):

        for del_item in items2delete:

            if os.path.isfile(del_item):
                os.remove(del_item)

    elif isinstance(items2delete, str):

        if os.path.isdir(items2delete):

            for del_item in os.listdir(items2delete):

                item2delete = '{}/{}'.format(items2delete, del_item)

                if os.path.isfile(item2delete):
                    os.remove(item2delete)

            shutil.rmtree(items2delete)


def overwrite_file(file2overwrite):

    """
    Removes a file + associated parts

    Args:
        file2overwrite (str): The file to remove.
    """

    di_name, fo_name = os.path.split(file2overwrite)
    fo_base, fo_ext = os.path.splitext(fo_name)

    # List all files that contain the
    #   file name + extension.
    associated_files = fnmatch.filter(os.listdir(di_name), '*{}*'.format(fo_name))

    associated_files = fnmatch.filter(associated_files, '*{}*'.format(fo_ext))

    if associated_files:

        for associated_file in associated_files:

            associated_file_full = '{}/{}'.format(di_name, associated_file)

            if os.path.isfile(associated_file_full):
                os.remove(associated_file_full)


def shp2dataframe(input_shapefile):

    """
    Converts a shapefile .dbf to a Pandas dataframe

    Args:
        input_shapefile (str): The input shapefile.
    """

    dfs = pysal.open(input_shapefile.replace('.shp', '.dbf'), 'r')
    dfs = OrderedDict([(col, np.array(dfs.by_col(col))) for col in dfs.header])

    return pd.DataFrame(dfs)


def dataframe2dbf(df, dbf_file, my_specs=None):

    """
    Converts a pandas.DataFrame into a dbf.

    Author:
        Dani Arribas-Bel <darribas@asu.edu>

    Reference:
        https://github.com/GeoDaSandbox/sandbox/blob/master/pyGDsandbox/dataIO.py#L56

        Copyright (c) 2007-2011, GeoDa Center for Geospatial Analysis and Computation
        All rights reserved.

        Redistribution and use in source and binary forms, with or without
        modification, are permitted provided that the following conditions are met:

        * Redistributions of source code must retain the above copyright notice, this
          list of conditions and the following disclaimer.

        * Redistributions in binary form must reproduce the above copyright
          notice, this list of conditions and the following disclaimer in the
          documentation and/or other materials provided with the distribution.

        * Neither the name of the GeoDa Center for Geospatial Analysis and Computation
          nor the names of its contributors may be used to endorse or promote products
          derived from this software without specific prior written permission.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
        CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
        INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
        MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
        DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
        CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
        SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
        LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
        USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
        ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
        LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
        ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
        POSSIBILITY OF SUCH DAMAGE.

    Args:
        df (object): Pandas dataframe.
        dbf_file (str): The output .dbf file.
        my_specs (Optional[list]): List with the field_specs to use for each column. Defaults to None and
            applies the following scheme:
                int: ('N', 14, 0)
                float: ('N', 14, 14)
                str: ('C', 14, 0)

    Returns:
        None, writes to ``dbf_file``.
    """

    if my_specs:
        specs = my_specs
    else:
        type2spec = {int: ('N', 20, 0),
                     np.int64: ('N', 20, 0),
                     float: ('N', 36, 15),
                     np.float64: ('N', 36, 15),
                     str: ('C', 14, 0)}

        types = [type(df[i].iloc[0]) for i in df.columns]

        specs = [type2spec[t] for t in types]

    db = pysal.open(dbf_file, 'w')
    db.header = list(df.columns)

    db.field_spec = specs

    for i, row in df.T.iteritems():
        db.write(row)

    db.close()


def n_rows_cols(pixel_index, block_size, rows_cols):

    if pixel_index + block_size < rows_cols:
        samp_out = block_size
    else:
        samp_out = rows_cols - pixel_index

    return samp_out


def get_block_chunks(im_rows, im_cols, chunk_size, kernel_size):

    """
    Indexes:
        0: i :: actual row index
        1: isub :: adjusted row starting position
        2: iplus :: adjusted row end position
        3: ip :: row start position to read back the chunk
        4: j :: actual column index
        5: jsub :: adjusted column starting position
        6: jplus :: adjusted column end position
        7: jp :: column start position to read back the chunk
        8: n_rows :: row chunk size for GDAL
        9: n_cols :: column chunk size for GDAL
    """

    block_chunks = []

    chunk = kernel_size + chunk_size + kernel_size

    for i in xrange(0, im_rows, chunk_size-kernel_size):

        isub = i - kernel_size

        if isub < 0:
            isub = 0
            ip = 0
        else:
            ip = kernel_size

        iplus = i + chunk_size + kernel_size
        iplus = im_rows-1 if iplus >= im_rows else iplus

        n_rows = n_rows_cols(isub, chunk, im_rows)

        for j in xrange(0, im_cols, chunk_size-kernel_size):

            jsub = j - kernel_size

            if jsub < 0:
                jsub = 0
                jp = 0
            else:
                jp = kernel_size

            jplus = j + chunk_size + kernel_size
            jplus = im_cols-1 if jplus >= im_cols else jplus

            n_cols = n_rows_cols(jsub, chunk, im_cols)

            block_chunks.append([i, isub, iplus, ip, j, jsub, jplus, jp, n_rows, n_cols])

    return block_chunks


def move_files2back(image_list, list2move):

    """
    Moves files to the front of list (back of mosaic)

    Args:
        image_list (str list): The full image list.
        list2move (str list): List of image base names to move.
    """

    for scene in list2move:

        for full_scene in image_list:

            if scene in full_scene:

                image_list.insert(0, image_list.pop(image_list.index(full_scene)))

    return image_list


def check_and_create_dir(dir2create):

    if len(dir2create) > 0:

        if not os.path.isdir(dir2create):

            try:
                os.makedirs(dir2create)
            except OSError:
                raise OSError('Could not create the directory {}'.format(dir2create))
