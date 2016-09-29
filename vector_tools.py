#!/usr/bin/env python

"""
@author: Jordan Graesser
Date Created: 9/29/2016
""" 

import os
import sys
import shutil
import time
import argparse
import subprocess
import ast
import copy
import fnmatch
import atexit

import raster_tools

# GDAL
try:
    from osgeo.gdalconst import *
    from osgeo import ogr, osr
except ImportError:
    raise ImportError('GDAL must be installed')

# NumPy
try:
    import numpy as np
except ImportError:
    raise ImportError('NumPy must be installed')

# Bottleneck
try:
    import bottleneck as bn
except ImportError:
    raise ImportError('Bottleneck must be installed')

# Pandas
try:
    import pandas as pd
except ImportError:
    raise ImportError('Pandas must be installed')

# SciPy
try:
    from scipy import stats
except ImportError:
    raise ImportError('SciPy must be installed')

# PySAL
try:
    import pysal
except:
    print('PySAL is not installed')

# Rtree
# try:
#     import rtree
# except:
#     print('Rtree is not installed')

# PyTables
try:
    import tables
except:
    print('PyTables is not installed')


class RegisterDriver(object):

    """
    Registers a vector driver.

    Args:
        vector_file (str): The vector to register.

    Attributes:
        driver (object)
        f_base (str)
        file_format (str)
    """

    def __init__(self, vector_file):

        self.out_vector = vector_file

        self._get_file_format()

        self.driver = ogr.GetDriverByName(self.file_format)
        self.driver.Register

    def _get_file_format(self):

        __, f_name = os.path.split(self.out_vector)
        self.f_base, file_extension = os.path.splitext(f_name)

        # if 'shp' not in file_extension.lower():
        #     raise NameError('\nOnly shapefiles are currently supported.\n')

        formats = {'.shp': 'ESRI Shapefile',
                   '.mem': 'MEMORY'}

        self.file_format = formats[file_extension]


class vinfo(RegisterDriver):

    """
    Gets vector information and file pointer object.

    Args:
        file_name (str): Vector location, name, and extension.
        open2read (Optional[bool]): Whether to open vector as 'read only' (True) or writeable (False).
            Default is True.
        epsg (Optional[int]): An EPSG code to declare when the .prj file is missing. Default is None.

    Attributes:
        shp (object)
        lyr (object)
        lyr_def (object)
        feature (object)
        shp_geom (list)
        shp_geom_name (str)
        n_feas (int)
        layer_count (int)
        projection (str list)
        extent (float list)
        left (float)
        top (float)
        right (float)
        bottom (float)
    """

    def __init__(self, file_name, open2read=True, epsg=None):

        self.file_name = file_name
        self.open2read = open2read
        self.epsg = epsg

        self.d_name, self.f_name = os.path.split(self.file_name)
        self.f_base, self.f_ext = os.path.splitext(self.f_name)

        RegisterDriver.__init__(self, self.file_name)

        self.open()
    
        self.get_info()

        # Check open files before closing.
        atexit.register(self.exit)

    def exit(self):

        if hasattr(self, 'file_open') and self.file_open:
            self.close()

    def open(self):

        if self.open2read:
            self.shp = ogr.Open(self.file_name, GA_ReadOnly)
        else:
            self.shp = ogr.Open(self.file_name, GA_Update)

        if self.shp is None:
            raise NameError('\nUnable to open {}.\n'.format(self.file_name))

        self.file_open = True

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
    
    def get_info(self):

        # get the layer
        self.lyr = self.shp.GetLayer()
        self.lyr_def = self.lyr.GetLayerDefn()
        
        self.feature = self.lyr.GetFeature(0)
        
        self.shp_geom = self.feature.GetGeometryRef()

        try:
            self.shp_geom_name = self.shp_geom.GetGeometryName()
        except:
            self.shp_geom_name = None

        # get the number of features in the layer
        self.n_feas = self.lyr.GetFeatureCount()

        # get the number of layers in the shapefile
        self.layer_count = self.shp.GetLayerCount()

        # get the projection
        if isinstance(self.epsg, int):

            try:

                self.spatial_reference = osr.SpatialReference()
                self.spatial_reference.ImportFromEPSG(self.epsg)

            except:
                print('Could not get the spatial reference')

        else:

            try:
                self.spatial_reference = self.lyr.GetSpatialRef()
            except:
                print('Could not get the spatial reference')

        self.projection = self.spatial_reference.ExportToWkt()

        # get the extent
        self.extent = self.lyr.GetExtent()
        
        self.left = self.extent[0]
        self.top = self.extent[3]
        self.right = self.extent[1]
        self.bottom = self.extent[2]

        self.field_names = [self.lyr_def.GetFieldDefn(i).GetName() for i in xrange(0, self.lyr_def.GetFieldCount())]

    def copy(self):

        """
        Copies the object instance
        """

        return copy.copy(self)

    def copy2(self, output_file):

        """
        Copies the input vector to another vector

        Args:
            output_file (str): The output vector.

        Returns:
            None, writes to ``output_file``.
        """

        __ = self.driver.CopyDataSource(self.shp, output_file)

    def close(self):

        self.shp.Destroy()

        self.file_open = False

    def delete(self):

        """
        Deletes an open file
        """

        if not self.open2read:
            raise NameError('The file must be opened in read-only mode.')

        try:
            self.driver.DeleteDataSource(self.file_name)
        except IOError:
            raise IOError('\n{} could not be deleted. Check for a file lock.\n'.format(self.file_name))

        self._cleanup()

    def _cleanup(self):

        """
        Cleans undeleted files
        """

        file_list = fnmatch.filter(os.listdir(self.d_name), '{}*'.format(self.f_name))

        if file_list:

            for rf in file_list:
                os.remove('{}/{}'.format(self.d_name, rf))


def copy_vector(file_name, output_file):

    """
    Copies a vector file

    Args:
        file_name (str): The file to copy.
        output_file (str): The file to copy to.

    Returns:
        None
    """

    with vinfo(file_name) as v_info:
        v_info.copy2(output_file)


def delete_vector(file_name):

    """
    Deletes a vector file

    Args:
        file_name (str): The file to delete.

    Returns:
        None
    """

    with vinfo(file_name) as v_info:
        v_info.delete()


class CreateDriver(RegisterDriver):

    """
    Creates a vector driver.

    Args:
        out_vector (str): The vector to create.
        overwrite (bool): Whether to overwrite an existing file.

    Attributes:
        datasource (object)
    """

    def __init__(self, out_vector, overwrite):

        RegisterDriver.__init__(self, out_vector)

        if overwrite:

            if os.path.isfile(out_vector):
                self.driver.DeleteDataSource(out_vector)

        # create the output driver
        self.datasource = self.driver.CreateDataSource(out_vector)

    def close(self):
        self.datasource.Destroy()


class create_vector(CreateDriver):

    """
    Creates a vector file.

    Args:
        out_vector (str): The output file name.
        field_names (Optional[str list]): The field names to create. Default is ['Id'].
        epsg (Optional[int]): The projection of the output vector, given by EPSG projection code. Default is 0.
        projection_from_file (Optional[str]): An file to grab the projection from. Default is None.
        projection (Optional[int]): The projection of the output vector, given as a string. Default is None.
        field_type (Optional[str]): The output field type. Default is 'int'.
        geom_type (Optional[str]): The output geometry type. Default is 'point'. Choices are ['point', 'polygon'].
        overwrite (Optional[bool]): Whether to overwrite an existing file. Default is True.

    Attributes:
        time_stamp (str)
        lyr (object)
        lyr_def (object)
        field_defs (object)

    Returns:
        None
    """

    def __init__(self, out_vector, field_names=['Id'], epsg=0, projection_from_file=None,
                 projection=None, field_type='int', geom_type='point', overwrite=True):

        self.time_stamp = time.asctime(time.localtime(time.time()))

        CreateDriver.__init__(self, out_vector, overwrite)

        if geom_type == 'point':
            geom_type = ogr.wkbPoint
        elif geom_type == 'polygon':
            geom_type = ogr.wkbPolygon

        if epsg > 0:

            sp_ref = osr.SpatialReference()
            sp_ref.ImportFromEPSG(epsg)

            # create the point layer
            self.lyr = self.datasource.CreateLayer(self.f_base, geom_type=geom_type, srs=sp_ref)

        elif isinstance(projection_from_file, str):

            sp_ref = osr.SpatialReference()
            sp_ref.ImportFromWkt(projection_from_file)

            # create the point layer
            self.lyr = self.datasource.CreateLayer(self.f_base, geom_type=geom_type, srs=sp_ref)

        elif isinstance(projection, str):

            self.lyr = self.datasource.CreateLayer(self.f_base, geom_type=geom_type, srs=projection)

        else:

            self.lyr = self.datasource.CreateLayer(self.f_base, geom_type=geom_type)

        self.lyr_def = self.lyr.GetLayerDefn()

        # create the field
        if field_type == 'int':

            self.field_defs = [ogr.FieldDefn(field, ogr.OFTInteger) for field in field_names]

        elif field_type == 'float':

            self.field_defs = [ogr.FieldDefn(field, ogr.OFTReal) for field in field_names]

        elif field_type == 'string':

            self.field_defs = []

            for field in field_names:

                field_def = ogr.FieldDefn(field, ogr.OFTString)
                field_def.SetWidth(20)
                self.field_defs.append(field_def)

        # create the fields
        [self.lyr.CreateField(field_def) for field_def in self.field_defs]


def rename_vector(input_file, output_file):

    """
    Renames a shapefile and all of its associated files

    Args:
        input_file (str): The file to rename.
        output_file (str): The renamed file.

    Examples:
        >>> from mappy import vector_tools
        >>> vector_tools.rename_vector('/in_vector.shp', '/out_vector.shp')

    Returns:
        None
    """

    d_name, f_name = os.path.split(input_file)
    f_base, f_ext = os.path.splitext(f_name)

    od_name, of_name = os.path.split(output_file)
    of_base, of_ext = os.path.splitext(of_name)

    associated_files = os.listdir(d_name)

    for associated_file in associated_files:

        a_base, a_ext = os.path.splitext(associated_file)

        if a_base == f_base:

            try:
                os.rename('{}/{}'.format(d_name, associated_file), '{}/{}{}'.format(od_name, of_base, a_ext))
            except OSError:
                raise OSError('Could not write {} to file.'.format(of_base))


def merge_vectors(shps2merge, merged_shapefile):

    """
    Merges a list of shapefiles into one shapefile

    Args:
        shps2merge (str list): A list of shapefiles to merge.
        merged_shapefile (str): The output merged shapefile.

    Examples:
        >>> from mappy import vector_tools
        >>> vector_tools.merge_vectors(['/in_shp_01.shp', '/in_shp_02.shp'],
        >>>                            '/merged_file.shp')

    Returns:
        None, writes to ``merged_shapefile``.
    """

    # First copy any of the shapefiles in
    # the list so that we have something
    # to merge to.
    d_name, f_name = os.path.split(shps2merge[0])
    f_base, f_ext = os.path.splitext(f_name)

    od_name, of_name = os.path.split(merged_shapefile)
    of_base, of_ext = os.path.splitext(of_name)

    associated_files = os.listdir(d_name)

    for associated_file in associated_files:

        a_base, a_ext = os.path.splitext(associated_file)

        if a_base == f_base:

            out_file = '{}/{}{}'.format(od_name, of_base, a_ext)

            if not os.path.isfile(out_file):
                shutil.copy2('{}/{}'.format(d_name, associated_file), out_file)

    # Then merge each shapefile into the
    # output file.
    for shp2merge in shps2merge[1:]:

        print 'Merging {} ...'.format(shp2merge)

        com = 'ogr2ogr -f "ESRI Shapefile" -update -append {} {} -nln {}'.format(merged_shapefile, shp2merge, of_base)

        subprocess.call(com, shell=True)


def add_point(x, y, layer_object, field, value2write):

    """
    Adds a point to an existing vector.

    Args:
        x (float)
        y (float)
        layer_object (object)
        field (str)
        value2write (str)

    Returns:
        None
    """

    pt_geom = ogr.Geometry(ogr.wkbPoint)

    # add the point
    pt_geom.AddPoint(x, y)

    # create a new feature
    feat = ogr.Feature(layer_object.lyr_def)

    feat.SetGeometry(pt_geom)

    # set the field value
    feat.SetField(field, value2write)

    # create the point
    layer_object.lyr.CreateFeature(feat)

    feat.Destroy()


def add_polygon(vector_object, xy_pairs=None, field_vals={}, geometry=None):

    """
    Args:
        vector_object (object): Class instance of ``create_vector``.
        xy_pairs (Optional[tuple]): List of x, y coordinates that make the feature. Default is None.
        field_vals (Optional[dict]): A dictionary of field values to write. They should match the order
            of ``field_defs``. Default is [].
        geometry (Optional[object]): A polygon geometry object to write (in place of ``xy_pairs``). Default is None.

    Returns:
        None
    """

    poly_geom = ogr.Geometry(ogr.wkbLinearRing)

    # Add the points
    if isinstance(xy_pairs, tuple):

        for pair in xy_pairs:
            poly_geom.AddPoint(float(pair[0]), float(pair[1]))

        poly = ogr.Geometry(ogr.wkbPolygon)

        poly.AddGeometry(poly_geom)

    else:
        poly = geometry

    feature = ogr.Feature(vector_object.lyr_def)
    feature.SetGeometry(poly)

    # set the fields
    if field_vals:

        for field, val in field_vals.iteritems():
            feature.SetField(field, val)

    vector_object.lyr.CreateFeature(feature)

    vector_object.lyr.SetFeature(feature)

    feature.Destroy()


def dataframe2dbf(df, dbf_file, my_specs=None):

    """
    Converts a pandas.DataFrame into a dbf.

    Author:  Dani Arribas-Bel <darribas@asu.edu>
        https://github.com/GeoDaSandbox/sandbox/blob/master/pyGDsandbox/dataIO.py#L56

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


def shp2dataframe(input_shp):

    """
    Uses PySAL to convert shapefile .dbf to a Pandas dataframe

    Author: Dani Arribas-Bel <darribas@asu.edu>
        https://github.com/GeoDaSandbox/sandbox/blob/master/pyGDsandbox/dataIO.py#L56

    Args:
        input_shp (str): The input shapefile

    Returns:
        Pandas dataframe
    """

    df = pysal.open(input_shp.replace('.shp', '.dbf'), 'r')

    df = dict([(col, np.array(df.by_col(col))) for col in df.header])

    return pd.DataFrame(df)


class ZonalStats(object):

    """
    Computes zonal statistics of points or raster values within polygon features

    Args:
        poly_shp (str): The polygon shapefile.
        point_shp (Optional[str]): A point shapefile to use for the zone values. Default is None.
        raster_file (Optional[str]): A raster file to use for the zone values. Default is None.
        point_h5 (Optional[str]): A HDF5 database with point tuples to use for the zone values. Default is None.
        poly_field (Optional[str]): The polygon zone field. Default is 'UNQ'.
        point_field (Optional[str]): The point value field. Default is 'Id'.
        statistic (Optional[str or str list]): The statistic to compute. Default is 'sum'. Choices are ['all', 'min',
            'max', 'mean', '5p', '25p', 'med', '75p', '95p', 'std', 'var', 'cv', 'sum', 'skew'].
        no_data (Optional[float or int]): A 'no data' value to ignore with ``raster_file``. Default is 0.
        point_h5_x (Optional[str]): The ``point_h5`` x field. Default is None.
        point_h5_y (Optional[str]): The ``point_h5`` y field. Default is None.
        point_h5_v (Optional[str]): The ``point_h5`` value field. Default is None.
        h5_title (Optional[str]): The HDF table name. Default is None.

    Methods:
        run
        write2file

    Examples:
        >>> from mappy.vector_tools import ZonalStats
        >>>
        >>> # Use a point shapefile
        >>> zs = ZonalStats('/in_poly.shp', point_shp='/in_points.shp', statistic='sum')
        >>> zs.run()
        >>> zs.write2file('/out_stats.csv')
        >>>
        >>> # Use a point list
        >>> zs = ZonalStats('/in_poly.shp', statistic=['sum', 'mean'],
        >>>                 point_list=[(x1, y1, value1), (x2, y2, value2), ..., (xN, yN, valueN)])
        >>> zs.run()
        >>>
        >>> # Use a HDF file
        >>> zs = ZonalStats('/in_poly.shp', point_h5='/in_hdf.h5', statistic='all',
        >>>                 point_h5_x='x', point_h5_y='y', point_h5_v='size')
        >>> zs.run()
    """

    def __init__(self, poly_shp, point_shp=None, raster_file=None, point_h5=None,
                 poly_field='UNQ', point_field='Id', statistic='sum', no_data=0,
                 ignore_id=0, point_h5_x=None, point_h5_y=None, point_h5_v=None,
                 h5_title=None):

        self.poly_shp = poly_shp
        self.point_shp = point_shp
        self.raster_file = raster_file
        self.point_h5 = point_h5

        self.poly_field = poly_field
        self.point_field = point_field

        self.statistic = statistic
        self.no_data = no_data
        self.ignore_id = ignore_id
        self.point_h5_x = point_h5_x
        self.point_h5_y = point_h5_y
        self.point_h5_v = point_h5_v
        self.h5_title = h5_title

        if not isinstance(self.poly_shp, str):
            raise TypeError('`poly_shp` must be a string')

        if not isinstance(self.poly_field, str):
            raise TypeError('`poly_field` must be a string')

        if not isinstance(self.point_field, str):
            raise TypeError('`point_field` must be a string')

        if isinstance(self.statistic, str) and self.statistic == 'all':
            self.statistic = [option for option in self._options() if option != 'all']

        if isinstance(self.statistic, str):

            if self.statistic not in self._options():
                raise KeyError('Choose from {}'.format(self._options()))

        elif isinstance(self.statistic, list):

            for stat in self.statistic:

                if stat not in self._options():
                    raise KeyError('Choose from {}'.format(self._options()))

        # Open the files
        self.poly_info = vinfo(self.poly_shp)

        if isinstance(self.point_shp, str) and isinstance(self.raster_file, str):
            raise ValueError('Use either `point_shp` or `raster_file`, but not both.')

        if isinstance(self.point_shp, str):
            self.point_info = vinfo(self.point_shp)

    def _options(self):

        return ['all', 'min', 'max', 'mean', '5p', '25p', 'med', '75p', '95p',
                'std', 'var', 'cv', 'cvmed', 'sum', 'skew']

    def _create_dictionary(self):

        """
        Creates a Rtree dictionary of the polygon features
        """

        # Setup the Rtree index.
        # self.rtree_index = rtree.index.Index(interleaved=False)

        # Create the feature Id dictionary.
        self.feature_dict = {}
        self.count_dict = {}

        # Add each polygon feature to
        #   Rtree index.
        for n in xrange(0, self.poly_info.n_feas):

            # current feature
            poly_feature = self.poly_info.lyr.GetFeature(n)

            # Set the polygon geometry.
            poly_geometry = poly_feature.GetGeometryRef()

            # Get the feature extent.
            left, right, bottom, top = poly_geometry.GetEnvelope()

            # self.rtree_index.insert(n, (left, right, bottom, top))

            # feature Id
            poly_feature_id = poly_feature.GetField(self.poly_field)

            if poly_feature_id in self.feature_dict:
                continue

            self.count_dict[str(int(poly_feature_id))] = 0

            if self.statistic == 'sum':
                self.feature_dict[str(int(poly_feature_id))] = 0.
            elif self.statistic == 'min':
                self.feature_dict[str(int(poly_feature_id))] = 9999999.
            elif self.statistic == 'max':
                self.feature_dict[str(int(poly_feature_id))] = -9999999.
            else:
                self.feature_dict[str(int(poly_feature_id))] = []

    def run(self):

        self._create_dictionary()

        d_name, f_name = os.path.split(self.poly_shp)
        f_base, __ = os.path.splitext(f_name)

        # Create a temporary folder
        if isinstance(self.raster_file, str):

            temp_dir = '{}/temp_{}'.format(d_name, '{:.4f}'.format(abs(np.random.randn(1)[0]))[-4:])

            if not os.path.isdir(temp_dir):
                os.makedirs(temp_dir)

        if isinstance(self.raster_file, str):
            i_info = raster_tools.rinfo(self.raster_file)

        ############
        # HDF5 table
        ############

        if isinstance(self.point_h5, str):

            if not os.path.isfile(self.point_h5):
                raise OSError('The HDF file does not exist.')

            # Open the HDF5 file.
            if not self.h5_title:
                h5_file = tables.open_file(self.point_h5, mode='a')
            else:
                h5_file = tables.open_file(self.point_h5, mode='a', title=self.h5_title)

            # Open the HDF5 table.
            table = h5_file.root.metadata

        # Iterate over each feature in the polygon.
        for n in xrange(0, self.poly_info.n_feas):

            if n % 50 == 0:

                if (n + 49) > self.poly_info.n_feas:
                    end_feature = self.poly_info.n_feas
                else:
                    end_feature = n + 49

                print 'Features {:,d}--{:,d} of {:,d} ...'.format(n, end_feature, self.poly_info.n_feas)

            # Get the current polygon feature.
            poly_feature = self.poly_info.lyr.GetFeature(n)

            # Get the id name.
            poly_name = poly_feature.GetField(self.poly_field)

            # Set the polygon geometry for
            #   the current zone feature.
            poly_geometry = poly_feature.GetGeometryRef()

            ############
            # HDF5 table
            ############

            if isinstance(self.point_h5, str):

                # Get the feature extent.
                left, right, bottom, top = poly_geometry.GetEnvelope()

                # Get points within the extent.
                #   *This also includes points outside
                #   of the envelope, but it restricts the
                #   following search.
                points_list = [dict(x=item[self.point_h5_x], y=item[self.point_h5_y], v=item[self.point_h5_v])
                               for item in table.where("""({} > {:f}) & ({} < {:f}) & \
                               ({} > {:f}) & ({} < {:f})""".format(self.point_h5_x, left,
                                                                   self.point_h5_x, right,
                                                                   self.point_h5_y, bottom,
                                                                   self.point_h5_y, top))]

                # Create a temporary point file. The
                #   value field is created and called 'Value'.
                cv = create_vector('temp_points.mem',
                                   field_names=['Value'],
                                   projection=self.poly_shp,
                                   geom_type='point')

                # Add projected points (in memory) to
                #   the current frame.
                for current_point_dict in points_list:

                    # ``current_point_dict`` = dict('x': x coordinate, 'y': y coordinate, 'v': value)

                    # Add a point to the memory file and fill
                    #   the 'Value' field with the point value.
                    add_point(current_point_dict['x'], current_point_dict['y'], cv, 'Value', current_point_dict['v'])

                    # point = ogr.Geometry(ogr.wkbPoint)
                    # point.AddPoint(current_point[0], current_point[1])
                    # point.ExportToWkt()
                    # point.Intersection(poly_geometry)

                # Set a spatial filter for points within
                #   the current feature (i.e., envelope).
                cv.lyr.SetSpatialFilter(poly_geometry)

                # Get the statistics for the
                #   current point ('Value' field).
                point_values = [point_feature.GetField('Value') for point_feature in cv.lyr]

                if self.statistic == 'sum':
                    self.feature_dict[str(int(poly_name))] += sum(point_values)
                elif self.statistic == 'min':
                    self.feature_dict[str(int(poly_name))] = min(self.feature_dict[str(int(poly_name))], point_values)
                elif self.statistic == 'max':
                    self.feature_dict[str(int(poly_name))] = max(self.feature_dict[str(int(poly_name))], point_values)
                else:
                    self.feature_dict[str(int(poly_name))] += point_values

                # Clear the spatial filter.
                cv.lyr.SetSpatialFilter(None)

                # Remove the temporary vector.
                cv = None

            #################
            # POINT SHAPEFILE
            #################

            elif isinstance(self.point_shp, str):

                # Set a spatial filter for points within
                #   current feature.
                self.point_info.lyr.SetSpatialFilter(poly_geometry)

                # Get the statistics for the
                #   current feature.
                point_values = [point_feature.GetField(self.point_field) for point_feature in self.point_info.lyr]

                if self.statistic == 'sum':
                    self.feature_dict[str(int(poly_name))] += sum(point_values)
                elif self.statistic == 'min':
                    self.feature_dict[str(int(poly_name))] = min(self.feature_dict[str(int(poly_name))], point_values)
                elif self.statistic == 'max':
                    self.feature_dict[str(int(poly_name))] = max(self.feature_dict[str(int(poly_name))], point_values)
                else:
                    self.feature_dict[str(int(poly_name))] += point_values

                    # Clear the filter.
                self.point_info.lyr.SetSpatialFilter(None)

            #############
            # RASTER FILE
            #############

            if isinstance(self.raster_file, str):

                # Get the feature extent.
                left, right, bottom, top = poly_geometry.GetEnvelope()

                # Check if the image boundary is outside
                # of the zones boundary.
                if i_info.outside(dict(left=left, right=right, bottom=bottom, top=top)):
                    continue

                out_zones_raster = '{}/{}_{}_temp.tif'.format(temp_dir, f_base, str(poly_name))

                # Rasterize the current feature.
                com = 'gdal_rasterize -q -init 0 -a {} -te {:f} {:f} {:f} {:f} -tr {:f} {:f} -ot UInt32 \
                -where "{}="{}"" --config GDAL_CACHEMAX 256 -co TILED=YES \
                -l {} {} {}'.format(self.poly_field, left, bottom, right, top, i_info.cellY, i_info.cellY,
                                    self.poly_field, str(poly_name), f_base, self.poly_shp, out_zones_raster)

                subprocess.call(com, shell=True)

                out_value_raster = '{}/{}_{}_temp.vrt'.format(temp_dir, f_base, str(poly_name))

                # Subset the value raster.
                com = 'gdalwarp -q -multi -wm 256 --config GDAL_CACHEMAX 256 -of VRT \
                -te {:f} {:f} {:f} {:f} -tr {:f} {:f} -ot Float32 \
                {} {}'.format(left, bottom, right, top, i_info.cellY, i_info.cellY, self.raster_file, out_value_raster)

                subprocess.call(com, shell=True)

                if not os.path.isfile(out_value_raster):

                    if os.path.isfile(out_zones_raster):
                        os.remove(out_zones_raster)

                    continue

                if not os.path.isfile(out_zones_raster):

                    if os.path.isfile(out_value_raster):
                        os.remove(out_value_raster)

                    continue

                s_info = raster_tools.rinfo(out_value_raster)
                v_info = raster_tools.rinfo(out_zones_raster)

                blk_rows, blk_cols = raster_tools.block_dimensions(s_info.rows, s_info.cols,
                                                                   row_block_size=2048,
                                                                   col_block_size=2048)

                for i in xrange(0, s_info.rows, blk_rows):

                    n_rows = raster_tools.n_rows_cols(i, blk_rows, s_info.rows)

                    for j in xrange(0, s_info.cols, blk_cols):

                        n_cols = raster_tools.n_rows_cols(j, blk_cols, s_info.cols)

                        # Open the arrays.
                        vct_arr = v_info.mparray(i=i, j=j, rows=n_rows, cols=n_cols, d_type='uint32')

                        if vct_arr.max() == self.ignore_id:
                            continue

                        img_arr = s_info.mparray(i=i, j=j, rows=n_rows, cols=n_cols, d_type='float32')

                        if img_arr.max() == self.no_data:
                            continue

                        img_arr[(vct_arr != int(poly_name)) | (img_arr == self.no_data)] = np.nan

                        if self.statistic == 'sum':
                            self.feature_dict[str(int(poly_name))] += bn.nansum(img_arr)
                        elif self.statistic == 'min':
                            self.feature_dict[str(int(poly_name))] = np.minimum(self.feature_dict[str(int(poly_name))],
                                                                                bn.nanmin(img_arr))
                        elif self.statistic == 'max':
                            self.feature_dict[str(int(poly_name))] = np.maximum(self.feature_dict[str(int(poly_name))],
                                                                                bn.nanmax(img_arr))
                        else:

                            idx = np.where(~np.isnan(img_arr))
                            self.feature_dict[str(int(poly_name))] += [iv for iv in img_arr[idx].ravel()]

                s_info.close()
                v_info.close()

                os.remove(out_zones_raster)
                os.remove(out_value_raster)

        if isinstance(self.point_h5, str):

            if h5_file.isopen:
                h5_file.close()

        if isinstance(self.raster_file, str):

            shutil.rmtree(temp_dir)

            i_info.close()

    def write2file(self, output, append_data=False, join2original=False, group_uniques=False, overwrite=False):

        """
        Args:
            output (str): The output statistics file.
            append_data (Optional[bool]): Whether to append raw data to the output statistics. Default is False.
            join2original (Optional[bool]): Whether to join the statistics table to ``self.poly_shp``. Default
                is False.
            group_uniques (Optional[bool]): Whether to group unique values of ``poly_field`` (in the case of multiple
                cases of unique features). Default is False.
            overwrite (Optional[bool]): Whether to overwrite an existing table. Default is False.
        """

        d_name, f_name = os.path.split(output)
        f_base, f_ext = os.path.splitext(f_name)

        if self.statistic in ['min', 'max', 'sum']:

            sum_field = self.statistic.upper()

            dfo = pd.DataFrame(self.feature_dict.items(), columns=[self.poly_field, sum_field])

            # Control the decimal places.
            dfo[sum_field] = dfo[sum_field].map('{:.4f}'.format)

        else:

            # The data list column will be named 1.
            dfo = pd.DataFrame(self.feature_dict.items())

            dfo = self.calculate_stats(dfo)

            if append_data:

                # Rename the data column.
                dfo.rename(columns={1: 'DATA'}, inplace=True)

                # Convert the data list into a string.
                dfo['DATA'] = dfo.apply(self.join_data, axis=1)

            else:

                # Remove the data column.
                dfo.drop(1, axis=1, inplace=True)

            # Rename the unique column to match
            #   the polygon id field.
            dfo.rename(columns={0: self.poly_field}, inplace=True)

        # Open the shapefile table.
        df = shp2dataframe(self.poly_shp)

        # Force column types to strings.
        df[self.poly_field] = df[self.poly_field].astype(str)
        dfo[self.poly_field] = dfo[self.poly_field].astype(str)

        # Join the shapefile table and the data table.
        dfo_ = pd.merge(df, dfo, on=self.poly_field, how='inner')

        # Set the index column name.
        dfo_.index.name = 'Index_Id'

        if group_uniques:
            dfo_ = dfo_.groupby(self.poly_field).first()

        # if join2original:
        #
        #     temp_table = '{}/{}_stats.csv'.format(d_name, f_base)
        #     output_shp = '{}/{}_stats.shp'.format(d_name, f_base)
        #
        #     dfo.to_csv(temp_table, sep=',')
        #
        #     join2shapefile(self.poly_shp, temp_table, output_shp, shp_field=self.poly_field, tbl_field=self.poly_field)
        #
        #     os.remove(temp_table)

        if overwrite:

            if os.path.isfile(output):

                try:
                    os.remove(output)
                except:
                    raise OSError('Could not remove existing table.')

        # Write the merged file to CSV.
        dfo_.to_csv(output, sep=',')

    def calculate_stats(self, dataframe):

        for data_field in self.statistic:
            dataframe[data_field] = dataframe.apply(self.apply2cell, axis=1, args=(data_field.upper(),))

        return dataframe

    def apply2cell(self, dataframe_cell, field):

        if field == 'MIN':
            return '{:.4f}'.format(min(dataframe_cell[1]))
        elif field == 'MAX':
            return '{:.4f}'.format(max(dataframe_cell[1]))
        elif field == 'MEAN':
            return '{:.4f}'.format(np.mean(dataframe_cell[1]))
        elif field == '5p':
            return '{:.4f}'.format(np.percentile(dataframe_cell[1], 5))
        elif field == '25p':
            return '{:.4f}'.format(np.percentile(dataframe_cell[1], 25))
        elif field == 'MED':
            return '{:.4f}'.format(np.median(dataframe_cell[1]))
        elif field == '75p':
            return '{:.4f}'.format(np.percentile(dataframe_cell[1], 75))
        elif field == '95p':
            return '{:.4f}'.format(np.percentile(dataframe_cell[1], 95))
        elif field == 'STD':
            return '{:.4f}'.format(np.std(dataframe_cell[1]))
        elif field == 'VAR':
            return '{:.4f}'.format(np.var(dataframe_cell[1]))
        elif field == 'CV':
            return '{:.4f}'.format(np.std(dataframe_cell[1]) / np.mean(dataframe_cell[1]))
        elif field == 'CVMED':
            return '{:.4f}'.format(np.std(dataframe_cell[1]) / np.median(dataframe_cell[1]))
        elif field == 'SUM':
            return '{:.4f}'.format(sum(dataframe_cell[1]))
        elif field == 'SKEW':
            return '{:.4f}'.format(stats.skew(dataframe_cell[1]))

    def join_data(self, dataframe_cell):
        return ','.join([str(v) for v in sorted(dataframe_cell['DATA'])])


def is_within(x, y, image_info):

    """
    Checks whether x, y coordinates are within an image extent.

    Args:
        x (float): The x coordinate.
        y (float): The y coordinate.
        image_info (object): Object of ``mappy.rinfo``.

    Returns:
        ``True`` if ``x`` and ``y`` are within ``image_info``, otherwise ``False``.
    """

    if not isinstance(image_info, raster_tools.rinfo):
        raise TypeError('`image_info` must be an instance of `rinfo`.')

    if not hasattr(image_info, 'left') or not hasattr(image_info, 'right') \
            or not hasattr(image_info, 'bottom') or not hasattr(image_info, 'top'):

        raise AttributeError('The `image_info` object must have left, right, bottom, top attributes.')

    if (x > image_info.left) and (x < image_info.right) and (y > image_info.bottom) and (y < image_info.top):
        return True
    else:
        return False


class Transform(object):

    """
    Transforms a x, y coordinate pair

    Args:
        x (float): The source x coordinate.
        y (float): The target y coordinate.
        source_epsg (int): The source EPSG code.
        target_epsg (int): The target EPSG code.

    Examples:
        >>> from mappy.vector_tools import Transfrom
        >>>
        >>> ptr = Transform(740000., 2260000., 102033, 4326)
        >>> print ptr.x, ptr.y
        >>> print ptr.x_transform, ptr.y_transform
    """

    def __init__(self, x, y, source_epsg, target_epsg):

        self.x = x
        self.y = y

        source_sr = osr.SpatialReference()
        source_sr.ImportFromEPSG(source_epsg)

        target_sr = osr.SpatialReference()
        target_sr.ImportFromEPSG(target_epsg)

        coord_trans = osr.CoordinateTransformation(source_sr, target_sr)

        self.point = ogr.Geometry(ogr.wkbPoint)

        self.point.AddPoint(self.x, self.y)
        self.point.Transform(coord_trans)

        self.x_transform = self.point.GetX()
        self.y_transform = self.point.GetY()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        self.point.Destroy()


def create_point(coordinate_pair, projection_file):

    """
    Creates a point in memory

    Args:
        coordinate_pair (list or tuple): The x, y coordinate pair.
        projection_file (str): The file with the EPSG projection info.
    """

    rsn = '{:f}'.format(abs(np.random.randn(1)[0]))[-4:]

    # Create a temporary point file. The
    #   value field is created and called 'Value'.
    cv = create_vector('temp_points_{}.mem'.format(rsn),
                       field_names=['Value'],
                       projection_from_file=projection_file,
                       geom_type='point')

    # Add a point.
    add_point(coordinate_pair[0], coordinate_pair[1], cv, 'Value', 1)

    return cv


def intersects_boundary(meta_dict, boundary_file):

    """
    Checks if an image extent intersects a polygon boundary.

    Args:
        meta_dict (dict): A dictionary of extent information.
            E.g., dict(UL=[x, y], UR=[x, y], LL=[x, y], LR=[x, y]).
        boundary_file (str): A boundary shapefile to check.

    Returns:
        True if ``meta_dict`` coordinates intersect ``boundary_shp``, otherwise False.
    """

    with vinfo(boundary_file) as bdy_info:

        bdy_feature = bdy_info.lyr.GetFeature(0)

        bdy_geometry = bdy_feature.GetGeometryRef()

        # Create a polygon object from the coordinates.
        coord_wkt = 'POLYGON (({:f} {:f}, {:f} {:f}, {:f} {:f}, {:f} {:f}, {:f} {:f}))'.format(meta_dict['UL'][0],
                                                                                               meta_dict['UL'][1],
                                                                                               meta_dict['UR'][0],
                                                                                               meta_dict['UR'][1],
                                                                                               meta_dict['LR'][0],
                                                                                               meta_dict['LR'][1],
                                                                                               meta_dict['LL'][0],
                                                                                               meta_dict['LL'][1],
                                                                                               meta_dict['UL'][0],
                                                                                               meta_dict['UL'][1])

        coord_poly = ogr.CreateGeometryFromWkt(coord_wkt)

        # If the polygon object is empty,
        #   then the two do not intersect.
        is_empty = bdy_geometry.Intersection(coord_poly).IsEmpty()

    if is_empty:
        return False
    else:
        return True

    # # for key, coordinate_pair in meta_dict.iteritems():
    #
    # # Create a point in memory.
    # # cv = create_point(coordinate_pair, boundary_shp)
    #
    # # Set a spatial filter to check if the
    # #   point is within the current feature
    # #   (i.e., envelope).
    # cv.lyr.SetSpatialFilterRect(poly_geometry)
    #
    # # Store the point in a list if it
    # #   is within the envelope.
    # n_points = cv.lyr.GetFeatureCount()
    #
    # # Clear the spatial filter.
    # cv.lyr.SetSpatialFilter(None)
    #
    # # Remove the temporary vector.
    # cv = None
    #
    # if n_points > 0:
    #
    #     poly_feature.Destroy()
    #
    #     return True
    #
    # poly_feature.Destroy()
    #
    # return False


def _get_xy_offsets(x, left, right, y, top, bottom, cell_size_x, cell_size_y, round_offset, check):

    # Xs (longitudes)
    if check:
        if (x < left) or (x > right):
            raise ValueError('The x is out of the image extent.')

    if ((x > 0) and (left < 0)) or ((left > 0) and (x < 0)):

        if round_offset:
            x_offset = int(round((abs(x) + abs(left)) / abs(cell_size_x)))
        else:
            x_offset = int((abs(x) + abs(left)) / abs(cell_size_x))

    else:

        if round_offset:
            x_offset = int(round(abs(abs(x) - abs(left)) / abs(cell_size_x)))
        else:
            x_offset = int(abs(abs(x) - abs(left)) / abs(cell_size_x))

    # Ys (latitudes)
    if check:
        if (y > top) or (y < bottom):
            raise ValueError('The y is out of the image extent.')

    if ((y > 0) and (top < 0)) or ((top > 0) and (y < 0)):

        if round_offset:
            y_offset = int(round((abs(y) + abs(top)) / cell_size_y))
        else:
            y_offset = int((abs(y) + abs(top)) / cell_size_y)

    else:

        if round_offset:
            y_offset = int(round(abs(abs(y) - abs(top)) / cell_size_y))
        else:
            y_offset = int(abs(abs(y) - abs(top)) / cell_size_y)

    return x_offset, y_offset


def get_xy_offsets(image_info=None, image_list=[], x=None, y=None, feature=None, xy_info=None,
                   round_offset=False, check_position=True):

    """
    Get coordinate offsets

    Args:
        image_info (object): Object of ``mappy.rinfo``.
        image_list (Optional[list]): [left, top, right, bottom, cellx, celly]. Default is [].
        x (Optional[float]): An x coordinate. Default is None.
        y (Optional[float]): A y coordinate. Default is None.
        feature (Optional[object]): Object of ``ogr.Feature``. Default is None.
        xy_info (Optional[object]): Object of ``mappy.vinfo`` or ``mappy.rinfo``. Default is None.
        round_offset (Optional[bool]): Whether to round offsets. Default is False.
        check_position (Optional[bool])

    Examples:
        >>> from mappy import vector_tools
        >>>
        >>> # With an image and x, y coordinates.
        >>> x, y, x_offset, y_offset = vector_tools.get_xy_coordinates(image_info=i_info, x=x, y=y)
        >>>
        >>> # With an image and a feature object.
        >>> x, y, x_offset, y_offset = vector_tools.get_xy_coordinates(image_info=i_info, feature=feature)
        >>>
        >>> # With an image and a ``rinfo`` or ``vinfo` instance.
        >>> x, y, x_offset, y_offset = vector_tools.get_xy_coordinates(image_info=i_info, xy_info=v_info)

    Returns:
        X coordinate, Y coordinate, X coordinate offset, Y coordinate offset
    """

    # The offset is given as a single x, y coordinate.
    if isinstance(feature, ogr.Feature):

        # Get point geometry.
        geometry = feature.GetGeometryRef()

        # Get X,Y coordinates.
        x = geometry.GetX()
        y = geometry.GetY()

    # The offset is from a vector or
    #   raster information object.
    elif isinstance(xy_info, vinfo) or isinstance(xy_info, raster_tools.rinfo):

        x = xy_info.left
        y = xy_info.top

    else:
        if not isinstance(x, float):
            raise ValueError('A coordinate or feature object must be given.')

    # Check if a list or an
    #   object/instance is given.
    if image_list:

        left = image_list[0]
        top = image_list[1]
        right = image_list[2]
        bottom = image_list[3]
        cell_x = image_list[4]
        cell_y = image_list[5]

    else:

        left = image_info.left
        top = image_info.top
        right = image_info.right
        bottom = image_info.bottom
        cell_x = image_info.cellX
        cell_y = image_info.cellY

    # Compute pixel offsets.
    x_offset, y_offset = _get_xy_offsets(x, left, right, y, top, bottom, cell_x, cell_y, round_offset, check_position)

    return x, y, x_offset, y_offset


class get_xy_coordinates(object):

    """
    Converts i, j indices to map coordinates.

    Args:
        i (int): The row index position.
        j (int): The column index position.
        rows (int): The number of rows in the array.
        cols (int): The number of columns in the array.
        image_info (object): An instance of ``raster_tools.rinfo``.
    """

    def __init__(self, i, j, rows, cols, image_info):

        self.get_extent(i, j, rows, cols, image_info)

    def get_extent(self, i, j, rows, cols, image_info):

        if (image_info.top > 0) and (image_info.bottom < 0):

            # Get the number of pixels top of center.
            n_pixels_top = int(np.ceil(image_info.top / image_info.cellY))

            if i > n_pixels_top:
                self.top = -(i - n_pixels_top) * image_info.cellY
            else:
                self.top = image_info.top - (i * image_info.cellY)

        else:
            self.top = image_info.top - (i * image_info.cellY)

        if (image_info.right > 0) and (image_info.left < 0):

            # Get the number of pixels left of center.
            n_pixels_left = int(np.ceil(abs(image_info.left) / image_info.cellY))

            if j > n_pixels_left:
                self.left = (j - n_pixels_left) * image_info.cellY
            else:
                self.left = image_info.left + (j * image_info.cellY)

        else:
            self.left = image_info.left + (j * image_info.cellY)

        self.bottom = self.top - (rows * image_info.cellY)
        self.right = self.left + (cols * image_info.cellY)


def spatial_intersection(select_shp, intersect_shp, output_shp, epsg=None):

    """
    Creates a new shapefile from a spatial intersection of two shapefiles

    Args:
        select_shp (str): The shapefile to select from.
        intersect_shp (str): The shapefile to test for intersection.
        output_shp (str): The output shapefile.
        epsg (Optional[int]): An EPSG code to declare when the .prj file is missing. Default is None.
    """

    # Open the files.
    with vinfo(select_shp, epsg=epsg) as select_info, vinfo(intersect_shp, epsg=epsg) as intersect_info:

        tracker_list = []

        # Create the output shapefile
        field_names = list_field_names(select_shp, be_quiet=True)

        if epsg > 0:

            o_shp = create_vector(output_shp, field_names=field_names,
                                  geom_type=select_info.shp_geom_name.lower(), epsg=epsg)

        else:

            o_shp = create_vector(output_shp, field_names=field_names, projection_from_file=select_shp,
                                  geom_type=select_info.shp_geom_name.lower())

        # Iterate over each select feature in the polygon.
        for m in xrange(0, select_info.n_feas):

            if m % 500 == 0:

                if (m + 499) > select_info.n_feas:
                    end_feature = select_info.n_feas
                else:
                    end_feature = m + 499

                print 'Select features {:d}--{:d} of {:d} ...'.format(m, end_feature, select_info.n_feas)

            # Get the current polygon feature.
            select_feature = select_info.lyr.GetFeature(m)

            # Set the polygon geometry.
            select_geometry = select_feature.GetGeometryRef()

            # Iterate over each intersecting feature in the polygon.
            for n in xrange(0, intersect_info.n_feas):

                # Get the current polygon feature.
                intersect_feature = intersect_info.lyr.GetFeature(n)

                # Set the polygon geometry.
                intersect_geometry = intersect_feature.GetGeometryRef()

                left, right, bottom, top = intersect_geometry.GetEnvelope()

                # No need to check intersecting features
                # if outside bounds.
                if (left > select_info.right) or (right < select_info.left) or (top < select_info.bottom) \
                        or (bottom > select_info.top):
                    continue

                # Test the intersection.
                if select_geometry.Intersect(intersect_geometry):

                    # Get the id name of the select feature.
                    # select_id = select_feature.GetField(select_field)

                    # Don't add a feature on top of existing one.
                    if m not in tracker_list:

                        field_values = {}

                        # Get the field names and values.
                        for field in field_names:
                            field_values[field] = select_feature.GetField(field)

                        # Add the feature.
                        add_polygon(o_shp, field_vals=field_values, geometry=select_geometry)

                        tracker_list.append(m)

        o_shp.close()


def select_and_save(file_name, out_vector, select_field=None, select_value=None,
                    expression=None, overwrite=True, epsg=None):

    """
    Selects a vector feature by an attribute and save to new file.

    Args:
        file_name (str): The file name to select from.
        out_vector (str): The output vector file.
        select_field (str): The field to select from.
        select_value (str): The field value to select.
        expression (Optional[str]): A conditional expression. E.g., "FIELD = 'VALUE' OR FIELD = 'VALUE2'".
        overwrite (Optional[bool]): Whether to overwrite an existing file. Default is True.
        epsg (Optional[int]): An EPSG code to declare when the .prj file is missing. Default is None.

    Returns:
        None

    Examples:
        >>> import mappy as mp
        >>>
        >>> # Save features where 'Id' is equal to 1.
        >>> mp.select_and_save('/in_shapefile.shp', '/out_shapefile.shp', 'Id', '1')
    """

    if not os.path.isfile(file_name):
        raise NameError('\n{} does not exist'.format(file_name))

    d_name, f_name = os.path.split(out_vector)
    f_base, f_ext = os.path.splitext(f_name)

    if not os.path.isdir(d_name):
        os.makedirs(d_name)

    # Open the input shapefile.
    with vinfo(file_name, epsg=epsg) as v_info:

        # Select the attribute by an expression.
        if not isinstance(expression, str):
            v_info.lyr.SetAttributeFilter("{} = '{}'".format(select_field, select_value))
        else:
            v_info.lyr.SetAttributeFilter(expression)

        # Create the output shapefile.
        out_driver_source = CreateDriver(out_vector, overwrite)

        out_lyr = out_driver_source.datasource.CopyLayer(v_info.lyr, f_base)

        out_lyr = None


def list_field_names(in_shapefile, be_quiet=False, epsg=None):

    """
    Lists all field names in a shapefile

    Args:
        in_shapefile (str)
        be_quiet (Optional[bool])
        epsg (Optional[int]): An EPSG code to declare when the .prj file is missing. Default is None.

    Returns:
        List of field names
    """

    d_name, f_name = os.path.split(in_shapefile)

    # Open the input shapefile.
    with vinfo(in_shapefile, epsg=epsg) as v_info:

        df_fields = pd.DataFrame(columns=['Name', 'Type', 'Length'])

        for i in xrange(0, v_info.lyr_def.GetFieldCount()):

            df_fields.loc[i, 'Name'] = v_info.lyr_def.GetFieldDefn(i).GetName()
            df_fields.loc[i, 'Type'] = v_info.lyr_def.GetFieldDefn(i).GetTypeName()
            df_fields.loc[i, 'Length'] = v_info.lyr_def.GetFieldDefn(i).GetWidth()

    if not be_quiet:

        print '\n{} has the following fields:\n'.format(f_name)
        print df_fields

    return df_fields


def euclidean_distance(lons, lats):
    return np.sqrt(((lons[0] - lons[1])**2.) + ((lats[0] - lats[1])**2.))


def buffer_vector(file_name, out_vector, distance=None, epsg=None, field_name=None):

    """
    Buffers a vector file.

    Args:
        file_name (str): The vector file to buffer.hex_shp
        out_vector (str): The output, buffered vector file.
        distance (Optional[float]): The buffer distance, in projection units. Default is None.
        epsg (Optional[int]): An EPSG code to declare when the .prj file is missing. Default is None.
        field_name (Optional[str]): A field to use as the buffer value. Default is None.

    Returns:
        None

    Examples:
        >>> import mappy as mp
        >>>
        >>> # 10 km buffer
        >>> mp.buffer_vector('/in_shapefile.shp', '/out_buffer.shp', distance=10000.)
        >>>
        >>> # Buffer by field name
        >>> mp.buffer_vector('/in_shapefile.shp', '/out_buffer.shp', field_name='Buffer')
    """

    if not os.path.isfile(file_name):
        raise NameError('\n{} does not exist'.format(file_name))

    if not isinstance(distance, float) and not isinstance(field_name, str):
        raise ValueError('Either the distance or field name must be given.')

    d_name, f_name = os.path.split(out_vector)

    if not os.path.isdir(d_name):
        os.makedirs(d_name)

    # open the input shapefile
    with vinfo(file_name, epsg=epsg) as v_info:

        if isinstance(distance, float):
            print '\nBuffering {} by {:f} distance ...'.format(f_name, distance)
        else:
            print '\nBuffering {} by field {} ...'.format(f_name, field_name)

        # create the output shapefile
        if isinstance(epsg, int):
            cv = create_vector(out_vector, epsg=epsg, geom_type='polygon')
        else:
            cv = create_vector(out_vector, projection_from_file=v_info.projection, geom_type='polygon')

        df_fields = list_field_names(file_name, be_quiet=True)

        field_names = df_fields['Name'].values.tolist()

        cv = create_fields(cv, field_names,
                           df_fields['Type'].values.tolist(),
                           df_fields['Length'].values.tolist())

        for feature in v_info.lyr:

            in_geom = feature.GetGeometryRef()

            if isinstance(field_name, str):

                try:
                    distance = float(feature.GetField(field_name))
                except:
                    continue

                if distance is None or distance == 'None':
                    continue

            geom_buffer = in_geom.Buffer(distance)

            out_feature = ogr.Feature(cv.lyr_def)
            out_feature.SetGeometry(geom_buffer)

            for fn in field_names:
                out_feature.SetField(fn, feature.GetField(fn))

            cv.lyr.CreateFeature(out_feature)

        cv.close()


def convex_hull(in_shp, out_shp):

    """
    Creates a convex hull of a polygon shapefile

    Reference:
        This code was slightly modified to fit into MapPy.

        Project:        Geothon (https://github.com/MBoustani/Geothon)
        File:           Conversion_Tools/shp_convex_hull.py
        Description:    This code generates convex hull shapefile for point, line and polygon shapefile
        Author:         Maziyar Boustani (github.com/MBoustani)

    Args:
        in_shp (str): The input vector file.
        out_shp (str): The output convex hull polygon vector file.

    Returns:
        None
    """

    v_info = vinfo(in_shp)

    # output convex hull polygon
    cv = create_vector(out_shp, projection=v_info.projection, geom_type='polygon')

    # define convex hull feature
    convex_hull_feature = ogr.Feature(cv.lyr_def)

    # define multipoint geometry to store all points
    multipoint = ogr.Geometry(ogr.wkbMultiPoint)

    # iterate over each feature
    for each_feature in xrange(0, v_info.n_feas):

        shp_feature = v_info.lyr.GetFeature(each_feature)

        feature_geom = shp_feature.GetGeometryRef()

        # if geometry is MULTIPOLYGON then need to get POLYGON then LINEARRING to be able to get points
        if feature_geom.GetGeometryName() == 'MULTIPOLYGON':

            for polygon in feature_geom:
                for linearring in polygon:
                    points = linearring.GetPoints()
                    for point in points:
                        point_geom = ogr.Geometry(ogr.wkbPoint)
                        point_geom.AddPoint(point[0], point[1])
                        multipoint.AddGeometry(point_geom)

        # if geometry is POLYGON then need to get LINEARRING to be able to get points
        elif feature_geom.GetGeometryName() == 'POLYGON':

            for linearring in feature_geom:
                points = linearring.GetPoints()
                for point in points:
                    point_geom = ogr.Geometry(ogr.wkbPoint)
                    point_geom.AddPoint(point[0], point[1])
                    multipoint.AddGeometry(point_geom)

        # if geometry is MULTILINESTRING then need to get LINESTRING to be able to get points
        elif feature_geom.GetGeometryName() == 'MULTILINESTRING':

            for multilinestring in feature_geom:
                for linestring in multilinestring:
                    points = linestring.GetPoints()
                    for point in points:
                        point_geom = ogr.Geometry(ogr.wkbPoint)
                        point_geom.AddPoint(point[0], point[1])
                        multipoint.AddGeometry(point_geom)

        # if geometry is MULTIPOINT then need to get POINT to be able to get points
        elif feature_geom.GetGeometryName() == 'MULTIPOINT':

            for multipoint in feature_geom:
                for each_point in multipoint:
                    points = each_point.GetPoints()
                    for point in points:
                        point_geom = ogr.Geometry(ogr.wkbPoint)
                        point_geom.AddPoint(point[0], point[1])
                        multipoint.AddGeometry(point_geom)

        #  if the geometry is POINT or LINESTRING then get points
        else:
            points = feature_geom.GetPoints()
        for point in points:
            point_geom = ogr.Geometry(ogr.wkbPoint)
            point_geom.AddPoint(point[0], point[1])
            multipoint.AddGeometry(point_geom)

    # convert multipoint to convex hull geometry
    convex_hull = multipoint.ConvexHull()

    # set the geomerty of convex hull shapefile feature
    convex_hull_feature.SetGeometry(convex_hull)

    # add the feature to convex hull layer
    cv.lyr.CreateFeature(convex_hull_feature)

    # close the datasets
    v_info.close()
    cv.close()


def create_fields(v_info, field_names, field_types, field_widths):

    """
    Creates fields in an existing vector

    Args:
        v_info (object): A ``vinfo`` object.
        field_names (str list): A list of field names to create.
        field_types (str list): A list of field types to create. Choices are ['real', 'float', 'int', str'].
        field_widths (int list): A list of field widths.

    Examples:
        >>> from mappy import vector_tools
        >>>
        >>> vector_tools.create_fields(v_info, ['Id'], ['int'], [5])

    Returns:
        The ``vinfo`` object.
    """

    type_dict = {'float': ogr.OFTReal, 'Real': ogr.OFTReal,
                 'int': ogr.OFTInteger, 'Integer': ogr.OFTInteger,
                 'int64': ogr.OFTInteger64, 'Integer64': ogr.OFTInteger64,
                 'str': ogr.OFTString, 'String': ogr.OFTString}

    # Create the fields.
    field_defs = []

    for field_name, field_type, field_width in zip(field_names, field_types, field_widths):

        field_def = ogr.FieldDefn(field_name, type_dict[field_type])

        if field_type in ['str', 'String']:
            field_def.SetWidth(field_width)
        elif field_type in ['float', 'Real']:
            field_def.SetPrecision(4)

        field_defs.append(field_def)

        v_info.lyr.CreateField(field_def)

    return v_info


def add_fields(input_vector, output_vector=None, field_names=['x', 'y'], method='field-xy',
               area_units='km', constant=1, epsg=None, field_breaks=None, default_value=None,
               field_type=None):

    """
    Adds x, y coordinate fields to an existing vector.

    Args:
        input_vector (str): The input vector.
        output_vector (Optional[str]): An output vector iwth ``method``='dissolve'. Default is None.
        field_names (Optional[str list]): The field names. Default is ['x', 'y'].
        method (Optional[str]): The method to use. Default is 'field-xy'. Choices are
            ['field-xy', 'field-id', 'field-area', 'field-constant', 'field-dissolve'].
        area_units (Optional[str]): The units to use for calculating area. Default is 'km', or square km.
            *Assumes the input units are meters if you use 'km'. Choices area ['ha', 'km'].
        constant (Optional[int]): A constant value when ``method`` is equal to field-constant. Default is 1.
        epsg (Optional[int]): An EPSG code to declare when the .prj file is missing. Default is None.
        field_breaks (Optional[dict]): The field breaks. Default is None.
        default_value (Optional[int, float, or str]): The default break value. Default is None.

    Returns:
        None, writes to ``input_vector`` in place.
    """

    if method in ['field-xy', 'field-area']:
        field_type = 'float'
    elif method == 'field-id':
        field_type = 'int'
    elif method == 'field-merge':
        field_type = 'str'
    else:

        if not field_type:

            if isinstance(default_value, float):
                field_type = 'float'
            elif isinstance(default_value, int):
                field_type = 'int'
            elif isinstance(default_value, str):
                field_type = 'str'
            else:
                field_type = 'int'

    __, f_name = os.path.split(input_vector)
    f_base, __ = os.path.splitext(f_name)

    # First open the vector file.
    v_info = vinfo(input_vector, open2read=False, epsg=epsg)

    # Create the new id field.
    field_names_ = [v_info.lyr_def.GetFieldDefn(i).GetName() for i in xrange(0, v_info.lyr_def.GetFieldCount())]

    if method == 'field-xy':

        if len(field_names) != 2:
            raise ValueError('There should be two {} field names'.format(method))

        for field_name in field_names:
            if field_name not in field_names_:
                v_info = create_fields(v_info, [field_name], [field_type], [None])

        # Add the centroids to each feature.
        for fi, feature in enumerate(v_info.lyr):

            if (fi + 99) < v_info.n_feas:
                remaining = fi + 99
            else:
                remaining = (v_info.n_feas - fi) + fi

            if fi % 100 == 0:
                print 'Features {:d}--{:d} of {:d} ...'.format(fi, remaining, v_info.n_feas)

            geometry = feature.GetGeometryRef()

            centroid = geometry.Centroid()

            feature.SetField(field_names[0], centroid.GetX())
            feature.SetField(field_names[1], centroid.GetY())

            v_info.lyr.SetFeature(feature)

            feature.Destroy()

    elif method == 'field-breaks':

        if len(field_names) != 2:
            raise ValueError('There should be two {} field names'.format(method))

        for field_name in field_names:
            if field_name not in field_names_:
                v_info = create_fields(v_info, [field_name], [field_type], [50])

        for fi, feature in enumerate(v_info.lyr):

            if (fi + 99) < v_info.n_feas:
                remaining = fi + 99
            else:
                remaining = (v_info.n_feas - fi) + fi

            if fi % 100 == 0:
                print 'Features {:d}--{:d} of {:d} ...'.format(fi, remaining, v_info.n_feas)

            try:
                field_value = float(feature.GetField(field_names[0]))
            except:
                continue

            if field_value is None or field_value == 'None':
                continue

            value_found = False

            for key, break_values in field_breaks.iteritems():

                if isinstance(break_values, int) or isinstance(break_values, float):

                    if field_value >= break_values:

                        feature.SetField(field_names[1], key)

                        value_found = True

                        break

                else:

                    if break_values[0] <= field_value < break_values[1]:

                        feature.SetField(field_names[1], key)

                        value_found = True

                        break

            if not value_found:
                feature.SetField(field_names[1], default_value)

            v_info.lyr.SetFeature(feature)

            feature.Destroy()

    elif method == 'field-id':

        if len(field_names) != 1:
            raise ValueError('There should be one {} field name'.format(method))

        if field_names[0] not in field_names_:
            v_info = create_fields(v_info, field_names, [field_type], [None])

        # Add the id to each feature.
        for fi, feature in enumerate(v_info.lyr):

            if (fi + 99) < v_info.n_feas:
                remaining = fi + 99
            else:
                remaining = (v_info.n_feas - fi) + fi

            if fi % 100 == 0:
                print 'Features {:d}--{:d} of {:d} ...'.format(fi, remaining, v_info.n_feas)

            feature.SetField(field_names[0], fi+1)

            v_info.lyr.SetFeature(feature)

            feature.Destroy()

    elif method == 'field-constant':

        if len(field_names) != 1:
            raise ValueError('There should be one {} field name'.format(method))

        if field_names[0] not in field_names_:
            v_info = create_fields(v_info, field_names, [field_type], [None])

        # Add the id to each feature.
        for fi, feature in enumerate(v_info.lyr):

            if (fi + 99) < v_info.n_feas:
                remaining = fi + 99
            else:
                remaining = (v_info.n_feas - fi) + fi

            if fi % 100 == 0:
                print 'Features {:d}--{:d} of {:d} ...'.format(fi, remaining, v_info.n_feas)

            feature.SetField(field_names[0], constant)

            v_info.lyr.SetFeature(feature)

            feature.Destroy()

    elif method == 'field-area':

        if len(field_names) != 1:
            raise ValueError('There should be one {} field name'.format(method))

        if field_names[0] not in field_names_:
            v_info = create_fields(v_info, field_names, [field_type], [None])

        # Add the id to each feature.
        for fi, feature in enumerate(v_info.lyr):

            if (fi + 99) < v_info.n_feas:
                remaining = fi + 99
            else:
                remaining = (v_info.n_feas - fi) + fi

            if fi % 100 == 0:
                print 'Features {:d}--{:d} of {:d} ...'.format(fi, remaining, v_info.n_feas)

            geometry = feature.GetGeometryRef()

            area = geometry.GetArea()

            # Convert square meters to square kilometers or to hectares
            if area_units == 'km':
                area *= .000001
            elif area_units == 'ha':
                area *= .0001

            # float('%.4f' % area)

            feature.SetField(field_names[0], area)

            v_info.lyr.SetFeature(feature)

            feature.Destroy()

    elif method == 'field-dissolve':

        if len(field_names) != 1:
            raise ValueError('There should be one {} field name'.format(method))

        if not isinstance(output_vector, str):
            raise ValueError('The output vector must be given.')

        # Dissolve the field.
        com = 'ogr2ogr {} {} -dialect sqlite -sql "SELECT ST_Union(geometry), \
        {} FROM {} GROUP BY {}"'.format(output_vector, input_vector, field_names[0], f_base, field_names[0])

        print 'Dissolving {} by {} ...'.format(input_vector, field_names[0])

        subprocess.call(com, shell=True)

    elif method == 'field-label':

        if len(field_names) != 2:
            raise ValueError('There should be two {} field names'.format(method))

        for field_name in field_names:
            if field_name not in field_names_:
                v_info = create_fields(v_info, [field_name], [field_type], [5])

        # Get the class names.
        all_class_names = []
        for fi, feature in enumerate(v_info.lyr):
            all_class_names.append(str(feature.GetField(field_names[0])))

        class_names = list(set(all_class_names))

        class_dictionary = dict(zip(class_names, range(1, len(class_names)+1)))

        # Reopen the shapefile.
        v_info.close()
        v_info = vinfo(input_vector, open2read=False, epsg=epsg)

        # Add the class values.
        for fi, feature in enumerate(v_info.lyr):

            if (fi + 99) < v_info.n_feas:
                remaining = fi + 99
            else:
                remaining = (v_info.n_feas - fi) + fi

            if fi % 100 == 0:
                print 'Features {:d}--{:d} of {:d} ...'.format(fi, remaining, v_info.n_feas)

            feature.SetField(field_names[1], class_dictionary[str(feature.GetField(field_names[0]))])

            v_info.lyr.SetFeature(feature)

            feature.Destroy()

    elif method == 'field-merge':

        if len(field_names) != 3:
            raise ValueError('There should be three {} field names'.format(method))

        if field_names[2] not in field_names_:
            v_info = create_fields(v_info, [field_names[2]], [field_type], [20])

        # Merge the two fields.
        for fi, feature in enumerate(v_info.lyr):

            if (fi + 99) < v_info.n_feas:
                remaining = fi + 99
            else:
                remaining = (v_info.n_feas - fi) + fi

            if fi % 100 == 0:
                print 'Features {:d}--{:d} of {:d} ...'.format(fi, remaining, v_info.n_feas)

            feature.SetField(field_names[2],
                             ''.join([str(feature.GetField(field_names[0])), str(feature.GetField(field_names[1]))]))

            v_info.lyr.SetFeature(feature)

            feature.Destroy()

    else:
        raise NameError('{} is not a method'.format(method))

    v_info.close()


def _examples():

    sys.exit("""\

    #############
    # INFORMATION
    #############

    # Get vector information
    vector_tools.py -i /in_shape.shp --method info

    # List the field names
    vector_tools.py -i /in_shape.shp -m fields

    ###########
    # PROCESSES
    ###########

    # Create a 10 km buffer
    vector_tools.py -i /in_shape.shp -o /out_buffer.shp -d 10000 -m buffer

    # Select the Id field where it is equal to 1, then save to new file
    vector_tools.py -i /in_shape.shp -o /out_selection.shp -f Id -v 1 -m select
    # OR
    vector_tools.py -i /in_shape.shp -o /out_selection.shp --expression "Id = '1'" --method select

    # Select features of A.shp that intersect B.shp
    vector_tools.py -iss A.shp -isi B.shp -o output.shp -m spatial --epsg 102033

    # Rename a shapefile in place
    vector_tools.py -i /in_vector.shp -o /out_vector.shp --method rename

    # Copy a shapefile
    vector_tools.py -i /in_vector.shp -o /out_vector.shp --method copy2

    # Delete a shapefile
    vector_tools.py -i /in_vector.shp --method delete

    # Merge multiple shapefiles
    vector_tools.py -sm /in_vector_01.shp /in_vector_02.shp -o /merged.shp --method merge

    # Dissolve a shapefile by a field.
    vector_tools.py -i /in_vector.shp -o /dissolved.shp --method field-dissolve --field-names DissolveField

    ########
    # FIELDS
    ########

    # Add x, y coordinate fields
    vector_tools.py -i /in_vector.shp --method field-xy --field-names X Y

    # Add an ordered id field
    vector_tools.py -i /in_vector.shp --method field-id --field-names id

    # Add an area field
    vector_tools.py -i /in_vector.shp --method field-area --field-names Area

    # Add unique class labels based on a named field. In this example, the field
    #   that contains the class names is 'Name' and the class id field to be
    #   created is 'Id'.
    vector_tools.py -i /in_vector.shp --method field-label --field-names Name Id

    # Merge two field names (f1 and f2) into a new field (merged).
    vector_tools.py -i /in_vector.shp --method field-merge --field-names f1 f2 merged

    # Set field values based on range parameters. In this example, the test is:
    #   If the value of 'f1' is --> (1 <= value < 10), then set field 'f2' as 1.
    #   If the value of 'f1' is --> (10 <= value < 20), then set field 'f2' as 2.
    vector_tools.py -i /in_vector.shp --method field-breaks --field-names f1 f2 --field-breaks "{1: [1, 10], 2: [10, 20]}"

    """)


def main():

    parser = argparse.ArgumentParser(description='Vector tools',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-e', '--examples', dest='examples', action='store_true', help='Show usage examples and exit')
    parser.add_argument('-i', '--input', dest='input', help='The input shapefile', default=None)
    parser.add_argument('-iss', '--input_select', dest='input_select', help='The select shapefile with -m spatial',
                        default=None)
    parser.add_argument('-isi', '--input_intersect', dest='input_intersect',
                        help='The intersect shapefile with -m spatial', default=None)
    parser.add_argument('-o', '--output', dest='output', help='The output shapefile', default=None)
    parser.add_argument('-m', '--method', dest='method', help='The method to run', default=None,
                        choices=['buffer', 'copy2', 'delete', 'fields', 'info', 'merge', 'rename', 'select', 'spatial',
                                 'field-xy', 'field-id', 'field-area', 'field-constant',
                                 'field-dissolve', 'field-merge', 'field-label', 'field-breaks'])
    parser.add_argument('-d', '--distance', dest='distance', help='The buffer distance', default=None, type=float)
    parser.add_argument('-f', '--field', dest='field', help='The field to select', default=None)
    parser.add_argument('-v', '--value', dest='value', help='The field selection value', default=None)
    parser.add_argument('-sm', '--shps2merge', dest='shps2merge', help='A list of shapefiles to merge', default=None,
                        nargs='+')
    parser.add_argument('-fn', '--field-name', dest='field_name', help='The field name', default=None)
    parser.add_argument('-fns', '--field-names', dest='field_names',
                        help='The field name(s) to add', default=['x', 'y'], nargs='+')
    parser.add_argument('-b', '--field-breaks', dest='field_breaks', help='The field breaks', default="{}")
    parser.add_argument('-dv', '--default-value', dest='default_value', help='The default break value', default=None)
    parser.add_argument('--area_units', dest='area_units', help='The units to use for area calcuation', default='km')
    parser.add_argument('--constant', dest='constant', help='A constant value for -m field-constant', default='1')
    parser.add_argument('--expression', dest='expression', help='A query expression', default=None)
    parser.add_argument('--epsg', dest='epsg', help='An EPSG projection code', default=0, type=int)

    args = parser.parse_args()

    if args.examples:
        _examples()

    print('\nStart date & time --- (%s)\n' % time.asctime(time.localtime(time.time())))

    start_time = time.time()

    if args.method == 'info':

        v_info = vinfo(args.input, epsg=args.epsg)

        print '\nThe projection:\n'
        print v_info.projection

        print '\nThe extent (left, right, top, bottom):\n'
        print '{:f}, {:f}, {:f}, {:f}'.format(v_info.left, v_info.right, v_info.top, v_info.bottom)

        print '\nThe geometry:\n'
        print v_info.shp_geom_name

        v_info.close()

    elif args.method == 'buffer':
        buffer_vector(args.input, args.output, distance=args.distance,
                         epsg=args.epsg, field_name=args.field_name)
    elif args.method == 'fields':
        list_field_names(args.input, epsg=args.epsg)
    elif args.method == 'select':
        select_and_save(args.input, args.output, args.field, args.value,
                        expression=args.expression, epsg=args.epsg)
    elif args.method == 'spatial':
        spatial_intersection(args.input_select, args.input_intersect, args.output, epsg=args.epsg)
    elif args.method == 'rename':
        rename_vector(args.input, args.output)
    elif args.method == 'copy2':
        copy_vector(args.input, args.output)
    elif args.method == 'delete':
        delete_vector(args.input)
    elif args.method == 'merge':
        merge_vectors(args.shps2merge, args.output)
    elif args.method in ['field-xy', 'field-id', 'field-area', 'field-constant',
                         'field-dissolve', 'field-merge', 'field-label', 'field-breaks']:

        add_fields(args.input, output_vector=args.output, method=args.method,
                   field_names=args.field_names, area_units=args.area_units,
                   constant=args.constant, epsg=args.epsg,
                   field_breaks=ast.literal_eval(args.field_breaks),
                   default_value=args.default_value)

    print('\nEnd data & time -- (%s)\nTotal processing time -- (%.2gs)\n' %
          (time.asctime(time.localtime(time.time())), (time.time()-start_time)))

if __name__ == '__main__':
    main()
