#!/usr/bin/env python

"""
@author: Jordan Graesser
Date Created: 9/24/2011
"""

import os
import sys
import time
import subprocess
from copy import copy
import argparse
import fnmatch
from joblib import Parallel, delayed

from poly2points import poly2points
from error_matrix import error_matrix

from mpglue import raster_tools
from mpglue import vector_tools
from mpglue.helpers import _iteration_parameters_values

# NumPy
try:
    import numpy as np
except ImportError:
    raise ImportError('NumPy is not installed')

# GDAL
try:
    from osgeo import gdal
    from osgeo.gdalconst import GA_ReadOnly
except ImportError:
    raise ImportError('GDAL is not installed')


def _sample_parallel(band_position, image_name, c_list, accuracy, feature_length):

    datasource = gdal.Open(image_name, GA_ReadOnly)

    band_object = datasource.GetRasterBand(band_position)

    # image_info = raster_tools.rinfo(image_name)

    # print 'Band {:d} of {:d} ...'.format(band_position, image_info.bands)
    print 'Band {:d} of {:d} ...'.format(band_position, datasource.RasterCount)

    # image_info.get_band(band_position)

    # 1d array to store image values.
    value_list = np.zeros(feature_length, dtype='float32')

    # Sort by feature index position.
    for vi, values in enumerate(sorted(c_list)):

        # values[1] = [x, y, x offset, y offset, label]
        # values[1][2] = x offset position
        # values[1][3] = y offset position
        # pixel_value = image_info.mparray(i=values[1][3],
        #                                  j=values[1][2],
        #                                  rows=1,
        #                                  cols=1,
        #                                  d_type='float32')[0, 0]
        pixel_value = float(band_object.ReadAsArray(values[1][2], values[1][3], 1, 1)[0, 0])

        if not accuracy:
            pixel_value = float(('{:.4f}'.format(pixel_value)))
        else:
            pixel_value = int(pixel_value)

        # Update the list with raster values.
        value_list[vi] = pixel_value

    # image_info.close()

    del band_object, datasource

    band_object = None
    datasource = None

    return value_list


class SampleImage(object):

    """
    A class for image sampling

    Args:
        points_file (str): The shapefile.
        image_file (str): The raster file to sample.
        out_dir (str)
        class_id (str)
        accuracy (Optional[bool])
        n_jobs (Optional[int])
        neighbors (Optional[bool])
        field_type (Optional[str])
        use_extent (Optional[bool])
    """

    def __init__(self, points_file, image_file, out_dir, class_id, accuracy=False, n_jobs=0,
                 neighbors=False, field_type='int', use_extent=True, sql_expression_attr=[],
                 sql_expression_field='Id'):

        self.points_file = points_file
        self.image_file = image_file
        self.out_dir = out_dir
        self.class_id = class_id
        self.accuracy = accuracy
        self.n_jobs = n_jobs
        self.neighbors = neighbors
        self.field_type = field_type
        self.use_extent = use_extent
        self.sql_expression_attr = sql_expression_attr
        self.sql_expression_field = sql_expression_field

        if not os.path.isfile(self.points_file):
            raise IOError('\n{} does not exist. It should be a point shapefile.'.format(self.points_file))

        if not os.path.isfile(self.image_file):
            raise IOError('\n{} does not exist. It should be a raster image.'.format(self.image_file))

        if self.neighbors and (self.n_jobs != 0):
            print('Cannot sample neighbors in parallel, so setting ``n_jobs`` to 0.')
            self.n_jobs = 0

        if self.accuracy:
            self.header = False
        else:
            self.header = True

        self.d_name_points, f_name_points = os.path.split(self.points_file)
        self.f_base_points, __ = os.path.splitext(f_name_points)

        # Filter by SQL expression.
        if self.sql_expression_attr:
            self.points_file = self.sql()

        __, self.f_name_rst = os.path.split(self.image_file)
        self.f_base_rst, __ = os.path.splitext(self.f_name_rst)

        if not self.out_dir:
            self.out_dir = copy(self.d_name_points)
            print '\nNo output directory was given. Results will be saved to {}'.format(self.out_dir)

        self.setup_names()

    def sql(self):

        out_points = '{}/{}_sql.shp'.format(self.d_name_points, self.f_base_points)

        ogr_com = 'ogr2ogr -overwrite {} {}'.format(out_points, self.points_file)

        ogr_com_sql = '"SELECT * FROM {} WHERE {}'.format(self.f_base_points, self.sql_expression_field)

        for attr in xrange(0, len(self.sql_expression_attr)):

            if attr == 0:
                ogr_com_sql = '{} = \'{}\''.format(ogr_com_sql, self.sql_expression_attr[attr])
            else:
                ogr_com_sql = '{} OR {} = \'{}\''.format(ogr_com_sql, self.sql_expression_field,
                                                         self.sql_expression_attr[attr])

        ogr_com = '{} -sql {}"'.format(ogr_com, ogr_com_sql)

        print '\nSubsetting {} by classes {} ...\n'.format(self.points_file, ','.join(self.sql_expression_attr))

        subprocess.call(ogr_com, shell=True)

        self.d_name_points, f_name_points = os.path.split(out_points)
        self.f_base_points, __ = os.path.splitext(f_name_points)

        return out_points

    def setup_names(self):

        """
        File names and directories
        """

        self.out_dir = self.out_dir.replace('\\', '/')

        # Open the samples.
        self.shp_info = vector_tools.vinfo(self.points_file)

        # Convert polygon to points.
        if 'POINT' not in self.shp_info.shp_geom_name:

            self.points_file = self.convert2points()

            # Close the polygon shapefile
            self.shp_info.shp.Destroy()
            self.shp_info.feature.Destroy()

            self.d_name_points, f_name_points = os.path.split(self.points_file)
            self.f_base_points, __ = os.path.splitext(f_name_points)

            self.shp_info = vector_tools.vinfo(self.points_file)

        self.n_feas = self.shp_info.n_feas

        self.lyr = self.shp_info.lyr

        self.get_class_count()

        self.data_file = '{}/{}__{}_samples.txt'.format(self.out_dir, self.f_base_points, self.f_base_rst)

        # samples file
        self.sample_writer = open(self.data_file, 'w')

        # number of samples file
        self.n_samps = '{}/{}__{}_info.txt'.format(self.out_dir, self.f_base_points, self.f_base_rst)

        self.n_sample_writer = open(self.n_samps, 'w')

        # create array of zeros for the class counter
        self.count_arr = np.zeros(len(self.n_classes), dtype='uint8')

    def convert2points(self):

        out_points = '{}/{}_points.shp'.format(self.d_name_points, self.f_base_points)

        if not os.path.isfile(out_points):

            poly2points(self.points_file, out_points, self.image_file,
                        class_id=self.class_id, field_type=self.field_type, use_extent=self.use_extent)

        return out_points

    def get_class_count(self):

        """
        Input class counts
        """

        try:
            self.n_classes = [self.shp_info.lyr.GetFeature(n).GetField(self.class_id)
                              for n in xrange(0, self.shp_info.n_feas)]
        except:
            raise IOError('\nField <{}> does not exist or there is a feature issue.\n'.format(self.class_id))

        if 0 in self.n_classes:
            self.zs = True
        else:
            self.zs = False

        self.n_classes = sorted(reduce(lambda x, y: x + y if y[0] not in x else x, map(lambda x: [x], self.n_classes)))

    def sample(self):

        print '  \nSampling {} ...\n'.format(self.f_name_rst)

        # Open the image.
        self.m_info = raster_tools.rinfo(self.image_file)

        self.write_headers()

        self.fill_dictionary()

        # Return if no samples were within
        #   the raster frame.
        if len(self.coords_offsets) == 0:
            self.finish()

        value_array = self.sample_image()

        self.write2file(value_array)

        self.shp_info.close()
        self.m_info.close()

    def write_headers(self):

        """
        Writes text headers
        """

        if self.header:

            # First, x,y
            self.sample_writer.write('Id,X,Y,')

            # Then <image name.band position> format.
            [self.sample_writer.write('{}.{:d},'.format(self.f_base_rst, b)) for b in xrange(1, self.m_info.bands+1)]

            # Last, response, or class id.
            self.sample_writer.write('response\n')

    def write2file(self, value_array):

        """
        Writes samples to file
        """

        # Convert the data to strings and
        #   write to text.
        value_array = np.char.mod('%f', value_array)

        value_array = ['{}\n'.format(','.join(val_a)) for val_a in value_array]

        self.sample_writer.writelines(value_array)

    def fill_dictionary(self):

        """
        Creates a dictionary where for each feature,
            feature 1 = [x coordinate, y coordinate, x offset, y offset, class label]
            ...
            feature n = [x coordinate, y coordinate, x offset, y offset, class label]
        """

        # Dictionary to store sampled data.
        self.coords_offsets = {}

        if self.neighbors:
            self.updater = 5
        else:
            self.updater = 1

        # Iterate over each feature
        #   in the vector file.
        for n in xrange(0, self.n_feas):

            # Get the current point.
            self.feature = self.shp_info.lyr.GetFeature(n)

            # Get point geometry.
            geometry = self.feature.GetGeometryRef()

            # Get X,Y coordinates.
            x = geometry.GetX()
            y = geometry.GetY()

            # Get the class label.
            pt_id = self.feature.GetField(self.class_id)

            # Check if the sample points fall
            #   within [current] raster boundary.
            if vector_tools.is_within(x, y, self.m_info):

                # Get x, y coordinates and offsets.
                x, y, x_off, y_off = vector_tools.get_xy_offsets(image_info=self.m_info, x=x, y=y)

                # Update the counter array with the current label.
                self.count_arr[self.n_classes.index(pt_id)] += self.updater

                x = float('{:.6f}'.format(x))
                y = float('{:.6f}'.format(y))

                # Add x, y coordinates, image offset indices,
                #   and class value to the dictionary.
                self.coords_offsets[n] = [x, y, x_off, y_off, pt_id]

    def sample_image(self):

        """
        The main image sampler
        """

        # Convert position items to a list.
        c_list = self.coords_offsets.items()

        feature_length = len(self.coords_offsets)

        if self.n_jobs != 0:

            value_arr = Parallel(n_jobs=self.n_jobs)(delayed(_sample_parallel)(f_bd,
                                                                               self.image_file,
                                                                               c_list,
                                                                               self.accuracy,
                                                                               feature_length)
                                                     for f_bd in xrange(1, self.m_info.bands+1))

            value_arr = np.asarray(value_arr).T

            # The order is the same as the point labels
            #   because we iterate over the sorted (by
            #   feature position) dictionary items
            #   in both cases.

            # nx2 coordinate array
            xy_coordinates = np.zeros((feature_length, 3), dtype='float32')

            # 1d of n length labels array
            labels = np.zeros(feature_length, dtype='float32')

            # Convert position items to a list.
            co_list = self.coords_offsets.items()

            # Sort by feature index position.
            for vi, values in enumerate(sorted(co_list)):

                # values[0] = feature index position
                # values[1] = list of coordinate data

                # Fill index + x & y coordinates
                xy_coordinates[vi] = [vi, values[1][0], values[1][1]]

                # Fill labels
                labels[vi] = values[1][4]

            # Combine all of the data.
            try:
                value_arr = np.c_[xy_coordinates, value_arr, labels]
            except:
                print xy_coordinates.shape
                print value_arr.shape
                print labels.shape
                sys.exit()

        else:

            # Create the array to write values to and
            #   add three to columns -- one for the class
            #   label, two for the x, y coordinates.
            neighbor_offsets = [[0, -1], [1, 0], [0, 1], [-1, 0]]

            value_arr = np.zeros((feature_length*self.updater, self.m_info.bands+4)).astype(np.float32)

            print '\nSampling {:d} samples from {:d} image layers ...\n'.format(feature_length, self.m_info.bands)

            ctr, pbar = _iteration_parameters_values(self.m_info.bands, feature_length)

            to_delete = []

            # Iterate over each band.
            for f_bd in xrange(1, self.m_info.bands+1):

                band = self.m_info.datasource.GetRasterBand(f_bd)

                point_iter = 0      # necessary because of neighbors

                # Iterate over each point.
                #   values = [x, y, x_off, y_off, pt_id]

                # Sort by feature index position.
                for vi, values in enumerate(sorted(c_list)):

                    # Get the image offset indices.
                    x_off = values[1][2]
                    y_off = values[1][3]

                    # Get the image value.
                    try:
                        value = band.ReadAsArray(x_off, y_off, 1, 1).astype(np.float32)[0, 0]
                    except:
                        print f_bd
                        print x_off, y_off
                        band.Checksum()
                        print gdal.GetLastErrorType()
                        sys.exit()

                    if not self.accuracy:
                        value = float(('{:.4f}'.format(value)))
                    else:
                        value = int(value)

                    # Update value the list.
                    value_arr[point_iter, 0] = vi               # Index id
                    value_arr[point_iter, 1:3] = values[1][:2]  # x,y coordinates
                    value_arr[point_iter, f_bd+2] = value       # raster values
                    value_arr[point_iter, -1] = values[1][4]    # class label

                    if self.neighbors:

                        """
                        | |1| |
                        |4|x|2|
                        | |3| |
                                            1         2       3       4
                        neighbor_offsets = [[0, -1], [1, 0], [0, 1], [-1, 0]]
                        """

                        for noff in xrange(1, self.updater):

                            if (x_off + neighbor_offsets[noff-1][0] >= self.m_info.cols) or \
                                    (y_off + neighbor_offsets[noff-1][1] >= self.m_info.rows):

                                to_delete.append(point_iter + noff)
                                self.count_arr[self.n_classes.index(values[1][4])] -= 1

                                continue

                            else:

                                value = band.ReadAsArray(x_off+neighbor_offsets[noff-1][0],
                                                         y_off+neighbor_offsets[noff-1][1], 1, 1).astype(np.float32)[0, 0]

                            if not self.accuracy:
                                value = float(('{:.4f}'.format(value)))
                            else:
                                value = int(value)

                            # Write to array.
                            value_arr[point_iter+noff, 0] = \
                                values[1][0] + (neighbor_offsets[noff-1][0] * self.m_info.cellY)

                            value_arr[point_iter+noff, 1] = \
                                values[1][1] + (neighbor_offsets[noff-1][1] * -self.m_info.cellY)

                            value_arr[point_iter+noff, f_bd+1] = value
                            value_arr[point_iter+noff, -1] = values[1][4]

                    point_iter += self.updater

                    pbar.update(ctr)
                    ctr += 1

                band.FlushCache()
                band = None

            pbar.finish()

            if to_delete:
                value_arr = np.delete(value_arr, np.array(to_delete), axis=0)

        value_arr[np.isnan(value_arr) | np.isinf(value_arr)] = 0.

        return value_arr

    def finish(self):

        # Write the number of samples from
        #   the counter array.
        for nc in self.n_classes:

            self.n_sample_writer.write('Class {:d}: {:d}\n'.format(int(nc),
                                                                   int(self.count_arr[self.n_classes.index(nc)])))

        # write the total number of samples
        self.n_sample_writer.write('Total: {:d}'.format(np.sum(self.count_arr)))

        self.sample_writer.close()
        self.n_sample_writer.close()

        self.feature.Destroy()
        self.shp_info.close()

        if max(self.count_arr) == 0:
            os.remove(self.data_file)
            os.remove(self.n_samps)
        else:

            if self.accuracy:

                s = '********************************'
                print s
                print '<Confusion matrix>'
                print s
                print

                # Output confusion matrix text file.
                error_file = '{}/{}__{}_acc.txt'.format(self.out_dir, self.f_base_points, self.f_base_rst)

                emat = error_matrix()
                emat.get_stats(po_text=self.data_file)
                emat.write_stats(error_file)


def sample_raster(points, image, out_dir=None, option=1, class_id='Id', accuracy=False,
                  field_type='int', use_extent=True, sql_expression_field='Id',
                  sql_expression_attr=[], neighbors=False, search_ext=['tif'], n_jobs=0):
    
    """
    Samples an image, or imagery, using a point, or points, shapefile.

    Args:
        points (str): Points shapefile or directory containing point shapefiles.
        image (str): Image or directory containing imagery.
        out_dir (Optional[str]): The directory to save text files. Default is None, or the same directory as 
            the points shapefile(s).
        option (Optional[int]): Default is 1.
            Options:
                1 :: One point shapefile    ---> One raster file
                2 :: One point shapefile    ---> Many raster files
                3 :: Many point shapefiles  ---> Many raster files
        class_id (Optional[str]): Shapefile field id containing class values. Default is 'Id'.
        accuracy (Optional[bool]): Whether to compute accuracy from ``image``. Default is False.
        field_type (Optional[str]): The field type of ``class_id``. Default is 'int'.
        use_extent (Optional[bool]): Whether to use the extent of ``image``. Default is True.
        sql_expression_field (Optional[str]): Default is 'Id'.
        sql_expression_attr (Optional[str]): Default is [].
        neighbors (Optional[bool]): Whether to sample neighboring pixels. Default is False.
        search_ext (Optional[str list]): A list of file extensions to search. Default is ['tif'].
        n_jobs (Optional[int]): The number of parallel jobs. Default is 0.

    Returns:
        None, writes results to ``out_dir``.
    """

    # 1:1
    if option == 1:

        si = SampleImage(points, image, out_dir, class_id, accuracy=accuracy, n_jobs=n_jobs,
                         field_type=field_type, use_extent=use_extent, neighbors=neighbors,
                         sql_expression_attr=sql_expression_attr, sql_expression_field=sql_expression_field)

        si.sample()

    # 1:--
    elif option == 2:

        search_ext = ['*.{}'.format(se) for se in search_ext]

        image_list = []
        for se in search_ext:
            [image_list.append(fn) for fn in fnmatch.filter(os.listdir(image), se)]

        for im in image_list:

            im_ = '{}/{}'.format(image, im)

            si = SampleImage(points, im_, out_dir, class_id, accuracy=accuracy, n_jobs=n_jobs,
                             field_type=field_type, use_extent=use_extent, neighbors=neighbors,
                             sql_expression_attr=sql_expression_attr, sql_expression_field=sql_expression_field)

            si.sample()

    # --:1
    elif option == 3:

        point_list = fnmatch.filter(os.listdir(points), '*.shp')

        for pt in point_list:

            pt_ = '{}/{}'.format(points, pt)

            si = SampleImage(pt_, image, out_dir, class_id, accuracy=accuracy, n_jobs=n_jobs,
                             field_type=field_type, use_extent=use_extent, neighbors=neighbors,
                             sql_expression_attr=sql_expression_attr, sql_expression_field=sql_expression_field)

            si.sample()


def _options():
    
    sys.exit("""\

    1 :: One point shapefile    ---> One raster file
    2 :: One point shapefile    ---> Many raster files
    3 :: Many point shapefiles  ---> Many raster files

    """)


def _examples():
    
    sys.exit("""\

    # Sample some_image.tif with pts.shp, returning one set of sample data
    spfeas_sample_raster -s /pts.shp -i /some_image.tif -o /out_dir

    # Sample all rasters in /some_dir with pts.shp, returning one set of sample data
    spfeas_sample_raster -s /pts.shp -i /some_dir -opt 2

    # Sample all rasters in /some_dir with all shapefiles in /some_dir_pts, returning sample data for each raster
    spfeas_sample_raster -s /some_dir_pts -i /some_dir --option 3

    # Query the <trees> and <shrubs> fields in <polys.shp> prior to sampling
    spfeas_sample_raster -s /polys.shp -c CLASS -i /image.tif -o /out_dir --sql_field name --sql_attr trees shrubs

    # compute the accuracy of some_image.tif
    spfeas_sample_raster -s /pts.shp -i /some_image.tif --accuracy

    """)


def main():

    parser = argparse.ArgumentParser(description='Sample raster(s) with shapefile(s)',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-e', '--examples', dest='examples', action='store_true', help='Show usage examples and exit')
    parser.add_argument('-s', '--shapefile', dest='shapefile', help='The shapefile to sample with', default=None)
    parser.add_argument('-i', '--input', dest='input', help='The input image to sample', default=None)
    parser.add_argument('-c', '--classid', dest='classid', help='The field class id name', default='Id')
    parser.add_argument('-f', '--fieldtype', dest='fieldtype', help='The field type of the class field', default='int')
    parser.add_argument('-o', '--output', dest='output', help='Output directory or base name of text extension',
                        default=None)
    parser.add_argument('-opt', '--option', dest='option', help='The option to use', default=1, type=int,
                        choices=[1, 2, 3])
    parser.add_argument('-n', '--neighbors', dest='neighbors', help='Whether to use neighboring pixels',
                        action='store_true')
    parser.add_argument('-a', '--accuracy', dest='accuracy', help='Whether to compute accuracy', action='store_true')
    parser.add_argument('-j', '--n_jobs', dest='n_jobs', help='Number of parallel jobs', default=0, type=int)
    parser.add_argument('--sql_attr', dest='sql_attr', help='The SQL field attributes', default=[], nargs='+')
    parser.add_argument('--sql_field', dest='sql_field', help='The SQL class field', default='Id')
    parser.add_argument('--options', dest='options', help='Whether to show sampling options', action='store_true')

    args = parser.parse_args()

    if args.options:
        _options()

    if args.examples:
        _examples()

    print('\nStart date & time --- (%s)\n' % time.asctime(time.localtime(time.time())))

    start_time = time.time()

    sample_raster(args.shapefile, args.input, out_dir=args.output, option=args.option, class_id=args.classid,
                  accuracy=args.accuracy, field_type=args.fieldtype, neighbors=args.neighbors,
                  n_jobs=args.n_jobs, sql_expression_attr=args.sql_attr, sql_expression_field=args.sql_field)

    print('\nEnd data & time -- (%s)\nTotal processing time -- (%.2gs)\n' %
          (time.asctime(time.localtime(time.time())), (time.time()-start_time)))

if __name__ == '__main__':
    main()
