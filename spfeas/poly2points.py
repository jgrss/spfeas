#!/usr/bin/env python

"""
@author: Jordan Graesser
Date Created: 12/20/2012
"""

import os
import sys
import time
import subprocess
from copy import copy

from mpglue import raster_tools
from mpglue import vector_tools

# GDAL
try:
    from osgeo import ogr
except ImportError:
    raise ImportError('GDAL did not load')
    
# NumPy
try:
    import numpy as np
except ImportError:
    raise ImportError('NumPy did not load')


def _add_points_from_raster(out_shp, class_id, field_type, in_rst, epsg=None, proj=None, no_data_val=-1, skip=1):

    # create the new shapefile
    # lyr, shp_src = createVct(out_shp, class_id, vct_info.proj, field_type)
    mpc = vector_tools.create_vector(out_shp, field_names=[class_id], epsg=epsg, projection=proj, field_type=field_type)

    pt_geom = ogr.Geometry(ogr.wkbPoint)                            # create the geometry

    ## write the points
    m_info = raster_tools.rinfo(in_rst)

    m_info.storage = 'byte'

    band = m_info.datasource.GetRasterBand(1)

    block = band.ReadAsArray().astype(np.float32)

    if m_info.rows > 512:
        blk_size_rws = 512
    else:
        blk_size_rws = np.copy(m_info.rows)

    if m_info.cols > 512:
        blk_size_cls = 512
    else:
        blk_size_cls = np.copy(m_info.cols)

    ttl_blks = int((np.ceil(float(m_info.rows) / float(blk_size_rws))) * \
                   (np.ceil(float(m_info.cols) / float(blk_size_cls))))
    ttl_blks_ct = 1

    print '\nConverting to points ...\n'

    for i in xrange(0, m_info.rows, blk_size_rws):

        n_rows = raster_tools.n_rows_cols(i, blk_size_rws, m_info.rows)

        for j in xrange(0, m_info.cols, blk_size_cls):

            if ttl_blks_ct == 1:

                sPl = ttl_blks_ct + 8

                print '  Blocks %d -- %d of %d ...' % (ttl_blks_ct, sPl, ttl_blks)

                newI = ttl_blks_ct + 9

            elif ttl_blks_ct == newI:

                if (ttl_blks_ct + 9) > ttl_blks:
                    sPl = copy(ttl_blks)
                else:
                    sPl = ttl_blks_ct + 9

                print '  Blocks %d -- %d of %d ...' % (ttl_blks_ct, sPl, ttl_blks)

                newI = ttl_blks_ct + 10

            n_cols = raster_tools.n_rows_cols(j, blk_size_cls, m_info.cols)

            block = band.ReadAsArray(j, i, n_cols, n_rows).astype(np.float32)

            blk_mean = block.mean()
            blk_mean = float('%.2f' % blk_mean)

            if blk_mean != no_data_val:

                top = m_info.top - (float(i) * m_info.cellY)

                for i2 in xrange(0, n_rows, skip):

                    left = m_info.left + (float(j) * m_info.cellY)

                    for j2 in xrange(0, n_cols, skip):

                        val = block[i2, j2]

                        if int(str(val)[str(val).find('.')+1]) == 0:
                            val = int(val)
                        else:
                            val = float('%.2f' % val)

                        if val != no_data_val:

                            left_shift = left + (m_info.cellY / 2.)
                            top_shift = top - (m_info.cellY / 2.)

                            # left_shift = left + ((m_info.cellY * skip) - (m_info.cellY / 2.))
                            # top_shift = top - ((m_info.cellY * skip) - (m_info.cellY / 2.))

                            # create a point at left, top
                            vector_tools.add_point(left_shift, top_shift, mpc, class_id, val)

                        left += (m_info.cellY * skip)

                    top -= (m_info.cellY * skip)

            ttl_blks_ct += 1

    band, m_info.datasource = None, None

    pt_geom.Destroy()
    mpc.shp_src.Destroy()

    if os.path.isfile(in_rst):
        try:
            os.remove(in_rst)
        except:
            pass

    
def poly2points(poly, out_shp, targ_img, class_id='Id', cell_size=None, field_type='int', use_extent=True, \
                no_data_val=-1.):

    """
    Converts polygons to points.

    Args:    
        poly (str): Path, name, and extension of polygon vector to compute.
        out_shp (str): Path, name, and extension of output vector points.
        targ_img (str): Path, name, and extension of image to align to.
        class_id (Optional[str]): The field id in ``poly`` to get class values from. Default is 'Id'.
        cell_size (Optional[float]): The cell size for point spacing. Default is None, or cell size of ``targ_img``.

    Examples:
        >>> from mappy.sample import poly2points
        >>> poly2points('C:/someDir/somePoly.shp', 'C:/someDir/somePts.shp')
    
        Command line usage
        ------------------
        .. mappy\sample\poly2points.py -i C:\someDir\somePoly.shp -o C:\someOutDir\somePts.shp

    Returns:
        None, writes to ``out_shp``.
    """

    d_name, f_name = os.path.split(out_shp)
    f_base, f_ext = os.path.splitext(f_name)
        
    out_rst = '%s/%s.tif' % (d_name, f_base)

    if os.path.isfile(out_rst):

        try:
            os.remove(out_rst)
        except:
            sys.exit('ERROR!! Could not delete the output raster.')
            
    m_info = raster_tools.rinfo(targ_img)
    
    m_info.storage = 'byte'
    
    if not cell_size:
        cell_size = m_info.cellY

    # get vector info 
    vct_info = vector_tools.vinfo(poly)
    
    # Check if the shapefile is UTM North or South. gdal_rasterize has trouble with UTM South
    # if 'S' in vct_info.proj.GetAttrValue('PROJCS')[-1]: # GetUTMZone()
    #     sys.exit('\nERROR!! The shapefile should be projected to UTM North (even for the Southern Hemisphere).\n')

    if use_extent:
        com = 'gdal_rasterize -init %d -a %s -te %f %f %f %f -tr %f %f -ot Float32 %s %s' % \
              (no_data_val, class_id, m_info.left, m_info.bottom, m_info.right, m_info.top, cell_size, cell_size, \
               poly, out_rst)
    else:
        com = 'gdal_rasterize -init %d -a %s -tr %f %f -ot Float32 %s %s' % \
              (no_data_val, class_id, cell_size, cell_size, poly, out_rst)

    print '\nRasterizing %s ...\n' % f_name

    subprocess.call(com, shell=True)

    _add_points_from_raster(out_shp, class_id, field_type, out_rst, proj=vct_info.proj)


def _usage():
    
    sys.exit("""\
    poly2points.py ...
    [-p <Input polygon (str)>]
    [-o <Output points (str)>]
    [-tr <Target raster--for alignment (str)>]
    [-fId <Field id :: Default=Id>]
    [-c <Cell size (float) :: Default=-tr>]
    [-fldt <Field type (int or float) (str) :: Default=int>]
    [-batch <Batch directory (str) :: Default is None>]
    """)


def main():

    argv = sys.argv
        
    if argv is None:
        sys.exit(0)

    poly = None
    out_shp = None
    targ_img = None
    class_id = 'Id'
    field_type = 'int'
    cell_size = None
    batch_dir = None
        
    # Parse command line arguments.
    i = 1
    while i < len(argv):
        arg = argv[i]

        if arg == '-p':
            i += 1
            poly = argv[i]
        
        elif arg == '-o':
            i += 1
            out_shp = argv[i]
            
        elif arg == '-tr':
            i += 1
            targ_img = argv[i]

        elif arg == '-fId':
            i += 1
            class_id = argv[i]

        elif arg == '-c':
            i += 1
            cell_size = float(argv[i])

        elif arg == '-fldt':
            i += 1
            field_type = argv[i]
            
        elif arg == '-batch':
            i += 1
            batch_dir = argv[i]

        elif arg == '-help':
            _usage()
            sys.exit(1)                    
            
        elif arg[:1] == ':':
            print('Unrecognized command option: %s' % arg)
            _usage()
            sys.exit(1)            

        i += 1

    if not batch_dir:
        if not poly:
            sys.exit('\nERROR!! The polygon (-p) must be specified.')      
        else:
            if not os.path.isfile(poly):
                sys.exit('\nERROR!! %s does not exist.' % poly)   

    if not batch_dir:
        if not out_shp:
            sys.exit('\nERROR!! The output points (-o) file must be specified.')           
                
    if not targ_img:
        sys.exit('\nERROR!! The target image (-tr) must be specified.')
    else:
        if not os.path.isfile(targ_img):
            sys.exit('\nERROR!! %s does not exist.' % targ_img)    
        
    print('\nStart date & time --- (%s)\n' % time.asctime(time.localtime(time.time())))

    startB = time.time()            
        
    if batch_dir:
    
        if not os.path.isdir(batch_dir):
            sys.exit('\nERROR!! %s does not exist.' % batch_dir)
            
        shp_list = [d for d in os.listdir(batch_dir) if d[-4:].lower() in ['.shp']]
            
        for shp in shp_list:
        
            f_base, f_ext = os.path.splitext(shp)
        
            poly = '%s/%s' % (batch_dir, shp)
            out_shp = '%s/%s_pts.shp' % (batch_dir, f_base)
        
            poly2points(poly, out_shp, targ_img, class_id, cell_size, field_type=field_type)
    else:
    
        poly2points(poly, out_shp, targ_img, class_id, cell_size, field_type=field_type)

    print('\nEnd data & time -- (%s)\nTotal processing time -- (%.2gs)\n' % (time.asctime(time.localtime(time.time())), (time.time()-startB)))                

if __name__ == '__main__':
    main()