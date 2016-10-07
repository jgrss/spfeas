import os


def get_mappy_path():
    return os.path.dirname(os.path.realpath(__file__))

gdal_path = 'gdalwin64-11'

starfm_path = '{}/helpers/StarFM/source'.format(get_mappy_path())
