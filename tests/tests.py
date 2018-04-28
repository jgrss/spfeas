from __future__ import print_function

import os
import unittest


def _import():
    """Test the main import"""
    import spfeas
    return True


def _mean():

    """Test the mean feature"""

    import spfeas

    path = os.path.dirname(os.path.realpath(__file__))
    data_path = path.replace('tests', 'data')
    image = os.path.join(data_path, 'test_image.tif')
    out_dir = os.path.join(data_path, 'features')

    spfeas.spatial_features(image,
                            out_dir,
                            band_positions=[1],
                            block=4,
                            scales=[8],
                            triggers=['mean'])

    os.remove(out_dir)


class TestUM(unittest.TestCase):

    def setUp(self):
        pass

    def test_import(self):
        self.assertTrue(_import())

    def test_mean(self):
        self.assertTrue(_mean())


if __name__ == '__main__':
    unittest.main()
