import os


test_image = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_image.tif')

training_1m_01 = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'shp', 'training_1m_01.tif')
training_1m_02 = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'shp', 'training_1m_02.tif')
training_4m_01 = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'shp', 'training_4m_01.tif')
training_4m_02 = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'shp', 'training_4m_02.tif')

features_4m_01 = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              '_features',
                              'test_image__BD1_BK4_SC8_TRmean.vrt')
