import os


test_image = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_image.tif')

training_01_2m = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'shp', 'grid01_2m.tif')
training_02_2m = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'shp', 'grid02_2m.tif')
training_03_2m = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'shp', 'grid03_2m.tif')
training_04_2m = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'shp', 'grid04_2m.tif')
training_05_2m = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'shp', 'grid05_2m.tif')

training_01_4m = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'shp', 'grid01_4m.tif')
training_02_4m = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'shp', 'grid02_4m.tif')
training_03_4m = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'shp', 'grid03_4m.tif')
training_04_4m = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'shp', 'grid04_4m.tif')
training_05_4m = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'shp', 'grid05_4m.tif')

features_01__BD1_BK4_SC8 = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                        '_features',
                                        'test_image__BD1_BK4_SC8_TRmean.vrt')

# features_02__BD1_BK4_SC2_8_16_32 = os.path.join(os.path.dirname(os.path.realpath(__file__)),
#                                                 '_features',
#                                                 'test_image__BD1_BK4_SC8-16-32_TRdmp-grad-hog-mean-pantex-orb-saliency-rbvi.vrt')

features_02__BD1_BK2_SC8_16 = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                           '_features',
                                           'test_image__BD1_BK2_SC8-16_TRdmp-hog-mean-saliency-rbvi.vrt')
