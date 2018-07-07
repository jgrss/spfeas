from __future__ import division
from builtins import int

import itertools
from joblib import Parallel, delayed

from .sphelpers.gabor_filter_bank import prep_gabor
from .sphelpers import lsr
from .sphelpers._stats import fill_labels, fill_key_points

from mpglue.stats._rolling_stats import rolling_stats

try:
    from skimage.exposure import rescale_intensity
    from skimage.feature import hog as HOG
    from skimage.feature import local_binary_pattern as LBP
    from skimage.feature import greycomatrix, greycoprops
    from skimage.exposure import histogram
    from skimage.color import rgb2rgbcie
    from skimage.segmentation import felzenszwalb
    from skimage.measure import regionprops
    from skimage.morphology import reconstruction
except ImportError:
    raise ImportError('Scikits-image must be installed')

try:
    import numpy as np
except ImportError:
    raise ImportError('NumPy must be installed')
    
try:
    import cv2
except ImportError:
    raise ImportError('OpenCV must be installed')

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError('Matplotlib must be installed')


def get_kernels():

    # Robert's
    roberts_filter_y_1 = np.array([[0, -1],
                                   [1, 0]], dtype='float32')

    roberts_filter_x_1 = np.array([[0, 1],
                                   [-1, 0]], dtype='float32')

    roberts_filter_y_2 = np.array([[-1, 0],
                                   [0, 1]], dtype='float32')

    roberts_filter_x_2 = np.array([[1, 0],
                                   [0, -1]], dtype='float32')

    roberts_filter_y_3 = np.array([[-1, -1],
                                   [1, 1]], dtype='float32')

    roberts_filter_x_3 = np.array([[1, 1],
                                   [-1, -1]], dtype='float32')

    roberts_filter_y_4 = np.array([[-1, 1],
                                   [-1, 1]], dtype='float32')

    roberts_filter_x_4 = np.array([[1, -1],
                                   [1, -1]], dtype='float32')

    return [[roberts_filter_y_1, roberts_filter_x_1], [roberts_filter_y_2, roberts_filter_x_2],
            [roberts_filter_y_3, roberts_filter_x_3], [roberts_filter_y_4, roberts_filter_x_4]]


def get_mag_avg(img):

    img = np.sqrt(img)

    kernels = get_kernels()

    mag = np.zeros(img.shape, dtype='float32')

    for kernel_filter in kernels:

        gx = cv2.filter2D(np.float32(img), cv2.CV_32F, kernel_filter[1], borderType=cv2.BORDER_REFLECT)
        gy = cv2.filter2D(np.float32(img), cv2.CV_32F, kernel_filter[0], borderType=cv2.BORDER_REFLECT)

        mag += cv2.magnitude(gx, gy)

    mag /= len(kernels)

    return np.uint8(mag)


def get_mag_ang(img):

    """
    Gets image gradient (magnitude) and orientation (angle)

    Args:
        img

    Returns:
        Gradient, orientation
    """

    img = np.sqrt(img)

    gx = cv2.Sobel(np.float32(img), cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(np.float32(img), cv2.CV_32F, 0, 1)

    mag, ang = cv2.cartToPolar(gx, gy)

    return mag, ang, gx, gy


def grad_mag(ch_bd):

    # normalize
    mu_ = ch_bd.mean()
    std_ = ch_bd.std()
    ch_bd = np.divide(np.subtract(ch_bd, mu_), std_)

    ch_bd[np.isnan(ch_bd)] = 0
    ch_bd += abs(ch_bd.min())

    # compute gradient orientation and magnitude
    return get_mag_ang(ch_bd)


def azimuthal_avg(image, center=None):

    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is
             None, which then uses the center of the image (including
             fracitonal pixels).

    """

    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max() - x.min()) / 2.0, (x.max() - x.min()) / 2.0])

    #   : get hypotenuse
    r = np.hypot(np.subtract(x, center[0]), np.subtract(y, center[1]))

    #   : get sorted radii indices
    ind = np.argsort(r.flat)
    rSorted = r.flat[ind]

    #   : get image values from index positions
    iSorted = image.flat[ind]

    #   : Get the integer part of the radii (bin size = 1)
    rInt = rSorted.astype(int)

    #   : Find all pixels that fall within each radial bin.
    deltar = np.subtract(rInt[1:], rInt[:-1])  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = np.subtract(rind[1:], rind[:-1])        # number of radius bin

    #   : Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(iSorted, dtype=float)
    tbin = np.subtract(csim[rind[1:]], csim[rind[:-1]])

    radialProf = np.divide(tbin, nr)

    return radialProf


def fourier_transform(ch_bd):

    dft = cv2.dft(np.float32(ch_bd), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # get the Power Spectrum
    magnitude_spectrum = 20. * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    psd1D = azimuthal_avg(magnitude_spectrum)

    return list(cv2.meanStdDev(psd1D))


def feature_fourier(chBd, blk, scs, end_scale):

    rows, cols = chBd.shape
    scales_half = int(end_scale / 2.0)
    scales_blk = end_scale - blk
    out_len = 0
    pix_ctr = 0

    for i in range(0, rows-scales_blk, blk):
        for j in range(0, cols-scales_blk, blk):
            for k in scs:
                out_len += 2

    # set the output list
    out_list = np.zeros(out_len, dtype='float32')

    for i in range(0, rows-scales_blk, blk):

        for j in range(0, cols-scales_blk, blk):

            for k in scs:

                k_half = int(k / 2.0)

                ch_bd = chBd[i+scales_half-k_half:i+scales_half-k_half+k,
                             j+scales_half-k_half:j+scales_half-k_half+k]

                # get the Fourier Transform
                dft = cv2.dft(np.float32(ch_bd), flags=cv2.DFT_COMPLEX_OUTPUT)
                dft_shift = np.fft.fftshift(dft)

                # get the Power Spectrum
                magnitude_spectrum = 20.0 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

                psd1D = azimuthal_avg(magnitude_spectrum)

                sts = list(cv2.meanStdDev(psd1D))

                # plt.subplot(121)
                # plt.imshow(ch_bd, cmap='gray')
                # plt.subplot(122)
                # plt.imshow(magnitude_spectrum, interpolation='nearest')
                # plt.show()
                # print psd1D
                # sys.exit()

                for st in sts:

                    if np.isnan(st[0][0]):
                        out_list[pix_ctr] = 0.0
                    else:
                        out_list[pix_ctr] = st[0][0]

                    pix_ctr += 1

    out_list[np.isnan(out_list) | np.isinf(out_list)] = 0.0

    return out_list


def call_lsr(edoim_s, edmim_s, dx_s, dy_s, scs, end_scale):

    scale_stats = list()
    scales_half = int(end_scale / 2)

    for k in scs:

        if k != scs[-1]:

            k_half = int(k / 2)

            ifst = scales_half - k_half
            isnd = scales_half - k_half + k
        else:
            ifst = None
            isnd = None

        scale_stats += list(lsr.feature_lsr(edoim_s[ifst:isnd, ifst:isnd],
                                            edmim_s[ifst:isnd, ifst:isnd],
                                            dx_s[ifst:isnd, ifst:isnd],
                                            dy_s[ifst:isnd, ifst:isnd]))

    return scale_stats


def feature_lsr(ch_bd, blk, scs, end_scale):

    rows, cols = ch_bd.shape
    out_list = list()

    edge_mag, edge_ori, deriv_x, deriv_y = grad_mag(ch_bd)
    
    for i in range(0, rows-(end_scale-blk), blk):

        out_list += list(itertools.chain.from_iterable(Parallel(n_jobs=-1,
                                                                max_nbytes=None)(delayed(call_lsr)(edge_ori[i:i+end_scale,
                                                                                                            j:j+end_scale],
                                                                                                   edge_mag[i:i+end_scale,
                                                                                                            j:j+end_scale],
                                                                                                   deriv_x[i:i+end_scale,
                                                                                                           j:j+end_scale],
                                                                                                   deriv_y[i:i+end_scale,
                                                                                                           j:j+end_scale],
                                                                                                   scs, end_scale)
                                                                                 for j in range(0, cols-(end_scale-blk), blk))))

        # for j in range(0, cols-(end_scale-blk), blk):
        #
        #
        #     sts = Parallel(n_jobs=len(scs),
        #                    max_nbytes=None)(delayed(call_lsr)(edge_ori[i+scales_half-int(k/2):i+scales_half-int(k/2)+k,
        #                                                                j+scales_half-int(k/2):j+scales_half-int(k/2)+k],
        #                                                       edge_mag[i+scales_half-int(k/2):i+scales_half-int(k/2)+k,
        #                                                                j+scales_half-int(k/2):j+scales_half-int(k/2)+k],
        #                                                       deriv_x[i+scales_half-int(k/2):i+scales_half-int(k/2)+k,
        #                                                               j+scales_half-int(k/2):j+scales_half-int(k/2)+k],
        #                                                       deriv_y[i+scales_half-int(k/2):i+scales_half-int(k/2)+k,
        #                                                               j+scales_half-int(k/2):j+scales_half-int(k/2)+k])
        #                                     for k in scs)
        #
        #     for st in sts:
        #         for st_ in st:
        #             out_list.append(st_)
            
    return out_list


def scale_rgb(layers, min_max, lidx):

    layers_c = np.empty(layers.shape, dtype='float32')

    # Rescale and blur.
    for li in range(0, 3):

        layer = layers[li]

        layer = np.float32(rescale_intensity(layer,
                                             in_range=(min_max[li][0],
                                                       min_max[li][1]),
                                             out_range=(0, 1)))

        layers_c[lidx[li]] = rescale_intensity(cv2.GaussianBlur(layer,
                                                                ksize=(3, 3),
                                                                sigmaX=3),
                                               in_range=(0, 1),
                                               out_range=(-1, 1))

    return layers_c


def get_saliency_tile_mean(im, min_max=None, vis_order=None):

    """
    Gets the mean lab averages per tile
    """

    if vis_order == 'bgr':
        lidx = [2, 1, 0]
    else:
        lidx = [0, 1, 2]

    layers = scale_rgb(im[0], min_max, lidx)

    # Transpose the image to RGB
    layers = layers.transpose(1, 2, 0)

    # Perform RGB to CIE Lab color space conversion
    layers = rgb2rgbcie(layers)

    # Compute Lab average values
    lm = layers[:, :, 0].mean(axis=0).mean()
    am = layers[:, :, 1].mean(axis=0).mean()
    bm = layers[:, :, 2].mean(axis=0).mean()

    lab_means = (lm, am, bm)

    return None, lab_means


def saliency(i_info, parameter_object, i_sect, j_sect, n_rows, n_cols):

    """
    References:
        Federico Perazzi, Philipp Krahenbul, Yael Pritch, Alexander Hornung. Saliency Filters. (2012).
            Contrast Based Filtering for Salient Region Detection. IEEE CVPR, Providence, Rhode Island, USA, June 16-21.

            https://graphics.ethz.ch/~perazzif/saliency_filters/

        Ming-Ming Cheng, Niloy J. Mitra, Xiaolei Huang, Philip H. S. Torr, Shi-Min Hu. (2015).
            Global Contrast based Salient Region detection. IEEE TPAMI.
    """

    # min_max = sputilities.get_layer_min_max(i_info)
    min_max = [(parameter_object.image_min, parameter_object.image_max)] * 3

    if parameter_object.vis_order == 'bgr':
        lidx = [2, 1, 0]
    else:
        lidx = [0, 1, 2]

    # Read the section.
    layers = i_info.read(bands2open=[1, 2, 3],
                         i=i_sect,
                         j=j_sect,
                         rows=n_rows,
                         cols=n_cols,
                         d_type='float32')

    layers = scale_rgb(layers, min_max, lidx)

    # Transpose the image to RGB
    layers = layers.transpose(1, 2, 0)

    # Perform RGB to CIE Lab color space conversion
    layers = rgb2rgbcie(layers)

    # Compute Lab average values
    # lm = layers[:, :, 0].mean(axis=0).mean()
    # am = layers[:, :, 1].mean(axis=0).mean()
    # bm = layers[:, :, 2].mean(axis=0).mean()
    lm = parameter_object.lab_means[0]
    am = parameter_object.lab_means[1]
    bm = parameter_object.lab_means[2]

    return np.uint8(rescale_intensity((layers[:, :, 0] - lm)**2. +
                                      (layers[:, :, 1] - am)**2. +
                                      (layers[:, :, 2] - bm)**2.,
                                      in_range=(-1, 1),
                                      out_range=(0, 255)))


def segment_image(im, parameter_object):

    dims, rows, cols = im.shape

    image2segment = np.dstack((rescale_intensity(im[0],
                                                 in_range=(parameter_object.image_min,
                                                           parameter_object.image_max),
                                                 out_range=(0, 255)),
                               rescale_intensity(im[1],
                                                 in_range=(parameter_object.image_min,
                                                           parameter_object.image_max),
                                                 out_range=(0, 255)),
                               rescale_intensity(im[2],
                                                 in_range=(parameter_object.image_min,
                                                           parameter_object.image_max),
                                                 out_range=(0, 255))))

    felzer = felzenszwalb(np.uint8(image2segment),
                          scale=50,
                          sigma=.01,
                          min_size=5,
                          multichannel=True).reshape(rows, cols)

    props = regionprops(felzer)
    props = np.array([p.area for p in props], dtype='uint64')

    return fill_labels(np.uint64(felzer), props)


def get_slopes(xv, yv):

    """
    Args:
        xv (2d array): Samples X M
        yv (1d array): M
    """

    return ((xv*yv).mean(axis=1) - xv.mean() * yv.mean(axis=1)) / ((xv**2).mean() - (xv.mean())**2)


def get_dmp(bd, image_min, image_max, ses=None):

    """
    Calculates the Differential Morphological Profile

    Args:
        bd (2d array)
        image_min (int or float)
        image_max (int or float)
        ses (Optional[list]): The structuring elements.

    Returns:

    """

    if not ses:
        ses = [3, 5, 7, 9, 11, 13, 15]

    if bd.dtype != 'uint8':

        bd = np.uint8(rescale_intensity(bd,
                                        in_range=(image_min,
                                                  image_max),
                                        out_range=(0, 255)))

    section_rows, section_cols = bd.shape

    dmp_counter = 0

    marker = bd.copy()
    previous = bd.copy()

    # The DMP holder
    # closings --> 1st len(ses) bands
    # openings --> last len(ses) bands
    dims = len(ses) * 2

    dmp_array = np.empty((dims, section_rows, section_cols), dtype='uint8')

    # Morphological opening
    for se_size in ses:

        se = cv2.getStructuringElement(cv2.MORPH_RECT, (se_size, se_size))

        marker = cv2.erode(marker, se, iterations=1)

        current = reconstruction(marker, bd, method='dilation', selem=se)

        dmp_array[dmp_counter] = previous - current

        previous = current.copy()

        dmp_counter += 1

    marker = bd.copy()
    previous = bd.copy()

    # Morphological closing
    for se_size in ses:

        se = cv2.getStructuringElement(cv2.MORPH_RECT, (se_size, se_size))

        marker = cv2.dilate(marker, se, iterations=1)

        current = reconstruction(marker, bd, method='erosion', selem=se)

        dmp_array[dmp_counter] = current - previous

        previous = current.copy()

        dmp_counter += 1

    return np.uint8(np.gradient(dmp_array, axis=0).mean(axis=0))

    # Reshape to [dims x samples].
    # X_min, X_max = rolling_stats(dmp_array.transpose(1, 2, 0).reshape(section_rows*section_cols, dims).T,
    #                              stat='slope',
    #                              window_size=dims)

    # return np.uint8(X_max.reshape(section_rows, section_cols))

    # Reshape to [samples X dimensions].
    # dmp_array = dmp_array.reshape(dims,
    #                               section_rows,
    #                               section_cols).transpose(1, 2, 0).reshape(section_rows*section_cols,
    #                                                                        dims)

    # Get the derivative of the
    #   morphological openings
    #   and closings.
    # return np.uint8(np.gradient(dmp_array,
    #                             axis=1).T.reshape(dims,
    #                                               section_rows,
    #                                               section_cols))

    # Reshape to [samples X dimensions] and
    #   get the slope for each sample.
    # return get_slopes(np.arange(dims, dtype='float32'),
    #                   dmp_array.reshape(dims,
    #                                     section_rows,
    #                                     section_cols).transpose(1, 2, 0).reshape(section_rows*section_cols,
    #                                                                              dims)).reshape(section_rows,
    #                                                                                             section_cols)


def get_orb_keypoints(bd, image_min, image_max):

    """
    Computes the ORB key points

    Args:
        bd (2d array)
        image_min (int or float)
        image_max (int or float)
    """

    # We want odd patch sizes.
    # if parameter_object.scales[-1] % 2 == 0:
    #     patch_size = parameter_object.scales[-1] - 1

    if bd.dtype != 'uint8':

        bd = np.uint8(rescale_intensity(bd,
                                        in_range=(image_min,
                                                  image_max),
                                        out_range=(0, 255)))

    patch_size = 31
    patch_size_d = patch_size * 3

    # Initiate ORB detector
    orb = cv2.ORB_create(nfeatures=int(.25*(bd.shape[0]*bd.shape[1])),
                         edgeThreshold=patch_size,
                         scaleFactor=1.2,
                         nlevels=8,
                         patchSize=patch_size,
                         WTA_K=4,
                         scoreType=cv2.ORB_FAST_SCORE)

    # Add padding because ORB ignores edges.
    bd = cv2.copyMakeBorder(bd, patch_size_d, patch_size_d, patch_size_d, patch_size_d, cv2.BORDER_REFLECT)

    # Compute ORB keypoints
    key_points = orb.detectAndCompute(bd, None)[0]

    # img = cv2.drawKeypoints(np.uint8(ch_bd), key_points, np.uint8(ch_bd).copy())

    return fill_key_points(np.float32(bd), key_points)[patch_size_d:-patch_size_d, patch_size_d:-patch_size_d]


def convolve_gabor(bd, image_min, image_max, scales):

    """
    Convolves an image with a series of Gabor kernels

    Args:
        bd (2d array)
        image_min (int or float)
        image_max (int or float)
        scales (1d array like)
    """

    if bd.dtype != 'uint8':

        bd = np.uint8(rescale_intensity(bd,
                                        in_range=(image_min,
                                                  image_max),
                                        out_range=(0, 255)))

    # Each set of Gabor kernels
    #   has 8 orientations.
    out_block = np.empty((8*len(scales),
                          bd.shape[0],
                          bd.shape[1]), dtype='uint8')

    ki = 0

    for scale in scales:

        # Check for even or
        #   odd scale size.
        if scale % 2 == 0:
            ssub = 1
        else:
            ssub = 0

        gabor_kernels = prep_gabor(kernel_size=(scale-ssub, scale-ssub))

        for kernel in gabor_kernels:

            out_block[ki] = cv2.filter2D(bd, cv2.CV_8U, kernel)

            ki += 1

    return out_block
