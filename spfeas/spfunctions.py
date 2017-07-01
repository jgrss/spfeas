import sys
import itertools
from joblib import Parallel, delayed

from .sphelpers import lsr

from skimage.exposure import rescale_intensity

try:
    from skimage.feature import hog as HOG
    from skimage.feature import local_binary_pattern as LBP
    from skimage.feature import greycomatrix, greycoprops
    from skimage.exposure import histogram
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

    return mag


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
    scales_half = end_scale / 2
    scales_blk = end_scale - blk
    out_len = 0
    pix_ctr = 0

    for i in xrange(0, rows-scales_blk, blk):
        for j in xrange(0, cols-scales_blk, blk):
            for k in scs:
                out_len += 2

    # set the output list
    out_list = np.zeros(out_len).astype(np.float32)

    for i in xrange(0, rows-scales_blk, blk):
        for j in xrange(0, cols-scales_blk, blk):
            for k in scs:

                ch_bd = chBd[i+scales_half-(k/2):i+scales_half-(k/2)+k, j+scales_half-(k/2):j+scales_half-(k/2)+k]

                # get the Fourier Transform
                dft = cv2.dft(np.float32(ch_bd), flags=cv2.DFT_COMPLEX_OUTPUT)
                dft_shift = np.fft.fftshift(dft)

                # get the Power Spectrum
                magnitude_spectrum = 20. * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

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
                        out_list[pix_ctr] = 0.
                    else:
                        out_list[pix_ctr] = st[0][0]

                    pix_ctr += 1

    out_list[np.isnan(out_list) | np.isinf(out_list)] = 0.

    return out_list


def call_lsr(edoim_s, edmim_s, dx_s, dy_s, scs, scales_half):

    scale_stats = []

    for k in scs:

        if k != scs[-1]:

            k_half = int(k / 2)

            ifst = scales_half - k_half
            isnd = scales_half - k_half + k

            edoim_s = edoim_s[ifst:isnd, ifst:isnd]
            edmim_s = edmim_s[ifst:isnd, ifst:isnd]
            dx_s = dx_s[ifst:isnd, ifst:isnd]
            dy_s = dy_s[ifst:isnd, ifst:isnd]

        scale_stats += list(lsr.feature_lsr(edoim_s, edmim_s, dx_s, dy_s))

    return scale_stats


def feature_lsr(ch_bd, blk, scs, end_scale):

    rows, cols = ch_bd.shape
    out_list = []
    scales_half = int(end_scale / 2)

    edge_mag, edge_ori, deriv_x, deriv_y = grad_mag(ch_bd)
    
    for i in xrange(0, rows-(end_scale-blk), blk):

        ifst_ = scales_half - scales_half
        isnd_ = scales_half - scales_half + end_scale

        out_list += list(itertools.chain.from_iterable(Parallel(n_jobs=64,
                                                                max_nbytes=None)(delayed(call_lsr)(edge_ori[i+ifst_:i+isnd_,
                                                                                                            j+ifst_:j+isnd_],
                                                                                                   edge_mag[i+ifst_:i+isnd_,
                                                                                                            j+ifst_:j+isnd_],
                                                                                                   deriv_x[i+ifst_:i+isnd_,
                                                                                                           j+ifst_:j+isnd_],
                                                                                                   deriv_y[i+ifst_:i+isnd_,
                                                                                                           j+ifst_:j+isnd_],
                                                                                                   scs, scales_half)
                                                                                 for j in xrange(0, cols-(end_scale-blk), blk))))

        # for j in xrange(0, cols-(end_scale-blk), blk):
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


# def surfFunc(surfArr, kPts, j, i, k, scs):
#
#     """
#     Get the moments
#     """
#
#     start_y = i+(scs[-1]/2)-(k/2)
#     start_x = j+(scs[-1]/2)-(k/2)
#
#     if surfArr.max() == 0:
#         return 0.	# 21 length vector to match pyramid histogram length (1+4+16)
#     else:
#         if kPts:
#             # return desc_stats[m](pyramid_hist_sift(surfArr, kPts, start_x, start_y).sp_hist)
#             return get_moments(pyramid_hist_sift(surfArr, kPts, start_x, start_y).sp_hist)
#         else:
#             return [0., 0., 0., 0.]


# def feaSURF(chBd, blk, scs):
#
#     rows, cols	= chBd.shape
#
#     # compute SURF features
#     kPts, descrip = cv2.SURF(50).detectAndCompute(chBd, None)
#
#     for i in xrange(0, rows-scales_blk, blk):
#         for j in xrange(0, cols-scales_blk, blk):
#             for k in scs:
#
#                 sts = surfFunc(chBd[i+scales_half-(k/2):i+scales_half-(k/2)+k, \
#                                j+scales_half-(k/2):j+scales_half-(k/2)+k], kPts, j, i, k, scs)
#
#
#     return [ surfFunc(chBd[i+(scs[-1]/2)-(k/2):i+(scs[-1]/2)-(k/2)+k, j+(scs[-1]/2)-(k/2):j+(scs[-1]/2)-(k/2)+k], kPts, j, i, k, scs, m) \
#             for k in scs for m in xrange(0, 4) for (i,j) in itertools.product(xrange(0, rows-(scs[-1]-blk), blk), xrange(0, cols-(scs[-1]-blk), blk)) ]
