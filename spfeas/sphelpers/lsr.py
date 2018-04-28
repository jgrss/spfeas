"""
@author: Jordan Graesser
Adapted from Anil Cheriyadat's MATLAB code
"""

from __future__ import absolute_import, division
from builtins import int

from . import _lsr

try:
    from skimage.measure import label as lab_img
    from skimage.morphology import remove_small_objects, skeletonize
except ImportError:
    raise ImportError('Scikit-image must be installed')

# try:
#     import pyfftw
# except ImportError:
#     raise ImportError('PyFFTW must be installed')

try:
    # from scipy.ndimage.measurements import label as lab_img
    from scipy.fftpack import fftshift, fft
    # from skimage.segmentation import find_boundaries, mark_boundaries
except ImportError:
    raise ImportError('SciPy.ndimage and/or .fftpack did not load')    
   
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


def get_edge_pixels(ori_img, mag_img, mag_thresh):

    """
    Ignore gradients with small magnitude
    """
    
    edge_pixs = np.where(mag_img.ravel() > mag_thresh)
    
    ori_img[(ori_img < 0)] += 360.
    
    return ori_img.ravel()[edge_pixs], edge_pixs


class BinQ(object):
    
    def __init__(self, data, edgePixs, lsr_thresh, dx, dy, rows, cols):

        self.bin(data, edgePixs, lsr_thresh, dx, dy, rows, cols)
    
    def bin(self, data, edgePixs, lsr_thresh, dx, dy, rows, cols):
    
        # Here we divide them into bins with bin boundaries as 
        # ... 0,45,90,135,180,225,270,315,0
        binidx = np.searchsorted(range(0, 360+45, 45), data)

        lsfim1 = np.zeros((2, rows, cols), dtype='float32')
        lsfarr = np.zeros((1, 6), dtype='float32')
        
        for k in range(1, len(range(0, 360+45, 45))):  # the range is for 45, ..n, 360, by 45

            curr_bin = np.where(binidx == k)

            if len(curr_bin[0]):
                
                edge_img = np.zeros(rows*cols, dtype='float32')
                edge_img[edgePixs[0][curr_bin[0]]] = 1

                # plt.subplot(121)
                # plt.imshow(edge_img.reshape(rows, cols))
                # plt.subplot(122)
                # plt.imshow(np.uint8(skeletonize(edge_img.copy().reshape(rows, cols))))
                # plt.show()
                # sys.exit()

                # Extract LSR by grouping pixels with similar orientations
                lsfim1, lsfarr = self.generate_regions(edge_img.reshape(rows, cols),
                                                       lsr_thresh, lsfim1, lsfarr, dx, dy, rows, cols)

        # Here we divide them into bins with bin boundaries as    
        # ... 22.5,67.5,112.5,157.5,202.5,247.5,292.5,337.5,22.5    
        binidx = np.searchsorted(list(np.linspace(22.5, 360, num=np.floor((360-22.5)/45.))), data)
        
        lsfim2 = np.zeros((2, rows, cols), dtype='float32')

        # the range is for 22.5, ..n, 337.5, by 45
        for k in range(1, len(list(np.linspace(22.5, 360, num=np.floor((360-22.5)/45.))))):
        
            curr_bin = np.where(binidx == k)
            
            if len(curr_bin[0]):
                
                edge_img = np.zeros(rows*cols, dtype='float32')
                edge_img[edgePixs[0][curr_bin[0]]] = 1

                # Extract LSR by grouping pixels with similar orientations
                lsfim2, lsfarr = self.generate_regions(edge_img.reshape(rows, cols), lsr_thresh, lsfim2,
                                                       lsfarr, dx, dy, rows, cols)

        edge_img = np.zeros(rows*cols, dtype='float32')
                
        binidx = np.searchsorted([337.5, 360], data)

        edge_img[edgePixs[0][np.where(binidx == 1)[0]]] = 1
        
        binidx = np.searchsorted([0., 22.5], data)
                
        edge_img[edgePixs[0][np.where(binidx == 1)[0]]] = 1
        
        edge_img = edge_img.reshape(rows, cols)
        
        # Extract LSR by grouping pixels with similar orientations
        lsfim2, lsfarr = self.generate_regions(edge_img, lsr_thresh, lsfim2, lsfarr, dx, dy, rows, cols)

        lsfarr = lsfarr[1:, :]	    # first row was a dummy

        if lsfarr.shape[0] > 0:

            lsfarr = _lsr.get_features(lsfarr, lsfim1, lsfim2, rows, cols)

            self.lsfarr = lsfarr[(lsfarr[:, 5] > 0)][:, :5]

            if len(self.lsfarr) == 0:
                self.lsfarr = np.zeros((1, 5), dtype='float32')

        else:
            self.lsfarr = np.zeros((1, 5), dtype='float32')
        
    def generate_regions(self, edge_img, lsr_thresh, lsfim, lsfarr, dx, dy, rows, cols):

        # Create independent edges
        __, edge_img = cv2.threshold(np.uint8(skeletonize(np.uint8(edge_img))), 0, 1, cv2.THRESH_BINARY_INV)
        edge_img = cv2.distanceTransform(edge_img, cv2.DIST_L1, 3)
        edge_img = np.where(edge_img == 2, 1, 0)

        # Label the edges
        ori, num_objs = lab_img(edge_img, connectivity=1, return_num=True)

        # plt.imshow(ori)
        # plt.show()
        # sys.exit()

        if lsfim.max() == 0:
            lsfima = np.zeros((rows, cols), dtype='float32')
            lsfimb = np.zeros((rows, cols), dtype='float32')
        else:
            lsfima = lsfim[0]
            lsfimb = lsfim[1]
        
        cnt = lsfarr.shape[0] - 1

        # TEST
        # ax = plt.figure().add_subplot(111)
        # ax.imshow(ori, interpolation='nearest')
        # TEST

        for n in range(1, num_objs):
        
            bidx = np.where(ori == n)

            if bidx[0] is None:
                continue

            y = bidx[0]
            x = bidx[1]

            # threshold for line length
            if len(y) <= lsr_thresh:
                continue

            # TEST
            # st = list(x).index(x.min())
            # ed = list(x).index(x.max())
            # ax.plot((x[st], x[ed]), (y[st], y[ed]))
            # continue
            # TEST

            # _fft = pyfftw.builders.fft(x*y,
            #                            n=len(x),
            #                            threads=8,
            #                            auto_align_input=True,
            #                            planner_effort='FFTW_ESTIMATE')

            # a = fftshift(_fft())

            a = fftshift(fft(x*y, len(x)))
            a = np.divide(a, len(x))

            idx = int(np.floor(len(x) / 2) + 1)

            lmx = a[idx].real
            lmy = a[idx].imag								

            llen = 2 * (np.abs(a[idx+1]) + np.abs(a[idx-1]))
            lorn = (np.arctan2(a[idx+1].imag, a[idx+1].real) + np.arctan2(a[idx-1].imag, a[idx-1].real)) / 2.
            lcon = np.max(np.maximum(abs(dx[bidx]), abs(dy[bidx])))

            cnt += 1
            
            lsfima[bidx] = cnt
            lsfimb[bidx] = llen			
            
            # line features
            cl_list = [llen, lmx, lmy, lorn, lcon, 0]

            lsfarr_row = np.zeros(6, dtype='float32')

            for cl in range(0, 6):
                lsfarr_row[cl] = cl_list[cl]
            
            lsfarr = np.vstack((lsfarr, lsfarr_row))

        lsfim[0] = lsfima
        lsfim[1] = lsfimb

        # TEST
        # plt.show()
        # sys.exit()
        # TEST

        return lsfim, lsfarr
    

def feature_lsr(orientation, magnitude, x_deriv, y_deriv):

    """
    Args:
        orientation: Gradient orientation
        magnitude: Gradient magnitude
    """

    rows, cols = x_deriv.shape

    # Threshold the edge magnitude
    data, edge_pixels = get_edge_pixels(orientation, magnitude, .5)

    # quantize gradient orientations
    obj = BinQ(data, edge_pixels, 5, x_deriv, y_deriv, rows, cols)   # any LSR below 5 pixels can be ignored
    
    lsfarr = obj.lsfarr

    bin_count = np.float32(np.searchsorted(range(5, 200+4, 4), lsfarr[:, 0]))
    lenpmf = bin_count / bin_count.sum()

    bin_count = np.float32(np.searchsorted(list(np.linspace(0, 10, num=np.floor(10./.5))), lsfarr[:, 4]))

    contrastpmf = bin_count / bin_count.sum()

    fea1 = -(np.multiply(lenpmf, np.log(np.add(lenpmf, 1e-5))).sum())
    fea2 = lsfarr[:, 4].mean()
    fea3 = -(np.multiply(contrastpmf, np.log(np.add(contrastpmf, 1e-5))).sum())

    feas = np.array([fea1, fea2, fea3])
    feas[(np.isnan(feas))] = 0.

    return feas
