try:
    import numpy as np
except ImportError:
    raise ImportError('NumPy must be installed')

try:
    from skimage.filters import gabor_kernel
    from skimage.transform import resize
except ImportError:
    raise ImportWarning('Skimage.filter.gabor_kernel did not load')


def prep_gabor(n_orientations=32, frequency=.2, sigmas=[1, 2, 4], kernel_size=(15, 15)):

    """
    Prepare the Gabor kernels
    """

    # prepare filter bank kernels
    kernels = []

    theta = np.pi * 0 / n_orientations

    kernel = resize(gabor_kernel(frequency, theta=theta, sigma_x=sigmas[1], sigma_y=sigmas[1]).real, kernel_size, order=3)

    kernels.append(np.float32(kernel))

    for th in xrange(3, n_orientations, 4):

        theta = np.pi * th / n_orientations

        # for sigma in sigmas:

        # for gamma in frequencies:

        kernel = resize(gabor_kernel(frequency, theta=theta, sigma_x=sigmas[1], sigma_y=sigmas[1]).real, kernel_size, order=3)
        # kernel = cv2.getGaborKernel((21, 21), sigmas[1], th, 10, frequencies[1])	# kernel size, std dev, direction, wavelength, frequency

        kernels.append(np.float32(kernel))

    return kernels[:-1]
