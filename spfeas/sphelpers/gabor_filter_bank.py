try:
    import numpy as np
except ImportError:
    raise ImportError('NumPy must be installed')

try:
    from skimage.filters import gabor_kernel
except ImportError:
    raise ImportWarning('Skimage.filter.gabor_kernel did not load')


def prep_gabor(n_orientations=32, sigmas=[1, 2, 4]):

    """
    Prepare the Gabor kernels
    """

    # prepare filter bank kernels
    kernels = []

    frequencies = [.05, .25]

    theta = np.pi * 0 / n_orientations

    kernel = gabor_kernel(frequencies[1], theta=theta, sigma_x=sigmas[1], sigma_y=sigmas[1]).real

    kernels.append(np.float32(kernel))

    for th in xrange(3, n_orientations, 4):

        theta = np.pi * th / n_orientations

        # for sigma in sigmas:

        # for gamma in frequencies:

        kernel = gabor_kernel(frequencies[1], theta=theta, sigma_x=sigmas[1], sigma_y=sigmas[1]).real
        # kernel = cv2.getGaborKernel((21, 21), sigmas[1], th, 10, frequencies[1])	# kernel size, std dev, direction, wavelength, frequency

        kernels.append(np.float32(kernel))

    return kernels[:-1]
