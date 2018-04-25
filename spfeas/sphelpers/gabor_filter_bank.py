from __future__ import division

try:
    import numpy as np
except ImportError:
    raise ImportError('NumPy must be installed')

try:
    import cv2
except ImportError:
    raise ImportError('OpenCV must be installed')

# try:
#     from skimage.filters import gabor_kernel
#     from skimage.transform import resize
# except ImportError:
#     raise ImportWarning('Skimage.filter.gabor_kernel did not load')


def prep_gabor(n_orientations=32, sigma=3., lambd=10., gamma=.5, psi=1., kernel_size=None, theta_skip=4):

    """
    Prepare the Gabor kernels

    Args:
        n_orientations (int)
        sigma (float): The standard deviation.
        lambd (float): The wavelength of the sinusoidal factor.
        gamma (float): The spatial aspect ratio.
        psi (float): The phase offset.
        kernel_size (tuple): The Gabor kernel size.
        theta_skip (int): The `theta` skip factor.
    """

    if not isinstance(kernel_size, tuple):
        kernel_size = (15, 15)

    # Prepare Gabor kernels.
    kernels = list()

    # kernel = resize(gabor_kernel(frequency,
    #                              theta=theta,
    #                              bandwidth=lambd,
    #                              sigma_x=sigma,
    #                              sigma_y=sigma,
    #                              offset=psi)

    for th in range(0, n_orientations, theta_skip):

        # The kernel orientation.
        theta = np.pi * th / n_orientations

        kernel = cv2.getGaborKernel(kernel_size, sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)

        kernel /= 1.5 * kernel.sum()

        kernels.append(np.float32(kernel))

    return kernels


def visualize(out_fig, cmap, grid_rows, grid_cols, **kwargs):

    import matplotlib.pyplot as plt

    gabor_kernels = prep_gabor(**kwargs)

    fig = plt.figure()

    for gi, gabor_kernel in enumerate(gabor_kernels):

        ax = fig.add_subplot(grid_rows, grid_cols, gi+1)

        ax.imshow(gabor_kernel, interpolation='lanczos', cmap=cmap)

        plt.axis('off')

    plt.tight_layout(h_pad=.1,
                     w_pad=.1)

    plt.savefig(out_fig,
                transparent=True,
                dpi=300)
