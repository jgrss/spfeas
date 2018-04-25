from progressbar.progressbar import ProgressBar
from progressbar import widgets

try:
    import numpy as np
except ImportError:
    raise ImportError('NumPy must be installed')


def _iteration_parameters(image_rows, image_cols, row_block_size, col_block_size, y_overlap=0, x_overlap=0, bands=1):

    maximum_blocks = 0

    for i in range(0, image_rows, row_block_size-y_overlap):

        for j in range(0, image_cols, col_block_size-x_overlap):

            if bands > 1:
                for band in range(1, bands+1):
                    maximum_blocks += 1
            else:
                maximum_blocks += 1

    progress_widgets = [' Percent: ', widgets.Percentage(), ' ',
                        widgets.Bar(marker='*', left='[', right=']'), ' ', widgets.ETA(), ' ',
                        widgets.FileTransferSpeed()]

    progress_bar = ProgressBar(widgets=progress_widgets, maxval=maximum_blocks)

    progress_bar.start()

    return 1, progress_bar


def _iteration_parameters_values(value1, value2):

    # Set widget and pbar
    progress_widgets = [' Perc: ', widgets.Percentage(), ' ',
                        widgets.Bar(marker='*', left='[', right=']'), ' ',
                        widgets.ETA(), ' ', widgets.FileTransferSpeed()]

    progress_bar = ProgressBar(widgets=progress_widgets, maxval=value1*value2)

    progress_bar.start()

    return 1, progress_bar
