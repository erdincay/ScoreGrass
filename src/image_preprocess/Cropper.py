import numpy as np

__author__ = 'Kern'


def extreme_value(coords):
    if len(coords.shape) != 2 or coords.shape[1] != 2:
        raise TypeError("input is not a list of 2D coordinate")

    row_min = np.amin(coords[:, 0])
    row_max = np.amax(coords[:, 0])
    col_min = np.amin(coords[:, 1])
    col_max = np.amax(coords[:, 1])

    return row_min, row_max, col_min, col_max


def diagonal_cropping(image, coords):
    row_min, row_max, col_min, col_max = extreme_value(coords)

    if len(image.shape) == 2:
        return image[row_min: row_max, col_min: col_max]
    elif len(image.shape) == 3:
        return image[row_min: row_max, col_min: col_max, :]
    else:
        raise TypeError("input is not a numpy array image")