import numpy as np

__author__ = 'Kern'


def extreme_value(coords):
    """
    find the max/min value in an 2D array for each axis
    :param coords: input array
    :return: :raise ValueError:
    """
    if len(coords.shape) != 2 or coords.shape[1] != 2:
        raise ValueError("input is not a list of 2D coordinate")

    row_min = np.amin(coords[:, 0])
    row_max = np.amax(coords[:, 0])
    col_min = np.amin(coords[:, 1])
    col_max = np.amax(coords[:, 1])

    return row_min, row_max, col_min, col_max


def sequence_extreme_value(coords, index):
    """
    find the nth max/min value in an 2D array for each axis
    :param coords: input array
    :param index: specify the nth index
    :return: :raise ValueError:
    """
    if len(coords.shape) != 2 or coords.shape[1] != 2:
        raise ValueError("input is not a list of 2D coordinate")

    min_index = index
    max_index = -1 - index
    row_min = coords[:, 0][np.argpartition(coords[:, 0], min_index)[min_index]]
    row_max = coords[:, 0][np.argpartition(coords[:, 0], max_index)[max_index]]
    col_min = coords[:, 1][np.argpartition(coords[:, 1], min_index)[min_index]]
    col_max = coords[:, 1][np.argpartition(coords[:, 1], max_index)[max_index]]

    return row_min, row_max, col_min, col_max


def diagonal_cropping(image, coords, extreme_extraction, offset=0):
    """
    Crop image into rectangle, the diagonal line is calculate via minimum and maximum value from a set of coordinate
    :param image: input image
    :param coords: a set of coordinates that indicate the interest region of the image
    :return: :raise ValueError: cropped image
    """
    row_min, row_max, col_min, col_max = extreme_extraction(coords)

    row_min += offset
    col_min += offset
    row_max -= offset
    col_max -= offset

    if len(image.shape) == 2:
        return image[row_min: row_max, col_min: col_max]
    elif len(image.shape) == 3:
        return image[row_min: row_max, col_min: col_max, :]
    else:
        raise ValueError("input is not a numpy array image")