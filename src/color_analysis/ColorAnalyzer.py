from skimage import io
import numpy
from ColorData import ColorData
from ColorFilter import ColorFilter
__author__ = 'Alec'

#323

def getColorVals(np):
    """
    Creates and initializes a ColorData object using specified image data.

    :param np: image data to extract data from
    :return: new ColorData object
    """

    c_data = ColorData(np)
    c_data.calc_color_values()

    return c_data

def getColorFilter(np):
    """
    Creates and initializes a ColorFilter object using specified image data.

    :param np: image data to apply the filter to
    :return: new ColorFilter object
    """

    c_filter = ColorFilter(np)

    return c_filter