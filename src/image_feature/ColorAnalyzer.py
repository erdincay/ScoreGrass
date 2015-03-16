import numpy as np
import pandas as pd
from enum import Enum
from skimage import color, img_as_float

__author__ = 'kern.ding'

feature_name_average = 'avr'
feature_name_coverage = 'cover'
feature_name_filtered = "filtered"


class ColorChannel(Enum):
    Red = 1,
    Green = 2,
    Blue = 3,
    Hue = 4


def hue_range_to_float(hue):
    return hue / 360


def rgb_range_to_float(rgb):
    return rgb / 256


def average_color(image):
    if len(image.shape) != 2:
        raise ValueError("input is not a single color channel image")

    return np.average(image)


def color_coverage(image, minimum, maximum):
    if len(image.shape) != 2:
        raise ValueError("input is not a single color channel image")

    bool_matrix = (image >= minimum) & (image <= maximum)
    pixels_in_range = np.count_nonzero(bool_matrix)
    percent_cover = pixels_in_range / (image.shape[0] * image.shape[1])
    filtered_color = np.sum(image[bool_matrix]) / pixels_in_range

    return percent_cover, filtered_color


def compute_feats(image, minimum, maximum, color_channel):
    image = img_as_float(image)
    if color_channel == ColorChannel.Hue:
        img_hsv = color.rgb2hsv(image)
        single_avr = average_color(img_hsv[:, :, 0])
        single_cover, single_filtered = color_coverage(img_hsv[:, :, 0], hue_range_to_float(minimum),
                                                       hue_range_to_float(maximum))
    else:
        single_avr = average_color(image[:, :, color_channel.value[0] - 1])
        single_cover, single_filtered = color_coverage(image[:, :, color_channel.value[0] - 1],
                                                       rgb_range_to_float(minimum), rgb_range_to_float(maximum))

    label_l2 = [feature_name_average, feature_name_coverage, feature_name_filtered]
    label_l1 = [color_channel.name] * len(label_l2)

    index = pd.MultiIndex.from_tuples(list(zip(label_l1, label_l2)), names=['method', 'attr'])

    return pd.Series([single_avr, single_cover, single_filtered], index)
