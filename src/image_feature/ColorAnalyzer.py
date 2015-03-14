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


def average_color(image):
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("input is not a color image")

    return np.average(image[:, :, 0]), np.average(image[:, :, 1]), np.average(image[:, :, 2])


def color_coverage(image_grey, minimum, maximum):
    if len(image_grey.shape) != 2:
        raise ValueError("input is not a single color channel image")

    bool_matrix = (image_grey >= minimum) & (image_grey <= maximum)
    pixels_in_range = np.count_nonzero(bool_matrix)
    percent_cover = pixels_in_range / (image_grey.shape[0] * image_grey.shape[1])
    filtered_color = np.sum(image_grey[bool_matrix]) / pixels_in_range

    return percent_cover, filtered_color


def compute_feats(image, minimum, maximum, color_channel):
    if color_channel == ColorChannel.Hue:
        img_hsv = color.rgb2hsv(img_as_float(image))
        single_avr = average_color(img_hsv)[0]
        single_cover, single_filtered = color_coverage(img_hsv[:, :, 0], minimum, maximum)
    else:
        single_avr = average_color(image)[color_channel.value - 1]
        single_cover, single_filtered = color_coverage(image[:, :, color_channel.value - 1], minimum, maximum)

    label_l2 = [feature_name_average, feature_name_coverage, feature_name_filtered]
    label_l1 = [color_channel.name] * len(label_l2)

    index = pd.MultiIndex.from_tuples(list(zip(label_l1, label_l2)), names=['method', 'attr'])

    return pd.Series([single_avr, single_cover, single_filtered], index)
