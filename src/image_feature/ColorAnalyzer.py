import numpy as np
import pandas as pd
from skimage import color, img_as_float

__author__ = 'kern.ding'

feature_method_name = 'Green'
feature_name_green_average = 'avr'
feature_name_green_coverage = 'cover'
feature_name_green_filtered = "filtered"


def average_color(image):
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("input is not a color image")

    return np.average(image[:, :, 0]), np.average(image[:, :, 1]), np.average(image[:, :, 2])


def average_green(image):
    return average_color(image)[1]


def average_hue(image):
    return average_color(color.rgb2hsv(img_as_float(image)))[0]


def color_coverage(image_grey, minimum, maximum):
    if len(image_grey.shape) != 2:
        raise ValueError("input is not a single color channel image")

    bool_matrix = (image_grey >= minimum) & (image_grey <= maximum)
    pixels_in_range = np.count_nonzero(bool_matrix)
    percent_cover = pixels_in_range / (image_grey.shape[0] * image_grey.shape[1])
    filtered_color = np.sum(image_grey[bool_matrix]) / pixels_in_range

    return percent_cover, filtered_color


def compute_feats(image, minimum, maximum):
    green_avr = average_green(image)
    green_cover, green_filtered = color_coverage(image[:, :, 1], minimum, maximum)

    label_l2 = [feature_name_green_average, feature_name_green_coverage, feature_name_green_filtered]
    label_l1 = [feature_method_name] * len(label_l2)

    index = pd.MultiIndex.from_tuples(list(zip(label_l1, label_l2)), names=['method', 'attr'])

    return pd.Series([green_avr, green_cover, green_filtered], index)
