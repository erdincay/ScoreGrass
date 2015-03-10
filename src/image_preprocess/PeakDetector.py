from skimage import img_as_float
from skimage import color
import numpy as np
from skimage.feature import peak_local_max, corner_peaks

__author__ = 'Kern'


def generate_color_map(image, color_mask):
    img_f = img_as_float(image)
    black_mask = color.rgb2gray(img_f) < 0.1
    distance = color.rgb2gray(1 - np.abs(img_f - color_mask))
    distance[black_mask] = 0
    return distance


def generate_red_map(image):
    return generate_color_map(image, (1, 0, 0))


def generate_green_map(image):
    return generate_color_map(image, (0, 1, 0))


def generate_pink_map(image):
    return generate_color_map(image, (1, 0.5, 0.5))


def peak_detector(distance_map, threshold, min_d):
    return peak_local_max(distance_map, threshold_rel=threshold, min_distance=min_d, num_peaks=8)


def peak_corner_detector(distance_map, threshold, min_d):
    """
    well, no idea what is the difference from skimage.feature.peak_local_max
    :param distance_map:
    :param threshold:
    :param min_d:
    :return:
    """
    return corner_peaks(distance_map, threshold_rel=threshold, min_distance=min_d, num_peaks=8)
