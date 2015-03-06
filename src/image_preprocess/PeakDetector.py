from skimage import img_as_float
from skimage import color
import numpy as np
from skimage.feature import peak_local_max

__author__ = 'Kern'


def generate_red_map(image):
    img_f = img_as_float(image)
    black_mask = color.rgb2gray(img_f) < 0.1
    distance_red = color.rgb2gray(1 - np.abs(img_f - (1, 0, 0)))
    distance_red[black_mask] = 0
    return distance_red


def peak_detector(distance_map):
    return peak_local_max(distance_map, threshold_rel=0.825, min_distance=50, num_peaks=8)


