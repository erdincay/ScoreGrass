from skimage.color import rgb2grey
from skimage.filters import threshold_otsu
from skimage.measure import find_contours
from skimage.morphology import closing, square
from skimage.feature import canny
import numpy as np

__author__ = 'Kern'


def binarize(image):
    image = rgb2grey(image)

    threshold = threshold_otsu(image)
    return closing(image > threshold, square(3))


def canny_edge_detector(binary_img, sigma):
    return canny(binary_img, sigma)


def find_edges(edge_map):
    return find_contours(edge_map, 0.5, fully_connected='high')


def edge_coordinate(contours):
    ret = []
    for contour in contours:
        ret.extend(contour)
    return np.array(ret)
