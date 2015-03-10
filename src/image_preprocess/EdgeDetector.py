from skimage.color import rgb2grey
from skimage.filters import threshold_otsu
from skimage.measure import find_contours
from skimage.morphology import closing, square
from skimage.feature import canny

__author__ = 'Kern'


def binarize(image):
    if len(image.shape) > 2:
        image = rgb2grey(image)

    threshold = threshold_otsu(image)
    return closing(image > threshold, square(3))


def canny_edge_detector(binary_img, sigma):
    return canny(binary_img, sigma)


def edge_coordinate(edge_map):
    ret = []

    for contour in find_contours(edge_map, 0.5, fully_connected='high'):
        ret.extend(contour)

    return ret