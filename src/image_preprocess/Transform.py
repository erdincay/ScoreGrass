from skimage import transform

__author__ = 'Kern'


def resize(image, target_shape):
    return transform.resize(image, target_shape)


def rescale(image, factor):
    return transform.rescale(image, factor)