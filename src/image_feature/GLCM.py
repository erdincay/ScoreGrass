import numpy as np
from skimage.feature import greycomatrix
from skimage.feature import greycoprops

__author__ = 'Kern'


def compute_features(image, distances, angles):
    """
    compute the texture feature by grey level co-occurrence matrices
    :param image: is just numpy array
    :param distances: List of pixel pair distance offsets
    :param angles: List of pixel pair angles in radians for the offsets
    :return: numpy array [[diss1, corr1], [diss2, corr2], [diss3, corr3], [diss4, corr4]... ] stand for dissimilarity and correlation attribute of co-occurrence matrix  by different input parametes combinations [[dis1, ang1], [dis1, ang2],[dis2, ang1],[dis2, ang2]]. So there are totally len(distances) * len(angles) pairs of return features
    """
    glcm = greycomatrix(image, distances, angles, 256, symmetric=True, normed=True)
    dissimilarities = greycoprops(glcm, 'dissimilarity').flat
    correlations = greycoprops(glcm, 'correlation').flat
    return np.array([[d, c] for d in dissimilarities for c in correlations])