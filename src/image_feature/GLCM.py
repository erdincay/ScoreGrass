import numpy as np
from skimage.feature import greycomatrix
from skimage.feature import greycoprops

__author__ = 'Kern'


def compute_features(image, distances, angles):
    glcm = greycomatrix(image, distances, angles, 256, symmetric=True, normed=True)
    dissimilarities = greycoprops(glcm, 'dissimilarity').flat
    correlations = greycoprops(glcm, 'correlation').flat
    return np.array([[d, c] for d in dissimilarities for c in correlations])