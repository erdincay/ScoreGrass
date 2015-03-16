import pandas as pd
from skimage.feature import greycomatrix
from skimage.feature import greycoprops

__author__ = 'Kern'

feature_method_name = 'GLCM'
feature_name_dissimilarity = 'diss{0}'
feature_name_correlation = 'corr{0}'
feature_name_energy = 'eng{0}'


def compute_feats(image, distances, angles):
    """
    compute the texture feature by grey level co-occurrence matrices
    :param image: is just numpy array
    :param distances: List of pixel pair distance offsets
    :param angles: List of pixel pair angles in radians for the offsets
    :return: [[diss1, corr1], [diss2, corr2], [diss3, corr3], [diss4, corr4]... ] stand for dissimilarity and correlation attribute of co-occurrence matrix  by different input parametes combinations [[dis1, ang1], [dis1, ang2],[dis2, ang1],[dis2, ang2]]. So there are totally len(distances) * len(angles) pairs of return features, wrappd by pandas.Series
    """
    glcm = greycomatrix(image, distances, angles, 256, symmetric=True, normed=True)
    dissimilarities = greycoprops(glcm, 'dissimilarity').flat
    correlations = greycoprops(glcm, 'correlation').flat
    energy = greycoprops(glcm, 'energy').flat

    data = []
    label_l2 = []
    for idx, (d, c, e) in enumerate(zip(dissimilarities, correlations, energy)):
        data.append(d)
        label_l2.append(feature_name_dissimilarity.format(idx))

        data.append(c)
        label_l2.append(feature_name_correlation.format(idx))

        data.append(e)
        label_l2.append(feature_name_energy.format(idx))

    label_l1 = [feature_method_name] * len(data)
    index = pd.MultiIndex.from_tuples(list(zip(label_l1, label_l2)), names=['method', 'attr'])

    return pd.Series(data, index)