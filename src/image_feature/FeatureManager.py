import pandas as pd
import numpy as np
from skimage.color import rgb2grey

from src.image_feature import GaborFilter
from src.image_feature import GLCM
from src.image_feature import ColorAnalyzer


__author__ = 'kern.ding'


def register_feature_calculators():
    return [
        lambda img: GaborFilter.compute_feats(rgb2grey(img), [GaborFilter.generate_kernel(0.15, 0.25 * np.pi, 1)]),
        lambda img: GLCM.compute_feats(rgb2grey(img), [5], [0]),
        lambda img: ColorAnalyzer.compute_feats(img, 150, 255, ColorAnalyzer.ColorChannel.Green)
    ]


def compute_feats(image):
    """
    merge different feature calculator output together
    :param image:
    :return:
    """
    feats = pd.Series()
    for calculator in register_feature_calculators():
        feat = calculator(image)
        feats = feats.append(feat)

    return feats