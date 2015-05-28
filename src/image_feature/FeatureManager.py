import pandas as pd
from skimage.color import rgb2grey

from src.image_feature import GaborFilter
from src.image_feature import ColorAnalyzer

__author__ = 'kern.ding'


def register_feature_calculators():
    return [
        lambda img: GaborFilter.compute_feats(rgb2grey(img), GaborFilter.generate_kernels(2)),
        # lambda img: GLCM.compute_feats(rgb2grey(img), [1, 5, 10, 20], [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4]),
        lambda img: ColorAnalyzer.compute_feats(img, 150, 255, ColorAnalyzer.ColorChannel.Green),
        lambda img: ColorAnalyzer.compute_feats(img, 50, 150, ColorAnalyzer.ColorChannel.Hue)
    ]


def compute_feats(image):
    """
    merge different feature calculator output together
    :param image:
    :return:
    """
    return pd.concat([calculator(image) for calculator in register_feature_calculators()])
