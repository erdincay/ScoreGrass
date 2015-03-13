import pandas as pd

from src.image_feature import GaborFilter
from src.image_feature import GLCM


__author__ = 'kern.ding'


def register_feature_calculators():
    return [
        lambda img: GaborFilter.compute_feats(img, GaborFilter.generate_kernels(1)),
        lambda img: GLCM.compute_feats(img, [5], [0])
    ]


class FeatureManager:
    def __init__(self):
        self.calculators = register_feature_calculators()

    def compute_feats(self, image):
        """
        merge different feature calculator output together
        :param image:
        :return: 
        """
        feats = pd.Series()
        for calculator in self.calculators:
            feat = calculator(image)
            feats.append(feat)

        return feats

