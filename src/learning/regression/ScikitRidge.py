from sklearn.linear_model import Ridge

from src.learning.regression.ScikitLearning import ScikitModel

__author__ = 'Kern'


class ScikitRidge(ScikitModel):
    def __init__(self, s_id):
        super().__init__(Ridge(alpha=0.1), s_id)
