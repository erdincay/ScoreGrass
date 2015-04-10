from sklearn.linear_model import Lasso

from src.learning.regression.ScikitLearning import ScikitModel


__author__ = 'Kern'


class ScikitLasso(ScikitModel):
    def __init__(self, s_id):
        super().__init__(Lasso(alpha=0.1), s_id)