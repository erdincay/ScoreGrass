from sklearn.linear_model import Lasso
from src.learning.regression.ScikitLearning import ScikitModel

__author__ = 'Kern'


class Lasso(ScikitModel):
    def __init__(self):
        super().__init__(Lasso(alpha=0.1))