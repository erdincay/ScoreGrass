from sklearn.linear_model import Ridge
from src.learning.regression.ScikitLearning import ScikitModel

__author__ = 'Kern'


class Ridge(ScikitModel):
    def __init__(self):
        super().__init__(Ridge(alpha=0.1))