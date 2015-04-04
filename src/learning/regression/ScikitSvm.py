from sklearn.svm import SVR

from src.learning.regression.ScikitLearning import ScikitModel


__author__ = 'Kern'


class ScikitSvm(ScikitModel):
    def __init__(self, knr, s_id):
        super().__init__(SVR(kernel=knr), s_id)