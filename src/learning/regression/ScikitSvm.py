from sklearn.svm import SVR

from src.learning.evaluation.CrossValidation import train_test_split
from src.learning.regression.ScikitLearning import ScikitModel


__author__ = 'Kern'


class ScikitSvm(ScikitModel):
    def __init__(self, knr):
        ScikitModel.__init__(self)
        self.model = SVR(kernel=knr)