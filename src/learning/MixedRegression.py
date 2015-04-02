from src.learning.RegressionManager import RegressionManager
from src.learning.evaluation import CrossValidation
from src.learning.preprocess import Preprocessor
from src.learning.regression import ScikitMultiTaskLasso
from src.learning.regression import TheanetsANNs
from src.learning.regression import PyBrainANNs

__author__ = 'Kern'


def _register_regression(x_dim, y_dim):
    return [
        ScikitMultiTaskLasso.MultiTaskLasso(),
        TheanetsANNs.TheanetsANNs(x_dim, y_dim, x_dim / 2),
        PyBrainANNs.PyBrainANNs(x_dim, y_dim, x_dim / 2),
    ]


class MixedRegression(RegressionManager):
    def __init__(self, x_dim, y_dim):
        super().__init__(_register_regression(x_dim, y_dim))

    def train(self, x_data, y_data):
        x_train, x_test, y_train, y_test = CrossValidation.data_set_split(0.3)(x_data, y_data)
        scalar = Preprocessor.Standardization(x_train)
        x_train_scaled = scalar.transform(x_train)
        x_test_scaled = scalar.transform(x_test)

        for m in self.model_list:
            m.train(x_train_scaled, y_train)