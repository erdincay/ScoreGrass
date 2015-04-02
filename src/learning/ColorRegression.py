from src.learning.RegressionManager import RegressionManager
from src.learning.evaluation import CrossValidation
from src.learning.preprocess import Preprocessor
from src.learning.regression import ScikitRidge
from src.learning.regression import ScikitLasso

__author__ = 'Kern'


def _register_regression():
    return [
        ScikitLasso.Lasso(),
        ScikitRidge.Ridge(),
    ]


class ColorRegression(RegressionManager):
    def __init__(self):
        super().__init__(_register_regression)

    def train(self, x_data, y_data):
        x_train, x_test, y_train, y_test = CrossValidation.data_set_split(0.3)(x_data, y_data)
        scalar = Preprocessor.Standardization(x_train)
        x_train_scaled = scalar.transform(x_train)
        x_test_scaled = scalar.transform(x_test)

        for m in self.model_list:
            m.train(x_train_scaled, y_train)