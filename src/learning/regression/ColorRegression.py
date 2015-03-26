from src.learning.regression import ScikitRidge
from src.learning.regression import ScikitLasso

__author__ = 'Kern'


def register_regression_color():
    return [
        ScikitLasso.Lasso(),
        ScikitRidge.Ridge(),
    ]


def train_color(learn_models, x_data, y_data):
    for m in learn_models:
        m.train(x_data, y_data)