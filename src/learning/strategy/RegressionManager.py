import os
import math

from src.learning.evaluation import CrossValidation
from src.learning.preprocess import Preprocessor
from src.learning.regression.ScikitLasso import ScikitLasso
from src.learning.regression.ScikitRidge import ScikitRidge
from src.learning.regression.ScikitBayesianRidge import ScikitBayesianRidge
from src.learning.regression.PyBrainANNs import PyBrainANNs
from src.learning.regression.ScikitMultiTaskLasso import ScikitMultiTaskLasso
from src.learning.regression.ScikitSvm import ScikitSvm
from src.learning.regression.TheanetsANNs import TheanetsANNs


__author__ = 'Kern'

serialize_extension = '.model'


def combine_name(parent, self):
    return parent + '_' + self + serialize_extension


def _register_color_regression(str_parent):
    return [
        ScikitLasso(combine_name(str_parent, ScikitLasso.__name__)),
        ScikitRidge(combine_name(str_parent, ScikitRidge.__name__))
    ]


def _register_quality_regression(x_dim, y_dim, str_parent):
    return [
        ScikitBayesianRidge(combine_name(str_parent, ScikitBayesianRidge.__name__)),
        ScikitSvm('rbf', combine_name(str_parent, ScikitSvm.__name__)),
        TheanetsANNs(x_dim, y_dim, math.floor(x_dim / 2), combine_name(str_parent, TheanetsANNs.__name__)),
        PyBrainANNs(x_dim, y_dim, math.floor(x_dim / 2), combine_name(str_parent, PyBrainANNs.__name__))
    ]


def _register_mixed_regression(x_dim, y_dim, str_parent):
    return [
        ScikitMultiTaskLasso(combine_name(str_parent, ScikitMultiTaskLasso.__name__)),
        TheanetsANNs(x_dim, y_dim, math.floor(x_dim / 2), combine_name(str_parent, TheanetsANNs.__name__)),
        PyBrainANNs(x_dim, y_dim, math.floor(x_dim / 2), combine_name(str_parent, PyBrainANNs.__name__))
    ]


class RegressionManager:
    def __init__(self, model_list):
        self.model_list = model_list
        self.scalar = None

    def train(self, x_data, y_data):
        assert (x_data.shape[0] == y_data.shape[0])

        x_train, x_test, y_train, y_test = CrossValidation.train_test_split(0.25)(x_data, y_data)
        self.scalar = Preprocessor.Standardization(x_train)
        x_train_scaled = self.scalar.transform(x_train)
        x_test_scaled = self.scalar.transform(x_test)

        for model in self.model_list:
            model.train(x_train_scaled, y_train)

    def save(self, path):
        for model in self.model_list:
            model.save(os.path.join(path, model.serialize_id))

    def load(self, path):
        for model in self.model_list:
            model.load(os.path.join(path, model.serialize_id))

    def predict(self, x_data):
        x_data_scaled = self.scalar.transform(x_data)
        return [m.predict(x_data_scaled) for m in self.model_list]

    @classmethod
    def color_regression(cls):
        return cls(_register_color_regression())

    @classmethod
    def quality_regression(cls, x_dim, y_dim):
        return cls(_register_quality_regression(x_dim, y_dim))