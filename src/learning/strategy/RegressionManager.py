import os

from src.learning.evaluation import CrossValidation
from src.learning.preprocess import Preprocessor


__author__ = 'Kern'

serialize_extension = '.model'


def combine_name(parent, self):
    return parent + '_' + self + serialize_extension


def feature_dimensions(data_struct):
    if len(data_struct.shape) >= 2:
        return data_struct.shape[1]
    elif len(data_struct.shape) == 1:
        return 1

    raise TypeError('unknown pandas struct type')


class RegressionManager:
    def __init__(self, model_list, s_path):
        self.serialize_path = s_path
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
            model.save(os.path.join(self.serialize_path, model.serialize_id))

    def predict(self, x_data):
        x_data_scaled = self.scalar.transform(x_data)
        return [m.predict(x_data_scaled) for m in self.model_list]