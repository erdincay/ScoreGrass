from enum import Enum
from src.learning.evaluation import CrossValidation
from src.learning.preprocess import Preprocessor

__author__ = 'Kern'

serialize_extension = '.model'


def combine_name(parent, self):
    return parent + '_' + self + serialize_extension


class ModelType(Enum):
    train_only = 1,
    train_test = 2


class RegressionManager:
    def __init__(self, model_list):
        self.model_list = model_list

    def train(self, x_data, y_data):
        x_train, x_test, y_train, y_test = CrossValidation.data_set_split(0.3)(x_data, y_data)
        scalar = Preprocessor.Standardization(x_train)
        x_train_scaled = scalar.transform(x_train)
        x_test_scaled = scalar.transform(x_test)

        for model_t, model in self.model_list:
            if model_t == ModelType.train_test:
                model.train(x_train_scaled, y_train, x_test_scaled, y_test)
            else:
                model.train(x_train_scaled, y_train)

            model.save(model.serialize_id)

    def predict(self, x_data):
        return [m.predict(x_data) for m in self.model_list]