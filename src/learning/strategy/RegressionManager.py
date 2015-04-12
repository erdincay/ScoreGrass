import os

from src.file_io import PublicSupport
from src.learning.evaluation import CrossValidation
from src.learning.preprocess import Preprocessor


__author__ = 'Kern'


def combine_name(parent, self):
    return parent + '_' + self + '.model'


def deserialize_json(path, serialized_id):
    json_dict = PublicSupport.read_json(os.path.join(path, serialized_id + ".json"))
    return json_dict[RegressionManager.x_dim_name], json_dict[RegressionManager.y_dim_name]


class RegressionManager:
    x_dim_name = 'x_dim'
    y_dim_name = 'y_dim'

    def __init__(self, model_list, x_dim, y_dim):
        self.model_list = model_list
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.scalar = Preprocessor.Standardization()

    def _scale(self, x_data):
        if self.scalar:
            return self.scalar.transform(x_data)
        else:
            return x_data

    def train(self, x_data, y_data):
        assert (x_data.shape[0] == y_data.shape[0])
        self.scalar = Preprocessor.Standardization(x_data)
        for model in self.model_list:
            model.train(self._scale(x_data), y_data)

    def predict(self, x_data):
        ret_dict = {}
        for model in self.model_list:
            predicted = model.predict(self._scale(x_data))
            if len(predicted.shape) == 1:
                ret_dict[model.serialize_id] = predicted
            elif len(predicted.shape) == 2:
                for index in range(predicted.shape[1]):
                    ret_dict[model.serialize_id + str(index)] = predicted[:, index]
            else:
                raise ValueError('wrong dimensions of prediction')
        return ret_dict

    def score(self, x_data, y_data):
        assert (x_data.shape[0] == y_data.shape[0])
        return {model.serialize_id: model.score(self._scale(x_data), y_data) for model in self.model_list}

    def validation(self, x_data, y_data, factor):
        assert (x_data.shape[0] == y_data.shape[0])
        x_train, x_test, y_train, y_test = CrossValidation.train_test_split(factor)(x_data, y_data)
        self.train(x_train, y_train)
        return self.score(x_test, y_test)

    def save(self, path):
        json_file = os.path.join(path, self.__class__.__name__ + '.json')
        PublicSupport.write_json({RegressionManager.x_dim_name: self.x_dim, RegressionManager.y_dim_name: self.y_dim},
                                 json_file)
        scalar_file = os.path.join(path, self.__class__.__name__ + '.scalar')
        self.scalar.save(scalar_file)
        for model in self.model_list:
            model.save(os.path.join(path, model.serialize_id))
        return json_file

    def load(self, path):
        scalar_file = os.path.join(path, self.__class__.__name__ + '.scalar')
        self.scalar = Preprocessor.Standardization().load(scalar_file)
        for model in self.model_list:
            model.load(os.path.join(path, model.serialize_id))
        return self
