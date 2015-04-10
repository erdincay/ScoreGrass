import os

from src.file_io import PublicSupport
from src.learning.evaluation import CrossValidation
from src.learning.preprocess import Preprocessor

__author__ = 'Kern'

serialize_extension = '.model'


def combine_name(parent, self):
    return parent + '_' + self + serialize_extension


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
        self.scalar = None

    def train(self, x_data, y_data):
        assert (x_data.shape[0] == y_data.shape[0])
        x_train, x_test, y_train, y_test = CrossValidation.train_test_split(0.25)(x_data, y_data)
        self.scalar = Preprocessor.Standardization(x_train)
        x_train_scaled = self.scalar.transform(x_train)
        x_test_scaled = self.scalar.transform(x_test)
        for model in self.model_list:
            model.train(x_train_scaled, y_train)
            print(model.serialize_id, ' validate on test dataset: ', model.score(x_test_scaled, y_test))

    def predict(self, x_data):
        x_data_scaled = self.scalar.transform(x_data)
        return [m.predict(x_data_scaled) for m in self.model_list]

    def save(self, path):
        file_id = os.path.join(path, self.__class__.__name__ + '.json')
        PublicSupport.write_json({RegressionManager.x_dim_name: self.x_dim, RegressionManager.y_dim_name: self.y_dim}, file_id)
        for model in self.model_list:
            model.save(os.path.join(path, model.serialize_id))
        return file_id

    def load(self, path):
        for model in self.model_list:
            model.load(os.path.join(path, model.serialize_id))
        return self
