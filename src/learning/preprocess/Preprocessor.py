from sklearn import preprocessing
from sklearn.externals import joblib

__author__ = 'Kern'


class Normalization:
    def __init__(self, x_train):
        self.normalizer = preprocessing.Normalizer().fit(x_train)

    def transform(self, data):
        return self.transform(data)


class Standardization:
    def __init__(self, x_train=None):
        if x_train is not None:
            self.scalar = preprocessing.StandardScaler().fit(x_train)
        else:
            self.scalar = preprocessing.StandardScaler()

    def scale(self, data):
        self.scalar.fit(data)

    def transform(self, data):
        return self.scalar.transform(data)

    def save(self, path):
        joblib.dump(self.scalar, path)

    def load(self, path):
        self.scalar = joblib.load(path)
