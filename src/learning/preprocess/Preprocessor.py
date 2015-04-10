from sklearn import preprocessing

__author__ = 'Kern'


class Normalization:
    def __init__(self, x_train):
        self.normalizer = preprocessing.Normalizer().fit(x_train)

    def transform(self, data):
        return self.transform(data)


class Standardization:
    def __init__(self, x_train):
        self.scalar = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit(x_train)

    def mean(self):
        return self.scalar.mean_

    def variance(self):
        return self.scalar.std_

    def transform(self, data):
        return self.scalar.transform(data)