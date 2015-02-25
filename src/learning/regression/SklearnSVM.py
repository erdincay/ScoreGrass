from sklearn.svm import SVR
from sklearn import cross_validation

__author__ = 'Kern'


class SklearnSVM:
    def __init__(self, kernel):
        self.svr = SVR(kernel)

    def training(self, in_data, labels):
        x_train, x_test, y_train, y_test = cross_validation.train_test_split(in_data, labels,
                                                                             test_size=0.3, random_state=0)
        self.svr.fit(x_train, y_train)
        self.svr.score(x_test, y_test)

    def predict(self, in_data):
        return self.svr.predict(in_data)