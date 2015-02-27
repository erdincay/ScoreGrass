__author__ = 'Kern'
import numpy as np
from sklearn import cross_validation


def train_test_split(factor):
    def function(x_data, y_data):
        cross_validation.train_test_split(x_data, y_data, test_size=factor, random_state=0)

    return function


def bootstrop(factor, n_iter):
    def function(x_data, y_data):
        return np.array([[x_data[train_index], y_data[test_index]] for train_index, test_index in
                         cross_validation.Bootstrap(len(x_data), n_iter=n_iter, test_size=factor, random_state=0)])

    return function
