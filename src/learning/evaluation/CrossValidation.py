import numpy as np
from sklearn import cross_validation

__author__ = 'Kern'


def train_test_split(factor):
    """
    split the data set by factor
    :param factor: test part size
    :return: function that accept all samples
    """

    def function(x_data, y_data):
        return cross_validation.train_test_split(x_data, y_data, test_size=factor, random_state=0)

    return function


def bootstrop(factor, n_iter):
    """
    bootstrop split data set by factor and iteration times
    :param factor: test part size
    :param n_iter: iteration times
    :return: function that accept all samples
    """

    def function(x_data, y_data):
        return np.array([[x_data[train_index], y_data[test_index]] for train_index, test_index in
                         cross_validation.Bootstrap(len(x_data), n_iter=n_iter, test_size=factor, random_state=0)])

    return function


def data_set_split(factor):
    """
    pybrain data set split method
    :param factor: test part size
    :return: function that accept all samples
    """

    def function(data_set):
        return data_set.splitWithProportion(1-factor)

    return function