import os

__author__ = 'Kern'


def extract_filename_by_path(path):
    return os.path.split(path)[1]


def save_dataframe(dataframe, filename):
    dataframe.to_csv(filename + '.csv')