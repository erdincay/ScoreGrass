import fnmatch
import os
import pandas as pd

__author__ = 'Kern'


def extract_filename_by_path(path):
    return os.path.split(path)[1]


def save_dataframe(dataframe, filename):
    dataframe.to_csv(filename + '.csv')


def load_dataframe(filename):
    return pd.read_csv(filename, header=[0, 1])


def find_newest_file(filepath):
    return max([os.path.join(filepath, file) for file in os.listdir(filepath) if fnmatch.fnmatch(file, '*.csv')], key=os.path.getctime)