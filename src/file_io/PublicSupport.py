import fnmatch
import os
import pandas as pd

__author__ = 'Kern'


def extract_filename_by_path(path):
    return os.path.split(path)[1]


def save_dataframe(dataframe, filename):
    dataframe.to_csv(filename + '.csv')


def load_dataframe(filename):
    if filename and os.path.isfile(filename):
        return pd.read_csv(filename, header=[0, 1], index_col=0)

    return None


def find_newest_file(filepath, pattern):
    file_list = [os.path.join(filepath, file) for file in os.listdir(filepath) if fnmatch.fnmatch(file, pattern)]
    if file_list:
        return max(file_list, key=os.path.getctime)

    return None