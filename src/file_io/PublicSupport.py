import fnmatch
import json
import os
import pandas as pd

__author__ = 'Kern'


def extract_filename_by_path(path):
    return os.path.split(path)[1]


def extract_dir_by_path(path):
    return os.path.split(path)[0]


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


def read_json(path):
    with open(path) as json_stream:
        json_dict = json.load(json_stream)
    json_stream.close()

    return json_dict


def write_json(content, path):
    with open(path, 'w') as output_json:
        json.dump(content, output_json)
    output_json.close()


def create_path(path):
    if not os.path.isdir(path):
        os.mkdir(path)