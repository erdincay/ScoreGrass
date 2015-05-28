import fnmatch
import os

import pandas as pd

from src.file_io import ImageOrganizer

__author__ = 'Kern'


def prepare_preprocessing_image(preprocessed_path, original_path, pattern):
    filename_list = [filename for filename in os.listdir(original_path) if fnmatch.fnmatch(filename, pattern)]
    return ImageOrganizer.prepare_images(filename_list, original_path, preprocessed_path)


def prepare_image_data(image_dict, file_column_name, l2_label_name):
    name_list = [file_column_name]
    multi_index_list = [[l2_label_name] * len(name_list), name_list]

    return [(img, pd.Series([name], multi_index_list)) for name, img in image_dict.items()]
