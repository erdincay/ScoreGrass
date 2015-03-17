import os
import pandas as pd
import numpy as np
from skimage import io

from src.image_preprocess import PreprocessingManager


__author__ = 'Kern'

l2_label_name = 'Subjective'


def load_preprocessed(image_collection):
    return {PreprocessingManager.get_file_name(img_path): img for img, img_path in
            zip(image_collection, image_collection.files)}


def prepare_preprocessing_image(excel_df, preprocessed_path, original_path, file_column_name):
    file_list = excel_df[file_column_name]
    filter_mask = file_list.apply(lambda fname: os.path.isfile(os.path.join(preprocessed_path, fname)))
    files_to_preprocess = file_list[np.logical_not(filter_mask)].apply(lambda fname: os.path.join(original_path, fname))

    preprocess_dict = PreprocessingManager.pre_process(
        io.imread_collection(files_to_preprocess.tolist(), conserve_memory=True))

    for name, img in preprocess_dict.items():
        io.imsave(os.path.join(preprocessed_path, name), img)

    files_has_preprocessed = file_list[filter_mask].apply(lambda fname: os.path.join(preprocessed_path, fname))
    preprocess_dict.update(load_preprocessed(io.imread_collection(files_has_preprocessed.tolist())))

    return preprocess_dict


def prepare_training_data(excel_df, image_dict, file_column_name, color_column_name, quality_column_name):
    name_list = [file_column_name, color_column_name, quality_column_name]
    multi_index_list = [[l2_label_name] * len(name_list), name_list]

    return [(image_dict[row[file_column_name]], pd.Series((row[name_list]).tolist(), multi_index_list))
            for index, row in excel_df.iterrows() if row[file_column_name] in image_dict]


def save_dataframe(dataframe, filename):
    dataframe.to_csv(filename + '.csv')