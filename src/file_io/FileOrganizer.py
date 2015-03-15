import os

import numpy as np
from skimage import io

from src.image_preprocess import PreprocessingManager


__author__ = 'Kern'


def load_preprocessed(image_collection):
    ret = {}
    for img, img_path in zip(image_collection, image_collection.files):
        name = PreprocessingManager.get_file_name(img_path)
        ret[name] = img
    return ret


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
    ret = []
    for index, row in excel_df.iterrows():
        file_name = row[file_column_name]
        if file_name in image_dict:
            ret.append((image_dict[file_name], row[[file_column_name, color_column_name, quality_column_name]]))
    return ret


def save_dataframe(dataframe, filename):
    # dataframe.reset_index().to_json(filename + '.json')
    dataframe.to_csv(filename + '.csv')