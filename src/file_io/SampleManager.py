import pandas as pd

from src.file_io import ImageOrganizer


__author__ = 'Kern'


def prepare_preprocessing_image(excel_df, preprocessed_path, original_path, file_column_name):
    return ImageOrganizer.prepare_images(excel_df[file_column_name], original_path, preprocessed_path)


def prepare_image_data(excel_df, image_dict, file_column_name, color_column_name, quality_column_name,
                          l2_label_name):
    name_list = [file_column_name, color_column_name, quality_column_name]
    multi_index_list = [[l2_label_name] * len(name_list), name_list]

    return [(image_dict[row[file_column_name]], pd.Series((row[name_list]).tolist(), multi_index_list)) for index, row
            in excel_df.iterrows() if row[file_column_name] in image_dict]