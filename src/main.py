import os
import sys

import pandas as pd
from datetime import datetime

from src.file_io import FileOrganizer
from src.image_feature import FeatureManager

__author__ = 'Kern'

if len(sys.argv) < 10:
    raise ValueError("Usage:", sys.argv[0], " Missing some argument to indicate input files")

data_home = os.path.abspath(sys.argv[1])
original_data_home = os.path.join(data_home, sys.argv[2])
preprocessed_data_home = os.path.join(data_home, sys.argv[3])
feature_data_home = os.path.join(data_home, sys.argv[4])
excel_file_path = os.path.join(original_data_home, sys.argv[5])
excel_sheet_name = sys.argv[6]
file_column_name = sys.argv[7]
color_column_name = sys.argv[8]
quality_column_name = sys.argv[9]

# load file and image preprocessing
excel_dataframe = (pd.read_excel(excel_file_path, excel_sheet_name, index_col=None, na_values=['NA'])).dropna(axis=0)
image_dict = FileOrganizer.prepare_preprocessing_image(excel_dataframe, preprocessed_data_home, original_data_home,
                                                       file_column_name)
training_data = FileOrganizer.prepare_training_data(excel_dataframe, image_dict, file_column_name, color_column_name,
                                                    quality_column_name)

# feature extraction
feats_list = []
for img, info in training_data:
    feats = FeatureManager.compute_feats(img)
    feats = feats.append(info)
    feats_list.append(feats)

feat_df = pd.DataFrame(feats_list)
FileOrganizer.save_dataframe(feat_df, os.path.join(feature_data_home,
                                                   'features ' + datetime.strftime(datetime.now().date(), '%Y-%m-%d')))