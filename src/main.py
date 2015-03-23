import os
import sys

import pandas as pd
from datetime import datetime

from src.file_io import SampleManager
from src.file_io import PredictorManager
from src.file_io import PublicSupport
from src.image_feature import FeatureManager

__author__ = 'Kern'


def feature(image_data):
    # feature extraction
    feats_list = [FeatureManager.compute_feats(img).append(info) for img, info in image_data]
    feat_df = pd.DataFrame(feats_list)
    PublicSupport.save_dataframe(feat_df, os.path.join(feature_data_home, 'features ' + datetime.now().strftime("%Y-%m-%d %H.%M.%S")))

    return feat_df


if len(sys.argv) < 10:
    raise ValueError("Usage:", sys.argv[0], " Missing some argument to indicate input files")

data_home = os.path.abspath(sys.argv[1])
original_data_home = os.path.join(data_home, sys.argv[2])
predict_data_home = os.path.join(data_home, sys.argv[3])
preprocessed_data_home = os.path.join(data_home, sys.argv[4])
feature_data_home = os.path.join(data_home, sys.argv[5])
excel_file_path = os.path.join(original_data_home, sys.argv[6])
excel_sheet_name = sys.argv[7]
file_column_name = sys.argv[8]
color_column_name = sys.argv[9]
quality_column_name = sys.argv[10]


def train():
    # load file and image preprocessing
    excel_dataframe = (pd.read_excel(excel_file_path, excel_sheet_name, index_col=None, na_values=['NA'])).dropna(axis=0)
    image_dict = SampleManager.prepare_preprocessing_image(excel_dataframe, preprocessed_data_home, original_data_home, file_column_name)
    training_data = SampleManager.prepare_training_data(excel_dataframe, image_dict, file_column_name, color_column_name, quality_column_name, 'Subjective')
    feat_df = feature(training_data)


def predict():
    image_dict = PredictorManager.prepare_preprocessing_image(preprocessed_data_home, original_data_home, "*.jpg")
    prediction_data = PredictorManager.prepare_training_data(image_dict, file_column_name, 'Subjective')
    feat_df = feature(prediction_data)


train()