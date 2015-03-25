import os
import sys

import pandas as pd
from datetime import datetime

from src.file_io import SampleManager
from src.file_io import PredictorManager
from src.file_io import PublicSupport
from src.image_feature import FeatureManager
from src.learning.evaluation import CrossValidation
from src.learning.preprocess import Preprocessor

__author__ = 'Kern'


def feature(image_data):
    # feature extraction
    feats_list = [FeatureManager.compute_feats(img).append(info) for img, info in image_data]
    feat_df = pd.DataFrame(feats_list)
    PublicSupport.save_dataframe(feat_df, os.path.join(feature_data_home, 'features ' + datetime.now().strftime("%Y-%m-%d %H.%M.%S")))

    return feat_df


if len(sys.argv) < 12:
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
subjective_column_name = sys.argv[11]


def train():
    # load file and image preprocessing
    excel_dataframe = (pd.read_excel(excel_file_path, excel_sheet_name, index_col=None, na_values=['NA'])).dropna(axis=0)
    image_dict = SampleManager.prepare_preprocessing_image(excel_dataframe, preprocessed_data_home, original_data_home, file_column_name)
    training_data = SampleManager.prepare_training_data(excel_dataframe, image_dict, file_column_name, color_column_name, quality_column_name, subjective_column_name)
    feat_df = feature(training_data)

    x_data = feat_df.drop(subjective_column_name, axis=1)
    y_data = feat_df[subjective_column_name]

    x_train, x_test, y_train, y_test = CrossValidation.data_set_split(0.3)(x_data, y_data)
    scalar = Preprocessor.Standardization(x_train)
    x_train_scaled = scalar.transform(x_train)
    x_test_scaled = scalar.transform(x_test)



    return scalar


def predict(scalar):
    image_dict = PredictorManager.prepare_preprocessing_image(preprocessed_data_home, original_data_home, "*.jpg")
    prediction_data = PredictorManager.prepare_training_data(image_dict, file_column_name, subjective_column_name)
    feat_df = feature(prediction_data)

    x_data = feat_df.drop(subjective_column_name, axis=1)
    y_data = feat_df[subjective_column_name]

    x_data_scaled = scalar.transform(x_data)


scalar = train()
predict(scalar)