import json
import os
import sys

import pandas as pd

from datetime import datetime
from src.file_io import SampleManager
from src.file_io import PredictorManager
from src.file_io import PublicSupport
from src.image_feature import FeatureManager
from src.learning.strategy import MixedRegression
from src.learning.strategy import QualityRegression
from src.learning.strategy import ColorRegression
from src.learning.strategy.RegressionManager import feature_dimensions


__author__ = 'Kern'


def feature(images_data, feature_name):
    # feature extraction
    feats_list = [FeatureManager.compute_feats(img).append(info) for img, info in images_data]
    feat_df = pd.DataFrame(feats_list)
    PublicSupport.save_dataframe(feat_df, os.path.join(feature_data_home, feature_name + datetime.now().strftime("%Y-%m-%d %H.%M.%S")))

    return feat_df


def calc_features(b_training):
    if b_training:
        # load file and image preprocessing
        excel_dataframe = (pd.read_excel(excel_file, excel_sheet_name, index_col=None, na_values=['NA'])).dropna(axis=0)
        preprocessed_path = os.path.join(original_data_home, preprocessed_folder)
        create_path(preprocessed_path)
        image_dict = SampleManager.prepare_preprocessing_image(excel_dataframe, preprocessed_path, original_data_home, file_column_name)
        training_data = SampleManager.prepare_training_data(excel_dataframe, image_dict, file_column_name, color_column_name, quality_column_name, subjective_column_name)
        feat_df = feature(training_data, 'feature_train')
    else:
        # load file and image preprocessing
        preprocessed_path = os.path.join(predict_data_home, preprocessed_folder)
        create_path(preprocessed_path)
        image_dict = PredictorManager.prepare_preprocessing_image(preprocessed_path, predict_data_home, "*.jpg")
        prediction_data = PredictorManager.prepare_training_data(image_dict, file_column_name, subjective_column_name)
        feat_df = feature(prediction_data, 'feature_prediction')

    return feat_df


def load_features(feat_path, pattern):
    return PublicSupport.load_dataframe(PublicSupport.find_newest_file(feat_path, pattern))


def train():
    # feat_df = calc_features(True)
    feat_df = load_features(feature_data_home, '*train*.csv')

    x_data = feat_df.drop(subjective_column_name, axis=1, level=0)
    y_data = feat_df[subjective_column_name]

    # simple linear models to train on color score
    color_x = x_data[hue_column_name]
    color_y = y_data[color_column_name]
    color_models = ColorRegression.ColorRegression(model_data_home)
    color_models.train(color_x, color_y)

    # kinds of models to train on quality score
    quality_x = x_data
    quality_y = y_data[quality_column_name]
    quality_models = QualityRegression.QualityRegression(feature_dimensions(quality_x), feature_dimensions(quality_y), model_data_home)
    quality_models.train(quality_x, quality_y)

    # modes to train on both color and quality
    mixed_x = x_data
    mixed_y = y_data[[color_column_name, quality_column_name]]
    mixed_models = MixedRegression.MixedRegression(feature_dimensions(mixed_x), feature_dimensions(mixed_y), model_data_home)
    mixed_models.train(mixed_x, mixed_y)

    return color_models, quality_models, mixed_models


def predict(color_models, quality_models, mixed_models):
    feat_df = calc_features(False)
    # feat_df = load_features(feature_data_home, '*prediction*.csv')

    x_data = feat_df.drop(subjective_column_name, axis=1, level=0)

    print(color_models.predict(x_data[hue_column_name]))
    print(quality_models.predict(x_data))
    print(mixed_models.predict(x_data))


def create_path(path):
    if not os.path.isdir(path):
        os.mkdir(path)


if len(sys.argv) < 2:
    raise ValueError("Usage:", sys.argv[0], " Missing some argument to indicate input files")

with open(sys.argv[1]) as json_stream:
    json_dict = json.load(json_stream)
json_stream.close()

# folder struct
data_home = os.path.abspath(json_dict['data_home'])
create_path(data_home)
original_data_home = os.path.join(data_home, json_dict['original_data_home'])
create_path(original_data_home)
predict_data_home = os.path.join(data_home, json_dict['predict_data_home'])
create_path(predict_data_home)
feature_data_home = os.path.join(data_home, json_dict['feature_data_home'])
create_path(feature_data_home)
model_data_home = os.path.join(data_home, json_dict['model_data_home'])
create_path(model_data_home)
preprocessed_folder = json_dict['preprocessed_folder']

# excel file for subjective score
excel_file = os.path.join(original_data_home, json_dict['excel_file_path'])

# table name
excel_sheet_name = json_dict['excel_sheet_name']
file_column_name = json_dict['file_column_name']
color_column_name = json_dict['color_column_name']
quality_column_name = json_dict['quality_column_name']
subjective_column_name = json_dict['subjective_column_name']
hue_column_name = json_dict['hue_column_name']

predict(*train())