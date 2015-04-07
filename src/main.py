import json
import os
import sys

import pandas as pd

from datetime import datetime
from src.file_io import SampleManager
from src.file_io import PredictorManager
from src.file_io import PublicSupport
from src.image_feature import FeatureManager
from src.learning.strategy import RegressionManager

__author__ = 'Kern'


def feature_dimensions(data_struct):
    if len(data_struct.shape) >= 2:
        return data_struct.shape[1]
    elif len(data_struct.shape) == 1:
        return 1

    raise TypeError('unknown pandas struct type')


def feature(images_data, feature_name, feature_data_path):
    # feature extraction
    feats_list = [FeatureManager.compute_feats(img).append(info) for img, info in images_data]
    feat_df = pd.DataFrame(feats_list)
    PublicSupport.save_dataframe(feat_df, os.path.join(feature_data_path, feature_name + datetime.now().strftime("%Y-%m-%d %H.%M.%S")))

    return feat_df


def calc_train_features(original_data_path, preprocessed_dir, excel, sheet_name, feature_data_path):
    excel_dataframe = (pd.read_excel(excel, sheet_name, index_col=None, na_values=['NA'])).dropna(axis=0)
    preprocessed_dir = os.path.join(original_data_path, preprocessed_dir)
    create_path(preprocessed_dir)
    image_dict = SampleManager.prepare_preprocessing_image(excel_dataframe, preprocessed_dir, original_data_path, file_column_name)
    training_data = SampleManager.prepare_training_data(excel_dataframe, image_dict, file_column_name, color_column_name, quality_column_name, subjective_column_name)
    return feature(training_data, 'feature_train', feature_data_path)


def calc_predict_features(predict_data_path, preprocessed_dir, feature_data_path):
    preprocessed_path = os.path.join(predict_data_path, preprocessed_dir)
    create_path(preprocessed_path)
    image_dict = PredictorManager.prepare_preprocessing_image(preprocessed_path, predict_data_path, "*.jpg")
    prediction_data = PredictorManager.prepare_training_data(image_dict, file_column_name, subjective_column_name)
    return feature(prediction_data, 'feature_prediction', feature_data_path)


def load_features(feat_path, pattern):
    return PublicSupport.load_dataframe(PublicSupport.find_newest_file(feat_path, pattern))


def load_models(model_list, model_data_path):
    mng = RegressionManager.RegressionManager(model_list)
    mng.load(model_data_path)

    return mng


def train(original_data_path, preprocessed_dir, excel, sheet_name, feature_data_path, model_data_path):
    feat_df = calc_train_features(original_data_path, preprocessed_dir, excel, sheet_name, feature_data_path)
    # feat_df = load_features(feature_data_home, '*train*.csv')

    x_data = feat_df.drop(subjective_column_name, axis=1, level=0)
    y_data = feat_df[subjective_column_name]

    # simple linear models to train on color score
    color_x = x_data[hue_column_name]
    color_y = y_data[color_column_name]
    color_models = RegressionManager.RegressionManager(RegressionManager.register_color_regression())
    color_models.train(color_x, color_y)
    color_models.save(model_data_path)

    # kinds of models to train on quality score
    quality_x = x_data
    quality_y = y_data[quality_column_name]
    quality_models = RegressionManager.RegressionManager(RegressionManager.register_quality_regression(feature_dimensions(quality_x), feature_dimensions(quality_y)))
    quality_models.train(quality_x, quality_y)
    quality_models.save(model_data_path)

    # modes to train on both color and quality
    mixed_x = x_data
    mixed_y = y_data[[color_column_name, quality_column_name]]
    mixed_models = RegressionManager.RegressionManager(RegressionManager.register_mixed_regression(feature_dimensions(mixed_x), feature_dimensions(mixed_y)))
    mixed_models.train(mixed_x, mixed_y)
    mixed_models.save(model_data_path)

    return color_models, quality_models, mixed_models


def predict(predict_data_path, preprocessed_dir, feature_data_path, color_models, quality_models, mixed_models):
    feat_df = calc_predict_features(predict_data_path, preprocessed_dir, feature_data_path)
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


# color_m = load_models(RegressionManager.register_color_regression())
color_m, quality_m, mixed_m = train(original_data_home, preprocessed_folder, excel_file, excel_sheet_name, feature_data_home, model_data_home)
predict(predict_data_home, preprocessed_folder, feature_data_home, color_m, quality_m, mixed_m)