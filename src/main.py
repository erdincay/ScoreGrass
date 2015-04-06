import os
import sys
from datetime import datetime

import pandas as pd

from src.file_io import SampleManager
from src.file_io import PredictorManager
from src.file_io import PublicSupport
from src.image_feature import FeatureManager
from src.learning.strategy import MixedRegression
from src.learning.strategy import QualityRegression
from src.learning.strategy import ColorRegression
from src.learning.strategy.RegressionManager import feature_dimensions

__author__ = 'Kern'

if len(sys.argv) < 14:
    raise ValueError("Usage:", sys.argv[0], " Missing some argument to indicate input files")

data_home = os.path.abspath(sys.argv[1])
original_data_home = os.path.join(data_home, sys.argv[2])
predict_data_home = os.path.join(data_home, sys.argv[3])
preprocessed_folder = sys.argv[4]
feature_data_home = os.path.join(data_home, sys.argv[5])
model_data_home = os.path.join(data_home, sys.argv[6])
excel_file_path = os.path.join(original_data_home, sys.argv[7])
excel_sheet_name = sys.argv[8]
file_column_name = sys.argv[9]
color_column_name = sys.argv[10]
quality_column_name = sys.argv[11]
subjective_column_name = sys.argv[12]
hue_column_name = sys.argv[13]


def feature(images_data, feature_name):
    # feature extraction
    feats_list = [FeatureManager.compute_feats(img).append(info) for img, info in images_data]
    feat_df = pd.DataFrame(feats_list)
    PublicSupport.save_dataframe(feat_df, os.path.join(feature_data_home, feature_name + datetime.now().strftime("%Y-%m-%d %H.%M.%S")))

    return feat_df


def calc_features(b_training):
    if b_training:
        # load file and image preprocessing
        excel_dataframe = (pd.read_excel(excel_file_path, excel_sheet_name, index_col=None, na_values=['NA'])).dropna(axis=0)
        image_dict = SampleManager.prepare_preprocessing_image(excel_dataframe, os.path.join(original_data_home, preprocessed_folder), original_data_home, file_column_name)
        training_data = SampleManager.prepare_training_data(excel_dataframe, image_dict, file_column_name, color_column_name, quality_column_name, subjective_column_name)
        feat_df = feature(training_data, 'feature_train')
    else:
        # load file and image preprocessing
        image_dict = PredictorManager.prepare_preprocessing_image(os.path.join(predict_data_home, preprocessed_folder), predict_data_home, "*.jpg")
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

predict(*train())