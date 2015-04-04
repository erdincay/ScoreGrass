import os
import sys
from datetime import datetime

import pandas as pd

from src.file_io import SampleManager
from src.file_io import PredictorManager
from src.file_io import PublicSupport
from src.image_feature import FeatureManager
from src.learning.strategy import ColorRegression, MixedRegression, QualityRegression


__author__ = 'Kern'

if len(sys.argv) < 13:
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
hue_column_name = sys.argv[12]


def feature(images_data):
    # feature extraction
    feats_list = [FeatureManager.compute_feats(img).append(info) for img, info in images_data]
    feat_df = pd.DataFrame(feats_list)
    PublicSupport.save_dataframe(feat_df, os.path.join(feature_data_home, 'features ' + datetime.now().strftime("%Y-%m-%d %H.%M.%S")))

    return feat_df


def train():
    # load file and image preprocessing
    excel_dataframe = (pd.read_excel(excel_file_path, excel_sheet_name, index_col=None, na_values=['NA'])).dropna(axis=0)
    image_dict = SampleManager.prepare_preprocessing_image(excel_dataframe, preprocessed_data_home, original_data_home, file_column_name)
    training_data = SampleManager.prepare_training_data(excel_dataframe, image_dict, file_column_name, color_column_name, quality_column_name, subjective_column_name)
    feat_df = feature(training_data)

    x_data = feat_df.drop(subjective_column_name, axis=1)
    y_data = feat_df[subjective_column_name]

    # simple linear models to train on color score
    color_models = ColorRegression.ColorRegression()
    color_models.train(x_data[hue_column_name], y_data[subjective_column_name, color_column_name])

    # kinds of models to train on quality score
    quality_x = x_data
    quality_y = y_data[subjective_column_name, quality_column_name]
    quality_models = QualityRegression.QualityRegression(len(quality_x.columns), len(quality_y.columns))
    quality_models.train(quality_x, quality_y)

    # modes to train on both color and quality
    mixed_x = x_data
    mixed_y = y_data[subjective_column_name][color_column_name, quality_column_name]
    mixed_models = MixedRegression.MixedRegression(len(mixed_x.columns), len(mixed_y.columns))
    mixed_models.train(mixed_x, mixed_y)

    return color_models, quality_models, mixed_models


def predict(color_models, quality_models, mixed_models):
    image_dict = PredictorManager.prepare_preprocessing_image(preprocessed_data_home, original_data_home, "*.jpg")
    prediction_data = PredictorManager.prepare_training_data(image_dict, file_column_name, subjective_column_name)
    feat_df = feature(prediction_data)

    x_data = feat_df.drop(subjective_column_name, axis=1)
    y_data = feat_df[subjective_column_name]

    color_models.predict(x_data)
    quality_models.predict(x_data)
    mixed_models.predict(x_data)

predict(*train())