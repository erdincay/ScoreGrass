import os
import sys
from datetime import datetime

import pandas as pd

from src.file_io import SampleManager
from src.file_io import PredictorManager
from src.file_io import PublicSupport
from src.image_feature import FeatureManager
from src.learning.strategy.ColorRegression import ColorRegression
from src.learning.strategy.MixedRegression import MixedRegression
from src.learning.strategy.QualityRegression import QualityRegression


__author__ = 'Kern'


def feature_dimensions(data_struct):
    if len(data_struct.shape) >= 2:
        return data_struct.shape[1]
    elif len(data_struct.shape) == 1:
        return 1

    raise TypeError('unknown struct type')


def __calc_features(images_data, feature_name, feature_data_path):
    # feature extraction
    feat_df = pd.DataFrame([FeatureManager.compute_feats(img).append(info) for img, info in images_data])
    PublicSupport.save_dataframe(feat_df, os.path.join(feature_data_path,
                                                       feature_name + datetime.now().strftime("%Y-%m-%d %H.%M.%S")))

    return feat_df


def _calc_train_features(original_data_path, preprocessed_dir, excel, sheet_name, feature_data_path):
    excel_dataframe = (pd.read_excel(excel, sheet_name, index_col=None, na_values=['NA'])).dropna(axis=0)
    preprocessed_dir = os.path.join(original_data_path, preprocessed_dir)
    PublicSupport.create_path(preprocessed_dir)
    image_dict = SampleManager.prepare_preprocessing_image(excel_dataframe, preprocessed_dir, original_data_path,
                                                           file_column_name)
    training_data = SampleManager.prepare_image_data(excel_dataframe, image_dict, file_column_name, color_column_name,
                                                     quality_column_name, subjective_column_name)
    return __calc_features(training_data, 'feature_train', feature_data_path)


def _calc_predict_features(predict_data_path, preprocessed_dir, feature_data_path):
    preprocessed_path = os.path.join(predict_data_path, preprocessed_dir)
    PublicSupport.create_path(preprocessed_path)
    image_dict = PredictorManager.prepare_preprocessing_image(preprocessed_path, predict_data_path, "*.jpg")
    prediction_data = PredictorManager.prepare_image_data(image_dict, file_column_name, subjective_column_name)
    return __calc_features(prediction_data, 'feature_prediction', feature_data_path)


def _load_features(feat_path, pattern):
    return PublicSupport.load_dataframe(PublicSupport.find_newest_file(feat_path, pattern))


def load_models(model_data_path):
    return ColorRegression.deserialize_regression(model_data_path), QualityRegression.deserialize_regression(
        model_data_path), MixedRegression.deserialize_regression(model_data_path)


def train(original_data_path, preprocessed_dir, excel, sheet_name, feature_data_path, model_data_path,
          output_result_path):
    if b_load_train_feat:
        feat_df = _load_features(feature_data_home, '*train*.csv')
    else:
        feat_df = _calc_train_features(original_data_path, preprocessed_dir, excel, sheet_name, feature_data_path)

    x_data = feat_df.drop(subjective_column_name, axis=1, level=0)
    y_data = feat_df[subjective_column_name]

    # simple linear models to train on color score
    color_x = x_data[hue_column_name]
    color_y = y_data[color_column_name]
    color_models = ColorRegression(feature_dimensions(color_x), feature_dimensions(color_y))
    model_score_dict = color_models.validation(color_x, color_y, 0.25)
    color_models.save(model_data_path)

    # kinds of models to train on quality score
    quality_x = x_data
    quality_y = y_data[quality_column_name]
    quality_models = QualityRegression(feature_dimensions(quality_x), feature_dimensions(quality_y))
    model_score_dict.update(quality_models.validation(quality_x, quality_y, 0.25))
    quality_models.save(model_data_path)

    # modes to train on both color and quality
    mixed_x = x_data
    mixed_y = y_data[[color_column_name, quality_column_name]]
    mixed_models = MixedRegression(feature_dimensions(mixed_x), feature_dimensions(mixed_y))
    model_score_dict.update(mixed_models.validation(mixed_x, mixed_y, 0.25))
    mixed_models.save(model_data_path)

    # store cross_validation scores
    PublicSupport.write_json(model_score_dict, os.path.join(output_result_path, 'model_score' +
                                                            datetime.now().strftime("%Y-%m-%d %H.%M.%S") + '.json'))

    return color_models, quality_models, mixed_models


def predict(predict_data_path, preprocessed_dir, feature_data_path, color_models, quality_models, mixed_models,
            output_result_path):
    if b_load_predict_feat:
        feat_df = _load_features(feature_data_home, '*prediction*.csv')
    else:
        feat_df = _calc_predict_features(predict_data_path, preprocessed_dir, feature_data_path)

    x_data = feat_df.drop(subjective_column_name, axis=1, level=0)

    color_df = pd.DataFrame(color_models.predict(x_data[hue_column_name]))
    quality_df = pd.DataFrame(quality_models.predict(x_data))
    mixed_df = pd.DataFrame(mixed_models.predict(x_data))

    all_df = pd.concat([color_df, quality_df, mixed_df, feat_df[subjective_column_name]], axis=1,
                       keys=[color_models.__class__.__name__, quality_models.__class__.__name__,
                             mixed_models.__class__.__name__, subjective_column_name])

    # store prediction result
    PublicSupport.save_dataframe(all_df, os.path.join(output_result_path,
                                                      'prediction' + 'model_score' + datetime.now().strftime(
                                                          "%Y-%m-%d %H.%M.%S")))


if len(sys.argv) < 2:
    raise ValueError("Usage:", sys.argv[0], " Missing some argument to indicate input files")

json_dict = PublicSupport.read_json(sys.argv[1])

# input folder struct
data_home = os.path.abspath(json_dict['data_home'])
PublicSupport.create_path(data_home)
original_data_home = os.path.join(data_home, json_dict['original_data_home'])
PublicSupport.create_path(original_data_home)
predict_data_home = os.path.join(data_home, json_dict['predict_data_home'])
PublicSupport.create_path(predict_data_home)
feature_data_home = os.path.join(data_home, json_dict['feature_data_home'])
PublicSupport.create_path(feature_data_home)
model_data_home = os.path.join(data_home, json_dict['model_data_home'])
PublicSupport.create_path(model_data_home)
output_result_home = os.path.join(data_home, json_dict['output_result_home'])
PublicSupport.create_path(output_result_home)
preprocessed_folder = json_dict['preprocessed_folder']

# input excel file for subjective score
excel_file = os.path.join(original_data_home, json_dict['excel_file_path'])

# input table name, which is used globally in this file
excel_sheet_name = json_dict['excel_sheet_name']
file_column_name = json_dict['file_column_name']
color_column_name = json_dict['color_column_name']
quality_column_name = json_dict['quality_column_name']
subjective_column_name = json_dict['subjective_column_name']
hue_column_name = json_dict['hue_column_name']

# loading vs calculation options
b_load_model = json_dict['b_load_model']
b_load_train_feat = json_dict['b_load_train_feat']
b_load_predict_feat = json_dict['b_load_predict_feat']

if b_load_model:
    color_m, quality_m, mixed_m = load_models(model_data_home)
else:
    color_m, quality_m, mixed_m = train(original_data_home, preprocessed_folder, excel_file, excel_sheet_name,
                                        feature_data_home, model_data_home, output_result_home)
predict(predict_data_home, preprocessed_folder, feature_data_home, color_m, quality_m, mixed_m, output_result_home)