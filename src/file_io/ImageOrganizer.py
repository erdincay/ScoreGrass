import os

from skimage import io

from src.file_io import PublicSupport
from src.image_preprocess import PreprocessingManager

__author__ = 'Kern'

l2_label_name = 'Subjective'


def load_images(image_collection):
    return {
        PublicSupport.extract_filename_by_path(img_path): img for img, img_path in
        zip(image_collection, image_collection.files)
        }


def prepare_images(filename_list, original_path, preprocessed_path):
    # filter the images that is already preprocessed
    need_preprocessing_list = [os.path.join(original_path, filename) for filename in filename_list if
                               not os.path.isfile(os.path.join(preprocessed_path, filename))]
    image_dict = PreprocessingManager.pre_process(io.imread_collection(need_preprocessing_list, conserve_memory=True))

    # save the preprocessed images
    for name, img in image_dict.items():
        io.imsave(os.path.join(preprocessed_path, name), img)

    # load the preprocessed images
    has_preprocessed_list = [os.path.join(original_path, filename) for filename in filename_list if
                             os.path.isfile(os.path.join(preprocessed_path, filename))]
    image_dict.update(load_images(io.imread_collection(has_preprocessed_list, conserve_memory=True)))

    return image_dict
