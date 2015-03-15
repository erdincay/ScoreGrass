from enum import Enum

from src.image_preprocess import EdgeDetector
from src.image_preprocess import PeakDetector
from src.image_preprocess import Cropper


__author__ = 'Kern'

crop_by_dot = 'plot'
crop_by_frame = 'patch'


class CropType(Enum):
    dot = 1,
    frame = 2,
    none = 3


def get_file_name(path):
    return str.split(path, '\\')[-1]


def crop_image_by_pink_dot(image):
    distance_mark = PeakDetector.generate_pink_map(image)
    coords_mark = PeakDetector.peak_corner_detector(distance_mark, 0.825, 80)
    return Cropper.diagonal_cropping(image, coords_mark, Cropper.extreme_value)


def crop_image_by_frame(image):
    binary_image = EdgeDetector.binarize(image)
    edges_map = EdgeDetector.canny_edge_detector(binary_image, 4.5)
    contours = EdgeDetector.find_edges(edges_map)
    coords_frame = EdgeDetector.edge_coordinate(contours)
    return Cropper.diagonal_cropping(image, coords_frame, Cropper.extreme_value, 100)


def crop_image(image, crop_type):
    if crop_type == CropType.dot:
        cropped = crop_image_by_pink_dot(image)
    elif crop_type == CropType.frame:
        cropped = crop_image_by_frame(image)
    else:
        raise TypeError('unknown cropped method, crop_type = ', crop_type.name)

    return cropped


def map_crop_type(path):
    name = get_file_name(path)
    crop_type = CropType.none
    if crop_by_dot in name.lower():
        crop_type = CropType.dot
    elif crop_by_frame in name.lower():
        crop_type = CropType.frame

    return name, crop_type


def pre_process(image_collection):
    ret = {}
    for img, img_path in zip(image_collection, image_collection.files):
        name, crop_type = map_crop_type(img_path)
        cropped_img = crop_image(img, crop_type)
        # ret[name] = transform.resize(cropped_img, (1152, 864))
        ret[name] = cropped_img

    return ret



