import inspect
import os
import sys
import logging

import numpy as np

from datetime import datetime
from skimage import io, img_as_uint
from skimage.color import rgb2hsv

from src.tools.image_region.strategy.EdgeMarker import make_region_edge
from src.tools.image_region.strategy.RegionCalculator import iterate_regions
from src.tools.image_region.strategy.RegionInitializer import init_regions

__author__ = 'Kern'

logging.basicConfig(filename=inspect.getfile(inspect.currentframe()) + '.log', level=logging.DEBUG)


def _logging_regions(regions):
    for r in regions:
        logging.info(r)
    logging.info("total: " + str(len(regions)))


def calc_regions(input_image, split_num, max_eigen_diff, max_merged_num, path):
    regions = init_regions(img_as_uint(rgb2hsv(image)), split_num)

    new_set_len = 0
    old_set_len = len(regions)

    while new_set_len < old_set_len:
        old_set_len = len(regions)
        regions = iterate_regions(regions, max_eigen_diff, max_merged_num)
        new_set_len = len(regions)

        _logging_regions(regions)

    output_image = input_image
    for index, region in enumerate(regions):
        edge = make_region_edge(region)
        output_image = edge.draw(output_image, [255, 255, 255])

    io.imsave(os.path.join(path, str(split_num) + '-' + "{:10.2f}".format(max_eigen_diff) + '-' + str(max_merged_num) + '-.jpg'), output_image)


if len(sys.argv) < 3:
    raise ValueError("Usage:", sys.argv[0], " Missing some argument to indicate input files")

input_file = sys.argv[1]
output_path = sys.argv[2]

current_output_path = os.path.join(output_path, datetime.now().strftime("%Y-%m-%d %H_%M_%S"))
if not os.path.exists(current_output_path):
    os.makedirs(current_output_path)

readret = io.imread(input_file)
if len(readret.shape) == 3:
    image = readret
elif len(readret.shape) == 1:
    image = readret[0]
else:
    raise ValueError("cannot read the input image: " + input_file)

io.imsave(os.path.join(current_output_path, "original.jpg"), image)


calc_regions(np.copy(image), 200, 80, 1, current_output_path)
