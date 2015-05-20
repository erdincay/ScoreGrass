import math

__author__ = 'Kern'


def InitRegions(image, initialized_regions_num):
    if len(image.shapre) < 2:
        raise ValueError("input is not an image")

    split_num = math.sqrt(initialized_regions_num)
    row_interval = image.shapre[0] / split_num
    col_interval = image.shapre[1] / split_num

    for row_index in range(split_num):
        for col_index in range(split_num):

            for
