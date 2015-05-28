import math
from src.tools.image_region.model.Line import generate_square_linked_coordinates
from src.tools.image_region.model.Region import Region

__author__ = 'Kern'


def _init_linked_region(regions_map, boundary):
    for coord, region in regions_map.items():
        for linked_coord in {c for c in generate_square_linked_coordinates(coord)
                             if 0 <= c[0] <= boundary[0] and 0 <= c[1] <= boundary[1]}:
            linked_region = regions_map.get(linked_coord)
            if linked_region is not None:
                region.add_neighbor(linked_region)


def init_regions(image, split_number):
    if len(image.shape) < 2:
        raise ValueError("input is not an image")

    if isinstance(split_number, int):
        split_row = split_number
        split_col = split_number
    elif len(split_number) >= 2:
        split_row = split_number[0]
        split_col = split_number[1]
    else:
        raise ValueError("Invalid split number input")

    row_interval = math.ceil(image.shape[0] / split_row)
    col_interval = math.ceil(image.shape[1] / split_col)

    regions_map = {}
    for row_region in range(split_row):
        for col_region in range(split_col):
            coord_map = {}
            for row_coord in range(row_interval):
                for col_coord in range(col_interval):
                    row_index = row_region * row_interval + row_coord
                    col_index = col_region * col_interval + col_coord
                    if 0 <= row_index < image.shape[0] and 0 <= col_index < image.shape[1]:
                        coord_map[(row_index, col_index)] = image[row_index, col_index]
            if coord_map:
                region = Region(coord_map, set())
                regions_map[(row_region, col_region)] = region

    _init_linked_region(regions_map, image.shape)

    return {region for coord, region in regions_map.items()}
