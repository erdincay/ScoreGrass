import math
from src.tools.ImageRegion.Model.Region import Region

__author__ = 'Kern'

def _init_linked_region(regions_map, boundary):
    for coord, region in regions_map.items():
        for linked_coord in {c for c in _generate_square_linked_coordinates(coord)
                             if 0 <= c[0] <= boundary[0] and 0 <= c[1] <= boundary[1]}:
            linked_region = regions_map.get(linked_coord)
            if linked_region is not None:
                region.add_neighbor(linked_region)


def _generate_square_linked_coordinates(coord):
    row_list = [coord[0] - 1, coord[0], coord[0] + 1]
    col_list = [coord[1] - 1, coord[1], coord[1] + 1]
    return {(row, col) for row in row_list for col in col_list if row != coord[0] or col != coord[1]}

def init_regions(image, initialized_regions_num):
    if len(image.shape) < 2:
        raise ValueError("input is not an image")

    split_num = math.floor(math.sqrt(initialized_regions_num))
    row_interval = math.ceil(image.shape[0] / split_num)
    col_interval = math.ceil(image.shape[1] / split_num)

    regions_map = {}
    for row_region in range(split_num):
        for col_region in range(split_num):
            coord_map = {}
            for row_coord in range(row_interval):
                for col_coord in range(col_interval):
                    row_index = row_region * row_interval + row_coord
                    col_index = col_region * col_interval + col_coord
                    if 0 <= row_index < image.shape[0] and 0 <= col_index < image.shape[1]:
                        coord_map[(row_index, col_index)] = image[row_index, col_index]
            region = Region(coord_map, set())
            regions_map[(row_region, col_region)] = region

    _init_linked_region(regions_map, image.shape)

    return {region for coord, region in regions_map.items()}
