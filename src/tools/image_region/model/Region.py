import numpy as np

from src.tools.DictionaryExtension import merge_two_dicts
from src.tools.image_region.model.Eigen import Eigen

__author__ = 'Kern'


def merge_two_regions(region1, region2):
    coord_map_new = merge_two_dicts(region1.coord_map, region2.coord_map)

    linked_regions_new = region1.linked_regions | region2.linked_regions
    linked_regions_new.discard(region1)
    linked_regions_new.discard(region2)

    eigen_value_new = (region1.get_eigen() * len(region1.coord_map) / len(coord_map_new)) + (
        region2.get_eigen() * len(region2.coord_map) / len(coord_map_new))

    return Region(coord_map_new, linked_regions_new, eigen_value_new)


def merge_regions(regions):
    if not regions:
        raise ValueError("cannot merge empty region set", regions)

    coord_map_new = dict()
    linked_regions_new = set()
    eigen_value_new = np.zeros(len(next(iter(regions)).get_eigen()))

    count_coords = 0
    for region in regions:
        count_coords += len(region.coord_map)
        coord_map_new.update(region.coord_map)
        linked_regions_new |= region.linked_regions

    assert count_coords == len(coord_map_new)

    for region in regions:
        eigen_value_new += region.get_eigen() * len(region.coord_map) / len(coord_map_new)
        linked_regions_new.discard(region)

    return Region(coord_map_new, linked_regions_new, eigen_value_new)


class Region:
    def __init__(self, coord_map, linked_regions, eigen_val=None):
        self.coord_map = coord_map

        linked_regions.discard(self)
        self.linked_regions = linked_regions

        self.eigen = Eigen(self.coord_map, eigen_val)

    def __str__(self):
        return "{id: " + format(id(self), '#018X') + ", coord_map: " + str(len(self.coord_map)) + ", linked: " + str(len(self.linked_regions)) + \
               ", eigen: " + str(self.get_eigen()) + "}"

    def merge(self, oth_region):
        return merge_two_regions(self, oth_region)

    def add_neighbor(self, region):
        if self is not region:
            self.linked_regions.add(region)

    def remove_neighbor(self, region):
        self.linked_regions.remove(region)

    def discard_neighbor(self, region):
        self.linked_regions.discard(region)

    def get_eigen(self):
        return self.eigen.get_eigen()

    def get_greeness(self):
        return self.get_eigen()[1]

    def get_hue(self):
        return self.get_eigen()[0]

    def get_saturation(self):
        return self.get_eigen()[1]
