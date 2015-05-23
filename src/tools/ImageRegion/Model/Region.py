import logging
from src.tools.DictionaryExtension import merge_two_dicts

__author__ = 'Kern'


def merge_two_regions(region1, region2):
    coord_map_new = merge_two_dicts(region1.coord_map, region2.coord_map)

    linked_regions_new = region1.linked_regions | region2.linked_regions
    linked_regions_new.discard(region1)
    linked_regions_new.discard(region2)

    eigen_new = (region1.eigen * len(region1.coord_map) / len(coord_map_new)) + (
        region2.eigen * len(region2.coord_map) / len(coord_map_new))

    return Region(coord_map_new, linked_regions_new, eigen_new)


def merge_regions(regions):
    coord_map_new = dict()
    linked_regions_new = set()
    eigen_new = 0

    logging.debug("merge_regions: num=" + str(len(regions)))
    for region in regions:
        logging.debug(region)
        coord_map_new.update(region.coord_map)
        linked_regions_new |= region.linked_regions

    for region in regions:
        eigen_new += region.eigen * len(region.coord_map) / len(coord_map_new)
        linked_regions_new.discard(region)

    ret = Region(coord_map_new, linked_regions_new, eigen_new)
    logging.debug("new region: " + str(ret))

    return ret


class Region:
    def __init__(self, coord_map, linked_regions, eigen=0):
        self.coord_map = coord_map

        linked_regions.discard(self)
        self.linked_regions = linked_regions

        if eigen == 0:
            self.eigen = self.eigen_value(1)
        else:
            self.eigen = eigen

    def __str__(self):
        return "{id: " + format(id(self), '#018X') + ", coord_map: " + str(len(self.coord_map)) + ", linked: " + str(len(self.linked_regions)) + \
               ", eigen: " + str(self.eigen) + ", eigen_function: " + str(self.eigen_value(1)) + "}"

    def merge(self, oth_region):
        return merge_two_regions(self, oth_region)

    def add_neighbor(self, region):
        if self is not region:
            self.linked_regions.add(region)

    def remove_neighbor(self, region):
        self.linked_regions.remove(region)

    def discard_neighbor(self, region):
        self.linked_regions.discard(region)

    def eigen_value(self, index):
        color_list = ([color[index] for color in self.coord_map.values()])
        return sum(color_list) / len(color_list)
