from src.tools.DictionaryExtension import merge_two_dicts

__author__ = 'Kern'


def merge_two_regions(region1, region2):
    coord_map_new = merge_two_dicts(region1.coord_map, region2.coord_map)

    linked_regions_new = region1.linked_regions | region2.linked_regions
    linked_regions_new.discard(region1)
    linked_regions_new.discard(region2)

    return Region(coord_map_new, linked_regions_new)


class Region:
    def __init__(self, coord_map, linked_regions):
        self.coord_map = coord_map
        self.linked_regions = linked_regions

    def merge(self, oth_region):
        return merge_two_regions(self, oth_region)
