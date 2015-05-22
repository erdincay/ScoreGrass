from src.tools.DictionaryExtension import merge_two_dicts, merge_dicts

__author__ = 'Kern'


def merge_two_regions(region1, region2):
    coord_map_new = merge_two_dicts(region1.coord_map, region2.coord_map)

    linked_regions_new = region1.linked_regions | region2.linked_regions
    linked_regions_new.discard(region1)
    linked_regions_new.discard(region2)

    eigen_new = (region1.eigen * len(region1.coord_map) / len(coord_map_new)) + (
        region2.eigen * len(region2.coord_map) / len(coord_map_new))

    return Region(coord_map_new, linked_regions_new, eigen_new)


def merge_regions(*regions):
    coord_map_new = dict()
    linked_regions_new = set()
    eigen_new = 0

    for region in regions:
        coord_map_new.update(region.coord_map)
        linked_regions_new |= region.linked_regions
        eigen_new += region.eigen * len(region.coord_map) / len(coord_map_new)

    return Region(coord_map_new, linked_regions_new, eigen_new)


class Region:
    def __init__(self, coord_map, linked_regions, eigen=0):
        self.coord_map = coord_map
        self.linked_regions = linked_regions
        if eigen == 0:
            self.eigen = self.eigen_value(1)
        else:
            self.eigen = eigen

    def merge(self, oth_region):
        return merge_two_regions(self, oth_region)

    def add_linked(self, region):
        self.linked_regions.add(region)

    def eigen_value(self, index):
        color_list = ([color[index] for color in self.coord_map.values()])
        return sum(color_list) / len(color_list)
