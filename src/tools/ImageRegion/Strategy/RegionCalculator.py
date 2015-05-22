import operator
from src.tools.ImageRegion.Model.Region import merge_regions

__author__ = 'Kern'


def iterate_regions(regions_set):
    pre_merge_regions = {}
    merge_index = 0
    regions_set_copy = regions_set.copy()

    for region in regions_set_copy:
        linked_region = min(region.linked_regions, key=lambda lr: abs(region.eigen - lr.eigen))
        region_index = pre_merge_regions.get(region)
        linked_index = pre_merge_regions.get(linked_region)
        if region_index is None and linked_index:
            pre_merge_regions[region] = linked_index
        elif region_index and linked_index is None:
            pre_merge_regions[linked_region] = region_index
        elif region_index is None and linked_index is None:
            pre_merge_regions[region] = merge_index
            pre_merge_regions[linked_region] = merge_index
            merge_index += 1
        elif region_index != linked_index:
            for pre_region in [region for region, merge_index in pre_merge_regions if
                               merge_index == region_index or merge_index == linked_index]:
                pre_merge_regions[pre_region] = merge_index
            merge_index += 1

    sorted_pre_regions = sorted(pre_merge_regions.items(), key=operator.itemgetter(1))
    first_tuple = next(iter(sorted_pre_regions), None)
    if first_tuple:
        init_index = first_tuple[1]
        merge_list = []
        for region, merge_index in sorted_pre_regions:
            regions_set_copy.discard(region)
            if merge_index == init_index:
                merge_list.append(region)
            else:
                regions_set_copy.add(merge_regions(merge_list))
                init_index = merge_index
                merge_list = [region]

    return regions_set_copy
