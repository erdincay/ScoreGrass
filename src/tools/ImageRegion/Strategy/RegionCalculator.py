import logging
import operator
from src.tools.ImageRegion.Model.Region import merge_regions

__author__ = 'Kern'


def iterate_regions(regions_set, max_eigen_diff):
    merge_index = 0
    pre_merge_regions_map = dict()
    regions_set_copy = regions_set.copy()

    for region in regions_set_copy:
        if region.linked_regions:
            closest_regions = min(region.linked_regions, key=lambda lr: abs(region.eigen - lr.eigen))
            if abs(region.eigen - closest_regions.eigen) <= max_eigen_diff:
                region_index = pre_merge_regions_map.get(region)
                linked_index = pre_merge_regions_map.get(closest_regions)
                if region_index is None and linked_index:
                    pre_merge_regions_map[region] = linked_index
                elif region_index and linked_index is None:
                    pre_merge_regions_map[closest_regions] = region_index
                elif region_index is None and linked_index is None:
                    pre_merge_regions_map[region] = merge_index
                    pre_merge_regions_map[closest_regions] = merge_index
                    merge_index += 1
                elif region_index != linked_index:
                    for pre_region in [region for region, merge_index in pre_merge_regions_map.items() if
                                       merge_index == region_index or merge_index == linked_index]:
                        pre_merge_regions_map[pre_region] = merge_index
                    merge_index += 1

    sorted_pre_regions = sorted(pre_merge_regions_map.items(), key=operator.itemgetter(1))

    logging.debug(str(len(sorted_pre_regions)))
    for region, index in sorted_pre_regions:
        logging.debug("index=" + str(index) + ": " + str(region))

    first_tuple = next(iter(sorted_pre_regions), None)
    if first_tuple:
        init_index = first_tuple[1]
        prepared_merge_set = set()
        old_region_to_new_region_map = dict()
        new_regions_set = set()

        for region, merge_index in sorted_pre_regions:
            regions_set_copy.remove(region)
            if merge_index == init_index:
                prepared_merge_set.append(region)
            else:
                new_region = merge_regions(prepared_merge_set)
                new_regions_set.add(new_region)
                for old_region in prepared_merge_set:
                    old_region_to_new_region_map[old_region] = new_region

                init_index = merge_index
                prepared_merge_set = {region}

        for unmerged_region in regions_set_copy:
            for neighbor in unmerged_region.linked_regions:
                new_region = old_region_to_new_region_map.get(neighbor)
                if new_region:
                    unmerged_region.remove_neighbor(neighbor)
                    unmerged_region.add_neighbor(new_region)

        regions_set_copy |= new_regions_set

    return regions_set_copy
