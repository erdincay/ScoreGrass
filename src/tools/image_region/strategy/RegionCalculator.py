import operator
from src.tools.image_region.model.Region import merge_regions

__author__ = 'Kern'


def _generate_regions_merge_map(regions_set, max_eigen_diff, max_merged_num):
    """
    generate a hash table that key is regions will be merged, value is the merged group NO.(new regions NO.)
    :param max_merged_num:
    :param regions_set:
    :param max_eigen_diff:
    :return: the hash table
    """
    pre_merge_regions_map = dict()
    merge_index = 0

    for region in regions_set:
        if region.linked_regions:
            for close_rank, closest_region in enumerate(sorted(region.linked_regions, key=lambda lr: abs(region.get_hue() - lr.get_hue()) + abs(region.get_saturation() - lr.get_saturation()))):
                if close_rank < max_merged_num and abs(region.get_hue() - closest_region.get_hue()) > max_eigen_diff:
                    region_index = pre_merge_regions_map.get(region)
                    linked_index = pre_merge_regions_map.get(closest_region)
                    if region_index is None and linked_index:
                        pre_merge_regions_map[region] = linked_index
                    elif region_index and linked_index is None:
                        pre_merge_regions_map[closest_region] = region_index
                    elif region_index is None and linked_index is None:
                        pre_merge_regions_map[region] = merge_index
                        pre_merge_regions_map[closest_region] = merge_index
                        merge_index += 1
                    elif region_index != linked_index:
                        for pre_region in [region for region, merge_index in pre_merge_regions_map.items() if
                                           merge_index == region_index or merge_index == linked_index]:
                            pre_merge_regions_map[pre_region] = merge_index
                        merge_index += 1

    return sorted(pre_merge_regions_map.items(), key=operator.itemgetter(1))


def __remap_linked_regions(regions_set, old_to_new_map):
    """
    modify regions_set.linked_regions
    :param regions_set: regions need to be checked
    :param old_to_new_map: old regions to new regions map
    """
    for region in regions_set:
        linked_regions_copy = region.linked_regions.copy()
        for neighbor in linked_regions_copy:
            new_region = old_to_new_map.get(neighbor)
            if new_region:
                region.remove_neighbor(neighbor)
                region.add_neighbor(new_region)


def __merge(prepared_merge_set, old_to_new_map):
    new_region = merge_regions(prepared_merge_set)
    for old_region in prepared_merge_set:
        old_to_new_map[old_region] = new_region

    return new_region


def merge_regions_map(sorted_prepared_regions, original_regions_set):
    """
    real merge operation based on prepared merged regions map
    :param sorted_prepared_regions: the map
    :param original_regions_set:
    :return: new regions set
    """
    new_regions_set = set()

    if len(sorted_prepared_regions) >= 2:
        init_index = sorted_prepared_regions[0][1]
        prepared_merge_set = set()
        old_region_to_new_region_map = dict()

        for region, merge_index in sorted_prepared_regions:
            original_regions_set.remove(region)
            if merge_index == init_index:
                prepared_merge_set.add(region)
            else:
                new_regions_set.add(__merge(prepared_merge_set, old_region_to_new_region_map))
                init_index = merge_index
                prepared_merge_set = {region}

        if prepared_merge_set:
            new_regions_set.add(__merge(prepared_merge_set, old_region_to_new_region_map))

        if old_region_to_new_region_map:
            __remap_linked_regions(original_regions_set, old_region_to_new_region_map)
            __remap_linked_regions(new_regions_set, old_region_to_new_region_map)

    return original_regions_set | new_regions_set


def iterate_regions(regions_set, max_eigen_diff, max_merged_num):
    """
    Merge regions one time
    :param regions_set: all the regions to be merge
    :param max_eigen_diff: eigen value threshold to prevent two regions with large gap eigen values to be merged.
    :return: merged regions
    """
    sorted_pre_regions = _generate_regions_merge_map(regions_set, max_eigen_diff, max_merged_num)
    return merge_regions_map(sorted_pre_regions, regions_set.copy())
