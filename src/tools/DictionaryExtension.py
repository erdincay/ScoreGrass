__author__ = 'Kern'


def merge_two_dicts(dict_x, dict_y):
    dict_new = dict_x.copy()
    dict_new.update(dict_y)
    return dict_new


def merge_dicts(*dict_args):
    dict_new = {}
    for dictionary in dict_args:
        dict_new.update(dictionary)
    return dict_new

