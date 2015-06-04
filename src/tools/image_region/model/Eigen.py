__author__ = 'kern.ding'

import numpy as np


class Eigen:
    def __init__(self, coord_map, eigen_val):
        self.coord_map = coord_map
        if eigen_val is None:
            self.eigen_val = self._calc_eigen()
        else:
            self.eigen_val = eigen_val

    def _calc_eigen(self):
        if not self.coord_map:
            raise ValueError("cannot calculate eigen value from empty coordinates map", self.coord_map)

        channel_sum = np.zeros(len(next(iter(self.coord_map.values()))))
        for pixel in self.coord_map.values():
            channel_sum += pixel

        return channel_sum / len(self.coord_map)

    def get_eigen(self):
        return self.eigen_val
