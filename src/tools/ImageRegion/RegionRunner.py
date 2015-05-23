import inspect
import sys
import logging

from skimage import io
from src.tools.ImageRegion.Strategy.RegionCalculator import iterate_regions
from src.tools.ImageRegion.Strategy.RegionInitializer import init_regions

__author__ = 'Kern'

logging.basicConfig(filename=inspect.getfile(inspect.currentframe()) + '.log', level=logging.DEBUG)


def logging_regions(regions):
    for regn in regions:
        logging.info(regn)
    logging.info("total: " + str(len(regions)))
    logging.info(
        "==============================================================================================================================================")


if len(sys.argv) < 2:
    raise ValueError("Usage:", sys.argv[0], " Missing some argument to indicate input files")
path = sys.argv[1]

regions_set = init_regions(io.imread(path)[0], 1600)

for i in range(1000):
    logging_regions(regions_set)

    old_set_len = len(regions_set)
    regions_set = iterate_regions(regions_set, 255)
    if len(regions_set) == old_set_len:
        break

logging_regions(regions_set)
