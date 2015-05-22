from skimage import io
import sys
from src.tools.ImageRegion.Strategy.RegionCalculator import iterate_regions
from src.tools.ImageRegion.Strategy.RegionInitializer import init_regions

__author__ = 'Kern'

if len(sys.argv) < 2:
    raise ValueError("Usage:", sys.argv[0], " Missing some argument to indicate input files")
path = sys.argv[1]

regions_set = init_regions(io.imread(path), 1600)

for i in range(100):
    regions_set = iterate_regions(regions_set)