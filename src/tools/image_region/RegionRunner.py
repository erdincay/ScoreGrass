import inspect
import sys
import logging

from skimage import io

from src.tools.image_region.Strategy.EdgeMarker import make_region_edge
from src.tools.image_region.Strategy.RegionCalculator import iterate_regions
from src.tools.image_region.Strategy.RegionInitializer import init_regions

__author__ = 'Kern'

logging.basicConfig(filename=inspect.getfile(inspect.currentframe()) + '.log', level=logging.DEBUG)


def logging_regions(regions):
    for r in regions:
        logging.info(r)
    logging.info("total: " + str(len(regions)))


if len(sys.argv) < 2:
    raise ValueError("Usage:", sys.argv[0], " Missing some argument to indicate input files")

path = sys.argv[1]
readret = io.imread(path)
if len(readret.shape) == 3:
    image = readret
elif len(readret.shape) == 1:
    image = readret[0]
else:
    raise ValueError("cannot read the image: " + path)

regions_set = init_regions(image, 2500)

new_set_len = 0
old_set_len = len(regions_set)

while new_set_len < old_set_len:
    old_set_len = len(regions_set)
    regions_set = iterate_regions(regions_set, 7)
    new_set_len = len(regions_set)

logging_regions(regions_set)

for index, region in enumerate(regions_set):
    edge = make_region_edge(region)
    logging.info(str(index) + ": " + str(edge))
    image = edge.draw(image, [255, 255, 255])

io.imshow(image)
io.show()
