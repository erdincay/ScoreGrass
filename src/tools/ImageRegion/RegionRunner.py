import inspect
import sys
import logging

from skimage import io
from src.tools.ImageRegion.Strategy.EdgeMarker import make_region_edge
from src.tools.ImageRegion.Strategy.RegionCalculator import iterate_regions
from src.tools.ImageRegion.Strategy.RegionInitializer import init_regions

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
if len(readret) > 1:
    image = readret[0]
else:
    image = readret

regions_set = init_regions(image, 3600)


for i in range(20):
    old_set_len = len(regions_set)
    regions_set = iterate_regions(regions_set, 7)
    if len(regions_set) >= old_set_len:
        logging.info("iterations = " + str(i))
        break

logging_regions(regions_set)

for index, region in enumerate(regions_set):
    edge = make_region_edge(region)
    logging.info(str(index) + ": " + str(edge))
    image = edge.draw(image, [255, 255, 255])

io.imshow(image)
io.show()


