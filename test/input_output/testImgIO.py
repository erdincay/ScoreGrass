import sys

from skimage import io, img_as_float
from skimage.color import rgb2grey
from skimage.viewer import ImageViewer
from skimage.filters import sobel

__author__ = 'Kern'

if len(sys.argv) < 2:
    raise ValueError("Usage:", sys.argv[0], " Missing some argument to indicate input files")
path = sys.argv[1]

image = io.imread(path)

image = img_as_float(image)

image_gray = rgb2grey(image)

elevation_map = sobel(image_gray)

ImageViewer(elevation_map).show()
