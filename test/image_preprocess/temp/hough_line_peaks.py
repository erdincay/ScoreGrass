from scipy.ndimage import io, uniform_filter
from skimage import transform
from skimage import color
import sys
from src.image_preprocess.PeakDetector import generate_green_map

__author__ = 'Kern'


if len(sys.argv) < 2:
    raise ValueError("Usage:", sys.argv[0], " Missing some argument to indicate input files")
path = sys.argv[1]

image_input = io.imread(path)
image = uniform_filter(image_input)
image = generate_green_map(image)

image_grey = color.rgb2gray(image)

hspace, angles, dists = transform.hough_line(image_grey)
hspace, angles, dists = transform.hough_line_peaks(hspace, angles, dists)
print(hspace, angles, dists)