from scipy.ndimage import io
from skimage import transform
from skimage import color

__author__ = 'Kern'


image = io.imread(
    "D:\\Cloud\\Dropbox\\Turfgrass software project\\Pictures taken\\Patch 7.JPG")
image = transform.resize(image, (800, 800))
image_grey = color.rgb2gray(image)

hspace, angles, dists = transform.hough_line(image_grey)
hspace, angles, dists = transform.hough_line_peaks(hspace, angles, dists)
print(hspace, angles, dists)