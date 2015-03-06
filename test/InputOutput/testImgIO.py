from skimage import io
from skimage.color import rgb2grey
from skimage.viewer import ImageViewer
from skimage.filter import sobel

__author__ = 'Kern'

image = io.imread(
    "D:\\Dropbox\\Turfgrass software project\\Pictures from the summer\\July 31\\7-31-2014 plot 6.JPG")

image_gray = rgb2grey(image)

elevation_map = sobel(image_gray)

ImageViewer(elevation_map).show()