import sys

from skimage import io
import matplotlib.pyplot as plt

from src.image_preprocess.EdgeDetector import *


__author__ = 'kern.ding'

if len(sys.argv) < 2:
    raise ValueError("Usage:", sys.argv[0], " Missing some argument to indicate input files")
path = sys.argv[1]

image = io.imread(path)
binary_image = binarize(image)
edges_map = canny_edge_detector(binary_image, 4.5)
contours = find_edges(edges_map)

f, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(20, 15))
ax0.imshow(image)
ax0.set_title('Input image')
ax1.imshow(binary_image, cmap=plt.cm.jet)
ax1.set_title('binary image')
ax2.imshow(edges_map, cmap=plt.cm.gray)
ax2.set_title('detected edges')
ax3.imshow(image, interpolation='nearest')
ax3.set_title('found contours')
for n, contour in enumerate(contours):
    ax3.plot(contour[:, 1], contour[:, 0], linewidth=2)
plt.show()