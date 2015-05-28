import sys

from skimage import io
import matplotlib.pyplot as plt

from src.image_preprocess.PeakDetector import *

__author__ = 'Kern'

if len(sys.argv) < 2:
    raise ValueError("Usage:", sys.argv[0], " Missing some argument to indicate input files")
path = sys.argv[1]

image = io.imread(path)
distance_mark = generate_pink_map(image)
coords_red = peak_detector(distance_mark, 0.825, 50)

print(coords_red.shape)

f, (ax0, ax2, ax1) = plt.subplots(1, 3, figsize=(20, 15))
ax0.imshow(image)
ax0.set_title('Input image')
ax1.imshow(image)
ax1.set_title('Marker locations')
ax1.plot(coords_red[:, 1], coords_red[:, 0], 'ro')
ax2.imshow(distance_mark, interpolation='nearest', cmap='gray')
ax2.set_title('Distance to pure red')
plt.show()
