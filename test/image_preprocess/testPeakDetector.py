from src.image_preprocess.PeakDetector import *

__author__ = 'Kern'

from skimage import io, img_as_float

import matplotlib.pyplot as plt

image = img_as_float(
    io.imread("D:\\Dropbox\\Turfgrass software project\\Pictures from the summer\\July 31\\7-31-2014 plot 6.JPG"))

distance_red = generate_red_map(image)
coords_red = peak_detector(distance_red)

print(coords_red.shape)

f, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 10))
ax0.imshow(image)
ax0.set_title('Input image')
ax1.imshow(image)
ax1.set_title('Marker locations')
ax1.plot(coords_red[:, 1], coords_red[:, 0], 'ro')
ax1.axis('image')
ax2.imshow(distance_red, interpolation='nearest', cmap='gray')
ax2.set_title('Distance to pure red')
plt.show()