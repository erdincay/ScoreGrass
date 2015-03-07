from skimage import io
from src.image_preprocess.Cropper import diagonal_cropping
from src.image_preprocess.PeakDetector import generate_red_map, peak_detector

import matplotlib.pyplot as plt

__author__ = 'Kern'

image = io.imread(
    "D:\\Cloud\\Dropbox\\Turfgrass software project\\Pictures from the summer\\July 31\\7-31-2014 plot 6.JPG")

distance_red = generate_red_map(image)
coords_red = peak_detector(distance_red, 0.825, 80)

crop = diagonal_cropping(image, coords_red)

f, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(15, 10))
ax0.imshow(image)
ax0.set_title('Input image')
ax1.imshow(image)
ax1.set_title('Marker locations')
ax1.plot(coords_red[:, 1], coords_red[:, 0], 'ro')
ax1.axis('image')
ax2.imshow(distance_red, interpolation='nearest', cmap='gray')
ax2.set_title('Distance to pure red')
ax3.imshow(crop)
ax3.set_title('Cropping image')

plt.show()