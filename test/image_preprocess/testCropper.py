import sys
import functools

from skimage import io
import matplotlib.pyplot as plt

from src.image_preprocess.Cropper import *
from src.image_preprocess.PeakDetector import *


__author__ = 'Kern'

if len(sys.argv) < 2:
    raise ValueError("Usage:", sys.argv[0], " Missing some argument to indicate input files")
path = sys.argv[1]

image = io.imread(path)

distance_red = generate_red_map(image)
coords_red = peak_corner_detector(distance_red, 0.825, 150)
print(coords_red.shape)

crop_first_val = diagonal_cropping(image, coords_red, extreme_value)
crop_second_val = diagonal_cropping(image, coords_red, functools.partial(sequence_extreme_value, index=1))
crop_third_val = diagonal_cropping(image, coords_red, functools.partial(sequence_extreme_value, index=2))

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
ax1.imshow(image)
ax1.set_title('Marker locations')
ax1.plot(coords_red[:, 1], coords_red[:, 0], 'ro')
ax1.axis('image')
ax2.imshow(crop_first_val)
ax2.set_title('cropped by max/min coordinate')
ax3.imshow(crop_second_val)
ax3.set_title('cropped by secondary coordinate')
ax4.imshow(crop_third_val)
ax4.set_title('cropped by third coordinate')

plt.show()