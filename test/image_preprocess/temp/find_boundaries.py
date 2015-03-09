from scipy.ndimage import io, uniform_filter
from skimage import color
from skimage.segmentation import find_boundaries, mark_boundaries
import matplotlib.pyplot as plt
import sys
from src.image_preprocess.PeakDetector import generate_green_map

__author__ = 'Kern'

if len(sys.argv) < 2:
    raise ValueError("Usage:", sys.argv[0], " Missing some argument to indicate input files")
path = sys.argv[1]

image = io.imread(path)
# image = uniform_filter(image_input)
image_green = generate_green_map(image)


# image_grey = color.rgb2gray(image)

image_out = find_boundaries(image)
image_mark = mark_boundaries(image, image_green)

fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
ax0.imshow(image)
ax0.set_title('Input image')
ax1.imshow(image_out)
ax1.set_title('boundaries image')
ax1.axis('image')
ax2.imshow(image_mark)
ax2.set_title("grey image")

plt.show()