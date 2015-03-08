from scipy.ndimage import io
from skimage.segmentation import find_boundaries
import matplotlib.pyplot as plt

__author__ = 'Kern'


image = io.imread(
    "D:\\Cloud\\Dropbox\\Turfgrass software project\\Pictures taken\\Patch 7.JPG")
# image = transform.resize(image, (800, 800))
# image_grey = color.rgb2gray(image)

image_out = find_boundaries(image)

fig, (ax0, ax1) = plt.subplots(1, 2)
ax0.imshow(image)
ax0.set_title('Input image')
ax1.imshow(image_out)
ax1.set_title('boundaries image')
ax1.axis('image')

plt.show()