"""
===================
Label image regions
===================

This example shows how to segment an image with image labelling. The following
steps are applied:

1. Thresholding with automatic Otsu method
2. Close small holes with binary closing
3. Remove artifacts touching image border
4. Measure image regions to filter small objects

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import uniform_filter
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing, square
from skimage.measure import regionprops
from skimage.color import label2rgb
from skimage.transform import resize
from skimage.filter import threshold_otsu
from skimage import io
from skimage import color
from skimage.viewer import ImageViewer


image = io.imread(
    "D:\\Dropbox\\Turfgrass software project\\Pictures taken\\Patch 7.JPG")
image = resize(image, (800, 800))
image = color.rgb2gray(image)

image = uniform_filter(image)
ImageViewer(image).show()

# apply threshold
thresh = threshold_otsu(image)
bw = closing(image > thresh, square(3))
ImageViewer(bw).show()

# remove artifacts connected to image border
cleared = bw.copy()
clear_border(cleared)
ImageViewer(cleared).show()


# label image regions
label_image = label(cleared)
borders = np.logical_xor(bw, cleared)
label_image[borders] = -1

image_label_overlay = label2rgb(label_image, image=image)
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
ax.imshow(image_label_overlay)

for region in regionprops(label_image):

    # skip small images
    if region.area < 100:
        continue

    # draw rectangle around segmented coins
    minr, minc, maxr, maxc = region.bbox
    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                              fill=False, edgecolor='white', linewidth=2)
    ax.add_patch(rect)

plt.show()
