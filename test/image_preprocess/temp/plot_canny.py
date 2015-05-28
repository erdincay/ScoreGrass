"""
===================
Canny edge detector
===================

The Canny filter is a multi-stage edge detector. It uses a filter based on the
derivative of a Gaussian in order to compute the intensity of the gradients.The
Gaussian reduces the effect of noise present in the image. Then, potential
edges are thinned down to 1-pixel curves by removing non-maximum pixels of the
gradient magnitude. Finally, edge pixels are kept or removed using hysteresis
thresholding on the gradient magnitude.

The Canny has three adjustable parameters: the width of the Gaussian (the
noisier the image, the greater the width), and the low and high threshold for
the hysteresis thresholding.

"""
import sys

import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2grey
from skimage.feature import canny
from skimage import measure
from skimage.filters import threshold_otsu
from skimage.morphology import closing, square


if len(sys.argv) < 2:
    raise ValueError("Usage:", sys.argv[0], " Missing some argument to indicate input files")
path = sys.argv[1]

image = io.imread(path)
image = rgb2grey(image)
# image = generate_green_map(image)
# im = uniform_filter(image)
thresh = threshold_otsu(image)
# im = image > thresh
im = closing(image > thresh, square(3))

_sigma = 4.5
# Compute the Canny filter for two values of sigma
edges1 = canny(im)
edges2 = canny(im, sigma=_sigma)

# display results
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3))

ax1.imshow(im, cmap=plt.cm.jet)
ax1.axis('off')
ax1.set_title('noisy image', fontsize=20)

ax2.imshow(edges1, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('Canny filter, $\sigma=1$', fontsize=20)

ax3.imshow(edges2, cmap=plt.cm.gray)
ax3.axis('off')
ax3.set_title('Canny filter, $\sigma=3$', fontsize=20)

fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9,
                    bottom=0.02, left=0.02, right=0.98)

plt.show()

contours = measure.find_contours(edges2, 0.5, fully_connected='high')
print(len(contours))
if len(contours) < 10:
    for contour in contours:
        print(contour)

# Display the image and plot all contours found
fig, ax = plt.subplots()
ax.imshow(edges2, interpolation='nearest', cmap=plt.cm.gray)

for n, contour in enumerate(contours):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()
