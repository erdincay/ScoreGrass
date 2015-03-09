"""
===============
Contour finding
===============

``skimage.measure.find_contours`` uses a marching squares method to find
constant valued contours in an image.  Array values are linearly interpolated
to provide better precision of the output contours.  Contours which intersect
the image edge are open; all others are closed.

The `marching squares algorithm
<http://www.essi.fr/~lingrand/MarchingCubes/algo.html>`__ is a special case of
the marching cubes algorithm (Lorensen, William and Harvey E. Cline. Marching
Cubes: A High Resolution 3D Surface Construction Algorithm. Computer Graphics
(SIGGRAPH 87 Proceedings) 21(4) July 1987, p. 163-170).

"""
import sys

import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from skimage import measure
from skimage import io
from skimage import color
from skimage.viewer import ImageViewer


if len(sys.argv) < 2:
    raise ValueError("Usage:", sys.argv[0], " Missing some argument to indicate input files")
path = sys.argv[1]

image = io.imread(path)
image = uniform_filter(image)
# image = transform.resize(image, (800, 800))
image_grey = color.rgb2gray(image)

# ImageViewer(image_grey).show()

contours = measure.find_contours(image_grey, 10, fully_connected='low')
print(len(contours))

# Display the image and plot all contours found
fig, ax = plt.subplots()
ax.imshow(image_grey, interpolation='nearest', cmap=plt.cm.gray)

for n, contour in enumerate(contours):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()
