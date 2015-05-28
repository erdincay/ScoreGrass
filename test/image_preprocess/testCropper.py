import sys

from skimage import io
import matplotlib.pyplot as plt

from src.image_preprocess.Cropper import *
from src.image_preprocess.EdgeDetector import *
from src.image_preprocess.PeakDetector import *

io.use_plugin('matplotlib')

__author__ = 'Kern'

if len(sys.argv) < 3:
    raise ValueError("Usage:", sys.argv[0], " Missing some argument to indicate input files")
path_dot = sys.argv[1]
path_frame = sys.argv[2]

image_dot = io.imread(path_dot)
image_frame = io.imread(path_frame)

distance_mark = generate_pink_map(image_dot)
coords_mark = peak_corner_detector(distance_mark, 0.825, 80, 9)
crop_dot = diagonal_cropping(image_dot, coords_mark, lambda coords: sequence_extreme_value(coords, 1))

binary_image = binarize(image_frame)
edges_map = canny_edge_detector(binary_image, 4.5)
contours = find_edges(edges_map)
coords_frame = edge_coordinate(contours)
crop_frame = diagonal_cropping(image_frame, coords_frame, extreme_value, 100)

f, ((dot1, dot2, dot3), (frame1, frame2, frame3)) = plt.subplots(2, 3, figsize=(15, 10))

dot1.imshow(image_dot)
dot1.set_title('Input Image')
dot2.imshow(image_dot)
dot2.plot(coords_mark[:, 1], coords_mark[:, 0], 'ro')
dot2.set_title('Image Marker')
dot3.imshow(crop_dot)
dot3.set_title('Cropped Image')

frame1.imshow(image_frame)
frame1.set_title('Input Image')
frame2.imshow(image_frame)
for n, contour in enumerate(contours):
    frame2.plot(contour[:, 1], contour[:, 0], linewidth=2)
frame2.set_title('Image Frame')
frame3.imshow(crop_frame)
frame3.set_title('Cropped Image')

plt.show()

io.imsave('test.jpg', crop_dot)
