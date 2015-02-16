__author__ = 'Kern'

from skimage import io
from skimage import data
from skimage import filter

lena = io.imread("./G0010109.jpg")
print(type(lena))