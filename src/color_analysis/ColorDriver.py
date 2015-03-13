from skimage import io
from src.color_analysis import ColorAnalyzer

__author__ = 'Alec'

imdata = io.imread("testimg.JPG")

colorInfo = ColorAnalyzer.getColorVals(imdata)
colorInfo.printValues()  # using its attributes: avgR, avgG, avgB, avgH, avgS, and avgV

filterer = ColorAnalyzer.getColorFilter(imdata)
filterer.filterToColorRange(30, 100)
filterer.printValues()  # using its attributes: percentCover, filteredR, filteredG, and filteredB

filterer.filterGreen(100, 150)
filterer.printValues()