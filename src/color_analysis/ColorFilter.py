import numpy

class ColorFilter:

    def __init__(self, image):
        self.percentCover = 0.0
        self.filteredR = 0.0
        self.filteredG = 0.0
        self.filteredB = 0.0
        self.image = image

    def resetFilterToImg(self):
        """
        Used to reset the filter to default values.

        :return:
        """

        self.percentCover = 0.0
        self.filteredR = 0.0
        self.filteredG = 0.0
        self.filteredB = 0.0

    def filterToColorRange(self, min, max, color=0):
        """
        This function is used to apply a filter and calculate the filter values, whether using a hue range
        or a specific range of R, G, or B for filtering (indicated by the optional color parameter).
        Calculated values are set to this object's attributes.

        :param min: minimum value for a pixel to stay in the filter
        :param max: maximum value for a pixel to stay in the filter
        :param color: the color to use for filtering (R, G, B), or 0 to indicate a hue range
        :return:
        """

        pixelCount = 0.0
        pixelsThatCount = 0.0
        tempR = 0.0
        tempG = 0.0
        tempB = 0.0

        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                if color == 0:
                    val = hueFromRGB(float(self.image[i, j, 0]), float(self.image[i, j, 1]), float(self.image[i, j, 2]))
                elif color == 1:
                    val = self.image[i, j, 0]
                elif color == 2:
                    val = self.image[i, j, 1]
                elif color == 3:
                    val = self.image[i, j, 2]
                else:
                    return

                if val >= min and val <= max:
                    pixelsThatCount += 1
                    tempR += self.image[i, j, 0]
                    tempG += self.image[i, j, 1]
                    tempB += self.image[i, j, 2]

                pixelCount += 1

        if pixelsThatCount > 0 and pixelCount > 0:
            self.percentCover = pixelsThatCount / pixelCount
            self.filteredR = tempR / pixelsThatCount
            self.filteredG = tempG / pixelsThatCount
            self.filteredB = tempB / pixelsThatCount
        else:
            self.percentCover = 0.0
            self.filteredR = 0.0
            self.filteredG = 0.0
            self.filteredB = 0.0

    def filterRed(self, min, max):
        self.filterToColorRange(min, max, 1)

    def filterGreen(self, min, max):
        self.filterToColorRange(min, max, 2)

    def filterBlue(self, min, max):
        self.filterToColorRange(min, max, 3)

    def printValues(self):
        print("Percent coverage: ", (self.percentCover * 100), "%")
        print("Average R in filter: ", self.filteredR)
        print("Average G in filter: ", self.filteredG)
        print("Average B in filter: ", self.filteredB)


def hueFromRGB(r, g, b):
    """
    Utility function used to simply obtain a hue value calculated from R, G, and B values.

    :param r: R value used
    :param g: G value used
    :param b: B value used
    :return: calculated hue value
    """

    h = 0.0

    rgbMax = max(r, g, b)
    rgbMin = min(r, g, b)

    if rgbMax == rgbMin:
        return 0.0

    if rgbMax == r:
        h = (g - b) / (rgbMax - rgbMin)
    elif rgbMax == g:
        h = 2.0 + (b - r) / (rgbMax - rgbMin)
    else:
        h = 4.0 + (r - g) / (rgbMax - rgbMin)

    h *= 60

    if h < 0:
        h += 360

    return h