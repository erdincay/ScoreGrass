__author__ = 'Alec'

"""Class representing the basic color information of an image (i.e. average RGB/HSV values)."""


class ColorData:
    def __init__(self, img):
        self.image = img
        self.avgR = 0.0
        self.avgG = 0.0
        self.avgB = 0.0
        self.avgH = 0.0
        self.avgS = 0.0
        self.avgV = 0.0

    def calc_color_values(self):
        """
        Used for calculating the color values of the object initially.
        Calculates average RGB of an image, then converts that to HSV and holds the values.

        :return:
        """

        sumR = 0.0
        sumG = 0.0
        sumB = 0.0
        totalPix = 0

        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                sumR += self.image[i, j, 0]
                sumG += self.image[i, j, 1]
                sumB += self.image[i, j, 2]
                totalPix += 1

        if not totalPix == 0:
            self.avgR = sumR / totalPix
            self.avgG = sumG / totalPix
            self.avgB = sumB / totalPix
            self.calc_hsv(self.avgR, self.avgG, self.avgB, True)

        return

    def printValues(self):
        print("Average R: ", self.avgR)
        print("Average G: ", self.avgG)
        print("Average B: ", self.avgB)
        print("Average H: ", self.avgH)
        print("Average S: ", self.avgS)
        print("Average V: ", self.avgV)

    def calc_hsv(self, r, g, b, forSelf=True):
        """
        Converts the specified R, G, and B values to HSV format, and stores them in this class's attributes for
        referencing.

        :param r: the red value used for conversion
        :param g: the green value used for conversion
        :param b: the blue value used for conversion
        :param forSelf: whether the calculation should replace this object's average HSV values
        :return:
        """

        h = 0.0
        s = 0.0
        v = 0.0
        rgbMin = 0.0

        v = max(r, g, b)
        rgbMin = min(r, g, b)

        if v != 0.0:
            s = (v - rgbMin) / v

            if s != 0.0:

                if v == r:
                    est = (g - b) / (v - rgbMin)
                elif v == g:
                    est = 2.0 + (b - r) / (v - rgbMin)
                else:
                    est = 4.0 + (r - g) / (v - rgbMin)

                est *= 60

                if est >= 0:
                    h = est
                else:
                    h = est + 360.0

        v /= 255

        if forSelf:
            self.avgH = h
            self.avgS = s
            self.avgV = v
        else:
            self.h = h
            self.s = s
            self.v = v

        return
