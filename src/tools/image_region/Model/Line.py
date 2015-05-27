__author__ = 'Kern'


class Line:
    def __init__(self, coords):
        self.coords = coords

    def draw(self, image, color):
        for coord in self.coords:
            image[coord] = color
        return image

    def __str__(self):
        return str(self.coords)
