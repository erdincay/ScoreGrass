__author__ = 'Kern'

def generate_square_linked_coordinates(coord):
    row_list = [coord[0] - 1, coord[0], coord[0] + 1]
    col_list = [coord[1] - 1, coord[1], coord[1] + 1]
    return {(row, col) for row in row_list for col in col_list if row != coord[0] or col != coord[1]}


class Line:
    def __init__(self, coords):
        self.coords = coords

    def draw(self, image, color):
        for coord in self.coords:
            image[coord] = color
        return image

    def __str__(self):
        return str(self.coords)
