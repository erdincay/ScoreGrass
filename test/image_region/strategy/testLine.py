from src.tools.image_region.model.Line import generate_square_linked_coordinates

__author__ = 'kern.ding'

import unittest


class MyTestCase(unittest.TestCase):
    def test_neighbor_coordinate(self):
        coord = 0, 0
        neighbors = generate_square_linked_coordinates(coord)

        self.assertEqual(len(neighbors), 8)

if __name__ == '__main__':
    unittest.main()
