import unittest
import os
from src.learning.strategy.MixedRegression import MixedRegression
from src.learning.strategy.QualityRegression import QualityRegression
from src.learning.strategy.ColorRegression import ColorRegression

__author__ = 'Kern'


class ModelTestCase(unittest.TestCase):
    def deserializeModel(self, model, deserialize):
        path = os.path.curdir
        file = model.save(path)
        self.assertTrue(os.path.isfile(file))
        return deserialize(path)

    def testSerializeColorModel(self):
        x_dim = 80
        y_dim = 1
        original = ColorRegression(x_dim, y_dim)
        new = self.deserializeModel(original, ColorRegression.deserialize_regression)

        self.assertEqual(new.x_dim, x_dim)
        self.assertEqual(new.y_dim, y_dim)

    def testSerializeQualityModel(self):
        x_dim = 77
        y_dim = 1
        original = QualityRegression(x_dim, y_dim)
        new = self.deserializeModel(original, QualityRegression.deserialize_regression)

        self.assertEqual(new.x_dim, x_dim)
        self.assertEqual(new.y_dim, y_dim)

    def testSerializeMixedModel(self):
        x_dim = 92
        y_dim = 1
        original = MixedRegression(x_dim, y_dim)
        new = self.deserializeModel(original, MixedRegression.deserialize_regression)

        self.assertEqual(new.x_dim, x_dim)
        self.assertEqual(new.y_dim, y_dim)


if __name__ == '__main__':
    unittest.main()
