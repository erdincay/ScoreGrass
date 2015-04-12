import unittest
import os
import numpy as np
import pandas as pd
from src.file_io import PublicSupport

from src.learning.strategy.MixedRegression import MixedRegression
from src.learning.strategy.QualityRegression import QualityRegression
from src.learning.strategy.ColorRegression import ColorRegression


__author__ = 'Kern'


class ModelTestCase(unittest.TestCase):
    def serializeModel(self, data_num, x_dim, y_dim, model_constructor, model_deserializer):
        self.assertGreaterEqual(data_num, 1)
        self.assertGreaterEqual(x_dim, 1)
        self.assertGreaterEqual(y_dim, 1)

        if x_dim > 1:
            x_data = pd.DataFrame(np.random.randn(data_num, x_dim))
        else:
            x_data = pd.Series(np.random.randn(data_num))

        if y_dim > 1:
            y_data = pd.DataFrame(np.random.randn(data_num, y_dim))
        else:
            y_data = pd.Series(np.random.randn(data_num))

        original = model_constructor(x_dim, y_dim)
        original.validation(x_data, y_data, 0.25)

        path = os.path.join(os.path.curdir, 'serialized')
        PublicSupport.create_path(path)
        file = original.save(path)
        self.assertTrue(os.path.isfile(file))
        new = model_deserializer(path)

        self.assertEqual(new.x_dim, x_dim)
        self.assertEqual(new.y_dim, y_dim)

    def testSerializeColorModel(self):
        self.serializeModel(20, 80, 1, ColorRegression, ColorRegression.deserialize_regression)

    def testSerializeQualityModel(self):
        self.serializeModel(20, 77, 1, QualityRegression, QualityRegression.deserialize_regression)

    def testSerializeMixedModel(self):
        self.serializeModel(20, 92, 2, MixedRegression, MixedRegression.deserialize_regression)


if __name__ == '__main__':
    unittest.main()
