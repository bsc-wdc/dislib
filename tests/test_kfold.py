import unittest

import numpy as np

from dislib.data import load_data
from dislib.model_selection import KFold


class KFoldTest(unittest.TestCase):

    def test_split(self):
        """Tests KFold.split() method"""
        x = np.random.rand(1000, 3)
        y = np.arange(1000)
        data = load_data(x=x, y=y, subset_size=111)
        cv = KFold()
        n_splits = 0
        for train_ds, test_ds in cv.split(data):
            n_splits += 1
            self.assertEqual(len(test_ds.labels), 200)
        self.assertEqual(cv.n_splits, n_splits)

    def test_init_params(self):
        """Tests that KFold() raises errors on invalid parameters"""
        with self.assertRaises(ValueError):
            KFold(n_splits=0.5)
        with self.assertRaises(ValueError):
            KFold(n_splits=1)
        with self.assertRaises(TypeError):
            KFold(shuffle=None)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
