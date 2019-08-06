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
        self.assertEqual(cv.get_n_splits(), n_splits)

    def test_split_no_shuffle(self):
        """Tests KFold.split() method with shuffle=False"""
        x = np.random.rand(1000, 3)
        y = np.arange(1000)
        data = load_data(x=x, y=y, subset_size=111)
        cv = KFold(shuffle=False)
        n_splits = 0
        for train_ds, test_ds in cv.split(data):
            n_splits += 1
            self.assertEqual(len(test_ds.labels), 200)
        self.assertEqual(cv.get_n_splits(), n_splits)

    def test_split_no_shuffle_uneven_folds(self):
        """Tests KFold.split() method with shuffle=False and uneven folds"""
        x = np.random.rand(1000, 3)
        y = np.arange(1000)
        data = load_data(x=x, y=y, subset_size=334)
        cv = KFold(n_splits=3, shuffle=False)
        n_splits = 0
        for train_ds, test_ds in cv.split(data):
            n_splits += 1
            size = len(test_ds.labels)
            self.assertTrue(size == 333 or size == 334,
                            'Fold size is ' + str(size) +
                            ' and should be 333 or 334.')
        self.assertEqual(cv.get_n_splits(), n_splits)
        self.assertEqual(3, n_splits)

    def test_split_single_subset_no_shuffle_uneven_folds(self):
        """Tests KFold.split() from single subset, shuffle=False and uneven"""
        x = np.random.rand(1000, 3)
        y = np.arange(1000)
        data = load_data(x=x, y=y, subset_size=1000)
        cv = KFold(n_splits=6, shuffle=False)
        n_splits = 0
        for train_ds, test_ds in cv.split(data):
            n_splits += 1
            size = len(test_ds.labels)
            self.assertTrue(size == 166 or size == 167,
                            'Fold size is ' + str(size) +
                            ' but should be 166 or 167.')
        self.assertEqual(cv.get_n_splits(), n_splits)

    def test_init_params(self):
        """Tests that KFold() raises errors on invalid parameters"""
        with self.assertRaises(ValueError):
            KFold(n_splits=2.5)
        with self.assertRaises(ValueError):
            KFold(n_splits=1)
        with self.assertRaises(TypeError):
            KFold(shuffle=None)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
