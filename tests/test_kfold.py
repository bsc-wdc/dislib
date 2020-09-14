import unittest

import numpy as np

import dislib as ds
from dislib.model_selection import KFold


class KFoldTest(unittest.TestCase):

    def test_split(self):
        """Tests KFold.split() method"""
        x_np = np.random.rand(1000, 3)
        y_np = np.arange(1000)
        x = ds.array(x_np, (111, 3))
        y = ds.array(y_np[:, np.newaxis], (111, 1))
        cv = KFold()
        n_splits = 0
        for train_ds, test_ds in cv.split(x, y):
            len_x_test = test_ds[0].shape[0]
            self.assertEqual(len_x_test, 200)
            n_splits += 1
        self.assertEqual(cv.get_n_splits(), n_splits)

    def test_split_no_shuffle(self):
        """Tests KFold.split() method with shuffle=False"""
        x_np = np.random.rand(1000, 3)
        y_np = np.arange(1000)
        x = ds.array(x_np, (111, 3))
        y = ds.array(y_np[:, np.newaxis], (111, 1))
        cv = KFold(shuffle=False)
        n_splits = 0
        for train_ds, test_ds in cv.split(x, y):
            len_x_test = test_ds[0].shape[0]
            self.assertEqual(len_x_test, 200)
            n_splits += 1
        self.assertEqual(cv.get_n_splits(), n_splits)

    def test_split_no_shuffle_uneven_folds(self):
        """Tests KFold.split() method with shuffle=False and uneven folds"""
        x_np = np.random.rand(1000, 3)
        y_np = np.arange(1000)
        x = ds.array(x_np, (334, 3))
        y = ds.array(y_np[:, np.newaxis], (334, 1))
        cv = KFold(n_splits=3, shuffle=False)
        n_splits = 0
        for train_ds, test_ds in cv.split(x, y):
            len_x_test = test_ds[0].shape[0]
            self.assertTrue(len_x_test == 333 or len_x_test == 334,
                            'Fold size is ' + str(len_x_test) +
                            ' and should be 333 or 334.')
            n_splits += 1
        self.assertEqual(cv.get_n_splits(), n_splits)
        self.assertEqual(3, n_splits)

    def test_split_single_subset_no_shuffle_uneven_folds(self):
        """Tests KFold.split() from single subset, shuffle=False and uneven"""
        x_np = np.random.rand(1000, 3)
        y_np = np.arange(1000)
        x = ds.array(x_np, (1000, 3))
        y = ds.array(y_np[:, np.newaxis], (1000, 1))
        cv = KFold(n_splits=6, shuffle=False)
        n_splits = 0
        for train_ds, test_ds in cv.split(x, y):
            len_x_train = train_ds[0].shape[0]
            len_y_train = train_ds[1].shape[0]
            self.assertEquals(len_x_train, len_y_train)
            len_x_test = test_ds[0].shape[0]
            len_y_test = test_ds[1].shape[0]
            self.assertEquals(len_x_test, len_y_test)
            self.assertEquals(len_x_train + len_x_test, 1000)
            self.assertTrue(len_x_test == 166 or len_x_test == 167,
                            'Fold size is ' + str(len_x_test) +
                            ' but should be 166 or 167.')
            n_splits += 1
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
