import unittest

import numpy as np
from scipy import sparse

import dislib as ds
from dislib.utils import shuffle
import math

from tests import BaseTimedTestCase


class TrainTestSplitTest(BaseTimedTestCase):

    def test_train_test_split_x(self):
        x = np.random.rand(40, 40)
        x_ds = ds.array(x, (9, 9))
        train, test = ds.utils.train_test_split(x_ds)
        self.assertTrue(train.shape[0] == int((math.floor(9 * 0.75) *
                                               (train._n_blocks[0] - 1))
                                              + 4 * 0.75))
        self.assertTrue(test.shape[0] == int((math.floor(9 * 0.25) *
                                              (train._n_blocks[0] - 1))
                                             + 4 * 0.25))
        self.assertTrue(train.shape[1] == int(40))
        self.assertTrue(test.shape[1] == int(40))
        self.assertTrue(train._reg_shape[0] == int(9 * 0.75))
        self.assertTrue(test._reg_shape[0] == int(9 * 0.25))
        self.assertTrue(train._reg_shape[1] == int(9))
        self.assertTrue(test._reg_shape[1] == int(9))
        train = train.collect()
        test = test.collect()
        for idx, x_row in enumerate(x):
            found = False
            for train_row in train:
                if (train_row == x_row).all():
                    found = True
            for test_row in test:
                if (test_row == x_row).all():
                    found = True
            self.assertTrue(found)

    def test_train_test_split_x_reg_shape(self):
        x = np.random.rand(40, 40)
        x_ds = ds.array(x, (8, 8))
        train, test = ds.utils.train_test_split(x_ds)
        self.assertTrue(train.shape[0] == int(40 * 0.75))
        self.assertTrue(test.shape[0] == int(40 * 0.25))
        self.assertTrue(train.shape[1] == int(40))
        self.assertTrue(test.shape[1] == int(40))
        self.assertTrue(train._reg_shape[0] == int(8 * 0.75))
        self.assertTrue(test._reg_shape[0] == int(8 * 0.25))
        self.assertTrue(train._reg_shape[1] == int(8))
        self.assertTrue(test._reg_shape[1] == int(8))
        train = train.collect()
        test = test.collect()
        for idx, x_row in enumerate(x):
            found = False
            for train_row in train:
                if (train_row == x_row).all():
                    found = True
            for test_row in test:
                if (test_row == x_row).all():
                    found = True
            self.assertTrue(found)

    def test_train_test_split_xy_reg_shape(self):
        np.random.seed(0)
        x = np.random.rand(40, 40)
        y = np.random.rand(40, 1)
        x_ds = ds.array(x, (8, 8))
        y_ds = ds.array(y, (8, 1))
        train, test, y_train, y_test = ds.utils.train_test_split(x_ds, y_ds)
        self.assertTrue(train.shape[0] == int(40 * 0.75))
        self.assertTrue(test.shape[0] == int(40 * 0.25))
        self.assertTrue(train.shape[1] == int(40))
        self.assertTrue(test.shape[1] == int(40))
        self.assertTrue(train._reg_shape[0] == int(8 * 0.75))
        self.assertTrue(test._reg_shape[0] == int(8 * 0.25))
        self.assertTrue(train._reg_shape[1] == int(8))
        self.assertTrue(test._reg_shape[1] == int(8))
        self.assertTrue(y_train.shape[0] == int(40 * 0.75))
        self.assertTrue(y_test.shape[0] == int(40 * 0.25))
        self.assertTrue(y_train.shape[1] == 1)
        self.assertTrue(y_test.shape[1] == 1)
        self.assertTrue(y_train._reg_shape[0] == int(8 * 0.75))
        self.assertTrue(y_test._reg_shape[0] == int(8 * 0.25))
        self.assertTrue(y_train._reg_shape[1] == 1)
        self.assertTrue(y_test._reg_shape[1] == 1)
        train = train.collect()
        test = test.collect()
        for idx, x_row in enumerate(x):
            found = False
            for train_row in train:
                if (train_row == x_row).all():
                    found = True
            for test_row in test:
                if (test_row == x_row).all():
                    found = True
            self.assertTrue(found)
        y_train = y_train.collect()
        y_test = y_test.collect()
        for idx, x_row in enumerate(y):
            found = False
            for train_row in y_train:
                if (train_row == x_row).all():
                    found = True
            for test_row in y_test:
                if (test_row == x_row).all():
                    found = True
            self.assertTrue(found)

    def test_train_test_split_xy(self):
        np.random.seed(0)
        x = np.random.rand(40, 40)
        y = np.random.rand(40, 1)
        x_ds = ds.array(x, (9, 9))
        y_ds = ds.array(y, (9, 1))
        train, test, y_train, y_test = ds.utils.train_test_split(x_ds, y_ds)
        self.assertTrue(train.shape[0] == int((math.floor(9 * 0.75) *
                                               (train._n_blocks[0] - 1))
                                              + 4 * 0.75))
        self.assertTrue(test.shape[0] == int((math.ceil(9 * 0.25) *
                                              (train._n_blocks[0] - 1))
                                             + 4 * 0.25))
        self.assertTrue(train.shape[1] == int(40))
        self.assertTrue(test.shape[1] == int(40))
        self.assertTrue(train._reg_shape[0] == int(9 * 0.75))
        self.assertTrue(test._reg_shape[0] == math.ceil(9 * 0.25))
        self.assertTrue(train._reg_shape[1] == int(9))
        self.assertTrue(test._reg_shape[1] == int(9))
        self.assertTrue(y_train.shape[0] == int((math.floor(9 * 0.75) *
                                                 (train._n_blocks[0] - 1))
                                                + 4 * 0.75))
        self.assertTrue(y_test.shape[0] == int((math.ceil(9 * 0.25) *
                                                (train._n_blocks[0] - 1))
                                               + 4 * 0.25))
        self.assertTrue(y_train.shape[1] == 1)
        self.assertTrue(y_test.shape[1] == 1)
        self.assertTrue(y_train._reg_shape[0] == int(9 * 0.75))
        self.assertTrue(y_test._reg_shape[0] == math.ceil(9 * 0.25))
        self.assertTrue(y_train._reg_shape[1] == 1)
        self.assertTrue(y_test._reg_shape[1] == 1)
        train = train.collect()
        test = test.collect()
        for idx, x_row in enumerate(x):
            found = False
            for train_row in train:
                if (train_row == x_row).all():
                    found = True
            for test_row in test:
                if (test_row == x_row).all():
                    found = True
            self.assertTrue(found)
        y_train = y_train.collect()
        y_test = y_test.collect()
        for idx, x_row in enumerate(y):
            found = False
            for train_row in y_train:
                if (train_row == x_row).all():
                    found = True
            for test_row in y_test:
                if (test_row == x_row).all():
                    found = True
            self.assertTrue(found)

    def test_train_test_split_x_train_size(self):
        x = np.random.rand(40, 40)
        x_ds = ds.array(x, (10, 10))
        train, test = ds.utils.train_test_split(x_ds, train_size=0.60)
        self.assertTrue(train.shape[0] == int(40 * 0.60))
        self.assertTrue(test.shape[0] == int(40 * 0.40))
        self.assertTrue(train.shape[1] == int(40))
        self.assertTrue(test.shape[1] == int(40))
        self.assertTrue(train._reg_shape[0] == int(10 * 0.60))
        self.assertTrue(test._reg_shape[0] == int(10 * 0.40))
        self.assertTrue(train._reg_shape[1] == int(10))
        self.assertTrue(test._reg_shape[1] == int(10))
        train = train.collect()
        test = test.collect()
        for idx, x_row in enumerate(x):
            found = False
            for train_row in train:
                if (train_row == x_row).all():
                    found = True
            for test_row in test:
                if (test_row == x_row).all():
                    found = True
            self.assertTrue(found)

    def test_train_test_split_x_test_size(self):
        x = np.random.rand(40, 40)
        x_ds = ds.array(x, (8, 8))
        train, test = ds.utils.train_test_split(x_ds, test_size=0.5)
        self.assertTrue(train.shape[0] == int(40 * 0.50))
        self.assertTrue(test.shape[0] == int(40 * 0.50))
        self.assertTrue(train.shape[1] == int(40))
        self.assertTrue(test.shape[1] == int(40))
        self.assertTrue(train._reg_shape[0] == int(8 * 0.50))
        self.assertTrue(test._reg_shape[0] == int(8 * 0.50))
        self.assertTrue(train._reg_shape[1] == int(8))
        self.assertTrue(test._reg_shape[1] == int(8))
        train = train.collect()
        test = test.collect()
        for idx, x_row in enumerate(x):
            found = False
            for train_row in train:
                if (train_row == x_row).all():
                    found = True
            for test_row in test:
                if (test_row == x_row).all():
                    found = True
            self.assertTrue(found)

    def test_train_test_split_excep(self):
        x = ds.random_array((3000, 3000), (400, 400), random_state=88)
        y = ds.random_array((3000, 1), (400, 1), random_state=88)
        with self.assertRaises(ValueError):
            ds.utils.train_test_split(x, y, test_size=1.25, train_size=1)
        with self.assertRaises(ValueError):
            ds.utils.train_test_split(x, y, test_size=0.95, train_size=0.1)
        with self.assertRaises(ValueError):
            ds.utils.train_test_split(x, y=np.zeros((2, 2)), test_size=0.95,
                                      train_size=0.1)
        with self.assertRaises(ValueError):
            ds.utils.train_test_split(x=np.zeros((2, 2)), test_size=0.95,
                                      train_size=0.1)
        with self.assertRaises(ValueError):
            ds.utils.train_test_split(x, y=np.zeros((2, 2)), test_size=0.95)
        with self.assertRaises(ValueError):
            ds.utils.train_test_split(x=np.zeros((2, 2)), test_size=0.95)


class UtilsTest(BaseTimedTestCase):

    def test_shuffle_x(self):
        """ Tests shuffle for given x and random_state. Tests that the
        shuffled array contains the same rows as the original data,
        and that the position has changed for some row.
        """
        x = np.random.rand(8, 3)
        x_ds = ds.array(x, (3, 2))

        shuffled_x = shuffle(x_ds, random_state=0)
        shuffled_x = shuffled_x.collect()

        # Assert that at least one of the first 2 samples has changed
        self.assertFalse(np.array_equal(x[0:2], shuffled_x[0:2]))
        # Assert that the shuffled data has the same shape.
        self.assertEqual(shuffled_x.shape, x.shape)
        # Assert that all rows from x are found in the shuffled_x.
        for x_row in x:
            found = False
            for shuffled_idx, shuffle_x_row in enumerate(shuffled_x):
                if (shuffle_x_row == x_row).all():
                    found = True
                    break
            self.assertTrue(found)

    def test_shuffle_xy(self):
        """ Tests shuffle for given x, y and random_state. Tests that the
        shuffled arrays contain the same rows as the original data,
        and that the position has changed for some row.
        """
        np.random.seed(0)
        x = np.random.rand(8, 3)
        y = np.random.rand(8, 1)
        x_ds = ds.array(x, (3, 2))
        y_ds = ds.array(y, (4, 1))

        shuffled_x, shuffled_y = shuffle(x_ds, y_ds, random_state=0)
        shuffled_x = shuffled_x.collect()
        shuffled_y = shuffled_y.collect()

        # Assert that at least one of the first 2 samples has changed
        self.assertFalse(np.array_equal(x[0:2], shuffled_x[0:2]))
        # Assert that the shuffled data has the same shape.
        self.assertEqual(shuffled_x.shape, x.shape)
        self.assertEqual(shuffled_y.shape[0], y.shape[0])
        # Assert that all rows from x are found in the shuffled_x, and that the
        # same permutation has been used to shuffle x and y.
        for idx, x_row in enumerate(x):
            found = False
            for shuffled_idx, shuffle_x_row in enumerate(shuffled_x):
                if (shuffle_x_row == x_row).all():
                    found = True
                    self.assertEqual(y[idx], shuffled_y[shuffled_idx])
                    break
            self.assertTrue(found)

    def test_shuffle_x_sparse(self):
        """ Tests shuffle for given sparse x, and random_state. Tests that the
        shuffled array contains the same rows as the original data, and that
        the position has changed for some row.
        """
        np.random.seed(0)
        x = sparse.random(8, 10, density=0.5).tocsr()
        x_ds = ds.array(x, (3, 5))

        shuffled_x = shuffle(x_ds, random_state=0)
        shuffled_x = shuffled_x.collect()

        # Assert that at least one of the first 2 samples has changed
        self.assertFalse((x[0:2] != shuffled_x[0:2]).nnz == 0)
        # Assert that the shuffled data has the same shape.
        self.assertEqual(shuffled_x.shape, x.shape)
        # Assert that all rows from x are found in the shuffled_x.
        for x_row in x:
            found = False
            for shuffled_idx, shuffle_x_row in enumerate(shuffled_x):
                if (shuffle_x_row != x_row).nnz == 0:  # If rows are equal
                    found = True
                    break
            self.assertTrue(found)

    def test_shuffle_xy_sparse(self):
        """ Tests shuffle for given sparse x and sparse y, and random_state.
        Tests that the shuffled arrays contain the same rows as the original
        data, and that the position has changed for some row.
        """
        np.random.seed(0)
        x = sparse.random(8, 10, density=0.5).tocsr()
        x_ds = ds.array(x, (3, 5))
        y = sparse.random(8, 1, density=0.5).tocsr()
        y_ds = ds.array(y, (4, 1))

        shuffled_x, shuffled_y = shuffle(x_ds, y_ds, random_state=0)
        shuffled_x = shuffled_x.collect()
        shuffled_y = shuffled_y.collect()

        # Assert that at least one of the first 2 samples has changed
        self.assertFalse((x[0:2] != shuffled_x[0:2]).nnz == 0)
        # Assert that the shuffled data has the same shape.
        self.assertEqual(shuffled_x.shape, x.shape)
        self.assertEqual(shuffled_y.shape[0], y.shape[0])
        # Assert that all rows from x are found in the shuffled_x, and that the
        # same permutation has been used to shuffle x and y.
        for idx, x_row in enumerate(x):
            found = False
            for shuffled_idx, shuffle_x_row in enumerate(shuffled_x):
                if (shuffle_x_row != x_row).nnz == 0:  # If rows are equal
                    found = True
                    self.assertEqual(y[idx, 0], shuffled_y[shuffled_idx, 0])
                    break
            self.assertTrue(found)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
