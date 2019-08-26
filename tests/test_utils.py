import unittest

import numpy as np
from scipy import sparse

import dislib as ds
from dislib.utils import shuffle


class UtilsTest(unittest.TestCase):
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
