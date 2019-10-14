import unittest

import numpy as np
from scipy.sparse import random as sp_random
from pycompss.api.api import compss_wait_on

import dislib as ds
from dislib.regression import LinearRegression
from dislib.data import random_array


class LinearRegressionTest(unittest.TestCase):

    def test_fit_and_predict(self):
        """Tests LinearRegression's fit() and predict()"""
        x_data = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
        y_data = np.array([2, 1, 1, 2, 4.5]).reshape(-1, 1)

        bn, bm = 2, 2

        x = ds.array(x=x_data, block_size=(bn, bm))
        y = ds.array(x=y_data, block_size=(bn, bm))

        reg = LinearRegression()
        reg.fit(x, y)
        # y = 0.6 * x + 0.3

        reg.coef_ = compss_wait_on(reg.coef_)
        reg.intercept_ = compss_wait_on(reg.intercept_)

        self.assertTrue(np.allclose(reg.coef_, 0.6))
        self.assertTrue(np.allclose(reg.intercept_, 0.3))

        x_test = np.array([3, 5]).reshape(-1, 1)
        test_data = ds.array(x=x_test, block_size=(bn, bm))
        pred = reg.predict(test_data).collect()

        self.assertTrue(np.allclose(pred, [2.1, 3.3]))

    def test_sparse(self):
        """Tests LR raises NotImplementedError for sparse data."""
        np.random.seed(0)
        coo_matrix = sp_random(10, 1, density=0.5)
        sparse_arr = ds.array(x=coo_matrix, block_size=(5, 1))
        reg = LinearRegression()
        with self.assertRaises(NotImplementedError):
            reg.fit(sparse_arr, sparse_arr)
        dense_arr = random_array((10, 1), (5, 1))
        reg.fit(dense_arr, dense_arr)
        with self.assertRaises(NotImplementedError):
            reg.predict(sparse_arr)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
