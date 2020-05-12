import unittest

import numpy as np
from scipy.sparse import random as sp_random

import dislib as ds
from dislib.regression import LinearRegression
from dislib.data import random_array


class LinearRegressionTest(unittest.TestCase):

    def test_univariate(self):
        """Tests fit() and predict(), univariate."""
        x_data = np.array([1, 2, 3, 4, 5])
        y_data = np.array([2, 1, 1, 2, 4.5])

        bn, bm = 2, 1

        x = ds.array(x=x_data, block_size=(bn, bm))
        y = ds.array(x=y_data, block_size=(bn, bm))

        reg = LinearRegression()
        reg.fit(x, y)
        self.assertTrue(np.allclose(reg.coef_.collect(), 0.6))
        self.assertTrue(np.allclose(reg.intercept_.collect(), 0.3))

        # Predict one sample
        x_test = np.array([3])
        test_data = ds.array(x=x_test, block_size=(1, 1))
        pred = reg.predict(test_data).collect()
        self.assertTrue(np.allclose(pred, 2.1))

        # Predict multiple samples
        x_test = np.array([3, 5, 6])
        test_data = ds.array(x=x_test, block_size=(bn, bm))
        pred = reg.predict(test_data).collect()
        self.assertTrue(np.allclose(pred, [2.1, 3.3, 3.9]))

    def test_univariate_no_intercept(self):
        """Tests fit() and predict(), univariate, fit_intercept=False."""
        x_data = np.array([1, 2, 3, 4, 5])
        y_data = np.array([2, 1, 1, 2, 4.5])

        bn, bm = 2, 1

        x = ds.array(x=x_data, block_size=(bn, bm))
        y = ds.array(x=y_data, block_size=(bn, bm))

        reg = LinearRegression(fit_intercept=False)
        reg.fit(x, y)
        self.assertTrue(np.allclose(reg.coef_.collect(), 0.68181818))
        self.assertTrue(np.allclose(reg.intercept_.collect(), 0))

        # Predict one sample
        x_test = np.array([3])
        test_data = ds.array(x=x_test, block_size=(1, 1))
        pred = reg.predict(test_data).collect()
        self.assertTrue(np.allclose(pred, 2.04545455))

        # Predict multiple samples
        x_test = np.array([3, 5, 6])
        test_data = ds.array(x=x_test, block_size=(bn, bm))
        pred = reg.predict(test_data).collect()
        self.assertTrue(np.allclose(pred, [2.04545455, 3.4090909, 4.0909091]))

    def test_multivariate(self):
        """Tests fit() and predict(), multivariate."""
        x_data = np.array([[1, 2], [2, 0], [3, 1], [4, 4], [5, 3]])
        y_data = np.array([2, 1, 1, 2, 4.5])

        bn, bm = 2, 2

        x = ds.array(x=x_data, block_size=(bn, bm))
        y = ds.array(x=y_data, block_size=(bn, 1))

        reg = LinearRegression()
        reg.fit(x, y)
        self.assertTrue(np.allclose(reg.coef_.collect(), [0.421875, 0.296875]))
        self.assertTrue(np.allclose(reg.intercept_.collect(), 0.240625))

        # Predict one sample
        x_test = np.array([3, 2])
        test_data = ds.array(x=x_test, block_size=(1, bm))
        pred = reg.predict(test_data).collect()
        self.assertTrue(np.allclose(pred, 2.1))

        # Predict multiple samples
        x_test = np.array([[3, 2], [4, 4], [1, 3]])
        test_data = ds.array(x=x_test, block_size=(bn, bm))
        pred = reg.predict(test_data).collect()
        self.assertTrue(np.allclose(pred, [2.1, 3.115625, 1.553125]))

    def test_multivariate_no_intercept(self):
        """Tests fit() and predict(), multivariate, fit_intercept=False."""
        x_data = np.array([[1, 2], [2, 0], [3, 1], [4, 4], [5, 3]])
        y_data = np.array([2, 1, 1, 2, 4.5])

        bn, bm = 2, 2

        x = ds.array(x=x_data, block_size=(bn, bm))
        y = ds.array(x=y_data, block_size=(bn, 1))

        reg = LinearRegression(fit_intercept=False)
        reg.fit(x, y)
        self.assertTrue(np.allclose(reg.coef_.collect(),
                                    [0.48305085, 0.30367232]))
        self.assertTrue(np.allclose(reg.intercept_.collect(), 0))

        # Predict one sample
        x_test = np.array([3, 2])
        test_data = ds.array(x=x_test, block_size=(1, bm))
        pred = reg.predict(test_data).collect()
        self.assertTrue(np.allclose(pred, [2.05649718]))

        # Predict multiple samples
        x_test = np.array([[3, 2], [4, 4], [1, 3]])
        test_data = ds.array(x=x_test, block_size=(bn, bm))
        pred = reg.predict(test_data).collect()
        self.assertTrue(np.allclose(pred, [2.05649718, 3.14689266, 1.3940678]))

    def test_multivariate_multiobjective(self):
        """Tests fit() and predict(), multivariate, multiobjective."""
        x_data = np.array([[1, 2, 3], [2, 0, 4], [3, 1, 8],
                           [4, 4, 2], [5, 3, 1], [2, 7, 1]])
        y_data = np.array([[2, 0, 3], [1, 5, 2], [1, 3, 4],
                           [2, 7, 9], [4.5, -1, 4], [0, 0, 0]])

        bn, bm = 2, 2

        x = ds.array(x=x_data, block_size=(bn, bm))
        y = ds.array(x=y_data, block_size=(bn, bm))

        reg = LinearRegression()
        reg.fit(x, y)

        # Predict one sample
        x_test = np.array([3, 2, 1])
        test_data = ds.array(x=x_test, block_size=(1, bm))
        pred = reg.predict(test_data).collect()
        self.assertTrue(np.allclose(pred, [3.0318415, 1.97164872, 3.85410906]))

        # Predict multiple samples
        x_test = np.array([[3, 2, 1], [4, 3, 3], [1, 1, 1]])
        test_data = ds.array(x=x_test, block_size=(bn, bm))
        pred = reg.predict(test_data).collect()
        self.assertTrue(np.allclose(pred, [[3.0318415, 1.97164872, 3.85410906],
                                           [2.5033157, 2.65809327, 5.05310495],
                                           [2.145797, 1.4840121, 1.5739791]]))

        # Check attributes values
        self.assertTrue(np.allclose(reg.coef_.collect(),
                                    [[0.65034768, 0.34673933, 1.22176283],
                                     [-0.41465084, -0.20584208, -0.16339571],
                                     [-0.38211131, 0.27277365, 0.07031439]]))
        self.assertTrue(np.allclose(reg.intercept_.collect(),
                                    [2.29221145, 1.07034124, 0.44529761]))

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
