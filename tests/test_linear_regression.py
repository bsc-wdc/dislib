import unittest

import numpy as np
from pycompss.api.api import compss_wait_on

import dislib as ds
from dislib.regression import LinearRegression


class LinearRegressionTest(unittest.TestCase):

    def test_fit_and_predict(self):
        """Tests LinearRegression's fit() and predict()"""
        x_data = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
        y_data = np.array([2, 1, 1, 2, 4.5]).reshape(-1, 1)

        bn, bm = 2, 2

        x = ds.array(x=x_data, blocks_shape=(bn, bm))
        y = ds.array(x=y_data, blocks_shape=(bn, bm))

        reg = LinearRegression()
        reg.fit(x, y)
        # y = 0.6 * x + 0.3

        reg.coef_ = compss_wait_on(reg.coef_)
        reg.intercept_ = compss_wait_on(reg.intercept_)

        self.assertTrue(np.allclose(reg.coef_, 0.6))
        self.assertTrue(np.allclose(reg.intercept_, 0.3))

        x_test = np.array([3, 5]).reshape(-1, 1)
        test_data = ds.array(x=x_test, blocks_shape=(bn, bm))
        pred = reg.predict(test_data).collect()

        self.assertTrue(np.allclose(pred, [2.1, 3.3]))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
