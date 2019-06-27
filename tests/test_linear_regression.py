import unittest

import numpy as np
from pycompss.api.api import compss_wait_on

from dislib.data import load_data
from dislib.regression import LinearRegression


class LinearRegressionTest(unittest.TestCase):

    def test_fit_and_predict(self):
        """Tests LinearRegression's fit() and predict()"""
        x = np.array([1, 2, 3, 4, 5])[:, np.newaxis]
        y = np.array([2, 1, 1, 2, 4.5])
        train_data = load_data(x=x, y=y, subset_size=2)
        reg = LinearRegression()
        reg.fit(train_data)
        # y = 0.6 * x + 0.3

        reg.coef_ = compss_wait_on(reg.coef_)
        reg.intercept_ = compss_wait_on(reg.intercept_)

        self.assertTrue(np.allclose(reg.coef_, 0.6))
        self.assertTrue(np.allclose(reg.intercept_, 0.3))

        x_test = np.array([3, 5])[:, np.newaxis]
        test_data = load_data(x=x_test, subset_size=2)
        reg.predict(test_data)
        prediction = test_data.labels
        self.assertTrue(np.allclose(prediction, [2.1, 3.3]))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
