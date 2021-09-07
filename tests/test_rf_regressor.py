import unittest

import numpy as np
from pycompss.api.api import compss_wait_on
from sklearn.datasets import make_regression

import dislib as ds
from dislib.regression import RandomForestRegressor


def _determination_coefficient(y_true, y_pred):
    u = np.sum(np.square(y_true - y_pred))
    v = np.sum(np.square(y_true - np.mean(y_true)))
    return 1 - u / v


class RandomForestRegressorTest(unittest.TestCase):
    def test_make_regression(self):
        """Tests RandomForestRegressor fit and score with default params."""
        x, y = make_regression(
            n_samples=3000,
            n_features=10,
            n_informative=4,
            shuffle=True,
            random_state=0,
        )
        x_train = ds.array(x[::2], (300, 10))
        y_train = ds.array(y[::2][:, np.newaxis], (300, 1))
        x_test = ds.array(x[1::2], (300, 10))
        y_test = ds.array(y[1::2][:, np.newaxis], (300, 1))

        rf = RandomForestRegressor(random_state=0)

        rf.fit(x_train, y_train)
        accuracy1 = compss_wait_on(rf.score(x_test, y_test))

        y_pred = rf.predict(x_test).collect()
        y_true = y[1::2]
        accuracy2 = _determination_coefficient(y_true, y_pred)

        self.assertGreater(accuracy1, 0.85)
        self.assertGreater(accuracy2, 0.85)
        self.assertAlmostEqual(accuracy1, accuracy2)

    def test_make_regression_predict_and_distr_depth(self):
        """Tests RandomForestRegressor fit and predict with a distr_depth."""
        x, y = make_regression(
            n_samples=3000,
            n_features=10,
            n_informative=4,
            shuffle=True,
            random_state=0,
        )
        x_train = ds.array(x[::2], (300, 10))
        y_train = ds.array(y[::2][:, np.newaxis], (300, 1))
        x_test = ds.array(x[1::2], (300, 10))
        y_test = ds.array(y[1::2][:, np.newaxis], (300, 1))

        rf = RandomForestRegressor(distr_depth=2, random_state=0)

        rf.fit(x_train, y_train)
        accuracy1 = compss_wait_on(rf.score(x_test, y_test))

        y_pred = rf.predict(x_test).collect()
        y_true = y[1::2]
        accuracy2 = _determination_coefficient(y_true, y_pred)

        self.assertGreater(accuracy1, 0.85)
        self.assertGreater(accuracy2, 0.85)
        self.assertAlmostEqual(accuracy1, accuracy2)

    def test_make_regression_sklearn_max_predict(self):
        """Tests RandomForestRegressor predict with sklearn_max."""
        x, y = make_regression(
            n_samples=3000,
            n_features=10,
            n_informative=4,
            shuffle=True,
            random_state=0,
        )
        x_train = ds.array(x[::2], (300, 10))
        y_train = ds.array(y[::2][:, np.newaxis], (300, 1))
        x_test = ds.array(x[1::2], (300, 10))
        y_test = ds.array(y[1::2][:, np.newaxis], (300, 1))

        rf = RandomForestRegressor(random_state=0, sklearn_max=10)

        rf.fit(x_train, y_train)
        accuracy1 = compss_wait_on(rf.score(x_test, y_test))

        y_pred = rf.predict(x_test).collect()
        y_true = y[1::2]
        accuracy2 = _determination_coefficient(y_true, y_pred)

        self.assertGreater(accuracy1, 0.85)
        self.assertGreater(accuracy2, 0.85)
        self.assertAlmostEqual(accuracy1, accuracy2)


def main():
    unittest.main()


if __name__ == "__main__":
    main()
