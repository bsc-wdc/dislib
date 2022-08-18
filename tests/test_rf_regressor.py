import unittest

import numpy as np
from pycompss.api.api import compss_wait_on
from sklearn.datasets import make_regression

import dislib as ds
from dislib.regression import RandomForestRegressor
import dislib.data.util.model as utilmodel

from tests import BaseTimedTestCase


def _determination_coefficient(y_true, y_pred):
    u = np.sum(np.square(y_true - y_pred))
    v = np.sum(np.square(y_true - np.mean(y_true)))
    return 1 - u / v


class RandomForestRegressorTest(BaseTimedTestCase):
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

    def test_save_load(self):
        """Tests the save and the load methods of the RandomForestRegressor
        class"""
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

        rf = RandomForestRegressor(random_state=0, n_estimators=2)
        rf.fit(x_train, y_train)
        rf.save_model("./rf_regressor")

        rf2 = RandomForestRegressor(random_state=0, n_estimators=2)
        rf2.load_model("./rf_regressor")

        accuracy1 = compss_wait_on(rf.score(x_test, y_test))
        accuracy_loaded1 = compss_wait_on(rf2.score(x_test, y_test))

        y_pred = rf.predict(x_test).collect()
        y_pred_loaded = rf2.predict(x_test).collect()
        y_true = y[1::2]
        accuracy2 = _determination_coefficient(y_true, y_pred)
        accuracy_loaded2 = _determination_coefficient(y_true, y_pred_loaded)

        self.assertEqual(accuracy1, accuracy_loaded1)
        self.assertEqual(accuracy2, accuracy_loaded2)

        rf.save_model("./rf_regressor", save_format="cbor")

        rf2 = RandomForestRegressor(random_state=0, n_estimators=2)
        rf2.load_model("./rf_regressor", load_format="cbor")

        accuracy_loaded1 = compss_wait_on(rf2.score(x_test, y_test))

        y_pred_loaded = rf2.predict(x_test).collect()
        accuracy_loaded2 = _determination_coefficient(y_true, y_pred_loaded)

        self.assertEqual(accuracy1, accuracy_loaded1)
        self.assertEqual(accuracy2, accuracy_loaded2)

        rf.save_model("./rf_regressor", save_format="pickle")

        rf2 = RandomForestRegressor(random_state=0, n_estimators=2)
        rf2.load_model("./rf_regressor", load_format="pickle")

        accuracy_loaded1 = compss_wait_on(rf2.score(x_test, y_test))

        y_pred_loaded = rf2.predict(x_test).collect()
        accuracy_loaded2 = _determination_coefficient(y_true, y_pred_loaded)

        self.assertEqual(accuracy1, accuracy_loaded1)
        self.assertEqual(accuracy2, accuracy_loaded2)

        with self.assertRaises(ValueError):
            rf.save_model("./rf_regressor", save_format="txt")

        with self.assertRaises(ValueError):
            rf2 = RandomForestRegressor(random_state=0, n_estimators=2)
            rf2.load_model("./rf_regressor", load_format="txt")

        rf1 = RandomForestRegressor(random_state=0, n_estimators=1)
        rf1.save_model("./rf_regressor", overwrite=False)

        rf2 = RandomForestRegressor(random_state=0, n_estimators=2)
        rf2.load_model("./rf_regressor", load_format="pickle")

        accuracy_loaded1 = compss_wait_on(rf2.score(x_test, y_test))

        y_pred_loaded = rf2.predict(x_test).collect()
        accuracy_loaded2 = _determination_coefficient(y_true, y_pred_loaded)

        self.assertEqual(accuracy1, accuracy_loaded1)
        self.assertEqual(accuracy2, accuracy_loaded2)

        cbor2_module = utilmodel.cbor2
        utilmodel.cbor2 = None
        with self.assertRaises(ModuleNotFoundError):
            rf.save_model("./rf_regressor", save_format="cbor")
        with self.assertRaises(ModuleNotFoundError):
            rf2.load_model("./rf_regressor", load_format="cbor")
        utilmodel.cbor2 = cbor2_module


def main():
    unittest.main()


if __name__ == "__main__":
    main()
