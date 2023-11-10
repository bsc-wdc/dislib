import numpy as np
from pycompss.api.api import compss_wait_on
from sklearn.datasets import make_regression

import dislib as ds
from dislib.regression import RandomForestRegressor
import dislib.data.util.model as utilmodel

from tests import BaseTimedTestCase
from pycompss.api.task import task
from math import isclose


def _determination_coefficient(y_true, y_pred):
    u = np.sum(np.square(y_true - y_pred))
    v = np.sum(np.square(y_true - np.mean(y_true)))
    return 1 - u / v


def test_make_regression():
    """Tests RandomForestRegressor fit and score with default params."""
    x, y = make_regression(
        n_samples=12000,
        n_features=40,
        n_informative=4,
        shuffle=True,
        random_state=0,
    )
    x_train = ds.array(x[::2], (4000, 20))
    y_train = ds.array(y[::2][:, np.newaxis], (4000, 1))
    x_test = ds.array(x[1::2], (4000, 20))
    y_test = ds.array(y[1::2][:, np.newaxis], (4000, 1))

    rf = RandomForestRegressor(distr_depth=1, random_state=0,
                               n_estimators=2, mmap=False,
                               nested=True)

    rf.fit(x_train, y_train)
    accuracy1 = compss_wait_on(rf.score(x_test, y_test))

    y_pred = rf.predict(x_test).collect()
    y_true = y[1::2]
    accuracy2 = _determination_coefficient(y_true, y_pred)

    return accuracy1 > 0.5 and accuracy2 > 0.5 and \
        isclose(accuracy1, accuracy2)


def test_make_regression_predict_and_distr_depth():
    """Tests RandomForestRegressor fit and predict with a distr_depth."""
    x, y = make_regression(
        n_samples=3000,
        n_features=10,
        n_informative=4,
        shuffle=True,
        random_state=0,
    )
    x_train = ds.array(x[::2], (1000, 10))
    y_train = ds.array(y[::2][:, np.newaxis], (1000, 1))
    x_test = ds.array(x[1::2], (1000, 10))
    y_test = ds.array(y[1::2][:, np.newaxis], (1000, 1))

    rf = RandomForestRegressor(distr_depth=1, random_state=0,
                               n_estimators=2,
                               mmap=False, nested=True)

    rf.fit(x_train, y_train)
    accuracy1 = compss_wait_on(rf.score(x_test, y_test))

    y_pred = rf.predict(x_test).collect()
    y_true = y[1::2]
    accuracy2 = _determination_coefficient(y_true, y_pred)

    return accuracy1 > 0.75 and accuracy2 > 0.75 and \
        isclose(accuracy1, accuracy2)


def test_make_regression_sklearn_max_predict():
    """Tests RandomForestRegressor predict with sklearn_max."""
    x, y = make_regression(
        n_samples=3000,
        n_features=10,
        n_informative=4,
        shuffle=True,
        random_state=0,
    )
    x_train = ds.array(x[::2], (1000, 10))
    y_train = ds.array(y[::2][:, np.newaxis], (1000, 1))
    x_test = ds.array(x[1::2], (1000, 10))
    y_test = ds.array(y[1::2][:, np.newaxis], (1000, 1))

    rf = RandomForestRegressor(distr_depth=2, n_estimators=2,
                               random_state=0, sklearn_max=10, mmap=False,
                               nested=True)

    rf.fit(x_train, y_train)
    accuracy1 = compss_wait_on(rf.score(x_test, y_test))

    y_pred = rf.predict(x_test).collect()
    y_true = y[1::2]
    accuracy2 = _determination_coefficient(y_true, y_pred)

    return accuracy1 > 0.75 and accuracy2 > 0.75 and \
        isclose(accuracy1, accuracy2)


def test_save_load():
    """Tests the save and the load methods of the RandomForestRegressor
    class"""
    x, y = make_regression(
        n_samples=3000,
        n_features=10,
        n_informative=4,
        shuffle=True,
        random_state=0,
    )
    x_train = ds.array(x[::2], (1000, 10))
    y_train = ds.array(y[::2][:, np.newaxis], (1000, 1))
    x_test = ds.array(x[1::2], (1000, 10))
    y_test = ds.array(y[1::2][:, np.newaxis], (1000, 1))

    rf = RandomForestRegressor(distr_depth=1,
                               random_state=0, n_estimators=2, mmap=False,
                               nested=True)
    rf.fit(x_train, y_train)
    rf.save_model("./rf_regressor")

    rf2 = RandomForestRegressor(distr_depth=1,
                                random_state=0, n_estimators=2, mmap=False,
                                nested=True)
    rf2.load_model("./rf_regressor")

    accuracy1 = compss_wait_on(rf.score(x_test, y_test))
    accuracy_loaded1 = compss_wait_on(rf2.score(x_test, y_test))

    y_pred = rf.predict(x_test).collect()
    y_pred_loaded = rf2.predict(x_test).collect()
    y_true = y[1::2]
    accuracy2 = _determination_coefficient(y_true, y_pred)
    accuracy_loaded2 = _determination_coefficient(y_true, y_pred_loaded)

    condition = accuracy1 == accuracy_loaded1
    condition = condition and accuracy2 == accuracy_loaded2

    rf.save_model("./rf_regressor", save_format="cbor")

    rf2 = RandomForestRegressor(distr_depth=1,
                                random_state=0, n_estimators=2, mmap=False,
                                nested=True)
    rf2.load_model("./rf_regressor", load_format="cbor")

    accuracy_loaded1 = compss_wait_on(rf2.score(x_test, y_test))

    y_pred_loaded = rf2.predict(x_test).collect()
    accuracy_loaded2 = _determination_coefficient(y_true, y_pred_loaded)

    condition = condition and accuracy1 == accuracy_loaded1
    condition = condition and accuracy2 == accuracy_loaded2

    rf.save_model("./rf_regressor", save_format="pickle")

    rf2 = RandomForestRegressor(distr_depth=1,
                                random_state=0, n_estimators=2, mmap=False,
                                nested=True)
    rf2.load_model("./rf_regressor", load_format="pickle")

    accuracy_loaded1 = compss_wait_on(rf2.score(x_test, y_test))

    y_pred_loaded = rf2.predict(x_test).collect()
    accuracy_loaded2 = _determination_coefficient(y_true, y_pred_loaded)

    condition = condition and accuracy1 == accuracy_loaded1
    condition = condition and accuracy2 == accuracy_loaded2

    try:
        rf.save_model("./rf_regressor", save_format="txt")
    except ValueError:
        condition_error = True
    condition = condition and condition_error

    try:
        rf2 = RandomForestRegressor(distr_depth=1,
                                    random_state=0, n_estimators=2,
                                    mmap=False, nested=True)
        rf2.load_model("./rf_regressor", load_format="txt")
    except ValueError:
        condition_error = True
    condition = condition and condition_error

    rf1 = RandomForestRegressor(distr_depth=1,
                                random_state=0, n_estimators=1,
                                mmap=False, nested=True)
    rf1.save_model("./rf_regressor", overwrite=False)

    rf2 = RandomForestRegressor(distr_depth=1,
                                random_state=0, n_estimators=2,
                                mmap=False, nested=True)
    rf2.load_model("./rf_regressor", load_format="pickle")

    accuracy_loaded1 = compss_wait_on(rf2.score(x_test, y_test))

    y_pred_loaded = rf2.predict(x_test).collect()
    accuracy_loaded2 = _determination_coefficient(y_true, y_pred_loaded)

    condition = condition and accuracy1 == accuracy_loaded1
    condition = condition and accuracy2 == accuracy_loaded2

    cbor2_module = utilmodel.cbor2
    utilmodel.cbor2 = None
    try:
        rf.save_model("./rf_regressor", save_format="cbor")
    except ModuleNotFoundError:
        condition_error = True
    condition = condition and condition_error
    try:
        rf2.load_model("./rf_regressor", load_format="cbor")
    except ModuleNotFoundError:
        condition_error = True
    condition = condition and condition_error
    utilmodel.cbor2 = cbor2_module
    return condition


class RandomForestRegressorTest(BaseTimedTestCase):
    def test_make_regression(self):
        """Tests RandomForestRegressor fit and score with default params."""
        x, y = make_regression(
            n_samples=12000,
            n_features=40,
            n_informative=4,
            shuffle=True,
            random_state=0,
        )
        x_train = ds.array(x[::2], (4000, 20))
        y_train = ds.array(y[::2][:, np.newaxis], (4000, 1))
        x_test = ds.array(x[1::2], (4000, 20))
        y_test = ds.array(y[1::2][:, np.newaxis], (4000, 1))

        rf = RandomForestRegressor(distr_depth=2, random_state=0,
                                   n_estimators=2, mmap=False, nested=True)

        rf.fit(x_train, y_train)
        accuracy1 = compss_wait_on(rf.score(x_test, y_test))

        y_pred = rf.predict(x_test).collect()
        y_true = y[1::2]
        accuracy2 = _determination_coefficient(y_true, y_pred)

        self.assertGreater(accuracy1, 0.50)
        self.assertGreater(accuracy2, 0.50)
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
        x_train = ds.array(x[::2], (1000, 10))
        y_train = ds.array(y[::2][:, np.newaxis], (1000, 1))
        x_test = ds.array(x[1::2], (1000, 10))
        y_test = ds.array(y[1::2][:, np.newaxis], (1000, 1))

        rf = RandomForestRegressor(distr_depth=1, random_state=0,
                                   n_estimators=2,
                                   mmap=False, nested=True)

        rf.fit(x_train, y_train)
        accuracy1 = compss_wait_on(rf.score(x_test, y_test))

        y_pred = rf.predict(x_test).collect()
        y_true = y[1::2]
        accuracy2 = _determination_coefficient(y_true, y_pred)

        self.assertGreater(accuracy1, 0.6)
        self.assertGreater(accuracy2, 0.6)
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
        x_train = ds.array(x[::2], (1000, 10))
        y_train = ds.array(y[::2][:, np.newaxis], (1000, 1))
        x_test = ds.array(x[1::2], (1000, 10))
        y_test = ds.array(y[1::2][:, np.newaxis], (1000, 1))

        rf = RandomForestRegressor(distr_depth=1, n_estimators=2,
                                   random_state=0, sklearn_max=10, mmap=False,
                                   nested=True)

        rf.fit(x_train, y_train)
        accuracy1 = compss_wait_on(rf.score(x_test, y_test))

        y_pred = rf.predict(x_test).collect()
        y_true = y[1::2]
        accuracy2 = _determination_coefficient(y_true, y_pred)

        self.assertGreater(accuracy1, 0.75)
        self.assertGreater(accuracy2, 0.75)
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
        x_train = ds.array(x[::2], (1000, 10))
        y_train = ds.array(y[::2][:, np.newaxis], (1000, 1))
        x_test = ds.array(x[1::2], (1000, 10))
        y_test = ds.array(y[1::2][:, np.newaxis], (1000, 1))

        rf = RandomForestRegressor(distr_depth=1,
                                   random_state=0, n_estimators=2, mmap=False,
                                   nested=True)
        rf.fit(x_train, y_train)
        rf.save_model("./rf_regressor")

        rf2 = RandomForestRegressor(distr_depth=1,
                                    random_state=0, n_estimators=2, mmap=False,
                                    nested=True)
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

        rf2 = RandomForestRegressor(distr_depth=1,
                                    random_state=0, n_estimators=2, mmap=False,
                                    nested=True)
        rf2.load_model("./rf_regressor", load_format="cbor")

        accuracy_loaded1 = compss_wait_on(rf2.score(x_test, y_test))

        y_pred_loaded = rf2.predict(x_test).collect()
        accuracy_loaded2 = _determination_coefficient(y_true, y_pred_loaded)

        self.assertEqual(accuracy1, accuracy_loaded1)
        self.assertEqual(accuracy2, accuracy_loaded2)

        rf.save_model("./rf_regressor", save_format="pickle")

        rf2 = RandomForestRegressor(distr_depth=1,
                                    random_state=0, n_estimators=2, mmap=False,
                                    nested=True)
        rf2.load_model("./rf_regressor", load_format="pickle")

        accuracy_loaded1 = compss_wait_on(rf2.score(x_test, y_test))

        y_pred_loaded = rf2.predict(x_test).collect()
        accuracy_loaded2 = _determination_coefficient(y_true, y_pred_loaded)

        self.assertEqual(accuracy1, accuracy_loaded1)
        self.assertEqual(accuracy2, accuracy_loaded2)

        with self.assertRaises(ValueError):
            rf.save_model("./rf_regressor", save_format="txt")

        with self.assertRaises(ValueError):
            rf2 = RandomForestRegressor(distr_depth=1,
                                        random_state=0, n_estimators=2,
                                        mmap=False, nested=True)
            rf2.load_model("./rf_regressor", load_format="txt")

        rf1 = RandomForestRegressor(distr_depth=1,
                                    random_state=0, n_estimators=1,
                                    mmap=False, nested=True)
        rf1.save_model("./rf_regressor", overwrite=False)

        rf2 = RandomForestRegressor(distr_depth=1,
                                    random_state=0, n_estimators=2,
                                    mmap=False, nested=True)
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


@task()
def main():
    test = test_make_regression()
    test2 = test_make_regression_predict_and_distr_depth()
    test3 = test_make_regression_sklearn_max_predict()
    test4 = test_save_load()
    print("TEST", flush=True)
    print(test)
    print(test2)
    print(test3)
    print(test4, flush=True)
    test = test and test2 and test3 and test4
    if test:
        print("Result tests: Passed", flush=True)
    else:
        print("Result tests: Failed", flush=True)


if __name__ == "__main__":
    main()
