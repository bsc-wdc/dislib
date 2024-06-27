import numpy as np
from sklearn.metrics import r2_score

import dislib as ds
from dislib.regression import Lasso
import dislib.data.util.model as utilmodel
from tests import BaseTimedTestCase


class LassoTest(BaseTimedTestCase):

    def test_fit_predict(self):
        """ Tests fit and predicts methods """

        np.random.seed(42)

        n_samples, n_features = 50, 100
        X = np.random.randn(n_samples, n_features)

        # Decreasing coef w. alternated signs for visualization
        idx = np.arange(n_features)
        coef = (-1) ** idx * np.exp(-idx / 10)
        coef[10:] = 0  # sparsify coef
        y = np.dot(X, coef)

        # Add noise
        y += 0.01 * np.random.normal(size=n_samples)

        n_samples = X.shape[0]
        X_train, y_train = X[:n_samples // 2], y[:n_samples // 2]
        X_test, y_test = X[n_samples // 2:], y[n_samples // 2:]

        lasso = Lasso(lmbd=0.1, max_iter=50)

        lasso.fit(ds.array(X_train, (5, 100)), ds.array(y_train, (5, 1)))
        y_pred_lasso = lasso.predict(ds.array(X_test, (25, 100)))
        r2_score_lasso = r2_score(y_test, y_pred_lasso.collect())

        self.assertAlmostEqual(r2_score_lasso, 0.9481746925431124)

    def test_save_load(self):
        """ Tests load and save methods """

        np.random.seed(42)

        n_samples, n_features = 50, 100
        X = np.random.randn(n_samples, n_features)

        # Decreasing coef w. alternated signs for visualization
        idx = np.arange(n_features)
        coef = (-1) ** idx * np.exp(-idx / 10)
        coef[10:] = 0  # sparsify coef
        y = np.dot(X, coef)

        # Add noise
        y += 0.01 * np.random.normal(size=n_samples)

        n_samples = X.shape[0]
        X_train, y_train = X[:n_samples // 2], y[:n_samples // 2]
        X_test, y_test = X[n_samples // 2:], y[n_samples // 2:]

        lasso = Lasso(lmbd=0.1, max_iter=50)

        lasso.fit(ds.array(X_train, (5, 100)), ds.array(y_train, (5, 1)))
        lasso.save_model("./lasso_model")

        lasso2 = Lasso()
        lasso2.load_model("./lasso_model")
        y_pred_lasso = lasso2.predict(ds.array(X_test, (25, 100)))
        r2_score_lasso = r2_score(y_test, y_pred_lasso.collect())

        self.assertAlmostEqual(r2_score_lasso, 0.9481746925431124)

        lasso.save_model("./lasso_model", save_format="cbor")

        lasso2 = Lasso()
        lasso2.load_model("./lasso_model", load_format="cbor")
        y_pred_lasso = lasso2.predict(ds.array(X_test, (25, 100)))
        r2_score_lasso = r2_score(y_test, y_pred_lasso.collect())

        self.assertAlmostEqual(r2_score_lasso, 0.9481746925431124)

        lasso.save_model("./lasso_model", save_format="pickle")

        lasso2 = Lasso()
        lasso2.load_model("./lasso_model", load_format="pickle")
        y_pred_lasso = lasso2.predict(ds.array(X_test, (25, 100)))
        r2_score_lasso = r2_score(y_test, y_pred_lasso.collect())

        self.assertAlmostEqual(r2_score_lasso, 0.9481746925431124)

        with self.assertRaises(ValueError):
            lasso.save_model("./lasso_model", save_format="txt")

        with self.assertRaises(ValueError):
            lasso2 = Lasso()
            lasso2.load_model("./lasso_model", load_format="txt")

        y2 = np.dot(X, coef)
        y2 += 0.1 * np.random.normal(size=n_samples)

        n_samples = X.shape[0]
        X_train, y_train = X[:n_samples // 2], y2[:n_samples // 2]

        lasso = Lasso(lmbd=0.1, max_iter=50)

        lasso.fit(ds.array(X_train, (5, 100)), ds.array(y_train, (5, 1)))
        lasso.save_model("./lasso_model", overwrite=False)

        lasso2 = Lasso()
        lasso2.load_model("./lasso_model", load_format="pickle")
        y_pred_lasso = lasso2.predict(ds.array(X_test, (25, 100)))
        r2_score_lasso = r2_score(y_test, y_pred_lasso.collect())

        self.assertAlmostEqual(r2_score_lasso, 0.9481746925431124)

        cbor2_module = utilmodel.cbor2
        utilmodel.cbor2 = None
        with self.assertRaises(ModuleNotFoundError):
            lasso.save_model("./lasso_model", save_format="cbor")
        with self.assertRaises(ModuleNotFoundError):
            lasso2.load_model("./lasso_model", load_format="cbor")
        utilmodel.cbor2 = cbor2_module
