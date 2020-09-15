import unittest

import numpy as np
from sklearn.metrics import r2_score

import dislib as ds
from dislib.regression import Lasso


class LassoTest(unittest.TestCase):

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

        self.assertEqual(r2_score_lasso, 0.9481746925431124)
