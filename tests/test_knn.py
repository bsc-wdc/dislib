import unittest

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import KNeighborsClassifier as skKNeighborsClassifier
from sklearn.datasets import make_classification

import dislib as ds
from dislib.classification import KNeighborsClassifier


class KNearestNeighborsTest(unittest.TestCase):

    def test_kneighbors(self):
        """ Tests kneighbors against scikit-learn """

        X, Y = make_classification(n_samples=200, n_features=5)
        x, y = ds.array(X, (50, 5)), ds.array(Y, (50, 1))

        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(x, y)
        ds_y_hat = knn.predict(x)
        knn.score(x, y)

        sknn = skKNeighborsClassifier(n_neighbors=3)
        sknn.fit(X, Y)
        sk_y_hat = sknn.predict(X)

        self.assertTrue(np.all(ds_y_hat.collect() == sk_y_hat))

    def test_kneighbors_sparse(self):
        """ Tests kneighbors against scikit-learn with sparse data """
        X, Y = make_classification(n_samples=200, n_features=5)
        X, Y = csr_matrix(X), Y
        x, y = ds.array(X, (50, 5)), ds.array(Y, (50, 1))

        knn = KNeighborsClassifier(n_neighbors=3, weights='')
        knn.fit(x, y)
        ds_y_hat = knn.predict(x)

        sknn = skKNeighborsClassifier(n_neighbors=3, weights='distance')
        sknn.fit(X, Y)
        sk_y_hat = sknn.predict(X)

        self.assertTrue(np.all(ds_y_hat.collect() == sk_y_hat))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
