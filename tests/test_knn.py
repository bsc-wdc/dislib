import unittest

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors as SKNearestNeighbors

from dislib.data import load_data
from dislib.neighbors import NearestNeighbors


class NearestNeighborsTest(unittest.TestCase):
    def test_kneighbors(self):
        """ Tests kneighbors against scikit-learn """
        x = np.random.random((1500, 5))
        dataset = load_data(x, subset_size=500)

        knn = NearestNeighbors(n_neighbors=10)
        knn.fit(dataset)
        dist, ind = knn.kneighbors(dataset)

        sknn = SKNearestNeighbors(n_neighbors=10)
        sknn.fit(X=dataset.samples)
        skdist, skind = sknn.kneighbors(X=dataset.samples)

        self.assertTrue(np.array_equal(dist, skdist))
        self.assertTrue(np.array_equal(ind, skind))

    def test_kneighbors_sparse(self):
        """ Tests kneighbors against scikit-learn with sparse data """
        x = csr_matrix(np.random.random((1500, 5)))
        dataset = load_data(x, subset_size=500)

        knn = NearestNeighbors(n_neighbors=10)
        knn.fit(dataset)
        dist, ind = knn.kneighbors(dataset)

        sknn = SKNearestNeighbors(n_neighbors=10)
        sknn.fit(X=dataset.samples)
        skdist, skind = sknn.kneighbors(X=dataset.samples)

        self.assertTrue(np.array_equal(dist, skdist))
        self.assertTrue(np.array_equal(ind, skind))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
