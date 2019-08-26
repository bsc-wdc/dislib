import unittest

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors as SKNearestNeighbors

import dislib as ds
from dislib.neighbors import NearestNeighbors


class NearestNeighborsTest(unittest.TestCase):
    def test_kneighbors(self):
        """ Tests kneighbors against scikit-learn """
        x = np.random.random((1500, 5))
        data = ds.array(x, block_size=(500, 3))
        q_data = ds.array(x, block_size=(101, 2))

        knn = NearestNeighbors(n_neighbors=10)
        knn.fit(data)
        dist, ind = knn.kneighbors(q_data)

        sknn = SKNearestNeighbors(n_neighbors=10)
        sknn.fit(X=x)
        skdist, skind = sknn.kneighbors(X=x)

        self.assertTrue(np.allclose(dist.collect(), skdist, atol=1e-7))
        self.assertTrue(np.array_equal(ind.collect(), skind))

    def test_kneighbors_sparse(self):
        """ Tests kneighbors against scikit-learn with sparse data """
        x = csr_matrix(np.random.random((1500, 5)))
        data = ds.array(x, block_size=(500, 5))

        knn = NearestNeighbors(n_neighbors=10)
        knn.fit(data)
        dist, ind = knn.kneighbors(data)

        sknn = SKNearestNeighbors(n_neighbors=10)
        sknn.fit(X=x)
        skdist, skind = sknn.kneighbors(X=x)

        self.assertTrue(np.allclose(dist.collect(), skdist, atol=1e-7))
        self.assertTrue(np.array_equal(ind.collect(), skind))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
