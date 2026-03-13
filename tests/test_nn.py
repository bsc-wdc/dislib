import unittest
import importlib

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors as SKNearestNeighbors

import dislib as ds
from dislib.neighbors import NearestNeighbors


cupy_available = importlib.util.find_spec("cupy") is not None


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

    @unittest.skipIf(not cupy_available, "cupy not installed")
    def test_kneighbors_gpu_sparse_dense(self):
        """ Tests GPU kneighbors with sparse and dense input handling """

        from dislib.neighbors.base import _get_kneighbors_gpu

        # Dense input
        x = np.random.rand(20, 5)
        q = np.random.rand(10, 5)
        x_blocks = [[x]]
        q_blocks = [[q]]
        dist_dense, ind_dense = _get_kneighbors_gpu(x_blocks, q_blocks, 3, 0)
        self.assertEqual(dist_dense.shape, (10, 3))
        self.assertEqual(ind_dense.shape, (10, 3))

        # Sparse input
        x_sp = csr_matrix(x)
        q_sp = csr_matrix(q)
        x_blocks_sp = [[x_sp]]
        q_blocks_sp = [[q_sp]]
        dist_sparse, ind_sparse = _get_kneighbors_gpu(
            x_blocks_sp, q_blocks_sp, 3, 0)
        self.assertEqual(dist_sparse.shape, (10, 3))
        self.assertEqual(ind_sparse.shape, (10, 3))

        # Mixed input
        dist_mixed, ind_mixed = _get_kneighbors_gpu(
            x_blocks_sp, q_blocks, 3, 0)
        self.assertEqual(dist_mixed.shape, (10, 3))
        self.assertEqual(ind_mixed.shape, (10, 3))
        dist_mixed2, ind_mixed2 = _get_kneighbors_gpu(
            x_blocks, q_blocks_sp, 3, 0)
        self.assertEqual(dist_mixed2.shape, (10, 3))
        self.assertEqual(ind_mixed2.shape, (10, 3))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
