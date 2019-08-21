import unittest

import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler as SKScaler

import dislib as ds
from dislib.preprocessing import StandardScaler


class StandardScalerTest(unittest.TestCase):
    def test_fit_transform(self):
        """ Tests fit_transform against scikit-learn.
        """
        n_samples = 1500
        x, y = make_blobs(n_samples=n_samples, random_state=170)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        x = np.dot(x, transformation)
        ds_arr = ds.array(x, blocks_shape=(300, 2))

        sc1 = SKScaler()
        scaled_x = sc1.fit_transform(x)
        sc2 = StandardScaler()
        ds_scaled = sc2.fit_transform(ds_arr).collect()

        self.assertTrue(np.allclose(scaled_x, ds_scaled))
        self.assertTrue(np.allclose(sc1.mean_, sc2.mean_.collect()))
        self.assertTrue(np.allclose(sc1.var_, sc2.var_.collect()))

    def test_sparse(self):
        """ Tests fit_transforms with sparse data"""
        n_samples = 1500
        x, y = make_blobs(n_samples=n_samples, random_state=170)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        x = np.dot(x, transformation)

        dense_arr = ds.array(x, blocks_shape=(300, 2))
        sparse_arr = ds.array(csr_matrix(x), blocks_shape=(300, 2))

        sc = StandardScaler()
        dense_scaled = sc.fit_transform(dense_arr)
        dense_mean = sc.mean_.collect()
        dense_var = sc.var_.collect()

        sparse_scaled = sc.fit_transform(sparse_arr)
        sparse_mean = sc.mean_.collect()
        sparse_var = sc.var_.collect()

        csr_scaled = sparse_scaled.collect()
        arr_scaled = dense_scaled.collect()

        self.assertTrue(issparse(csr_scaled))
        self.assertTrue(sparse_scaled._sparse)
        self.assertFalse(sc.var_._sparse)
        self.assertFalse(sc.mean_._sparse)
        self.assertFalse(issparse(sparse_mean))
        self.assertFalse(issparse(sparse_var))

        self.assertTrue(np.allclose(csr_scaled.toarray(), arr_scaled))
        self.assertTrue(np.allclose(sparse_mean, dense_mean))
        self.assertTrue(np.allclose(sparse_var, dense_var))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
