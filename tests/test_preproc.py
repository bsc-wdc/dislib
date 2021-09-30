import unittest

import numpy as np
from numpy.testing._private.parameterized import parameterized
from pycompss.api.api import compss_wait_on
from scipy.sparse import csr_matrix, issparse
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler as SkStandardScaler
from sklearn.preprocessing import MinMaxScaler as SkMinMaxScaler

import dislib as ds
from dislib.preprocessing import StandardScaler, MinMaxScaler


class MinMaxScalerTest(unittest.TestCase):
    @parameterized.expand([((0, 1),),
                           ((-1, 1),)])
    def test_fit_transform(self, feature_range):
        """ Tests fit_transform against scikit-learn.
        """
        n_samples = 1500
        x, y = make_blobs(n_samples=n_samples, random_state=170)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        x = np.dot(x, transformation)
        ds_arr = ds.array(x, block_size=(300, 2))

        sc1 = SkMinMaxScaler(feature_range=feature_range)
        scaled_x = sc1.fit_transform(x)
        sc2 = MinMaxScaler(feature_range=feature_range)
        ds_scaled = sc2.fit_transform(ds_arr)

        self.assertTrue(np.allclose(scaled_x, ds_scaled.collect()))
        self.assertTrue(np.allclose(sc1.data_min_, sc2.data_min_.collect()))
        self.assertTrue(np.allclose(sc1.data_max_, sc2.data_max_.collect()))
        self.assertEqual(ds_scaled._top_left_shape,
                         ds_scaled._blocks[0][0].shape)
        self.assertEqual(ds_arr._reg_shape, ds_scaled._reg_shape)
        self.assertEqual(ds_arr._top_left_shape, ds_scaled._top_left_shape)
        self.assertEqual(ds_arr.shape, ds_scaled.shape)
        self.assertEqual(ds_arr._n_blocks, ds_scaled._n_blocks)

    @parameterized.expand([((0, 1),),
                           ((-1, 1),)])
    def test_sparse(self, feature_range):
        """ Tests fit_transforms with sparse data"""
        n_samples = 1500
        x, y = make_blobs(n_samples=n_samples, random_state=170)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        x = np.dot(x, transformation)

        dense_arr = ds.array(x, block_size=(300, 2))
        sparse_arr = ds.array(csr_matrix(x), block_size=(300, 2))

        sc = MinMaxScaler(feature_range=feature_range)
        dense_scaled = sc.fit_transform(dense_arr)
        dense_min = sc.data_min_.collect()
        dense_max = sc.data_max_.collect()

        sparse_scaled = sc.fit_transform(sparse_arr)
        sparse_min = sc.data_min_.collect()
        sparse_max = sc.data_max_.collect()

        csr_scaled = sparse_scaled.collect()
        arr_scaled = dense_scaled.collect()

        self.assertTrue(issparse(csr_scaled))
        self.assertTrue(sparse_scaled._sparse)
        self.assertTrue(sc.data_min_._sparse)
        self.assertTrue(sc.data_max_._sparse)
        self.assertTrue(issparse(sparse_min))
        self.assertTrue(issparse(sparse_max))

        self.assertTrue(np.allclose(csr_scaled.toarray(), arr_scaled))
        self.assertTrue(np.allclose(sparse_min.toarray(), dense_min))
        self.assertTrue(np.allclose(sparse_max.toarray(), dense_max))

    @parameterized.expand([((0, 1),),
                           ((-1, 1),)])
    def test_irregular(self, feature_range):
        """ Test with an irregular array """
        n_samples = 1500
        x, y = make_blobs(n_samples=n_samples, random_state=170)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        x = np.dot(x, transformation)
        ds_arr = ds.array(x, block_size=(300, 2))
        ds_arr = ds_arr[297:602]
        x = x[297:602]

        sc1 = SkMinMaxScaler(feature_range=feature_range)
        scaled_x = sc1.fit_transform(x)
        sc2 = MinMaxScaler(feature_range=feature_range)
        ds_scaled = sc2.fit_transform(ds_arr)

        self.assertTrue(np.allclose(scaled_x, ds_scaled.collect()))
        self.assertTrue(np.allclose(sc1.data_min_, sc2.data_min_.collect()))
        self.assertTrue(np.allclose(sc1.data_max_, sc2.data_max_.collect()))
        self.assertEqual(ds_scaled._top_left_shape,
                         compss_wait_on(ds_scaled._blocks[0][0]).shape)
        self.assertEqual(ds_arr._reg_shape, ds_scaled._reg_shape)
        self.assertEqual(ds_arr._top_left_shape, ds_scaled._top_left_shape)
        self.assertEqual(ds_arr.shape, ds_scaled.shape)
        self.assertEqual(ds_arr._n_blocks, ds_scaled._n_blocks)


class StandardScalerTest(unittest.TestCase):
    def test_fit_transform(self):
        """ Tests fit_transform against scikit-learn.
        """
        n_samples = 1500
        x, y = make_blobs(n_samples=n_samples, random_state=170)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        x = np.dot(x, transformation)
        ds_arr = ds.array(x, block_size=(300, 2))

        sc1 = SkStandardScaler()
        scaled_x = sc1.fit_transform(x)
        sc2 = StandardScaler()
        ds_scaled = sc2.fit_transform(ds_arr)

        self.assertTrue(np.allclose(scaled_x, ds_scaled.collect()))
        self.assertTrue(np.allclose(sc1.mean_, sc2.mean_.collect()))
        self.assertTrue(np.allclose(sc1.var_, sc2.var_.collect()))
        self.assertEqual(ds_scaled._top_left_shape,
                         ds_scaled._blocks[0][0].shape)
        self.assertEqual(ds_arr._reg_shape, ds_scaled._reg_shape)
        self.assertEqual(ds_arr._top_left_shape, ds_scaled._top_left_shape)
        self.assertEqual(ds_arr.shape, ds_scaled.shape)
        self.assertEqual(ds_arr._n_blocks, ds_scaled._n_blocks)

    def test_sparse(self):
        """ Tests fit_transforms with sparse data"""
        n_samples = 1500
        x, y = make_blobs(n_samples=n_samples, random_state=170)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        x = np.dot(x, transformation)

        dense_arr = ds.array(x, block_size=(300, 2))
        sparse_arr = ds.array(csr_matrix(x), block_size=(300, 2))

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
        self.assertTrue(sc.var_._sparse)
        self.assertTrue(sc.mean_._sparse)
        self.assertTrue(issparse(sparse_mean))
        self.assertTrue(issparse(sparse_var))

        self.assertTrue(np.allclose(csr_scaled.toarray(), arr_scaled))
        self.assertTrue(np.allclose(sparse_mean.toarray(), dense_mean))
        self.assertTrue(np.allclose(sparse_var.toarray(), dense_var))

    def test_irregular(self):
        """ Test with an irregular array """
        n_samples = 1500
        x, y = make_blobs(n_samples=n_samples, random_state=170)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        x = np.dot(x, transformation)
        ds_arr = ds.array(x, block_size=(300, 2))
        ds_arr = ds_arr[297:602]
        x = x[297:602]

        sc1 = SkStandardScaler()
        scaled_x = sc1.fit_transform(x)
        sc2 = StandardScaler()
        ds_scaled = sc2.fit_transform(ds_arr)

        self.assertTrue(np.allclose(scaled_x, ds_scaled.collect()))
        self.assertTrue(np.allclose(sc1.mean_, sc2.mean_.collect()))
        self.assertTrue(np.allclose(sc1.var_, sc2.var_.collect()))
        self.assertEqual(ds_scaled._top_left_shape,
                         compss_wait_on(ds_scaled._blocks[0][0]).shape)
        self.assertEqual(ds_arr._reg_shape, ds_scaled._reg_shape)
        self.assertEqual(ds_arr._top_left_shape, ds_scaled._top_left_shape)
        self.assertEqual(ds_arr.shape, ds_scaled.shape)
        self.assertEqual(ds_arr._n_blocks, ds_scaled._n_blocks)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
