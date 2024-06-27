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
import dislib.data.util.model as utilmodel
from tests import BaseTimedTestCase


class ScalerTest(BaseTimedTestCase):
    def setUp(self) -> None:
        self.x, _ = make_blobs(n_samples=1500, n_features=8,
                               random_state=170, cluster_std=33)
        block_size = (300, 2)
        self.ds_arr = ds.array(self.x, block_size=block_size)
        self.sparse_arr = ds.array(csr_matrix(self.x), block_size=block_size)
        return super().setUp()


class MinMaxScalerTest(ScalerTest):

    @parameterized.expand([((0, 1),),
                           ((-1, 1),)])
    def test_fit_transform(self, feature_range):
        """ Tests fit_transform against scikit-learn.
        """

        sc1 = SkMinMaxScaler(feature_range=feature_range)
        scaled_x = sc1.fit_transform(self.x)
        sc2 = MinMaxScaler(feature_range=feature_range)
        ds_scaled = sc2.fit_transform(self.ds_arr)

        self.assertTrue(np.allclose(scaled_x, ds_scaled.collect()))
        self.assertTrue(np.allclose(sc1.data_min_, sc2.data_min_.collect()))
        self.assertTrue(np.allclose(sc1.data_max_, sc2.data_max_.collect()))
        self.assertEqual(ds_scaled._top_left_shape,
                         ds_scaled._blocks[0][0].shape)
        self.assertEqual(self.ds_arr._reg_shape, ds_scaled._reg_shape)
        self.assertEqual(self.ds_arr._top_left_shape,
                         ds_scaled._top_left_shape)
        self.assertEqual(self.ds_arr.shape, ds_scaled.shape)
        self.assertEqual(self.ds_arr._n_blocks, ds_scaled._n_blocks)

    @parameterized.expand([((0, 1),),
                           ((-1, 1),)])
    def test_inverse_transform(self, feature_range):
        """ Tests inverse_transform against scikit-learn.
        """
        n_samples = 500
        x, y = make_blobs(n_samples=n_samples, random_state=170)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        x = np.dot(x, transformation)
        ds_arr = ds.array(x, block_size=(300, 2))

        sc1 = SkMinMaxScaler(feature_range=feature_range)
        scaled_x = sc1.fit_transform(x)
        scaled_x = sc1.inverse_transform(scaled_x)
        sc2 = MinMaxScaler(feature_range=feature_range)
        ds_scaled = sc2.fit_transform(ds_arr)
        ds_scaled = sc2.inverse_transform(ds_scaled)

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

        sc = MinMaxScaler(feature_range=feature_range)
        dense_scaled = sc.fit_transform(self.ds_arr)
        dense_min = sc.data_min_.collect()
        dense_max = sc.data_max_.collect()

        sparse_scaled = sc.fit_transform(self.sparse_arr)
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
    def test_sparse_inverse_transform(self, feature_range):
        """ Tests inverse_transform with sparse data"""
        n_samples = 500
        x, y = make_blobs(n_samples=n_samples, random_state=170)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        x = np.dot(x, transformation)

        dense_arr = ds.array(x, block_size=(300, 2))
        sparse_arr = ds.array(csr_matrix(x), block_size=(300, 2))

        sc = MinMaxScaler(feature_range=feature_range)
        dense_scaled = sc.fit_transform(dense_arr)
        dense_scaled = sc.inverse_transform(dense_scaled)
        dense_min = sc.data_min_.collect()
        dense_max = sc.data_max_.collect()

        sparse_scaled = sc.fit_transform(sparse_arr)
        sparse_scaled = sc.inverse_transform(sparse_scaled)
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

        ds_arr = self.ds_arr[297:602]
        x = self.x[297:602]

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

    def test_save_load(self):
        """
        Tests that the save and load methods work properly with the three
        expected formats and that an exception is raised when a non-supported
        format is provided.
        """
        x = ds.array(np.array([[0.2, 3.4], [5.7, 2.3], [-1.2, 9.8],
                               [15.7, -12.3]]), block_size=(2, 2))

        scaler = MinMaxScaler()
        scaler.fit(x)
        scaler.save_model("./saved_model")

        scaler_2 = MinMaxScaler()
        scaler_2.load_model("./saved_model")
        self.assertTrue(np.all(scaler.transform(x).collect() ==
                               scaler_2.transform(x).collect()))

        scaler.save_model("./saved_model", save_format="cbor")

        scaler_2 = MinMaxScaler()
        scaler_2.load_model("./saved_model", load_format="cbor")

        self.assertTrue(np.all(scaler.transform(x).collect() ==
                               scaler_2.transform(x).collect()))

        scaler.save_model("./saved_model", save_format="pickle")

        scaler_2 = MinMaxScaler()
        scaler_2.load_model("./saved_model", load_format="pickle")
        self.assertTrue(np.all(scaler.transform(x).collect() ==
                               scaler_2.transform(x).collect()))

        with self.assertRaises(ValueError):
            scaler.save_model("./saved_model", save_format="txt")

        with self.assertRaises(ValueError):
            scaler_2 = MinMaxScaler()
            scaler_2.load_model("./saved_model", load_format="txt")

        scaler = MinMaxScaler(feature_range=(0, 2))
        scaler.fit(x)
        scaler.save_model("./saved_model", overwrite=False)

        scaler_2 = MinMaxScaler(feature_range=(0, 2))
        scaler_2.load_model("./saved_model", load_format="pickle")
        self.assertFalse(np.all(scaler.transform(x).collect() ==
                                scaler_2.transform(x).collect()))

        scaler.save_model("./saved_model", overwrite=True)

        scaler_2 = MinMaxScaler(feature_range=(0, 2))
        scaler_2.load_model("./saved_model")
        self.assertTrue(np.all(scaler.transform(x).collect() ==
                               scaler_2.transform(x).collect()))

        cbor2_module = utilmodel.cbor2
        utilmodel.cbor2 = None
        with self.assertRaises(ModuleNotFoundError):
            scaler.save_model("./saved_model_error", save_format="cbor")
        with self.assertRaises(ModuleNotFoundError):
            scaler_2.load_model("./saved_model_error", load_format="cbor")
        utilmodel.cbor2 = cbor2_module


class StandardScalerTest(ScalerTest):
    def test_fit_transform(self):
        """ Tests fit_transform against scikit-learn.
        """

        sc1 = SkStandardScaler()
        scaled_x = sc1.fit_transform(self.x)
        sc2 = StandardScaler()
        ds_scaled = sc2.fit_transform(self.ds_arr)

        self.assertTrue(np.allclose(scaled_x, ds_scaled.collect()))
        self.assertTrue(np.allclose(sc1.mean_, sc2.mean_.collect()))
        self.assertTrue(np.allclose(sc1.var_, sc2.var_.collect()))
        self.assertEqual(ds_scaled._top_left_shape,
                         ds_scaled._blocks[0][0].shape)
        self.assertEqual(self.ds_arr._reg_shape,
                         ds_scaled._reg_shape)
        self.assertEqual(self.ds_arr._top_left_shape,
                         ds_scaled._top_left_shape)
        self.assertEqual(self.ds_arr.shape, ds_scaled.shape)
        self.assertEqual(self.ds_arr._n_blocks, ds_scaled._n_blocks)

    def test_inverse_transform(self):
        """ Tests inverse_transform against scikit-learn.
        """
        n_samples = 500
        x, y = make_blobs(n_samples=n_samples, random_state=170)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        x = np.dot(x, transformation)
        ds_arr = ds.array(x, block_size=(300, 2))

        sc1 = SkStandardScaler()
        scaled_x = sc1.fit_transform(x)
        scaled_x = sc1.inverse_transform(scaled_x)
        sc2 = StandardScaler()
        ds_scaled = sc2.fit_transform(ds_arr)
        ds_scaled = sc2.inverse_transform(ds_scaled)

        self.assertTrue(np.allclose(scaled_x, ds_scaled.collect()))
        self.assertTrue(np.allclose(sc1.mean_, sc2.mean_.collect()))
        self.assertTrue(np.allclose(sc1.var_, sc2.var_.collect()))
        self.assertEqual(ds_scaled._top_left_shape,
                         ds_scaled._blocks[0][0].shape)
        self.assertEqual(ds_arr._reg_shape, ds_scaled._reg_shape)
        self.assertEqual(ds_arr._top_left_shape,
                         ds_scaled._top_left_shape)
        self.assertEqual(ds_arr.shape, ds_scaled.shape)
        self.assertEqual(ds_arr._n_blocks, ds_scaled._n_blocks)

    def test_sparse(self):
        """ Tests fit_transforms with sparse data"""

        dense_arr = self.ds_arr

        sc = StandardScaler()
        dense_scaled = sc.fit_transform(dense_arr)
        dense_mean = sc.mean_.collect()
        dense_var = sc.var_.collect()

        sparse_scaled = sc.fit_transform(self.sparse_arr)
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

    def test_sparse_inverse_transform(self):
        """ Tests inverse_transform with sparse data"""
        n_samples = 500
        x, y = make_blobs(n_samples=n_samples, random_state=170)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        x = np.dot(x, transformation)

        dense_arr = ds.array(x, block_size=(300, 2))
        sparse_arr = ds.array(csr_matrix(x), block_size=(300, 2))

        sc = StandardScaler()
        dense_scaled = sc.fit_transform(dense_arr)
        dense_scaled = sc.inverse_transform(dense_scaled)
        dense_mean = sc.mean_.collect()
        dense_var = sc.var_.collect()

        sparse_scaled = sc.fit_transform(sparse_arr)
        sparse_scaled = sc.inverse_transform(sparse_scaled)
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

        ds_arr = self.ds_arr[297:602]
        x = self.x[297:602]

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

    def test_save_load(self):
        """
        Tests that the save and load methods work properly with the three
        expected formats and that an exception is raised when a non-supported
        format is provided.
        """
        x = ds.array(np.array([[0.2, 3.4], [5.7, 2.3], [-1.2, 9.8],
                               [15.7, -12.3]]), block_size=(2, 2))

        scaler = StandardScaler()
        scaler.fit(x)
        scaler.save_model("./saved_model")

        scaler_2 = StandardScaler()
        scaler_2.load_model("./saved_model")
        self.assertTrue(np.all(scaler.transform(x).collect() ==
                               scaler_2.transform(x).collect()))

        scaler.save_model("./saved_model", save_format="cbor")

        scaler_2 = StandardScaler()
        scaler_2.load_model("./saved_model", load_format="cbor")

        self.assertTrue(np.all(scaler.transform(x).collect() ==
                               scaler_2.transform(x).collect()))

        scaler.save_model("./saved_model", save_format="pickle")

        scaler_2 = StandardScaler()
        scaler_2.load_model("./saved_model", load_format="pickle")
        self.assertTrue(np.all(scaler.transform(x).collect() ==
                               scaler_2.transform(x).collect()))

        scaler = StandardScaler()
        scaler.fit(x)
        scaler.save_model("./saved_model", overwrite=False)

        scaler_2 = StandardScaler()
        scaler_2.load_model("./saved_model", load_format="pickle")
        self.assertTrue(np.all(scaler.transform(x).collect() ==
                               scaler_2.transform(x).collect()))

        with self.assertRaises(ValueError):
            scaler.save_model("./saved_model", save_format="txt")

        with self.assertRaises(ValueError):
            scaler_2 = MinMaxScaler()
            scaler_2.load_model("./saved_model", load_format="txt")


def main():
    unittest.main()


if __name__ == '__main__':
    main()
