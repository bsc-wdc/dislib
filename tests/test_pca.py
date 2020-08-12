import unittest

import numpy as np
from parameterized import parameterized
from sklearn.datasets import make_blobs

import dislib as ds
from dislib.decomposition import PCA


class PCATest(unittest.TestCase):

    @parameterized.expand([("eig"),
                           ("svd")])
    def test_fit(self, method):
        """Tests PCA.fit()"""
        np.random.seed(8)
        n_samples = 150
        n_features = 5
        n_blobs = 3

        # Create normal clusters along a diagonal line
        data = []
        cov = np.eye(n_features)
        size = n_samples // n_blobs
        for i in range(n_blobs):
            mu = [i] * n_features
            data.append(np.random.multivariate_normal(mu, cov, size))
        data = np.vstack(data)
        bn, bm = 25, 2
        dataset = ds.array(x=data, block_size=(bn, bm))

        pca = PCA(method=method)
        pca.fit(dataset)

        expected_eigvec = np.array([
            [0.46266375, 0.40183351, 0.50259945, 0.41394469, 0.44778976],
            [0.6548706, -0.63369901, 0.27008578, -0.15049713, -0.27198225],
            [-0.32039683, 0.24776797, 0.72090966, -0.18219517, -0.53202545],
            [-0.27511718, -0.36710543, 0.04871129, 0.85622265, -0.23249543],
            [-0.42278027, -0.4907138, 0.39033823, -0.19921873, 0.62322129]
        ])
        expected_eigval = np.array(
            [4.97145301, 1.3622795, 1.24328201, 0.96016518, 0.81758398]
        )

        comp = pca.components_.collect()
        var = pca.explained_variance_.collect()

        # use abs because signs do not really matter
        self.assertTrue(np.allclose(abs(comp), abs(expected_eigvec)))
        self.assertTrue(np.allclose(abs(var), abs(expected_eigval)))

    @parameterized.expand([("eig"),
                           ("svd")])
    def test_fit_transform(self, method):
        """Tests PCA.fit_transform()"""
        x, _ = make_blobs(n_samples=10, n_features=4, random_state=0)
        bn, bm = 10, 2
        dataset = ds.array(x=x, block_size=(bn, bm))

        pca = PCA(n_components=3, method=method)
        transformed = pca.fit_transform(dataset)
        self.assertEqual(transformed.shape, (10, 3))
        transformed = transformed.collect()

        expected = np.array([
            [-6.35473531, -2.7164493, -1.56658989],
            [7.929884, -1.58730182, -0.34880254],
            [-6.38778631, -2.42507746, -1.14037578],
            [-3.05289416, 5.17150174, 1.7108992],
            [-0.04603327, 3.83555442, -0.62579556],
            [7.40582319, -3.03963075, 0.32414659],
            [-6.46857295, -4.08706644, 2.32695512],
            [-1.10626548, 3.28309797, -0.56305687],
            [0.72446701, 2.41434103, -0.54476492],
            [7.35611329, -0.84896939, 0.42738466]
        ])
        self.assertEqual(transformed.shape, (10, 3))

        for i in range(transformed.shape[1]):
            features_equal = np.allclose(transformed[:, i], expected[:, i])
            features_opposite = np.allclose(transformed[:, i], -expected[:, i])
            self.assertTrue(features_equal or features_opposite)

    def test_sparse(self):
        """ Tests PCA produces the same results using dense and sparse
        data structures. """
        file_ = "tests/files/libsvm/2"
        x_sp, _ = ds.load_svmlight_file(file_, (10, 300), 780, True)
        x_ds, _ = ds.load_svmlight_file(file_, (10, 300), 780, False)

        pca = PCA()
        transform_dense = pca.fit_transform(x_ds).collect()
        dense_variance = pca.explained_variance_.collect()

        pca = PCA()
        transform_sparse = pca.fit_transform(x_sp).collect()
        sparse_variance = pca.explained_variance_.collect()

        self.assertTrue(np.allclose(transform_sparse, transform_dense))
        self.assertTrue(np.allclose(sparse_variance, dense_variance))

        # Test error for sparse data and method=svd
        with self.assertRaises(NotImplementedError):
            pca = PCA(method='svd')
            pca.fit(x_sp)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
