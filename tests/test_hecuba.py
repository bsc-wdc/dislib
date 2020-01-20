import gc
import unittest

import numpy as np
from hecuba import config
from pycompss.api.api import compss_wait_on
from sklearn.datasets import make_blobs

import dislib as ds
from dislib.cluster import KMeans
from dislib.decomposition import PCA
from dislib.neighbors import NearestNeighbors
from dislib.regression import LinearRegression


class HecubaTest(unittest.TestCase):

    def test_iterate_rows(self):
        """
        Tests iterating through the rows of the Hecuba array
        """
        config.session.execute("TRUNCATE TABLE hecuba.istorage")
        config.session.execute("DROP KEYSPACE IF EXISTS hecuba_dislib")
        block_size = (20, 10)
        x = np.array([[i] * 10 for i in range(100)])

        data = ds.array(x=x, block_size=block_size, backend="hecuba",
                        name="hecuba_dislib.test_array")

        for i, chunk in enumerate(data._iterator(axis="rows")):
            r_data = chunk.collect()
            r_x = np.array([[j] * 10
                            for j in range(i * block_size[0],
                                           i * block_size[0] + block_size[0])])
            self.assertTrue(np.array_equal(r_data, r_x))

        self.assertEqual(i + 1, len(data._blocks))

    def test_fit_predict(self):
        """ Tests fit_predict."""
        config.session.execute("TRUNCATE TABLE hecuba.istorage")
        config.session.execute("DROP KEYSPACE IF EXISTS hecuba_dislib")

        x, y = make_blobs(n_samples=1500, random_state=170)
        x_filtered = np.vstack(
            (x[y == 0][:500], x[y == 1][:100], x[y == 2][:10]))

        block_size = (x_filtered.shape[0] // 10, x_filtered.shape[1])

        x_train = ds.array(x_filtered, block_size=block_size)
        x_train_hecuba = ds.array(x=x_filtered, block_size=block_size,
                                  backend="hecuba",
                                  name="hecuba_dislib.test_array2")

        kmeans = KMeans(n_clusters=3, random_state=170, verbose=True)
        labels = kmeans.fit_predict(x_train).collect()

        kmeans2 = KMeans(n_clusters=3, random_state=170, verbose=True)
        h_labels = kmeans2.fit_predict(x_train_hecuba).collect()

        self.assertTrue(np.allclose(kmeans.centers, kmeans2.centers))
        self.assertTrue(np.allclose(labels, h_labels))

    def test_already_persistent(self):
        config.session.execute("TRUNCATE TABLE hecuba.istorage")
        config.session.execute("DROP KEYSPACE IF EXISTS hecuba_dislib")
        x, y = make_blobs(n_samples=1500, random_state=170)
        x_filtered = np.vstack(
            (x[y == 0][:500], x[y == 1][:100], x[y == 2][:10]))

        block_size = (x_filtered.shape[0] // 10, x_filtered.shape[1])

        x_train = ds.array(x_filtered, block_size=block_size)
        x_train_hecuba = ds.array(x=x_filtered, block_size=block_size,
                                  backend="hecuba",
                                  name="hecuba_dislib.test_array2")

        # ensure that all data is released from memory
        blocks = x_train_hecuba._blocks
        for block in blocks:
            del block
        del x_train_hecuba
        gc.collect()

        x_train_hecuba = ds.array(x=None, block_size=block_size,
                                  backend="hecuba",
                                  name="hecuba_dislib.test_array2")

        kmeans = KMeans(n_clusters=3, random_state=170)
        labels = kmeans.fit_predict(x_train).collect()

        kmeans2 = KMeans(n_clusters=3, random_state=170)
        h_labels = kmeans2.fit_predict(x_train_hecuba).collect()

        self.assertTrue(np.allclose(kmeans.centers, kmeans2.centers))
        self.assertTrue(np.allclose(labels, h_labels))

    def test_linear_fit_predict(self):
        config.session.execute("TRUNCATE TABLE hecuba.istorage")
        config.session.execute("DROP KEYSPACE IF EXISTS hecuba_dislib")

        x_data = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
        y_data = np.array([2, 1, 1, 2, 4.5]).reshape(-1, 1)

        block_size = (x_data.shape[0] // 3, x_data.shape[1])

        x = ds.array(x=x_data, block_size=block_size, backend="hecuba",
                     name="hecuba_dislib.test_array_x")
        y = ds.array(x=y_data, block_size=block_size, backend="hecuba",
                     name="hecuba_dislib.test_array_y")

        reg = LinearRegression()
        reg.fit(x, y)
        # y = 0.6 * x + 0.3

        reg.coef_ = compss_wait_on(reg.coef_)
        reg.intercept_ = compss_wait_on(reg.intercept_)
        self.assertTrue(np.allclose(reg.coef_, 0.6))
        self.assertTrue(np.allclose(reg.intercept_, 0.3))

        x_test = np.array([3, 5]).reshape(-1, 1)
        test_data = ds.array(x=x_test, block_size=block_size,
                             backend="hecuba",
                             name="hecuba_dislib.test_array_test")
        pred = reg.predict(test_data).collect()
        self.assertTrue(np.allclose(pred, [2.1, 3.3]))

    def test_knn_fit(self):
        config.session.execute("TRUNCATE TABLE hecuba.istorage")
        config.session.execute("DROP KEYSPACE IF EXISTS hecuba_dislib")

        x = np.random.random((1500, 5))
        block_size = (x.shape[0] // 10, 3)
        block_size2 = (x.shape[0] // 20, 2)

        data = ds.array(x, block_size=block_size)
        q_data = ds.array(x, block_size=block_size2)

        data_h = ds.array(x, block_size=block_size, backend="hecuba",
                          name="hecuba_dislib.test_array")
        q_data_h = ds.array(x, block_size=block_size2, backend="hecuba",
                            name="hecuba_dislib.test_array_q")

        knn = NearestNeighbors(n_neighbors=10)
        knn.fit(data)
        dist, ind = knn.kneighbors(q_data)

        knn_h = NearestNeighbors(n_neighbors=10)
        knn_h.fit(data_h)
        dist_h, ind_h = knn_h.kneighbors(q_data_h)

        self.assertTrue(np.allclose(dist.collect(), dist_h.collect(),
                                    atol=1e-7))
        self.assertTrue(np.array_equal(ind.collect(), ind_h.collect()))

    def test_pca_fit_transform(self):
        config.session.execute("TRUNCATE TABLE hecuba.istorage")
        config.session.execute("DROP KEYSPACE IF EXISTS hecuba_dislib")

        x, _ = make_blobs(n_samples=10, n_features=4, random_state=0)
        bn, bm = 25, 5
        dataset = ds.array(x=x, block_size=(bn, bm), backend="hecuba",
                           name="hecuba_dislib.test_array")

        pca = PCA(n_components=3)
        transformed = pca.fit_transform(dataset).collect()
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


def main():
    unittest.main()


if __name__ == '__main__':
    main()
