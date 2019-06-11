import unittest

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

from dislib.cluster import DBSCAN
from dislib.data import load_data
from dislib.utils import as_grid


class DBSCANTest(unittest.TestCase):
    def test_n_clusters_blobs(self):
        """ Tests that DBSCAN finds the correct number of clusters with blob
        data.
        """
        n_samples = 1500
        x, y = make_blobs(n_samples=n_samples, n_features=2, random_state=8)
        dbscan = DBSCAN(n_regions=1, eps=.3)
        x = StandardScaler().fit_transform(x)
        dataset = load_data(x=x, y=y, subset_size=300)
        dbscan.fit(dataset)
        self.assertEqual(dbscan.n_clusters, 3)

    def test_n_clusters_circles(self):
        """ Tests that DBSCAN finds the correct number of clusters with
        circle data.
        """
        n_samples = 1500
        x, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)
        dbscan = DBSCAN(n_regions=1, eps=.15)
        x = StandardScaler().fit_transform(x)
        dataset = load_data(x=x, y=y, subset_size=300)
        dbscan.fit(dataset)
        self.assertEqual(dbscan.n_clusters, 2)

    def test_n_clusters_moons(self):
        """ Tests that DBSCAN finds the correct number of clusters with
        moon data.
        """
        n_samples = 1500
        x, y = make_moons(n_samples=n_samples, noise=.05)
        dbscan = DBSCAN(n_regions=1, eps=.3)
        x = StandardScaler().fit_transform(x)
        dataset = load_data(x=x, y=y, subset_size=300)
        dbscan.fit(dataset)
        self.assertEqual(dbscan.n_clusters, 2)

    def test_n_clusters_aniso(self):
        """ Tests that DBSCAN finds the correct number of clusters with
        anisotropicly distributed data.
        """
        n_samples = 1500
        x, y = make_blobs(n_samples=n_samples, random_state=170)
        dbscan = DBSCAN(n_regions=1, eps=.15)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        x = np.dot(x, transformation)
        x = StandardScaler().fit_transform(x)
        dataset = load_data(x=x, y=y, subset_size=300)
        dbscan.fit(dataset)
        y_pred = dataset.labels
        true_sizes = {19, 496, 491, 488, 6}
        cluster_sizes = {y_pred[y_pred == -1].size,
                         y_pred[y_pred == 0].size,
                         y_pred[y_pred == 1].size,
                         y_pred[y_pred == 2].size,
                         y_pred[y_pred == 3].size}

        self.assertEqual(dbscan.n_clusters, 4)
        self.assertEqual(true_sizes, cluster_sizes)

    def test_n_clusters_blobs_max_samples(self):
        """ Tests that DBSCAN finds the correct number of clusters when
        defining max_samples with blob data.
        """
        n_samples = 1500
        x, y = make_blobs(n_samples=n_samples, n_features=2, random_state=8)
        dbscan = DBSCAN(n_regions=1, eps=.3, max_samples=500)
        x = StandardScaler().fit_transform(x)
        dataset = load_data(x=x, y=y, subset_size=300)
        dbscan.fit(dataset)
        self.assertEqual(dbscan.n_clusters, 3)

    def test_n_clusters_circles_max_samples(self):
        """ Tests that DBSCAN finds the correct number of clusters when
        defining max_samples with circle data.
        """
        n_samples = 1500
        x, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)
        dbscan = DBSCAN(n_regions=1, eps=.15, max_samples=500)
        x = StandardScaler().fit_transform(x)
        dataset = load_data(x=x, y=y, subset_size=300)
        dbscan.fit(dataset)
        self.assertEqual(dbscan.n_clusters, 2)

    def test_n_clusters_moons_max_samples(self):
        """ Tests that DBSCAN finds the correct number of clusters when
        defining max_samples with moon data.
        """
        n_samples = 1500
        x, y = make_moons(n_samples=n_samples, noise=.05)
        dbscan = DBSCAN(n_regions=1, eps=.3, max_samples=500)
        x = StandardScaler().fit_transform(x)
        dataset = load_data(x=x, y=y, subset_size=300)
        dbscan.fit(dataset)
        self.assertEqual(dbscan.n_clusters, 2)

    def test_n_clusters_aniso_max_samples(self):
        """ Tests that DBSCAN finds the correct number of clusters when
        defining max_samples with anisotropicly distributed data.
        """
        n_samples = 1500
        x, y = make_blobs(n_samples=n_samples, random_state=170)
        dbscan = DBSCAN(n_regions=1, eps=.15, max_samples=500)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        x = np.dot(x, transformation)
        x = StandardScaler().fit_transform(x)
        dataset = load_data(x=x, y=y, subset_size=300)
        dbscan.fit(dataset)
        y_pred = dataset.labels
        true_sizes = {19, 496, 491, 488, 6}
        cluster_sizes = {y_pred[y_pred == -1].size,
                         y_pred[y_pred == 0].size,
                         y_pred[y_pred == 1].size,
                         y_pred[y_pred == 2].size,
                         y_pred[y_pred == 3].size}

        self.assertEqual(dbscan.n_clusters, 4)
        self.assertEqual(true_sizes, cluster_sizes)

    def test_n_clusters_blobs_grid(self):
        """ Tests that DBSCAN finds the correct number of clusters when
        setting n_regions > 1 with blob data.
        """
        n_samples = 1500
        x, y = make_blobs(n_samples=n_samples, n_features=2, random_state=8)
        dbscan = DBSCAN(n_regions=4, eps=.3, max_samples=300)
        x = StandardScaler().fit_transform(x)
        dataset = load_data(x=x, y=y, subset_size=300)
        dbscan.fit(dataset)
        self.assertEqual(dbscan.n_clusters, 3)

    def test_n_clusters_circles_grid(self):
        """ Tests that DBSCAN finds the correct number of clusters when
        setting n_regions > 1 with circle data.
        """
        n_samples = 1500
        x, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)
        dbscan = DBSCAN(n_regions=4, eps=.15, max_samples=700)
        x = StandardScaler().fit_transform(x)
        dataset = load_data(x=x, y=y, subset_size=300)
        dbscan.fit(dataset)
        self.assertEqual(dbscan.n_clusters, 2)

    def test_n_clusters_moons_grid(self):
        """ Tests that DBSCAN finds the correct number of clusters when
        setting n_regions > 1 with moon data.
        """
        n_samples = 1500
        x, y = make_moons(n_samples=n_samples, noise=.05)
        dbscan = DBSCAN(n_regions=4, eps=.3, max_samples=600)
        x = StandardScaler().fit_transform(x)
        dataset = load_data(x=x, y=y, subset_size=300)
        dbscan.fit(dataset)
        self.assertEqual(dbscan.n_clusters, 2)

    def test_n_clusters_aniso_grid(self):
        """ Tests that DBSCAN finds the correct number of clusters when
        setting n_regions > 1 with anisotropicly distributed data.
        """
        n_samples = 1500
        x, y = make_blobs(n_samples=n_samples, random_state=170)
        dbscan = DBSCAN(n_regions=4, eps=.15, max_samples=500)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        x = np.dot(x, transformation)
        x = StandardScaler().fit_transform(x)
        dataset = load_data(x=x, y=y, subset_size=300)
        dbscan.fit(dataset)
        y_pred = dataset.labels
        true_sizes = {19, 496, 491, 488, 6}
        cluster_sizes = {y_pred[y_pred == -1].size,
                         y_pred[y_pred == 0].size,
                         y_pred[y_pred == 1].size,
                         y_pred[y_pred == 2].size,
                         y_pred[y_pred == 3].size}

        self.assertEqual(dbscan.n_clusters, 4)
        self.assertEqual(true_sizes, cluster_sizes)

    def test_zero_samples(self):
        """ Tests DBSCAN fit when some regions contain zero samples.
        """
        n_samples = 2
        x, y = make_blobs(n_samples=n_samples, n_features=2, random_state=8)
        dbscan = DBSCAN(n_regions=3, eps=.2, arrange_data=True,
                        max_samples=100)
        x = StandardScaler().fit_transform(x)
        dataset = load_data(x=x, y=y, subset_size=300)
        dbscan.fit(dataset)
        self.assertEqual(dbscan.n_clusters, 0)

    def test_n_clusters_aniso_dimensions(self):
        """ Tests that DBSCAN finds the correct number of clusters when
        dimensions is not None.
        """
        n_samples = 1500
        x, y = make_blobs(n_samples=n_samples, random_state=170)
        dbscan = DBSCAN(n_regions=5, dimensions=[1], eps=.15)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        x = np.dot(x, transformation)
        x = StandardScaler().fit_transform(x)
        dataset = load_data(x=x, y=y, subset_size=300)
        dbscan.fit(dataset)
        y_pred = dataset.labels
        true_sizes = {19, 496, 491, 488, 6}
        cluster_sizes = {y_pred[y_pred == -1].size,
                         y_pred[y_pred == 0].size,
                         y_pred[y_pred == 1].size,
                         y_pred[y_pred == 2].size,
                         y_pred[y_pred == 3].size}

        self.assertEqual(dbscan.n_clusters, 4)
        self.assertEqual(true_sizes, cluster_sizes)

    def test_n_clusters_arranged(self):
        """ Tests DBSCAN clustering with previously arranged data. """
        n_samples = 1500
        x, y = make_blobs(n_samples=n_samples, random_state=170)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        x = np.dot(x, transformation)
        x = StandardScaler().fit_transform(x)
        dataset = load_data(x=x, y=y, subset_size=300)
        grid_data = as_grid(dataset, n_regions=4)
        dbscan = DBSCAN(arrange_data=False, eps=.15)
        dbscan.fit(grid_data)
        y_pred = grid_data.labels
        true_sizes = {19, 496, 491, 488, 6}
        cluster_sizes = {y_pred[y_pred == -1].size,
                         y_pred[y_pred == 0].size,
                         y_pred[y_pred == 1].size,
                         y_pred[y_pred == 2].size,
                         y_pred[y_pred == 3].size}

        self.assertEqual(dbscan.n_clusters, 4)
        self.assertEqual(true_sizes, cluster_sizes)

    def test_sparse(self):
        """ Tests that DBSCAN produces the same results with sparse and
        dense data.
        """
        n_samples = 1500
        x, y = make_blobs(n_samples=n_samples, random_state=170)
        dbscan = DBSCAN(n_regions=1, eps=.15)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        x = np.dot(x, transformation)
        x = StandardScaler().fit_transform(x)

        dense = load_data(x=x, y=y, subset_size=300)
        sparse = load_data(x=csr_matrix(x), y=y, subset_size=300)

        dbscan.fit(dense)
        dbscan.fit(sparse)

        self.assertTrue(np.array_equal(dense.labels, sparse.labels))

    def test_small_cluster_1(self):
        """ Tests that DBSCAN can find clusters with less than min_samples. """
        x = np.array([[0, 0], [0, 1], [1, 0], [3, 0], [5.1, 0], [6, 0], [6, 1],
                      [10, 10]])
        ds = load_data(x=x, subset_size=5)

        # n_regions=1
        dbscan1 = DBSCAN(n_regions=1, eps=2.5, min_samples=4)
        dbscan1.fit(ds)
        self.assertEqual(dbscan1.n_clusters, 2)

    def test_small_cluster_2(self):
        """ Tests that DBSCAN can find clusters with less than min_samples. """
        x = np.array([[0, 0], [0, 1], [1, 0], [3, 0], [5.1, 0], [6, 0], [6, 1],
                      [10, 10]])
        ds = load_data(x=x, subset_size=5)

        # n_regions=10
        dbscan2 = DBSCAN(n_regions=10, eps=2.5, min_samples=4)
        dbscan2.fit(ds)
        self.assertEqual(dbscan2.n_clusters, 2)

    def test_cluster_between_regions_1(self):
        """ Tests that DBSCAN can find clusters between regions. """
        x = np.array([[0, 0], [3.9, 0], [4.1, 0], [4.1, 0.89], [4.1, 0.88],
                      [5.9, 0], [5.9, 0.89], [5.9, 0.88], [6.1, 0], [10, 10],
                      [4.6, 0], [5.4, 0]])
        ds = load_data(x=x, subset_size=5)

        dbscan = DBSCAN(n_regions=10, eps=0.9, min_samples=4)
        dbscan.fit(ds)
        self.assertEqual(dbscan.n_clusters, 1)

    def test_cluster_between_regions_2(self):
        """ Tests that DBSCAN can find clusters between regions. """
        x = np.array([[0, 0], [0.6, 0], [0.9, 0], [1.1, 0.2], [0.9, 0.6],
                      [1.1, 0.8], [1.4, 0.8], [2, 2]])
        ds = load_data(x=x, subset_size=5)

        dbscan = DBSCAN(n_regions=2, eps=0.5, min_samples=3)
        dbscan.fit(ds)
        self.assertEqual(dbscan.n_clusters, 1)

    def test_cluster_between_regions_3(self):
        """ Tests that DBSCAN can find clusters between regions. """
        x = np.array([[0, 0], [0.6, 0], [0.6, 0.01], [0.9, 0], [1.1, 0.2],
                      [1.4, 0.2], [1.4, 0.21], [0.9, 0.6], [0.6, 0.6],
                      [0.6, 0.61], [1.1, 0.8], [1.4, 0.8], [1.4, 0.81],
                      [2, 2]])
        ds = load_data(x=x, subset_size=5)

        dbscan = DBSCAN(n_regions=2, eps=0.5, min_samples=3)
        dbscan.fit(ds)
        self.assertEqual(dbscan.n_clusters, 1)

    def test_random_clusters_1(self):
        """ Tests DBSCAN on random data with multiple clusters. """
        # 1 dimension
        np.random.seed(1)
        x = np.random.uniform(0, 10, size=(1000, 1))
        ds = load_data(x=x, subset_size=300)
        dbscan = DBSCAN(n_regions=100, eps=0.1, min_samples=20)
        dbscan.fit(ds)

        self.assertEqual(dbscan.n_clusters, 18)
        self.assertEqual(np.count_nonzero(ds.labels == -1), 72)

    def test_random_clusters_2(self):
        """ Tests DBSCAN on random data with multiple clusters. """
        # 2 dimensions
        np.random.seed(2)
        x = np.random.uniform(0, 10, size=(1000, 2))
        ds = load_data(x=x, subset_size=300)
        dbscan = DBSCAN(n_regions=10, max_samples=10, eps=0.5, min_samples=10)
        dbscan.fit(ds)

        self.assertEqual(dbscan.n_clusters, 27)
        self.assertEqual(np.count_nonzero(ds.labels == -1), 206)

    def test_random_clusters_3(self):
        """ Tests DBSCAN on random data with multiple clusters. """
        # 3 dimensions
        np.random.seed(3)
        x = np.random.uniform(0, 10, size=(1000, 3))
        ds = load_data(x=x, subset_size=300)
        dbscan = DBSCAN(n_regions=10, dimensions=[0, 1],
                        eps=0.9, min_samples=4)
        dbscan.fit(ds)

        self.assertEqual(dbscan.n_clusters, 50)
        self.assertEqual(np.count_nonzero(ds.labels == -1), 266)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
