import unittest

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

from dislib.cluster import DBSCAN
from dislib.data import load_data


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
        n_clusters = np.unique(dbscan.labels_)
        n_clusters = n_clusters[n_clusters >= 0].size
        self.assertEqual(n_clusters, 3)

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
        n_clusters = np.unique(dbscan.labels_)
        n_clusters = n_clusters[n_clusters >= 0].size
        self.assertEqual(n_clusters, 2)

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
        n_clusters = np.unique(dbscan.labels_)
        n_clusters = n_clusters[n_clusters >= 0].size
        self.assertEqual(n_clusters, 2)

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
        n_clusters = np.unique(dbscan.labels_)
        n_clusters = n_clusters[n_clusters >= 0].size
        self.assertEqual(n_clusters, 4)

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
        n_clusters = np.unique(dbscan.labels_)
        n_clusters = n_clusters[n_clusters >= 0].size
        self.assertEqual(n_clusters, 3)

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
        n_clusters = np.unique(dbscan.labels_)
        n_clusters = n_clusters[n_clusters >= 0].size
        self.assertEqual(n_clusters, 2)

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
        n_clusters = np.unique(dbscan.labels_)
        n_clusters = n_clusters[n_clusters >= 0].size
        self.assertEqual(n_clusters, 2)

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
        n_clusters = np.unique(dbscan.labels_)
        n_clusters = n_clusters[n_clusters >= 0].size
        self.assertEqual(n_clusters, 4)

    def test_n_clusters_blobs_grid(self):
        """ Tests that DBSCAN finds the correct number of clusters when
        setting n_regions > 1 with blob data.
        """
        n_samples = 1500
        x, y = make_blobs(n_samples=n_samples, n_features=2, random_state=8)
        dbscan = DBSCAN(n_regions=2, eps=.3, max_samples=300)
        x = StandardScaler().fit_transform(x)
        dataset = load_data(x=x, y=y, subset_size=300)
        dbscan.fit(dataset)
        n_clusters = np.unique(dbscan.labels_)
        n_clusters = n_clusters[n_clusters >= 0].size
        self.assertEqual(n_clusters, 3)

    def test_n_clusters_circles_grid(self):
        """ Tests that DBSCAN finds the correct number of clusters when
        setting n_regions > 1 with circle data.
        """
        n_samples = 1500
        x, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)
        dbscan = DBSCAN(n_regions=2, eps=.15, max_samples=700)
        x = StandardScaler().fit_transform(x)
        dataset = load_data(x=x, y=y, subset_size=300)
        dbscan.fit(dataset)
        n_clusters = np.unique(dbscan.labels_)
        n_clusters = n_clusters[n_clusters >= 0].size
        self.assertEqual(n_clusters, 2)

    def test_n_clusters_moons_grid(self):
        """ Tests that DBSCAN finds the correct number of clusters when
        setting n_regions > 1 with moon data.
        """
        n_samples = 1500
        x, y = make_moons(n_samples=n_samples, noise=.05)
        dbscan = DBSCAN(n_regions=2, eps=.3, max_samples=600)
        x = StandardScaler().fit_transform(x)
        dataset = load_data(x=x, y=y, subset_size=300)
        dbscan.fit(dataset)
        n_clusters = np.unique(dbscan.labels_)
        n_clusters = n_clusters[n_clusters >= 0].size
        self.assertEqual(n_clusters, 2)

    def test_n_clusters_aniso_grid(self):
        """ Tests that DBSCAN finds the correct number of clusters when
        setting n_regions > 1 with anisotropicly distributed data.
        """
        n_samples = 1500
        x, y = make_blobs(n_samples=n_samples, random_state=170)
        dbscan = DBSCAN(n_regions=2, eps=.15, max_samples=500)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        x = np.dot(x, transformation)
        x = StandardScaler().fit_transform(x)
        dataset = load_data(x=x, y=y, subset_size=300)
        dbscan.fit(dataset)
        n_clusters = np.unique(dbscan.labels_)
        n_clusters = n_clusters[n_clusters >= 0].size
        self.assertEqual(n_clusters, 4)

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
        n_clusters = np.unique(dbscan.labels_)
        n_clusters = n_clusters[n_clusters >= 0].size
        self.assertEqual(n_clusters, 0)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
