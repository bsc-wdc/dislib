import os
import sys
import unittest

sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class ImportTests(unittest.TestCase):

    def test_import_fft(self):
        from dislib.fft import fft

    def test_import_cascadecsvm(self):
        from dislib.classification import CascadeSVM

    def test_import_kmeans(self):
        from dislib.cluster import KMeans

    def test_import_dbscan(self):
        from dislib.cluster import DBSCAN


class ResultsTest(unittest.TestCase):

    def test_cascadecsvm(self):
        from dislib.classification import CascadeSVM


class DBSCANTest(unittest.TestCase):

    def test_n_clusters_blobs(self):
        from sklearn.datasets import make_blobs
        from dislib.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
        from dislib.data import load_data
        import numpy as np

        n_samples = 1500

        # Test blobs
        x, y = make_blobs(n_samples=n_samples, n_features=2, random_state=8)
        dbscan = DBSCAN(grid_dim=1, eps=.3)
        x = StandardScaler().fit_transform(x)
        dataset = load_data(x=x, y=y, subset_size=300)
        dbscan.fit(dataset)
        n_clusters = np.unique(dbscan.labels_)
        n_clusters = n_clusters[n_clusters >= 0].size
        self.assertEqual(n_clusters, 3)

    def test_n_clusters_circles(self):
        from sklearn.datasets import make_circles
        from dislib.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
        from dislib.data import load_data
        import numpy as np

        n_samples = 1500

        # Test circles
        x, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)
        dbscan = DBSCAN(grid_dim=1, eps=.15)
        x = StandardScaler().fit_transform(x)
        dataset = load_data(x=x, y=y, subset_size=300)
        dbscan.fit(dataset)
        n_clusters = np.unique(dbscan.labels_)
        n_clusters = n_clusters[n_clusters >= 0].size
        self.assertEqual(n_clusters, 2)

    def test_n_clusters_moons(self):
        from sklearn.datasets import make_moons
        from dislib.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
        from dislib.data import load_data
        import numpy as np

        n_samples = 1500

        # Test moons
        x, y = make_moons(n_samples=n_samples, noise=.05)
        dbscan = DBSCAN(grid_dim=1, eps=.3)
        x = StandardScaler().fit_transform(x)
        dataset = load_data(x=x, y=y, subset_size=300)
        dbscan.fit(dataset)
        n_clusters = np.unique(dbscan.labels_)
        n_clusters = n_clusters[n_clusters >= 0].size
        self.assertEqual(n_clusters, 2)

    def test_n_clusters_aniso(self):
        from sklearn.datasets import make_blobs
        from dislib.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
        from dislib.data import load_data
        import numpy as np

        n_samples = 1500

        # Test aniso
        x, y = make_blobs(n_samples=n_samples, random_state=170)
        dbscan = DBSCAN(grid_dim=1, eps=.15)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        x = np.dot(x, transformation)
        x = StandardScaler().fit_transform(x)
        dataset = load_data(x=x, y=y, subset_size=300)
        dbscan.fit(dataset)
        n_clusters = np.unique(dbscan.labels_)
        n_clusters = n_clusters[n_clusters >= 0].size
        self.assertEqual(n_clusters, 4)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
