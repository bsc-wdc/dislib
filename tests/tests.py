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


class DataTest(unittest.TestCase):

    def test_load_data(self):
        from sklearn.datasets import make_blobs
        from dislib.data import load_data
        import numpy as np

        x, y = make_blobs(n_samples=1500)
        data = load_data(x=x, y=y, subset_size=100)
        data.collect()

        read_x = np.empty((0, x.shape[1]))
        read_y = np.empty(0)

        for subset in data:
            read_x = np.concatenate((read_x, subset.samples))
            read_y = np.concatenate((read_y, subset.labels))

        self.assertTrue((read_x == x).all())
        self.assertTrue((read_y == y).all())
        self.assertEqual(len(data), 15)

    def test_load_libsvm_file_sparse(self):
        from dislib.data import load_libsvm_file
        from sklearn.datasets import load_svmlight_file
        import numpy as np

        data = load_libsvm_file("./tests/files/libsvm/2", 10, 780)
        data.collect()
        x,y = load_svmlight_file("./tests/files/libsvm/2", n_features=780)

        read_x = np.empty((0, x.shape[1]))
        read_y = np.empty(0)

        for subset in data:
            read_x = np.concatenate((read_x, subset.samples.toarray()))
            read_y = np.concatenate((read_y, subset.labels))

        self.assertTrue((read_x == x.toarray()).all())
        self.assertTrue((read_y == y).all())
        self.assertEqual(len(data), 6)

    def test_load_libsvm_file_dense(self):
        from dislib.data import load_libsvm_file
        from sklearn.datasets import load_svmlight_file
        import numpy as np

        data = load_libsvm_file("./tests/files/libsvm/1", 20, 780, False)
        data.collect()
        x, y = load_svmlight_file("./tests/files/libsvm/1", n_features=780)

        read_x = np.empty((0, x.shape[1]))
        read_y = np.empty(0)

        for subset in data:
            read_x = np.concatenate((read_x, subset.samples))
            read_y = np.concatenate((read_y, subset.labels))

        self.assertTrue((read_x == x.toarray()).all())
        self.assertTrue((read_y == y).all())
        self.assertEqual(len(data), 4)

    def test_load_libsvm_files_sparse(self):
        from dislib.data import load_libsvm_files
        from sklearn.datasets import load_svmlight_file
        import os

        dir_ = "./tests/files/libsvm"
        file_list = os.listdir(dir_)
        data = load_libsvm_files(dir_, 780)
        data.collect()

        for i, subset in enumerate(data):
            samples = subset.samples.toarray()
            file_ = os.path.join(dir_, file_list[i])
            x, y = load_svmlight_file(file_, n_features=780)

            self.assertTrue((samples == x).all())
            self.assertTrue((subset.labels == y).all())

        self.assertEqual(len(data), 3)

    def test_load_libsvm_files_dense(self):
        from dislib.data import load_libsvm_files
        from sklearn.datasets import load_svmlight_file
        import os

        dir_ = "./tests/files/libsvm"
        file_list = os.listdir(dir_)
        data = load_libsvm_files(dir_, 780, False)
        data.collect()

        for i, subset in enumerate(data):
            samples = subset.samples
            file_ = os.path.join(dir_, file_list[i])
            x,y = load_svmlight_file(file_, n_features=780)

            self.assertTrue((samples == x).all())
            self.assertTrue((subset.labels == y).all())

        self.assertEqual(len(data), 3)

    def test_load_csv_file(self):
        from dislib.data import load_csv_file
        import numpy as np

        csv_file = "./tests/files/csv/1"
        data = load_csv_file(csv_file, subset_size=300, n_features=122)
        data.collect()
        csv = np.loadtxt(csv_file, delimiter=",")

        read_x = np.empty((0, csv.shape[1]))

        for subset in data:
            read_x = np.concatenate((read_x, subset.samples))

        self.assertTrue((read_x == csv).all())
        self.assertEqual(len(data), 15)
        self.assertIsNone(subset.labels)

    def test_load_csv_file_labels_last(self):
        from dislib.data import load_csv_file
        import numpy as np

        csv_file = "./tests/files/csv/1"
        data = load_csv_file(csv_file, subset_size=1000, n_features=121,
                             label_col="last")
        data.collect()
        csv = np.loadtxt(csv_file, delimiter=",")

        read_x = np.empty((0, csv.shape[1] - 1))
        read_y = np.empty(0)

        for subset in data:
            read_x = np.concatenate((read_x, subset.samples))
            read_y = np.concatenate((read_y, subset.labels))

        self.assertTrue((read_x == csv[:, :-1]).all())
        self.assertTrue((read_y == csv[:, -1]).all())
        self.assertEqual(len(data), 5)

    def test_load_csv_file_labels_first(self):
        from dislib.data import load_csv_file
        import numpy as np

        csv_file = "./tests/files/csv/1"
        data = load_csv_file(csv_file, subset_size=1000, n_features=121,
                             label_col="first")
        data.collect()
        csv = np.loadtxt(csv_file, delimiter=",")

        read_x = np.empty((0, csv.shape[1] - 1))
        read_y = np.empty(0)

        for subset in data:
            read_x = np.concatenate((read_x, subset.samples))
            read_y = np.concatenate((read_y, subset.labels))

        self.assertTrue((read_x == csv[:, 1:]).all())
        self.assertTrue((read_y == csv[:, 0]).all())
        self.assertEqual(len(data), 5)

    def test_load_csv_files(self):
        from dislib.data import load_csv_files
        import numpy as np

        csv_dir = "./tests/files/csv"
        data = load_csv_files(csv_dir, n_features=122)
        data.collect()
        csv = np.loadtxt(csv_dir, delimiter=",")

        read_x = np.empty((0, csv.shape[1]))

        for subset in data:
            read_x = np.concatenate((read_x, subset.samples))

        self.assertTrue((read_x == csv).all())
        self.assertEqual(len(data), 15)
        self.assertIsNone(subset.labels)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
