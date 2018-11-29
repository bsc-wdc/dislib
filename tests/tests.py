import os
import sys
import unittest

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

from dislib.cluster import DBSCAN
from dislib.data import Subset
from dislib.data import load_csv_file
from dislib.data import load_csv_files
from dislib.data import load_data
from dislib.data import load_libsvm_file
from dislib.data import load_libsvm_files

sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# the import tests should be removed; import should be tests in class specific
#  tests
class ImportTests(unittest.TestCase):
    def test_import_fft(self):
        from dislib.fft import fft
        self.assertIsNotNone(fft)

    def test_import_cascadecsvm(self):
        from dislib.classification import CascadeSVM
        self.assertIsNotNone(CascadeSVM)

    def test_import_kmeans(self):
        from dislib.cluster import KMeans
        self.assertIsNotNone(KMeans)

    def test_import_dbscan(self):
        from dislib.cluster import DBSCAN
        self.assertIsNotNone(DBSCAN)


class ResultsTest(unittest.TestCase):
    def test_cascadecsvm(self):
        from dislib.classification import CascadeSVM
        self.assertIsNotNone(CascadeSVM)


class DBSCANTest(unittest.TestCase):
    def test_n_clusters_blobs(self):
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

    def test_n_clusters_blobs_max_samples(self):
        from sklearn.datasets import make_blobs
        from dislib.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
        from dislib.data import load_data
        import numpy as np

        n_samples = 1500

        # Test blobs
        x, y = make_blobs(n_samples=n_samples, n_features=2, random_state=8)
        dbscan = DBSCAN(grid_dim=1, eps=.3, max_samples=500)
        x = StandardScaler().fit_transform(x)
        dataset = load_data(x=x, y=y, subset_size=300)
        dbscan.fit(dataset)
        n_clusters = np.unique(dbscan.labels_)
        n_clusters = n_clusters[n_clusters >= 0].size
        self.assertEqual(n_clusters, 3)

    def test_n_clusters_circles_max_samples(self):
        from sklearn.datasets import make_circles
        from dislib.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
        from dislib.data import load_data
        import numpy as np

        n_samples = 1500

        # Test circles
        x, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)
        dbscan = DBSCAN(grid_dim=1, eps=.15, max_samples=500)
        x = StandardScaler().fit_transform(x)
        dataset = load_data(x=x, y=y, subset_size=300)
        dbscan.fit(dataset)
        n_clusters = np.unique(dbscan.labels_)
        n_clusters = n_clusters[n_clusters >= 0].size
        self.assertEqual(n_clusters, 2)

    def test_n_clusters_moons_max_samples(self):
        from sklearn.datasets import make_moons
        from dislib.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
        from dislib.data import load_data
        import numpy as np

        n_samples = 1500

        # Test moons
        x, y = make_moons(n_samples=n_samples, noise=.05)
        dbscan = DBSCAN(grid_dim=1, eps=.3, max_samples=500)
        x = StandardScaler().fit_transform(x)
        dataset = load_data(x=x, y=y, subset_size=300)
        dbscan.fit(dataset)
        n_clusters = np.unique(dbscan.labels_)
        n_clusters = n_clusters[n_clusters >= 0].size
        self.assertEqual(n_clusters, 2)

    def test_n_clusters_aniso_max_samples(self):
        from sklearn.datasets import make_blobs
        from dislib.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
        from dislib.data import load_data
        import numpy as np

        n_samples = 1500

        # Test aniso
        x, y = make_blobs(n_samples=n_samples, random_state=170)
        dbscan = DBSCAN(grid_dim=1, eps=.15, max_samples=500)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        x = np.dot(x, transformation)
        x = StandardScaler().fit_transform(x)
        dataset = load_data(x=x, y=y, subset_size=300)
        dbscan.fit(dataset)
        n_clusters = np.unique(dbscan.labels_)
        n_clusters = n_clusters[n_clusters >= 0].size
        self.assertEqual(n_clusters, 4)


class DataLoadingTest(unittest.TestCase):
    def test_load_data_with_labels(self):

        x, y = make_blobs(n_samples=1500)
        data = load_data(x=x, y=y, subset_size=100)

        read_x = np.empty((0, x.shape[1]))
        read_y = np.empty(0)

        for subset in data:
            read_x = np.concatenate((read_x, subset.samples))
            read_y = np.concatenate((read_y, subset.labels))

        self.assertTrue((read_x == x).all())
        self.assertTrue((read_y == y).all())
        self.assertEqual(len(data), 15)

    def test_load_data_without_labels(self):

        x, y = make_blobs(n_samples=1500)
        data = load_data(x=x, subset_size=100)

        read_x = np.empty((0, x.shape[1]))

        for subset in data:
            read_x = np.concatenate((read_x, subset.samples))

        self.assertTrue((read_x == x).all())
        self.assertEqual(len(data), 15)

    def test_load_libsvm_file_sparse(self):

        file_ = "tests/files/libsvm/2"
        data = load_libsvm_file(file_, 10, 780)
        data.collect()
        x, y = load_svmlight_file(file_, n_features=780)

        read_x = np.empty((0, x.shape[1]))
        read_y = np.empty(0)

        for subset in data:
            read_x = np.concatenate((read_x, subset.samples.toarray()))
            read_y = np.concatenate((read_y, subset.labels))

        self.assertTrue((read_x == x.toarray()).all())
        self.assertTrue((read_y == y).all())
        self.assertEqual(len(data), 6)

    def test_load_libsvm_file_dense(self):

        file_ = "tests/files/libsvm/1"
        data = load_libsvm_file(file_, 20, 780, False)
        data.collect()
        x, y = load_svmlight_file(file_, n_features=780)

        read_x = np.empty((0, x.shape[1]))
        read_y = np.empty(0)

        for subset in data:
            read_x = np.concatenate((read_x, subset.samples))
            read_y = np.concatenate((read_y, subset.labels))

        self.assertTrue((read_x == x.toarray()).all())
        self.assertTrue((read_y == y).all())
        self.assertEqual(len(data), 4)

    def test_load_libsvm_files_sparse(self):

        dir_ = "tests/files/libsvm"
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

        dir_ = "tests/files/libsvm"
        file_list = os.listdir(dir_)
        data = load_libsvm_files(dir_, 780, False)
        data.collect()

        for i, subset in enumerate(data):
            samples = subset.samples
            file_ = os.path.join(dir_, file_list[i])
            x, y = load_svmlight_file(file_, n_features=780)

            self.assertTrue((samples == x).all())
            self.assertTrue((subset.labels == y).all())

        self.assertEqual(len(data), 3)

    def test_load_csv_file(self):

        csv_file = "tests/files/csv/1"
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

        csv_file = "tests/files/csv/1"
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

        csv_file = "tests/files/csv/2"
        data = load_csv_file(csv_file, subset_size=100, n_features=121,
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
        self.assertEqual(len(data), 44)

    def test_load_csv_files(self):

        csv_dir = "tests/files/csv"
        file_list = os.listdir(csv_dir)
        data = load_csv_files(csv_dir, n_features=122)
        data.collect()

        for i, subset in enumerate(data):
            csv_file = os.path.join(csv_dir, file_list[i])
            csv = np.loadtxt(csv_file, delimiter=",")

            self.assertTrue((subset.samples == csv).all())

        self.assertEqual(len(data), 3)

    def test_load_csv_files_labels_last(self):

        csv_dir = "tests/files/csv"
        file_list = os.listdir(csv_dir)
        data = load_csv_files(csv_dir, n_features=122, label_col="last")
        data.collect()

        for i, subset in enumerate(data):
            csv_file = os.path.join(csv_dir, file_list[i])
            csv = np.loadtxt(csv_file, delimiter=",")

            self.assertTrue((subset.samples == csv[:, :-1]).all())
            self.assertTrue((subset.labels == csv[:, -1]).all())

        self.assertEqual(len(data), 3)

    def test_load_csv_files_labels_first(self):

        csv_dir = "tests/files/csv"
        file_list = os.listdir(csv_dir)
        data = load_csv_files(csv_dir, n_features=122, label_col="first")
        data.collect()

        for i, subset in enumerate(data):
            csv_file = os.path.join(csv_dir, file_list[i])
            csv = np.loadtxt(csv_file, delimiter=",")

            self.assertTrue((subset.samples == csv[:, 1:]).all())
            self.assertTrue((subset.labels == csv[:, 0]).all())

        self.assertEqual(len(data), 3)


class DataClassesTest(unittest.TestCase):
    def test_dataset_get_item(self):
        arr = np.array((range(10), range(10, 20)))
        dataset = load_data(arr, subset_size=2)
        samples = dataset[0].samples

        self.assertTrue((samples[0] == arr[0]).all())

    def test_dataset_len(self):
        arr = np.zeros((40, 25))
        dataset = load_data(arr, subset_size=5)

        self.assertEqual(len(dataset), 8)

    def test_dataset_append(self):
        arr = np.zeros((40, 25))
        dataset = load_data(arr, subset_size=5)
        subset = Subset(samples=arr)
        dataset.append(subset)

        self.assertEqual(len(dataset), 9)

    def test_dataset_extend(self):
        arr = np.zeros((40, 25))
        dataset = load_data(arr, subset_size=5)
        subset1 = Subset(samples=np.zeros((20, 18)))
        subset2 = Subset(samples=np.zeros((24, 18)))
        dataset.extend(subset1, subset2)

        self.assertEqual(len(dataset), 10)

    def test_dataset_collect(self):
        csv_file = "tests/files/csv/3"
        dataset = load_csv_file(csv_file, subset_size=300, n_features=122)
        dataset.collect()

        self.assertIsInstance(dataset[0], Subset)

    def test_subset_concatenate_dense(self):
        subset1 = Subset(samples=np.zeros((13, 2)))
        subset2 = Subset(samples=np.zeros((11, 2)))

        subset1.concatenate(subset2)

        self.assertEqual(subset1.samples.shape[0], 24)

    def test_subset_concatenate_sparse(self):
        m1 = csr_matrix(np.random.random((13, 2)))
        m2 = csr_matrix(np.random.random((11, 2)))
        subset1 = Subset(samples=m1)
        subset2 = Subset(samples=m2)

        subset1.concatenate(subset2)

        self.assertEqual(subset1.samples.shape[0], 24)

    def test_subset_concatenate_with_labels(self):
        subset1 = Subset(samples=np.zeros((13, 2)), labels=np.zeros((13)))
        subset2 = Subset(samples=np.zeros((11, 2)), labels=np.zeros((11)))

        subset1.concatenate(subset2)

        self.assertEqual(subset1.labels.shape[0], 24)

    def test_subset_concatenate_removing_duplicates(self):
        labels1 = np.random.random(25)
        labels2 = np.random.random(35)

        subset1 = Subset(samples=np.random.random((25, 8)), labels=labels1)
        subset2 = Subset(samples=np.random.random((35, 8)), labels=labels2)

        subset1.concatenate(subset2)
        subset2.concatenate(subset1, remove_duplicates=True)

        self.assertEqual(subset2.samples.shape[0], 60)

    def test_subset_set_label(self):
        subset = Subset(samples=np.random.random((25, 8)))
        subset.set_label(15, 3)

        self.assertEqual(subset.labels[15], 3)

    def test_subset_get_item(self):
        subset = Subset(samples=np.array([range(10), range(10, 20)]))
        item = subset[1]

        self.assertTrue((item.samples == np.array(range(10, 20))).all())

    def test_subset_get_item_with_labels(self):
        samples = np.array([range(10), range(10, 20)])
        labels = np.array([3, 4])

        subset = Subset(samples=samples, labels=labels)
        item = subset[1]

        self.assertTrue((item.samples == np.array(range(10, 20))).all())
        self.assertEqual(item.labels, 4)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
