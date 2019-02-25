import os
import unittest

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import make_blobs

from dislib.data import Subset, Dataset
from dislib.data import load_data
from dislib.data import load_libsvm_file, load_libsvm_files
from dislib.data import load_txt_file
from dislib.data import load_txt_files


class DataLoadingTest(unittest.TestCase):
    def test_load_data_with_labels(self):
        """ Tests load_data with a labeled dataset.
        """
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
        """ Tests load_data with an unlabeled sparse and dense dataset.
        """
        x = np.random.random((100, 2))
        dataset = load_data(x=x, subset_size=10)

        self.assertTrue(np.array_equal(dataset.samples, x))
        self.assertEqual(len(dataset), 10)
        self.assertFalse(dataset.sparse)

        x = csr_matrix(x)
        dataset = load_data(x=x, subset_size=10)

        self.assertTrue(np.array_equal(dataset.samples.toarray(), x.toarray()))
        self.assertEqual(len(dataset), 10)
        self.assertTrue(dataset.sparse)

    def test_load_libsvm_file_sparse(self):
        """ Tests loading a LibSVM file in sparse mode.
        """
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
        """ Tests loading a LibSVM file in dense mode.
        """
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
        """ Tests loading multiple LibSVM files in sparse mode.
        """
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
        """ Tests loading multiple LibSVM files in dense mode.
        """
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
        """ Tests loading a CSV file.
        """
        csv_file = "tests/files/csv/1"

        data = load_txt_file(csv_file, subset_size=300, n_features=122)
        data.collect()
        csv = np.loadtxt(csv_file, delimiter=",")

        read_x = np.empty((0, csv.shape[1]))

        for subset in data:
            read_x = np.concatenate((read_x, subset.samples))

        self.assertTrue((read_x == csv).all())
        self.assertEqual(len(data), 15)
        self.assertIsNone(subset.labels)

    def test_load_csv_file_labels_last(self):
        """ Tests loading a CSV file with labels at the last column.
        """
        csv_file = "tests/files/csv/1"

        data = load_txt_file(csv_file, subset_size=1000, n_features=121,
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
        """ Tests loading a CSV file with labels at the first column.
        """
        csv_file = "tests/files/csv/2"

        data = load_txt_file(csv_file, subset_size=100, n_features=121,
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
        """ Tests loading multiple CSV files.
        """
        csv_dir = "tests/files/csv"

        file_list = os.listdir(csv_dir)
        data = load_txt_files(csv_dir, n_features=122)
        data.collect()

        for i, subset in enumerate(data):
            csv_file = os.path.join(csv_dir, file_list[i])
            csv = np.loadtxt(csv_file, delimiter=",")

            self.assertTrue((subset.samples == csv).all())

        self.assertEqual(len(data), 3)

    def test_load_csv_files_labels_last(self):
        """ Tests loading multiple CSV files with labels at the last column.
        """
        csv_dir = "tests/files/csv"

        file_list = os.listdir(csv_dir)
        data = load_txt_files(csv_dir, n_features=122, label_col="last")
        data.collect()

        for i, subset in enumerate(data):
            csv_file = os.path.join(csv_dir, file_list[i])
            csv = np.loadtxt(csv_file, delimiter=",")

            self.assertTrue((subset.samples == csv[:, :-1]).all())
            self.assertTrue((subset.labels == csv[:, -1]).all())

        self.assertEqual(len(data), 3)

    def test_load_csv_files_labels_first(self):
        """ Tests loading multiple CSV files with labels at the first column.
        """
        csv_dir = "tests/files/csv"

        file_list = os.listdir(csv_dir)
        data = load_txt_files(csv_dir, n_features=122, label_col="first")
        data.collect()

        for i, subset in enumerate(data):
            csv_file = os.path.join(csv_dir, file_list[i])
            csv = np.loadtxt(csv_file, delimiter=",")

            self.assertTrue((subset.samples == csv[:, 1:]).all())
            self.assertTrue((subset.labels == csv[:, 0]).all())

        self.assertEqual(len(data), 3)

    def test_load_txt_delimiter(self):
        """ Tests load_txt_file with a custom delimiter """
        path_ = "tests/files/other/4"
        data = load_txt_file(path_, n_features=122, subset_size=1000,
                             delimiter=" ")
        csv = np.loadtxt(path_, delimiter=" ")

        self.assertTrue(np.array_equal(data.samples, csv))
        self.assertEqual(len(data), 5)
        self.assertIsNone(data.labels)

    def test_load_txt_files_delimiter(self):
        """ Tests loading multiple files with a custom delimiter"""
        path_ = "tests/files/other"

        file_list = os.listdir(path_)
        data = load_txt_files(path_, n_features=122, delimiter=" ")
        data.collect()

        for i, subset in enumerate(data):
            file_ = os.path.join(path_, file_list[i])
            read_data = np.loadtxt(file_, delimiter=" ")

            self.assertTrue(np.array_equal(subset.samples, read_data))

        self.assertEqual(len(data), 2)


class DatasetTest(unittest.TestCase):
    def test_get_item(self):
        """ Tests Dataset item getter. """
        arr = np.array((range(10), range(10, 20)))
        dataset = load_data(arr, subset_size=2)
        samples = dataset[0].samples

        self.assertTrue((samples[0] == arr[0]).all())

    def test_len(self):
        """ Tests len() on a Dataset. """
        arr = np.zeros((40, 25))
        dataset = load_data(arr, subset_size=5)

        self.assertEqual(len(dataset), 8)

    def test_append(self):
        """ Tests Dataset's append(). """
        arr = np.zeros((40, 25))
        dataset = load_data(arr, subset_size=5)
        subset = Subset(samples=arr)
        dataset.append(subset)

        self.assertEqual(len(dataset), 9)

    def test_extend(self):
        """ Tests Dataset's extend(). """
        arr = np.zeros((40, 25))
        dataset = load_data(arr, subset_size=5)
        subset1 = Subset(samples=np.zeros((20, 18)))
        subset2 = Subset(samples=np.zeros((24, 18)))
        dataset.extend([subset1, subset2])

        self.assertEqual(len(dataset), 10)

    def test_collect(self):
        """ Tests Dataset's collect(). """
        csv_file = "tests/files/csv/3"

        dataset = load_txt_file(csv_file, subset_size=300, n_features=122)
        dataset.collect()

        self.assertIsInstance(dataset[0], Subset)

    def test_samples_labels(self):
        """ Tests the access to Dataset.samples and Dataset.labels """
        csv_file = "tests/files/csv/3"

        dataset = load_txt_file(csv_file, subset_size=300, n_features=121,
                                label_col="last")

        self.assertEqual(dataset.samples.shape[0], 4179)
        self.assertEqual(dataset.labels.shape[0], 4179)

    def test_empty_labels(self):
        """ Tests the access Dataset.labels for unlabeled datasets """
        csv_file = "tests/files/csv/3"

        dataset = load_txt_file(csv_file, subset_size=300, n_features=122)

        self.assertIsNone(dataset.labels)

    def test_min_max_features(self):
        """ Tests that min_features and max_features correctly return min
        and max values in a toy dataset.
        """
        s1 = Subset(samples=np.array([[1, 2], [4, 5], [2, 2], [6, 6]]),
                    labels=np.array([0, 1, 1, 1]))
        s2 = Subset(samples=np.array([[7, 8], [9, 8], [0, 4]]),
                    labels=np.array([0, 1, 1]))
        s3 = Subset(samples=np.array([[3, 9], [0, 7], [6, 1], [0, 8]]),
                    labels=np.array([0, 1, 1, 1]))
        dataset = Dataset(n_features=2)
        dataset.append(s1)
        dataset.append(s2)
        dataset.append(s3)

        min_ = dataset.min_features()
        max_ = dataset.max_features()

        self.assertTrue(np.array_equal(min_, np.array([0, 1])))
        self.assertTrue(np.array_equal(max_, np.array([9, 9])))

    def test_min_max_features_sparse(self):
        """ Tests that min_features and max_features correctly return min
        and max values with sparse dataset. """

        file_ = "tests/files/libsvm/1"
        sparse = load_libsvm_file(file_, 10, 780)
        dense = load_libsvm_file(file_, 10, 780, store_sparse=False)

        max_sp = sparse.max_features()
        max_d = dense.max_features()
        min_sp = sparse.min_features()
        min_d = sparse.min_features()

        self.assertTrue(np.array_equal(max_sp, max_d))
        self.assertTrue(np.array_equal(min_sp, min_d))

    def test_samples_sparse(self):
        """ Tests that Dataset.samples works for sparse data."""
        file_ = "tests/files/libsvm/1"
        sparse = load_libsvm_file(file_, 10, 780)
        dense = load_libsvm_file(file_, 10, 780, store_sparse=False)
        samples_sp = sparse.samples.toarray()
        samples_d = dense.samples

        self.assertTrue(np.array_equal(samples_sp, samples_d))

    def test_transpose_dense(self):
        """ Tests Dataset.transpose() in dense format."""

        # Initial dataset: 1 subset
        data = np.random.random((4, 4))
        dataset = load_data(data, subset_size=4)

        # Transpose with same n_subsets
        data_t = data.transpose()
        dataset_t = dataset.transpose()

        self.assertTrue(np.allclose(dataset_t.samples, data_t))
        self.assertEqual(len(dataset), len(dataset_t))
        self.assertTrue(dataset_t.labels is None)

        # Transpose with n_subsets=2
        dataset_t = dataset.transpose(n_subsets=2)

        self.assertTrue(np.allclose(dataset_t.samples, data_t))
        self.assertEqual(len(dataset_t), 2)
        self.assertTrue(dataset_t.labels is None)

        # Initial dataset: 3 subsets
        data = np.random.random((12, 8))
        dataset = load_data(data, subset_size=4)

        # Transpose with same n_subsets
        data_t = data.transpose()
        dataset_t = dataset.transpose()

        self.assertTrue(np.allclose(dataset_t.samples, data_t))
        self.assertEqual(len(dataset), len(dataset_t))
        self.assertTrue(dataset_t.labels is None)
        self.assertEqual(dataset_t.subsets_sizes(), [2, 2, 4])

        # Transpose with n_subsets=2
        dataset_t = dataset.transpose(n_subsets=2)

        self.assertTrue(np.allclose(dataset_t.samples, data_t))
        self.assertEqual(len(dataset_t), 2)
        self.assertTrue(dataset_t.labels is None)
        self.assertEqual(dataset_t.subsets_sizes(), [4, 4])

    def test_transpose_sparse(self):
        """ Tests Dataset.transpose() in sparse csr format."""

        # Initial dataset: 1 subset
        data = csr_matrix(np.random.random((4, 4)))
        dataset = load_data(data, subset_size=4)

        # Transpose with same n_subsets
        data_t = data.transpose()
        dataset_t = dataset.transpose()

        self.assertTrue(
            np.allclose(dataset_t.samples.toarray(), data_t.toarray()))
        self.assertEqual(len(dataset), len(dataset_t))
        self.assertTrue(dataset_t.labels is None)

        # Transpose with n_subsets=2
        dataset_t = dataset.transpose(n_subsets=2)
        self.assertTrue(
            np.allclose(dataset_t.samples.toarray(), data_t.toarray()))
        self.assertEqual(len(dataset_t), 2)
        self.assertTrue(dataset_t.labels is None)

        # Initial dataset: 3 subsets
        data = csr_matrix(np.random.random((12, 8)))
        dataset = load_data(data, subset_size=4)

        # Transpose with same n_subsets
        data_t = data.transpose()
        dataset_t = dataset.transpose()

        self.assertTrue(
            np.allclose(dataset_t.samples.toarray(), data_t.toarray()))
        self.assertEqual(len(dataset), len(dataset_t))
        self.assertTrue(dataset_t.labels is None)
        self.assertEqual(dataset_t.subsets_sizes(), [2, 2, 4])

        # Transpose with n_subsets=2
        dataset_t = dataset.transpose(n_subsets=2)

        self.assertTrue(
            np.allclose(dataset_t.samples.toarray(), data_t.toarray()))
        self.assertEqual(len(dataset_t), 2)
        self.assertTrue(dataset_t.labels is None)
        self.assertEqual(dataset_t.subsets_sizes(), [4, 4])

    def test_subsets_sizes(self):
        """ Tests Dataset.subsets_sizes() returns the correct subset sizes."""

        data = np.random.random((100, 1))
        data_csr = csr_matrix(data)

        dense = load_data(data, subset_size=30)
        sparse = load_data(data_csr, subset_size=30)

        self.assertTrue(sparse.subsets_sizes(), [30, 30, 30, 10])
        self.assertTrue(dense.subsets_sizes(), [30, 30, 30, 10])

        dense = load_data(data, subset_size=25)
        sparse = load_data(data_csr, subset_size=25)

        self.assertTrue(sparse.subsets_sizes(), [25, 25, 25, 25])
        self.assertTrue(dense.subsets_sizes(), [25, 25, 25, 25])

        dense = load_data(data, subset_size=100)
        sparse = load_data(data_csr, subset_size=100)

        self.assertTrue(sparse.subsets_sizes(), [100])
        self.assertTrue(dense.subsets_sizes(), [100])

        dense = load_data(data, subset_size=1)
        sparse = load_data(data_csr, subset_size=1)

        self.assertTrue(sparse.subsets_sizes(), [1] * 100)
        self.assertTrue(dense.subsets_sizes(), [1] * 100)


class SubsetTest(unittest.TestCase):
    def test_concatenate_dense(self):
        """ Tests the concatenation of two dense Subsets. """
        subset1 = Subset(samples=np.zeros((13, 2)))
        subset2 = Subset(samples=np.zeros((11, 2)))

        subset1.concatenate(subset2)

        self.assertEqual(subset1.samples.shape[0], 24)

    def test_concatenate_sparse(self):
        """ Tests the concatenation of two sparse Subsets. """
        m1 = csr_matrix(np.random.random((13, 2)))
        m2 = csr_matrix(np.random.random((11, 2)))
        subset1 = Subset(samples=m1)
        subset2 = Subset(samples=m2)

        subset1.concatenate(subset2)

        self.assertEqual(subset1.samples.shape[0], 24)

    def test_concatenate_with_labels(self):
        """ Tests the concatenation of two dense labeled Subsets.
        """
        subset1 = Subset(samples=np.zeros((13, 2)), labels=np.zeros((13)))
        subset2 = Subset(samples=np.zeros((11, 2)), labels=np.zeros((11)))

        subset1.concatenate(subset2)

        self.assertEqual(subset1.labels.shape[0], 24)

    def test_concatenate_removing_duplicates(self):
        """ Tests that concatenate() removes duplicate samples.
        """
        labels1 = np.random.random(25)
        labels2 = np.random.random(35)

        subset1 = Subset(samples=np.random.random((25, 8)), labels=labels1)
        subset2 = Subset(samples=np.random.random((35, 8)), labels=labels2)

        subset1.concatenate(subset2)
        subset2.concatenate(subset1, remove_duplicates=True)

        self.assertEqual(subset2.samples.shape[0], 60)

    def test_set_label(self):
        """ Tests setting a label.
        """
        subset = Subset(samples=np.random.random((25, 8)))
        subset.set_label(15, 3)

        self.assertEqual(subset.labels[15], 3)

    def test_get_item(self):
        """ Tests Subset's item getter.
        """
        subset = Subset(samples=np.array([range(10), range(10, 20)]))
        item = subset[1]

        self.assertTrue((item.samples == np.array(range(10, 20))).all())

    def test_get_item_with_labels(self):
        """ Tests Subset's item getter with labeled samples.
        """
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
