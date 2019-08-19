# import os
import unittest
from math import ceil

import numpy as np
from scipy import sparse as sp
from scipy.sparse import issparse, csr_matrix
from sklearn.datasets import load_svmlight_file

import dislib as ds
from dislib.data.array import Array


def equal(arr1, arr2):
    if issparse(arr1) and not issparse(arr2):
        raise AttributeError('Arrays are of different type: %s != %s' % (
            type(arr1), type(arr2)))

    if issparse(arr1):
        equal = (arr1 != arr2).nnz == 0
    else:
        equal = not (arr1 != arr2).any()

    if not equal:
        print("\nArr1: \n%s" % arr1)
        print("Arr2: \n%s" % arr2)

    return equal


def equivalent_types(arr1, arr2):
    equivalent = type(arr1) == type(arr2) \
                 or (type(arr1) == np.ndarray and type(arr2) == np.matrix) \
                 or (type(arr1) == np.matrix and type(arr2) == np.ndarray)

    if not equivalent:
        print("Type(arr1): %s" % type(arr1))
        print("Type(arr2): %s" % type(arr2))

    return equivalent


def _validate_arrays(self, darray, x, block_shape):
    # Different size and type comparison if arrays have 1 or 2 dimensions
    if len(x.shape) == 1:
        n, m = len(x), 1
        self.assertEqual(type(darray.collect()[0]), type(x[0]))
        self.assertEqual(darray.shape[0], x.shape[0])
    else:
        n, m = x.shape
        self.assertEqual(type(darray.collect()[0, 0]), type(x[0, 0]))
        self.assertEqual(darray.shape, x.shape)

    bn, bm = block_shape

    self.assertTrue(equal(darray.collect(), x))
    self.assertTrue(equivalent_types(darray.collect(), x))
    self.assertEqual(type(darray), Array)

    self.assertEqual(darray._n_blocks, (ceil(n / bn), ceil(m / bm)))


def _sum_and_mult(arr, a=0, axis=0, b=1):
    return (np.sum(arr, axis=axis) + a) * b


class DataLoadingTest(unittest.TestCase):
    def test_array_constructor(self):
        """ Tests load_data
        """
        n, m = 6, 10
        bn, bm = 4, 3
        x = np.random.randint(0, 10, size=(n, m))
        darray = ds.array(x=x, blocks_shape=(bn, bm))

        _validate_arrays(self, darray, x, (bn, bm))

        x = sp.csr_matrix(x)
        darray = ds.array(x=x, blocks_shape=(bn, bm))

        _validate_arrays(self, darray, x, (bn, bm))

    def test_load_libsvm_file(self):
        """ Tests loading a LibSVM file in dense mode.
        """
        file_ = "tests/files/libsvm/1"

        x, y = load_svmlight_file(file_, n_features=780)

        bn, bm = 25, 100

        # Load SVM and store in sparse
        arr_x, arr_y = ds.load_svmlight_file(file_, (25, 100), n_features=780,
                                             store_sparse=True)

        _validate_arrays(self, arr_x, x, (bn, bm))
        _validate_arrays(self, arr_y, y, (bn, 1))

        # Load SVM and store in dense
        arr_x, arr_y = ds.load_svmlight_file(file_, (25, 100), n_features=780,
                                             store_sparse=False)

        _validate_arrays(self, arr_x, x.toarray(), (bn, bm))
        _validate_arrays(self, arr_y, y, (bn, 1))


#
#     def test_load_libsvm_files_sparse(self):
#         """ Tests loading multiple LibSVM files in sparse mode.
#         """
#         dir_ = "tests/files/libsvm"
#
#         file_list = os.listdir(dir_)
#         data = load_libsvm_files(dir_, 780)
#         data.collect()
#
#         for i, subset in enumerate(data):
#             samples = subset.samples.toarray()
#             file_ = os.path.join(dir_, file_list[i])
#             x, y = load_svmlight_file(file_, n_features=780)
#
#             self.assertTrue((samples == x).all())
#             self.assertTrue((subset.labels == y).all())
#
#         self.assertEqual(len(data), 3)
#
#     def test_load_libsvm_files_dense(self):
#         """ Tests loading multiple LibSVM files in dense mode.
#         """
#         dir_ = "tests/files/libsvm"
#
#         file_list = os.listdir(dir_)
#         data = load_libsvm_files(dir_, 780, False)
#         data.collect()
#
#         for i, subset in enumerate(data):
#             samples = subset.samples
#             file_ = os.path.join(dir_, file_list[i])
#             x, y = load_svmlight_file(file_, n_features=780)
#
#             self.assertTrue((samples == x).all())
#             self.assertTrue((subset.labels == y).all())
#
#         self.assertEqual(len(data), 3)
#
#     def test_load_csv_file(self):
#         """ Tests loading a CSV file.
#         """
#         csv_file = "tests/files/csv/1"
#
#         data = load_txt_file(csv_file, subset_size=300, n_features=122)
#         data.collect()
#         csv = np.loadtxt(csv_file, delimiter=",")
#
#         read_x = np.empty((0, csv.shape[1]))
#
#         for subset in data:
#             read_x = np.concatenate((read_x, subset.samples))
#
#         self.assertTrue((read_x == csv).all())
#         self.assertEqual(len(data), 15)
#         self.assertIsNone(subset.labels)
#
#     def test_load_csv_file_labels_last(self):
#         """ Tests loading a CSV file with labels at the last column.
#         """
#         csv_file = "tests/files/csv/1"
#
#         data = load_txt_file(csv_file, subset_size=1000, n_features=121,
#                              label_col="last")
#         data.collect()
#         csv = np.loadtxt(csv_file, delimiter=",")
#
#         read_x = np.empty((0, csv.shape[1] - 1))
#         read_y = np.empty(0)
#
#         for subset in data:
#             read_x = np.concatenate((read_x, subset.samples))
#             read_y = np.concatenate((read_y, subset.labels))
#
#         self.assertTrue((read_x == csv[:, :-1]).all())
#         self.assertTrue((read_y == csv[:, -1]).all())
#         self.assertEqual(len(data), 5)
#
#     def test_load_csv_file_labels_first(self):
#         """ Tests loading a CSV file with labels at the first column.
#         """
#         csv_file = "tests/files/csv/2"
#
#         data = load_txt_file(csv_file, subset_size=100, n_features=121,
#                              label_col="first")
#         data.collect()
#         csv = np.loadtxt(csv_file, delimiter=",")
#
#         read_x = np.empty((0, csv.shape[1] - 1))
#         read_y = np.empty(0)
#
#         for subset in data:
#             read_x = np.concatenate((read_x, subset.samples))
#             read_y = np.concatenate((read_y, subset.labels))
#
#         self.assertTrue((read_x == csv[:, 1:]).all())
#         self.assertTrue((read_y == csv[:, 0]).all())
#         self.assertEqual(len(data), 44)
#
#     def test_load_csv_files(self):
#         """ Tests loading multiple CSV files.
#         """
#         csv_dir = "tests/files/csv"
#
#         file_list = os.listdir(csv_dir)
#         data = load_txt_files(csv_dir, n_features=122)
#         data.collect()
#
#         for i, subset in enumerate(data):
#             csv_file = os.path.join(csv_dir, file_list[i])
#             csv = np.loadtxt(csv_file, delimiter=",")
#
#             self.assertTrue((subset.samples == csv).all())
#
#         self.assertEqual(len(data), 3)
#
#     def test_load_csv_files_labels_last(self):
#         """ Tests loading multiple CSV files with labels at the last column.
#         """
#         csv_dir = "tests/files/csv"
#
#         file_list = os.listdir(csv_dir)
#         data = load_txt_files(csv_dir, n_features=122, label_col="last")
#         data.collect()
#
#         for i, subset in enumerate(data):
#             csv_file = os.path.join(csv_dir, file_list[i])
#             csv = np.loadtxt(csv_file, delimiter=",")
#
#             self.assertTrue((subset.samples == csv[:, :-1]).all())
#             self.assertTrue((subset.labels == csv[:, -1]).all())
#
#         self.assertEqual(len(data), 3)
#
#     def test_load_csv_files_labels_first(self):
#         """ Tests loading multiple CSV files with labels at the first column.
#         """
#         csv_dir = "tests/files/csv"
#
#         file_list = os.listdir(csv_dir)
#         data = load_txt_files(csv_dir, n_features=122, label_col="first")
#         data.collect()
#
#         for i, subset in enumerate(data):
#             csv_file = os.path.join(csv_dir, file_list[i])
#             csv = np.loadtxt(csv_file, delimiter=",")
#
#             self.assertTrue((subset.samples == csv[:, 1:]).all())
#             self.assertTrue((subset.labels == csv[:, 0]).all())
#
#         self.assertEqual(len(data), 3)
#
#     def test_load_txt_delimiter(self):
#         """ Tests load_txt_file with a custom delimiter """
#         path_ = "tests/files/other/4"
#         data = load_txt_file(path_, n_features=122, subset_size=1000,
#                              delimiter=" ")
#         csv = np.loadtxt(path_, delimiter=" ")
#
#         self.assertTrue(np.array_equal(data.samples, csv))
#         self.assertEqual(len(data), 5)
#         self.assertIsNone(data.labels)
#
#     def test_load_txt_files_delimiter(self):
#         """ Tests loading multiple files with a custom delimiter"""
#         path_ = "tests/files/other"
#
#         file_list = os.listdir(path_)
#         data = load_txt_files(path_, n_features=122, delimiter=" ")
#         data.collect()
#
#         for i, subset in enumerate(data):
#             file_ = os.path.join(path_, file_list[i])
#             read_data = np.loadtxt(file_, delimiter=" ")
#
#             self.assertTrue(np.array_equal(subset.samples, read_data))
#
#         self.assertEqual(len(data), 2)
#
#


class ArrayTest(unittest.TestCase):
    def test_sizes(self):
        """ Tests sizes consistency. """

        x_size, y_size = 40, 25
        bn, bm = 9, 11
        x = np.random.randint(10, size=(x_size, y_size))
        darray = ds.array(x=x, blocks_shape=(bn, bm))

        self.assertEqual(darray.shape, (x_size, y_size))

        self.assertEqual(darray._n_blocks,
                         (ceil(x_size / bn), ceil(y_size / bm)))
        self.assertEqual(darray._blocks_shape, (bn, bm))

        x = sp.csr_matrix(x)
        darray = ds.array(x=x, blocks_shape=(bn, bm))

        self.assertEqual(darray.shape, (x_size, y_size))
        self.assertEqual(darray._n_blocks,
                         (ceil(x_size / bn), ceil(y_size / bm)))
        self.assertEqual(darray._blocks_shape, (bn, bm))

    def test_iterate_rows(self):
        """ Testing the row _iterator of the ds.array
        """
        x_size = 2
        # Dense
        x = np.random.randint(10, size=(10, 10))
        data = ds.array(x=x, blocks_shape=(x_size, 2))
        for i, r in enumerate(data._iterator(axis='rows')):
            r_data = r.collect()
            r_x = x[i * x_size:(i + 1) * x_size]
            self.assertTrue(equal(r_data, r_x))

        # Sparse
        x = sp.csr_matrix(x)
        data = ds.array(x=x, blocks_shape=(x_size, 2))
        for i, r in enumerate(data._iterator(axis='rows')):
            r_data = r.collect()
            r_x = x[i * x_size:(i + 1) * x_size]
            self.assertTrue(equal(r_data, r_x))

    def test_iterate_cols(self):
        """ Tests iterating through the rows of the ds.array
        """
        bn, bm = 2, 2
        # Dense
        x = np.random.randint(10, size=(10, 10))
        data = ds.array(x=x, blocks_shape=(bn, bm))

        for i, c in enumerate(data._iterator(axis='columns')):
            c_data = c.collect()
            c_x = x[:, i * bm:(i + 1) * bm]
            self.assertTrue(equal(c_data, c_x))

        # Sparse
        x = sp.csr_matrix(x)
        data = ds.array(x=x, blocks_shape=(bn, bm))

        for i, c in enumerate(data._iterator(axis='columns')):
            c_data = c.collect()
            c_x = x[:, i * bm:(i + 1) * bm]
            self.assertTrue(equal(c_data, c_x))

    def test_get_item(self):
        """ Tests get item of the ds.array
        """
        bn, bm = 2, 2
        x = np.random.randint(10, size=(10, 10))
        data = ds.array(x=x, blocks_shape=(bn, bm))

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                element = data[i, j].collect()
                self.assertEqual(element, x[i, j])

        # Try indexing with irregular array
        x = x[1:, 1:]
        data = data[1:, 1:]
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                element = data[i, j].collect()
                self.assertEqual(element, x[i, j])

    def test_get_slice_dense(self):
        """ Tests get a dense slice of the ds.array
        """
        bn, bm = 5, 5
        x = np.random.randint(100, size=(30, 30))
        data = ds.array(x=x, blocks_shape=(bn, bm))

        slice_indices = [(7, 22, 7, 22),  # many row-column
                         (6, 8, 6, 8),  # single block row-column
                         (6, 8, None, None),  # single-block rows, all columns
                         (None, None, 6, 8),  # all rows, single-block columns
                         (15, 16, 15, 16),  # single element
                         # (-10, -5, -10, -5),  # out-of-bounds (not
                         # implemented)
                         # (-10, 5, -10, 5),  # out-of-bounds (not implemented)
                         (21, 40, 21, 40)]  # out-of-bounds (correct)

        for top, bot, left, right in slice_indices:
            got = data[top:bot, left:right].collect()
            expected = x[top:bot, left:right]

            self.assertTrue(equal(got, expected))

        # Try slicing with irregular array
        x = x[1:, 1:]
        data = data[1:, 1:]

        for top, bot, left, right in slice_indices:
            got = data[top:bot, left:right].collect()
            expected = x[top:bot, left:right]

            self.assertTrue(equal(got, expected))

    def test_get_slice_sparse(self):
        """ Tests get a sparse slice of the ds.array
        """
        bn, bm = 5, 5
        x = np.random.randint(100, size=(30, 30))
        x = sp.csr_matrix(x)
        data = ds.array(x=x, blocks_shape=(bn, bm))

        slice_indices = [(7, 22, 7, 22),  # many row-column
                         (6, 8, 6, 8),  # single block row-column
                         (6, 8, None, None),  # single-block rows, all columns
                         (None, None, 6, 8),  # all rows, single-block columns
                         (15, 16, 15, 16),  # single element
                         # (-10, -5, -10, -5),  # out-of-bounds (not
                         # implemented)
                         # (-10, 5, -10, 5),  # out-of-bounds (not implemented)
                         (21, 40, 21, 40)]  # out-of-bounds (correct)

        for top, bot, left, right in slice_indices:
            got = data[top:bot, left:right].collect()
            expected = x[top:bot, left:right]

            self.assertTrue(equal(got, expected))

        # Try slicing with irregular array
        x = x[1:, 1:]
        data = data[1:, 1:]

        for top, bot, left, right in slice_indices:
            got = data[top:bot, left:right].collect()
            expected = x[top:bot, left:right]
            self.assertTrue(equal(got, expected))

    def test_index_rows_dense(self):
        """ Tests get a slice of rows from the ds.array using lists as index
        """
        bn, bm = 5, 5
        x = np.random.randint(100, size=(10, 10))
        # x = sp.csr_matrix(x)
        data = ds.array(x=x, blocks_shape=(bn, bm))

        # indices_lists = [([0, 5], [0, 5]),  # one from each block
        #                  ([0, 1, 3, 4], [0, 1, 2, 4]),  # all from first
        #                  ]
        indices_lists = [([0, 5], [0, 5])]

        for rows, cols in indices_lists:
            got = data[rows].collect()

            expected = x[rows]

            self.assertTrue(equal(got, expected))

        # Try slicing with irregular array
        x = x[1:, 1:]
        data = data[1:, 1:]

        for rows, cols in indices_lists:
            got = data[rows].collect()
            expected = x[rows]

            self.assertTrue(equal(got, expected))

    def test_index_cols_dense(self):
        """ Tests get a slice of cols from the ds.array using lists as index
        """
        bn, bm = 5, 5
        x = np.random.randint(100, size=(10, 10))
        # x = sp.csr_matrix(x)
        data = ds.array(x=x, blocks_shape=(bn, bm))

        indices_lists = [([0, 5], [0, 5]),  # one from each block
                         ([0, 1, 3, 4], [0, 1, 2, 4]),  # all from first
                         ]

        for rows, cols in indices_lists:
            got = data[:, cols].collect()
            expected = x[:, cols]

            self.assertTrue(equal(got, expected))

        # Try slicing with irregular array
        x = x[1:, 1:]
        data = data[1:, 1:]

        for rows, cols in indices_lists:
            got = data[:, cols].collect()
            expected = x[:, cols]

            self.assertTrue(equal(got, expected))

    def test_transpose(self):
        """ Tests array transpose."""

        x_size, y_size = 4, 68
        bn, bm = 2, 3

        x = np.random.randint(10, size=(x_size, y_size))
        darray = ds.array(x=x, blocks_shape=(bn, bm))

        darray_t = darray.transpose(mode='all')
        _validate_arrays(self, darray_t, x.transpose(), (bm, bn))
        # ensure that original data was not modified
        _validate_arrays(self, darray, x, (bn, bm))

        darray_t = darray.transpose(mode='rows')
        _validate_arrays(self, darray_t, x.transpose(), (bm, bn))
        # ensure that original data was not modified
        _validate_arrays(self, darray, x, (bn, bm))

        darray_t = darray.transpose(mode='columns')
        _validate_arrays(self, darray_t, x.transpose(), (bm, bn))
        # ensure that original data was not modified
        _validate_arrays(self, darray, x, (bn, bm))

        self.assertRaises(Exception, darray.transpose, 'invalid')

    def test_random(self):
        """ Tests random array """
        arr1 = ds.random_array((93, 177), (43, 31), random_state=88)

        self.assertEqual(arr1.shape, arr1.collect().shape)
        self.assertEqual(arr1._n_blocks, (3, 6))
        self.assertEqual(arr1._blocks_shape, (43, 31))
        self.assertEqual(arr1._blocks[2][0].shape, (7, 31))
        self.assertEqual(arr1._blocks[2][5].shape, (7, 22))
        self.assertEqual(arr1._blocks[0][5].shape, (43, 22))
        self.assertEqual(arr1._blocks[0][0].shape, (43, 31))

        arr2 = ds.random_array((93, 177), (43, 31), random_state=88)
        arr3 = ds.random_array((93, 177), (43, 31), random_state=666)

        arr4 = ds.random_array((193, 77), (21, 51))
        arr5 = ds.random_array((193, 77), (21, 51))

        self.assertTrue(np.array_equal(arr1.collect(), arr2.collect()))
        self.assertFalse(np.array_equal(arr1.collect(), arr3.collect()))
        self.assertFalse(np.array_equal(arr4.collect(), arr5.collect()))

    def test_apply_axis(self):
        """ Tests apply along axis"""
        x = ds.array(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                     blocks_shape=(2, 2))

        x1 = ds.apply_along_axis(_sum_and_mult, 0, x)
        self.assertTrue(x1.shape, (1, 3))
        self.assertTrue(x1._blocks_shape, (1, 2))
        self.assertTrue(np.array_equal(x1.collect(), np.array([12, 15, 18])))

        x1 = ds.apply_along_axis(_sum_and_mult, 1, x)
        self.assertTrue(x1.shape, (3, 1))
        self.assertTrue(x1._blocks_shape, (2, 1))
        self.assertTrue(np.array_equal(x1.collect(), np.array([6, 15, 24])))

        x1 = ds.apply_along_axis(_sum_and_mult, 1, x, 2)
        self.assertTrue(x1.shape, (3, 1))
        self.assertTrue(x1._blocks_shape, (2, 1))
        self.assertTrue(np.array_equal(x1.collect(), np.array([8, 17, 26])))

        x1 = ds.apply_along_axis(_sum_and_mult, 1, x, b=2)
        self.assertTrue(x1.shape, (3, 1))
        self.assertTrue(x1._blocks_shape, (2, 1))
        self.assertTrue(np.array_equal(x1.collect(), np.array([12, 30, 48])))

        x1 = ds.apply_along_axis(_sum_and_mult, 1, x, 1, b=2)
        self.assertTrue(x1.shape, (3, 1))
        self.assertTrue(x1._blocks_shape, (2, 1))
        self.assertTrue(np.array_equal(x1.collect(), np.array([14, 32, 50])))

        x = ds.array(sp.csr_matrix([[1, 0, -1], [0, 5, 0], [7, 8, 0]]),
                     blocks_shape=(2, 2))
        x1 = ds.apply_along_axis(_sum_and_mult, 0, x, 1, b=2)
        self.assertTrue(x1.shape, (1, 3))
        self.assertTrue(x1._blocks_shape, (1, 2))
        self.assertTrue(np.array_equal(x1.collect(), np.array([18, 28, 0])))

        x = ds.array(sp.csr_matrix([[1, 0, -1], [0, 5, 0], [7, 8, 0]]),
                     blocks_shape=(2, 2))
        x1 = ds.apply_along_axis(_sum_and_mult, 0, x, 1, b=2)
        self.assertTrue(x1.shape, (1, 3))
        self.assertTrue(x1._blocks_shape, (1, 2))
        self.assertTrue((x1.collect() == np.array([18, 28, 0])).all())

    def test_apply_sparse(self):
        """ Tests apply with sparse data """
        x_d = ds.array(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                       blocks_shape=(2, 2))

        x_sp = ds.array(csr_matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                        blocks_shape=(2, 2))

        x1 = ds.apply_along_axis(_sum_and_mult, 0, x_d)
        x2 = ds.apply_along_axis(_sum_and_mult, 0, x_sp)

        self.assertTrue(np.array_equal(x1.collect(), x2.collect()))

    def test_array_functions(self):
        """ Tests various array functions """
        x = ds.array(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                     blocks_shape=(2, 2))

        self.assertTrue((x.min().collect() == [1, 2, 3]).all())
        self.assertTrue((x.max().collect() == [7, 8, 9]).all())
        self.assertTrue((x.mean().collect() == [4, 5, 6]).all())
        self.assertTrue((x.sum().collect() == [12, 15, 18]).all())


def main():
    unittest.main()


if __name__ == '__main__':
    main()
