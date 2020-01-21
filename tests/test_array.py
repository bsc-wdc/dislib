import unittest
from math import ceil

import numpy as np
from parameterized import parameterized
from pycompss.api.api import compss_wait_on
from scipy import sparse as sp
from sklearn.datasets import load_svmlight_file

import dislib as ds


def _sum_and_mult(arr, a=0, axis=0, b=1):
    return (np.sum(arr, axis=axis) + a) * b


def _equal_arrays(x1, x2):
    if sp.issparse(x1):
        return np.allclose(x1.toarray(), x2.toarray())
    else:
        return np.allclose(np.squeeze(x1), np.squeeze(x2))


def _gen_random_arrays(fmt, shape=None, block_size=None):
    if not shape:
        shape = (np.random.randint(10, 100), np.random.randint(10, 100))
        block_size = (np.random.randint(1, shape[0]),
                      np.random.randint(1, shape[1]))

    if not block_size:
        block_size = (np.random.randint(1, shape[0]),
                      np.random.randint(1, shape[1]))

    if "dense" in fmt:
        x_np = np.random.random(shape)
        x = ds.array(x_np, block_size=block_size)
        return x, x_np
    elif "sparse" in fmt:
        x_sp = sp.csr_matrix(np.random.random(shape))
        x = ds.array(x_sp, block_size=block_size)
        return x, x_sp


def _gen_irregular_arrays(fmt, shape=None, block_size=None):
    if not shape:
        shape = (np.random.randint(10, 100), np.random.randint(10, 100))
        block_size = (np.random.randint(1, shape[0]),
                      np.random.randint(1, shape[1]))

    if not block_size:
        block_size = (np.random.randint(1, shape[0]),
                      np.random.randint(1, shape[1]))

    if "dense" in fmt:
        x_np = np.random.random(shape)
        x = ds.array(x_np, block_size=block_size)
        return x[1:, 1:], x_np[1:, 1:]
    elif "sparse" in fmt:
        x_sp = sp.csr_matrix(np.random.random(shape))
        x = ds.array(x_sp, block_size=block_size)
        return x[1:, 1:], x_sp[1:, 1:]


class DataLoadingTest(unittest.TestCase):

    @parameterized.expand([(_gen_random_arrays("dense", (6, 10), (4, 3))
                            + ((6, 10), (4, 3))),
                           (_gen_random_arrays("sparse", (6, 10), (4, 3))
                            + ((6, 10), (4, 3)))])
    def test_array_constructor(self, x, x_np, shape, block_size):
        """ Tests load_data
        """
        n, m = shape
        bn, bm = block_size

        self.assertTrue(x._n_blocks, ceil(n / bn) == ceil(m / bm))
        self.assertTrue(_equal_arrays(x.collect(), x_np))

    def test_load_svmlight_file(self):
        """ Tests loading a LibSVM file  """
        file_ = "tests/files/libsvm/1"

        x_np, y_np = load_svmlight_file(file_, n_features=780)

        # Load SVM and store in sparse
        x, y = ds.load_svmlight_file(file_, (25, 100), n_features=780,
                                     store_sparse=True)

        self.assertTrue(_equal_arrays(x.collect(), x_np))
        self.assertTrue(_equal_arrays(y.collect(), y_np))

        # Load SVM and store in dense
        x, y = ds.load_svmlight_file(file_, (25, 100), n_features=780,
                                     store_sparse=False)

        self.assertTrue(_equal_arrays(x.collect(), x_np.toarray()))
        self.assertTrue(_equal_arrays(y.collect(), y_np))

    def test_load_csv_file(self):
        """ Tests loading a CSV file. """
        csv_f = "tests/files/csv/1"

        data = ds.load_txt_file(csv_f, block_size=(300, 50))
        csv = np.loadtxt(csv_f, delimiter=",")

        self.assertEqual(data._top_left_shape, (300, 50))
        self.assertEqual(data._reg_shape, (300, 50))
        self.assertEqual(data.shape, (4235, 122))
        self.assertEqual(data._n_blocks, (15, 3))

        self.assertTrue(np.array_equal(data.collect(), csv))

        csv_f = "tests/files/other/4"
        data = ds.load_txt_file(csv_f, block_size=(1000, 122), delimiter=" ")
        csv = np.loadtxt(csv_f, delimiter=" ")

        self.assertTrue(np.array_equal(data.collect(), csv))


class ArrayTest(unittest.TestCase):

    @parameterized.expand([_gen_random_arrays("dense"),
                           _gen_random_arrays("sparse")])
    def test_sizes(self, x, x_np):
        """ Tests sizes consistency. """
        bshape = x._reg_shape
        shape = x_np.shape

        self.assertEqual(x.shape, shape)
        self.assertEqual(x._n_blocks, (ceil(shape[0] / bshape[0]),
                                       (ceil(shape[1] / bshape[1]))))

    @parameterized.expand([_gen_random_arrays("dense"),
                           _gen_random_arrays("sparse")])
    def test_iterate_rows(self, x, x_np):
        """ Testing the row _iterator of the ds.array """
        n_rows = x._reg_shape[0]

        for i, h_block in enumerate(x._iterator(axis='rows')):
            computed = h_block.collect()
            expected = x_np[i * n_rows: (i + 1) * n_rows]
            self.assertTrue(_equal_arrays(computed, expected))

    @parameterized.expand([_gen_random_arrays("dense"),
                           _gen_random_arrays("sparse")])
    def test_iterate_cols(self, x, x_np):
        """ Testing the row _iterator of the ds.array """
        n_cols = x._reg_shape[1]

        for i, v_block in enumerate(x._iterator(axis='columns')):
            computed = v_block.collect()
            expected = x_np[:, i * n_cols: (i + 1) * n_cols]
            self.assertTrue(_equal_arrays(computed, expected))

    def test_invalid_indexing(self):
        """ Tests invalid indexing """
        x = ds.random_array((5, 5), (1, 1))
        with self.assertRaises(IndexError):
            x[[3], [4]]
        with self.assertRaises(IndexError):
            x[7, 4]
        with self.assertRaises(IndexError):
            x["sss"]
        with self.assertRaises(NotImplementedError):
            x[:, 4]

    @parameterized.expand([_gen_random_arrays("dense"),
                           _gen_random_arrays("sparse"),
                           _gen_irregular_arrays("dense"),
                           _gen_irregular_arrays("sparse")])
    def test_indexing(self, x, x_np):
        """ Tests indexing """

        # Single row
        rows = np.random.randint(0, x.shape[0] - 1, size=min(3, x.shape[0]))

        for row in rows:
            ours = x[int(row)].collect()
            expected = x_np[row]
            self.assertTrue(_equal_arrays(ours, expected))

        # Single element
        rows = np.random.randint(0, x.shape[0] - 1, size=min(10, x.shape[0]))
        cols = np.random.randint(0, x.shape[1] - 1, size=min(10, x.shape[1]))

        for i in rows:
            for j in cols:
                element = x[int(i), int(j)].collect()
                self.assertEqual(element, x_np[int(i), int(j)])

        # Set of rows / columns
        frm = np.random.randint(0, x.shape[0] - 5, size=min(3, x.shape[0]))
        to = frm + 4

        for i, j in zip(frm, to):
            ours = x[int(i):int(j)].collect()
            expected = x_np[i:j]
            self.assertTrue(_equal_arrays(ours, expected))

        frm = np.random.randint(0, x.shape[1] - 5, size=min(3, x.shape[1]))
        to = frm + 4

        for i, j in zip(frm, to):
            ours = x[:, int(i):int(j)].collect()
            expected = x_np[:, i:j]
            self.assertTrue(_equal_arrays(ours, expected))

        # Set of elements
        i = int(np.random.randint(0, x.shape[0] - 5, size=1))
        j = int(np.random.randint(0, x.shape[1] - 5, size=1))

        ours = x[i:i + 1, j:j + 1].collect()
        expected = x_np[i:i + 1, j:j + 1]
        self.assertTrue(_equal_arrays(ours, expected))

        ours = x[i:i + 100, j:j + 100].collect()
        expected = x_np[i:i + 100, j:j + 100]
        self.assertTrue(_equal_arrays(ours, expected))

        ours = x[i:i + 4, j:j + 4].collect()
        expected = x_np[i:i + 4, j:j + 4]
        self.assertTrue(_equal_arrays(ours, expected))

    @parameterized.expand([_gen_random_arrays("dense"),
                           _gen_random_arrays("sparse"),
                           _gen_irregular_arrays("dense"),
                           _gen_irregular_arrays("sparse"),
                           _gen_irregular_arrays("sparse", (98, 10), (85, 2)) +
                           (None, [0, 1, 2, 5]),
                           _gen_irregular_arrays("sparse", (10, 98), (2, 85)) +
                           ([0, 1, 2, 5], None)])
    def test_fancy_indexing(self, x, x_np, rows=None, cols=None):
        """ Tests fancy indexing """

        # Non-consecutive rows / cols
        if not rows:
            rows = np.random.randint(0, x.shape[0] - 1, min(5, x.shape[0]))
            rows = np.unique(sorted(rows))

        ours = x[rows].collect()
        expected = x_np[rows]
        self.assertTrue(_equal_arrays(ours, expected))

        if not cols:
            cols = np.random.randint(0, x.shape[1] - 1, min(5, x.shape[1]))
            cols = np.unique(sorted(cols))

        ours = x[:, cols].collect()
        expected = x_np[:, cols]
        self.assertTrue(_equal_arrays(ours, expected))

    @parameterized.expand([_gen_random_arrays("dense"),
                           _gen_random_arrays("sparse"),
                           _gen_random_arrays("dense", (33, 34), (2, 33)),
                           _gen_irregular_arrays("dense"),
                           _gen_irregular_arrays("sparse")])
    def test_get_slice_shapes(self, x, x_np):
        """ Tests that shapes are correct after slicing """
        reg_shape = x._reg_shape
        tl_shape = x._top_left_shape[0]
        sliced = x[1:, x.shape[1] - 1:x.shape[1]]

        if tl_shape > 1:
            tl_shape -= 1
        else:
            tl_shape = compss_wait_on(x._blocks[1][0]).shape[0]

        self.assertEqual(sliced._top_left_shape, (tl_shape, 1))
        self.assertEqual(sliced._reg_shape, reg_shape)
        self.assertEqual(sliced.shape, (x.shape[0] - 1, 1))

        tl = compss_wait_on(sliced._blocks[0][0])
        reg = compss_wait_on(sliced._blocks[1][0])
        self.assertEqual(tl.shape, (tl_shape, 1))
        rshape = min(x._reg_shape[0], x.shape[0] - x._top_left_shape[0])
        self.assertEqual(reg.shape, (rshape, 1))

    @parameterized.expand([_gen_random_arrays("dense"),
                           _gen_random_arrays("dense", (1, 10), (1, 2)),
                           _gen_random_arrays("dense", (10, 1), (3, 1)),
                           _gen_random_arrays("sparse"),
                           _gen_irregular_arrays("dense"),
                           _gen_irregular_arrays("sparse")])
    def test_transpose(self, x, x_np):
        """ Tests array transpose."""
        x_np_t = x_np.transpose()
        b0, b1 = x._n_blocks

        x_t = x.transpose(mode="all")
        self.assertTrue(_equal_arrays(x_t.collect(), x_np_t))
        self.assertEqual((b1, b0), x_t._n_blocks)

        x_t = x.transpose(mode="rows")
        self.assertTrue(_equal_arrays(x_t.collect(), x_np_t))
        self.assertEqual((b1, b0), x_t._n_blocks)

        x_t = x.transpose(mode="columns")
        self.assertTrue(_equal_arrays(x_t.collect(), x_np_t))
        self.assertEqual((b1, b0), x_t._n_blocks)

        with self.assertRaises(Exception):
            x.transpose(mode="invalid")

    def test_random(self):
        """ Tests random array """
        arr1 = ds.random_array((93, 177), (43, 31), random_state=88)

        self.assertEqual(arr1.shape, arr1.collect().shape)
        self.assertEqual(arr1._n_blocks, (3, 6))
        self.assertEqual(arr1._reg_shape, (43, 31))
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

    @parameterized.expand([(ds.array([[1, 2, 3],
                                      [4, 5, 6],
                                      [7, 8, 9]], (2, 2)),),
                           (ds.array(sp.csr_matrix([[1, 2, 3],
                                                    [4, 5, 6],
                                                    [7, 8, 9]]), (2, 2)),)])
    def test_apply_axis(self, x):
        """ Tests apply along axis """
        x1 = ds.apply_along_axis(_sum_and_mult, 0, x)
        self.assertTrue(x1.shape, (1, 3))
        self.assertTrue(x1._reg_shape, (1, 2))
        self.assertTrue(
            np.array_equal(x1.collect(), np.array([12, 15, 18])))

        x1 = ds.apply_along_axis(_sum_and_mult, 1, x)
        self.assertTrue(x1.shape, (3, 1))
        self.assertTrue(x1._reg_shape, (2, 1))
        self.assertTrue(
            np.array_equal(x1.collect(), np.array([6, 15, 24])))

        x1 = ds.apply_along_axis(_sum_and_mult, 1, x, 2)
        self.assertTrue(x1.shape, (3, 1))
        self.assertTrue(x1._reg_shape, (2, 1))
        self.assertTrue(
            np.array_equal(x1.collect(), np.array([8, 17, 26])))

        x1 = ds.apply_along_axis(_sum_and_mult, 1, x, b=2)
        self.assertTrue(x1.shape, (3, 1))
        self.assertTrue(x1._reg_shape, (2, 1))
        self.assertTrue(
            np.array_equal(x1.collect(), np.array([12, 30, 48])))

        x1 = ds.apply_along_axis(_sum_and_mult, 1, x, 1, b=2)
        self.assertTrue(x1.shape, (3, 1))
        self.assertTrue(x1._reg_shape, (2, 1))
        self.assertTrue(
            np.array_equal(x1.collect(), np.array([14, 32, 50])))

    @parameterized.expand([(ds.array([[1, 2, 3],
                                      [4, 5, 6],
                                      [7, 8, 9]], (2, 2)),),
                           (ds.array(sp.csr_matrix([[1, 2, 3],
                                                    [4, 5, 6],
                                                    [7, 8, 9]]), (2, 2)),)])
    def test_array_functions(self, x):
        """ Tests various array functions """
        min = np.array([1, 2, 3])
        max = np.array([7, 8, 9])
        mean = np.array([4., 5., 6.])
        sum = np.array([12, 15, 18])

        self.assertTrue(_equal_arrays(x.min().collect(), min))
        self.assertTrue(_equal_arrays(x.max().collect(), max))
        self.assertTrue(_equal_arrays(x.mean().collect(), mean))
        self.assertTrue(_equal_arrays(x.sum().collect(), sum))
