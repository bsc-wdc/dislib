import unittest
from math import ceil

import numpy as np
from parameterized import parameterized
from scipy import sparse as sp
from sklearn.datasets import load_svmlight_file

import dislib as ds


def _sum_and_mult(arr, a=0, axis=0, b=1):
    return (np.sum(arr, axis=axis) + a) * b


def _check_array_shapes(x):
    x.collect()
    tl = x._blocks[0][0].shape
    br = x._blocks[-1][-1].shape

    # single element arrays might contain only the value and not a NumPy
    # array (and thus there is no shape)
    if not tl:
        tl = (1, 1)
    if not br:
        br = (1, 1)

    br0 = x.shape[0] - (x._reg_shape[0] *
                        max(x._n_blocks[0] - 2, 0)
                        + x._top_left_shape[0])
    br1 = x.shape[1] - (x._reg_shape[1] *
                        max(x._n_blocks[1] - 2, 0)
                        + x._top_left_shape[1])

    br0 = br0 if br0 > 0 else x._top_left_shape[0]
    br1 = br1 if br1 > 0 else x._top_left_shape[1]

    return tl == x._top_left_shape and br == (br0, br1)


def _equal_arrays(x1, x2):
    if sp.issparse(x1):
        return np.allclose(x1.toarray(), x2.toarray())
    else:
        return np.allclose(x1, x2)


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

        csv_f = "tests/files/csv/4"
        data = ds.load_txt_file(csv_f, block_size=(1, 2))
        csv = np.loadtxt(csv_f, delimiter=",")

        self.assertTrue(_equal_arrays(data.collect(), csv))

    def test_load_npy_file(self):
        """ Tests loading an npy file """
        path = "tests/files/npy/1.npy"

        x = ds.load_npy_file(path, block_size=(3, 9))
        x_np = np.load(path)

        self.assertTrue(_check_array_shapes(x))
        self.assertTrue(np.array_equal(x.collect(), x_np))

        with self.assertRaises(ValueError):
            ds.load_npy_file(path, block_size=(1000, 1000))

        with self.assertRaises(ValueError):
            ds.load_npy_file("tests/files/npy/3d.npy", block_size=(3, 3))


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
            computed = h_block
            expected = x_np[i * n_rows: (i + 1) * n_rows]
            self.assertTrue(_check_array_shapes(computed))
            self.assertTrue(_equal_arrays(computed.collect(), expected))

    @parameterized.expand([_gen_random_arrays("dense"),
                           _gen_random_arrays("sparse")])
    def test_iterate_cols(self, x, x_np):
        """ Testing the row _iterator of the ds.array """
        n_cols = x._reg_shape[1]

        for i, v_block in enumerate(x._iterator(axis='columns')):
            expected = x_np[:, i * n_cols: (i + 1) * n_cols]
            self.assertTrue(_check_array_shapes(v_block))
            self.assertTrue(_equal_arrays(v_block.collect().reshape(
                v_block.shape), expected))

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
                           _gen_random_arrays("dense", (33, 34), (2, 33)),
                           _gen_random_arrays("sparse"),
                           _gen_irregular_arrays("dense"),
                           _gen_irregular_arrays("sparse")])
    def test_indexing(self, x, x_np):
        """ Tests indexing """

        # Single row
        rows = np.random.randint(0, x.shape[0] - 1, size=min(3, x.shape[0]))

        for row in rows:
            ours = x[int(row)]
            expected = x_np[row]
            self.assertTrue(_check_array_shapes(ours))
            self.assertTrue(_equal_arrays(ours.collect(), expected))

        # Single element
        rows = np.random.randint(0, x.shape[0] - 1, size=min(10, x.shape[0]))
        cols = np.random.randint(0, x.shape[1] - 1, size=min(10, x.shape[1]))

        for i in rows:
            for j in cols:
                element = x[int(i), int(j)]
                self.assertTrue(_check_array_shapes(element))
                self.assertEqual(element.collect(), x_np[int(i), int(j)])

        # Set of rows / columns
        frm = np.random.randint(0, x.shape[0] - 5, size=min(3, x.shape[0]))
        to = frm + 4

        for i, j in zip(frm, to):
            ours = x[int(i):int(j)]
            expected = x_np[i:j]
            self.assertTrue(_check_array_shapes(ours))
            self.assertTrue(_equal_arrays(ours.collect(), expected))

        frm = np.random.randint(0, x.shape[1] - 5, size=min(3, x.shape[1]))
        to = frm + 4

        for i, j in zip(frm, to):
            ours = x[:, int(i):int(j)]
            expected = x_np[:, i:j]
            self.assertTrue(_check_array_shapes(ours))
            self.assertTrue(_equal_arrays(ours.collect(), expected))

        # Set of elements
        i = int(np.random.randint(0, x.shape[0] - 5, size=1))
        j = int(np.random.randint(0, x.shape[1] - 5, size=1))

        ours = x[i:i + 1, j:j + 1]
        expected = x_np[i:i + 1, j:j + 1]
        self.assertTrue(_check_array_shapes(ours))
        self.assertTrue(_equal_arrays(ours.collect(), expected))

        ours = x[i:i + 100, j:j + 100]
        expected = x_np[i:i + 100, j:j + 100]
        self.assertTrue(_check_array_shapes(ours))
        self.assertTrue(_equal_arrays(ours.collect(), expected))

        ours = x[i:i + 4, j:j + 4]
        expected = x_np[i:i + 4, j:j + 4]
        self.assertTrue(_check_array_shapes(ours))
        self.assertTrue(_equal_arrays(ours.collect(), expected))

    @parameterized.expand([_gen_random_arrays("dense"),
                           _gen_random_arrays("sparse"),
                           _gen_irregular_arrays("dense"),
                           _gen_irregular_arrays("sparse"),
                           _gen_irregular_arrays("sparse", (98, 10), (85, 2)) +
                           (None, [0, 1, 2, 5]),
                           _gen_irregular_arrays("sparse", (10, 98), (2, 85)) +
                           ([0, 1, 2, 5], None),
                           _gen_irregular_arrays("dense", (22, 49), (3, 1)) +
                           (None, [18, 20, 41, 44]),
                           _gen_irregular_arrays("dense", (49, 22), (1, 3)) +
                           ([18, 20, 41, 44], None),
                           _gen_random_arrays("dense", (5, 4), (3, 3)) +
                           ([0, 1, 3, 4], None),
                           _gen_random_arrays("dense", (4, 5), (3, 3)) +
                           (None, [0, 1, 3, 4])])
    def test_fancy_indexing(self, x, x_np, rows=None, cols=None):
        """ Tests fancy indexing """

        # Non-consecutive rows / cols
        if not rows:
            rows = np.random.randint(0, x.shape[0] - 1, min(5, x.shape[0]))
            rows = np.unique(sorted(rows))

        ours = x[rows]
        expected = x_np[rows]
        self.assertTrue(_check_array_shapes(ours))
        self.assertTrue(_equal_arrays(ours.collect(), expected))

        if not cols:
            cols = np.random.randint(0, x.shape[1] - 1, min(5, x.shape[1]))
            cols = np.unique(sorted(cols))

        ours = x[:, cols]
        expected = x_np[:, cols]
        self.assertTrue(_check_array_shapes(ours))
        self.assertTrue(_equal_arrays(ours.collect(), expected))

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
        self.assertTrue(
            _equal_arrays(x_t.collect().reshape(x_t.shape), x_np_t))
        self.assertEqual((b1, b0), x_t._n_blocks)
        self.assertTrue(_check_array_shapes(x_t))

        x_t = x.transpose(mode="rows")
        self.assertTrue(
            _equal_arrays(x_t.collect().reshape(x_t.shape), x_np_t))
        self.assertEqual((b1, b0), x_t._n_blocks)
        self.assertTrue(_check_array_shapes(x_t))

        x_t = x.transpose(mode="columns")
        self.assertTrue(
            _equal_arrays(x_t.collect().reshape(x_t.shape), x_np_t))
        self.assertEqual((b1, b0), x_t._n_blocks)
        self.assertTrue(_check_array_shapes(x_t))

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
        self.assertTrue(_check_array_shapes(arr1))

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
        self.assertTrue(_check_array_shapes(x1))

        x1 = ds.apply_along_axis(_sum_and_mult, 1, x)
        self.assertTrue(x1.shape, (3, 1))
        self.assertTrue(x1._reg_shape, (2, 1))
        self.assertTrue(
            np.array_equal(x1.collect(), np.array([6, 15, 24])))
        self.assertTrue(_check_array_shapes(x1))

        x1 = ds.apply_along_axis(_sum_and_mult, 1, x, 2)
        self.assertTrue(x1.shape, (3, 1))
        self.assertTrue(x1._reg_shape, (2, 1))
        self.assertTrue(
            np.array_equal(x1.collect(), np.array([8, 17, 26])))
        self.assertTrue(_check_array_shapes(x1))

        x1 = ds.apply_along_axis(_sum_and_mult, 1, x, b=2)
        self.assertTrue(x1.shape, (3, 1))
        self.assertTrue(x1._reg_shape, (2, 1))
        self.assertTrue(
            np.array_equal(x1.collect(), np.array([12, 30, 48])))
        self.assertTrue(_check_array_shapes(x1))

        x1 = ds.apply_along_axis(_sum_and_mult, 1, x, 1, b=2)
        self.assertTrue(x1.shape, (3, 1))
        self.assertTrue(x1._reg_shape, (2, 1))
        self.assertTrue(
            np.array_equal(x1.collect(), np.array([14, 32, 50])))
        self.assertTrue(_check_array_shapes(x1))

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

    @parameterized.expand([(ds.random_array((20, 30), (5, 6)),
                            ds.random_array((30, 10), (6, 2))),

                           (ds.random_array((1, 10), (1, 5)),
                            ds.random_array((10, 7), (5, 2))),

                           (ds.random_array((5, 10), (2, 2)),
                            ds.random_array((10, 1), (2, 1))),

                           (ds.random_array((17, 13), (3, 3)),
                            ds.random_array((13, 9), (3, 2))),

                           (ds.array(sp.csr_matrix(np.random.random((10, 12))),
                                     (5, 2)),
                            ds.array(sp.csr_matrix(np.random.random((12, 3))),
                                     (2, 1))),
                           (ds.random_array((1, 30), (1, 7)),
                            ds.random_array((30, 1), (7, 1)))])
    def test_matmul(self, x1, x2):
        """ Tests ds-array multiplication """
        expected = x1.collect() @ x2.collect()
        computed = x1 @ x2
        self.assertTrue(_equal_arrays(expected, computed.collect()))

    def test_matmul_error(self):
        """ Tests matmul not implemented cases """

        with self.assertRaises(ValueError):
            x1 = ds.random_array((5, 3), (5, 3))
            x2 = ds.random_array((5, 3), (5, 3))
            x1 @ x2

        with self.assertRaises(ValueError):
            x1 = ds.random_array((5, 3), (5, 3))
            x2 = ds.random_array((3, 5), (2, 5))
            x1 @ x2

        with self.assertRaises(ValueError):
            x1 = ds.array([[1, 2, 3], [4, 5, 6]], (2, 3))
            x2 = ds.array(sp.csr_matrix([[1, 2], [4, 5], [7, 6]]), (3, 2))
            x1 @ x2
