import unittest
import os
import shutil

import numpy as np
from parameterized import parameterized
from scipy import sparse as sp
from sklearn.datasets import load_svmlight_file
import pandas as pd
import dislib as ds
from math import ceil
from tests import BaseTimedTestCase


def _sum_and_mult(arr, a=0, axis=0, b=1):
    return (np.sum(arr, axis=axis) + a) * b


def _validate_array(x):
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

    return (tl == x._top_left_shape and br == (br0, br1) and
            sp.issparse(x._blocks[0][0]) == x._sparse)


def _equal_arrays(x1, x2):
    if sp.issparse(x1):
        x1 = x1.toarray()

    if sp.issparse(x2):
        x2 = x2.toarray()

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


class DataLoadingTest(BaseTimedTestCase):

    @parameterized.expand([(_gen_random_arrays("dense", (6, 10), (4, 3))
                            + ((6, 10), (4, 3))),
                           (_gen_random_arrays("sparse", (6, 10), (4, 3))
                            + ((6, 10), (4, 3)))])
    def test_array_constructor(self, x, x_np, shape, block_size):
        """ Tests array constructor """
        n, m = shape
        bn, bm = block_size

        self.assertTrue(x._n_blocks, ceil(n / bn) == ceil(m / bm))
        self.assertTrue(_equal_arrays(x.collect(), x_np))

    def test_array_creation(self):
        """ Tests array creation """
        data = [[1, 2, 3], [4, 5, 6]]

        x_np = np.array(data)
        x = ds.array(data, (2, 3))
        self.assertTrue(_validate_array(x))
        self.assertTrue(_equal_arrays(x.collect(), x_np))

        x = ds.array(x_np, (2, 3))
        self.assertTrue(_validate_array(x))
        self.assertTrue(_equal_arrays(x.collect(), x_np))

        x_np = np.random.random(10)
        x = ds.array(x_np, (1, 5))
        self.assertTrue(_validate_array(x))
        self.assertTrue(_equal_arrays(x.collect(), x_np))

        x_np = np.random.random(10)
        x = ds.array(x_np, (5, 1))
        self.assertTrue(_validate_array(x))
        self.assertTrue(_equal_arrays(x.collect(), x_np))

        with self.assertRaises(ValueError):
            x_np = np.random.random(10)
            ds.array(x_np, (5, 5))

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
        self.assertTrue(_validate_array(arr1))

        arr2 = ds.random_array((93, 177), (43, 31), random_state=88)
        arr3 = ds.random_array((93, 177), (43, 31), random_state=666)

        arr4 = ds.random_array((193, 77), (21, 51))
        arr5 = ds.random_array((193, 77), (21, 51))

        self.assertTrue(np.array_equal(arr1.collect(), arr2.collect()))
        self.assertFalse(np.array_equal(arr1.collect(), arr3.collect()))
        self.assertFalse(np.array_equal(arr4.collect(), arr5.collect()))

    def test_full(self):
        """ Tests full functions """
        x = ds.zeros((10, 10), (3, 7), dtype=int)
        x_np = np.zeros((10, 10), dtype=int)
        self.assertTrue(_validate_array(x))
        self.assertTrue(_equal_arrays(x.collect(), x_np))

        x = ds.full((11, 11), (3, 5), 15, dtype=float)
        x_np = np.full((11, 11), 15, dtype=float)
        self.assertTrue(_validate_array(x))
        self.assertTrue(_equal_arrays(x.collect(), x_np))

    @parameterized.expand([(2, ), (3, ), (5, ), (8, ), (13, ), (21, ), (44, )])
    def test_identity(self, n):
        """ Tests identity function """
        x = ds.identity(n, (2, 2))
        x_np = np.identity(n)
        self.assertTrue(_validate_array(x))
        self.assertTrue(_equal_arrays(x.collect(), x_np))

    @parameterized.expand([(2, 3), (3, 2), (5, 3), (3, 5), (8, 5),
                           (5, 8), (13, 8), (8, 13), (21, 13),
                           (13, 21), (44, 21), (21, 44)])
    def test_eye(self, n, m):
        """ Tests eye function """
        x = ds.eye(n, m, (2, 2))
        x_np = np.eye(n, m)
        self.assertTrue(_validate_array(x))
        self.assertTrue(_equal_arrays(x.collect(), x_np))

    def test_eye_exceptions(self):
        """ Tests eye function exceptions """

        with self.assertRaises(ValueError):
            ds.eye(10, 20, (20, 10))

        with self.assertRaises(ValueError):
            ds.eye(20, 10, (10, 20))

    def test_load_svmlight_file(self):
        """ Tests loading a LibSVM file  """
        file_ = "tests/datasets/libsvm/1"

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
        csv_f = "tests/datasets/csv/1"

        data = ds.load_txt_file(csv_f, block_size=(300, 50))
        csv = np.loadtxt(csv_f, delimiter=",")

        self.assertEqual(data._top_left_shape, (300, 50))
        self.assertEqual(data._reg_shape, (300, 50))
        self.assertEqual(data.shape, (4235, 122))
        self.assertEqual(data._n_blocks, (15, 3))

        self.assertTrue(np.array_equal(data.collect(), csv))

        csv_f = "tests/datasets/other/4"
        data = ds.load_txt_file(csv_f, block_size=(1000, 122), delimiter=" ")
        csv = np.loadtxt(csv_f, delimiter=" ")

        self.assertTrue(np.array_equal(data.collect(), csv))

        csv_f = "tests/datasets/csv/4"
        data = ds.load_txt_file(csv_f, block_size=(1, 2))
        csv = np.loadtxt(csv_f, delimiter=",")

        self.assertTrue(_equal_arrays(data.collect(), csv))

    def test_load_csv_file_without_first_row_and_col(self):
        """ Tests loading a CSV file removing the first row and column. """
        csv_f = "tests/datasets/csv/iris_csv.csv"

        data = ds.load_txt_file(csv_f, discard_first_row=True,
                                block_size=(20, 5))
        csv = pd.read_csv(csv_f, delimiter=",")
        feature_cols = ["sepallength", "sepalwidth", "petallength",
                        "petalwidth", "class"]
        csv_x = csv[feature_cols]
        csv_x = csv_x.values
        self.assertEqual(data._top_left_shape, (20, 5))
        self.assertEqual(data._reg_shape, (20, 5))
        self.assertEqual(data.shape, (150, 5))
        self.assertEqual(data._n_blocks, (8, 1))
        self.assertTrue(np.array_equal(data.collect(), csv_x))

        csv_f = "tests/datasets/csv/iris_csv.csv"
        data = ds.load_txt_file(csv_f, discard_first_row=True,
                                col_of_index=True, block_size=(20, 4))
        csv = pd.read_csv(csv_f, delimiter=",")

        self.assertEqual(data._top_left_shape, (20, 4))
        self.assertEqual(data._reg_shape, (20, 4))
        self.assertEqual(data.shape, (150, 4))

        self.assertFalse(np.array_equal(data.collect(), csv[:][1:]))

    def test_load_npy_file(self):
        """ Tests loading an npy file """
        path = "tests/datasets/npy/1.npy"

        x = ds.load_npy_file(path, block_size=(3, 9))
        x_np = np.load(path)

        self.assertTrue(_validate_array(x))
        self.assertTrue(np.array_equal(x.collect(), x_np))

        with self.assertRaises(ValueError):
            ds.load_npy_file(path, block_size=(1000, 1000))

        with self.assertRaises(ValueError):
            ds.load_npy_file("tests/datasets/npy/3d.npy", block_size=(3, 3))

    def test_load_mdcrd_file(self):
        """ Tests loading an mdcrd file """
        path = "tests/datasets/traj10samples_12atoms.mdcrd"

        x = ds.load_mdcrd_file(path, block_size=(4, 5), n_atoms=12)
        self.assertTrue(_validate_array(x))

        x = ds.load_mdcrd_file(path, block_size=(5, 4), n_atoms=12, copy=True)
        self.assertTrue(_validate_array(x))


class LoadBlocksRechunkTest(BaseTimedTestCase):
    def test_rechunk_new_block_size_exception(self):
        """ Tests that load_blocks_rechunk function throws an exception
        when the block_size returned of the rechunk is greater than the shape
        of the array."""
        array = ds.random_array((20, 20), (2, 2))
        blocks = []
        for block in array._blocks:
            for block_block in block:
                blocks.append(block_block)
        with self.assertRaises(ValueError):
            ds.data.load_blocks_rechunk(blocks, (20, 20), (2, 2), (21, 21))

    def test_rechunk(self):
        """ Tests load_blocks_rechunk function """
        array = ds.random_array((20, 20), (2, 2))
        array_aux = array.copy()
        blocks = []
        for block in array._blocks:
            for block_block in block:
                blocks.append(block_block)
        x1 = ds.data.load_blocks_rechunk(blocks, (20, 20), (2, 2), (10, 10))
        array_collected = array_aux.collect()
        self.assertTrue(_equal_arrays(x1.collect(), array_collected))
        array = ds.random_array((20, 20), (2, 2))
        array_aux = array.copy()
        blocks = []
        for block in array._blocks:
            for block_block in block:
                blocks.append(block_block)
        x2 = ds.data.load_blocks_rechunk(blocks, (40, 10), (2, 2), (10, 10))
        x3 = x2.collect()
        array_collected = array_aux.collect()
        self.assertTrue(_equal_arrays(x3[0:2], array_collected[0:2, 0:10]))
        self.assertTrue(_equal_arrays(x3[2:4],
                                      array_collected[0:2, 10:20]))
        self.assertTrue(_equal_arrays(x3[10:12],
                                      array_collected[4:6, 10:20]))
        self.assertTrue(_equal_arrays(x3[38:], array_collected[18:20, 10:20]))

    def test_rechunk_block_size_exception(self):
        """ Tests that load_blocks_rechunk throws an exception when
        the block_size specified is greater than the real block size"""
        array = ds.random_array((20, 20), (2, 2))
        blocks = []
        for block in array._blocks:
            for block_block in block:
                blocks.append(block_block)
        with self.assertRaises(ValueError):
            ds.data.load_blocks_rechunk(blocks, (20, 20), (25, 25), (5, 5))


class LoadHStackNpyFilesTest(BaseTimedTestCase):
    folder = 'load_hstack_npy_files_test_folder'
    arrays = [np.random.rand(3, 4) for _ in range(5)]

    def setUp(self):
        os.mkdir(self.folder)
        for i, arr in enumerate(self.arrays):
            np.save(os.path.join(self.folder, str(i)), arr)

    def tearDown(self):
        shutil.rmtree(self.folder)

    def test_load_hstack_npy_files(self):
        """ Tests load_hstack_npy_files """
        x = ds.data.load_hstack_npy_files(self.folder)
        self.assertTrue(_validate_array(x))
        self.assertTrue(np.allclose(x.collect(), np.hstack(self.arrays)))

    def test_load_hstack_npy_files_2(self):
        """ Tests load_hstack_npy_files with cols_per_block parameter"""
        x = ds.data.load_hstack_npy_files(self.folder, cols_per_block=9)
        self.assertTrue(_validate_array(x))
        self.assertTrue(np.allclose(x.collect(), np.hstack(self.arrays)))


class SaveTxtTest(BaseTimedTestCase):
    folder = 'save_txt_test_folder'

    def tearDown(self):
        shutil.rmtree(self.folder)

    @parameterized.expand([_gen_random_arrays('dense'),
                           _gen_irregular_arrays('dense')])
    def test_save_txt(self, x, x_np):
        """Tests saving chunk by chunk into a folder"""
        folder = self.folder
        ds.data.save_txt(x, folder)
        blocks = []
        for i in range(x._n_blocks[0]):
            blocks.append([])
            for j in range(x._n_blocks[1]):
                fname = '{}_{}'.format(i, j)
                path = os.path.join(folder, fname)
                blocks[-1].append(np.loadtxt(path, ndmin=2))

        self.assertTrue(_equal_arrays(np.block(blocks), x_np))

    @parameterized.expand([_gen_random_arrays('dense'),
                           _gen_irregular_arrays('dense')])
    def test_save_txt_merge_rows(self, x, x_np):
        """Tests saving chunk by chunk into a folder"""
        folder = self.folder
        ds.data.save_txt(x, folder, merge_rows=True)
        h_blocks = []
        for i in range(x._n_blocks[0]):
            path = os.path.join(folder, str(i))
            h_blocks.append(np.loadtxt(path))

        self.assertTrue(_equal_arrays(np.vstack(h_blocks), x_np))


class SaveNpyTest(BaseTimedTestCase):
    folder = 'save_npy_test_folder'

    def tearDown(self):
        shutil.rmtree(self.folder)

    @parameterized.expand([_gen_random_arrays('dense'),
                           _gen_irregular_arrays('dense')])
    def test_save_npy(self, x, x_np):
        """Tests saving chunk by chunk into a folder"""
        ds.data.save_npy_file(x, 'save_npy_test_folder')
        blocks = []
        for i in range(x._n_blocks[0]):
            blocks.append([])
            for j in range(x._n_blocks[1]):
                fname = '{}_{}.npy'.format(i, j)
                path = os.path.join('save_npy_test_folder', fname)
                blocks[-1].append(np.load(path))

        self.assertTrue(_equal_arrays(np.block(blocks), x_np))

    @parameterized.expand([_gen_random_arrays('dense'),
                           _gen_irregular_arrays('dense')])
    def test_save_npy_merge_rows(self, x, x_np):
        """Tests saving chunk by chunk into a folder"""
        ds.data.save_npy_file(x, 'save_npy_test_folder', merge_rows=True)
        h_blocks = []
        for i in range(x._n_blocks[0]):
            path = os.path.join('save_npy_test_folder', str(i)+'.npy')
            h_blocks.append(np.load(path))

        self.assertTrue(_equal_arrays(np.vstack(h_blocks), x_np))

    def test_load_npy_files(self):
        """Tests loading chunk by chunk into a folder"""
        array = ds.random_array((7, 2), (2, 2))
        array.collect()
        ds.data.save_npy_file(array, 'save_npy_test_folder')
        loaded_array = ds.data.load_npy_files('save_npy_test_folder',
                                              shape=(7, 2))
        self.assertTrue(_equal_arrays(loaded_array.collect(), array.collect()))

        with self.assertRaises(ValueError):
            ds.data.load_npy_files('save_npy_test_folder')


class ArrayTest(BaseTimedTestCase):

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
            self.assertTrue(_validate_array(computed))
            self.assertTrue(_equal_arrays(computed.collect(), expected))

    @parameterized.expand([_gen_random_arrays("dense"),
                           _gen_random_arrays("sparse")])
    def test_iterate_cols(self, x, x_np):
        """ Testing the row _iterator of the ds.array """
        n_cols = x._reg_shape[1]

        for i, v_block in enumerate(x._iterator(axis='columns')):
            expected = x_np[:, i * n_cols: (i + 1) * n_cols]
            self.assertTrue(_validate_array(v_block))
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
        with self.assertRaises(NotImplementedError):
            x[-3:-1, 2]

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
            self.assertTrue(_validate_array(ours))
            self.assertTrue(_equal_arrays(ours.collect(), expected))

        # Single element
        rows = np.random.randint(0, x.shape[0] - 1, size=min(10, x.shape[0]))
        cols = np.random.randint(0, x.shape[1] - 1, size=min(10, x.shape[1]))

        for i in rows:
            for j in cols:
                element = x[int(i), int(j)]
                self.assertTrue(_validate_array(element))
                self.assertEqual(element.collect(), x_np[int(i), int(j)])

        # Set of rows / columns
        frm = np.random.randint(0, x.shape[0] - 5, size=min(3, x.shape[0]))
        to = frm + 4

        for i, j in zip(frm, to):
            ours = x[int(i):int(j)]
            expected = x_np[i:j]
            self.assertTrue(_validate_array(ours))
            self.assertTrue(_equal_arrays(ours.collect(), expected))

        frm = np.random.randint(0, x.shape[1] - 5, size=min(3, x.shape[1]))
        to = frm + 4

        for i, j in zip(frm, to):
            ours = x[:, int(i):int(j)]
            expected = x_np[:, i:j]
            self.assertTrue(_validate_array(ours))
            self.assertTrue(_equal_arrays(ours.collect(), expected))

        # Set of elements
        i = int(np.random.randint(0, x.shape[0] - 5, size=1))
        j = int(np.random.randint(0, x.shape[1] - 5, size=1))

        ours = x[i:i + 1, j:j + 1]
        expected = x_np[i:i + 1, j:j + 1]
        self.assertTrue(_validate_array(ours))
        self.assertTrue(_equal_arrays(ours.collect(), expected))

        ours = x[i:i + 100, j:j + 100]
        expected = x_np[i:i + 100, j:j + 100]
        self.assertTrue(_validate_array(ours))
        self.assertTrue(_equal_arrays(ours.collect(), expected))

        ours = x[i:i + 4, j:j + 4]
        expected = x_np[i:i + 4, j:j + 4]
        self.assertTrue(_validate_array(ours))
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

        np.random.seed(1234)

        # Non-consecutive rows / cols
        if not rows:
            rows = np.random.randint(0, x.shape[0] - 1, min(5, x.shape[0]))
            rows = np.unique(sorted(rows))

        ours = x[rows]
        expected = x_np[rows]
        self.assertTrue(_validate_array(ours))
        self.assertTrue(_equal_arrays(ours.collect(), expected))

        if not cols:
            cols = np.random.randint(0, x.shape[1] - 1, min(5, x.shape[1]))
            cols = np.unique(sorted(cols))

        ours = x[:, cols]
        expected = x_np[:, cols]
        self.assertTrue(_validate_array(ours))
        self.assertTrue(_equal_arrays(ours.collect(), expected))

    @parameterized.expand([("dense", (50, 60), None, (0, 13), 20),
                           ("sparse", (60, 50), None, (3, 20), 10),
                           ("dense", (98, 10), None, (0, 98), 1),
                           ("sparse", (98, 10), None, (50, 98), 9),
                           ("sparse", (98, 10), (85, 2), (7, 70), 3),
                           ("sparse", (10, 98), (2, 85), (2, 5), 90),
                           ("dense", (22, 49), (3, 1), (0, 5), 48),
                           ("dense", (49, 22), (1, 3), (20, 30), 2),
                           ("dense", (5, 4), (3, 3), (0, 5), 3),
                           ("dense", (4, 5), (3, 3), (3, 4), 4)])
    def test_set_column(self, fmt, arr_shape, block_size, rows_slice, column):
        """ Tests setting a column of values """
        np.random.seed(1234)
        x, x_np = _gen_random_arrays(fmt, arr_shape, block_size)

        gen_vector1 = np.random.random(rows_slice[1] - rows_slice[0])
        x1 = x.copy()
        x_np1 = x_np.copy()
        x1[rows_slice[0]:rows_slice[1], column] = gen_vector1

        # scipy before 1.5.4 requires vectors (n, 1)
        # while numpy accepts only (n,)
        if fmt == "sparse":
            x_np1[rows_slice[0]:rows_slice[1], column] = gen_vector1.reshape(
                (gen_vector1.shape[0], 1)
            )
        else:
            x_np1[rows_slice[0]:rows_slice[1], column] = gen_vector1

        self.assertTrue(_validate_array(x1))
        self.assertTrue(_equal_arrays(x1.collect(), x_np1))

        gen_vector2 = np.random.random(x.shape[0])
        x2 = x.copy()
        x_np2 = x_np.copy()
        x2[:, column] = gen_vector2

        # scipy before 1.5.4 requires vectors (n, 1)
        # while numpy accepts only (n,)
        if fmt == "sparse":
            x_np2[:, column] = gen_vector2.reshape((gen_vector2.shape[0], 1))
        else:
            x_np2[:, column] = gen_vector2

        self.assertTrue(_validate_array(x2))
        self.assertTrue(_equal_arrays(x2.collect(), x_np2))

    def test_set_column_exceptions(self):
        """ Tests raising proper exceptions while setting a column """
        np.random.seed(1234)
        x, _ = _gen_random_arrays("dense", (100, 100), (10, 10))

        with self.assertRaises(NotImplementedError):
            # slicing for multiple columns is not implemented
            x[:, :] = x

        with self.assertRaises(IndexError):
            # different shapes
            x[:, 10] = np.random.random(5)

        with self.assertRaises(IndexError):
            # second dimension is not of size 1
            x[0:5, 10] = np.random.random((5, 5))

        with self.assertRaises(IndexError):
            # too many dimensions
            x[0:5, 10] = np.random.random((5, 5, 5))

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
        self.assertTrue(_validate_array(x_t))

        x_t = x.T
        self.assertTrue(
            _equal_arrays(x_t.collect().reshape(x_t.shape), x_np_t))
        self.assertEqual((b1, b0), x_t._n_blocks)
        self.assertTrue(_validate_array(x_t))

        x_t = x.transpose(mode="columns")
        self.assertTrue(
            _equal_arrays(x_t.collect().reshape(x_t.shape), x_np_t))
        self.assertEqual((b1, b0), x_t._n_blocks)
        self.assertTrue(_validate_array(x_t))

        with self.assertRaises(Exception):
            x.transpose(mode="invalid")

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
        self.assertTrue(_equal_arrays(x1.collect(), np.array([12, 15, 18])))
        self.assertTrue(_validate_array(x1))

        x1 = ds.apply_along_axis(_sum_and_mult, 1, x)
        self.assertTrue(x1.shape, (3, 1))
        self.assertTrue(x1._reg_shape, (2, 1))
        self.assertTrue(_equal_arrays(x1.collect(False),
                                      np.array([[6], [15], [24]])))
        self.assertTrue(_validate_array(x1))

        x1 = ds.apply_along_axis(_sum_and_mult, 1, x, 2)
        self.assertTrue(x1.shape, (3, 1))
        self.assertTrue(x1._reg_shape, (2, 1))
        self.assertTrue(_equal_arrays(x1.collect(False),
                                      np.array([[8], [17], [26]])))
        self.assertTrue(_validate_array(x1))

        x1 = ds.apply_along_axis(_sum_and_mult, 1, x, b=2)
        self.assertTrue(x1.shape, (3, 1))
        self.assertTrue(x1._reg_shape, (2, 1))
        self.assertTrue(_equal_arrays(x1.collect(False),
                                      np.array([[12], [30], [48]])))
        self.assertTrue(_validate_array(x1))

        x1 = ds.apply_along_axis(_sum_and_mult, 1, x, 1, b=2)
        self.assertTrue(x1.shape, (3, 1))
        self.assertTrue(x1._reg_shape, (2, 1))
        self.assertTrue(_equal_arrays(x1.collect(False),
                                      np.array([[14], [32], [50]])))
        self.assertTrue(_validate_array(x1))

    @parameterized.expand([(ds.array([[1, 2, 3],
                                      [4, 5, 6],
                                      [7, 8, 9]], (2, 2)),),
                           (ds.array(sp.csr_matrix([[1, 2, 3],
                                                    [4, 5, 6],
                                                    [7, 8, 9]]), (2, 2)),)])
    def test_array_functions(self, x):
        """ Tests various array functions, applicable to both dense
        and sparse matrices """
        min = np.array([1, 2, 3])
        max = np.array([7, 8, 9])
        mean = np.array([4., 5., 6.])
        sum = np.array([12, 15, 18])

        self.assertTrue(_equal_arrays(x.min().collect(), min))
        self.assertTrue(_equal_arrays(x.max().collect(), max))
        self.assertTrue(_equal_arrays(x.mean().collect(), mean))
        self.assertTrue(_equal_arrays(x.sum().collect(), sum))

    @parameterized.expand([(np.full((10, 10), 3, complex),),
                           (sp.csr_matrix(np.full((10, 10), 5, complex)),),
                           (np.random.rand(10, 10) +
                            1j * np.random.rand(10, 10),)])
    def test_conj(self, x_np):
        """ Tests the complex conjugate """
        bs0 = np.random.randint(1, x_np.shape[0] + 1)
        bs1 = np.random.randint(1, x_np.shape[1] + 1)

        x = ds.array(x_np, (bs0, bs1))
        self.assertTrue(_equal_arrays(x.conj().collect(), x_np.conj()))

    @parameterized.expand([(ds.array([[1, 2, 3], [4, 5, 6],
                                      [7, 8, 9], [1, 2, 3],
                                      [4, 5, 6]], (5, 3)),
                            ds.array([[1, 1, 1], [1, 1, 1],
                                      [1, 1, 1], [1, 1, 1],
                                      [1, 1, 1]], (5, 3)),
                            np.array([[0, 1, 2], [3, 4, 5],
                                      [6, 7, 8], [0, 1, 2],
                                      [3, 4, 5]]),
                            ),
                           (ds.array([[1, 2, 3], [4, 5, 6],
                                      [7, 8, 9], [1, 2, 3],
                                      [4, 5, 6]], (5, 3)),
                            ds.array([[2, 2, 2], [4, 5, 6],
                                      [9, 8, 7], [3, 2, 1],
                                      [6, 5, 4]], (5, 3)),
                            np.array([[-1, 0, 1], [0, 0, 0],
                                      [-2, 0, 2], [-2, 0, 2],
                                      [-2, 0, 2]]),),
                           (ds.array([[-1, 2, 3], [4, -5, 6]], (2, 3)),
                            ds.array([[-2, 2, -2], [4, 5, 6]], (2, 3)),
                            np.array([[1, 0, 5], [0, -10, 0]]),
                            )])
    def test_matsubtract(self, x, y, z):
        """ Tests subtraction of two ds-array """
        self.assertTrue(_equal_arrays(ds.data.matsubtract(x, y).collect(),
                                      z))

    def test_matsubtract_error(self):
        """ Tests the implementation of errors in matsubtract """

        with self.assertRaises(ValueError):
            x1 = ds.array([[1, 2, 3], [4, 5, 6]], (2, 3))
            x1.__init__([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3],
                        [4, 5, 6]], top_left_shape=(1, 3), reg_shape=(2, 3),
                        shape=(5, 3), sparse=False)
            x2 = ds.array([[1, 1, 1], [1, 1, 1], [1, 1, 1],
                           [1, 1, 1], [1, 1, 1]], (2, 3))
            ds.data.matsubtract(x1, x2)

        with self.assertRaises(ValueError):
            x1 = ds.random_array((5, 5), (2, 5))
            x2 = ds.random_array((3, 5), (2, 5))
            ds.data.matsubtract(x1, x2)

        with self.assertRaises(ValueError):
            x1 = ds.random_array((5, 5), (2, 5))
            x2 = ds.random_array((5, 5), (1, 5))
            ds.data.matsubtract(x1, x2)

    @parameterized.expand([(ds.array([[1, 2, 3], [4, 5, 6],
                                      [7, 8, 9], [1, 2, 3],
                                      [4, 5, 6]], (1, 3)),
                            ds.array([[1, 1, 1], [1, 1, 1],
                                      [1, 1, 1], [1, 1, 1],
                                      [1, 1, 1]], (1, 3)),
                            np.array([[2, 3, 4], [5, 6, 7],
                                      [8, 9, 10], [2, 3, 4],
                                      [5, 6, 7]]),
                            ),
                           (ds.array([[-1, -2, -3], [4, 5, 6],
                                      [7, 8, 9], [1, 2, 3],
                                      [4, 5, 6]], (2, 3)),
                            ds.array([[2, 2, 2], [4, 5, 6],
                                      [9, 8, 7], [3, 2, 1],
                                      [6, 5, 4]], (2, 3)),
                            np.array([[1, 0, -1], [8, 10, 12],
                                      [16, 16, 16], [4, 4, 4],
                                      [10, 10, 10]]),),
                           (ds.array([[-1, 2, 3], [4, -5, 6]], (2, 3)),
                            ds.array([[-2, 2, -2], [4, 5, 6]], (2, 3)),
                            np.array([[-3, 4, 1], [8, 0, 12]]),
                            )])
    def test_matadd(self, x, y, z):
        """ Tests addition of two ds-array """
        self.assertTrue(_equal_arrays(ds.data.matadd(x, y).collect(),
                                      z))

    def test_matadd_error(self):
        """ Tests the implementation of errors in matadd """

        with self.assertRaises(ValueError):
            x1 = ds.array([[1, 2, 3], [4, 5, 6]], (2, 3))
            x1.__init__([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3],
                        [4, 5, 6]], top_left_shape=(1, 3),
                        reg_shape=(2, 3), shape=(5, 3), sparse=False)
            x2 = ds.array([[1, 1, 1], [1, 1, 1],
                                      [1, 1, 1], [1, 1, 1],
                                      [1, 1, 1]], (2, 3))
            ds.data.matadd(x1, x2)

        with self.assertRaises(ValueError):
            x1 = ds.random_array((5, 5), (3, 5))
            x2 = ds.random_array((3, 5), (3, 5))
            ds.data.matadd(x1, x2)

        with self.assertRaises(ValueError):
            x1 = ds.random_array((5, 5), (2, 5))
            x2 = ds.random_array((5, 5), (1, 5))
            ds.data.matadd(x1, x2)

    @parameterized.expand([((ds.array([[1, 2, 3], [4, 5, 6]], (1, 3)),
                            ds.array([[1, 1, 1, 2, 3, 4], [1, 1, 1, 8, 2, 3]],
                                     (1, 3)),
                            np.array([[1, 2, 3, 1, 1, 1, 2, 3, 4],
                                      [4, 5, 6, 1, 1, 1, 8, 2, 3]]), )),
                           ((ds.array([[1, 2, 3, 4], [4, 5, 6, 8]], (1, 2)),
                             ds.array([[1, 1], [1, 1]], (1, 2)),
                             np.array([[1, 2, 3, 4, 1, 1],
                                       [4, 5, 6, 8, 1, 1]]), )),
                           ])
    def test_concat_columns(self, x, y, z):
        """ Tests concatenation of two ds-arrays by columns"""
        self.assertTrue(_equal_arrays(ds.data.concat_columns(x, y).
                                      collect(), z))

    def test_concat_columns_error(self):
        x1 = ds.array([[1, 2, 3], [4, 5, 6]], (1, 3))
        x2 = ds.array([[1, 1, 1, 2], [1, 1, 1, 8], [1, 1, 1, 8]], (1, 4))
        with self.assertRaises(ValueError):
            ds.data.concat_columns(x1, x2)
        x1 = ds.array([[1, 2, 3], [4, 5, 6]], (1, 3))
        x2 = ds.array([[4, 4, 4, 4], [2, 3, 4, 5]], (1, 4))
        with self.assertRaises(ValueError):
            ds.data.concat_columns(x1, x2)

    @parameterized.expand([((20, 30), (30, 10), False),
                           ((1, 10), (10, 7), False),
                           ((5, 10), (10, 1), False),
                           ((17, 13), (13, 9), False),
                           ((1, 30), (30, 1), False),
                           ((10, 1), (1, 20), False),
                           ((20, 30), (30, 10), True),
                           ((1, 10), (10, 7), True),
                           ((5, 10), (10, 1), True),
                           ((17, 13), (13, 9), True),
                           ((1, 30), (30, 1), True),
                           ((10, 1), (1, 20), True)])
    def test_matmul(self, shape_a, shape_b, sparse):
        """ Tests ds-array multiplication """
        a_np = np.random.random(shape_a)
        b_np = np.random.random(shape_b)

        if sparse:
            a_np = sp.csr_matrix(a_np)
            b_np = sp.csr_matrix(b_np)

        b0 = np.random.randint(1, a_np.shape[0] + 1)
        b1 = np.random.randint(1, a_np.shape[1] + 1)
        b2 = np.random.randint(1, b_np.shape[1] + 1)

        a = ds.array(a_np, (b0, b1))
        b = ds.array(b_np, (b1, b2))

        expected = a_np @ b_np
        computed = a @ b
        self.assertTrue(_equal_arrays(expected, computed.collect(False)))

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

    @parameterized.expand([((21, 33), (10, 15), (5, 18)),
                           ((10, 8), (2, 5), (5, 3)),
                           ((11, 12), (4, 6), (5, 12)),
                           ((9, 15), (8, 15), (1, 9)),
                           ((1, 1), (1, 1), (1, 1)),
                           ((5, 5), (2, 3), (1, 1))])
    def test_rechunk(self, shape, bsize_in, bsize_out):
        """ Tests the rechunk function """
        x = ds.random_array(shape, bsize_in)
        re = x.rechunk(bsize_out)
        self.assertEqual(re._reg_shape, bsize_out)
        self.assertEqual(re._top_left_shape, bsize_out)
        self.assertTrue(_validate_array(re))
        self.assertTrue(_equal_arrays(x.collect(), re.collect()))

    def test_rechunk_exceptions(self):
        """ Tests exceptions of the rechunk function """
        x = ds.random_array((50, 50), (10, 10))
        with self.assertRaises(ValueError):
            x.rechunk((100, 10))

        x = ds.random_array((50, 50), (10, 10))
        with self.assertRaises(ValueError):
            x.rechunk((10, 100))

    def test_set_item(self):
        """ Tests setting a single value """
        x = ds.random_array((10, 10), (3, 3))
        x[5, 5] = -1
        x[0, 0] = -2
        x[9, 9] = -3

        self.assertTrue(_validate_array(x))

        x_np = x.collect()

        self.assertEqual(x_np[5][5], -1)
        self.assertEqual(x_np[0][0], -2)
        self.assertEqual(x_np[9][9], -3)

        with self.assertRaises(ValueError):
            x[0, 0] = [2, 3, 4]

        with self.assertRaises(IndexError):
            x[10, 2] = 3

        with self.assertRaises(NotImplementedError):
            x[0] = 3

    def test_power(self):
        """ Tests ds-array power and sqrt """
        orig = np.array([[1, 2, 3], [4, 5, 6]])
        x = ds.array(orig, block_size=(2, 1))
        xp = x ** 2
        xs = xp.sqrt()

        self.assertTrue(_validate_array(xp))
        self.assertTrue(_validate_array(xs))

        expected = np.array([[1, 4, 9], [16, 25, 36]])

        self.assertTrue(_equal_arrays(expected, xp.collect()))
        self.assertTrue(_equal_arrays(orig, xs.collect()))

        orig = sp.csr_matrix([[1, 2, 3], [4, 5, 6]])
        x = ds.array(orig, block_size=(2, 1))
        xp = x ** 2
        xs = xp.sqrt()

        self.assertTrue(_validate_array(xp))
        self.assertTrue(_validate_array(xs))

        expected = sp.csr_matrix([[1, 4, 9], [16, 25, 36]])

        self.assertTrue(_equal_arrays(expected, xp.collect()))
        self.assertTrue(_equal_arrays(orig, xs.collect()))

        with self.assertRaises(NotImplementedError):
            x ** x

    def test_add(self):
        orig = np.array([[1, 2, 3], [4, 5, 6]])
        x = ds.array(orig, block_size=(2, 1))
        orig_vector = np.array([[1, 2, 3]])
        vector = ds.array(orig_vector, block_size=(1, 1))

        b = x + vector
        self.assertTrue(_validate_array(b))
        expected = np.array([[2, 4, 6], [5, 7, 9]])

        self.assertTrue(_equal_arrays(expected, b.collect()))

        orig = sp.csr_matrix([[1, 2, 3], [4, 5, 6]])
        x = ds.array(orig, block_size=(2, 1))
        orig_vector = sp.csr_matrix([[1, 2, 3]])
        vector = ds.array(orig_vector, block_size=(1, 1))

        b = x + vector
        self.assertTrue(_validate_array(b))
        expected = sp.csr_matrix([[2, 4, 6], [5, 7, 9]])

        self.assertTrue(_equal_arrays(expected, b.collect()))

        with self.assertRaises(NotImplementedError):
            orig = np.array([[1, 2, 3], [4, 5, 6]])
            x = ds.array(orig, block_size=(2, 1))
            orig_vector = np.array([[1, 2]])
            vector = ds.array(orig_vector, block_size=(1, 1))
            b = x + vector

    def test_sub(self):
        orig = np.array([[1, 2, 3], [4, 5, 6]])
        x = ds.array(orig, block_size=(2, 1))
        orig_vector = np.array([[1, 2, 3]])
        vector = ds.array(orig_vector, block_size=(1, 1))

        b = x - vector
        self.assertTrue(_validate_array(b))
        expected = np.array([[0, 0, 0], [3, 3, 3]])

        self.assertTrue(_equal_arrays(expected, b.collect()))

        with self.assertRaises(NotImplementedError):
            orig = np.array([[1, 2, 3], [4, 5, 6]])
            x = ds.array(orig, block_size=(2, 1))
            orig_vector = np.array([[1, 3]])
            vector = ds.array(orig_vector, block_size=(1, 1))
            b = x - vector

    def test_iadd(self):
        """ Tests ds-array magic method __iadd__ """
        orig = np.array([[1, 2, 3], [4, 5, 6]])
        x = ds.array(orig, block_size=(2, 1))
        orig_vector = np.array([[1, 2, 3]])
        vector = ds.array(orig_vector, block_size=(1, 1))

        x += vector

        self.assertTrue(_validate_array(x))
        self.assertTrue(_validate_array(vector))

        expected = np.array([[2, 4, 6], [5, 7, 9]])

        self.assertTrue(_equal_arrays(expected, x.collect()))
        self.assertTrue(_equal_arrays(orig_vector, vector.collect()))

        orig = np.array([[1, 2, 3], [4, 5, 6]])
        x = ds.array(orig, block_size=(2, 1))
        orig_mat = np.array([[1, 2, 3], [4, 5, 6]])
        matrix = ds.array(orig_mat, block_size=(2, 1))

        x += matrix

        self.assertTrue(_validate_array(x))
        self.assertTrue(_validate_array(matrix))

        expected = np.array([[2, 4, 6], [8, 10, 12]])

        self.assertTrue(_equal_arrays(expected, x.collect()))
        self.assertTrue(_equal_arrays(orig_mat, matrix.collect()))

        with self.assertRaises(NotImplementedError):
            orig = np.array([[1, 2, 3], [4, 5, 6]])
            x = ds.array(orig, block_size=(2, 1))
            orig_vector = np.array([[1, 2]])
            vector = ds.array(orig_vector, block_size=(1, 1))
            x += vector

        with self.assertRaises(NotImplementedError):
            orig = np.array([[1, 2, 3], [4, 5, 6]])
            x = ds.array(orig, block_size=(2, 1))
            orig_vector = np.array([[1, 2], [4, 5]])
            vector = ds.array(orig_vector, block_size=(2, 1))
            x += vector

        with self.assertRaises(ValueError):
            x1 = ds.array([[1, 2, 3], [4, 5, 6]], (2, 3))
            x1.__init__([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3],
                         [4, 5, 6]], top_left_shape=(1, 3),
                        reg_shape=(2, 3), shape=(5, 3), sparse=False)
            x2 = ds.array([[1, 1, 1], [1, 1, 1],
                           [1, 1, 1], [1, 1, 1],
                           [1, 1, 1]], (2, 3))
            x1 += x2
        with self.assertRaises(ValueError):
            orig = np.array([[1, 2, 3], [4, 5, 6]])
            x = ds.array(orig, block_size=(2, 1))
            orig_vector = np.array([[1, 2, 3], [4, 5, 6]])
            vector = ds.array(orig_vector, block_size=(2, 2))
            x += vector

    def test_isub(self):
        """ Tests ds-array magic method __isub__ """
        orig = np.array([[1, 2, 3], [4, 5, 6]])
        x = ds.array(orig, block_size=(2, 1))
        orig_vector = np.array([[1, 2, 3]])
        vector = ds.array(orig_vector, block_size=(1, 1))

        x -= vector

        self.assertTrue(_validate_array(x))
        self.assertTrue(_validate_array(vector))

        expected = np.array([[0, 0, 0], [3, 3, 3]])

        self.assertTrue(_equal_arrays(expected, x.collect()))
        self.assertTrue(_equal_arrays(orig_vector, vector.collect()))

        orig = np.array([[1, 2, 3], [4, 5, 6]])
        x = ds.array(orig, block_size=(2, 1))
        orig_mat = np.array([[1, 2, 3], [4, 5, 6]])
        matrix = ds.array(orig_mat, block_size=(2, 1))

        x -= matrix

        self.assertTrue(_validate_array(x))
        self.assertTrue(_validate_array(matrix))

        expected = np.array([[0, 0, 0], [0, 0, 0]])

        self.assertTrue(_equal_arrays(expected, x.collect()))
        self.assertTrue(_equal_arrays(orig_mat, matrix.collect()))
        with self.assertRaises(NotImplementedError):
            orig = np.array([[1, 2, 3], [4, 5, 6]])
            x = ds.array(orig, block_size=(2, 1))
            orig_vector = np.array([[1, 2]])
            vector = ds.array(orig_vector, block_size=(1, 1))
            x -= vector
        with self.assertRaises(NotImplementedError):
            orig = np.array([[1, 2, 3], [4, 5, 6]])
            x = ds.array(orig, block_size=(2, 1))
            orig_vector = np.array([[1, 2], [4, 5]])
            vector = ds.array(orig_vector, block_size=(2, 1))
            x -= vector

        with self.assertRaises(ValueError):
            x1 = ds.array([[1, 2, 3], [4, 5, 6]], (2, 3))
            x1.__init__([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3],
                         [4, 5, 6]], top_left_shape=(1, 3),
                        reg_shape=(2, 3), shape=(5, 3), sparse=False)
            x2 = ds.array([[1, 1, 1], [1, 1, 1],
                           [1, 1, 1], [1, 1, 1],
                           [1, 1, 1]], (2, 3))
            x1 -= x2
        with self.assertRaises(ValueError):
            orig = np.array([[1, 2, 3], [4, 5, 6]])
            x = ds.array(orig, block_size=(2, 1))
            orig_vector = np.array([[1, 2, 3], [4, 5, 6]])
            vector = ds.array(orig_vector, block_size=(2, 2))
            x -= vector

    def test_norm(self):
        """ Tests the norm """
        x_np = np.array([[1, 2, 3], [4, 5, 6]])
        x = ds.array(x_np, block_size=(2, 1))
        xn = x.norm()

        self.assertTrue(_validate_array(xn))

        expected = np.linalg.norm(x_np, axis=0)

        self.assertTrue(_equal_arrays(expected, xn.collect()))

        xn = x.norm(axis=1)

        self.assertTrue(_validate_array(xn))

        expected = np.linalg.norm(x_np, axis=1)

        self.assertTrue(_equal_arrays(expected, xn.collect()))

    def test_median(self):
        """ Tests the median """
        x_np = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        x = ds.array(x_np, block_size=(2, 2))
        xm = x.median()

        self.assertTrue(_validate_array(xm))

        expected = np.median(x_np, axis=0)

        self.assertTrue(_equal_arrays(expected, xm.collect()))

        xm = x.median(axis=1)

        self.assertTrue(_validate_array(xm))

        expected = np.median(x_np, axis=1)

        self.assertTrue(_equal_arrays(expected, xm.collect()))

        with self.assertRaises(NotImplementedError):
            x_csr = ds.array(sp.csr_matrix([[1, 2, 3],
                                            [4, 5, 6],
                                            [7, 8, 9]]), (2, 2))
            x_csr.median()


class MathTest(BaseTimedTestCase):

    @parameterized.expand([((21, 33), (10, 15), False),
                           ((5, 10), (8, 1), False),
                           ((17, 13), (1, 9), False),
                           ((6, 1), (12, 23), False),
                           ((1, 22), (25, 16), False),
                           ((1, 12), (1, 3), False),
                           ((14, 1), (4, 1), False),
                           ((10, 1), (1, 19), False),
                           ((1, 30), (12, 1), False)])
    def test_kron(self, shape_a, shape_b, sparse):
        """ Tests kronecker product """
        np.random.seed()

        a_np = np.random.random(shape_a)
        b_np = np.random.random(shape_b)
        expected = np.kron(a_np, b_np)

        if sparse:
            a_np = sp.csr_matrix(a_np)
            b_np = sp.csr_matrix(b_np)

        b0 = np.random.randint(1, a_np.shape[0] + 1)
        b1 = np.random.randint(1, a_np.shape[1] + 1)
        b2 = np.random.randint(1, b_np.shape[0] + 1)
        b3 = np.random.randint(1, b_np.shape[1] + 1)

        a = ds.array(a_np, (b0, b1))
        b = ds.array(b_np, (b2, b3))

        b4 = np.random.randint(1, (b0 * b2) + 1)
        b5 = np.random.randint(1, (b1 * b3) + 1)

        computed = ds.kron(a, b, (b4, b5))

        self.assertTrue(_validate_array(computed))

        computed = computed.collect(False)

        # convert to ndarray because there is no kron for sparse matrices in
        # scipy
        if a._sparse:
            computed = computed.toarray()

        self.assertTrue(_equal_arrays(expected, computed))

    @parameterized.expand([((15, 13), (3, 6), (9, 6), (3, 2)),
                           ((7, 8), (2, 3), (1, 15), (1, 15))])
    def test_kron_regular(self, a_shape, a_bsize, b_shape, b_bsize):
        """ Tests kron when blocks of b are all equal """
        a = ds.random_array(a_shape, a_bsize)
        b = ds.random_array(b_shape, b_bsize)

        computed = ds.kron(a, b)
        expected = np.kron(a.collect(), b.collect())

        self.assertTrue(_validate_array(computed))
        self.assertTrue(_equal_arrays(computed.collect(), expected))


def main():
    unittest.main()


if __name__ == "__main__":
    main()
