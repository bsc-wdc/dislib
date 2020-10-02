import unittest

import numpy as np
from parameterized import parameterized
from scipy import sparse as sp
from sklearn.datasets import load_svmlight_file
from hecuba import config
import dislib as ds
from math import ceil

from pycompss.api.api import compss_wait_on , compss_barrier
import time
from tests.func_sum_and_mult import _sum_and_mult


def _validate_array(x):
    x._blocks=compss_wait_on(x._blocks)
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



def _gen_random_arrays(fmt, shape=None, block_size=None, persistent=None):
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
    elif "sparse" in fmt:
        x_np = sp.csr_matrix(np.random.random(shape))
        x = ds.array(x_np, block_size=block_size)  
    return x, x_np, persistent


def _gen_irregular_arrays(fmt, shape=None, block_size=None, persistent=None):
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
        return x[1:, 1:], x_np[1:, 1:], persistent
    elif "sparse" in fmt:
        x_sp = sp.csr_matrix(np.random.random(shape))
        x = ds.array(x_sp, block_size=block_size)
        return x[1:, 1:], x_sp[1:, 1:], persistent

class DataLoadingTest(unittest.TestCase):

    @parameterized.expand([(_gen_random_arrays("dense", (6, 10), (4, 3))
                            + ((6, 10), (4, 3))),
                           (_gen_random_arrays("sparse", (6, 10), (4, 3))
                            + ((6, 10), (4, 3))),
                            (_gen_random_arrays("dense", (6, 10), (4, 3), "test1")
                            + ((6, 10), (4, 3))),
                            (_gen_random_arrays("dense", (6, 11), (4, 3), "test2")
                            + ((6, 11), (4, 3)))])
    def test_array_constructor(self, x, x_np, persistent, shape, block_size):
        """ Tests array constructor """
        n, m = shape
        bn, bm = block_size       
        if persistent!= None:
            x.make_persistent(name="hecuba_dislib.test_array_constructor")

        self.assertTrue(x._n_blocks, ceil(n / bn) == ceil(m / bm))
        self.assertTrue(_equal_arrays(x.collect(), x_np))

    

    def test_array_creation_persistent(self):
        """ Tests array creation """
        data = [[1, 2, 3], [4, 5, 6]]

        x_np = np.array(data)
        x = ds.array(data, (2, 3))
        x.make_persistent(name="hecuba_dislib.test_array_creation1")         
        self.assertTrue(_validate_array(x))
        self.assertTrue(_equal_arrays(x.collect(), x_np))

        x = ds.array(x_np, (2, 3))
        x.make_persistent(name="hecuba_dislib.test_array_creation2")         
        self.assertTrue(_validate_array(x))
        self.assertTrue(_equal_arrays(x.collect(), x_np))

        x_np = np.random.random(10)
        x = ds.array(x_np, (1, 5))
        x.make_persistent(name="hecuba_dislib.test_array_creation3")
        self.assertTrue(_validate_array(x))
        self.assertTrue(_equal_arrays(x.collect(), x_np))

        x_np = np.random.random(10)
        x = ds.array(x_np, (5, 1))
        x.make_persistent(name="hecuba_dislib.test_array_creation4")
        self.assertTrue(_validate_array(x))
        self.assertTrue(_equal_arrays(x.collect(), x_np))

        with self.assertRaises(ValueError):
            x_np = np.random.random(10)
            ds.array(x_np, (5, 5))

    

class ArrayTest(unittest.TestCase):

    @parameterized.expand([_gen_random_arrays(fmt = "dense"),
                           _gen_random_arrays(fmt = "sparse"),
                           _gen_random_arrays(fmt = "dense", persistent = "test1")])
    def test_sizes(self, x, x_np, persistent):
        """ Tests sizes consistency. """
        if persistent!= None:
            x.make_persistent(name="hecuba_dislib.test_sizes")
        bshape = x._reg_shape
        shape = x_np.shape
        
        self.assertEqual(x.shape, shape)
        self.assertEqual(x._n_blocks, (ceil(shape[0] / bshape[0]),
                                       (ceil(shape[1] / bshape[1]))))

    @parameterized.expand([_gen_random_arrays(fmt = "dense"),
                           _gen_random_arrays(fmt = "sparse"),
                           _gen_random_arrays(fmt = "dense", persistent = "t1")])
    def test_iterate_rows(self, x, x_np, persistent):
        """ Testing the row _iterator of the ds.array """
        if persistent!= None:
            x.make_persistent(name="hecuba_dislib.ite"+persistent)

        n_rows = x._reg_shape[0]
        for i, h_block in enumerate(x._iterator(axis='rows')):
            computed = h_block
            expected = x_np[i * n_rows: (i + 1) * n_rows]
            self.assertTrue(_validate_array(computed))
            self.assertTrue(_equal_arrays(computed.collect(), expected))


    @parameterized.expand([_gen_random_arrays(fmt = "dense"),
                           _gen_random_arrays(fmt = "sparse"),
                           _gen_random_arrays(fmt = "dense", persistent = "t2")])
    def test_iterate_cols(self, x, x_np, persistent):
        if persistent!= None:
            x.make_persistent(name="hecuba_dislib.test_ite"+persistent)

        """ Testing the row _iterator of the ds.array """
        n_cols = x._reg_shape[1]

        for i, v_block in enumerate(x._iterator(axis='columns')):
            expected = x_np[:, i * n_cols: (i + 1) * n_cols]
            self.assertTrue(_validate_array(v_block))
            self.assertTrue(_equal_arrays(v_block.collect().reshape(
                v_block.shape), expected))

    

    @parameterized.expand([_gen_random_arrays(fmt = "dense", persistent = "test12"),
                           _gen_random_arrays(fmt = "dense", persistent = "test12"),
                           _gen_random_arrays(fmt = "dense", shape=(33, 34), block_size= (2, 33), persistent = "test21"),
                           _gen_irregular_arrays(fmt = "dense", persistent="test22")])
    def test_indexing(self, x, x_np, persistent=None):
        """ Tests indexing """
        # Single row
        if persistent!= None:
            x.make_persistent(name="hecuba_dislib.test_indexing"+persistent)

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


    @parameterized.expand([_gen_random_arrays("dense", persistent="test22"),
                           _gen_random_arrays("dense", persistent="test25"),
                           _gen_irregular_arrays("dense", persistent="test24"),
                           _gen_irregular_arrays("dense", (22, 49), (3, 1), persistent="test28") +
                           (None, [18, 20, 41, 44]),
                           _gen_irregular_arrays("dense", (49, 22), (1, 3), persistent="test29") +
                           ([18, 20, 41, 44], None),
                           _gen_random_arrays("dense", (5, 4), (3, 3), persistent="test30") +
                           ([0, 1, 3, 4], None),
                           _gen_random_arrays("dense", (4, 5), (3, 3), persistent="test31") +
                           (None, [0, 1, 3, 4])])
    def test_fancy_indexing(self, x, x_np, persistent=None, rows=None, cols=None):
        """ Tests fancy indexing """
        if persistent!= None:
            x.make_persistent(name="hecuba_dislib.test_indexing"+persistent)
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


    @parameterized.expand([_gen_random_arrays("dense", persistent="t1"),
                           _gen_random_arrays("dense", (1, 10), (1, 2), persistent="t2"),
                           _gen_random_arrays("dense", (10, 1), (3, 1), persistent="t3"),
                           _gen_irregular_arrays("dense", persistent="t4")])  
    def test_transpose(self, x, x_np, persistent):
        """ Tests array transpose."""
        if persistent!= None:
            x.make_persistent(name="hecuba_dislib.test_transpose"+persistent)
        
        b0, b1 = x._n_blocks
        x_t = x.transpose(mode="all")
        x_np_t = x_np.transpose()

        x_t._blocks=compss_wait_on(x_t._blocks)

        self.assertTrue(
            _equal_arrays(x_t.collect().reshape(x_t.shape), x_np_t))
        self.assertEqual((b1, b0), x_t._n_blocks)
        self.assertTrue(_validate_array(x_t))

        x_t = x.T
        x_t._blocks=compss_wait_on(x_t._blocks)
        self.assertTrue(
            _equal_arrays(x_t.collect().reshape(x_t.shape), x_np_t))
        self.assertEqual((b1, b0), x_t._n_blocks)
        self.assertTrue(_validate_array(x_t))

        x_t = x.transpose(mode="columns")
        x_t._blocks=compss_wait_on(x_t._blocks)
        self.assertTrue(
            _equal_arrays(x_t.collect().reshape(x_t.shape), x_np_t))
        self.assertEqual((b1, b0), x_t._n_blocks)
        self.assertTrue(_validate_array(x_t))

        with self.assertRaises(Exception):
            x.transpose(mode="invalid")


    


    @parameterized.expand([(ds.array(np.array([[1, 2, 3],
                                      [4, 5, 6],
                                      [7, 8, 9]]), (2, 2)),)])
    def test_apply_axis_persistent(self, x):
        """ Tests apply along axis """
        if x._sparse == False:
            x.make_persistent(name='hecuba_dislib.test_applyaxis')

        x1 = ds.apply_along_axis(_sum_and_mult, 0, x)
        self.assertTrue(x1.shape, (1, 3))
        self.assertTrue(x1._reg_shape, (1, 2))
        self.assertTrue(
            np.array_equal(x1.collect(), np.array([12, 15, 18])))
        self.assertTrue(_validate_array(x1))

        x1 = ds.apply_along_axis(_sum_and_mult, 1, x)
        self.assertTrue(x1.shape, (3, 1))
        self.assertTrue(x1._reg_shape, (2, 1))
        self.assertTrue(
            np.array_equal(x1.collect(), np.array([6, 15, 24])))
        self.assertTrue(_validate_array(x1))

        x1 = ds.apply_along_axis(_sum_and_mult, 1, x, 2)
        self.assertTrue(x1.shape, (3, 1))
        self.assertTrue(x1._reg_shape, (2, 1))
        self.assertTrue(
            np.array_equal(x1.collect(), np.array([8, 17, 26])))
        self.assertTrue(_validate_array(x1))

        x1 = ds.apply_along_axis(_sum_and_mult, 1, x, b=2)
        self.assertTrue(x1.shape, (3, 1))
        self.assertTrue(x1._reg_shape, (2, 1))
        self.assertTrue(
            np.array_equal(x1.collect(), np.array([12, 30, 48])))
        self.assertTrue(_validate_array(x1))

        x1 = ds.apply_along_axis(_sum_and_mult, 1, x, 1, b=2)
        self.assertTrue(x1.shape, (3, 1))
        self.assertTrue(x1._reg_shape, (2, 1))
        self.assertTrue(
            np.array_equal(x1.collect(), np.array([14, 32, 50])))
        self.assertTrue(_validate_array(x1))

   
    @parameterized.expand([((20, 30), (30, 10), False, "t1"),
                           ((1, 10), (10, 7), False, "t2"),
                           ((5, 10), (10, 1), False, "t3"),
                           ((17, 13), (13, 9), False, "t4"),
                           ((1, 30), (30, 1), False, "t5"),
                           ((10, 1), (1, 20), False, "t6")])
    def test_matmul_persistent(self, shape_a, shape_b, sparse, persistent=None):
        """ Tests ds-array multiplication persistent"""
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

        if persistent != None:
            a.make_persistent(name="hecuba_dislib.test_matmul_a_"+persistent)
            b.make_persistent(name="hecuba_dislib.test_matmul_b_"+persistent)
        

        computed = a @ b
        computed._blocks=compss_wait_on(computed._blocks)
        self.assertTrue(_equal_arrays(expected, computed.collect(False)))


   

    def test_set_item_persistent(self):
        """ Tests setting a single value """
        x = ds.random_array((10, 10), (3, 3))
        x.make_persistent(name="hecuba_dislib.test_set_item_persistent")

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

        with self.assertRaises(IndexError):
            x[0] = 3


class CleanTest(unittest.TestCase):
    def clean_set(self):
        """ Tests clean """
        config.session.execute("TRUNCATE TABLE hecuba.istorage")
        config.session.execute("DROP KEYSPACE IF EXISTS hecuba_dislib")


def main():
    config.session.execute("TRUNCATE TABLE hecuba.istorage")
    config.session.execute("DROP KEYSPACE IF EXISTS hecuba_dislib")
    unittest.main(verbosity=2)



if __name__ == '__main__':
    main()
    