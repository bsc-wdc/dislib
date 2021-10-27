import unittest

import numpy as np
from parameterized import parameterized
from scipy import sparse as sp

import dislib as ds


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


class MatmulTest(unittest.TestCase):

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
        computed = ds.matmul(a, b)
        self.assertTrue(_validate_array(computed))
        self.assertTrue(_equal_arrays(expected, computed.collect(False)))

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
    def test_matmul_transpose_a(self, shape_a, shape_b, sparse):
        """ Tests ds-array multiplication """
        a_np = np.random.random((shape_a[1], shape_a[0]))
        b_np = np.random.random(shape_b)

        if sparse:
            a_np = sp.csr_matrix(a_np)
            b_np = sp.csr_matrix(b_np)

        b0 = np.random.randint(1, min(a_np.shape[0], b_np.shape[0]) + 1)
        b1 = np.random.randint(1, a_np.shape[1] + 1)
        b2 = np.random.randint(1, b_np.shape[1] + 1)

        a = ds.array(a_np, (b0, b1))
        b = ds.array(b_np, (b0, b2))

        expected = a_np.T @ b_np
        computed = ds.matmul(a, b, transpose_a=True)
        self.assertTrue(_validate_array(computed))
        self.assertTrue(_equal_arrays(expected, computed.collect(False)))

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
    def test_matmul_transpose_b(self, shape_a, shape_b, sparse):
        """ Tests ds-array multiplication """
        a_np = np.random.random(shape_a)
        b_np = np.random.random((shape_b[1], shape_b[0]))

        if sparse:
            a_np = sp.csr_matrix(a_np)
            b_np = sp.csr_matrix(b_np)

        b0 = np.random.randint(1, a_np.shape[0] + 1)
        b1 = np.random.randint(1, b_np.shape[0] + 1)
        b2 = np.random.randint(1, min(a_np.shape[1], b_np.shape[1]) + 1)

        a = ds.array(a_np, (b0, b2))
        b = ds.array(b_np, (b1, b2))

        expected = a_np @ b_np.T
        computed = ds.matmul(a, b, transpose_b=True)
        self.assertTrue(_validate_array(computed))
        self.assertTrue(_equal_arrays(expected, computed.collect(False)))

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
    def test_matmul_transpose_ab(self, shape_a, shape_b, sparse):
        """ Tests ds-array multiplication """
        a_np = np.random.random((shape_a[1], shape_a[0]))
        b_np = np.random.random((shape_b[1], shape_b[0]))

        if sparse:
            a_np = sp.csr_matrix(a_np)
            b_np = sp.csr_matrix(b_np)

        b0 = np.random.randint(1, min(a_np.shape[0], b_np.shape[1]) + 1)
        b1 = np.random.randint(1, a_np.shape[1] + 1)
        b2 = np.random.randint(1, b_np.shape[0] + 1)

        a = ds.array(a_np, (b0, b1))
        b = ds.array(b_np, (b2, b0))

        expected = a_np.T @ b_np.T
        computed = ds.matmul(a, b, transpose_a=True, transpose_b=True)
        self.assertTrue(_validate_array(computed))
        self.assertTrue(_equal_arrays(expected, computed.collect(False)))

    def test_matmul_error(self):
        """ Tests matmul errors """

        with self.assertRaises(NotImplementedError):
            x1 = ds.random_array((5, 3), (2, 2))
            x1 = x1[1:]
            x2 = ds.random_array((3, 5), (2, 2))
            ds.matmul(x1, x2)

        with self.assertRaises(NotImplementedError):
            x1 = ds.random_array((5, 3), (2, 2))
            x2 = ds.random_array((3, 5), (2, 2))
            x2 = x2[1:]
            ds.matmul(x1, x2)

        with self.assertRaises(ValueError):
            x1 = ds.random_array((5, 3), (5, 3))
            x2 = ds.random_array((3, 5), (2, 5))
            ds.matmul(x1, x2, transpose_a=True)


def main():
    unittest.main()


if __name__ == "__main__":
    main()
