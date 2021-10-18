import unittest

import numpy as np
from numpy.linalg import qr as qr_numpy
from parameterized import parameterized
from pycompss.api.api import compss_wait_on

from dislib.data.array import random_array
from dislib.decomposition import qr
from dislib.decomposition.qr.base import (
    _dot_task,
    _qr_task,
    _transpose_block,
    _validate_ds_array,
    ZEROS,
    IDENTITY
)


class QRTest(unittest.TestCase):

    @parameterized.expand([
        (1, 1, 2), (1, 1, 4), (2, 2, 2), (2, 2, 4),
        (3, 3, 3), (3, 3, 4), (4, 4, 2), (4, 4, 3),
        (4, 4, 4), (6, 6, 6), (8, 8, 8), (2, 1, 2),
        (2, 1, 4), (3, 2, 2), (3, 2, 4), (4, 3, 3),
        (4, 3, 4), (5, 4, 2), (10, 6, 6), (10, 6, 6),
    ])
    def test_qr(self, m_size, n_size, b_size):
        """Tests qr_blocked full mode"""
        np.set_printoptions(precision=2)
        np.random.seed(8)

        shape = (m_size * b_size, n_size * b_size)
        m2b_ds = random_array(shape, (b_size, b_size))

        (q, r) = qr(m2b_ds)

        q = q.collect()
        r = r.collect()
        m2b = m2b_ds.collect()

        # check if Q matrix is orthogonal
        self.assertTrue(np.allclose(q.dot(q.T), np.identity(m_size * b_size)))
        # check if R matrix is upper triangular
        self.assertTrue(np.allclose(np.triu(r), r))
        # check if the product Q * R is the original matrix
        self.assertTrue(np.allclose(q.dot(r), m2b))

    @parameterized.expand([
        ((7, 6), (3, 3)), ((7, 5), (2, 2)), ((10, 4), (3, 3)),
        ((4, 4), (3, 3)), ((6, 4), (3, 3)), ((6, 5), (2, 2)),
    ])
    def test_qr_with_padding(self, m_shape, b_shape):
        """Tests qr_blocked with padding"""
        np.set_printoptions(precision=2)
        np.random.seed(8)

        m2b_ds = random_array(m_shape, b_shape)

        (q, r) = qr(m2b_ds, mode='full')

        q = q.collect()
        r = r.collect()

        # check if Q matrix is orthogonal
        self.assertTrue(np.allclose(q.dot(q.T), np.identity(m_shape[0])))
        # check if R matrix is upper triangular
        self.assertTrue(np.allclose(np.triu(r), r))
        # check if the product Q * R is the original matrix
        self.assertTrue(np.allclose(q.dot(r), m2b_ds.collect()))

        (q, r) = qr(m2b_ds, mode="economic")

        q = q.collect()
        r = r.collect()
        m2b = m2b_ds.collect()

        # check if Q matrix is orthogonal
        self.assertTrue(np.allclose(q.T.dot(q), np.identity(m_shape[1])))
        # check if R matrix is upper triangular
        self.assertTrue(np.allclose(np.triu(r), r))
        # check if the product Q * R is the original matrix
        self.assertTrue(np.allclose(q.dot(r), m2b))
        # check the dimensions of Q
        self.assertTrue(q.shape == (m_shape[0], m_shape[1]))
        # check the dimensions of R
        self.assertTrue(r.shape == (m_shape[1], m_shape[1]))

    @parameterized.expand([
        (1, 1, 2), (1, 1, 4), (2, 2, 2), (2, 2, 4),
        (3, 3, 3), (3, 3, 4), (4, 4, 2), (4, 4, 3),
        (4, 4, 4), (6, 6, 6), (8, 8, 8), (2, 1, 2),
        (2, 1, 4), (3, 2, 2), (3, 2, 4), (4, 3, 3),
        (4, 3, 4), (5, 4, 2), (10, 6, 6), (10, 6, 6),
    ])
    def test_qr_economic(self, m_size, n_size, b_size):
        """Tests qr_blocked economic mode"""
        np.set_printoptions(precision=2)
        np.random.seed(8)

        shape = (m_size * b_size, n_size * b_size)
        m2b_ds = random_array(shape, (b_size, b_size))

        (q, r) = qr(m2b_ds, mode="economic")

        q = q.collect()
        r = r.collect()
        m2b = m2b_ds.collect()

        # check if Q matrix is orthogonal
        self.assertTrue(np.allclose(q.T.dot(q), np.identity(n_size * b_size)))
        # check if R matrix is upper triangular
        self.assertTrue(np.allclose(np.triu(r), r))
        # check if the product Q * R is the original matrix
        self.assertTrue(np.allclose(q.dot(r), m2b))
        # check the dimensions of Q
        self.assertTrue(q.shape == (m_size * b_size, n_size * b_size))
        # check the dimensions of R
        self.assertTrue(r.shape == (n_size * b_size, n_size * b_size))

    @parameterized.expand([
        (1, 1, 2), (1, 1, 4), (2, 2, 2), (2, 2, 4),
        (3, 3, 3), (3, 3, 4), (4, 4, 2), (4, 4, 3),
        (4, 4, 4), (6, 6, 6), (8, 8, 8), (2, 1, 2),
        (2, 1, 4), (3, 2, 2), (3, 2, 4), (4, 3, 3),
        (4, 3, 4), (5, 4, 2), (10, 6, 6), (10, 6, 6),
    ])
    def test_qr_r(self, m_size, n_size, b_size):
        """Tests qr_blocked r mode"""
        np.set_printoptions(precision=2)
        np.random.seed(8)

        shape = (m_size * b_size, n_size * b_size)
        m2b_ds = random_array(shape, (b_size, b_size))

        r = qr(m2b_ds, mode="r")
        _, r_full = qr(m2b_ds, mode="full")

        r = r.collect()
        r_full = r_full.collect()

        # check if R matrix is upper triangular
        self.assertTrue(np.allclose(np.triu(r), r))
        # check if R matrix is the same as when the full mode is applied
        self.assertTrue(np.allclose(r, r_full))

    def test_special_cases(self):
        """Tests special cases not covered in other tests"""
        np.set_printoptions(precision=2)
        np.random.seed(8)

        # testing a block transpose
        a = None
        a_type = [[ZEROS]]
        a_transposed_type, a = _transpose_block(a, a_type)
        self.assertEqual(a_type, a_transposed_type)
        self.assertIsNone(a)

        a = None
        a_type = [[IDENTITY]]
        a_transposed_type, a = _transpose_block(a, a_type)
        self.assertEqual(a_type, a_transposed_type)
        self.assertIsNone(a)

        # testing array validation
        a = random_array((10, 10), (2, 5))
        with self.assertRaises(ValueError):
            _validate_ds_array(a)

        # testing transpose dot
        a = np.random.rand(3, 2)
        b = np.random.rand(2, 5)
        a_dot_b_transposed = compss_wait_on(_dot_task(a, b,
                                                      transpose_result=True))
        self.assertTrue(np.allclose(np.transpose(np.dot(a, b)),
                                    a_dot_b_transposed))

        # testing qr of zeros or identity
        b_size = (5, 5)
        a = None
        q, r = _qr_task(a, np.full((1, 1), ZEROS), b_size)
        q = compss_wait_on(q)
        r = compss_wait_on(r)
        q_numpy, r_numpy = qr_numpy(np.zeros(b_size), mode='reduced')
        self.assertTrue(np.allclose(q_numpy, q))
        self.assertTrue(np.allclose(r_numpy, r))
        q, r = _qr_task(a, np.full((1, 1), IDENTITY), b_size)
        q = compss_wait_on(q)
        r = compss_wait_on(r)
        q_numpy, r_numpy = qr_numpy(np.identity(b_size[0]), mode='reduced')
        self.assertTrue(np.allclose(q_numpy, q))
        self.assertTrue(np.allclose(r_numpy, r))

    def test_economic_overwrite(self):
        """Tests economic qr with overwrite = True"""
        np.set_printoptions(precision=2)
        np.random.seed(8)
        m_size = 25
        n_size = 10
        b_size = 5
        m2b_ds = random_array((m_size, n_size), (b_size, b_size))
        (q, r) = qr(m2b_ds, mode="economic", overwrite_a=True)

        q = q.collect()
        r = r.collect()
        m2b = m2b_ds.collect()

        # check if arrays original and r arrays are different
        # as it is not possible to overwrite the original matrix in the
        # economic mode
        self.assertFalse(np.array_equal(m2b, r))

        # check if Q matrix is orthogonal
        self.assertTrue(np.allclose(q.T.dot(q), np.identity(n_size)))
        # check if R matrix is upper triangular
        self.assertTrue(np.allclose(np.triu(r), r))
        # check if the product Q * R is the original matrix
        self.assertTrue(np.allclose(q.dot(r), m2b))
        # check the dimensions of Q
        self.assertTrue(q.shape == (m_size, n_size))
        # check the dimensions of R
        self.assertTrue(r.shape == (n_size, n_size))

    def test_exceptions(self):
        m2b_ds = random_array((100, 100), (10, 10))
        with self.assertRaises(ValueError):
            # unsupported mode
            qr(m2b_ds, mode="x")

        m2b_ds = random_array((50, 100), (10, 10))
        with self.assertRaises(ValueError):
            # m > n is required for matrices m x n
            qr(m2b_ds)

        m2b_ds = random_array((100, 100), (10, 5))
        with self.assertRaises(ValueError):
            # Square blocks are required
            qr(m2b_ds)

        m2b_ds = random_array((100, 100), (10, 10))
        m2b_ds = m2b_ds[:, 0:5]
        with self.assertRaises(ValueError):
            qr(m2b_ds)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
