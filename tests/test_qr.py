import unittest

import numpy as np
from parameterized import parameterized
from pycompss.api.api import compss_barrier, compss_wait_on

from dislib.data.array import random_array
from dislib.math import qr


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

        compss_barrier()

        (q, r) = qr(m2b_ds)

        q = compss_wait_on(q).collect()
        r = compss_wait_on(r).collect()
        m2b_ds = compss_wait_on(m2b_ds)
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
        (30, 5, 10)
    ])
    def test_qr_economic(self, m_size, n_size, b_size):
        """Tests qr_blocked economic mode"""
        np.set_printoptions(precision=2)
        np.random.seed(8)

        shape = (m_size * b_size, n_size * b_size)

        m2b_ds = random_array(shape, (b_size, b_size))

        compss_barrier()

        (q, r) = qr(m2b_ds, mode="economic")

        q = compss_wait_on(q).collect()
        r = compss_wait_on(r).collect()
        m2b_ds = compss_wait_on(m2b_ds)
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

        compss_barrier()

        r = qr(m2b_ds, mode="r")
        _, r_full = qr(m2b_ds, mode="full")

        r = compss_wait_on(r).collect()
        r_full = compss_wait_on(r_full).collect()

        # check if R matrix is upper triangular
        self.assertTrue(np.allclose(np.triu(r), r))
        # check if R matrix is the same as when the full mode is applied
        self.assertTrue(np.allclose(r, r_full))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
