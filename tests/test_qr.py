import unittest

import numpy as np
from parameterized import parameterized
from pycompss.api.api import compss_barrier, compss_wait_on

from dislib.data.array import random_array
from dislib.math import qr


class QRTest(unittest.TestCase):

    @parameterized.expand([
        (1, 1, 2, False), (1, 1, 4, False), (2, 2, 2, False), (2, 2, 4, False),
        (3, 3, 3, False), (3, 3, 4, False), (4, 4, 2, False), (4, 4, 3, False),
        (4, 4, 4, False), (6, 6, 6, False), (8, 8, 8, False), (2, 1, 2, False),
        (2, 1, 4, False), (3, 2, 2, False), (3, 2, 4, False), (4, 3, 3, False),
        (4, 3, 4, False), (5, 4, 2, False), (10, 6, 6, False), (10, 6, 6, False),
        (1, 1, 2, True), (1, 1, 4, True), (2, 2, 2, True), (2, 2, 4, True),
        (3, 3, 3, True), (3, 3, 4, True), (4, 4, 2, True), (4, 4, 3, True),
        (4, 4, 4, True), (6, 6, 6, True), (8, 8, 8, True), (2, 1, 2, True),
        (2, 1, 4, True), (3, 2, 2, True), (3, 2, 4, True), (4, 3, 3, True),
        (4, 3, 4, True), (5, 4, 2, True), (10, 6, 6, True), (10, 6, 6, True),
    ])
    def test_qr(self, m_size, n_size, b_size, save_memory):
        """Tests qr_blocked"""
        np.set_printoptions(precision=2)
        np.random.seed(8)

        shape = (m_size * b_size, n_size * b_size)

        m2b_ds = random_array(shape, (b_size, b_size))

        compss_barrier()

        (q, r) = qr(m2b_ds, save_memory=save_memory)

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
        ((7, 6), (3, 3), False), ((7, 5), (2, 2), False), ((10, 4), (3, 3), False),
        ((4, 4), (3, 3), False), ((6, 4), (3, 3), False), ((6, 5), (2, 2), False),
    ])
    def test_qr_with_padding(self, m_shape, b_shape, save_memory):
        """Tests qr_blocked with padding"""
        np.set_printoptions(precision=2)
        np.random.seed(8)

        m2b_ds = random_array(m_shape, b_shape)

        (q, r) = qr(m2b_ds, save_memory=save_memory)

        q = compss_wait_on(q).collect()
        r = compss_wait_on(r).collect()
        m2b_ds = compss_wait_on(m2b_ds)
        m2b = m2b_ds.collect()

        # check if Q matrix is orthogonal
        self.assertTrue(np.allclose(q.dot(q.T), np.identity(m_shape[0])))
        # check if R matrix is upper triangular
        self.assertTrue(np.allclose(np.triu(r), r))
        # check if the product Q * R is the original matrix
        self.assertTrue(np.allclose(q.dot(r), m2b))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
