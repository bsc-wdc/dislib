import unittest

import numpy as np
from parameterized import parameterized
from pycompss.api.api import compss_barrier, compss_wait_on

from dislib.data.array import random_array
from dislib.math import qr


#class QRTest(unittest.TestCase):
class QRTest(object):

    @parameterized.expand([
        #(1, 1, 2), (1, 1, 4), (2, 2, 2), (2, 2, 4), (3, 3, 3), (3, 3, 4), (4, 4, 2),
        #(4, 4, 3), (4, 4, 4), (6, 6, 6), (8, 8, 8), (2, 1, 2), (2, 1, 4), (3, 2, 2),
        #(3, 2, 4), (4, 3, 3), (4, 3, 4), (5, 4, 2), (10, 6, 6)
        (10, 6, 6)
    ])
    def test_qr(self, m_size, n_size, b_size):
        """Tests qr_blocked"""
        np.set_printoptions(precision=2)
        np.random.seed(8)

        shape = (m_size * b_size, n_size * b_size)

        m2b_ds = random_array(shape, (b_size, b_size))

        compss_barrier()

        (Q, R) = qr(m2b_ds, save_memory=True)

        Q = compss_wait_on(Q).collect()
        R = compss_wait_on(R).collect()
        m2b_ds = compss_wait_on(m2b_ds)
        m2b = m2b_ds.collect()

        Q_blocked_np = self._ds_to_np(Q)
        R_blocked_np = self._ds_to_np(R)

        print("results")
        print(Q_blocked_np.dot(Q_blocked_np.T).shape)
        print(Q_blocked_np.dot(Q_blocked_np.T))
        print(R_blocked_np)
        print("original")
        print(m2b)
        print("results")
        print(Q_blocked_np.dot(R_blocked_np))

        # check if Q matrix is orthogonal
        self.assertTrue(np.allclose(Q_blocked_np.dot(Q_blocked_np.T), np.identity(m_size * b_size)))
        # check if R matrix is upper triangular
        self.assertTrue(np.allclose(np.triu(R_blocked_np), R_blocked_np))
        # check if the product Q * R is the original matrix
        self.assertTrue(np.allclose(Q_blocked_np.dot(R_blocked_np), m2b))

    def _ds_to_np(self, ds):
        ds_np = np.zeros(ds.shape)
        for i in range(ds.shape[0]):
            for j in range(ds.shape[1]):
                ds_np[i, j] = ds[i, j]
        return ds_np


#def main():
#    unittest.main()


#if __name__ == '__main__':
#    main()
