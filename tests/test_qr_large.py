import unittest

import numpy as np
from numpy.linalg import qr as qr_numpy
from pycompss.api.api import compss_wait_on

from dislib.data.array import random_array
from dislib.decomposition import qr


class QRLargeTest(unittest.TestCase):

    def test_qr(self):
        """Tests qr_blocked full mode"""
        np.set_printoptions(precision=2)
        np.random.seed(8)

        m2b_ds = random_array((20000, 20000), (2500, 2500))

        (q, r) = qr(m2b_ds)

        q = q.collect()
        r = r.collect()
        m2b = m2b_ds.collect()

        print(q.dtype)
        print()
        print()
        print(r.dtype)
        print()
        print()
        print(q.dot(q.T).dtype)

        # check if Q matrix is orthogonal
        self.assertTrue(np.allclose(q.dot(q.T), np.identity(20000)))
        # check if R matrix is upper triangular
        self.assertTrue(np.allclose(np.triu(r), r))
        # check if the product Q * R is the original matrix
        self.assertTrue(np.allclose(q.dot(r), m2b))

    


def main():
    unittest.main()


if __name__ == '__main__':
    main()
