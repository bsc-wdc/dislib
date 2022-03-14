import unittest
from dislib.decomposition import tsqr
from dislib.data.array import random_array
import dislib as ds
import numpy as np
from parameterized import parameterized


class QRTest(unittest.TestCase):
    @parameterized.expand([
        (2, 1, 64, 36), (4, 1, 32, 36), (16, 1, 20, 10),
    ])
    def test_tsqr(self, m_size, n_size, b_size_r, b_size_c):
        """Tests tsqr"""

        shape = (m_size * b_size_r, n_size * b_size_c)
        m2b_ds = random_array(shape, (b_size_r, b_size_c))

        (q, r) = tsqr(m2b_ds)

        q = q.collect()
        r = r.collect()
        m2b = m2b_ds.collect()
        # check if R matrix is upper triangular
        self.assertTrue(np.allclose(np.triu(r), r))
        # check if the product Q * R is the original matrix
        self.assertTrue(np.allclose(q.dot(r), m2b))

    def test_tsqr_warning(self):
        m2b_ds = random_array((50, 20), (10, 10))
        (q, r) = tsqr(m2b_ds)
        q = q.collect()
        r = r.collect()
        m2b_ds = m2b_ds.collect()
        # check if R matrix is upper triangular
        self.assertTrue(np.allclose(np.triu(r), r))
        # check if the product Q * R is the original matrix
        self.assertTrue(np.allclose(q.dot(r), m2b_ds))

    def test_tsqr_exceptions(self):
        m2b_ds = ds.array([[1, 2, 3], [4, 5, 6]], (2, 3))
        m2b_ds.__init__([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3],
                         [4, 5, 6]], top_left_shape=(1, 3), reg_shape=(2, 3),
                        shape=(5, 3), sparse=False)
        with self.assertRaises(ValueError):
            # top left shape blocks needs to have the same shape
            # as other blocks
            tsqr(m2b_ds)
        m2b_ds = random_array((50, 100), (10, 100))
        with self.assertRaises(ValueError):
            # m > n is required for matrices m x n
            tsqr(m2b_ds)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
