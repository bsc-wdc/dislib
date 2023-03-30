import unittest
from dislib.decomposition import tsqr
from dislib.data.array import random_array
import dislib as ds
import numpy as np
from parameterized import parameterized

from dislib.decomposition.tsqr.base import _is_not_power_of_two
from tests import BaseTimedTestCase


class QRTest(BaseTimedTestCase):
    @parameterized.expand([
        (2, 1, 64, 36), (3, 1, 64, 36), (4, 1, 32, 36), (16, 1, 20, 10),
    ])
    def test_tsqr(self, m_size, n_size, b_size_r, b_size_c):
        """Tests tsqr"""

        shape = (m_size * b_size_r, n_size * b_size_c)
        m2b_ds = random_array(shape, (b_size_r, b_size_c))

        (q, r) = tsqr(m2b_ds)
        assigned_q_shape = q.shape
        assigned_r_shape = r.shape
        q = q.collect()
        r = r.collect()
        m2b = m2b_ds.collect()
        self.assertEqual(assigned_q_shape, q.shape)
        self.assertEqual(assigned_r_shape, r.shape)
        # check if R matrix is upper triangular
        self.assertTrue(np.allclose(np.triu(r), r))
        # check if Q matrix is orthogonal
        self.assertTrue(np.allclose(np.dot(q, q.transpose()),
                                    np.eye(q.shape[0])))
        # check if the product Q * R is the original matrix
        self.assertTrue(np.allclose(q.dot(r), m2b))

    @parameterized.expand([
        (2, 1, 64, 36), (3, 1, 64, 36), (4, 1, 32, 36), (16, 1, 20, 10),
    ])
    def test_tsqr_complete_indexes(self, m_size, n_size, b_size_r, b_size_c):
        """Tests tsqr"""

        shape = (m_size * b_size_r, n_size * b_size_c)
        m2b_ds = random_array(shape, (b_size_r, b_size_c))
        (q, r) = tsqr(m2b_ds, indexes=[2, 3, 4])
        assigned_q_shape = q.shape
        assigned_r_shape = r.shape
        q = q.collect()
        r = r.collect()
        self.assertEqual(assigned_q_shape, q.shape)
        self.assertEqual(assigned_r_shape, r.shape)
        # check if R matrix is upper triangular
        self.assertTrue(np.allclose(np.triu(r), r))
        # check if the Q matrix contains the number of specified indexes
        self.assertTrue(q.shape == (q.shape[0], 3))

    @parameterized.expand([
        (2, 1, 64, 36), (4, 1, 32, 36), (16, 1, 20, 10),
    ])
    def test_tsqr_inverse(self, m_size, n_size, b_size_r, b_size_c):
        """Tests tsqr"""

        shape = (m_size * b_size_r, n_size * b_size_c)
        m2b_ds = random_array(shape, (b_size_r, b_size_c))

        (q, r) = tsqr(m2b_ds, mode="complete_inverse")
        assigned_q_shape = q.shape
        assigned_r_shape = r.shape
        q = q.collect()
        r = r.collect()
        m2b = m2b_ds.collect()
        self.assertEqual(assigned_q_shape, q.shape)
        self.assertEqual(assigned_r_shape, r.shape)
        # check if R matrix is upper triangular
        self.assertTrue(np.allclose(np.triu(r), r))
        # check if Q matrix is orthogonal
        self.assertTrue(np.allclose(np.dot(q, q.transpose()),
                                    np.eye(q.shape[0])))
        # check if the product Q * R is the original matrix
        self.assertTrue(np.allclose(q.dot(r), m2b))

    @parameterized.expand([
        (2, 1, 64, 36), (4, 1, 32, 36), (16, 1, 20, 10),
    ])
    def test_tsqr_inverse_indexes(self, m_size, n_size, b_size_r, b_size_c):
        """Tests tsqr"""

        shape = (m_size * b_size_r, n_size * b_size_c)
        m2b_ds = random_array(shape, (b_size_r, b_size_c))

        (q, r) = tsqr(m2b_ds, mode="complete_inverse", indexes=[2, 3, 4])
        assigned_q_shape = q.shape
        assigned_r_shape = r.shape
        q = q.collect()
        r = r.collect()
        self.assertEqual(assigned_q_shape, q.shape)
        self.assertEqual(assigned_r_shape, r.shape)
        # check if R matrix is upper triangular
        self.assertTrue(np.allclose(np.triu(r), r))
        # self.assertTrue(q.shape == (q.shape[0], 3))

    @parameterized.expand([
        (2, 1, 64, 36), (3, 1, 64, 36), (4, 1, 36, 32), (16, 1, 20, 10),
    ])
    def test_tsqr_reduced(self, m_size, n_size, b_size_r, b_size_c):
        """Tests tsqr"""

        shape = (m_size * b_size_r, n_size * b_size_c)
        m2b_ds = random_array(shape, (b_size_r, b_size_c))

        (q, r) = tsqr(m2b_ds, mode="reduced")
        assigned_q_shape = q.shape
        assigned_r_shape = r.shape
        q = q.collect()
        r = r.collect()
        self.assertEqual(assigned_q_shape, q.shape)
        self.assertEqual(assigned_r_shape, r.shape)
        m2b = m2b_ds.collect()
        # check if R matrix is upper triangular
        self.assertTrue(np.allclose(np.triu(r), r))
        # check if the product Q * R is the original matrix
        self.assertTrue(np.allclose(q.dot(r), m2b))

    @parameterized.expand([
        (2, 1, 64, 36), (4, 1, 36, 32), (16, 1, 20, 10),
    ])
    def test_tsqr_reduced_inverse(self, m_size, n_size, b_size_r, b_size_c):
        """Tests tsqr"""

        shape = (m_size * b_size_r, n_size * b_size_c)
        m2b_ds = random_array(shape, (b_size_r, b_size_c))

        (q, r) = tsqr(m2b_ds, mode="reduced_inverse")
        assigned_q_shape = q.shape
        assigned_r_shape = r.shape
        q = q.collect()
        r = r.collect()
        m2b = m2b_ds.collect()
        self.assertEqual(assigned_q_shape, q.shape)
        self.assertEqual(assigned_r_shape, r.shape)
        # check if R matrix is upper triangular
        self.assertTrue(np.allclose(np.triu(r), r))
        # check if the product Q * R is the original matrix
        self.assertTrue(np.allclose(q.dot(r), m2b))

    @parameterized.expand([
        (2, 1, 64, 36), (4, 1, 36, 32), (16, 1, 20, 10),
    ])
    def test_tsqr_reduced_inverse_indexes(self, m_size, n_size,
                                          b_size_r, b_size_c):
        """Tests tsqr"""

        shape = (m_size * b_size_r, n_size * b_size_c)
        m2b_ds = random_array(shape, (b_size_r, b_size_c))

        (q, r) = tsqr(m2b_ds, mode="reduced_inverse", indexes=[2, 3, 4])
        assigned_q_shape = q.shape
        assigned_r_shape = r.shape
        q = q.collect()
        r = r.collect()
        self.assertEqual(assigned_q_shape, q.shape)
        self.assertEqual(assigned_r_shape, r.shape)
        # check if R matrix is upper triangular
        self.assertTrue(np.allclose(np.triu(r), r))
        # check if the product Q columns retrived are equal to the ones in the
        # Q computed with numpy
        self.assertTrue(q.shape == (q.shape[0], 3))

    @parameterized.expand([
        (2, 1, 64, 36), (3, 1, 64, 36), (4, 1, 36, 32), (16, 1, 20, 10),
    ])
    def test_tsqr_compute_r(self, m_size, n_size, b_size_r, b_size_c):
        """Tests tsqr"""

        shape = (m_size * b_size_r, n_size * b_size_c)
        m2b_ds = random_array(shape, (b_size_r, b_size_c))

        r = tsqr(m2b_ds, mode="r_complete")
        assigned_r_shape = r.shape
        r = r.collect()
        self.assertEqual(assigned_r_shape, r.shape)
        # check if R matrix is upper triangular
        self.assertTrue(np.allclose(np.triu(r), r))

    @parameterized.expand([
        (2, 1, 64, 36), (3, 1, 64, 36), (4, 1, 36, 32), (16, 1, 20, 10),
    ])
    def test_tsqr_compute_r_reduced(self, m_size, n_size, b_size_r, b_size_c):
        """Tests tsqr"""

        shape = (m_size * b_size_r, n_size * b_size_c)
        m2b_ds = random_array(shape, (b_size_r, b_size_c))

        r = tsqr(m2b_ds, mode="r_reduced")
        assigned_r_shape = r.shape
        r = r.collect()
        self.assertEqual(assigned_r_shape, r.shape)
        # check if R matrix is upper triangular
        self.assertTrue(np.allclose(np.triu(r), r))

    def test_tsqr_warning(self):
        m2b_ds = random_array((50, 20), (10, 10))
        (q, r) = tsqr(m2b_ds)
        q = q.collect()
        r = r.collect()
        m2b_ds = m2b_ds.collect()
        # check if R matrix is upper triangular
        self.assertTrue(np.allclose(np.triu(r), r))
        # check if Q matrix is orthogonal
        self.assertTrue(np.allclose(np.dot(q, q.transpose()),
                                    np.eye(q.shape[0])))
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
        m2b_ds = random_array((500, 100), (100, 100))
        with self.assertRaises(ValueError):
            # power of 2 is required
            tsqr(m2b_ds, mode="complete_inverse")
        m2b_ds = random_array((500, 100), (100, 100))
        with self.assertRaises(ValueError):
            # power of n_reduction is required
            tsqr(m2b_ds, n_reduction=3, mode="complete_inverse")
        m2b_ds = random_array((500, 100), (100, 100))
        with self.assertRaises(ValueError):
            # power of 2 is required
            tsqr(m2b_ds, mode="reduced_inverse")
        m2b_ds = random_array((500, 100), (100, 100))
        with self.assertRaises(ValueError):
            # power of n_reduction is required
            tsqr(m2b_ds, n_reduction=3, mode="reduced_inverse")

    def test_power_two_returns_false(self):
        self.assertFalse(_is_not_power_of_two(0))
        self.assertFalse(_is_not_power_of_two(-1))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
