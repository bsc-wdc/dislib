import unittest

import numpy as np
import itertools

import dislib as ds
from parameterized import parameterized
from dislib.math.base import svd_col_combs


class SVDTest(unittest.TestCase):

    def test_pairing(self):
        for n_cols in range(10):

            all_combs = list(itertools.combinations(range(n_cols), 2))
            cols_combs = svd_col_combs(n_cols)

            assert set(all_combs) == set(cols_combs)

    @parameterized.expand([(ds.array([[1, 0, 0, 0],
                                      [0, 0, 0, 2],
                                      [0, 3, 0, 0],
                                      [2, 0, 0, 0]], (2, 2)),),
                           (ds.random_array((17, 5), (1, 1)),),
                           (ds.random_array((9, 7), (9, 6)),),
                           (ds.random_array((10, 10), (2, 2))[1:, 1:],)])
    def test_svd(self, x):
        x_np = x.collect()
        u, s, v = ds.svd(x)
        u = u.collect()
        s = np.diag(s.collect())
        v = v.collect()

        self.assertTrue(np.allclose(x_np, u @ s @ v.T))
        self.assertTrue(
            np.allclose(np.linalg.norm(u, axis=0), np.ones(u.shape[1])))
        self.assertTrue(
            np.allclose(np.linalg.norm(v, axis=0), np.ones(v.shape[1])))

        u, s, v = ds.svd(x, sort=False)
        u = u.collect()
        s = np.diag(s.collect())
        v = v.collect()

        self.assertTrue(np.allclose(x_np, u @ s @ v.T))
        self.assertTrue(
            np.allclose(np.linalg.norm(u, axis=0), np.ones(u.shape[1])))
        self.assertTrue(
            np.allclose(np.linalg.norm(v, axis=0), np.ones(v.shape[1])))

        s = ds.svd(x, compute_uv=False, sort=False)
        s = np.diag(s.collect())

        # use U and V from previous decomposition
        self.assertTrue(np.allclose(x_np, u @ s @ v.T))
        self.assertTrue(
            np.allclose(np.linalg.norm(u, axis=0), np.ones(u.shape[1])))
        self.assertTrue(
            np.allclose(np.linalg.norm(v, axis=0), np.ones(v.shape[1])))

        u, s, v = ds.svd(x, copy=False)
        u = u.collect()
        s = np.diag(s.collect())
        v = v.collect()

        self.assertTrue(np.allclose(x_np, u @ s @ v.T))
        self.assertTrue(
            np.allclose(np.linalg.norm(u, axis=0), np.ones(u.shape[1])))
        self.assertTrue(
            np.allclose(np.linalg.norm(v, axis=0), np.ones(v.shape[1])))

    def test_svd_errors(self):
        with self.assertRaises(ValueError):
            ds.svd(ds.random_array((3, 9), (2, 2)))

        with self.assertRaises(ValueError):
            ds.svd(ds.random_array((3, 3), (3, 3)))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
