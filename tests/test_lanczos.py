import unittest
from dislib.decomposition import lanczos_svd
import numpy as np
import dislib as ds


def create_matrix(m, n):
    min_exp = -14
    max_exp = 3
    singular_values = np.logspace(min_exp, max_exp, num=min(m, n))
    A = np.zeros((m, n))
    np.fill_diagonal(A, singular_values)
    U, s, V = np.linalg.svd(A, full_matrices=False)
    A = U.dot(np.diag(s)).dot(V)
    return A, singular_values[::-1]


class LanczosSVDTest(unittest.TestCase):

    def test_lanczos(self):
        A, sing_values = create_matrix(10000, 100)
        A = ds.data.array(A, block_size=(2000, 5))
        U, svds, V = lanczos_svd(A, 20, 5, 15, 10, 0.000001, 0.5, 2)
        self.assertTrue(svds.shape == (15, 15))
        self.assertTrue(U.shape == (A.shape[0], 15))
        self.assertTrue(V.shape == (A.shape[1], 15))
        svds = svds.collect().diagonal()
        self.assertTrue(np.allclose(sing_values[:10], svds[:10]))

    def test_convergence_lanczos(self):
        A = ds.data.random_array(shape=(10000, 100), block_size=(2000, 20),
                                 random_state=0)
        U, svds, V = lanczos_svd(A, 60, 20, 40, 30, 0.0001, 0.5, 2)
        svds = svds.collect().diagonal()
        self.assertTrue(len(svds) == 40)
        U, svds, V = lanczos_svd(A, 60, 20, 40, 30, 0.0001, 0.4, 2)
        svds = svds.collect().diagonal()
        self.assertTrue(len(svds) == 60)

    def test_exceptions(self):
        A = ds.data.random_array(shape=(10000, 100), block_size=(2000, 20),
                                 random_state=0)
        with self.assertRaises(ValueError):
            lanczos_svd(A, 120, 20, 120, 110, 0.0001, 0.0001, 2)
        with self.assertRaises(ValueError):
            lanczos_svd(A, 60, 20, 20, 40, 0.0001, 0.0001, 2)
        with self.assertRaises(ValueError):
            lanczos_svd(A, 40, 20, 40, 20, 0.0001, 0.0001, 2)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
