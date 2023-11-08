import unittest
from dislib.decomposition import random_svd
import numpy as np
import dislib as ds


def create_matrix(m, n):
    min_exp = -15
    max_exp = 2
    singular_values = np.logspace(min_exp, max_exp, num=min(m, n))
    A = np.zeros((m, n))
    np.fill_diagonal(A, singular_values)
    U, s, V = np.linalg.svd(A, full_matrices=False)
    A = U.dot(np.diag(s)).dot(V)
    return A, singular_values[::-1]


class RandomSVDTest(unittest.TestCase):

    def test_random(self):
        A, sing_values = create_matrix(10000, 50)
        A = ds.data.array(A, block_size=(2000, 10))
        U, svds, V = random_svd(A, iters=5, epsilon=0.5, tol=1e-3,
                                nsv=5, k=20, verbose=True)
        self.assertTrue(svds.shape == (20, 20))
        self.assertTrue(U.shape == (A.shape[0], 20))
        self.assertTrue(V.shape == (A.shape[1], 20))
        svds = svds.collect().diagonal()

    def test_convergence_random(self):
        A = ds.data.random_array(shape=(10000, 100), block_size=(2000, 20),
                                 random_state=0)
        U, svds, V = random_svd(A, iters=5, epsilon=0.4, tol=1e-3,
                                nsv=20, k=40)
        svds = svds.collect().diagonal()
        self.assertTrue(len(svds) == 60)

    def test_exceptions(self):
        A = ds.data.random_array(shape=(10000, 100), block_size=(2000, 20),
                                 random_state=0)
        with self.assertRaises(ValueError):
            random_svd(A, iters=10, epsilon=0.4, tol=1e-3, nsv=400, k=40)
        with self.assertRaises(ValueError):
            random_svd(A, iters=10, epsilon=0.4, tol=1e-3, nsv=20, k=10)
        with self.assertRaises(ValueError):
            random_svd(A, iters=10, epsilon=0.4, tol=1e-3, nsv=10, k=25)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
