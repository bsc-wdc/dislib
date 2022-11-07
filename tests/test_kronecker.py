import unittest

import numpy as np
from parameterized import parameterized

from dislib.data.array import random_array
from dislib.math import kron


class KroneckerTest(unittest.TestCase):

    @parameterized.expand([
        (3, 3, 3), (3, 3, 4), (4, 4, 2), (4, 4, 3),
    ])
    def test_qr(self, m_size, n_size, b_size):
        np.set_printoptions(precision=2)
        np.random.seed(8)

        shape = (m_size * b_size, n_size * b_size)
        a = random_array(shape, (b_size, b_size))
        b = random_array(shape, (b_size, b_size))

        res_ds = kron(a, b)

        a = a.collect()
        b = b.collect()
        res = res_ds.collect()

        kron_np = np.kron(a, b)

        self.assertTrue(np.allclose(kron_np, res))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
