import unittest

import numpy as np
from numpy.linalg import qr as qr_numpy
from parameterized import parameterized
from pycompss.api.api import compss_wait_on

from dislib.data.array import random_array
from dislib.math import kron
import dislib

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

        print('*****', res.shape)
        kron_np = np.kron(a, b)
        print('*****', kron_np.shape)
        print(); print()

        self.assertTrue(np.allclose(kron_np, res))

    
def main():
    unittest.main()


if __name__ == '__main__':
    print('****************************')
    print(dislib.__gpu_available__)
    print('****************************')
    main()
