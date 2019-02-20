import unittest

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler as SKScaler

from dislib.data import load_data
from dislib.preprocessing import StandardScaler


class StandardScalerTest(unittest.TestCase):
    def test_fit_transform(self):
        """ Tests fit_transform against scikit-learn.
        """
        n_samples = 1500
        x, y = make_blobs(n_samples=n_samples, random_state=170)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        x = np.dot(x, transformation)
        dataset = load_data(x=x, y=y, subset_size=300)

        sc1 = SKScaler()
        scaled_x = sc1.fit_transform(x)
        sc2 = StandardScaler()
        sc2.fit_transform(dataset)

        self.assertTrue(np.allclose(scaled_x, dataset.samples))
        self.assertTrue(np.allclose(sc1.mean_, sc2.mean_))
        self.assertTrue(np.allclose(sc2.var_, sc2.var_))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
