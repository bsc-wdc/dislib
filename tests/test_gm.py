import unittest

import numpy as np
from numpy.random.mtrand import RandomState
from pycompss.api.api import compss_wait_on
from sklearn.datasets import make_blobs

from dislib.cluster import GaussianMixture
from dislib.data import Dataset
from dislib.data import Subset
from dislib.data import load_data


class GaussianMixtureTest(unittest.TestCase):
    def test_init_params(self):
        n_components = 2
        covariance_type = 'diag'
        tol = 1e-4
        reg_covar = 1e-5
        max_iter = 3
        init_params = 'random'
        weights_init = np.array([0.4, 0.6])
        means_init = np.array([[0, 0], [2, 3]])
        precisions_init = 'todo'
        random_state = RandomState(666)
        gm = GaussianMixture(n_components=n_components,
                             covariance_type=covariance_type,
                             tol=tol,
                             reg_covar=reg_covar,
                             max_iter=max_iter,
                             init_params=init_params,
                             weights_init=weights_init,
                             means_init=means_init,
                             precisions_init=precisions_init,
                             random_state=random_state)
        expected = (n_components, covariance_type, tol, reg_covar,
                    max_iter, init_params, weights_init, means_init,
                    precisions_init, random_state)
        real = (gm.n_components, gm.covariance_type, gm.tol, gm.reg_covar,
                gm.max_iter, gm.init_params, gm.weights_init, gm.means_init,
                gm.precisions_init, gm.random_state)
        self.assertEqual(expected, real)

    def test_fit(self):
        dataset = Dataset(n_features=2)

        dataset.append(Subset(np.array([[1, 2], [2, 1], [-3, -3]])))
        dataset.append(Subset(np.array([[-1, -2], [-2, -1], [3, 3]])))

        gm = GaussianMixture(n_components=2, random_state=666)
        gm.fit(dataset)

        expected_weights = np.array([0.5, 0.5])
        expected_means = np.array([[-2, -2], [2, 2]])
        expected_cov = np.array([[[0.66671688, 0.33338255],
                                  [0.33338255, 0.66671688]],

                                 [[0.66671688, 0.33338255],
                                  [0.33338255, 0.66671688]]])
        expected_pc = np.array([[[1.22469875, -0.70714834],
                                 [0., 1.4141944]],

                                [[1.22469875, -0.70714834],
                                 [0., 1.4141944]]])

        gm.weights_ = compss_wait_on(gm.weights_)
        gm.means_ = compss_wait_on(gm.means_)
        gm.covariances_ = compss_wait_on(gm.covariances_)
        gm.precisions_cholesky_ = compss_wait_on(gm.precisions_cholesky_)

        self.assertTrue((np.allclose(gm.weights_, expected_weights)))
        self.assertTrue((np.allclose(gm.means_, expected_means)))
        self.assertTrue((np.allclose(gm.covariances_, expected_cov)))
        self.assertTrue((np.allclose(gm.precisions_cholesky_, expected_pc)))

    def test_predict(self):
        dataset = Dataset(n_features=2)
        p1, p2, p3, p4 = [1, 2], [2, 1], [-1, -2], [-2, -1]

        dataset.append(Subset(np.array([p1, p3])))
        dataset.append(Subset(np.array([p2, p4])))

        gm = GaussianMixture(n_components=2, random_state=666)
        gm.fit(dataset)

        p5, p6 = [2, 2], [-1, -3]
        l1, l2, l3, l4, l5, l6 = gm.predict([p1, p2, p3, p4, p5, p6])

        self.assertTrue(l1 != l3)
        self.assertTrue(l1 == l2 == l5)
        self.assertTrue(l3 == l4 == l6)

    def test_fit_predict(self):
        x, y = make_blobs(n_samples=1500, random_state=170)
        x_filtered = np.vstack(
            (x[y == 0][:500], x[y == 1][:100], x[y == 2][:10]))
        y_real = np.concatenate((np.zeros(500), np.ones(100), 2*np.ones(10)))

        dataset = load_data(x_filtered, subset_size=300)

        gm = GaussianMixture(n_components=3, random_state=170)
        gm.fit_predict(dataset)
        labels = []
        for subset in dataset:
            subset = compss_wait_on(subset)
            labels.append(subset.labels)
        labels = np.concatenate(labels)
        self.assertEqual(len(labels), 610)
        accuracy = np.count_nonzero(labels == y_real) / len(labels)
        self.assertGreater(accuracy, 0.99)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
