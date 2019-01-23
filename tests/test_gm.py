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
        """Tests that GaussianMixture params are set"""
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
        """Tests GaussianMixture.fit()"""
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
        """Tests GaussianMixture.predict()"""
        dataset = Dataset(n_features=2)
        p0, p1, p2, p3 = [1, 2], [-1, -2], [2, 1], [-2, -1]

        dataset.append(Subset(np.array([p0, p1])))
        dataset.append(Subset(np.array([p2, p3])))

        gm = GaussianMixture(n_components=2, random_state=666)
        gm.fit(dataset)

        p4, p5 = [2, 2], [-1, -3]
        dataset.append(Subset(np.array([p4, p5])))
        gm.predict(dataset)
        prediction = dataset.labels

        self.assertTrue(prediction[0] != prediction[1])
        self.assertTrue(prediction[0] == prediction[2] == prediction[4])
        self.assertTrue(prediction[1] == prediction[3] == prediction[5])

    def test_fit_predict(self):
        """Tests GaussianMixture.fit_predict()"""
        x, y = make_blobs(n_samples=1500, random_state=170)
        x_filtered = np.vstack(
            (x[y == 0][:500], x[y == 1][:100], x[y == 2][:10]))
        y_real = np.concatenate((np.zeros(500), np.ones(100), 2*np.ones(10)))

        dataset = load_data(x_filtered, subset_size=300)

        gm = GaussianMixture(n_components=3, random_state=170)
        gm.fit_predict(dataset)
        prediction = dataset.labels

        self.assertEqual(len(prediction), 610)
        accuracy = np.count_nonzero(prediction == y_real) / len(prediction)
        self.assertGreater(accuracy, 0.99)

    def test_check_n_components(self):
        """Tests GaussianMixture n_components validation"""
        x = np.array([[0, 0], [0, 1], [1, 0]])
        dataset = load_data(x, subset_size=10)
        with self.assertRaises(ValueError):
            gm = GaussianMixture(n_components=0)
            gm.fit(dataset)

    def test_check_tol(self):
        """Tests GaussianMixture tol validation"""
        x = np.array([[0, 0], [0, 1], [1, 0]])
        dataset = load_data(x, subset_size=10)
        with self.assertRaises(ValueError):
            gm = GaussianMixture(tol=-0.1)
            gm.fit(dataset)

    def test_check_max_iter(self):
        """Tests GaussianMixture max_iter validation"""
        x = np.array([[0, 0], [0, 1], [1, 0]])
        dataset = load_data(x, subset_size=10)
        with self.assertRaises(ValueError):
            gm = GaussianMixture(max_iter=0)
            gm.fit(dataset)

    def test_check_reg_covar(self):
        """Tests GaussianMixture reg_covar validation"""
        x = np.array([[0, 0], [0, 1], [1, 0]])
        dataset = load_data(x, subset_size=10)
        with self.assertRaises(ValueError):
            gm = GaussianMixture(reg_covar=-0.1)
            gm.fit(dataset)

    def test_check_covariance_type(self):
        """Tests GaussianMixture covariance_type validation"""
        x = np.array([[0, 0], [0, 1], [1, 0]])
        dataset = load_data(x, subset_size=10)
        with self.assertRaises(ValueError):
            gm = GaussianMixture(covariance_type='')
            gm.fit(dataset)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
