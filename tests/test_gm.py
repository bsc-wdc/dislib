import io
import sys
import unittest
import warnings

import numpy as np
from numpy.random.mtrand import RandomState
from pycompss.api.api import compss_wait_on
from sklearn.datasets import make_blobs, load_iris
from sklearn.exceptions import ConvergenceWarning

import dislib as ds
from dislib.cluster import GaussianMixture


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

        x = np.array([[1, 2], [2, 1], [-3, -3], [-1, -2], [-2, -1], [3, 3]])
        ds_x = ds.array(x, block_size=(3, 2))

        gm = GaussianMixture(n_components=2, random_state=666)
        gm.fit(ds_x)

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
        x_train = np.array([[1, 2], [-1, -2], [2, 1], [-2, -1]])
        ds_x_train = ds.array(x_train, block_size=(2, 2))

        gm = GaussianMixture(n_components=2, random_state=666)
        gm.fit(ds_x_train)

        x_test = np.concatenate((x_train, [[2, 2], [-1, -3]]))
        ds_x_test = ds.array(x_test, block_size=(2, 2))
        pred = gm.predict(ds_x_test).collect()

        self.assertTrue(pred[0] != pred[1])
        self.assertTrue(pred[0] == pred[2] == pred[4])
        self.assertTrue(pred[1] == pred[3] == pred[5])

    def test_fit_predict(self):
        """Tests GaussianMixture.fit_predict()"""
        x, y = make_blobs(n_samples=1500, random_state=170)
        x_filtered = np.vstack(
            (x[y == 0][:500], x[y == 1][:100], x[y == 2][:10]))
        y_real = np.concatenate((np.zeros(500), np.ones(100), 2 * np.ones(10)))

        ds_x = ds.array(x_filtered, block_size=(300, 2))

        gm = GaussianMixture(n_components=3, random_state=170)
        pred = gm.fit_predict(ds_x).collect()

        self.assertEqual(len(pred), 610)
        accuracy = np.count_nonzero(pred == y_real) / len(pred)
        self.assertGreater(accuracy, 0.99)

    def test_check_n_components(self):
        """Tests GaussianMixture n_components validation"""
        x = ds.array([[0, 0], [0, 1], [1, 0]], block_size=(3, 2))
        with self.assertRaises(ValueError):
            gm = GaussianMixture(n_components=0)
            gm.fit(x)

    def test_check_tol(self):
        """Tests GaussianMixture tol validation"""
        x = ds.array([[0, 0], [0, 1], [1, 0]], block_size=(3, 2))
        with self.assertRaises(ValueError):
            gm = GaussianMixture(tol=-0.1)
            gm.fit(x)

    def test_check_max_iter(self):
        """Tests GaussianMixture max_iter validation"""
        x = ds.array([[0, 0], [0, 1], [1, 0]], block_size=(3, 2))
        with self.assertRaises(ValueError):
            gm = GaussianMixture(max_iter=0)
            gm.fit(x)

    def test_check_reg_covar(self):
        """Tests GaussianMixture reg_covar validation"""
        x = ds.array([[0, 0], [0, 1], [1, 0]], block_size=(3, 2))
        with self.assertRaises(ValueError):
            gm = GaussianMixture(reg_covar=-0.1)
            gm.fit(x)

    def test_check_covariance_type(self):
        """Tests GaussianMixture covariance_type validation"""
        x = ds.array([[0, 0], [0, 1], [1, 0]], block_size=(3, 2))
        with self.assertRaises(ValueError):
            gm = GaussianMixture(covariance_type='')
            gm.fit(x)

    def test_check_init_params(self):
        """Tests GaussianMixture init_params validation"""
        x = ds.array([[0, 0], [0, 1], [1, 0]], block_size=(3, 2))
        with self.assertRaises(ValueError):
            gm = GaussianMixture(init_params='')
            gm.fit(x)

    def test_check_initial_parameters(self):
        """Tests GaussianMixture initial parameters validation"""
        x = ds.array([[0, 0], [0, 1], [1, 0]], block_size=(3, 2))
        with self.assertRaises(ValueError):
            gm = GaussianMixture(weights_init=[1, 2])
            gm.fit(x)
        with self.assertRaises(ValueError):
            gm = GaussianMixture(means_init=[1, 2])
            gm.fit(x)
        with self.assertRaises(ValueError):
            gm = GaussianMixture(precisions_init=[1, 2],
                                 covariance_type='full')
            gm.fit(x)
        with self.assertRaises(ValueError):
            gm = GaussianMixture(precisions_init=[1, 2],
                                 covariance_type='tied')
            gm.fit(x)
        with self.assertRaises(ValueError):
            gm = GaussianMixture(precisions_init=[1, 2],
                                 covariance_type='diag')
            gm.fit(x)
        with self.assertRaises(ValueError):
            gm = GaussianMixture(precisions_init=[1, 2],
                                 covariance_type='spherical')
            gm.fit(x)
        with self.assertRaises(ValueError):
            gm = GaussianMixture(means_init=[[1, 2, 3]],
                                 precisions_init=[[1, 2], [3, 4]],
                                 covariance_type='tied')
            gm.fit(x)

    def test_sparse(self):
        """ Tests GaussianMixture produces the same results using dense and
        sparse data structures """
        file_ = "tests/files/libsvm/2"

        x_sparse, _ = ds.load_svmlight_file(file_, (10, 780), 780, True)
        x_dense, _ = ds.load_svmlight_file(file_, (10, 780), 780, False)

        covariance_types = 'full', 'tied', 'diag', 'spherical'

        for cov_type in covariance_types:
            gm = GaussianMixture(n_components=4, random_state=0,
                                 covariance_type=cov_type)
            labels_sparse = gm.fit_predict(x_sparse).collect()
            labels_dense = gm.fit_predict(x_dense).collect()
            self.assertTrue(np.array_equal(labels_sparse, labels_dense))

    def test_init_random(self):
        """ Tests GaussianMixture random initialization """
        x = ds.random_array((50, 3), (10, 3), random_state=0)
        gm = GaussianMixture(init_params='random', n_components=4,
                             arity=2, random_state=170)
        gm.fit(x)
        self.assertGreater(gm.n_iter, 5)

    def test_covariance_types(self):
        """ Tests GaussianMixture covariance types """
        np.random.seed(0)
        n_samples = 600
        n_features = 2

        def create_anisotropic_dataset():
            """Create dataset with 2 anisotropic gaussians of different
            weight"""
            n0 = 2 * n_samples // 3
            n1 = n_samples // 3
            x0 = np.random.normal(size=(n0, n_features))
            x1 = np.random.normal(size=(n1, n_features))
            transformation = [[0.6, -0.6], [-0.4, 0.8]]
            x0 = np.dot(x0, transformation)
            x1 = np.dot(x1, transformation) + [0, 3]
            x = np.concatenate((x0, x1))
            y = np.concatenate((np.zeros(n0), np.ones(n1)))
            return x, y

        def create_spherical_blobs_dataset():
            """Create dataset with 2 spherical gaussians of different weight,
            variance and position"""
            n0 = 2 * n_samples // 3
            n1 = n_samples // 3
            x0 = np.random.normal(size=(n0, 2), scale=0.5, loc=[2, 0])
            x1 = np.random.normal(size=(n1, 2), scale=2.5)
            x = np.concatenate((x0, x1))
            y = np.concatenate((np.zeros(n0), np.ones(n1)))
            return x, y

        def create_uncorrelated_dataset():
            """Create dataset with 2 gaussians forming a cross of uncorrelated
            variables"""
            n0 = 2 * n_samples // 3
            n1 = n_samples // 3
            x0 = np.random.normal(size=(n0, n_features))
            x1 = np.random.normal(size=(n1, n_features))
            x0 = np.dot(x0, [[1.2, 0], [0, 0.5]]) + [0, 3]
            x1 = np.dot(x1, [[0.4, 0], [0, 2.5]]) + [1, 0]
            x = np.concatenate((x0, x1))
            y = np.concatenate((np.zeros(n0), np.ones(n1)))
            return x, y

        def create_correlated_dataset():
            """Create dataset with 2 gaussians forming a cross of correlated
            variables"""
            x, y = create_uncorrelated_dataset()
            x = np.dot(x, [[1, 1], [-1, 1]])
            return x, y

        datasets = {'aniso': create_anisotropic_dataset(),
                    'blobs': create_spherical_blobs_dataset(),
                    'uncorr': create_uncorrelated_dataset(),
                    'corr': create_correlated_dataset()}
        real_labels = {k: v[1] for k, v in datasets.items()}
        for k, v in datasets.items():
            datasets[k] = ds.array(v[0], block_size=(200, v[0].shape[1]))

        covariance_types = 'full', 'tied', 'diag', 'spherical'

        def compute_accuracy(real, predicted):
            """ Computes classification accuracy for binary (0/1) labels"""
            equal_labels = np.count_nonzero(predicted == real)
            equal_ratio = equal_labels / len(real)
            return max(equal_ratio, 1 - equal_ratio)

        pred_labels = {}
        for cov_type in covariance_types:
            pred_labels[cov_type] = {}
            gm = GaussianMixture(n_components=2, covariance_type=cov_type,
                                 random_state=0)
            for k, x in datasets.items():
                pred_labels[cov_type][k] = gm.fit_predict(x)
        accuracy = {}
        for cov_type in covariance_types:
            accuracy[cov_type] = {}
            for k, pred in pred_labels[cov_type].items():
                accuracy[cov_type][k] = \
                    compute_accuracy(real_labels[k], pred.collect())

        # Covariance type 'full'.
        # Assert good accuracy in all tested datasets.
        self.assertGreater(accuracy['full']['aniso'], 0.9)
        self.assertGreater(accuracy['full']['blobs'], 0.9)
        self.assertGreater(accuracy['full']['uncorr'], 0.9)
        self.assertGreater(accuracy['full']['corr'], 0.9)

        # Covariance type 'tied'.
        # Assert good accuracy only for 'aniso'.
        self.assertGreater(accuracy['tied']['aniso'], 0.9)
        self.assertLess(accuracy['tied']['blobs'], 0.9)
        self.assertLess(accuracy['tied']['uncorr'], 0.9)
        self.assertLess(accuracy['tied']['corr'], 0.9)

        # Covariance type 'diag'.
        # Assert good accuracy only for 'blobs' and 'uncorr'.
        self.assertLess(accuracy['diag']['aniso'], 0.9)
        self.assertGreater(accuracy['diag']['blobs'], 0.9)
        self.assertGreater(accuracy['diag']['uncorr'], 0.9)
        self.assertLess(accuracy['diag']['corr'], 0.9)

        # Covariance type 'spherical'.
        # Assert good accuracy only for 'blobs'.
        self.assertLess(accuracy['spherical']['aniso'], 0.9)
        self.assertGreater(accuracy['spherical']['blobs'], 0.9)
        self.assertLess(accuracy['spherical']['uncorr'], 0.9)
        self.assertLess(accuracy['spherical']['corr'], 0.9)

        # For a graphical plot of the results of this comparision, see
        # examples/gm_covariance_types_comparision.py

    def test_verbose(self):
        """ Tests GaussianMixture verbose mode prints text """
        x = ds.array([[0, 0], [0, 1], [1, 0]], (3, 2))
        gm = GaussianMixture(verbose=True, max_iter=2)

        saved_stdout = sys.stdout
        try:
            sys.stdout = io.StringIO()

            # Call code that has to print
            gm.fit(x)

            captured_output = sys.stdout.getvalue()
        finally:
            sys.stdout = saved_stdout

        self.assertTrue(len(captured_output) > 0)

    def test_not_converged_warning(self):
        """ Tests GaussianMixture warns when not converged """
        with self.assertWarns(ConvergenceWarning):
            x, _ = load_iris(return_X_y=True)
            x_ds = ds.array(x, (75, 4))
            gm = GaussianMixture(max_iter=1)
            gm.fit(x_ds)

    def test_fit_predict_vs_fit_and_predict(self):
        """Tests GaussianMixture fit_predict() eq. fit() and predict() for both
        converged and not converged runs (and a fixed random_state)."""
        x0 = np.random.normal(size=(1000, 2))
        x1 = np.random.normal(size=(2000, 2))
        x0 = np.dot(x0, [[1.2, 1], [0, 0.5]]) + [0, 3]
        x1 = np.dot(x1, [[0.4, 0], [1, 2.5]]) + [1, 0]
        x = np.concatenate((x0, x1))
        x_ds = ds.array(x, (1500, 2))

        # We check the cases with and without convergence
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for max_iter, converges in ((5, False), (100, True)):
                gm1 = GaussianMixture(n_components=2, max_iter=max_iter,
                                      random_state=0)
                gm1.fit(x_ds)
                labels1 = gm1.predict(x_ds)

                gm2 = GaussianMixture(n_components=2, max_iter=max_iter,
                                      random_state=0)
                labels2 = gm2.fit_predict(x_ds)

                self.assertTrue(np.all(labels1.collect() == labels2.collect()))
                self.assertEqual(gm1.n_iter, gm2.n_iter)
                self.assertEqual(converges, gm1.converged_)
                self.assertEqual(gm1.converged_, gm2.converged_)
                self.assertEqual(gm1.lower_bound_, gm2.lower_bound_)

                gm1.weights_ = compss_wait_on(gm1.weights_)
                gm1.means_ = compss_wait_on(gm1.means_)
                gm1.covariances_ = compss_wait_on(gm1.covariances_)
                gm2.weights_ = compss_wait_on(gm2.weights_)
                gm2.means_ = compss_wait_on(gm2.means_)
                gm2.covariances_ = compss_wait_on(gm2.covariances_)

                self.assertTrue(np.all(gm1.weights_ == gm2.weights_))
                self.assertTrue(np.all(gm1.means_ == gm2.means_))
                self.assertTrue(np.all(gm1.covariances_ == gm2.covariances_))

    def test_means_init_and_weights_init(self):
        """ Tests GaussianMixture means_init and weights_init parameters """
        x, _ = load_iris(return_X_y=True)
        x_ds = ds.array(x, (75, 4))
        weights_init = [1 / 3, 1 / 3, 1 / 3]
        means_init = np.array([[5, 3, 2, 0],
                               [6, 3, 4, 1],
                               [7, 3, 6, 2]])
        gm = GaussianMixture(random_state=0, n_components=3,
                             weights_init=weights_init, means_init=means_init)
        gm.fit(x_ds)
        self.assertTrue(gm.converged_)

    def test_precisions_init_full(self):
        """ Tests GaussianMixture with precisions_init='full' """
        x, _ = load_iris(return_X_y=True)
        x_ds = ds.array(x, (75, 4))
        weights_init = [1 / 3, 1 / 3, 1 / 3]
        means_init = [[5, 3, 2, 0],
                      [6, 3, 4, 1],
                      [7, 3, 6, 2]]
        np.random.seed(0)
        rand_matrices = [np.random.rand(4, 4) for _ in range(3)]
        precisions_init = [np.matmul(r, r.T) for r in rand_matrices]

        gm = GaussianMixture(covariance_type='full', random_state=0,
                             n_components=3, weights_init=weights_init,
                             means_init=means_init,
                             precisions_init=precisions_init)
        gm.fit(x_ds)
        self.assertTrue(gm.converged_)

    def test_precisions_init_tied(self):
        """ Tests GaussianMixture with precisions_init='tied' """
        x, _ = load_iris(return_X_y=True)
        x_ds = ds.array(x, (75, 4))
        weights_init = [1 / 3, 1 / 3, 1 / 3]
        means_init = [[5, 3, 2, 0],
                      [6, 3, 4, 1],
                      [7, 3, 6, 2]]
        np.random.seed(0)
        rand_matrix = np.random.rand(4, 4)
        precisions_init = np.matmul(rand_matrix, rand_matrix.T)

        gm = GaussianMixture(covariance_type='tied', random_state=0,
                             n_components=3, weights_init=weights_init,
                             means_init=means_init,
                             precisions_init=precisions_init)
        gm.fit(x_ds)
        self.assertTrue(gm.converged_)

    def test_precisions_init_diag(self):
        """ Tests GaussianMixture with precisions_init='diag' """
        x, _ = load_iris(return_X_y=True)
        x_ds = ds.array(x, (75, 4))
        weights_init = np.array([1 / 3, 1 / 3, 1 / 3])
        means_init = np.array([[5, 3, 2, 0],
                               [6, 3, 4, 1],
                               [7, 3, 6, 2]])
        np.random.seed(0)
        precisions_init = np.random.rand(3, 4) * 2

        gm = GaussianMixture(covariance_type='diag', random_state=0,
                             n_components=3, weights_init=weights_init,
                             means_init=means_init,
                             precisions_init=precisions_init)
        gm.fit(x_ds)
        self.assertTrue(gm.converged_)

    def test_precisions_init_spherical(self):
        """ Tests GaussianMixture with precisions_init='spherical' """
        x, _ = load_iris(return_X_y=True)
        x_ds = ds.array(x, (75, 4))
        weights_init = [1 / 3, 1 / 3, 1 / 3]
        means_init = np.array([[5, 3, 2, 0],
                               [6, 3, 4, 1],
                               [7, 3, 6, 2]])
        np.random.seed(0)
        precisions_init = np.random.rand(3) * 2

        gm = GaussianMixture(covariance_type='spherical', random_state=0,
                             n_components=3, weights_init=weights_init,
                             means_init=means_init,
                             precisions_init=precisions_init)
        gm.fit(x_ds)
        self.assertTrue(gm.converged_)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
