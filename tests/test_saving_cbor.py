import unittest

import numpy as np
from numpy.random.mtrand import RandomState
from scipy.sparse import csr_matrix
from sklearn import datasets
from sklearn.cluster import KMeans as SKMeans
from sklearn.metrics import r2_score
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs, load_iris

import dislib as ds
from dislib.cluster import KMeans
from dislib.cluster import GaussianMixture
from dislib.classification import CascadeSVM
from dislib.classification import RandomForestClassifier
from dislib.regression import Lasso
from dislib.regression import LinearRegression
from dislib.recommendation import ALS
from dislib.utils import save_model, load_model

from pycompss.api.api import compss_wait_on


class KMeansSavingTestCBOR(unittest.TestCase):
    filepath = "tests/files/saving/kmeans.cbor"

    def test_init_params_kmeans(self):
        """Tests that KMeans correctly sets the initialization
        parameters"""
        n_clusters = 2
        max_iter = 1
        tol = 1e-4
        seed = 666
        arity = 2
        init = "random"

        km = KMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            tol=tol,
            arity=arity,
            random_state=seed,
        )
        save_model(km, self.filepath, save_format="cbor")
        km2 = load_model(self.filepath, load_format="cbor")

        expected = (n_clusters, init, max_iter, tol, arity)
        real = (km.n_clusters, km.init, km.max_iter, km.tol, km.arity)
        real2 = (km2.n_clusters, km2.init, km2.max_iter, km2.tol, km2.arity)
        self.assertEqual(expected, real)
        self.assertEqual(expected, real2)

    def test_fit_kmeans(self):
        """Tests that the fit method returns the expected centers using toy
        data.
        """
        arr = np.array([[1, 2], [2, 1], [-1, -2], [-2, -1]])
        x = ds.array(arr, block_size=(2, 2))

        km = KMeans(n_clusters=2, random_state=666, verbose=False)
        km.fit(x)

        expected_centers = np.array([[1.5, 1.5], [-1.5, -1.5]])

        save_model(km, self.filepath, save_format="cbor")
        km2 = load_model(self.filepath, load_format="cbor")

        self.assertTrue((km.centers == expected_centers).all())
        self.assertTrue((km2.centers == expected_centers).all())

    def test_predict_kmeans(self):
        """Tests that labels are correctly predicted using toy data."""
        p1, p2, p3, p4 = [1, 2], [2, 1], [-1, -2], [-2, -1]

        arr1 = np.array([p1, p2, p3, p4])
        x = ds.array(arr1, block_size=(2, 2))

        km = KMeans(n_clusters=2, random_state=666)
        km.fit(x)

        save_model(km, self.filepath, save_format="cbor")
        km2 = load_model(self.filepath, load_format="cbor")

        p5, p6 = [10, 10], [-10, -10]

        arr2 = np.array([p1, p2, p3, p4, p5, p6])
        x_test = ds.array(arr2, block_size=(2, 2))

        labels = km.predict(x_test).collect()
        labels2 = km2.predict(x_test).collect()
        expected_labels = np.array([0, 0, 1, 1, 0, 1])

        self.assertTrue(np.array_equal(labels, expected_labels))
        self.assertTrue(np.array_equal(labels2, expected_labels))

    def test_fit_predict_kmeans(self):
        """Tests fit_predict."""
        x, y = make_blobs(n_samples=1500, random_state=170)
        x_filtered = np.vstack(
            (x[y == 0][:500], x[y == 1][:100], x[y == 2][:10])
        )

        x_train = ds.array(x_filtered, block_size=(300, 2))

        kmeans = KMeans(n_clusters=3, random_state=170)
        labels = kmeans.fit_predict(x_train).collect()

        save_model(kmeans, self.filepath, save_format="cbor")
        kmeans = load_model(self.filepath, load_format="cbor")

        skmeans = SKMeans(n_clusters=3, random_state=170)
        sklabels = skmeans.fit_predict(x_filtered)

        centers = np.array(
            [
                [-8.941375656533449, -5.481371322614891],
                [-4.524023204953875, 0.06235042593214654],
                [2.332994701667008, 0.37681003933082696],
            ]
        )

        self.assertTrue(np.allclose(centers, kmeans.centers))
        self.assertTrue(np.allclose(labels, sklabels))

    def test_sparse_kmeans(self):
        """Tests K-means produces the same results using dense and sparse
        data structures."""
        file_ = "tests/files/libsvm/2"

        x_sp, _ = ds.load_svmlight_file(file_, (10, 300), 780, True)
        x_ds, _ = ds.load_svmlight_file(file_, (10, 300), 780, False)

        kmeans = KMeans(random_state=170)
        kmeans.fit(x_sp)

        save_model(kmeans, self.filepath, save_format="cbor")
        kmeans2 = load_model(self.filepath, load_format="cbor")

        y_sparse = kmeans.predict(x_sp).collect()
        y_sparse2 = kmeans2.predict(x_sp).collect()

        sparse_c = kmeans.centers.toarray()
        sparse_c2 = kmeans2.centers.toarray()

        kmeans = KMeans(random_state=170)

        y_dense = kmeans.fit_predict(x_ds).collect()
        dense_c = kmeans.centers

        self.assertTrue(np.allclose(sparse_c, dense_c))
        self.assertTrue(np.allclose(sparse_c2, dense_c))
        self.assertTrue(np.array_equal(y_sparse, y_dense))
        self.assertTrue(np.array_equal(y_sparse2, y_dense))

    def test_init_kmeans(self):
        # With dense data
        x, y = make_blobs(n_samples=1500, random_state=170)
        x_filtered = np.vstack(
            (x[y == 0][:500], x[y == 1][:100], x[y == 2][:10])
        )
        x_train = ds.array(x_filtered, block_size=(300, 2))

        init = np.random.random((5, 2))
        km = KMeans(n_clusters=5, init=init)
        km.fit(x_train)

        save_model(km, self.filepath, save_format="cbor")
        km2 = load_model(self.filepath, load_format="cbor")

        self.assertTrue(np.array_equal(km.init, init))
        self.assertTrue(np.array_equal(km2.init, init))
        self.assertFalse(np.array_equal(km.centers, init))
        self.assertFalse(np.array_equal(km2.centers, init))

        # With sparse data
        x_sp = ds.array(csr_matrix(x_filtered), block_size=(300, 2))
        init = csr_matrix(np.random.random((5, 2)))

        km = KMeans(n_clusters=5, init=init)
        km.fit(x_sp)

        save_model(km, self.filepath, save_format="cbor")
        km2 = load_model(self.filepath, load_format="cbor")

        self.assertTrue(np.array_equal(km.init.toarray(), init.toarray()))
        self.assertTrue(np.array_equal(km2.init.toarray(), init.toarray()))
        self.assertFalse(np.array_equal(km.centers.toarray(), init.toarray()))
        self.assertFalse(np.array_equal(km2.centers.toarray(), init.toarray()))


class GaussianMixtureSavingTestCBOR(unittest.TestCase):
    filepath = "tests/files/saving/gm.cbor"

    def test_init_params(self):
        """Tests that GaussianMixture params are set"""
        n_components = 2
        covariance_type = "diag"
        tol = 1e-4
        reg_covar = 1e-5
        max_iter = 3
        init_params = "random"
        weights_init = np.array([0.4, 0.6])
        means_init = np.array([[0, 0], [2, 3]])
        precisions_init = "todo"
        random_state = RandomState(666)
        gm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            init_params=init_params,
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init,
            random_state=random_state,
        )

        save_model(gm, self.filepath, save_format="cbor")
        gm2 = load_model(self.filepath, load_format="cbor")

        real = (
            gm.n_components,
            gm.covariance_type,
            gm.tol,
            gm.reg_covar,
            gm.max_iter,
            gm.init_params,
            gm.weights_init.tolist(),
            gm.means_init.tolist(),
            gm.precisions_init,
            *[
                list(x) if isinstance(x, np.ndarray) else x
                for x in gm.random_state.get_state()
            ],
        )
        real2 = (
            gm2.n_components,
            gm2.covariance_type,
            gm2.tol,
            gm2.reg_covar,
            gm2.max_iter,
            gm2.init_params,
            gm2.weights_init.tolist(),
            gm2.means_init.tolist(),
            gm2.precisions_init,
            *[
                list(x) if isinstance(x, np.ndarray) else x
                for x in gm2.random_state.get_state()
            ],
        )

        self.assertEqual(real, real2)

    def test_fit(self):
        """Tests GaussianMixture.fit()"""

        x = np.array([[1, 2], [2, 1], [-3, -3], [-1, -2], [-2, -1], [3, 3]])
        ds_x = ds.array(x, block_size=(3, 2))

        gm = GaussianMixture(n_components=2, random_state=666)
        gm.fit(ds_x)

        expected_weights = np.array([0.5, 0.5])
        expected_means = np.array([[-2, -2], [2, 2]])
        expected_cov = np.array(
            [
                [[0.66671688, 0.33338255], [0.33338255, 0.66671688]],
                [[0.66671688, 0.33338255], [0.33338255, 0.66671688]],
            ]
        )
        expected_pc = np.array(
            [
                [[1.22469875, -0.70714834], [0.0, 1.4141944]],
                [[1.22469875, -0.70714834], [0.0, 1.4141944]],
            ]
        )

        save_model(gm, self.filepath, save_format="cbor")
        gm2 = load_model(self.filepath, load_format="cbor")

        gm.weights_ = compss_wait_on(gm.weights_)
        gm.means_ = compss_wait_on(gm.means_)
        gm.covariances_ = compss_wait_on(gm.covariances_)
        gm.precisions_cholesky_ = compss_wait_on(gm.precisions_cholesky_)

        gm2.weights_ = compss_wait_on(gm2.weights_)
        gm2.means_ = compss_wait_on(gm2.means_)
        gm2.covariances_ = compss_wait_on(gm2.covariances_)
        gm2.precisions_cholesky_ = compss_wait_on(gm2.precisions_cholesky_)

        self.assertTrue((np.allclose(gm.weights_, expected_weights)))
        self.assertTrue((np.allclose(gm.means_, expected_means)))
        self.assertTrue((np.allclose(gm.covariances_, expected_cov)))
        self.assertTrue((np.allclose(gm.precisions_cholesky_, expected_pc)))

        self.assertTrue((np.allclose(gm2.weights_, expected_weights)))
        self.assertTrue((np.allclose(gm2.means_, expected_means)))
        self.assertTrue((np.allclose(gm2.covariances_, expected_cov)))
        self.assertTrue((np.allclose(gm2.precisions_cholesky_, expected_pc)))

    def test_predict(self):
        """Tests GaussianMixture.predict()"""
        x_train = np.array([[1, 2], [-1, -2], [2, 1], [-2, -1]])
        ds_x_train = ds.array(x_train, block_size=(2, 2))

        gm = GaussianMixture(n_components=2, random_state=666)
        gm.fit(ds_x_train)

        save_model(gm, self.filepath, save_format="cbor")
        gm2 = load_model(self.filepath, load_format="cbor")

        x_test = np.concatenate((x_train, [[2, 2], [-1, -3]]))
        ds_x_test = ds.array(x_test, block_size=(2, 2))
        pred = gm.predict(ds_x_test).collect()
        pred2 = gm2.predict(ds_x_test).collect()

        self.assertTrue(pred[0] != pred[1])
        self.assertTrue(pred[0] == pred[2] == pred[4])
        self.assertTrue(pred[1] == pred[3] == pred[5])

        self.assertTrue(pred2[0] != pred2[1])
        self.assertTrue(pred2[0] == pred2[2] == pred2[4])
        self.assertTrue(pred2[1] == pred2[3] == pred2[5])

    def test_fit_predict(self):
        """Tests GaussianMixture.fit_predict()"""
        x, y = make_blobs(n_samples=1500, random_state=170)
        x_filtered = np.vstack(
            (x[y == 0][:500], x[y == 1][:100], x[y == 2][:10])
        )
        y_real = np.concatenate((np.zeros(500), np.ones(100), 2 * np.ones(10)))

        ds_x = ds.array(x_filtered, block_size=(300, 2))

        gm = GaussianMixture(n_components=3, random_state=170)
        pred = gm.fit_predict(ds_x).collect()

        save_model(gm, self.filepath, save_format="cbor")
        gm2 = load_model(self.filepath, load_format="cbor")

        pred2 = gm2.predict(ds_x).collect()

        self.assertEqual(len(pred), 610)
        accuracy = np.count_nonzero(pred == y_real) / len(pred)
        self.assertGreater(accuracy, 0.99)

        self.assertEqual(len(pred2), 610)
        accuracy2 = np.count_nonzero(pred2 == y_real) / len(pred2)
        self.assertGreater(accuracy2, 0.99)

    def test_sparse(self):
        """Tests GaussianMixture produces the same results using dense and
        sparse data structures"""
        file_ = "tests/files/libsvm/2"

        x_sparse, _ = ds.load_svmlight_file(file_, (10, 780), 780, True)
        x_dense, _ = ds.load_svmlight_file(file_, (10, 780), 780, False)

        covariance_types = "full", "tied", "diag", "spherical"

        for cov_type in covariance_types:
            gm = GaussianMixture(
                n_components=4, random_state=0, covariance_type=cov_type
            )
            gm.fit(x_sparse)
            save_model(gm, self.filepath, save_format="cbor")
            gm2 = load_model(self.filepath, load_format="cbor")
            labels_sparse = gm.predict(x_sparse).collect()
            labels_sparse2 = gm2.predict(x_sparse).collect()

            gm = GaussianMixture(
                n_components=4, random_state=0, covariance_type=cov_type
            )
            gm.fit(x_dense)
            save_model(gm, self.filepath, save_format="cbor")
            gm2 = load_model(self.filepath, load_format="cbor")
            labels_dense = gm.predict(x_dense).collect()
            labels_dense2 = gm2.predict(x_dense).collect()

            self.assertTrue(np.array_equal(labels_sparse, labels_sparse2))
            self.assertTrue(np.array_equal(labels_sparse, labels_dense))
            self.assertTrue(np.array_equal(labels_sparse2, labels_dense2))

    def test_init_random(self):
        """Tests GaussianMixture random initialization"""
        x = ds.random_array((50, 3), (10, 3), random_state=0)
        gm = GaussianMixture(
            init_params="random", n_components=4, arity=2, random_state=170
        )
        gm.fit(x)
        save_model(gm, self.filepath, save_format="cbor")
        gm2 = load_model(self.filepath, load_format="cbor")
        self.assertGreater(gm.n_iter, 5)
        self.assertGreater(gm2.n_iter, 5)

    def test_means_init_and_weights_init(self):
        """Tests GaussianMixture means_init and weights_init parameters"""
        x, _ = load_iris(return_X_y=True)
        x_ds = ds.array(x, (75, 4))
        weights_init = [1 / 3, 1 / 3, 1 / 3]
        means_init = np.array([[5, 3, 2, 0], [6, 3, 4, 1], [7, 3, 6, 2]])
        gm = GaussianMixture(
            random_state=0,
            n_components=3,
            weights_init=weights_init,
            means_init=means_init,
        )
        gm.fit(x_ds)
        save_model(gm, self.filepath, save_format="cbor")
        gm2 = load_model(self.filepath, load_format="cbor")
        self.assertTrue(gm.converged_)
        self.assertTrue(gm2.converged_)


class CSVMSavingTestCBOR(unittest.TestCase):
    filepath = "tests/files/saving/csvm.cbor"

    def test_init_params(self):
        """Test constructor parameters"""
        cascade_arity = 3
        max_iter = 1
        tol = 1e-4
        kernel = "rbf"
        c = 2
        gamma = 0.1
        check_convergence = True
        seed = 666
        verbose = False

        csvm = CascadeSVM(
            cascade_arity=cascade_arity,
            max_iter=max_iter,
            tol=tol,
            kernel=kernel,
            c=c,
            gamma=gamma,
            check_convergence=check_convergence,
            random_state=seed,
            verbose=verbose,
        )
        save_model(csvm, self.filepath, save_format="cbor")
        csvm2 = load_model(self.filepath, load_format="cbor")

        self.assertEqual(csvm.cascade_arity, cascade_arity)
        self.assertEqual(csvm.max_iter, max_iter)
        self.assertEqual(csvm.tol, tol)
        self.assertEqual(csvm.kernel, kernel)
        self.assertEqual(csvm.c, c)
        self.assertEqual(csvm.gamma, gamma)
        self.assertEqual(csvm.check_convergence, check_convergence)
        self.assertEqual(csvm.random_state, seed)
        self.assertEqual(csvm.verbose, verbose)

        self.assertEqual(csvm2.cascade_arity, cascade_arity)
        self.assertEqual(csvm2.max_iter, max_iter)
        self.assertEqual(csvm2.tol, tol)
        self.assertEqual(csvm2.kernel, kernel)
        self.assertEqual(csvm2.c, c)
        self.assertEqual(csvm2.gamma, gamma)
        self.assertEqual(csvm2.check_convergence, check_convergence)
        self.assertEqual(csvm2.random_state, seed)
        self.assertEqual(csvm2.verbose, verbose)

    def test_fit_private_params(self):
        kernel = "rbf"
        c = 2
        gamma = 0.1
        seed = 666
        file_ = "tests/files/libsvm/2"

        x, y = ds.load_svmlight_file(file_, (10, 300), 780, False)
        csvm = CascadeSVM(kernel=kernel, c=c, gamma=gamma, random_state=seed)
        csvm.fit(x, y)
        save_model(csvm, self.filepath, save_format="cbor")
        csvm2 = load_model(self.filepath, load_format="cbor")
        self.assertEqual(csvm._clf_params["kernel"], kernel)
        self.assertEqual(csvm._clf_params["C"], c)
        self.assertEqual(csvm._clf_params["gamma"], gamma)
        self.assertEqual(csvm2._clf_params["kernel"], kernel)
        self.assertEqual(csvm2._clf_params["C"], c)
        self.assertEqual(csvm2._clf_params["gamma"], gamma)

        kernel, c = "linear", 0.3
        csvm = CascadeSVM(kernel=kernel, c=c, random_state=seed)
        csvm.fit(x, y)
        save_model(csvm, self.filepath, save_format="cbor")
        csvm2 = load_model(self.filepath, load_format="cbor")
        self.assertEqual(csvm._clf_params["kernel"], kernel)
        self.assertEqual(csvm._clf_params["C"], c)
        self.assertEqual(csvm2._clf_params["kernel"], kernel)
        self.assertEqual(csvm2._clf_params["C"], c)

        # # check for exception when incorrect kernel is passed
        # self.assertRaises(AttributeError, CascadeSVM(kernel='fake_kernel'))

    def test_fit(self):
        seed = 666
        file_ = "tests/files/libsvm/2"

        x, y = ds.load_svmlight_file(file_, (10, 300), 780, False)

        csvm = CascadeSVM(
            cascade_arity=3,
            max_iter=5,
            tol=1e-4,
            kernel="linear",
            c=2,
            gamma=0.1,
            check_convergence=True,
            random_state=seed,
            verbose=False,
        )

        csvm.fit(x, y)
        save_model(csvm, self.filepath, save_format="cbor")
        csvm2 = load_model(self.filepath, load_format="cbor")

        self.assertTrue(csvm.converged)
        self.assertTrue(csvm2.converged)

        csvm = CascadeSVM(
            cascade_arity=3,
            max_iter=1,
            tol=1e-4,
            kernel="linear",
            c=2,
            gamma=0.1,
            check_convergence=False,
            random_state=seed,
            verbose=False,
        )

        csvm.fit(x, y)
        save_model(csvm, self.filepath, save_format="cbor")
        csvm2 = load_model(self.filepath, load_format="cbor")
        self.assertFalse(csvm.converged)
        self.assertEqual(csvm.iterations, 1)
        self.assertFalse(csvm2.converged)
        self.assertEqual(csvm2.iterations, 1)

    def test_predict(self):
        seed = 666

        # negative points belong to class 1, positives to 0
        p1, p2, p3, p4 = [1, 2], [2, 1], [-1, -2], [-2, -1]

        x = ds.array(np.array([p1, p4, p3, p2]), (2, 2))
        y = ds.array(np.array([0, 1, 1, 0]).reshape(-1, 1), (2, 1))

        csvm = CascadeSVM(
            cascade_arity=3,
            max_iter=10,
            tol=1e-4,
            kernel="linear",
            c=2,
            gamma=0.1,
            check_convergence=False,
            random_state=seed,
            verbose=False,
        )

        csvm.fit(x, y)
        save_model(csvm, self.filepath, save_format="cbor")
        csvm2 = load_model(self.filepath, load_format="cbor")

        # p5 should belong to class 0, p6 to class 1
        p5, p6 = np.array([1, 1]), np.array([-1, -1])

        x_test = ds.array(np.array([p1, p2, p3, p4, p5, p6]), (2, 2))

        y_pred = csvm.predict(x_test)
        y_pred2 = csvm2.predict(x_test)

        l1, l2, l3, l4, l5, l6 = y_pred.collect()
        self.assertTrue(l1 == l2 == l5 == 0)
        self.assertTrue(l3 == l4 == l6 == 1)

        l1, l2, l3, l4, l5, l6 = y_pred2.collect()
        self.assertTrue(l1 == l2 == l5 == 0)
        self.assertTrue(l3 == l4 == l6 == 1)

    def test_score(self):
        seed = 666

        # negative points belong to class 1, positives to 0
        p1, p2, p3, p4 = [1, 2], [2, 1], [-1, -2], [-2, -1]

        x = ds.array(np.array([p1, p4, p3, p2]), (2, 2))
        y = ds.array(np.array([0, 1, 1, 0]).reshape(-1, 1), (2, 1))

        csvm = CascadeSVM(
            cascade_arity=3,
            max_iter=10,
            tol=1e-4,
            kernel="rbf",
            c=2,
            gamma=0.1,
            check_convergence=True,
            random_state=seed,
            verbose=False,
        )

        csvm.fit(x, y)
        save_model(csvm, self.filepath, save_format="cbor")
        csvm2 = load_model(self.filepath, load_format="cbor")

        # points are separable, scoring the training dataset should have 100%
        # accuracy
        x_test = ds.array(np.array([p1, p2, p3, p4]), (2, 2))
        y_test = ds.array(np.array([0, 0, 1, 1]).reshape(-1, 1), (2, 1))

        accuracy = compss_wait_on(csvm.score(x_test, y_test))
        accuracy2 = compss_wait_on(csvm2.score(x_test, y_test))

        self.assertEqual(accuracy, 1.0)
        self.assertEqual(accuracy2, 1.0)

    def test_decision_func(self):
        seed = 666

        # negative points belong to class 1, positives to 0
        # all points are in the x-axis
        p1, p2, p3, p4 = [0, 2], [0, 1], [0, -2], [0, -1]

        x = ds.array(np.array([p1, p4, p3, p2]), (2, 2))
        y = ds.array(np.array([0, 1, 1, 0]).reshape(-1, 1), (2, 1))

        csvm = CascadeSVM(
            cascade_arity=3,
            max_iter=10,
            tol=1e-4,
            kernel="rbf",
            c=2,
            gamma=0.1,
            check_convergence=False,
            random_state=seed,
            verbose=False,
        )

        csvm.fit(x, y)
        save_model(csvm, self.filepath, save_format="cbor")
        csvm2 = load_model(self.filepath, load_format="cbor")

        # p1 should be equidistant to p3, and p2 to p4
        x_test = ds.array(np.array([p1, p2, p3, p4]), (2, 2))

        y_pred = csvm.decision_function(x_test)
        y_pred2 = csvm2.decision_function(x_test)

        d1, d2, d3, d4 = y_pred.collect()
        self.assertTrue(np.isclose(abs(d1) - abs(d3), 0))
        self.assertTrue(np.isclose(abs(d2) - abs(d4), 0))
        d1, d2, d3, d4 = y_pred2.collect()
        self.assertTrue(np.isclose(abs(d1) - abs(d3), 0))
        self.assertTrue(np.isclose(abs(d2) - abs(d4), 0))

        # p5 and p6 should be in the decision function (distance=0)
        p5, p6 = np.array([1, 0]), np.array([-1, 0])

        x_test = ds.array(np.array([p5, p6]), (1, 2))

        y_pred = csvm.decision_function(x_test)
        y_pred2 = csvm2.decision_function(x_test)

        d5, d6 = y_pred.collect()
        self.assertTrue(np.isclose(d5, 0))
        self.assertTrue(np.isclose(d6, 0))
        d5, d6 = y_pred2.collect()
        self.assertTrue(np.isclose(d5, 0))
        self.assertTrue(np.isclose(d6, 0))

    def test_sparse(self):
        """Tests that C-SVM produces the same results with sparse and dense
        data"""
        seed = 666
        train = "tests/files/libsvm/3"

        x_sp, y_sp = ds.load_svmlight_file(train, (10, 300), 780, True)
        x_d, y_d = ds.load_svmlight_file(train, (10, 300), 780, False)

        csvm_sp = CascadeSVM(random_state=seed)
        csvm_sp.fit(x_sp, y_sp)
        save_model(csvm_sp, self.filepath, save_format="cbor")
        csvm_sp2 = load_model(self.filepath, load_format="cbor")

        csvm_d = CascadeSVM(random_state=seed)
        csvm_d.fit(x_d, y_d)
        save_model(csvm_d, self.filepath, save_format="cbor")
        csvm_d2 = load_model(self.filepath, load_format="cbor")

        sv_d = csvm_d._clf.support_vectors_
        sv_sp = csvm_sp._clf.support_vectors_.toarray()
        sv_d2 = csvm_d2._clf.support_vectors_
        sv_sp2 = csvm_sp2._clf.support_vectors_.toarray()

        self.assertTrue(np.array_equal(sv_d, sv_sp))
        self.assertTrue(np.array_equal(sv_d2, sv_sp2))
        self.assertTrue(np.array_equal(sv_d, sv_d2))

        coef_d = csvm_d._clf.dual_coef_
        coef_sp = csvm_sp._clf.dual_coef_.toarray()
        coef_d2 = csvm_d2._clf.dual_coef_
        coef_sp2 = csvm_sp2._clf.dual_coef_.toarray()

        self.assertTrue(np.array_equal(coef_d, coef_sp))
        self.assertTrue(np.array_equal(coef_d2, coef_sp2))
        self.assertTrue(np.array_equal(coef_d, coef_d2))

    def test_duplicates(self):
        """Tests that C-SVM does not generate duplicate support vectors"""
        x = ds.array(
            np.array(
                [
                    [0, 1],
                    [1, 1],
                    [0, 1],
                    [1, 2],
                    [0, 0],
                    [2, 2],
                    [2, 1],
                    [1, 0],
                ]
            ),
            (2, 2),
        )

        y = ds.array(np.array([1, 0, 1, 0, 1, 0, 0, 1]).reshape(-1, 1), (2, 1))

        csvm = CascadeSVM(c=1, random_state=1, max_iter=100, tol=0)
        csvm.fit(x, y)
        save_model(csvm, self.filepath, save_format="cbor")
        csvm2 = load_model(self.filepath, load_format="cbor")

        csvm._collect_clf()
        csvm2._collect_clf()
        self.assertEqual(csvm._clf.support_vectors_.shape[0], 6)
        self.assertEqual(csvm2._clf.support_vectors_.shape[0], 6)


class RFSavingTestCBOR(unittest.TestCase):
    filepath = "tests/files/saving/rf.cbor"

    def test_make_classification_score(self):
        """Tests RandomForestClassifier fit and score with default params."""
        x, y = make_classification(
            n_samples=3000,
            n_features=10,
            n_classes=3,
            n_informative=4,
            n_redundant=2,
            n_repeated=1,
            n_clusters_per_class=2,
            shuffle=True,
            random_state=0,
        )
        x_train = ds.array(x[: len(x) // 2], (300, 10))
        y_train = ds.array(y[: len(y) // 2][:, np.newaxis], (300, 1))
        x_test = ds.array(x[len(x) // 2:], (300, 10))
        y_test = ds.array(y[len(y) // 2:][:, np.newaxis], (300, 1))

        rf = RandomForestClassifier(random_state=0)
        rf.fit(x_train, y_train)
        save_model(rf, self.filepath, save_format="cbor")
        rf2 = load_model(self.filepath, load_format="cbor")

        accuracy = compss_wait_on(rf.score(x_test, y_test))
        accuracy2 = compss_wait_on(rf2.score(x_test, y_test))
        self.assertGreater(accuracy, 0.7)
        self.assertGreater(accuracy2, 0.7)

    def test_make_classification_predict_and_distr_depth(self):
        """Tests RandomForestClassifier fit and predict with a distr_depth."""
        x, y = make_classification(
            n_samples=3000,
            n_features=10,
            n_classes=3,
            n_informative=4,
            n_redundant=2,
            n_repeated=1,
            n_clusters_per_class=2,
            shuffle=True,
            random_state=0,
        )
        x_train = ds.array(x[: len(x) // 2], (300, 10))
        y_train = ds.array(y[: len(y) // 2][:, np.newaxis], (300, 1))
        x_test = ds.array(x[len(x) // 2:], (300, 10))
        y_test = y[len(y) // 2:]

        rf = RandomForestClassifier(distr_depth=2, random_state=0)
        rf.fit(x_train, y_train)
        save_model(rf, self.filepath, save_format="cbor")
        rf2 = load_model(self.filepath, load_format="cbor")

        y_pred = rf.predict(x_test).collect()
        y_pred2 = rf2.predict(x_test).collect()
        accuracy = np.count_nonzero(y_pred == y_test) / len(y_test)
        accuracy2 = np.count_nonzero(y_pred2 == y_test) / len(y_test)
        self.assertGreater(accuracy, 0.7)
        self.assertGreater(accuracy2, 0.7)

    def test_make_classification_fit_predict(self):
        """Tests RandomForestClassifier fit_predict with default params."""
        x, y = make_classification(
            n_samples=3000,
            n_features=10,
            n_classes=3,
            n_informative=4,
            n_redundant=2,
            n_repeated=1,
            n_clusters_per_class=2,
            shuffle=True,
            random_state=0,
        )
        x_train = ds.array(x[: len(x) // 2], (300, 10))
        y_train = ds.array(y[: len(y) // 2][:, np.newaxis], (300, 1))

        rf = RandomForestClassifier(random_state=0)
        rf.fit(x_train, y_train)
        save_model(rf, self.filepath, save_format="cbor")
        rf2 = load_model(self.filepath, load_format="cbor")

        y_pred = rf.predict(x_train).collect()
        y_pred2 = rf2.predict(x_train).collect()
        y_train = y_train.collect()
        accuracy = np.count_nonzero(y_pred == y_train) / len(y_train)
        accuracy2 = np.count_nonzero(y_pred2 == y_train) / len(y_train)
        self.assertGreater(accuracy, 0.7)
        self.assertGreater(accuracy2, 0.7)

    def test_make_classification_sklearn_max_predict(self):
        """Tests RandomForestClassifier predict with sklearn_max."""
        x, y = make_classification(
            n_samples=3000,
            n_features=10,
            n_classes=3,
            n_informative=4,
            n_redundant=2,
            n_repeated=1,
            n_clusters_per_class=2,
            shuffle=True,
            random_state=0,
        )
        x_train = ds.array(x[: len(x) // 2], (300, 10))
        y_train = ds.array(y[: len(y) // 2][:, np.newaxis], (300, 1))
        x_test = ds.array(x[len(x) // 2:], (300, 10))
        y_test = y[len(y) // 2:]

        rf = RandomForestClassifier(random_state=0, sklearn_max=10)
        rf.fit(x_train, y_train)
        save_model(rf, self.filepath, save_format="cbor")
        rf2 = load_model(self.filepath, load_format="cbor")

        y_pred = rf.predict(x_test).collect()
        y_pred2 = rf2.predict(x_test).collect()
        accuracy = np.count_nonzero(y_pred == y_test) / len(y_test)
        accuracy2 = np.count_nonzero(y_pred2 == y_test) / len(y_test)
        self.assertGreater(accuracy, 0.7)
        self.assertGreater(accuracy2, 0.7)

    def test_make_classification_sklearn_max_predict_proba(self):
        """Tests RandomForestClassifier predict_proba with sklearn_max."""
        x, y = make_classification(
            n_samples=3000,
            n_features=10,
            n_classes=3,
            n_informative=4,
            n_redundant=2,
            n_repeated=1,
            n_clusters_per_class=2,
            shuffle=True,
            random_state=0,
        )
        x_train = ds.array(x[: len(x) // 2], (300, 10))
        y_train = ds.array(y[: len(y) // 2][:, np.newaxis], (300, 1))
        x_test = ds.array(x[len(x) // 2:], (300, 10))
        y_test = y[len(y) // 2:]

        rf = RandomForestClassifier(random_state=0, sklearn_max=10)
        rf.fit(x_train, y_train)
        save_model(rf, self.filepath, save_format="cbor")
        rf2 = load_model(self.filepath, load_format="cbor")

        probabilities = rf.predict_proba(x_test).collect()
        probabilities2 = rf2.predict_proba(x_test).collect()
        rf.classes = compss_wait_on(rf.classes)
        rf2.classes = compss_wait_on(rf2.classes)
        y_pred = rf.classes[np.argmax(probabilities, axis=1)]
        y_pred2 = rf2.classes[np.argmax(probabilities2, axis=1)]
        accuracy = np.count_nonzero(y_pred == y_test) / len(y_test)
        accuracy2 = np.count_nonzero(y_pred2 == y_test) / len(y_test)
        self.assertGreater(accuracy, 0.7)
        self.assertGreater(accuracy2, 0.7)

    def test_make_classification_hard_vote_predict(self):
        """Tests RandomForestClassifier predict with hard_vote."""
        x, y = make_classification(
            n_samples=3000,
            n_features=10,
            n_classes=3,
            n_informative=4,
            n_redundant=2,
            n_repeated=1,
            n_clusters_per_class=2,
            shuffle=True,
            random_state=0,
        )
        x_train = ds.array(x[: len(x) // 2], (300, 10))
        y_train = ds.array(y[: len(y) // 2][:, np.newaxis], (300, 1))
        x_test = ds.array(x[len(x) // 2:], (300, 10))
        y_test = y[len(y) // 2:]

        rf = RandomForestClassifier(
            random_state=0, sklearn_max=10, hard_vote=True
        )
        rf.fit(x_train, y_train)
        save_model(rf, self.filepath, save_format="cbor")
        rf2 = load_model(self.filepath, load_format="cbor")

        y_pred = rf.predict(x_test).collect()
        y_pred2 = rf2.predict(x_test).collect()
        accuracy = np.count_nonzero(y_pred == y_test) / len(y_test)
        accuracy2 = np.count_nonzero(y_pred2 == y_test) / len(y_test)
        self.assertGreater(accuracy, 0.7)
        self.assertGreater(accuracy2, 0.7)

    def test_make_classification_hard_vote_score_mix(self):
        """Tests RandomForestClassifier score with hard_vote, sklearn_max,
        distr_depth and max_depth."""
        x, y = make_classification(
            n_samples=3000,
            n_features=10,
            n_classes=3,
            n_informative=4,
            n_redundant=2,
            n_repeated=1,
            n_clusters_per_class=2,
            shuffle=True,
            random_state=0,
        )
        x_train = ds.array(x[: len(x) // 2], (300, 10))
        y_train = ds.array(y[: len(y) // 2][:, np.newaxis], (300, 1))
        x_test = ds.array(x[len(x) // 2:], (300, 10))
        y_test = ds.array(y[len(y) // 2:][:, np.newaxis], (300, 1))

        rf = RandomForestClassifier(
            random_state=0,
            sklearn_max=100,
            distr_depth=2,
            max_depth=12,
            hard_vote=True,
        )
        rf.fit(x_train, y_train)
        save_model(rf, self.filepath, save_format="cbor")
        rf2 = load_model(self.filepath, load_format="cbor")

        accuracy = compss_wait_on(rf.score(x_test, y_test))
        accuracy2 = compss_wait_on(rf2.score(x_test, y_test))
        self.assertGreater(accuracy, 0.7)
        self.assertGreater(accuracy2, 0.7)

    def test_iris(self):
        """Tests RandomForestClassifier with a minimal example."""
        x, y = datasets.load_iris(return_X_y=True)
        ds_fit = ds.array(x[::2], block_size=(30, 2))
        fit_y = ds.array(y[::2].reshape(-1, 1), block_size=(30, 1))
        ds_validate = ds.array(x[1::2], block_size=(30, 2))
        validate_y = ds.array(y[1::2].reshape(-1, 1), block_size=(30, 1))

        rf = RandomForestClassifier(
            n_estimators=1, max_depth=1, random_state=0
        )
        rf.fit(ds_fit, fit_y)
        save_model(rf, self.filepath, save_format="cbor")
        rf2 = load_model(self.filepath, load_format="cbor")

        accuracy = compss_wait_on(rf.score(ds_validate, validate_y))
        accuracy2 = compss_wait_on(rf2.score(ds_validate, validate_y))

        # Accuracy should be <= 2/3 for any seed, often exactly equal.
        self.assertAlmostEqual(accuracy, 2 / 3)
        self.assertAlmostEqual(accuracy2, 2 / 3)


class LassoSavingTestCBOR(unittest.TestCase):
    filepath = "tests/files/saving/lasso.cbor"

    def test_fit_predict(self):
        """Tests fit and predicts methods"""

        np.random.seed(42)

        n_samples, n_features = 50, 100
        X = np.random.randn(n_samples, n_features)

        # Decreasing coef w. alternated signs for visualization
        idx = np.arange(n_features)
        coef = (-1) ** idx * np.exp(-idx / 10)
        coef[10:] = 0  # sparsify coef
        y = np.dot(X, coef)

        # Add noise
        y += 0.01 * np.random.normal(size=n_samples)

        n_samples = X.shape[0]
        X_train, y_train = X[: n_samples // 2], y[: n_samples // 2]
        X_test, y_test = X[n_samples // 2:], y[n_samples // 2:]

        lasso = Lasso(lmbd=0.1, max_iter=50)

        lasso.fit(ds.array(X_train, (5, 100)), ds.array(y_train, (5, 1)))
        save_model(lasso, self.filepath, save_format="cbor")
        lasso2 = load_model(self.filepath, load_format="cbor")

        y_pred_lasso = lasso.predict(ds.array(X_test, (25, 100)))
        r2_score_lasso = r2_score(y_test, y_pred_lasso.collect())
        y_pred_lasso2 = lasso2.predict(ds.array(X_test, (25, 100)))
        r2_score_lasso2 = r2_score(y_test, y_pred_lasso2.collect())

        self.assertAlmostEqual(r2_score_lasso, 0.9481746925431124)
        self.assertAlmostEqual(r2_score_lasso2, 0.9481746925431124)


class LinearRegressionSavingTestCBOR(unittest.TestCase):
    filepath = "tests/files/saving/linear_regression.cbor"

    def test_univariate(self):
        """Tests fit() and predict(), univariate."""
        x_data = np.array([1, 2, 3, 4, 5])
        y_data = np.array([2, 1, 1, 2, 4.5])

        bn, bm = 2, 1

        x = ds.array(x=x_data, block_size=(bn, bm))
        y = ds.array(x=y_data, block_size=(bn, bm))

        reg = LinearRegression()
        reg.fit(x, y)
        save_model(reg, self.filepath, save_format="cbor")
        reg2 = load_model(self.filepath, load_format="cbor")

        self.assertTrue(np.allclose(reg.coef_.collect(), 0.6))
        self.assertTrue(np.allclose(reg.intercept_.collect(), 0.3))
        self.assertTrue(np.allclose(reg2.coef_.collect(), 0.6))
        self.assertTrue(np.allclose(reg2.intercept_.collect(), 0.3))

        # Predict one sample
        x_test = np.array([3])
        test_data = ds.array(x=x_test, block_size=(1, 1))
        pred = reg.predict(test_data).collect()
        pred2 = reg2.predict(test_data).collect()
        self.assertTrue(np.allclose(pred, 2.1))
        self.assertTrue(np.allclose(pred2, 2.1))

        # Predict multiple samples
        x_test = np.array([3, 5, 6])
        test_data = ds.array(x=x_test, block_size=(bn, bm))
        pred = reg.predict(test_data).collect()
        pred2 = reg2.predict(test_data).collect()
        self.assertTrue(np.allclose(pred, [2.1, 3.3, 3.9]))
        self.assertTrue(np.allclose(pred2, [2.1, 3.3, 3.9]))

    def test_univariate_no_intercept(self):
        """Tests fit() and predict(), univariate, fit_intercept=False."""
        x_data = np.array([1, 2, 3, 4, 5])
        y_data = np.array([2, 1, 1, 2, 4.5])

        bn, bm = 2, 1

        x = ds.array(x=x_data, block_size=(bn, bm))
        y = ds.array(x=y_data, block_size=(bn, bm))

        reg = LinearRegression(fit_intercept=False)
        reg.fit(x, y)
        save_model(reg, self.filepath, save_format="cbor")
        reg2 = load_model(self.filepath, load_format="cbor")

        self.assertTrue(np.allclose(reg.coef_.collect(), 0.68181818))
        self.assertTrue(np.allclose(reg2.coef_.collect(), 0.68181818))
        self.assertTrue(np.allclose(reg.intercept_.collect(), 0))
        self.assertTrue(np.allclose(reg2.intercept_.collect(), 0))

        # Predict one sample
        x_test = np.array([3])
        test_data = ds.array(x=x_test, block_size=(1, 1))
        pred = reg.predict(test_data).collect()
        pred2 = reg2.predict(test_data).collect()
        self.assertTrue(np.allclose(pred, 2.04545455))
        self.assertTrue(np.allclose(pred2, 2.04545455))

        # Predict multiple samples
        x_test = np.array([3, 5, 6])
        test_data = ds.array(x=x_test, block_size=(bn, bm))
        pred = reg.predict(test_data).collect()
        pred2 = reg.predict(test_data).collect()
        self.assertTrue(np.allclose(pred, [2.04545455, 3.4090909, 4.0909091]))
        self.assertTrue(np.allclose(pred2, [2.04545455, 3.4090909, 4.0909091]))

    def test_multivariate(self):
        """Tests fit() and predict(), multivariate."""
        x_data = np.array([[1, 2], [2, 0], [3, 1], [4, 4], [5, 3]])
        y_data = np.array([2, 1, 1, 2, 4.5])

        bn, bm = 2, 2

        x = ds.array(x=x_data, block_size=(bn, bm))
        y = ds.array(x=y_data, block_size=(bn, 1))

        reg = LinearRegression()
        reg.fit(x, y)
        save_model(reg, self.filepath, save_format="cbor")
        reg2 = load_model(self.filepath, load_format="cbor")

        self.assertTrue(np.allclose(reg.coef_.collect(), [0.421875, 0.296875]))
        self.assertTrue(
            np.allclose(reg2.coef_.collect(), [0.421875, 0.296875])
        )
        self.assertTrue(np.allclose(reg.intercept_.collect(), 0.240625))
        self.assertTrue(np.allclose(reg2.intercept_.collect(), 0.240625))

        # Predict one sample
        x_test = np.array([3, 2])
        test_data = ds.array(x=x_test, block_size=(1, bm))
        pred = reg.predict(test_data).collect()
        pred2 = reg2.predict(test_data).collect()
        self.assertTrue(np.allclose(pred, 2.1))
        self.assertTrue(np.allclose(pred2, 2.1))

        # Predict multiple samples
        x_test = np.array([[3, 2], [4, 4], [1, 3]])
        test_data = ds.array(x=x_test, block_size=(bn, bm))
        pred = reg.predict(test_data).collect()
        self.assertTrue(np.allclose(pred, [2.1, 3.115625, 1.553125]))

    def test_multivariate_no_intercept(self):
        """Tests fit() and predict(), multivariate, fit_intercept=False."""
        x_data = np.array([[1, 2], [2, 0], [3, 1], [4, 4], [5, 3]])
        y_data = np.array([2, 1, 1, 2, 4.5])

        bn, bm = 2, 2

        x = ds.array(x=x_data, block_size=(bn, bm))
        y = ds.array(x=y_data, block_size=(bn, 1))

        reg = LinearRegression(fit_intercept=False)
        reg.fit(x, y)
        save_model(reg, self.filepath, save_format="cbor")
        reg2 = load_model(self.filepath, load_format="cbor")

        self.assertTrue(
            np.allclose(reg.coef_.collect(), [0.48305085, 0.30367232])
        )
        self.assertTrue(
            np.allclose(reg2.coef_.collect(), [0.48305085, 0.30367232])
        )
        self.assertTrue(np.allclose(reg.intercept_.collect(), 0))
        self.assertTrue(np.allclose(reg2.intercept_.collect(), 0))

        # Predict one sample
        x_test = np.array([3, 2])
        test_data = ds.array(x=x_test, block_size=(1, bm))
        pred = reg.predict(test_data).collect()
        pred2 = reg2.predict(test_data).collect()
        self.assertTrue(np.allclose(pred, [2.05649718]))
        self.assertTrue(np.allclose(pred2, [2.05649718]))

        # Predict multiple samples
        x_test = np.array([[3, 2], [4, 4], [1, 3]])
        test_data = ds.array(x=x_test, block_size=(bn, bm))
        pred = reg.predict(test_data).collect()
        pred2 = reg2.predict(test_data).collect()
        self.assertTrue(np.allclose(pred, [2.05649718, 3.14689266, 1.3940678]))
        self.assertTrue(
            np.allclose(pred2, [2.05649718, 3.14689266, 1.3940678])
        )

    def test_multivariate_multiobjective(self):
        """Tests fit() and predict(), multivariate, multiobjective."""
        x_data = np.array(
            [[1, 2, 3], [2, 0, 4], [3, 1, 8], [4, 4, 2], [5, 3, 1], [2, 7, 1]]
        )
        y_data = np.array(
            [
                [2, 0, 3],
                [1, 5, 2],
                [1, 3, 4],
                [2, 7, 9],
                [4.5, -1, 4],
                [0, 0, 0],
            ]
        )

        bn, bm = 2, 2

        x = ds.array(x=x_data, block_size=(bn, bm))
        y = ds.array(x=y_data, block_size=(bn, bm))

        reg = LinearRegression()
        reg.fit(x, y)
        save_model(reg, self.filepath, save_format="cbor")
        reg2 = load_model(self.filepath, load_format="cbor")

        # Predict one sample
        x_test = np.array([3, 2, 1])
        test_data = ds.array(x=x_test, block_size=(1, bm))
        pred = reg.predict(test_data).collect()
        pred2 = reg2.predict(test_data).collect()
        self.assertTrue(np.allclose(pred, [3.0318415, 1.97164872, 3.85410906]))
        self.assertTrue(
            np.allclose(pred2, [3.0318415, 1.97164872, 3.85410906])
        )

        # Predict multiple samples
        x_test = np.array([[3, 2, 1], [4, 3, 3], [1, 1, 1]])
        test_data = ds.array(x=x_test, block_size=(bn, bm))
        pred = reg.predict(test_data).collect()
        pred2 = reg2.predict(test_data).collect()
        self.assertTrue(
            np.allclose(
                pred,
                [
                    [3.0318415, 1.97164872, 3.85410906],
                    [2.5033157, 2.65809327, 5.05310495],
                    [2.145797, 1.4840121, 1.5739791],
                ],
            )
        )
        self.assertTrue(
            np.allclose(
                pred2,
                [
                    [3.0318415, 1.97164872, 3.85410906],
                    [2.5033157, 2.65809327, 5.05310495],
                    [2.145797, 1.4840121, 1.5739791],
                ],
            )
        )

        # Check attributes values
        self.assertTrue(
            np.allclose(
                reg2.coef_.collect(),
                [
                    [0.65034768, 0.34673933, 1.22176283],
                    [-0.41465084, -0.20584208, -0.16339571],
                    [-0.38211131, 0.27277365, 0.07031439],
                ],
            )
        )
        self.assertTrue(
            np.allclose(
                reg2.coef_.collect(),
                [
                    [0.65034768, 0.34673933, 1.22176283],
                    [-0.41465084, -0.20584208, -0.16339571],
                    [-0.38211131, 0.27277365, 0.07031439],
                ],
            )
        )
        self.assertTrue(
            np.allclose(
                reg.intercept_.collect(), [2.29221145, 1.07034124, 0.44529761]
            )
        )
        self.assertTrue(
            np.allclose(
                reg2.intercept_.collect(), [2.29221145, 1.07034124, 0.44529761]
            )
        )


def load_movielens(train_ratio=0.9):
    file = "tests/files/sample_movielens_ratings.csv"

    # 'user_id', 'movie_id', 'rating', 'timestamp'

    data = np.genfromtxt(file, dtype="int", delimiter=",", usecols=range(3))

    # just in case there are movies/user without rating
    # movie_id
    n_m = max(len(np.unique(data[:, 1])), max(data[:, 1]) + 1)
    # user_id
    n_u = max(len(np.unique(data[:, 0])), max(data[:, 0]) + 1)

    idx = int(data.shape[0] * train_ratio)

    train_data = data[:idx]
    test_data = data[idx:]

    train = csr_matrix(
        (train_data[:, 2], (train_data[:, 0], train_data[:, 1])),
        shape=(n_u, n_m),
    )

    test = csr_matrix((test_data[:, 2], (test_data[:, 0], test_data[:, 1])))

    x_size, y_size = train.shape[0] // 4, train.shape[1] // 4
    train_arr = ds.array(train, block_size=(x_size, y_size))

    x_size, y_size = test.shape[0] // 4, test.shape[1] // 4
    test_arr = ds.array(test, block_size=(x_size, y_size))

    return train_arr, test_arr


class ALSSavingTestCBOR(unittest.TestCase):
    filepath = "tests/files/saving/als.cbor"

    def test_init_params(self):
        # Test all parameters
        seed = 666
        n_f = 100
        lambda_ = 0.001
        convergence_threshold = 0.1
        max_iter = 10
        verbose = True
        arity = 12

        als = ALS(
            random_state=seed,
            n_f=n_f,
            lambda_=lambda_,
            tol=convergence_threshold,
            max_iter=max_iter,
            verbose=verbose,
            arity=arity,
        )
        save_model(als, self.filepath, save_format="cbor")
        als2 = load_model(self.filepath, load_format="cbor")

        self.assertEqual(als.random_state, seed)
        self.assertEqual(als.n_f, n_f)
        self.assertEqual(als.lambda_, lambda_)
        self.assertEqual(als.tol, convergence_threshold)
        self.assertEqual(als.max_iter, max_iter)
        self.assertEqual(als.verbose, verbose)
        self.assertEqual(als.arity, arity)
        self.assertEqual(als2.random_state, seed)
        self.assertEqual(als2.n_f, n_f)
        self.assertEqual(als2.lambda_, lambda_)
        self.assertEqual(als2.tol, convergence_threshold)
        self.assertEqual(als2.max_iter, max_iter)
        self.assertEqual(als2.verbose, verbose)
        self.assertEqual(als2.arity, arity)

    def test_fit(self):
        train, test = load_movielens()

        als = ALS(
            tol=0.01,
            random_state=666,
            n_f=100,
            verbose=False,
            check_convergence=True,
        )

        als.fit(train, test)
        self.assertTrue(als.converged)

        als.fit(train)
        save_model(als, self.filepath, save_format="cbor")
        als2 = load_model(self.filepath, load_format="cbor")

        self.assertTrue(als.converged)
        self.assertTrue(als2.converged)

    def test_predict(self):
        data = np.array([[0, 0, 5], [3, 0, 5], [3, 1, 2]])
        ratings = csr_matrix(data)
        train = ds.array(x=ratings, block_size=(1, 1))
        als = ALS(tol=0.01, random_state=666, n_f=5, verbose=False)
        als.fit(train)
        save_model(als, self.filepath, save_format="cbor")
        als2 = load_model(self.filepath, load_format="cbor")

        predictions = als.predict_user(user_id=0)
        predictions2 = als2.predict_user(user_id=0)

        # Check that the ratings for user 0 are similar to user 1 because they
        # share preferences (third movie), thus it is expected that user 0
        # will rate movie 1 similarly to user 1.
        self.assertTrue(
            2.75 < predictions[0] < 3.25
            and predictions[1] < 1
            and predictions[2] > 4.5
        )
        self.assertTrue(
            2.75 < predictions2[0] < 3.25
            and predictions2[1] < 1
            and predictions2[2] > 4.5
        )
        self.assertTrue(
            np.array_equal(predictions, predictions2, equal_nan=True)
        )


def main():
    unittest.main()


if __name__ == "__main__":
    main()
