import unittest

import numpy as np
from parameterized import parameterized
from pycompss.api.api import compss_wait_on
from sklearn.datasets import make_classification

import dislib as ds
from dislib.classification import CascadeSVM
import dislib.data.util.model as utilmodel
from tests import BaseTimedTestCase


class CSVMTest(BaseTimedTestCase):
    def test_init_params(self):
        """ Test constructor parameters"""
        cascade_arity = 3
        max_iter = 1
        tol = 1e-4
        kernel = 'rbf'
        c = 2
        gamma = 0.1
        check_convergence = True
        seed = 666
        verbose = False

        csvm = CascadeSVM(cascade_arity=cascade_arity, max_iter=max_iter,
                          tol=tol, kernel=kernel, c=c, gamma=gamma,
                          check_convergence=check_convergence,
                          random_state=seed, verbose=verbose)
        self.assertEqual(csvm.cascade_arity, cascade_arity)
        self.assertEqual(csvm.max_iter, max_iter)
        self.assertEqual(csvm.tol, tol)
        self.assertEqual(csvm.kernel, kernel)
        self.assertEqual(csvm.c, c)
        self.assertEqual(csvm.gamma, gamma)
        self.assertEqual(csvm.check_convergence, check_convergence)
        self.assertEqual(csvm.random_state, seed)
        self.assertEqual(csvm.verbose, verbose)

    def test_fit_private_params(self):
        kernel = 'rbf'
        c = 2
        gamma = 0.1
        seed = 666
        file_ = "tests/datasets/libsvm/2"

        x, y = ds.load_svmlight_file(file_, (10, 300), 780, False)
        csvm = CascadeSVM(kernel=kernel, c=c, gamma=gamma, random_state=seed)
        csvm.fit(x, y)

        self.assertEqual(csvm._clf_params['kernel'], kernel)
        self.assertEqual(csvm._clf_params['C'], c)
        self.assertEqual(csvm._clf_params['gamma'], gamma)

        kernel, c = 'linear', 0.3
        csvm = CascadeSVM(kernel=kernel, c=c, random_state=seed)
        csvm.fit(x, y)
        self.assertEqual(csvm._clf_params['kernel'], kernel)
        self.assertEqual(csvm._clf_params['C'], c)

        # # check for exception when incorrect kernel is passed
        # self.assertRaises(AttributeError, CascadeSVM(kernel='fake_kernel'))

    def test_fit(self):
        seed = 666
        file_ = "tests/datasets/libsvm/2"

        x, y = ds.load_svmlight_file(file_, (10, 300), 780, False)

        csvm = CascadeSVM(cascade_arity=3, max_iter=5,
                          tol=1e-4, kernel='linear', c=2, gamma=0.1,
                          check_convergence=True,
                          random_state=seed, verbose=False)

        csvm.fit(x, y)

        self.assertTrue(csvm.converged)

        csvm = CascadeSVM(cascade_arity=3, max_iter=1,
                          tol=1e-4, kernel='linear', c=2, gamma=0.1,
                          check_convergence=False,
                          random_state=seed, verbose=False)

        csvm.fit(x, y)
        self.assertFalse(csvm.converged)
        self.assertEqual(csvm.iterations, 1)
        csvm = CascadeSVM(cascade_arity=3, max_iter=4,
                          tol=1e-4, kernel='rbf', c=2, gamma=0.1,
                          check_convergence=True,
                          random_state=seed, verbose=True)
        csvm.fit(x, y)
        self.assertTrue(csvm.converged)
        csvm = CascadeSVM(cascade_arity=3, max_iter=2,
                          tol=1e-7, kernel='rbf', c=1, gamma=0.01,
                          check_convergence=True,
                          random_state=seed, verbose=True)
        csvm.fit(x, y)
        self.assertFalse(csvm.converged)

    def test_fit_multiclass(self):
        seed = 666
        x, y = make_classification(
            n_samples=2400,
            n_features=13,
            n_classes=5,
            n_informative=3,
            n_redundant=1,
            n_repeated=2,
            n_clusters_per_class=1,
            shuffle=True,
            random_state=0,
        )
        x_train = ds.array(x[::2], (50, 13))
        y_train = ds.array(y[::2][:, np.newaxis], (50, 1))

        csvm = CascadeSVM(cascade_arity=3, max_iter=15,
                          tol=1e-6, kernel='rbf', c=1, gamma=0.05,
                          check_convergence=True,
                          random_state=seed, verbose=False)
        csvm.fit(x_train, y_train)
        self.assertTrue(csvm.converged)

        csvm = CascadeSVM(cascade_arity=3, max_iter=1,
                          tol=1e-4, kernel='rbf', c=2, gamma=0.1,
                          check_convergence=False,
                          random_state=seed, verbose=False)

        csvm.fit(x_train, y_train)
        self.assertFalse(csvm.converged)
        self.assertEqual(csvm.iterations, 1)

    def test_predict_multiclass(self):
        seed = 666

        p1, p2, p3, p4, p5, p6 = [1, 2], [2, 1], [-1, -2], \
                                 [-2, -1], [15, 15], [14, 14]

        x = ds.array(np.array([p1, p4, p3, p5, p2, p6]), (2, 2))
        y = ds.array(np.array([0, 1, 1, 2, 0, 2]).reshape(-1, 1), (2, 1))

        csvm = CascadeSVM(cascade_arity=3, max_iter=10,
                          tol=1e-4, kernel='linear', c=2, gamma=0.1,
                          check_convergence=False,
                          random_state=seed, verbose=False)

        csvm.fit(x, y)

        p7, p8, p9, p10 = np.array([1, 1]), np.array([-1, -1]),\
            np.array([14, 15]), np.array([15, 14])

        x_test = ds.array(np.array([p1, p2, p3, p4, p5, p6, p7,
                                    p8, p9, p10]), (2, 1))

        y_pred = csvm.predict(x_test)

        l1, l2, l3, l4, l5, l6, l7, l8, l9, l10 = y_pred.collect()

        self.assertTrue(l1 == l2 == l7 == 0)
        self.assertTrue(l3 == l4 == l8 == 1)
        self.assertTrue(l5 == l6 == l9 == l10 == 2)

    def test_fit_default_gamma(self):
        """ Tests that the fit method converges when using gamma=auto on a
        toy dataset """
        seed = 666
        file_ = "tests/datasets/libsvm/2"

        x, y = ds.load_svmlight_file(file_, (10, 300), 780, False)

        csvm = CascadeSVM(cascade_arity=3, max_iter=5,
                          tol=1e-4, kernel='linear', c=2,
                          check_convergence=True,
                          random_state=seed, verbose=False)

        csvm.fit(x, y)

        self.assertTrue(csvm.converged)

        csvm = CascadeSVM(cascade_arity=3, max_iter=1,
                          tol=1e-4, kernel='linear', c=2, gamma=0.1,
                          check_convergence=False,
                          random_state=seed, verbose=False)

        csvm.fit(x, y)
        self.assertFalse(csvm.converged)
        self.assertEqual(csvm.iterations, 1)

    def test_predict(self):
        seed = 666

        # negative points belong to class 1, positives to 0
        p1, p2, p3, p4 = [1, 2], [2, 1], [-1, -2], [-2, -1]

        x = ds.array(np.array([p1, p4, p3, p2]), (2, 2))
        y = ds.array(np.array([0, 1, 1, 0]).reshape(-1, 1), (2, 1))

        csvm = CascadeSVM(cascade_arity=3, max_iter=10,
                          tol=1e-4, kernel='linear', c=2, gamma=0.1,
                          check_convergence=False,
                          random_state=seed, verbose=False)

        csvm.fit(x, y)

        # p5 should belong to class 0, p6 to class 1
        p5, p6 = np.array([1, 1]), np.array([-1, -1])

        x_test = ds.array(np.array([p1, p2, p3, p4, p5, p6]), (2, 2))

        y_pred = csvm.predict(x_test)

        l1, l2, l3, l4, l5, l6 = y_pred.collect()

        self.assertTrue(l1 == l2 == l5 == 0)
        self.assertTrue(l3 == l4 == l6 == 1)

    @parameterized.expand([(True,), (False,)])
    def test_score(self, collect):
        seed = 666

        # negative points belong to class 1, positives to 0
        p1, p2, p3, p4 = [1, 2], [2, 1], [-1, -2], [-2, -1]

        x = ds.array(np.array([p1, p4, p3, p2]), (2, 2))
        y = ds.array(np.array([0, 1, 1, 0]).reshape(-1, 1), (2, 1))

        csvm = CascadeSVM(cascade_arity=3, max_iter=10,
                          tol=1e-4, kernel='rbf', c=2, gamma=0.1,
                          check_convergence=True,
                          random_state=seed, verbose=False)

        csvm.fit(x, y)

        # points are separable, scoring the training dataset should have 100%
        # accuracy
        x_test = ds.array(np.array([p1, p2, p3, p4]), (2, 2))
        y_test = ds.array(np.array([0, 0, 1, 1]).reshape(-1, 1), (2, 1))

        accuracy = csvm.score(x_test, y_test, collect)
        if not collect:
            accuracy = compss_wait_on(accuracy)

        self.assertEqual(accuracy, 1.0)

    @parameterized.expand([(True,), (False,)])
    def test_score_multiclass(self, collect):
        seed = 666

        # negative points belong to class 1, positives to 0
        x, y = make_classification(
            n_samples=2400,
            n_features=13,
            n_classes=5,
            n_informative=3,
            n_redundant=1,
            n_repeated=2,
            n_clusters_per_class=1,
            shuffle=True,
            random_state=0,
        )
        x_train = ds.array(x[::2], (50, 13))
        y_train = ds.array(y[::2][:, np.newaxis], (50, 1))
        x_test = ds.array(x[1::2], (50, 13))
        y_test = ds.array(y[1::2][:, np.newaxis], (50, 1))

        csvm = CascadeSVM(cascade_arity=3, max_iter=10,
                          tol=1e-4, kernel='rbf', c=2, gamma=0.1,
                          check_convergence=True,
                          random_state=seed, verbose=False)

        csvm.fit(x_train, y_train)

        accuracy = csvm.score(x_test, y_test, collect)
        if not collect:
            accuracy = compss_wait_on(accuracy)

        self.assertGreater(accuracy, 0.85)

    def test_decision_func(self):
        seed = 666

        # negative points belong to class 1, positives to 0
        # all points are in the x-axis
        p1, p2, p3, p4 = [0, 2], [0, 1], [0, -2], [0, -1]

        x = ds.array(np.array([p1, p4, p3, p2]), (2, 2))
        y = ds.array(np.array([0, 1, 1, 0]).reshape(-1, 1), (2, 1))

        csvm = CascadeSVM(cascade_arity=3, max_iter=10,
                          tol=1e-4, kernel='rbf', c=2, gamma=0.1,
                          check_convergence=False,
                          random_state=seed, verbose=False)

        csvm.fit(x, y)

        # p1 should be equidistant to p3, and p2 to p4
        x_test = ds.array(np.array([p1, p2, p3, p4]), (2, 2))

        y_pred = csvm.decision_function(x_test)

        d1, d2, d3, d4 = y_pred.collect()

        self.assertTrue(np.isclose(abs(d1) - abs(d3), 0))
        self.assertTrue(np.isclose(abs(d2) - abs(d4), 0))

        # p5 and p6 should be in the decision function (distance=0)
        p5, p6 = np.array([1, 0]), np.array([-1, 0])

        x_test = ds.array(np.array([p5, p6]), (1, 2))

        y_pred = csvm.decision_function(x_test)

        d5, d6 = y_pred.collect()

        self.assertTrue(np.isclose(d5, 0))
        self.assertTrue(np.isclose(d6, 0))

    def test_sparse(self):
        """ Tests that C-SVM produces the same results with sparse and dense
        data"""
        seed = 666
        train = "tests/datasets/libsvm/3"

        x_sp, y_sp = ds.load_svmlight_file(train, (10, 300), 780, True)
        x_d, y_d = ds.load_svmlight_file(train, (10, 300), 780, False)

        csvm_sp = CascadeSVM(random_state=seed)
        csvm_sp.fit(x_sp, y_sp)

        csvm_d = CascadeSVM(random_state=seed)
        csvm_d.fit(x_d, y_d)

        sv_d = csvm_d._clf.support_vectors_
        sv_sp = csvm_sp._clf.support_vectors_.toarray()

        self.assertTrue(np.array_equal(sv_d, sv_sp))

        coef_d = csvm_d._clf.dual_coef_
        coef_sp = csvm_sp._clf.dual_coef_.toarray()

        self.assertTrue(np.array_equal(coef_d, coef_sp))

    def test_duplicates(self):
        """ Tests that C-SVM does not generate duplicate support vectors """
        x = ds.array(np.array([[0, 1],
                               [1, 1],
                               [0, 1],
                               [1, 2],
                               [0, 0],
                               [2, 2],
                               [2, 1],
                               [1, 0]]), (2, 2))

        y = ds.array(np.array([1, 0, 1, 0, 1, 0, 0, 1]).reshape(-1, 1), (2, 1))

        csvm = CascadeSVM(c=1, random_state=1, max_iter=100, tol=0)
        csvm.fit(x, y)

        csvm._collect_clf()
        self.assertEqual(csvm._clf.support_vectors_.shape[0], 6)

    def test_save_load(self):
        """
        Tests that the save and load methods of the C-SVM work properly with
        the implemented formats and that an exception is retuned when the
        requested format is not supported.
        """
        seed = 666

        # negative points belong to class 1, positives to 0
        p1, p2, p3, p4 = [1, 2], [2, 1], [-1, -2], [-2, -1]

        x = ds.array(np.array([p1, p4, p3, p2]), (2, 2))
        y = ds.array(np.array([0, 1, 1, 0]).reshape(-1, 1), (2, 1))

        csvm = CascadeSVM(cascade_arity=3, max_iter=10,
                          tol=1e-4, kernel='linear', c=2, gamma=0.1,
                          check_convergence=False,
                          random_state=seed, verbose=False)

        csvm.fit(x, y)
        csvm.save_model("./saved_csvm")
        csvm2 = CascadeSVM()
        csvm2.load_model("./saved_csvm")
        p5, p6 = np.array([1, 1]), np.array([-1, -1])

        x_test = ds.array(np.array([p1, p2, p3, p4, p5, p6]), (2, 2))

        y_pred = csvm2.predict(x_test)

        l1, l2, l3, l4, l5, l6 = y_pred.collect()
        self.assertTrue(l1 == l2 == l5 == 0)
        self.assertTrue(l3 == l4 == l6 == 1)

        csvm.save_model("./saved_csvm", save_format="cbor")
        csvm2 = CascadeSVM()
        csvm2.load_model("./saved_csvm", load_format="cbor")

        y_pred = csvm2.predict(x_test)

        l1, l2, l3, l4, l5, l6 = y_pred.collect()
        self.assertTrue(l1 == l2 == l5 == 0)
        self.assertTrue(l3 == l4 == l6 == 1)

        csvm.save_model("./saved_csvm", save_format="pickle")
        csvm2 = CascadeSVM()
        csvm2.load_model("./saved_csvm", load_format="pickle")

        y_pred = csvm2.predict(x_test)

        l1, l2, l3, l4, l5, l6 = y_pred.collect()
        self.assertTrue(l1 == l2 == l5 == 0)
        self.assertTrue(l3 == l4 == l6 == 1)

        with self.assertRaises(ValueError):
            csvm.save_model("./saved_csvm", save_format="txt")

        with self.assertRaises(ValueError):
            csvm2 = CascadeSVM()
            csvm2.load_model("./saved_csvm", load_format="txt")

        p1, p2, p3, p4 = [-1, -2], [-2, -1], [1, 2], [2, 1]

        x = ds.array(np.array([p1, p4, p3, p2]), (2, 2))
        y = ds.array(np.array([0, 1, 1, 0]).reshape(-1, 1), (2, 1))

        csvm = CascadeSVM(cascade_arity=3, max_iter=10,
                          tol=1e-4, kernel='linear', c=2, gamma=0.1,
                          check_convergence=False,
                          random_state=seed, verbose=False)

        csvm.fit(x, y)
        csvm.save_model("./saved_csvm", overwrite=False)

        csvm2 = CascadeSVM()
        csvm2.load_model("./saved_csvm", load_format="pickle")

        y_pred = csvm2.predict(x_test)

        l1, l2, l3, l4, l5, l6 = y_pred.collect()
        self.assertTrue(l1 == l2 == l5 == 0)
        self.assertTrue(l3 == l4 == l6 == 1)

        cbor2_module = utilmodel.cbor2
        utilmodel.cbor2 = None
        with self.assertRaises(ModuleNotFoundError):
            csvm.save_model("./saved_csvm", save_format="cbor")
        with self.assertRaises(ModuleNotFoundError):
            csvm2.load_model("./saved_csvm", load_format="cbor")
        utilmodel.cbor2 = cbor2_module


def main():
    unittest.main()


if __name__ == '__main__':
    main()
