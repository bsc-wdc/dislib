import unittest

import numpy as np
from pycompss.api.api import compss_wait_on

import dislib as ds
from dislib.classification import CascadeSVM


class CSVMTest(unittest.TestCase):
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
        file_ = "tests/files/libsvm/2"

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
        file_ = "tests/files/libsvm/2"

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

    def test_fit_default_gamma(self):
        """ Tests that the fit method converges when using gamma=auto on a
        toy dataset """
        seed = 666
        file_ = "tests/files/libsvm/2"

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

    def test_score(self):
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

        accuracy = compss_wait_on(csvm.score(x_test, y_test))

        self.assertEqual(accuracy, 1.0)

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
        train = "tests/files/libsvm/3"

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


def main():
    unittest.main()


if __name__ == '__main__':
    main()
