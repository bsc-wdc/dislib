import unittest

import numpy as np

from dislib.classification import CascadeSVM
from dislib.data import Dataset
from dislib.data import Subset
from dislib.data import load_libsvm_file, load_data


class CSVMTest(unittest.TestCase):
    def test_init_params(self):
        # Test all parameters with rbf kernel
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
        self.assertEqual(csvm._arity, cascade_arity)
        self.assertEqual(csvm._max_iter, max_iter)
        self.assertEqual(csvm._tol, tol)
        self.assertEqual(csvm._clf_params['kernel'], kernel)
        self.assertEqual(csvm._clf_params['C'], c)
        self.assertEqual(csvm._clf_params['gamma'], gamma)
        self.assertEqual(csvm._check_convergence, check_convergence)
        self.assertEqual(csvm._random_state, seed)
        self.assertEqual(csvm._verbose, verbose)

        # test correct linear kernel and c param (other's are not changed)
        kernel, c = 'linear', 0.3
        csvm = CascadeSVM(kernel=kernel, c=c)
        self.assertEqual(csvm._clf_params['kernel'], kernel)
        self.assertEqual(csvm._clf_params['C'], c)

        # # check for exception when incorrect kernel is passed
        # self.assertRaises(AttributeError, CascadeSVM(kernel='fake_kernel'))

    def test_fit(self):
        seed = 666
        file_ = "tests/files/libsvm/2"

        dataset = load_libsvm_file(file_, 10, 780)

        csvm = CascadeSVM(cascade_arity=3, max_iter=5,
                          tol=1e-4, kernel='linear', c=2, gamma=0.1,
                          check_convergence=True,
                          random_state=seed, verbose=False)

        csvm.fit(dataset)

        self.assertTrue(csvm.converged)

        csvm = CascadeSVM(cascade_arity=3, max_iter=1,
                          tol=1e-4, kernel='linear', c=2, gamma=0.1,
                          check_convergence=False,
                          random_state=seed, verbose=False)

        csvm.fit(dataset)
        self.assertFalse(csvm.converged)
        self.assertEqual(csvm.iterations, 1)

    def test_fit_default_gamma(self):
        """ Tests that the fit method converges when using gamma=auto on a
        toy dataset """
        seed = 666
        file_ = "tests/files/libsvm/2"

        dataset = load_libsvm_file(file_, 10, 780)

        csvm = CascadeSVM(cascade_arity=3, max_iter=5,
                          tol=1e-4, kernel='linear', c=2,
                          check_convergence=True,
                          random_state=seed, verbose=False)

        csvm.fit(dataset)

        self.assertTrue(csvm.converged)

        csvm = CascadeSVM(cascade_arity=3, max_iter=1,
                          tol=1e-4, kernel='linear', c=2, gamma=0.1,
                          check_convergence=False,
                          random_state=seed, verbose=False)

        csvm.fit(dataset)
        self.assertFalse(csvm.converged)
        self.assertEqual(csvm.iterations, 1)

    def test_predict(self):
        seed = 666

        dataset = Dataset(n_features=2)
        # negative points belong to class 1, positives to 0
        p1, p2, p3, p4 = [1, 2], [2, 1], [-1, -2], [-2, -1]
        dataset.append(Subset(np.array([p1, p4]), np.array([0, 1])))
        dataset.append(Subset(np.array([p3, p2]), np.array([1, 0])))

        csvm = CascadeSVM(cascade_arity=3, max_iter=10,
                          tol=1e-4, kernel='linear', c=2, gamma=0.1,
                          check_convergence=False,
                          random_state=seed, verbose=False)

        csvm.fit(dataset)

        # p5 should belong to class 0, p6 to class 1
        p5, p6 = np.array([1, 1]), np.array([-1, -1])

        test_set = load_data(x=np.array([p1, p2, p3, p4, p5, p6]),
                             subset_size=2)
        csvm.predict(test_set)

        l1, l2, l3, l4, l5, l6 = test_set.labels

        self.assertTrue(l1 == l2 == l5 == 0)
        self.assertTrue(l3 == l4 == l6 == 1)

    def test_score(self):
        seed = 666

        dataset = Dataset(n_features=2)
        # negative points belong to class 1, positives to 0
        p1, p2, p3, p4 = [1, 2], [2, 1], [-1, -2], [-2, -1]
        dataset.append(Subset(np.array([p1, p4]), np.array([0, 1])))
        dataset.append(Subset(np.array([p3, p2]), np.array([1, 0])))

        csvm = CascadeSVM(cascade_arity=3, max_iter=10,
                          tol=1e-4, kernel='rbf', c=2, gamma=0.1,
                          check_convergence=True,
                          random_state=seed, verbose=False)

        csvm.fit(dataset)

        # points are separable, scoring the training dataset should have 100%
        # accuracy
        test_set = load_data(x=np.array([p1, p2, p3, p4]), subset_size=2,
                             y=np.array([0, 0, 1, 1]))
        accuracy = csvm.score(test_set)

        self.assertEqual(accuracy, 1.0)

    def test_decision_func(self):
        seed = 666

        dataset = Dataset(n_features=2)
        # negative points belong to class 1, positives to 0
        # all points are in the x-axis
        p1, p2, p3, p4 = [0, 2], [0, 1], [0, -2], [0, -1]
        dataset.append(Subset(np.array([p1, p4]), np.array([0, 1])))
        dataset.append(Subset(np.array([p3, p2]), np.array([1, 0])))

        csvm = CascadeSVM(cascade_arity=3, max_iter=10,
                          tol=1e-4, kernel='rbf', c=2, gamma=0.1,
                          check_convergence=False,
                          random_state=seed, verbose=False)

        csvm.fit(dataset)

        # p1 should be equidistant to p3, and p2 to p4
        test_set = load_data(x=np.array([p1, p2, p3, p4]), subset_size=2)
        csvm.decision_function(test_set)
        d1, d2, d3, d4 = test_set.labels

        self.assertTrue(np.isclose(abs(d1) - abs(d3), 0))
        self.assertTrue(np.isclose(abs(d2) - abs(d4), 0))

        # p5 and p6 should be in the decision function (distance=0)
        p5, p6 = np.array([1, 0]), np.array([-1, 0])
        test_set = load_data(x=np.array([p5, p6]), subset_size=1)
        csvm.decision_function(test_set)
        d5, d6 = test_set.labels
        self.assertTrue(np.isclose(d5, 0))
        self.assertTrue(np.isclose(d6, 0))

    def test_sparse(self):
        """ Tests that C-SVM produces the same results with sparse and dense
        data"""
        seed = 666
        train = "tests/files/libsvm/3"

        train_sp = load_libsvm_file(train, 10, 780)
        train_d = load_libsvm_file(train, 10, 780, False)

        csvm_sp = CascadeSVM(random_state=seed)
        csvm_sp.fit(train_sp)
        csvm_d = CascadeSVM(random_state=seed)
        csvm_d.fit(train_d)

        sv_d = csvm_d._clf.support_vectors_
        sv_sp = csvm_sp._clf.support_vectors_.toarray()

        self.assertTrue(np.array_equal(sv_d, sv_sp))

        coef_d = csvm_d._clf.dual_coef_
        coef_sp = csvm_sp._clf.dual_coef_.toarray()

        self.assertTrue(np.array_equal(coef_d, coef_sp))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
