from time import time

import numpy as np
from pycompss.api.api import compss_barrier as barrier
from pycompss.api.api import compss_delete_object
from pycompss.api.api import compss_wait_on
from pycompss.api.task import task
from scipy.sparse import issparse
from sklearn.svm import SVC


class CascadeSVM(object):
    name_to_kernel = {"linear": "_linear_kernel", "rbf": "_rbf_kernel"}

    def __init__(self, split_times=False):
        """

        :param split_times: boolean, optional (default=False)
            Whether to compute read and fit times separately.
        """

        self.iterations = []
        self.converged = []

        self.read_time = 0
        self.fit_time = 0
        self.total_time = 0

        self._split_times = split_times
        self._cascade_arity = []
        self._max_iterations = []
        self._nchunks = []
        self._tol = []
        self._last_w = []
        self._clf = []
        self._data = []
        self._clf_params = []
        self._kernel_f = []

    def fit(self, X, cascade_arity=None, cascade_iterations=None, tol=None,
            kernel=None, c=None, gamma=None, check_convergence=True):
        """
        Fits one or more models using training data. The training process of
        each dataset is performed in parallel. The resulting models are stored
        in self._clf.

        :param X: list
            List of datasets
        :param cascade_arity: list (int), optional (default=2)
            Arities of the reductions of the input datasets X.
        :param cascade_iterations: list (int), optional (default=5)
            Maximum number of iterations to perform for each dataset.
        :param tol: list (float), optional (default=1e-3)
            Tolerance for the stopping criterion for each dataset.
        :param kernel: list (string), optional (default='rbf')
            Specifies the kernel type to be used in the algorithm. It must be
            one of 'linear' or 'rbf'.
        :param c: list (float), optional (default=1.0)
            Penalty parameter C of the error term for each dataset.
        :param gamma: list (float), optional (default='auto')
            Kernel coefficient for 'rbf'. If gamma is 'auto' then 1/n_features
            will be used instead.
        :param check_convergence: boolean, optional (default=True)
            Whether to test for convergence. If False, the algorithm will run
            for the number of iterations specified in load_data.
            Checking for convergence adds a synchronization point, and can
            negatively affect performance if multiple datasets are fit in
            parallel.
        """
        self._data = X
        ndata = len(X)
        self._set_defaults(ndata, cascade_arity, cascade_iterations, tol,
                           kernel, c, gamma)
        self.iterations = [0] * ndata
        self.converged = [False] * ndata
        self._last_w = [None] * ndata
        self._clf = [None] * ndata

        if self._split_times:
            barrier()
            self.read_time = time() - self.read_time
            self.fit_time = time()

        self._do_fit(check_convergence)

        barrier()

        if self._split_times:
            self.fit_time = time() - self.fit_time

        self.total_time = time() - self.total_time

    def predict(self, X, i=0):
        """
        Perform classification on samples in X using model i.
        
        :param X: array-like, shape (n_samples, n_features)
        :param i: int, optional (default=0)
            Model index in case multiple models have been built in parallel. 
        :return y_pred: array, shape (n_samples,)
            Class labels for samples in X.
        """

        if len(self._clf) > i and self._clf[i]:
            return self._clf[i].predict(X)
        else:
            raise Exception("Model %s has not been initialized. Try calling "
                            "fit first." % i)

    def decision_function(self, X, i=0):
        """
        Distance of the samples X to the ith separating hyperplane.
        
        :param X: array-like, shape (n_samples, n_features)
        :param i: int, optional (default=0)
            Model index in case multiple models have been built in parallel. 
        :return: array-like, shape (n_samples, n_classes * (n_classes-1) / 2)        
            Returns the decision function of the sample for each class in the
            ith model.
        """

        if len(self._clf) > i and self._clf[i]:
            return self._clf[i].decision_function(X)
        else:
            raise Exception("Model %s has not been initialized. Try calling "
                            "fit first." % i)

    def score(self, X, y, i=0):
        """
        Returns the mean accuracy on the given test data and labels using model
        i.
        
        :param X: array-like, shape = (n_samples, n_features)
            Test samples.
        :param y: array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.  
        :param i: int, optional (default=0)
            Model index in case multiple models have been built in parallel.
        :return score: Mean accuracy of self.predict(X, i) wrt. y.
        """

        if len(self._clf) > i and self._clf[i]:
            return self._clf[i].score(X, y)
        else:
            raise Exception("Model %s has not been initialized. Try calling "
                            "fit first." % i)

    def _set_defaults(self, len, cascade_arity, cascade_iterations, tol,
                      kernel, c, gamma):
        if cascade_arity:
            self._cascade_arity = cascade_arity
        else:
            self._cascade_arity = [2] * len
        if cascade_iterations:
            self._max_iterations = cascade_iterations
        else:
            self._max_iterations = [5] * len
        if tol:
            self._tol = tol
        else:
            self._tol = [1 ** -3] * len
        if not c:
            c = [1] * len
        if not gamma:
            gamma = ["auto"] * len
        if not kernel:
            kernel = ["rbf"] * len

        for k, z, g in zip(kernel, c, gamma):
            self._clf_params.append({"kernel": k, "C": z, "gamma": g})

            try:
                k_func = CascadeSVM.name_to_kernel[k]
                self._kernel_f.append(getattr(self, k_func))
            except AttributeError:
                self._kernel_f.append(getattr(self, "_rbf_kernel"))

    def _do_fit(self, check_convergence):
        feedback = [None] * len(self._data)
        finished = [False] * len(self._data)

        while not np.array(finished).all():
            for idx, chunks in enumerate(self._data):
                if not finished[idx]:
                    self._do_iteration(check_convergence, chunks, feedback, idx)

                    if check_convergence:
                        self._check_convergence_and_update_w(feedback[idx], idx)
                        self._print_iteration(idx)

                    finished[idx] = self._check_finished(idx)

        if not check_convergence:
            self._retrieve_clf(feedback)

    def _retrieve_clf(self, feedback):
        for idx, fb in enumerate(feedback):
            _ignore, self._clf[idx] = compss_wait_on(fb)

    def _print_iteration(self, idx):
        print("Dataset %s iteration %s of %s. \n" % (idx, self.iterations[idx],
                                                     self._max_iterations[idx]))

    def _do_iteration(self, check_convergence, chunks, feedback, idx):
        q = []
        arity = self._cascade_arity[idx]
        params = self._clf_params[idx]

        # first level
        for chunk in chunks:
            data = filter(None, [chunk, feedback[idx]])
            q.append(_train(False, *data, **params))

        # reduction
        while len(q) > arity:
            data = q[:arity]
            del q[:arity]

            q.append(_train(False, *data, **params))

            # delete partial results
            for d in data:
                compss_delete_object(d)

        # last layer
        if check_convergence:
            result = _train(True, *q, **params)
            feedback[idx], self._clf[idx] = compss_wait_on(result)
        else:
            feedback[idx] = _train(self._is_last_iteration(idx), *q, **params)

        self.iterations[idx] += 1

    def _is_last_iteration(self, idx):
        return self.iterations[idx] == self._max_iterations[idx] - 1

    def _check_finished(self, idx):
        return self.iterations[idx] >= self._max_iterations[idx] or \
               self.converged[idx]

    def _lagrangian_fast(self, vectors, labels, coef, idx):
        set_sl = set(labels)
        assert len(set_sl) == 2, "Only binary problem can be handled"
        new_sl = labels.copy()
        new_sl[labels == 0] = -1

        if issparse(coef):
            coef = coef.todense()

        c1, c2 = np.meshgrid(coef, coef)
        l1, l2 = np.meshgrid(new_sl, new_sl)
        double_sum = c1 * c2 * l1 * l2 * self._kernel_f[idx](vectors, idx)
        double_sum = double_sum.sum()
        w = -0.5 * double_sum + coef.sum()

        return w

    def _check_convergence_and_update_w(self, sv, idx):
        vectors = sv.vectors
        labels = sv.labels
        self.converged[idx] = False
        clf = self._clf[idx]
        print("Checking convergence for model %s:" % idx)

        if clf:
            w = self._lagrangian_fast(vectors, labels, clf.dual_coef_, idx)
            print("     Computed W %s" % w)

            if self._last_w[idx]:
                delta = np.abs((w - self._last_w[idx]) / self._last_w[idx])
                if delta < self._tol[idx]:
                    print("     Converged with delta: %s " % delta)
                    self.converged[idx] = True
                else:
                    print("     No convergence with delta: %s " % delta)
            else:
                print("     First iteration, not testing convergence.")
            self._last_w[idx] = w
            print()

    def _rbf_kernel(self, x, idx):
        # Trick: || x - y || ausmultipliziert
        sigmaq = -1 / (2 * self._clf_params[idx]["gamma"])
        n = x.shape[0]
        k = x.dot(x.T) / sigmaq

        if issparse(k):
            k = k.todense()

        d = np.diag(k).reshape((n, 1))
        k = k - np.ones((n, 1)) * d.T / 2
        k = k - d * np.ones((1, n)) / 2
        k = np.exp(k)
        return k

    @staticmethod
    def _linear_kernel(x):
        return np.dot(x, x.T)


@task(returns=tuple)
def _train(return_classifier, *args, **kwargs):
    data = _merge(*args)

    clf = SVC(random_state=1, **kwargs)
    clf.fit(X=data.vectors, y=data.labels)

    sup_vec = data[clf.support_]

    if return_classifier:
        return sup_vec, clf
    else:
        return sup_vec


def _merge(*args):
    d0 = args[0]

    for dx in args[1:]:
        d0.concatenate(dx, remove_duplicates=True)

    return d0
