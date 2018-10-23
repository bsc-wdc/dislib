import numpy as np
from pycompss.api.api import compss_delete_object
from pycompss.api.api import compss_wait_on
from pycompss.api.task import task
from scipy.sparse import issparse
from sklearn.svm import SVC


class CascadeSVM(object):
    name_to_kernel = {"linear": "_linear_kernel", "rbf": "_rbf_kernel"}

    def __init__(self, cascade_arity=2, cascade_iterations=5, tol=1 ** -3,
                 kernel="rbf", c=1, gamma="scale", check_convergence=True):
        """

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

        assert (gamma is "auto" or gamma is "scale" or type(gamma) == float
                or type(float(gamma)) == float), "Invalid gamma"
        assert (kernel is None or kernel in self.name_to_kernel.keys()), \
            "Incorrect kernel value [%s], available kernels are %s" % (
                kernel, self.name_to_kernel.keys())
        assert (c is None or type(c) == float or type(float(c)) == float), \
            "Incorrect C type [%s], type : %s" % (c, type(c))
        assert (type(tol) == float or type(float(tol)) == float), \
            "Incorrect tol type [%s], type : %s" % (tol, type(tol))
        assert cascade_arity > 1, "Cascade arity must be greater than 1"
        assert cascade_iterations > 0, "Max iterations must be greater than 0"
        assert type(check_convergence) == bool, "Invalid value in " \
                                                "check_convergence"

        self._reset_model()

        self._cascade_arity = cascade_arity
        self._max_iterations = cascade_iterations
        self._tol = tol
        self._check_convergence = check_convergence
        self._clf_params = {"kernel": kernel, "C": c, "gamma": gamma}

        try:
            self._kernel_f = getattr(self, CascadeSVM.name_to_kernel[kernel])
        except AttributeError:
            self._kernel_f = getattr(self, "_rbf_kernel")

    def fit(self, data):
        """
        Fits one or more models using training data. The training process of
        each dataset is performed in parallel. The resulting models are stored
        in self._clf.

        :param data: Dataset
            Dataset
        """
        self._reset_model()

        while not self._check_finished():
            self._do_iteration(data)

            if self._check_convergence:
                self._check_convergence_and_update_w()
                self._print_iteration()

    def predict(self, x):
        """
        Perform classification on samples in X using model i.
        
        :param x: array-like, shape (n_samples, n_features)
        :param i: int, optional (default=0)
            Model index in case multiple models have been built in parallel. 
        :return y_pred: array, shape (n_samples,)
            Class labels for samples in X.
        """

        assert (self._clf is not None or self._feedback is not None), \
            "Model has not been initialized. Call fit() first."

        if self._clf is None:
            self._retrieve_clf()

        return self._clf.predict(x)

    def decision_function(self, x):
        """
        Distance of the samples x to the ith separating hyperplane.
        
        :param x: array-like, shape (n_samples, n_features)
        :param i: int, optional (default=0)
            Model index in case multiple models have been built in parallel. 
        :return: array-like, shape (n_samples, n_classes * (n_classes-1) / 2)        
            Returns the decision function of the sample for each class in the
            ith model.
        """

        assert (self._clf is not None or self._feedback is not None), \
            "Model has not been initialized. Call fit() first."

        if self._clf is None:
            self._retrieve_clf()

        return self._clf.decision_function(x)

    def score(self, x, y):
        """
        Returns the mean accuracy on the given test data and labels using model
        i.
        
        :param x: array-like, shape = (n_samples, n_features)
            Test samples.
        :param y: array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.  
        :param i: int, optional (default=0)
            Model index in case multiple models have been built in parallel.
        :return score: Mean accuracy of self.predict(X, i) wrt. y.
        """

        assert (self._clf is not None or self._feedback is not None), \
            "Model has not been initialized. Call fit() first."

        if self._clf is None:
            self._retrieve_clf()

        return self._clf.score(x, y)

    def _reset_model(self):
        self.iterations = 0
        self.converged = False
        self._last_w = None
        self._clf = None
        self._feedback = None

    def _retrieve_clf(self):
        self._feedback, self._clf = compss_wait_on(self._feedback)

    def _print_iteration(self):
        print("Iteration %s of %s." % (self.iterations, self._max_iterations))

    def _do_iteration(self, data):
        q = []
        arity = self._cascade_arity
        params = self._clf_params

        # first level
        for partition in data:
            data = filter(None, [partition, self._feedback])
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
        get_clf = (self._check_convergence or self._is_last_iteration())
        self._feedback = _train(get_clf, *q, **params)
        self.iterations += 1

    def _is_last_iteration(self):
        return self.iterations == self._max_iterations - 1

    def _check_finished(self):
        return self.iterations >= self._max_iterations or self.converged

    def _lagrangian_fast(self, vectors, labels, coef):
        set_sl = set(labels)
        assert len(set_sl) == 2, "Only binary problem can be handled"
        new_sl = labels.copy()
        new_sl[labels == 0] = -1

        if issparse(coef):
            coef = coef.todense()

        c1, c2 = np.meshgrid(coef, coef)
        l1, l2 = np.meshgrid(new_sl, new_sl)
        double_sum = c1 * c2 * l1 * l2 * self._kernel_f(vectors)
        double_sum = double_sum.sum()
        w = -0.5 * double_sum + coef.sum()

        return w

    def _check_convergence_and_update_w(self):
        self._retrieve_clf()
        vectors = self._feedback.vectors
        labels = self._feedback.labels

        print("Checking convergence...")
        w = self._lagrangian_fast(vectors, labels, self._clf.dual_coef_)
        print("     Computed W %s" % w)

        if self._last_w:
            delta = np.abs((w - self._last_w) / self._last_w)
            if delta < self._tol:
                print("     Converged with delta: %s " % delta)
                self.converged = True
            else:
                print("     No convergence with delta: %s " % delta)
        else:
            print("     First iteration, not testing convergence.")
        self._last_w = w
        print()

    def _rbf_kernel(self, x):
        # Trick: || x - y || ausmultipliziert
        sigmaq = -1 / (2 * self._clf_params["gamma"])
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
