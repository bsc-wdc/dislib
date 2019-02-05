import numpy as np
from pycompss.api.api import compss_delete_object
from pycompss.api.api import compss_wait_on
from pycompss.api.parameter import INOUT
from pycompss.api.task import task
from scipy.sparse import issparse
from sklearn.svm import SVC


class CascadeSVM(object):
    """ Cascade Support Vector classification.

    Implements distributed support vector classification based on
    Graf et al. [1]_. The optimization process is carried out using
    scikit-learn's `SVC <http://scikit-learn.org/stable/modules/generated
    /sklearn.svm.SVC.html>`_.

    Parameters
    ----------
    cascade_arity : int, optional (default=2)
        Arity of the reduction process.
    max_iter : int, optional (default=5)
        Maximum number of iterations to perform.
    tol : float, optional (default=1e-3)
        Tolerance for the stopping criterion.
    kernel : string, optional (default='rbf')
        Specifies the kernel type to be used in the algorithm. Supported
        kernels are 'linear' and 'rbf'.
    c : float, optional (default=1.0)
        Penalty parameter C of the error term.
    gamma : float, optional (default='auto')
        Kernel coefficient for 'rbf'.

        Default is 'auto', which uses 1 / (n_features).
    check_convergence : boolean, optional (default=True)
        Whether to test for convergence. If False, the algorithm will run
        for cascade_iterations. Checking for convergence adds a
        synchronization point after each iteration.

        If ``check_convergence=False'' synchronization does not happen until
        a call to ``predict'', ``decision_function'' or ``score''. This can
        be useful to fit multiple models in parallel.
    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator used when shuffling the
        data for probability estimates. If int, random_state is the seed used
        by the random number generator; If RandomState instance, random_state
        is the random number generator; If None, the random number generator is
        the RandomState instance used by np.random.
    verbose : boolean, optional (default=False)
        Whether to print progress information.

    Attributes
    ----------
    iterations : int
        Number of iterations performed.
    converged : boolean
        Whether the model has converged.

    References
    ----------

    .. [1] Graf, H. P., Cosatto, E., Bottou, L., Dourdanovic, I., & Vapnik, V.
        (2005). Parallel support vector machines: The cascade svm. In Advances
        in neural information processing systems (pp. 521-528).

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    >>> y = np.array([1, 1, 2, 2])
    >>> from dislib.data import load_data
    >>> train_data = load_data(x=x, y=y, subset_size=4)
    >>> from dislib.classification import CascadeSVM
    >>> svm = CascadeSVM()
    >>> svm.fit(train_data)
    >>> test_data = load_data(x=np.array([[-0.8, -1]]), subset_size=1)
    >>> svm.predict(test_data)
    >>> print(test_data.labels)
    """
    _name_to_kernel = {"linear": "_linear_kernel", "rbf": "_rbf_kernel"}

    def __init__(self, cascade_arity=2, max_iter=5, tol=1e-3,
                 kernel="rbf", c=1, gamma='auto', check_convergence=True,
                 random_state=None, verbose=False):

        assert (gamma == "auto" or type(gamma) == float
                or type(float(gamma)) == float), "Invalid gamma"
        assert (kernel is None or kernel in self._name_to_kernel.keys()), \
            "Incorrect kernel value [%s], available kernels are %s" % (
                kernel, self._name_to_kernel.keys())
        assert (c is None or type(c) == float or type(float(c)) == float), \
            "Incorrect C type [%s], type : %s" % (c, type(c))
        assert (type(tol) == float or type(float(tol)) == float), \
            "Incorrect tol type [%s], type : %s" % (tol, type(tol))
        assert cascade_arity > 1, "Cascade arity must be greater than 1"
        assert max_iter > 0, "Max iterations must be greater than 0"
        assert type(check_convergence) == bool, "Invalid value in " \
                                                "check_convergence"

        self._reset_model()

        self._arity = cascade_arity
        self._max_iter = max_iter
        self._tol = tol
        self._check_convergence = check_convergence
        self._random_state = random_state
        self._verbose = verbose
        self._gamma = gamma

        if kernel == "rbf":
            self._clf_params = {"kernel": kernel, "C": c, "gamma": gamma}
        else:
            self._clf_params = {"kernel": kernel, "C": c}

        try:
            self._kernel_f = getattr(self, CascadeSVM._name_to_kernel[kernel])
        except AttributeError:
            self._kernel_f = getattr(self, "_rbf_kernel")

    def fit(self, dataset):
        """ Fits a model using training data.

        Parameters
        ----------
        dataset : Dataset
            Training data.
        """
        self._reset_model()
        self._set_gamma(dataset.n_features)

        while not self._check_finished():
            self._do_iteration(dataset)

            if self._check_convergence:
                self._check_convergence_and_update_w()
                self._print_iteration()

    def predict(self, dataset):
        """ Perform classification on samples in dataset. This method stores
        labels in dataset.

        Parameters
        ----------
        dataset : Dataset
        """
        assert (self._clf is not None or self._feedback is not None), \
            "Model has not been initialized. Call fit() first."

        for subset in dataset:
            _predict(subset, self._clf)

    def decision_function(self, dataset):
        """ Computes distances of the samples in dataset to the separating
        hyperplane. Distances are stored in dataset.labels.

        Parameters
        ----------
        dataset : Dataset
        """
        assert (self._clf is not None or self._feedback is not None), \
            "Model has not been initialized. Call fit() first."

        for subset in dataset:
            _decision_function(subset, self._clf)

    def score(self, dataset):
        """
        Returns the mean accuracy on the given test dataset. This method
        assumes dataset.labels are true labels.

        Parameters
        ----------
        dataset : Dataset
            Dataset where dataset.labels are true labels for
            dataset.samples.

        Returns
        -------
        score : Mean accuracy of self.predict(dataset) wrt. dataset.labels.
        """
        assert (self._clf is not None or self._feedback is not None), \
            "Model has not been initialized. Call fit() first."

        partial_scores = []

        for subset in dataset:
            partial_scores.append(_score(subset, self._clf))

        score = compss_wait_on(_merge_scores(*partial_scores))

        return score

    def _reset_model(self):
        self.iterations = 0
        self.converged = False
        self._last_w = None
        self._clf = None
        self._feedback = None

    def _set_gamma(self, n_features):
        if self._gamma == "auto":
            self._gamma = 1. / n_features
            self._clf_params["gamma"] = self._gamma

    def _collect_clf(self):
        self._feedback, self._clf = compss_wait_on(self._feedback, self._clf)

    def _print_iteration(self):
        if self._verbose:
            print("Iteration %s of %s." % (self.iterations, self._max_iter))

    def _do_iteration(self, dataset):
        q = []
        arity = self._arity
        params = self._clf_params

        # first level
        for subset in dataset:
            data = filter(None, [subset, self._feedback])
            q.append(_train(False, self._random_state, *data, **params)[0])

        # reduction
        while len(q) > arity:
            data = q[:arity]
            del q[:arity]

            q.append(_train(False, self._random_state, *data, **params)[0])

            # delete partial results
            for partial in data:
                compss_delete_object(partial)

        # last layer
        get_clf = (self._check_convergence or self._is_last_iteration())
        _out = _train(get_clf, self._random_state, *q, **params)
        self._feedback, self._clf = _out
        self.iterations += 1

    def _is_last_iteration(self):
        return self.iterations == self._max_iter - 1

    def _check_finished(self):
        return self.iterations >= self._max_iter or self.converged

    def _lagrangian_fast(self, vectors, labels, coef):
        set_sl = set(labels)
        assert len(set_sl) == 2, "Only binary problem can be handled"
        new_sl = labels.copy()
        new_sl[labels == 0] = -1

        if issparse(coef):
            coef = coef.toarray()

        c1, c2 = np.meshgrid(coef, coef)
        l1, l2 = np.meshgrid(new_sl, new_sl)
        double_sum = c1 * c2 * l1 * l2 * self._kernel_f(vectors)
        double_sum = double_sum.sum()
        w = -0.5 * double_sum + coef.sum()

        return w

    def _check_convergence_and_update_w(self):
        self._collect_clf()
        samples = self._feedback.samples
        labels = self._feedback.labels

        w = self._lagrangian_fast(samples, labels, self._clf.dual_coef_)
        delta = 0

        if self._last_w:
            delta = np.abs((w - self._last_w) / self._last_w)

            if delta < self._tol:
                self.converged = True

        if self._verbose:
            self._print_convergence(delta, w)

        self._last_w = w

    def _print_convergence(self, delta, w):
        print("Computed W %s" % w)
        if self._last_w:
            print("Checking convergence...")

            if self.converged:
                print("     Converged with delta: %s " % delta)
            else:
                print("     No convergence with delta: %s " % delta)

    def _rbf_kernel(self, x):
        # Trick: || x - y || ausmultipliziert
        sigmaq = -1 / (2 * self._gamma)
        n = x.shape[0]
        k = x.dot(x.T) / sigmaq

        if issparse(k):
            k = k.toarray()

        d = np.diag(k).reshape((n, 1))
        k = k - np.ones((n, 1)) * d.T / 2
        k = k - d * np.ones((1, n)) / 2
        k = np.exp(k)
        return k

    @staticmethod
    def _linear_kernel(x):
        return np.dot(x, x.T)


@task(returns=2)
def _train(return_classifier, random_state, *subsets, **params):
    subset = _merge(*subsets)

    clf = SVC(random_state=random_state, **params)
    clf.fit(X=subset.samples, y=subset.labels)

    sup_vec = subset[clf.support_]

    if return_classifier:
        return sup_vec, clf
    else:
        return sup_vec, None


@task(subset=INOUT)
def _predict(subset, clf):
    labels = clf.predict(subset.samples)
    subset.labels = labels


@task(subset=INOUT)
def _decision_function(subset, clf):
    distances = clf.decision_function(subset.samples)
    subset.labels = distances


@task(returns=tuple)
def _score(subset, clf):
    labels = clf.predict(subset.samples)
    equal = np.equal(labels, subset.labels)

    return np.sum(equal), subset.samples.shape[0]


@task(returns=float)
def _merge_scores(*partials):
    total_correct = 0.
    total_size = 0.

    for correct, size in partials:
        total_correct += correct
        total_size += size

    return total_correct / total_size


def _merge(*subsets):
    set0 = subsets[0].copy()

    for setx in subsets[1:]:
        set0.concatenate(setx, remove_duplicates=True)

    return set0
