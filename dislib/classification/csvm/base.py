from uuid import uuid4

import numpy as np
from pycompss.api.api import compss_delete_object
from pycompss.api.api import compss_wait_on
from pycompss.api.parameter import COLLECTION_IN, Depth, Type
from pycompss.api.task import task
from scipy.sparse import hstack as hstack_sp
from scipy.sparse import issparse
from sklearn.svm import SVC

from dislib.data.array import Array


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
        self._hstack_f = None

        if kernel == "rbf":
            self._clf_params = {"kernel": kernel, "C": c, "gamma": gamma}
        else:
            self._clf_params = {"kernel": kernel, "C": c}

        try:
            self._kernel_f = getattr(self, CascadeSVM._name_to_kernel[kernel])
        except AttributeError:
            self._kernel_f = getattr(self, "_rbf_kernel")

    def fit(self, x, y):
        """ Fits a model using training data.

        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)
            Training samples.
        y : ds-array, shape=(n_samples, 1)
            Class labels of x.

        Returns
        -------
        self : object
        """

        self._check_xy(x, y)
        self._reset_model()
        self._set_gamma(x.shape[1])
        self._hstack_f = hstack_sp if x._sparse else np.hstack

        ids_list = [[_gen_ids(row._blocks)] for row in x._iterator(axis=0)]

        while not self._check_finished():
            self._do_iteration(x, y, ids_list)

            if self._check_convergence:
                self._check_convergence_and_update_w()
                self._print_iteration()

        return self

    def predict(self, x):
        """ Perform classification on samples.

        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)
            Input samples.

        Returns
        -------
        y : ds-array, shape(n_samples, 1)
            Class labels of x.
        """
        assert (self._clf is not None or self._svs is not None), \
            "Model has not been initialized. Call fit() first."

        y_list = []

        for row in x._iterator(axis=0):
            y_list.append([_predict(row._blocks, self._clf)])

        return Array(blocks=y_list, blocks_shape=(x._blocks_shape[0], 1),
                     shape=(x.shape[0], 1), sparse=False)

    def decision_function(self, x):
        """ Evaluates the decision function for the samples in x.

        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)
            Input samples.

        Returns
        -------
        df : ds-array, shape=(n_samples, 2)
            The decision function of the samples for each class in the model.
        """
        assert (self._clf is not None or self._svs is not None), \
            "Model has not been initialized. Call fit() first."

        df = []

        for row in x._iterator(axis=0):
            df.append([_decision_function(row._blocks, self._clf)])

        bshape = x._blocks_shape

        return Array(blocks=df, blocks_shape=(bshape[0], 1),
                     shape=(x.shape[0], 1), sparse=False)

    def score(self, x, y):
        """
        Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)
            Test samples.
        y : ds-array, shape=(n_samples, 1)
            True labels for x.

        Returns
        -------
        score : ds-array, shape=(1, 1)
            Mean accuracy of self.predict(x) wrt. y.
        """
        assert (self._clf is not None or self._svs is not None), \
            "Model has not been initialized. Call fit() first."

        partial_scores = []

        for x_row, y_row in zip(x._iterator(axis=0), y._iterator(axis=0)):
            partial = _score(x_row._blocks, y_row._blocks, self._clf)
            partial_scores.append(partial)

        score = [[_merge_scores(*partial_scores)]]
        return Array(blocks=score, blocks_shape=(1, 1), shape=(1, 1),
                     sparse=False)

    @staticmethod
    def _check_xy(x, y):
        # We force 'x' and 'y' to have the same number of row blocks. This
        # could be avoided by re-chunking 'y', or using slicing on 'y'
        # during the training process
        xshape = x._blocks_shape
        yshape = y._blocks_shape

        if len(xshape) != len(yshape) or \
                (isinstance(xshape, tuple) and xshape[0] != yshape[0]) or \
                (isinstance(xshape, list) and \
                 (xshape[0][0] != yshape[0][0] or \
                  xshape[-1][0] != yshape[-1][0])):
            raise AttributeError(
                "x and y must have the same number of blocks along the first "
                "axis")

        if x.shape[0] != y.shape[0]:
            raise AttributeError("The number of labels does not match the "
                                 "number of samples")

    def _reset_model(self):
        self.iterations = 0
        self.converged = False
        self._last_w = None
        self._clf = None
        self._svs = None
        self._sv_labels = None

    def _set_gamma(self, n_features):
        if self._gamma == "auto":
            self._gamma = 1. / n_features
            self._clf_params["gamma"] = self._gamma

    def _collect_clf(self):
        self._svs, self._sv_labels, self._clf = compss_wait_on(self._svs,
                                                               self._sv_labels,
                                                               self._clf)

    def _print_iteration(self):
        if self._verbose:
            print("Iteration %s of %s." % (self.iterations, self._max_iter))

    def _do_iteration(self, x, y, ids_list):
        q = []
        pars = self._clf_params
        arity = self._arity

        # first level
        for x_row, y_row, id_bk in zip(x._iterator(axis=0),
                                       y._iterator(axis=0), ids_list):
            x_data = x_row._blocks
            y_data = y_row._blocks
            ids = [id_bk]

            if self._svs is not None:
                x_data.append(self._svs)
                y_data.append([self._sv_labels])
                ids.append([self._sv_ids])

            _tmp = _train(x_data, y_data, ids, self._random_state, **pars)
            sv, sv_labels, sv_ids, self._clf = _tmp
            q.append((sv, sv_labels, sv_ids))

        # reduction
        while len(q) > arity:
            data = q[:arity]
            del q[:arity]

            x_data = [tup[0] for tup in data]
            y_data = [[tup[1]] for tup in data]
            ids = [[tup[2]] for tup in data]

            _tmp = _train(x_data, y_data, ids, self._random_state, **pars)
            sv, sv_labels, sv_ids, self._clf = _tmp
            q.append((sv, sv_labels, sv_ids))

            # delete partial results
            for partial in data:
                compss_delete_object(partial)

        # last layer
        x_data = [tup[0] for tup in q]
        y_data = [[tup[1]] for tup in q]
        ids = [[tup[2]] for tup in q]

        _tmp = _train(x_data, y_data, ids, self._random_state, **pars)
        self._svs, self._sv_labels, self._sv_ids, self._clf = _tmp

        self.iterations += 1

    def _check_finished(self):
        return self.iterations >= self._max_iter or self.converged

    def _lag_fast(self, vectors, labels, coef):
        set_sl = set(labels.ravel())
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

        vecs = self._hstack_f(self._svs)
        w = self._lag_fast(vecs, self._sv_labels, self._clf.dual_coef_)
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


@task(blocks={Type: COLLECTION_IN, Depth: 2}, returns=1)
def _gen_ids(blocks):
    samples = Array._merge_blocks(blocks)
    idx = [[uuid4().int] for _ in range(samples.shape[0])]
    return np.array(idx)


@task(x_list={Type: COLLECTION_IN, Depth: 2},
      y_list={Type: COLLECTION_IN, Depth: 2},
      id_list={Type: COLLECTION_IN, Depth: 2},
      returns=4)
def _train(x_list, y_list, id_list, random_state, **params):
    x, y, ids = _merge(x_list, y_list, id_list)

    clf = SVC(random_state=random_state, **params)
    clf.fit(X=x, y=y.ravel())

    sup = x[clf.support_]
    start, end = 0, 0
    sv = []

    for xi in x_list[0]:
        end += xi.shape[1]
        sv.append(sup[:, start:end])
        start = end

    sv_labels = y[clf.support_]
    sv_ids = ids[clf.support_]

    return sv, sv_labels, sv_ids, clf


@task(x_list={Type: COLLECTION_IN, Depth: 2}, returns=np.array)
def _predict(x_list, clf):
    x = Array._merge_blocks(x_list)
    return clf.predict(x).reshape(-1, 1)


@task(x_list={Type: COLLECTION_IN, Depth: 2}, returns=np.array)
def _decision_function(x_list, clf):
    x = Array._merge_blocks(x_list)
    return clf.decision_function(x).reshape(-1, 1)


@task(x_list={Type: COLLECTION_IN, Depth: 2},
      y_list={Type: COLLECTION_IN, Depth: 2}, returns=tuple)
def _score(x_list, y_list, clf):
    x = Array._merge_blocks(x_list)
    y = Array._merge_blocks(y_list)

    y_pred = clf.predict(x)
    equal = np.equal(y_pred, y.ravel())

    return np.sum(equal), x.shape[0]


@task(returns=float)
def _merge_scores(*partials):
    total_correct = 0.
    total_size = 0.

    for correct, size in partials:
        total_correct += correct
        total_size += size

    return total_correct / total_size


def _merge(x_list, y_list, id_list):
    samples = Array._merge_blocks(x_list)
    labels = Array._merge_blocks(y_list)
    sample_ids = Array._merge_blocks(id_list)

    _, uniques = np.unique(sample_ids, return_index=True)
    indices = np.argsort(uniques)
    uniques = uniques[indices]

    sample_ids = sample_ids[uniques]
    samples = samples[uniques]
    labels = labels[uniques]

    return samples, labels, sample_ids
