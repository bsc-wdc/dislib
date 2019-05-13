import warnings

import numpy as np
from numpy.random.mtrand import RandomState
from pycompss.api.api import compss_wait_on, compss_delete_object
from pycompss.api.parameter import INOUT
from scipy import linalg
from scipy.sparse import issparse
from sklearn.utils import validation
from sklearn.utils.extmath import row_norms
from sklearn.utils.fixes import logsumexp
from sklearn.exceptions import ConvergenceWarning

from pycompss.api.task import task

from dislib.cluster import KMeans
from dislib.data import Dataset, Subset


@task(returns=1)
def _estimate_parameters_subset(subset, resp):
    subsample = subset.samples
    resp = resp.samples
    nk_ss = resp.sum(axis=0)
    if issparse(subsample):
        means_ss = subsample.T.dot(resp).T
    else:
        means_ss = np.matmul(resp.T, subsample)

    return subsample.shape[0], nk_ss, means_ss


def _reduce_estimate_parameters(partials, arity):
    while len(partials) > 1:
        partials_subset = partials[:arity]
        partials = partials[arity:]
        partials.append(_merge_estimate_parameters(*partials_subset))
    return _finalize_parameters(partials[0])


@task(returns=1)
def _merge_estimate_parameters(*subsets_params):
    n_samples = sum(params[0] for params in subsets_params)
    nk = sum(params[1] for params in subsets_params)
    means = sum(params[2] for params in subsets_params)
    return n_samples, nk, means


@task(returns=3)
def _finalize_parameters(params):
    n_samples = params[0]
    nk = params[1]
    nk += 10 * np.finfo(nk.dtype).eps
    means = params[2] / nk[:, np.newaxis]
    weights = nk / n_samples
    return weights, nk, means


def _estimate_covariances(dataset, resp, nk, means, reg_covar, covar_type,
                          arity):
    """Estimate the covariances and compute the cholesky precisions.

    Parameters
    ----------
    dataset : dislib.data.Dataset
        The input data.
    resp : array-like, shape (n_samples, n_components)
        The responsibilities for each data sample in X.
    nk : array-like, shape (n_components,)
        The numbers of data samples (weighted by responsibility) in the
        current components.
    means : array-like, shape (n_components, n_features)
        The centers of the current components.
    reg_covar : float
        The regularization added to the diagonal of the covariance matrices.
    covar_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.
    arity : int
        Arity of the reductions.

    Returns
    -------
    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.
    cholesky_precisions : array-like, shape (n_components,)
        The numbers of data samples in the current components.
    """
    partials = []
    partial_covar = {
        "full": _partial_covar_full,
        "tied": lambda r, s, m: _partial_covar_tied(s),
        "diag": _partial_covar_diag,
        "spherical": _partial_covar_diag  # uses same partial as diag
        }[covar_type]
    for ss, resp_ss in zip(dataset, resp):
        partials.append(partial_covar(resp_ss, ss, means))
    while len(partials) > 1:
        partials_chunk = partials[:arity]
        partials = partials[arity:]
        partials.append(_sum_covar_partials(*partials_chunk))
    finalize_covariances = {
        "full": lambda t, r, n, m, p: _finalize_covar_full(t, r, n, p),
        "tied": _finalize_covar_tied,
        "diag": _finalize_covar_diag,
        "spherical": _finalize_covar_spherical
        }[covar_type]
    return finalize_covariances(covar_type, reg_covar, nk, means, partials[0])


@task(returns=1)
def _partial_covar_full(resp, subset, means):
    subsample = subset.samples
    resp = resp.samples
    n_components, n_features = means.shape
    covariances = np.empty((n_components, n_features, n_features))
    for k in range(n_components):
        if issparse(subsample):
            diff = (x - means[k] for x in subsample)
            partial_covs = (np.dot(r*d.T, d) for d, r in zip(diff, resp[:, k]))
            covariances[k] = sum(partial_covs)
        else:
            diff = subsample - means[k]
            covariances[k] = np.dot(resp[:, k] * diff.T, diff)
    return covariances


@task(returns=1)
def _partial_covar_tied(subset):
    subsample = subset.samples
    if issparse(subsample):
        avg_sample_2 = subsample.T.dot(subsample)
    else:
        avg_sample_2 = np.dot(subsample.T, subsample)
    return avg_sample_2


@task(returns=1)
def _partial_covar_diag(resp, subset, means):
    subsample = subset.samples
    resp = resp.samples
    if issparse(subsample):
        avg_resp_sample_2 = subsample.multiply(subsample).T.dot(resp).T
        avg_sample_means = means * subsample.T.dot(resp).T
    else:
        avg_resp_sample_2 = np.dot(resp.T, subsample * subsample)
        avg_sample_means = means * np.dot(resp.T, subsample)
    return avg_resp_sample_2 - 2 * avg_sample_means


@task(returns=1)
def _sum_covar_partials(*covar_partials):
    return sum(covar_partials)


@task(returns=2)
def _finalize_covar_full(covar_type, reg_covar, nk, covariances):
    n_components, n_features, _ = covariances.shape
    for k in range(n_components):
        covariances[k] /= nk[k]
        covariances[k].flat[::n_features + 1] += reg_covar
    precisions_chol = _compute_precision_cholesky(covariances, covar_type)
    return covariances, precisions_chol


@task(returns=2)
def _finalize_covar_tied(covar_type, reg_covar, nk, means, covariances):
    avg_means2 = np.dot(nk * means.T, means)
    covariances -= avg_means2
    covariances /= nk.sum()
    covariances.flat[::len(covariances) + 1] += reg_covar
    precisions_chol = _compute_precision_cholesky(covariances, covar_type)
    return covariances, precisions_chol


@task(returns=2)
def _finalize_covar_diag(covar_type, reg_covar, nk, means, covariances):
    covariances /= nk[:, np.newaxis]
    covariances += means ** 2
    covariances += reg_covar
    precisions_chol = _compute_precision_cholesky(covariances, covar_type)
    return covariances, precisions_chol


@task(returns=2)
def _finalize_covar_spherical(covar_type, reg_covar, nk, means, covariances):
    covariances /= nk[:, np.newaxis]
    covariances += means ** 2
    covariances += reg_covar
    covariances = covariances.mean(1)
    precisions_chol = _compute_precision_cholesky(covariances, covar_type)
    return covariances, precisions_chol


def _compute_precision_cholesky(covariances, covariance_type):
    """Compute the Cholesky decomposition of the precisions.

    Parameters
    ----------
    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.

    Returns
    -------
    precisions_cholesky : array-like
        The cholesky decomposition of sample precisions of the current
        components. The shape depends of the covariance_type.
    """
    estimate_precision_error_message = (
        "Fitting the mixture model failed because some components have "
        "ill-defined empirical covariance (for instance caused by singleton "
        "or collapsed samples). Try to decrease the number of components, "
        "or increase reg_covar.")

    if covariance_type in 'full':
        n_components, n_features, _ = covariances.shape
        precisions_chol = np.empty((n_components, n_features, n_features))
        for k, covariance in enumerate(covariances):
            try:
                cov_chol = linalg.cholesky(covariance, lower=True)
            except linalg.LinAlgError:
                raise ValueError(estimate_precision_error_message)
            precisions_chol[k] = linalg.solve_triangular(cov_chol,
                                                         np.eye(n_features),
                                                         lower=True).T
    elif covariance_type == 'tied':
        _, n_features = covariances.shape
        try:
            cov_chol = linalg.cholesky(covariances, lower=True)
        except linalg.LinAlgError:
            raise ValueError(estimate_precision_error_message)
        precisions_chol = linalg.solve_triangular(cov_chol,
                                                  np.eye(n_features),
                                                  lower=True).T
    else:
        if np.any(np.less_equal(covariances, 0.0)):
            raise ValueError(estimate_precision_error_message)
        precisions_chol = 1. / np.sqrt(covariances)
    return precisions_chol


@task(returns=1)
def _sum_log_prob_norm(*partials):
    total, count = map(sum, zip(*partials))
    return total, count


@task(returns=1)
def _finalize_sum_log_prob_norm(*partials):
    total, count = map(sum, zip(*partials))
    return total / count


@task(returns=2)
def _estimate_responsibilities(subset, weights, means, precisions_cholesky,
                               covariance_type):
    """Estimate log-likelihood and responsibilities for a subset.

    Compute the sum of log-likelihoods, the count of samples, and the
    responsibilities for each sample in the subset with respect to the
    current state of the model.

    Parameters
    ----------
    subset : dislib.datat.Subset
    weights : array-like, shape (n_components,)
        The weights of the current components.
    means : array-like, shape (n_components, n_features)
        The centers of the current components.
    precisions_cholesky : array-like
        The cholesky decomposition of sample precisions of the current
        components. The shape depends of the covariance_type.
    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.

    Returns
    -------
    log_prob_norm_subset : tuple
        tuple(sum, count) for log p(subset)

    responsibilities : dislib.data.Subset
    """
    weighted_log_prob = _estimate_weighted_log_prob(subset, weights, means,
                                                    precisions_cholesky,
                                                    covariance_type)
    log_prob_norm = logsumexp(weighted_log_prob, axis=1)
    log_prob_norm_sum = np.sum(log_prob_norm)
    count = len(log_prob_norm)
    with np.errstate(under='ignore'):
        # ignore underflow
        resp = np.exp(weighted_log_prob - log_prob_norm[:, np.newaxis])
    return (log_prob_norm_sum, count), Subset(resp)


def _estimate_weighted_log_prob(subset, weights, means, precisions_cholesky,
                                covariance_type):
    return _estimate_log_gaussian_prob(subset.samples, means,
                                       precisions_cholesky, covariance_type)\
           + _estimate_log_weights(weights)


def _estimate_log_weights(weights):
    return np.log(weights)


def _estimate_log_gaussian_prob(x, means, precisions_chol, covariance_type):
    """Estimate the log Gaussian probability.

    Parameters
    ----------
    x : array-like or csr_matrix, shape (n_samples, n_features)

    means : array-like, shape (n_components, n_features)

    precisions_chol : array-like,
        Cholesky decompositions of the precision matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    covariance_type : {'full', 'tied', 'diag', 'spherical'}

    Returns
    -------
    log_prob : array, shape (n_samples, n_components)
    """
    n_samples, n_features = x.shape
    n_components, _ = means.shape
    # det(precision_chol) is half of det(precision)
    log_det = _compute_log_det_cholesky(
        precisions_chol, covariance_type, n_features)

    if covariance_type == 'full':
        log_prob = np.empty((n_samples, n_components))
        for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
            if issparse(x):
                y = x.dot(prec_chol) - np.dot(mu, prec_chol)
            else:
                y = np.matmul(x, prec_chol) - np.dot(mu, prec_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)

    elif covariance_type == 'tied':
        log_prob = np.empty((n_samples, n_components))
        for k, mu in enumerate(means):
            if issparse(x):
                y = x.dot(precisions_chol) - np.dot(mu, precisions_chol)
            else:
                y = np.dot(x, precisions_chol) - np.dot(mu, precisions_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)

    elif covariance_type == 'diag':
        precisions = precisions_chol ** 2
        if issparse(x):
            log_prob = (np.sum((means ** 2 * precisions), 1) -
                        2. * (x * (means * precisions).T) +
                        x.multiply(x).dot(precisions.T))
        else:
            log_prob = (np.sum((means ** 2 * precisions), 1) -
                        2. * np.dot(x, (means * precisions).T) +
                        np.dot(x ** 2, precisions.T))

    elif covariance_type == 'spherical':
        precisions = precisions_chol ** 2
        if issparse(x):
            log_prob = (np.sum(means ** 2, 1) * precisions -
                        2 * (x * (means.T * precisions)) +
                        np.outer(row_norms(x, squared=True), precisions))
        else:
            log_prob = (np.sum(means ** 2, 1) * precisions -
                        2 * np.dot(x, means.T * precisions) +
                        np.outer(row_norms(x, squared=True), precisions))
    else:   # pragma: no cover
        raise ValueError()
    return -.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det


def _compute_log_det_cholesky(matrix_chol, covariance_type, n_features):
    """Compute the log-det of the cholesky decomposition of matrices.

    Parameters
    ----------
    matrix_chol : array-like,
        Cholesky decompositions of the matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    covariance_type : {'full', 'tied', 'diag', 'spherical'}

    n_features : int
        Number of features.

    Returns
    -------
    log_det_precision_chol : array-like, shape (n_components,)
        The determinant of the precision matrix for each component.
    """
    if covariance_type == 'full':
        n_components, _, _ = matrix_chol.shape
        log_det_chol = (np.sum(np.log(
            matrix_chol.reshape(
                n_components, -1)[:, ::n_features + 1]), 1))

    elif covariance_type == 'tied':
        log_det_chol = (np.sum(np.log(np.diag(matrix_chol))))

    elif covariance_type == 'diag':
        log_det_chol = (np.sum(np.log(matrix_chol), axis=1))

    else:
        log_det_chol = n_features * (np.log(matrix_chol))

    return log_det_chol


def _assign_predictions(dataset, responsabilities):
    for subset, resp in zip(dataset, responsabilities):
        _assign_subset_predictions(subset, resp)


@task(subset=INOUT)
def _assign_subset_predictions(subset, responsabilities):
    subset.labels = responsabilities.samples.argmax(axis=1)


class GaussianMixture:
    """Gaussian mixture model.

    Estimates the parameters of a Gaussian mixture model probability function
    that fits the data. Allows clustering.

    Parameters
    ----------
    n_components : int, optional (default=1)
        The number of components.
    covariance_type : str, (default='full')
        String describing the type of covariance parameters to use.
        Must be one of::

            'full' (each component has its own general covariance matrix),
            'tied' (all components share the same general covariance matrix),
            'diag' (each component has its own diagonal covariance matrix),
            'spherical' (each component has its own single variance).
    check_convergence : boolean, optional (default=True)
        Whether to test for convergence at the end of each iteration. Setting
        it to False removes control dependencies, allowing fitting this model
        in parallel with other tasks.
    tol : float, defaults to 1e-3.
        The convergence threshold. If the absolute change of the lower bound
        respect to the previous iteration is below this threshold, the
        iterations will stop. Ignored if `check_convergence` is False.
    reg_covar : float, defaults to 1e-6.
        Non-negative regularization added to the diagonal of covariance.
        Allows to assure that the covariance matrices are all positive.
    max_iter : int, defaults to 100.
        The number of EM iterations to perform.
    init_params : {'kmeans', 'random'}, defaults to 'kmeans'.
        The method used to initialize the weights, the means and the
        precisions. This method defines the responsibilities and a maximization
        step gives the model parameters. This is not used if `weights_init`,
        `means_init` and `precisions_init` are all provided.
        Must be one of::

            'kmeans' : responsibilities are initialized using kmeans,
            'random' : responsibilities are initialized randomly.
    weights_init : array-like, shape (n_components, ), optional
        The user-provided initial weights, defaults to None.
        If None, weights are initialized using the `init_params` method.
    means_init : array-like, shape (n_components, n_features), optional
        The user-provided initial means, defaults to None.
        If None, means are initialized using the `init_params` method.
    precisions_init : array-like, optional.
        The user-provided initial precisions (inverse of the covariance
        matrices), defaults to None.
        If None, precisions are initialized using the `init_params` method.
        The shape depends on 'covariance_type'::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'
    random_state : int, RandomState or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    arity : int, optional (default=50)
        Arity of the reductions.
    verbose: boolean, optional (default=False)
        Whether to print progress information.

    Attributes
    ----------
    weights_ : array-like, shape (n_components,)
        The weight of each mixture component.
    means_ : array-like, shape (n_components, n_features)
        The mean of each mixture component.
    covariances_ : array-like
        The covariance of each mixture component.
        The shape depends on `covariance_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'
    precisions_cholesky_ : array-like
        The cholesky decomposition of the precision matrices of each mixture
        component. A precision matrix is the inverse of a covariance matrix.
        A covariance matrix is symmetric positive definite so the mixture of
        Gaussian can be equivalently parameterized by the precision matrices.
        Storing the precision matrices instead of the covariance matrices makes
        it more efficient to compute the log-likelihood of new samples at test
        time. The shape depends on `covariance_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'
    converged_ : bool
        True if `check_convergence` is True and convergence is reached, False
        otherwise.
    n_iter : int
        Number of EM iterations done.
    lower_bound_ : float
        Lower bound value on the log-likelihood of the training data with
        respect to the model.

    Examples
    --------
    >>> from pycompss.api.api import compss_wait_on
    >>> from dislib.cluster import GaussianMixture
    >>> from dislib.data import load_data
    >>> import numpy as np
    >>> x = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    >>> train_data = load_data(x=x, subset_size=2)
    >>> gm = GaussianMixture(n_components=2, random_state=0)
    >>> gm.fit_predict(train_data)
    >>> print(train_data.labels)
    >>> test_data = load_data(x=np.array([[0, 0], [4, 4]]), subset_size=2)
    >>> gm.predict(test_data)
    >>> print(test_data.labels)
    >>> print(compss_wait_on(gm.means_))
    """

    def __init__(self, n_components=1, covariance_type='full',
                 check_convergence=True, tol=1e-3, reg_covar=1e-6,
                 max_iter=100, init_params='kmeans', weights_init=None,
                 means_init=None, precisions_init=None, arity=50,
                 verbose=False, random_state=None):

        self.n_components = n_components
        self._check_convergence = check_convergence
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.init_params = init_params
        self._arity = arity
        self._verbose = verbose
        self.random_state = random_state
        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init

    def fit(self, dataset):
        """Estimate model parameters with the EM algorithm.

        Iterates between E-steps and M-steps until convergence or until
        `max_iter` iterations are reached. It estimates the model parameters
        `weights_`, `means_` and `covariances_`.

        Parameters
        ----------
        dataset : dislib.data.Dataset
            Data points.

        Warns
        -----
        ConvergenceWarning
            If `tol` is not None and `max_iter` iterations are reached without
            convergence.
        """
        self._check_initial_parameters()

        self.converged_ = False
        self.n_iter = 0

        random_state = validation.check_random_state(self.random_state)

        self._initialize_parameters(dataset, random_state)
        self.lower_bound_ = -np.infty
        if self._verbose:
            print("GaussianMixture EM algorithm start")
        for self.n_iter in range(1, self.max_iter + 1):
            prev_lower_bound = self.lower_bound_

            self.lower_bound_, resp = self._e_step(dataset)
            self._m_step(dataset, resp)
            for resp_subset in resp:
                compss_delete_object(resp_subset)

            if self._check_convergence:
                self.lower_bound_ = compss_wait_on(self.lower_bound_)
                diff = abs(self.lower_bound_ - prev_lower_bound)

                if self._verbose:
                    iter_msg_template = "Iteration %s - Convergence crit. = %s"
                    print(iter_msg_template % (self.n_iter, diff))

                if diff < self.tol:
                    self.converged_ = True
                    break

        if self._check_convergence and not self.converged_:
            warnings.warn('The algorithm did not converge. '
                          'Try different init parameters, '
                          'or increase max_iter, tol '
                          'or check for degenerate data.',
                          ConvergenceWarning)

    def fit_predict(self, dataset):
        """Estimate model parameters and predict labels of the same dataset.

        Fits the model and, after fitting, uses the model to predict labels for
        the same training dataset.

        Parameters
        ----------
        dataset : dislib.data.Dataset
            Data points.

        Warns
        -----
        ConvergenceWarning
            If `tol` is not None and `max_iter` iterations are reached without
            convergence.
        """
        self.fit(dataset)
        self.predict(dataset)

    def predict(self, dataset):
        """Predict labels for the given dataset using trained model.

        Parameters
        ----------
        dataset : dislib.data.Dataset
            Data points.

        """
        validation.check_is_fitted(self,
                                   ['weights_', 'means_',
                                    'precisions_cholesky_'])
        _, resp = self._e_step(dataset)
        _assign_predictions(dataset, resp)

    def _e_step(self, dataset):
        """E step.

        Parameters
        ----------
        dataset : dislib.data.Dataset

        Returns
        -------
        log_prob_norm : float
            Mean of the logarithms of the probabilities of each sample in the
            dataset.

        responsibility : dislib.data.Dataset
            Posterior probabilities (or responsibilities) of each sample in the
            dataset.
        """
        log_prob_norm = []
        resp = Dataset(self.n_components)
        for s in dataset:
            log_prob_norm_s, resp_s = self._estimate_prob_resp(s)
            log_prob_norm.append(log_prob_norm_s)
            resp.append(resp_s)
        return self._reduce_log_prob_norm(log_prob_norm), resp

    def _estimate_prob_resp(self, subset):
        """Estimate log-likelihood and responsibilities for a subset.

        Compute the sum of log-likelihoods, the count of samples, and the
        responsibilities for each sample in the subset with respect to the
        current state of the model.

        Parameters
        ----------
        subset : dislib.data.Subset

        Returns
        -------
        log_prob_norm_subset : tuple
            tuple(sum, count) for log p(subset)

        responsibilities : dislib.data.Subset
            responsibilities for each sample and component
        """
        return _estimate_responsibilities(subset, self.weights_, self.means_,
                                          self.precisions_cholesky_,
                                          self.covariance_type)

    def _reduce_log_prob_norm(self, partials):
        while len(partials) > self._arity:
            partials_subset = partials[:self._arity]
            partials = partials[self._arity:]
            partials.append(_sum_log_prob_norm(*partials_subset))
        return _finalize_sum_log_prob_norm(*partials)

    def _m_step(self, dataset, resp):
        """M step.

        Parameters
        ----------
        dataset : dislib.data.Dataset

        resp : dislib.data.Dataset
            Posterior probabilities (or responsibilities) of the point of each
            sample in the dataset.
        """
        weights, nk, means = self._estimate_parameters(dataset, resp)
        self.weights_ = weights
        self.means_ = means

        cov, p_c = _estimate_covariances(dataset, resp, nk, means,
                                         self.reg_covar, self.covariance_type,
                                         self._arity)

        self.covariances_ = cov
        self.precisions_cholesky_ = p_c

    def _estimate_parameters(self, dataset, resp):
        """Estimate the Gaussian distribution weights and means.

        Parameters
        ----------
        dataset : dislib.data.Dataset
            The input data.
        resp : array-like, shape (n_samples, n_components)
            The responsibilities for each data sample in X.

        Returns
        -------
        weights : array-like, shape (n_components,)
            The weights of the current components.
        nk : array-like, shape (n_components,)
            The numbers of data samples (weighted by responsibility) in the
            current components.
        means : array-like, shape (n_components, n_features)
            The centers of the current components.
        """
        subsets_params = []
        for ss, resp_ss in zip(dataset, resp):
            ss_params = _estimate_parameters_subset(ss, resp_ss)
            subsets_params.append(ss_params)
        return _reduce_estimate_parameters(subsets_params, self._arity)

    def _check_initial_parameters(self):
        """Check values of the basic parameters."""
        if self.n_components < 1:
            raise ValueError("Invalid value for 'n_components': %d "
                             "Estimation requires at least one component"
                             % self.n_components)

        if self.tol < 0.:
            raise ValueError("Invalid value for 'tol': %.5f "
                             "Tolerance used by the EM must be non-negative"
                             % self.tol)

        if self.max_iter < 1:
            raise ValueError("Invalid value for 'max_iter': %d "
                             "Estimation requires at least one iteration"
                             % self.max_iter)

        if self.reg_covar < 0.:
            raise ValueError("Invalid value for 'reg_covar': %.5f "
                             "regularization on covariance must be "
                             "non-negative"
                             % self.reg_covar)

        if self.covariance_type not in ['spherical', 'tied', 'diag', 'full']:
            raise ValueError("Invalid value for 'covariance_type': %s "
                             "'covariance_type' should be in "
                             "['spherical', 'tied', 'diag', 'full']"
                             % self.covariance_type)

        self._prepare_init_parameters()
        n = self.n_components
        if self.weights_init is not None:
            shape = self.weights_init.shape
            if shape != (n,):
                raise ValueError("n_components=%d, " % n +
                                 "weights_init.shape=%s, " % str(shape) +
                                 "weights_init.shape should be "
                                 "(n_components,)")
        if self.means_init is not None:
            shape = self.means_init.shape
            if len(shape) != 2 or shape[0] != n:
                raise ValueError("n_components=%d, " % n +
                                 "means_init.shape=%s, " % str(shape) +
                                 "means_init.shape should be "
                                 "(n_components, n_features)")
        if self.precisions_init is not None:
            shape = self.precisions_init.shape
            cov_type = self.covariance_type
            if cov_type == 'spherical':
                if shape != (n,):
                    raise ValueError("n_components=%d, " % n +
                                     "precisions_init.shape=%s, " % str(shape)
                                     +
                                     "precisions_init.shape should be "
                                     "(n_components,) for "
                                     "covariance_type='spherical'")
            elif cov_type == 'tied':
                if len(shape) != 2 or shape[0] != shape[1]:
                    raise ValueError("precisions_init.shape=%s, " % str(shape)
                                     +
                                     "precisions_init.shape should be "
                                     "(n_features, n_features) for "
                                     "covariance_type='tied'")
            elif cov_type == 'diag':
                if len(shape) != 2 or shape[0] != n:
                    raise ValueError("n_components=%d, " % n +
                                     "precisions_init.shape=%s, " % str(shape)
                                     +
                                     "precisions_init.shape should be "
                                     "(n_components, n_features) for "
                                     "covariance_type='diag'")
            elif cov_type == 'full':
                if len(shape) != 3 or shape[0] != n or shape[1] != shape[2]:
                    raise ValueError("n_components=%d, " % n +
                                     "precisions_init.shape=%s, " % str(shape)
                                     +
                                     "precisions_init.shape should be "
                                     "(n_components, n_features, n_features) "
                                     "for covariance_type='full'")
        if self.means_init is not None and self.precisions_init is not None:
            if self.covariance_type in ('tied', 'diag', 'full'):
                if self.means_init.shape[1] != self.precisions_init.shape[1]:
                    raise ValueError("n_features mismatch in the dimensions "
                                     "of 'means_init' and 'precisions_init'")

    def _prepare_init_parameters(self):
        if isinstance(self.weights_init, (list, tuple)):
            self.weights_init = np.array(self.weights_init)
        if isinstance(self.means_init, (list, tuple)):
            self.means_init = np.array(self.means_init)
        if isinstance(self.precisions_init, (list, tuple)):
            self.precisions_init = np.array(self.precisions_init)

    def _initialize_parameters(self, dataset, random_state):
        """Initialization of the Gaussian mixture parameters.

        Parameters
        ----------
        dataset : dislib.data.Dataset

        random_state : RandomState
            A random number generator instance.
        """
        if self.weights_init is not None:
            self.weights_ = self.weights_init / np.sum(self.weights_init)
        if self.means_init is not None:
            self.means_ = self.means_init
        if self.precisions_init is not None:
            if self.covariance_type == 'full':
                self.precisions_cholesky_ = np.array(
                    [linalg.cholesky(prec_init, lower=True)
                     for prec_init in self.precisions_init])
            elif self.covariance_type == 'tied':
                self.precisions_cholesky_ = linalg.cholesky(
                    self.precisions_init, lower=True)
            else:
                self.precisions_cholesky_ = self.precisions_init
        initialize_params = (self.weights_init is None or
                             self.means_init is None or
                             self.precisions_init is None)
        if initialize_params:
            n_components = self.n_components
            resp = Dataset(n_components)
            if self.init_params == 'kmeans':
                if self._verbose:
                    print("KMeans initialization start")
                seed = random_state.randint(0, int(1e8))
                kmeans = KMeans(n_clusters=n_components, random_state=seed,
                                verbose=self._verbose)
                kmeans.fit_predict(dataset)
                self.kmeans = kmeans
                for labeled_subset in dataset:
                    resp.append(_resp_subset(labeled_subset, n_components))
            elif self.init_params == 'random':
                chunks = len(dataset)
                seeds = random_state.randint(np.iinfo(np.int32).max,
                                             size=chunks)
                for i in range(chunks):
                    subset = dataset[i]
                    resp.append(_random_resp_subset(subset, n_components,
                                                    seeds[i]))
            else:
                raise ValueError("Unimplemented initialization method '%s'"
                                 % self.init_params)

            weights, nk, means = self._estimate_parameters(dataset, resp)
            if self.means_init is None:
                self.means_ = means
            if self.weights_init is None:
                self.weights_ = weights

            if self.precisions_init is None:
                cov, p_c = _estimate_covariances(dataset, resp, nk,
                                                 self.means_, self.reg_covar,
                                                 self.covariance_type,
                                                 self._arity)
                self.covariances_ = cov
                self.precisions_cholesky_ = p_c

            for resp_subset in resp:
                compss_delete_object(resp_subset)


@task(returns=1)
def _resp_subset(labeled_subset, n_components):
    labels = labeled_subset.labels
    n_samples = len(labels)
    resp_chunk = np.zeros((n_samples, n_components))
    resp_chunk[np.arange(n_samples), labels.astype(int)] = 1
    return Subset(resp_chunk)


@task(returns=1)
def _random_resp_subset(subset, n_components, seed):
    n_samples = subset.samples.shape[0]
    resp_chunk = RandomState(seed).rand(n_samples, n_components)
    resp_chunk /= resp_chunk.sum(axis=1)[:, np.newaxis]
    return Subset(resp_chunk)
