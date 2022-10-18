import json
import os
import pickle
import warnings

import numpy as np
from numpy.random.mtrand import RandomState
from pycompss.api.api import compss_wait_on, compss_delete_object
from pycompss.api.constraint import constraint
from pycompss.api.parameter import Type, COLLECTION_IN, Depth
from pycompss.api.task import task
from scipy import linalg
from scipy.sparse import issparse
from scipy.special import logsumexp
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import BaseEstimator
from sklearn.utils import validation
from sklearn.utils.extmath import row_norms

from dislib.cluster import KMeans
from dislib.data.array import Array
from dislib.data.util import sync_obj, encoder_helper, decoder_helper

import dislib.data.util.model as utilmodel


class GaussianMixture(BaseEstimator):
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
    weights_init : array-like, shape=(n_components, ), optional
        The user-provided initial weights, defaults to None.
        If None, weights are initialized using the `init_params` method.
    means_init : array-like, shape=(n_components, n_features), optional
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
    weights_ : array-like, shape=(n_components,)
        The weight of each mixture component.
    means_ : array-like, shape=(n_components, n_features)
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
    >>> import dislib as ds
    >>> from dislib.cluster import GaussianMixture
    >>> from pycompss.api.api import compss_wait_on
    >>>
    >>>
    >>> if __name__ == '__main__':
    >>>     x = ds.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]],
    >>>                  (3, 2))
    >>>     gm = GaussianMixture(n_components=2, random_state=0)
    >>>     labels = gm.fit_predict(x).collect()
    >>>     print(labels)
    >>>     x_test = ds.array([[0, 0], [4, 4]], (2, 2))
    >>>     labels_test = gm.predict(x_test).collect()
    >>>     print(labels_test)
    >>>     print(compss_wait_on(gm.means_))
    """

    def __init__(self, n_components=1, covariance_type='full',
                 check_convergence=True, tol=1e-3, reg_covar=1e-6,
                 max_iter=100, init_params='kmeans', weights_init=None,
                 means_init=None, precisions_init=None, arity=50,
                 verbose=False, random_state=None):

        self.n_components = n_components
        self.check_convergence = check_convergence
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.init_params = init_params
        self.arity = arity
        self.verbose = verbose
        self.random_state = random_state
        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init

    def fit(self, x, y=None):
        """Estimate model parameters with the EM algorithm.

        Iterates between E-steps and M-steps until convergence or until
        `max_iter` iterations are reached. It estimates the model parameters
        `weights_`, `means_` and `covariances_`.

        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)
            Data points.
        y : ignored
            Not used, present here for API consistency by convention.

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

        self._initialize_parameters(x, random_state)
        self.lower_bound_ = -np.infty
        if self.verbose:
            print("GaussianMixture EM algorithm start")
        for self.n_iter in range(1, self.max_iter + 1):
            prev_lower_bound = self.lower_bound_

            self.lower_bound_, resp = self._e_step(x)
            self._m_step(x, resp)
            for resp_block in resp._blocks:
                compss_delete_object(resp_block)

            if self.check_convergence:
                self.lower_bound_ = compss_wait_on(self.lower_bound_)
                diff = abs(self.lower_bound_ - prev_lower_bound)

                if self.verbose:
                    iter_msg_template = "Iteration %s - Convergence crit. = %s"
                    print(iter_msg_template % (self.n_iter, diff))

                if diff < self.tol:
                    self.converged_ = True
                    break

        if self.check_convergence and not self.converged_:
            warnings.warn('The algorithm did not converge. '
                          'Try different init parameters, '
                          'or increase max_iter, tol '
                          'or check for degenerate data.',
                          ConvergenceWarning)

    def fit_predict(self, x):
        """Estimate model parameters and predict clusters for the same data.

        Fits the model and, after fitting, uses the model to predict cluster
        labels for the same training data.

        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)
            Data points.

        Returns
        -------
        y : ds-array, shape(n_samples, 1)
            Cluster labels for x.

        Warns
        -----
        ConvergenceWarning
            If `tol` is not None and `max_iter` iterations are reached without
            convergence.
        """
        self.fit(x)
        return self.predict(x)

    def predict(self, x):
        """Predict cluster labels for the given data using the trained model.

        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)
            Data points.

        Returns
        -------
        y : ds-array, shape(n_samples, 1)
            Cluster labels for x.

        """
        validation.check_is_fitted(self,
                                   ['weights_', 'means_',
                                    'precisions_cholesky_'])
        _, resp = self._e_step(x)
        return _resp_argmax(resp)

    def _e_step(self, x):
        """E step.

        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)
            Data points.

        Returns
        -------
        log_prob_norm : float
            Mean of the logarithms of the probabilities of each sample in the
            data.

        responsibility : ds-array, shape (n_samples, n_components)
            Posterior probabilities (or responsibilities) of each sample in the
            data.
        """
        log_prob_norm_partials = []
        resp_blocks = []
        for x_part in x._iterator(axis=0):
            log_prob_norm_part, resp_part = self._estimate_prob_resp(x_part)
            log_prob_norm_partials.append(log_prob_norm_part)
            resp_blocks.append([resp_part])
        log_prob_norm = self._reduce_log_prob_norm(log_prob_norm_partials)

        resp = Array(blocks=resp_blocks,
                     top_left_shape=(x._top_left_shape[0], self.n_components),
                     reg_shape=(x._reg_shape[0], self.n_components),
                     shape=(x.shape[0], self.n_components), sparse=False)
        return log_prob_norm, resp

    def _estimate_prob_resp(self, x_part):
        """Estimate log-likelihood and responsibilities for a subsample.

        Compute the sum of log-likelihoods, the count of samples, and the
        responsibilities for each sample in the data portion with respect to
        the current state of the model.

        Parameters
        ----------
        x_part : ds-array, shape=(x_part_size, n_features)
            Horizontal portion of the data.

        Returns
        -------
        log_prob_norm_subset : tuple
            tuple(sum, count) for log p(subset)

        responsibilities : ds-array, shape (x.shape[0], n_components)
            Responsibilities for each sample and component.
        """
        return _estimate_responsibilities(x_part._blocks, self.weights_,
                                          self.means_,
                                          self.precisions_cholesky_,
                                          self.covariance_type)

    def _reduce_log_prob_norm(self, partials):
        while len(partials) > self.arity:
            partials_subset = partials[:self.arity]
            partials = partials[self.arity:]
            partials.append(_sum_log_prob_norm(*partials_subset))
        return _finalize_sum_log_prob_norm(*partials)

    def _m_step(self, x, resp):
        """M step.

        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)
            Data points.

        resp : ds-array, shape (n_samples, n_components)
            Posterior probabilities (or responsibilities) of the point of each
            sample in the data.
        """
        weights, nk, means = self._estimate_parameters(x, resp)
        self.weights_ = weights
        self.means_ = means

        cov, p_c = _estimate_covariances(x, resp, nk, means,
                                         self.reg_covar, self.covariance_type,
                                         self.arity)

        self.covariances_ = cov
        self.precisions_cholesky_ = p_c

    def _estimate_parameters(self, x, resp):
        """Estimate the Gaussian distribution weights and means.

        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)
            Data points.
        resp : ds-array, shape (n_samples, n_components)
            The responsibilities for each data sample in x.

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
        all_partial_params = []
        for x_part, resp_part in zip(x._iterator(axis=0),
                                     resp._iterator(axis=0)):
            partial_params = _partial_estimate_parameters(x_part._blocks,
                                                          resp_part._blocks)
            all_partial_params.append(partial_params)
        return _reduce_estimate_parameters(all_partial_params, self.arity)

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

    def _initialize_parameters(self, x, random_state):
        """Initialization of the Gaussian mixture parameters.

        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)
            Data points.

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
            resp_blocks = []
            if self.init_params == 'kmeans':
                if self.verbose:
                    print("KMeans initialization start")
                seed = random_state.randint(0, int(1e8))
                kmeans = KMeans(n_clusters=n_components, random_state=seed,
                                verbose=self.verbose)
                y = kmeans.fit_predict(x)
                self.kmeans = kmeans
                for y_part in y._iterator(axis=0):
                    resp_blocks.append([_resp_subset(y_part._blocks,
                                                     n_components)])

            elif self.init_params == 'random':
                chunks = x._n_blocks[0]
                seeds = random_state.randint(np.iinfo(np.int32).max,
                                             size=chunks)
                for i, x_row in enumerate(x._iterator(axis=0)):
                    resp_blocks.append([_random_resp_subset(x_row.shape[0],
                                                            n_components,
                                                            seeds[i])])
            else:
                raise ValueError("Unimplemented initialization method '%s'"
                                 % self.init_params)
            resp = Array(blocks=resp_blocks,
                         top_left_shape=(x._top_left_shape[0], n_components),
                         reg_shape=(x._reg_shape[0], n_components),
                         shape=(x.shape[0], n_components), sparse=False)
            weights, nk, means = self._estimate_parameters(x, resp)
            if self.means_init is None:
                self.means_ = means
            if self.weights_init is None:
                self.weights_ = weights

            if self.precisions_init is None:
                cov, p_c = _estimate_covariances(x, resp, nk,
                                                 self.means_, self.reg_covar,
                                                 self.covariance_type,
                                                 self.arity)
                self.covariances_ = cov
                self.precisions_cholesky_ = p_c

            for resp_block in resp._blocks:
                compss_delete_object(resp_block)

    def save_model(self, filepath, overwrite=True, save_format="json"):
        """Saves a model to a file.
                The model is synchronized before saving and can be
                reinstantiated in the exact same state, without any of
                the code used for model definition or fitting.
                Parameters
                ----------
                filepath : str
                    Path where to save the model
                overwrite : bool, optional (default=True)
                    Whether any existing model at the target
                    location should be overwritten.
                save_format : str, optional (default='json)
                    Format used to save the models.
                Examples
                --------
                >>> from dislib.cluster import GaussianMixture
                >>> import numpy as np
                >>> import dislib as ds
                >>> x = np.array([[1, 2], [1, 4], [1, 0], [4, 2],
                >>> [4, 4], [4, 0]])
                >>> x_train = ds.array(x, (2, 2))
                >>> model = gm = GaussianMixture(n_components=2,
                >>> random_state=0)
                >>> model.fit(x_train)
                >>> model.save_model('/tmp/model')
                >>> loaded_model = gm = GaussianMixture()
                >>> loaded_model.load_model('/tmp/model')
                >>> x_test = ds.array(np.array([[0, 0], [4, 4]]), (2, 2))
                >>> loaded_model_pred = loaded_model.predict(x_test)
                """

        # Check overwrite
        if not overwrite and os.path.isfile(filepath):
            return

        sync_obj(self.__dict__)
        model_metadata = self.__dict__
        model_metadata["model_name"] = "gm"

        # Save model
        if save_format == "json":
            with open(filepath, "w") as f:
                json.dump(model_metadata, f, default=_encode_helper)
        elif save_format == "cbor":
            if utilmodel.cbor2 is None:
                raise ModuleNotFoundError("No module named 'cbor2'")
            with open(filepath, "wb") as f:
                utilmodel.cbor2.dump(model_metadata, f,
                                     default=_encode_helper_cbor)
        elif save_format == "pickle":
            with open(filepath, "wb") as f:
                pickle.dump(model_metadata, f)
        else:
            raise ValueError("Wrong save format.")

    def load_model(self, filepath, load_format="json"):
        """Loads a model from a file.
        The model is reinstantiated in the exact same state in which it was
        saved, without any of the code used for model definition or fitting.
        Parameters
        ----------
        filepath : str
            Path of the saved the model
        load_format : str, optional (default='json')
            Format used to load the model.
        Examples
        --------
        >>> from dislib.cluster import GaussianMixture
        >>> import numpy as np
        >>> import dislib as ds
        >>> x = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
        >>> x_train = ds.array(x, (2, 2))
        >>> model = GaussianMixture(n_components=2, random_state=0)
        >>> model.fit(x_train)
        >>> model.save_model('/tmp/model')
        >>> gm = GaussianMixture()
        >>> gm.load_model('/tmp/model')
        >>> x_test = ds.array(np.array([[0, 0], [4, 4]]), (2, 2))
        >>> loaded_model_pred = gm.predict(x_test)
        """
        # Load model
        if load_format == "json":
            with open(filepath, "r") as f:
                model_metadata = json.load(f, object_hook=_decode_helper)
        elif load_format == "cbor":
            if utilmodel.cbor2 is None:
                raise ModuleNotFoundError("No module named 'cbor2'")
            with open(filepath, "rb") as f:
                model_metadata = utilmodel.cbor2.\
                    load(f, object_hook=_decode_helper_cbor)
        elif load_format == "pickle":
            with open(filepath, "rb") as f:
                model_metadata = pickle.load(f)
        else:
            raise ValueError("Wrong load format.")

        for key, val in model_metadata.items():
            setattr(self, key, val)


def _encode_helper_cbor(encoder, obj):
    encoder.encode(_encode_helper(obj))


def _encode_helper(obj):
    encoded = encoder_helper(obj)
    if encoded is not None:
        return encoded
    else:
        return {
            "class_name": "GaussianMixture",
            "module_name": "cluster",
            "items": obj.__dict__,
        }


def _decode_helper_cbor(decoder, obj):
    """Special decoder wrapper for dislib using cbor2."""
    return _decode_helper(obj)


def _decode_helper(obj):
    if isinstance(obj, dict) and "class_name" in obj:
        class_name = obj["class_name"]
        decoded = decoder_helper(class_name, obj)
        if decoded is not None:
            return decoded
        elif class_name == "RandomState":
            random_state = np.random.RandomState()
            random_state.set_state(_decode_helper(obj["items"]))
            return random_state
        else:
            return GaussianMixture().__dict__.update(
                _decode_helper(obj["items"]))
    return obj


@constraint(computing_units="${ComputingUnits}")
@task(x={Type: COLLECTION_IN, Depth: 2},
      resp={Type: COLLECTION_IN, Depth: 2},
      returns=1)
def _partial_estimate_parameters(x, resp):
    x = Array._merge_blocks(x)
    resp = Array._merge_blocks(resp)
    partial_nk = resp.sum(axis=0)
    if issparse(x):
        partial_means = x.T.dot(resp).T
    else:
        partial_means = np.matmul(resp.T, x)

    return x.shape[0], partial_nk, partial_means


def _reduce_estimate_parameters(partials, arity):
    while len(partials) > 1:
        partials_chunk = partials[:arity]
        partials = partials[arity:]
        partials.append(_merge_estimate_parameters(*partials_chunk))
    return _finalize_parameters(partials[0])


@constraint(computing_units="${ComputingUnits}")
@task(returns=1)
def _merge_estimate_parameters(*partials_params):
    n_samples = sum(params[0] for params in partials_params)
    nk = sum(params[1] for params in partials_params)
    means = sum(params[2] for params in partials_params)
    return n_samples, nk, means


@constraint(computing_units="${ComputingUnits}")
@task(returns=3)
def _finalize_parameters(params):
    n_samples = params[0]
    nk = params[1]
    nk += 10 * np.finfo(nk.dtype).eps
    means = params[2] / nk[:, np.newaxis]
    weights = nk / n_samples
    return weights, nk, means


def _estimate_covariances(x, resp, nk, means, reg_covar, covar_type, arity):
    """Estimate the covariances and compute the cholesky precisions.

    Parameters
    ----------
    x : ds-array, shape (n_samples, n_features)
        The input data.
    resp : ds-array, shape (n_samples, n_components)
        The responsibilities for each data sample in x.
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
        "tied": lambda r, x, m: _partial_covar_tied(x),
        "diag": _partial_covar_diag,
        "spherical": _partial_covar_diag  # uses same partial_covar as diag
    }[covar_type]
    for x_part, resp_part in zip(x._iterator(axis=0), resp._iterator(axis=0)):
        partials.append(partial_covar(resp_part._blocks, x_part._blocks,
                                      means))
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


@constraint(computing_units="${ComputingUnits}")
@task(x={Type: COLLECTION_IN, Depth: 2},
      resp={Type: COLLECTION_IN, Depth: 2},
      returns=1)
def _partial_covar_full(resp, x, means):
    x = Array._merge_blocks(x)
    resp = Array._merge_blocks(resp)
    n_components, n_features = means.shape
    covariances = np.empty((n_components, n_features, n_features))
    for k in range(n_components):
        if issparse(x):
            diff = (x - means[k] for x in x)
            partial_covs = (np.dot(r * d.T, d) for d, r in
                            zip(diff, resp[:, k]))
            covariances[k] = sum(partial_covs)
        else:
            diff = x - means[k]
            covariances[k] = np.dot(resp[:, k] * diff.T, diff)
    return covariances


@constraint(computing_units="${ComputingUnits}")
@task(x={Type: COLLECTION_IN, Depth: 2},
      returns=1)
def _partial_covar_tied(x):
    x = Array._merge_blocks(x)
    if issparse(x):
        avg_sample_2 = x.T.dot(x)
    else:
        avg_sample_2 = np.dot(x.T, x)
    return avg_sample_2


@constraint(computing_units="${ComputingUnits}")
@task(x={Type: COLLECTION_IN, Depth: 2},
      resp={Type: COLLECTION_IN, Depth: 2},
      returns=1)
def _partial_covar_diag(resp, x, means):
    x = Array._merge_blocks(x)
    resp = Array._merge_blocks(resp)
    if issparse(x):
        avg_resp_sample_2 = x.multiply(x).T.dot(resp).T
        avg_sample_means = means * x.T.dot(resp).T
    else:
        avg_resp_sample_2 = np.dot(resp.T, x * x)
        avg_sample_means = means * np.dot(resp.T, x)
    return avg_resp_sample_2 - 2 * avg_sample_means


@constraint(computing_units="${ComputingUnits}")
@task(returns=1)
def _sum_covar_partials(*covar_partials):
    return sum(covar_partials)


@constraint(computing_units="${ComputingUnits}")
@task(returns=2)
def _finalize_covar_full(covar_type, reg_covar, nk, covariances):
    n_components, n_features, _ = covariances.shape
    for k in range(n_components):
        covariances[k] /= nk[k]
        covariances[k].flat[::n_features + 1] += reg_covar
    precisions_chol = _compute_precision_cholesky(covariances, covar_type)
    return covariances, precisions_chol


@constraint(computing_units="${ComputingUnits}")
@task(returns=2)
def _finalize_covar_tied(covar_type, reg_covar, nk, means, covariances):
    avg_means2 = np.dot(nk * means.T, means)
    covariances -= avg_means2
    covariances /= nk.sum()
    covariances.flat[::len(covariances) + 1] += reg_covar
    precisions_chol = _compute_precision_cholesky(covariances, covar_type)
    return covariances, precisions_chol


@constraint(computing_units="${ComputingUnits}")
@task(returns=2)
def _finalize_covar_diag(covar_type, reg_covar, nk, means, covariances):
    covariances /= nk[:, np.newaxis]
    covariances += means ** 2
    covariances += reg_covar
    precisions_chol = _compute_precision_cholesky(covariances, covar_type)
    return covariances, precisions_chol


@constraint(computing_units="${ComputingUnits}")
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


@constraint(computing_units="${ComputingUnits}")
@task(returns=1)
def _sum_log_prob_norm(*partials):
    total, count = map(sum, zip(*partials))
    return total, count


@constraint(computing_units="${ComputingUnits}")
@task(returns=1)
def _finalize_sum_log_prob_norm(*partials):
    total, count = map(sum, zip(*partials))
    return total / count


@constraint(computing_units="${ComputingUnits}")
@task(x={Type: COLLECTION_IN, Depth: 2}, returns=2)
def _estimate_responsibilities(x, weights, means, precisions_cholesky,
                               covariance_type):
    """Estimate log-likelihood and responsibilities for the given data portion.

    Compute the sum of log-likelihoods, the count of samples, and the
    responsibilities for each sample in the data portion with respect to the
    current state of the model.

    Parameters
    ----------
    x : collection of depth 2
        Blocks of a horizontal portion of the data.
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
    log_prob_norm_x : tuple
        tuple(sum, count) for log p(x)

    responsibilities : array-like, shape (x.shape[0], n_features)
    """
    x = Array._merge_blocks(x)
    weighted_log_prob = _estimate_weighted_log_prob(x, weights, means,
                                                    precisions_cholesky,
                                                    covariance_type)
    log_prob_norm = logsumexp(weighted_log_prob, axis=1)
    log_prob_norm_sum = np.sum(log_prob_norm)
    count = len(log_prob_norm)
    with np.errstate(under='ignore'):
        # ignore underflow
        resp = np.exp(weighted_log_prob - log_prob_norm[:, np.newaxis])
    return (log_prob_norm_sum, count), resp


def _estimate_weighted_log_prob(x_part, weights, means, precisions_cholesky,
                                covariance_type):
    return _estimate_log_gaussian_prob(x_part, means,
                                       precisions_cholesky, covariance_type) \
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
    else:  # pragma: no cover
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


def _resp_argmax(resp):
    pred_blocks = []
    for resp_row in resp._iterator(axis=0):
        pred_blocks.append([_partial_resp_argmax(resp_row._blocks)])
    pred = Array(blocks=pred_blocks,
                 top_left_shape=(resp._top_left_shape[0], 1),
                 reg_shape=(resp._reg_shape[0], 1),
                 shape=(resp.shape[0], 1), sparse=False)
    return pred


@constraint(computing_units="${ComputingUnits}")
@task(resp={Type: COLLECTION_IN, Depth: 2}, returns=1)
def _partial_resp_argmax(resp):
    resp = Array._merge_blocks(resp)
    return resp.argmax(axis=1)[:, np.newaxis]


@constraint(computing_units="${ComputingUnits}")
@task(labels={Type: COLLECTION_IN, Depth: 2}, returns=1)
def _resp_subset(labels, n_components):
    labels = Array._merge_blocks(labels).flatten()
    n_samples = len(labels)
    resp_chunk = np.zeros((n_samples, n_components))
    resp_chunk[np.arange(n_samples), labels.astype(int, copy=False)] = 1
    return resp_chunk


@constraint(computing_units="${ComputingUnits}")
@task(returns=1)
def _random_resp_subset(n_samples, n_components, seed):
    resp_chunk = RandomState(seed).rand(n_samples, n_components)
    resp_chunk /= resp_chunk.sum(axis=1)[:, np.newaxis]
    return resp_chunk
