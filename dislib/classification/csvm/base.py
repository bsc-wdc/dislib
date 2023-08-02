import json
import os
import pickle
from uuid import uuid4

import numpy as np
from pycompss.api.api import compss_delete_object
from pycompss.api.api import compss_wait_on
from pycompss.api.constraint import constraint
from pycompss.api.parameter import COLLECTION_IN, Depth, Type
from pycompss.api.task import task
from scipy.sparse import hstack as hstack_sp
from scipy.sparse import issparse
from sklearn.base import BaseEstimator
from sklearn.svm import SVC

from dislib.data.array import Array
from dislib.data.util import sync_obj, encoder_helper, decoder_helper
from dislib.utils.base import _paired_partition

import dislib.data.util.model as utilmodel


class CascadeSVM(BaseEstimator):
    """ Cascade Support Vector classification.

    Implements distributed support vector classification based on
    Graf et al. [1]_. The optimization process is carried out using
    scikit-learn's `SVC <http://scikit-learn.org/stable/modules/generated
    /sklearn.svm.SVC.html>`_. This method solves binary classification
    problems.

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
        Whether to test for convergence. If False, the algorithm will run for
        max_iter iterations. Checking for convergence adds a synchronization
        point after each iteration.

        If ``check_convergence=False'' synchronization does not happen until
        a call to ``predict'' or ``decision_function''. This can be useful to
        fit multiple models in parallel.
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
    >>> import dislib as ds
    >>> from dislib.classification import CascadeSVM
    >>> import numpy as np
    >>>
    >>>
    >>> if __name__ == '__main__':
    >>>     x = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    >>>     y = np.array([1, 1, 2, 2])
    >>>     train_data = ds.array(x, block_size=(4, 2))
    >>>     train_labels = ds.array(y, block_size=(4, 2))
    >>>     svm = CascadeSVM()
    >>>     svm.fit(train_data, train_labels)
    >>>     test_data = ds.array(np.array([[-0.8, -1]]), block_size=(1, 2))
    >>>     y_pred = svm.predict(test_data)
    >>>     print(y_pred)
    """
    _name_to_kernel = {"linear": "_linear_kernel", "rbf": "_rbf_kernel"}

    def __init__(self, cascade_arity=2, max_iter=5, tol=1e-3,
                 kernel="rbf", c=1, gamma='auto', check_convergence=True,
                 random_state=None, verbose=False):

        self.cascade_arity = cascade_arity
        self.max_iter = max_iter
        self.tol = tol
        self.kernel = kernel
        self.c = c
        self.gamma = gamma
        self.check_convergence = check_convergence
        self.random_state = random_state
        self.verbose = verbose

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
        self : CascadeSVM
        """
        self._check_initial_parameters()
        self._reset_model()
        self._set_gamma(x.shape[1])
        self._set_kernel()
        self._hstack_f = hstack_sp if x._sparse else np.hstack

        ids_list = [[_gen_ids(row.shape[0])] for row in x._iterator(axis=0)]

        while not self._check_finished():
            self._do_iteration(x, y, ids_list)

            if self.check_convergence:
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

        return Array(blocks=y_list, top_left_shape=(x._top_left_shape[0], 1),
                     reg_shape=(x._reg_shape[0], 1),
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

        return Array(blocks=df, top_left_shape=(x._top_left_shape[0], 1),
                     reg_shape=(x._reg_shape[0], 1),
                     shape=(x.shape[0], 1), sparse=False)

    def score(self, x, y, collect=False):
        """
        Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)
            Test samples.
        y : ds-array, shape=(n_samples, 1)
            True labels for x.
        collect : bool, optional (default=False)
            When True, a synchronized result is returned.

        Returns
        -------
        score : float (as future object)
            Mean accuracy of self.predict(x) wrt. y.
        """
        assert (self._clf is not None or self._svs is not None), \
            "Model has not been initialized. Call fit() first."

        partial_scores = []

        for x_row, y_row in _paired_partition(x, y):
            partial = _score(x_row._blocks, y_row._blocks, self._clf)
            partial_scores.append(partial)

        score = _merge_scores(*partial_scores)

        return compss_wait_on(score) if collect else score

    def _check_initial_parameters(self):
        gamma = self.gamma
        assert (gamma == "auto" or isinstance(gamma, float)
                or isinstance(float(gamma), float)), "Invalid gamma"
        kernel = self.kernel
        assert (kernel is None or kernel in self._name_to_kernel.keys()), \
            "Incorrect kernel value [%s], available kernels are %s" % (
                kernel, self._name_to_kernel.keys())
        c = self.c
        assert (c is None or isinstance(c, float) or
                isinstance(float(c), float)), \
            "Incorrect C type [%s], type : %s" % (c, type(c))
        tol = self.tol
        assert (isinstance(tol, float) or isinstance(float(tol), float)), \
            "Incorrect tol type [%s], type : %s" % (tol, type(tol))
        assert self.cascade_arity > 1, "Cascade arity must be greater than 1"
        assert self.max_iter > 0, "Max iterations must be greater than 0"
        assert isinstance(self.check_convergence, bool), "Invalid value in " \
                                                         "check_convergence"

    def _reset_model(self):
        self.iterations = 0
        self.converged = False
        self._last_w = None
        self._clf = None
        self._svs = None
        self._sv_labels = None

    def _set_gamma(self, n_features):
        if self.gamma == "auto":
            self._gamma = 1. / n_features
        else:
            self._gamma = self.gamma

    def _set_kernel(self):
        kernel = self.kernel
        c = self.c
        if kernel == "rbf":
            self._clf_params = {"kernel": kernel, "C": c, "gamma": self._gamma}
        else:
            self._clf_params = {"kernel": kernel, "C": c}

        try:
            self._kernel_f = getattr(self, CascadeSVM._name_to_kernel[kernel])
        except AttributeError:
            self._kernel_f = getattr(self, "_rbf_kernel")

    def _collect_clf(self):
        self._svs, self._sv_labels, self._clf = compss_wait_on(self._svs,
                                                               self._sv_labels,
                                                               self._clf)

    def _print_iteration(self):
        if self.verbose:
            print("Iteration %s of %s." % (self.iterations, self.max_iter))

    def _do_iteration(self, x, y, ids_list):
        q = []
        pars = self._clf_params
        arity = self.cascade_arity

        # first level
        for partition, id_bk in zip(_paired_partition(x, y), ids_list):
            x_data = partition[0]._blocks
            y_data = partition[1]._blocks
            ids = [id_bk]

            if self._svs is not None:
                x_data.append(self._svs)
                y_data.append([self._sv_labels])
                ids.append([self._sv_ids])

            _tmp = _train(x_data, y_data, ids, self.random_state, **pars)
            sv, sv_labels, sv_ids, self._clf = _tmp
            q.append((sv, sv_labels, sv_ids))

        # reduction
        while len(q) > arity:
            data = q[:arity]
            del q[:arity]

            x_data = [tup[0] for tup in data]
            y_data = [[tup[1]] for tup in data]
            ids = [[tup[2]] for tup in data]

            _tmp = _train(x_data, y_data, ids, self.random_state, **pars)
            sv, sv_labels, sv_ids, self._clf = _tmp
            q.append((sv, sv_labels, sv_ids))

            # delete partial results
            for partial in data:
                compss_delete_object(partial)

        # last layer
        x_data = [tup[0] for tup in q]
        y_data = [[tup[1]] for tup in q]
        ids = [[tup[2]] for tup in q]

        _tmp = _train(x_data, y_data, ids, self.random_state, **pars)
        self._svs, self._sv_labels, self._sv_ids, self._clf = _tmp

        self.iterations += 1

    def _check_finished(self):
        return self.iterations >= self.max_iter or self.converged

    def _lag_fast(self, vectors, labels, coef):
        set_sl = set(labels.ravel())
        if len(set_sl) > 2:
            new_sl = [labels.copy()]
            vectors_def = vectors
            for _ in range(len(set_sl) - 2):
                new_sl.append(labels.copy())
                vectors_def = np.concatenate((vectors_def, vectors))
        else:
            new_sl = labels.copy()
            new_sl[labels == 0] = -1
            vectors_def = vectors
        if issparse(coef):
            coef = coef.toarray()

        c1, c2 = np.meshgrid(coef, coef)
        l1, l2 = np.meshgrid(new_sl, new_sl)
        double_sum = c1 * c2 * l1 * l2 * self._kernel_f(vectors_def)
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

            if delta < self.tol:
                self.converged = True

        if self.verbose:
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

    def save_model(self, filepath, overwrite=True, save_format="json"):
        """Saves a model to a file.
        The model is synchronized before saving and can be reinstantiated in
        the exact same state, without any of the code used for model
        definition or fitting.
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
        >>> from dislib.classification import CascadeSVM
        >>> import numpy as np
        >>> import dislib as ds
        >>> x = ds.array(np.array([[1, 2], [2, 1], [-1, -2],
        >>> [-2, -1]]), (2, 2))
        >>> y = ds.array(np.array([0, 1, 1, 0]).reshape(-1, 1), (2, 1))
        >>> model = CascadeSVM(cascade_arity=3, max_iter=10,
        >>>              tol=1e-4, kernel='linear', c=2, gamma=0.1,
        >>>              check_convergence=False,
        >>>              random_state=seed, verbose=False)
        >>> model.fit(x, y)
        >>> model.save_model('/tmp/model')
        >>> loaded_model = CascadeSVM()
        >>> loaded_model.load_model('/tmp/model')
        >>> x_test = ds.array(np.array([[1, 2], [2, 1], [-1, -2], [-2, -1],
        >>> [1, 1], [-1, -1]]), (2, 2))
        >>> y_pred = model.predict(x_test)
        >>> y_loaded_pred = loaded_model.predict(x_test)
        >>> assert np.allclose(y_pred.collect(),
        >>> y_loaded_pred.collect())
        """

        # Check overwrite
        if not overwrite and os.path.isfile(filepath):
            return

        sync_obj(self.__dict__)
        model_metadata = self.__dict__
        model_metadata["model_name"] = "csvm"

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
        >>> from dislib.classification import CascadeSVM
        >>> import numpy as np
        >>> import dislib as ds
        >>> x = ds.array(np.array([[1, 2], [2, 1], [-1, -2],
        >>> [-2, -1]]), (2, 2))
        >>> y = ds.array(np.array([0, 1, 1, 0]).reshape(-1, 1), (2, 1))
        >>> model = CascadeSVM(cascade_arity=3, max_iter=10,
        >>>              tol=1e-4, kernel='linear', c=2, gamma=0.1,
        >>>              check_convergence=False,
        >>>              random_state=seed, verbose=False)
        >>> model.fit(x, y)
        >>> model.save_model('/tmp/model')
        >>> loaded_model = CascadeSVM()
        >>> loaded_model.load_model('/tmp/model')
        >>> x_test = ds.array(np.array([[1, 2], [2, 1], [-1, -2], [-2, -1],
        >>> [1, 1], [-1, -1]]), (2, 2))
        >>> y_pred = model.predict(x_test)
        >>> y_loaded_pred = loaded_model.predict(x_test)
        >>> assert np.allclose(y_pred.collect(), y_loaded_pred.collect())
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
    elif isinstance(obj, SVC):
        return {
            "class_name": obj.__class__.__name__,
            "module_name": obj.__module__,
            "items": obj.__dict__,
        }
    else:
        return {
            "class_name": "CascadeSVM",
            "module_name": "classification",
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
        elif (class_name == "SVC" and "sklearn" in obj["module_name"]):
            dict_ = _decode_helper(obj["items"])
            model = SVC()
            model.__dict__.update(dict_)
            return model
        else:
            return CascadeSVM().__dict__.update(_decode_helper(obj["items"]))
    return obj


@constraint(computing_units="${ComputingUnits}")
@task(returns=1)
def _gen_ids(n_samples):
    idx = [[uuid4().int] for _ in range(n_samples)]
    return np.array(idx)


@constraint(computing_units="${ComputingUnits}")
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


@constraint(computing_units="${ComputingUnits}")
@task(x_list={Type: COLLECTION_IN, Depth: 2}, returns=np.array)
def _predict(x_list, clf):
    x = Array._merge_blocks(x_list)
    return clf.predict(x).reshape(-1, 1)


@constraint(computing_units="${ComputingUnits}")
@task(x_list={Type: COLLECTION_IN, Depth: 2}, returns=np.array)
def _decision_function(x_list, clf):
    x = Array._merge_blocks(x_list)
    return clf.decision_function(x).reshape(-1, 1)


@constraint(computing_units="${ComputingUnits}")
@task(x_list={Type: COLLECTION_IN, Depth: 2},
      y_list={Type: COLLECTION_IN, Depth: 2}, returns=tuple)
def _score(x_list, y_list, clf):
    x = Array._merge_blocks(x_list)
    y = Array._merge_blocks(y_list)

    y_pred = clf.predict(x)
    equal = np.equal(y_pred, y.ravel())

    return np.sum(equal), x.shape[0]


@constraint(computing_units="${ComputingUnits}")
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
