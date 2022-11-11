import json
import os
import pickle

import numpy as np

import dislib
from pycompss.api.api import compss_wait_on
from pycompss.api.constraint import constraint
from pycompss.api.parameter import COLLECTION_IN, Depth, Type
from pycompss.api.task import task
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import paired_distances
from sklearn.utils import check_random_state, validation

from dislib.data.array import Array
from dislib.data.util import encoder_helper, decoder_helper, sync_obj
import dislib.data.util.model as utilmodel


class KMeans(BaseEstimator):
    """ Perform K-means clustering.

    Parameters
    ----------
    n_clusters : int, optional (default=8)
        The number of clusters to form as well as the number of centroids to
        generate.
    init : {'random', nd-array or sparse matrix}, optional (default='random')
        Method of initialization, defaults to 'random', which generates
        random centers at the beginning.

        If an nd-array or sparse matrix is passed, it should be of shape
        (n_clusters, n_features) and gives the initial centers.
    max_iter : int, optional (default=10)
        Maximum number of iterations of the k-means algorithm for a single run.
    tol : float, optional (default=1e-4)
        Tolerance for accepting convergence.
    arity : int, optional (default=50)
        Arity of the reduction carried out during the computation of the new
        centroids.
    random_state : int or RandomState, optional (default=None)
        Seed or numpy.random.RandomState instance to generate random numbers
        for centroid initialization.
    verbose: boolean, optional (default=False)
        Whether to print progress information.

    Attributes
    ----------
    centers : ndarray
        Computed centroids.
    n_iter : int
        Number of iterations performed.

    Examples
    --------
    >>> import dislib as ds
    >>> from dislib.cluster import KMeans
    >>> import numpy as np
    >>>
    >>>
    >>> if __name__ == '__main__':
    >>>     x = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    >>>     x_train = ds.array(x, (2, 2))
    >>>     kmeans = KMeans(n_clusters=2, random_state=0)
    >>>     labels = kmeans.fit_predict(x_train)
    >>>     print(labels)
    >>>     x_test = ds.array(np.array([[0, 0], [4, 4]]), (2, 2))
    >>>     labels = kmeans.predict(x_test)
    >>>     print(labels)
    >>>     print(kmeans.centers)
    """

    def __init__(self, n_clusters=8, init='random', max_iter=10, tol=1e-4,
                 arity=50, random_state=None, verbose=False):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.arity = arity
        self.verbose = verbose
        self.init = init

    def fit(self, x, y=None):
        """ Compute K-means clustering.

        Parameters
        ----------
        x : ds-array
            Samples to cluster.
        y : ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : KMeans
        """
        self.random_state = check_random_state(self.random_state)
        self._init_centers(x.shape[1], x._sparse)

        old_centers = None
        iteration = 0

        if dislib.__gpu_available__:
            partial_sum_func = _partial_sum_gpu
        else:
            partial_sum_func = _partial_sum

        while not self._converged(old_centers, iteration):
            old_centers = self.centers.copy()
            partials = []

            for row in x._iterator(axis=0):
                partial = partial_sum_func(row._blocks, old_centers)
                partials.append(partial)

            self._recompute_centers(partials)
            iteration += 1

        self.n_iter = iteration

        return self

    def fit_predict(self, x, y=None):
        """ Compute cluster centers and predict cluster index for each sample.

        Parameters
        ----------
        x : ds-array
            Samples to cluster.
        y : ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : ds-array, shape=(n_samples, 1)
            Index of the cluster each sample belongs to.
        """

        self.fit(x)
        return self.predict(x)

    def predict(self, x):
        """ Predict the closest cluster each sample in the data belongs to.

        Parameters
        ----------
        x : ds-array
            New data to predict.

        Returns
        -------
        labels : ds-array, shape=(n_samples, 1)
            Index of the cluster each sample belongs to.
        """
        validation.check_is_fitted(self, 'centers')
        blocks = []

        for row in x._iterator(axis=0):
            blocks.append([_predict(row._blocks, self.centers)])

        return Array(blocks=blocks, top_left_shape=(x._top_left_shape[0], 1),
                     reg_shape=(x._reg_shape[0], 1), shape=(x.shape[0], 1),
                     sparse=False)

    def _converged(self, old_centers, iteration):
        if old_centers is None:
            return False

        diff = np.sum(paired_distances(self.centers, old_centers))

        if self.verbose:
            print("Iteration %s - Convergence crit. = %s" % (iteration, diff))

        return diff < self.tol ** 2 or iteration >= self.max_iter

    def _recompute_centers(self, partials):
        while len(partials) > 1:
            partials_subset = partials[:self.arity]
            partials = partials[self.arity:]
            partials.append(_merge(*partials_subset))

        partials = compss_wait_on(partials)

        for idx, sum_ in enumerate(partials[0]):
            if sum_[1] != 0:
                self.centers[idx] = sum_[0] / sum_[1]

    def _init_centers(self, n_features, sparse):
        if isinstance(self.init, np.ndarray) \
                or isinstance(self.init, csr_matrix):
            if self.init.shape != (self.n_clusters, n_features):
                raise ValueError("Init array must be of shape (n_clusters, "
                                 "n_features)")
            self.centers = self.init.copy()
        elif self.init == "random":
            shape = (self.n_clusters, n_features)
            self.centers = self.random_state.random_sample(shape)

            if sparse:
                self.centers = csr_matrix(self.centers)
        else:
            raise ValueError("Init must be random, an nd-array, "
                             "or an sp.matrix")

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
        >>> from dislib.cluster import KMeans
        >>> import numpy as np
        >>> import dislib as ds
        >>> x = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
        >>> x_train = ds.array(x, (2, 2))
        >>> model = KMeans(n_clusters=2, random_state=0)
        >>> model.fit(x_train)
        >>> model.save_model('/tmp/model')
        >>> loaded_model = KMeans()
        >>> loaded_model.load_model('/tmp/model')
        >>> x_test = ds.array(np.array([[0, 0], [4, 4]]), (2, 2))
        >>> model_pred = model.predict(x_test)
        >>> loaded_model_pred = loaded_model.predict(x_test)
        >>> assert np.allclose(model_pred.collect(),
        >>> loaded_model_pred.collect())
        """

        # Check overwrite
        if not overwrite and os.path.isfile(filepath):
            return

        sync_obj(self.__dict__)
        model_metadata = self.__dict__
        model_metadata["model_name"] = "kmeans"

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
        >>> from dislib.cluster import KMeans
        >>> import numpy as np
        >>> import dislib as ds
        >>> x = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
        >>> x_train = ds.array(x, (2, 2))
        >>> model = KMeans(n_clusters=2, random_state=0)
        >>> model.fit(x_train)
        >>> model.save_model('/tmp/model')
        >>> loaded_model = KMeans()
        >>> loaded_model.load_model('/tmp/model')
        >>> x_test = ds.array(np.array([[0, 0], [4, 4]]), (2, 2))
        >>> model_pred = model.predict(x_test)
        >>> loaded_model_pred = loaded_model.predict(x_test)
        >>> assert np.allclose(model_pred.collect(),
        >>> loaded_model_pred.collect())
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
    return obj


@constraint(processors=[
                {"processorType": "CPU", "computingUnits": "1"},
                {"processorType": "GPU", "computingUnits": "1"},
            ])
@task(blocks={Type: COLLECTION_IN, Depth: 2}, returns=np.array)
def _partial_sum_gpu(blocks, centers):
    import cupy as cp

    partials = np.zeros((centers.shape[0], 2), dtype=object)
    arr = Array._merge_blocks(blocks).astype(np.float32)
    arr_gpu = cp.asarray(arr)
    centers_gpu = cp.asarray(centers).astype(cp.float32)

    close_centers_gpu = cp.argmin(distance_gpu(arr_gpu, centers_gpu), axis=1)
    arr_gpu, centers_gpu = None, None

    close_centers = cp.asnumpy(close_centers_gpu)

    for center_idx, _ in enumerate(centers):
        indices = np.argwhere(close_centers == center_idx).flatten()
        partials[center_idx][0] = np.sum(arr[indices], axis=0)
        partials[center_idx][1] = indices.shape[0]

    return partials


@constraint(computing_units="${ComputingUnits}")
@task(blocks={Type: COLLECTION_IN, Depth: 2}, returns=np.array)
def _partial_sum(blocks, centers):
    partials = np.zeros((centers.shape[0], 2), dtype=object)
    arr = Array._merge_blocks(blocks)

    close_centers = pairwise_distances(arr, centers).argmin(axis=1)

    for center_idx, _ in enumerate(centers):
        indices = np.argwhere(close_centers == center_idx).flatten()
        partials[center_idx][0] = np.sum(arr[indices], axis=0)
        partials[center_idx][1] = indices.shape[0]

    return partials


@constraint(computing_units="${ComputingUnits}")
@task(returns=dict)
def _merge(*data):
    accum = data[0].copy()

    for d in data[1:]:
        accum += d

    return accum


@constraint(computing_units="${ComputingUnits}")
@task(blocks={Type: COLLECTION_IN, Depth: 2}, returns=np.array)
def _predict(blocks, centers):
    arr = Array._merge_blocks(blocks)
    return pairwise_distances(arr, centers).argmin(axis=1).reshape(-1, 1)


def distance_gpu(a_gpu, b_gpu):
    import cupy as cp

    sq_sum_ker = get_sq_sum_kernel()

    aa_gpu = cp.empty(a_gpu.shape[0], dtype=cp.float32)
    bb_gpu = cp.empty(b_gpu.shape[0], dtype=cp.float32)

    sq_sum_ker(a_gpu, aa_gpu, axis=1)
    sq_sum_ker(b_gpu, bb_gpu, axis=1)

    size = len(aa_gpu) * len(bb_gpu)
    dist_gpu = cp.empty((len(aa_gpu), len(bb_gpu)), dtype=cp.float32)
    add_mix_kernel(len(b_gpu))(aa_gpu, bb_gpu, dist_gpu, size=size)
    aa_gpu, bb_gpu = None, None

    dist_gpu += -2.0 * cp.dot(a_gpu, b_gpu.T)

    return dist_gpu


def get_sq_sum_kernel():
    import cupy as cp

    return cp.ReductionKernel(
        'T x',  # input params
        'T y',  # output params
        'x * x',  # map
        'a + b',  # reduce
        'y = a',  # post-reduction map
        '0',  # identity value
        'sqsum'  # kernel name
    )


def add_mix_kernel(y_len):
    import cupy as cp

    return cp.ElementwiseKernel(
        'raw T x, raw T y', 'raw T z',
        f'z[i] = x[i / {y_len}] + y[i % {y_len}]',
        'add_mix')
