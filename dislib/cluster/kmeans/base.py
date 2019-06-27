import numbers

import numpy as np
from pycompss.api.api import compss_wait_on
from pycompss.api.parameter import COLLECTION_IN, Depth, Type
from pycompss.api.task import task
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import paired_distances

from dislib.data.array import Array


class KMeans:
    """ Perform K-means clustering.

    Parameters
    ----------
    n_clusters : int, optional (default=8)
        The number of clusters to form as well as the number of centroids to
        generate.
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
    >>> from dislib.cluster import KMeans
    >>> import numpy as np
    >>> import dislib as ds
    >>> x = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    >>> x_train = ds.array(x, (2, 2))
    >>> kmeans = KMeans(n_clusters=2, random_state=0)
    >>> labels = kmeans.fit_predict(x_train)
    >>> print(labels)
    >>> x_test = ds.array(np.array([[0, 0], [4, 4]]), (2, 2))
    >>> labels = kmeans.predict(x_test)
    >>> print(labels)
    >>> print(kmeans.centers)
    """

    def __init__(self, n_clusters=8, max_iter=10, tol=1e-4, arity=50,
                 random_state=None, verbose=False):
        self._n_clusters = n_clusters
        self._max_iter = max_iter
        self._tol = tol
        self._random_state = random_state
        self._arity = arity
        self.centers = None
        self.n_iter = 0
        self._verbose = verbose

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
        self
        """
        n_features = x.shape[1]
        sparse = x._sparse

        centers = _init_centers(n_features, sparse, self._n_clusters,
                                self._random_state)
        self.centers = compss_wait_on(centers)

        old_centers = None
        iteration = 0

        while not self._converged(old_centers, iteration):
            old_centers = self.centers.copy()
            partials = []

            for r_block in x.iterator(axis=0):
                partial = _partial_sum(r_block._blocks, old_centers)
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
        """ Predict the closest cluster each sample in dataset belongs to.

        Parameters
        ----------
        x : ds-array
            New data to predict.

        Returns
        -------
        labels : ds-array, shape=(n_samples, 1)
            Index of the cluster each sample belongs to.
        """
        blocks = [list()]

        for r_block in x.iterator(axis=0):
            blocks[0].append(_predict(r_block._blocks, self.centers))

        return Array(blocks=blocks, blocks_shape=(x._blocks_shape[0], 1),
                     shape=(x.shape[0], 1),
                     sparse=x._sparse)

    def _converged(self, old_centers, iteration):
        if old_centers is None:
            return False

        diff = np.sum(paired_distances(self.centers, old_centers))

        if self._verbose:
            print("Iteration %s - Convergence crit. = %s" % (iteration, diff))

        return diff < self._tol ** 2 or iteration >= self._max_iter

    def _recompute_centers(self, partials):
        while len(partials) > 1:
            partials_subset = partials[:self._arity]
            partials = partials[self._arity:]
            partials.append(_merge(*partials_subset))

        partials = compss_wait_on(partials)

        for idx, sum_ in enumerate(partials[0]):
            if sum_[1] != 0:
                self.centers[idx] = sum_[0] / sum_[1]


@task(returns=np.array)
def _init_centers(n_features, sparse, n_clusters, random_state):
    r_state = random_state

    if isinstance(r_state, (numbers.Integral, np.integer)):
        r_state = np.random.RandomState(r_state)

    centers = r_state.random_sample((n_clusters, n_features))

    if sparse:
        centers = csr_matrix(centers)

    return centers


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


@task(returns=dict)
def _merge(*data):
    accum = data[0].copy()

    for d in data[1:]:
        accum += d

    return accum


@task(blocks={Type: COLLECTION_IN, Depth: 2}, returns=np.array)
def _predict(blocks, centers):
    arr = Array._merge_blocks(blocks)
    return pairwise_distances(arr, centers).argmin(axis=1)
