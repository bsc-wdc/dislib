import numpy as np
from pycompss.api.api import compss_wait_on
from pycompss.api.parameter import INOUT
from pycompss.api.task import task
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import paired_distances


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
    >>> x = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    >>> from dislib.data import load_data
    >>> train_data = load_data(x=x, subset_size=2)
    >>> kmeans = KMeans(n_clusters=2, random_state=0)
    >>> kmeans.fit_predict(train_data)
    >>> print(train_data.labels)
    >>> test_data = load_data(x=np.array([[0, 0], [4, 4]]), subset_size=2)
    >>> kmeans.predict(test_data)
    >>> print(test_data.labels)
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

    def fit(self, dataset):
        """ Compute K-means clustering.

        Parameters
        ----------
        dataset : Dataset
            Samples to cluster.
        """
        self._do_fit(dataset, False)

    def fit_predict(self, dataset):
        """ Performs clustering on data, and sets the cluster labels of the
        input Dataset.

        Parameters
        ----------
        dataset : Dataset
            Samples to cluster.
        """

        self._do_fit(dataset, True)

    def predict(self, dataset):
        """ Predict the closest cluster each sample in dataset belongs to.
        Cluster labels are stored in dataset.

        Parameters
        ----------
        dataset : Dataset
            New data to predict.
        """
        for subset in dataset:
            _predict(subset, self.centers)

    def _do_fit(self, dataset, set_labels):
        n_features = dataset.n_features
        sparse = dataset.sparse

        centers = _init_centers(n_features, sparse, self._n_clusters,
                                self._random_state)
        self.centers = compss_wait_on(centers)

        old_centers = None
        iteration = 0

        while not self._converged(old_centers, iteration):
            old_centers = self.centers.copy()
            partials = []

            for subset in dataset:
                partial = _partial_sum(subset, old_centers, set_labels)
                partials.append(partial)

            self._recompute_centers(partials)
            iteration += 1

        self.n_iter = iteration

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
def _get_label(subset):
    return subset.labels


@task(returns=np.array)
def _init_centers(n_features, sparse, n_clusters, random_state):
    np.random.seed(random_state)
    centers = np.random.random((n_clusters, n_features))

    if sparse:
        centers = csr_matrix(centers)

    return centers


@task(subset=INOUT, returns=np.array)
def _partial_sum(subset, centers, set_labels):
    partials = np.zeros((centers.shape[0], 2), dtype=object)
    close_centers = pairwise_distances(subset.samples, centers).argmin(axis=1)

    if set_labels:
        subset.labels = close_centers

    for center_idx, _ in enumerate(centers):
        indices = np.argwhere(close_centers == center_idx).flatten()
        partials[center_idx][0] = np.sum(subset.samples[indices], axis=0)
        partials[center_idx][1] = indices.shape[0]

    return partials


@task(returns=dict)
def _merge(*data):
    accum = data[0].copy()

    for d in data[1:]:
        accum += d

    return accum


@task(subset=INOUT)
def _predict(subset, centers):
    subset.labels = pairwise_distances(subset.samples, centers).argmin(axis=1)


def _vec_euclid(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)
