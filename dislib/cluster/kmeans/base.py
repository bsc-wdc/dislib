import numpy as np
from pycompss.api.api import compss_wait_on
from pycompss.api.parameter import INOUT
from pycompss.api.task import task


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
    random_state : int, optional (default=None)
        Determines random number generation for centroid initialization. Use an
        int to make the randomness deterministic.

    Attributes
    ----------
    centers : ndarray
        Computed centroids.
    n_iter : int
        Number of iterations performed.

    Methods
    -------
    fit(dataset)
        Compute K-means clustering.
    fit_predict(dataset)
        Compute K-means clustering, and set and return cluster labels.
    predict(x)
        Predict the closest cluster each sample in x belongs to.
    """

    def __init__(self, n_clusters=8, max_iter=10, tol=1 ** -4, arity=50,
                 random_state=None):
        self._n_clusters = n_clusters
        self._max_iter = max_iter
        self._tol = tol
        self._random_state = random_state
        self._arity = arity
        self.centers = None
        self.n_iter = 0

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
        centers = _init_centers(dataset.n_features, self._n_clusters,
                                self._random_state)
        self.centers = compss_wait_on(centers)

        old_centers = None
        iteration = 0

        while not self._converged(old_centers, iteration):
            old_centers = np.array(self.centers)
            partials = []

            for subset in dataset:
                partials.append(_partial_sum(subset, old_centers, set_labels))

            self._recompute_centers(partials)
            iteration += 1

        self.n_iter = iteration

    def _converged(self, old_centers, iter):
        if old_centers is not None:
            diff = 0
            for i, center in enumerate(self.centers):
                diff += np.linalg.norm(center - old_centers[i])

            return diff < self._tol ** 2 or iter >= self._max_iter

    def _recompute_centers(self, partials):
        while len(partials) > 1:
            subset = partials[:self._arity]
            partials = partials[self._arity:]
            partials.append(_merge(*subset))

        partials = compss_wait_on(partials)

        for idx, sum in enumerate(partials[0]):
            if sum[1] != 0:
                self.centers[idx] = sum[0] / sum[1]


@task(returns=np.array)
def _get_label(subset):
    return subset.labels


@task(returns=np.array)
def _init_centers(n_features, n_clusters, random_state):
    np.random.seed(random_state)
    centers = np.random.random((n_clusters, n_features))
    return centers


@task(subset=INOUT, returns=np.array)
def _partial_sum(subset, centers, set_labels):
    partials = np.zeros((centers.shape[0], 2), dtype=object)

    for idx, sample in enumerate(subset.samples):
        dist = np.linalg.norm(sample - centers, axis=1)
        min_center = np.argmin(dist)

        if set_labels:
            subset.set_label(idx, min_center)

        partials[min_center][0] += sample
        partials[min_center][1] += 1

    return partials


@task(returns=dict)
def _merge(*data):
    accum = data[0]

    for d in data[1:]:
        accum += d

    return accum


@task(subset=INOUT)
def _predict(subset, centers):
    for sample_idx, sample in enumerate(subset):
        dist = np.linalg.norm(sample - centers, axis=1)
        label = np.argmin(dist)
        subset.set_label(sample_idx, label)
