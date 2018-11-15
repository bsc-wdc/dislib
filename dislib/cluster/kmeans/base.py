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
    fit(data)
        Compute K-means clustering.
    fit_predict(data)
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

    def fit(self, data):
        """ Compute k-means clustering.

        Parameters
        ----------
        data : List of Dataset
            Unlabeled data divided in partitions.
        """
        future = _init_centers(data[0], self._n_clusters, self._random_state)
        self.centers = compss_wait_on(future)

        old_centers = None
        iter = 0

        while not self._converged(old_centers, iter):
            old_centers = np.array(self.centers)
            partials = []
            for part in data:
                partials.append(_partial_sum(part, old_centers))

            self._recompute_centers(partials)
            iter += 1

        self.n_iter = iter

    def fit_predict(self, data):
        """ Performs clustering on data, and sets and returns the cluster
        labels.

        Parameters
        ----------
        data : List of Dataset
            Unlabeled data divided in partitions.

        Returns
        -------
        y : ndarray, shape (n_samples)
            Cluster labels.

        Notes
        -----
        This method modifies the input dataset by setting the cluster labels.
        """

        self.fit(data)
        labels = []

        for part in data:
            labels.append(_get_label(part))

        return np.array(compss_wait_on(labels))

    def predict(self, x):
        """ Predict the closest cluster each sample in x belongs to.

        Parameters
        ----------
        x : ndarray
            New data to predict.

        Returns
        -------
        labels : ndarray
            Index of the cluster each sample belongs to.
        """
        labels = []

        for x in x:
            dist = np.linalg.norm(x - self.centers, axis=1)
            labels.append(np.argmin(dist))
        return np.array(labels)

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
def _get_label(data):
    return data.labels


@task(returns=np.array)
def _init_centers(data, n_clusters, random_state):
    np.random.seed(random_state)
    n_features = data.vectors.shape[1]
    centers = np.random.random((n_clusters, n_features))
    return centers


@task(data=INOUT, returns=np.array)
def _partial_sum(data, centers):
    partials = np.zeros((centers.shape[0], 2), dtype=object)

    for idx, vec in enumerate(data.vectors):
        dist = np.linalg.norm(vec - centers, axis=1)
        min_center = np.argmin(dist)
        data.set_label(idx, min_center)
        partials[min_center][0] += vec
        partials[min_center][1] += 1

    return partials


@task(returns=dict)
def _merge(*data):
    accum = data[0]

    for d in data[1:]:
        accum += d

    return accum
