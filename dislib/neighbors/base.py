import numpy as np
from pycompss.api.parameter import Depth, Type, COLLECTION_IN
from pycompss.api.task import task
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors as SKNeighbors
from sklearn.utils import validation

from dislib.data.array import Array


class NearestNeighbors(BaseEstimator):
    """ Unsupervised learner for implementing neighbor searches.

    Parameters
    ----------
    n_neighbors : int, optional (default=5)
        Number of neighbors to use by default for kneighbors queries.

    Examples
    --------
    >>> from dislib.neighbors import NearestNeighbors
    >>> import dislib as ds
    >>> data = ds.random_array((100, 5), block_size=(25, 5))
    >>> knn = NearestNeighbors(n_neighbors=10)
    >>> knn.fit(data)
    >>> distances, indices = knn.kneighbors(data)
    """

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, x):
        """ Fit the model using training data.

        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)
            Training data.

        Returns
        -------
        self : NearestNeighbors
        """
        self._fit_data = x
        return self

    def kneighbors(self, x, n_neighbors=None, return_distance=True):
        """ Finds the K nearest neighbors of the input samples. Returns
        indices and distances to the neighbors of each sample.

        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)
            The query samples.
        n_neighbors: int, optional (default=None)
            Number of neighbors to get. If None, the value passed in the
            constructor is employed.
        return_distance : boolean, optional (default=True)
            Whether to return distances.

        Returns
        -------
        dist : ds-array, shape=(n_samples, n_neighbors)
            Array representing the lengths to points, only present if
            return_distance=True.
        ind : ds-array, shape=(n_samples, n_neighbors)
            Indices of the nearest samples in the fitted data.
        """
        validation.check_is_fitted(self, '_fit_data')

        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        distances = []
        indices = []

        for q_row in x._iterator(axis=0):
            queries = []

            for row in self._fit_data._iterator(axis=0):
                queries.append(_get_neighbors(row._blocks, q_row._blocks,
                                              n_neighbors))

            dist, ind = _merge_queries(*queries)
            distances.append([dist])
            indices.append([ind])

        ind_arr = Array(blocks=indices,
                        top_left_shape=(x._top_left_shape[0], n_neighbors),
                        reg_shape=(x._reg_shape[0], n_neighbors),
                        shape=(x.shape[0], n_neighbors), sparse=False)

        if return_distance:
            dst_arr = Array(blocks=distances,
                            top_left_shape=(x._top_left_shape[0], n_neighbors),
                            reg_shape=(x._reg_shape[0], n_neighbors),
                            shape=(x.shape[0], n_neighbors), sparse=False)
            return dst_arr, ind_arr

        return ind_arr


@task(returns=2)
def _merge_queries(*queries):
    final_dist, final_ind, offset = queries[0]

    for dist, ind, n_samples in queries[1:]:
        ind += offset
        offset += n_samples

        # keep the indices of the samples that are at minimum distance
        m_ind = _min_indices(final_dist, dist)
        comb_ind = np.hstack((final_ind, ind))

        final_ind = np.array([comb_ind[i][m_ind[i]]
                              for i in range(comb_ind.shape[0])])

        # keep the minimum distances
        final_dist = _min_distances(final_dist, dist)

    return final_dist, final_ind


@task(blocks={Type: COLLECTION_IN, Depth: 2},
      q_blocks={Type: COLLECTION_IN, Depth: 2}, returns=tuple)
def _get_neighbors(blocks, q_blocks, n_neighbors):
    samples = Array._merge_blocks(blocks)
    q_samples = Array._merge_blocks(q_blocks)

    n_samples = samples.shape[0]

    knn = SKNeighbors(n_neighbors=n_neighbors)
    knn.fit(X=samples)
    dist, ind = knn.kneighbors(X=q_samples)

    return dist, ind, n_samples


def _min_distances(d1, d2):
    size, num = d1.shape
    d = [np.sort(np.hstack((d1[i], d2[i])))[:num] for i in range(size)]
    return np.array(d)


def _min_indices(d1, d2):
    size, num = d1.shape
    d = [np.argsort(np.hstack((d1[i], d2[i])))[:num] for i in range(size)]
    return np.array(d)
