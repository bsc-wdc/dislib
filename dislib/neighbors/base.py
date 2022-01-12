import numpy as np
from pycompss.api.api import compss_delete_object
from pycompss.api.constraint import constraint
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
    >>> import dislib as ds
    >>> from dislib.neighbors import NearestNeighbors
    >>>
    >>>
    >>> if __name__ == '__main__':
    >>>     data = ds.random_array((100, 5), block_size=(25, 5))
    >>>     knn = NearestNeighbors(n_neighbors=10)
    >>>     knn.fit(data)
    >>>     distances, indices = knn.kneighbors(data)
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
        self._fit_data = list()

        for row in x._iterator(axis=0):
            sknnstruct = _compute_fit(row._blocks)
            n_samples = row.shape[0]
            self._fit_data.append((sknnstruct, n_samples))

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
            offset = 0

            for sknnstruct, n_samples in self._fit_data:
                queries.append(_get_kneighbors(sknnstruct, q_row._blocks, n_neighbors, offset))
                offset += n_samples

            dist, ind = _merge_kqueries(n_neighbors, *queries)
            compss_delete_object(*queries)
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

    # @abarcelo started to implement radius_neighbors and it was trivial.
    # Easy to implement, easy merge, everything was going smoothly.
    # Until I realized that the returning Array structure has variable row
    # size (the number of neighbors is not fixed). That is not supported
    # by the Array dislib structure (AFAIK).
    # @abarcelo then threw all the elegant implementation in a trash bin and
    # proceeded to go cry in a corner.
    # def radius_neighbors(self, x, radius=None, return_distance=True):


@constraint(computing_units="${ComputingUnits}")
@task(blocks={Type: COLLECTION_IN, Depth: 2}, returns=1)
def _compute_fit(blocks):
    samples = Array._merge_blocks(blocks)
    knn = SKNeighbors()
    return knn.fit(X=samples)


@constraint(computing_units="${ComputingUnits}")
@task(q_blocks={Type: COLLECTION_IN, Depth: 2}, returns=tuple)
def _get_kneighbors(sknnstruct, q_blocks, n_neighbors, offset):
    q_samples = Array._merge_blocks(q_blocks)

    # Note that the merge requires distances, so we ask for them
    dist, ind = sknnstruct.kneighbors(X=q_samples, n_neighbors=n_neighbors)

    # This converts the local indexes to global ones
    ind += offset

    return dist, ind


@constraint(computing_units="${ComputingUnits}")
@task(returns=2)
def _merge_kqueries(k, *queries):
    # Reorganize and flatten
    dist, ind = zip(*queries)
    aggr_dist = np.hstack(dist)
    aggr_ind = np.hstack(ind)

    # Final indexes of the indexes (sic)
    final_ii = np.argsort(aggr_dist)[:,:k]

    # Final results
    final_dist = np.take_along_axis(aggr_dist, final_ii, 1)
    final_ind = np.take_along_axis(aggr_ind, final_ii, 1)

    return final_dist, final_ind
