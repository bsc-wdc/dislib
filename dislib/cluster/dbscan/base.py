import numpy as np
from pycompss.api.api import compss_wait_on
from pycompss.api.constraint import constraint
from pycompss.api.parameter import COLLECTION_IN, COLLECTION_OUT, \
    Type, Depth
from pycompss.api.task import task
from scipy.sparse import issparse
from scipy.sparse import vstack as vstack_sparse
from sklearn.base import BaseEstimator
from sklearn.utils import validation

from dislib.cluster.dbscan.classes import Region
from dislib.data.array import Array


class DBSCAN(BaseEstimator):
    """ Perform DBSCAN clustering.

    This algorithm requires data to be arranged in a multidimensional grid.
    The fit method re-arranges input data before running the
    clustering algorithm. See ``fit()`` for more details.

    Parameters
    ----------
    eps : float, optional (default=0.5)
        The maximum distance between two samples for them to be considered as
        in the same neighborhood.
    min_samples : int, optional (default=5)
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.
    n_regions : int, optional (default=1)
        Number of regions per dimension in which to divide the feature space.
        The total number of regions generated is equal to ``n_regions`` ^
        ``len(dimensions)``.
    dimensions : iterable, optional (default=None)
        Integer indices of the dimensions of the feature space that should be
        divided. If None, all dimensions are divided.
    max_samples : int, optional (default=None)
        Setting max_samples to an integer results in the paralellization of
        the computation of distances inside each region of the grid. That
        is, each region is processed using various parallel tasks, where each
        task finds the neighbours of max_samples samples.

        This can be used to balance the load in scenarios where samples are not
        evenly distributed in the feature space.

    Attributes
    ----------
    n_clusters : int
        Number of clusters found. Accessing this member performs a
        synchronization.

    Examples
    --------
    >>> from dislib.cluster import DBSCAN
    >>> import dislib as ds
    >>> import numpy as np
    >>>
    >>>
    >>> if __name__ == '__main__':
    >>>     arr = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
    >>>     x = ds.array(arr, block_size=(2, 2))
    >>>     dbscan = DBSCAN(eps=3, min_samples=2)
    >>>     y = dbscan.fit_predict(x)
    >>>     print(y.collect())
    """

    def __init__(self, eps=0.5, min_samples=5, n_regions=1,
                 dimensions=None, max_samples=None):
        assert n_regions >= 1, \
            "Number of regions must be greater or equal to 1."

        self.eps = eps
        self.min_samples = min_samples
        self.n_regions = n_regions
        self.dimensions = dimensions
        self.max_samples = max_samples

    def fit(self, x, y=None):
        """ Perform DBSCAN clustering on x.

        Samples are initially rearranged in a multidimensional grid with
        ``n_regions`` regions per dimension in ``dimensions``. All regions
        in a specific dimension have the same size.

        Parameters
        ----------
        x : ds-array
            Input data.
        y : ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : DBSCAN
        """
        assert self.n_regions >= 1, \
            "Number of regions must be greater or equal to 1."

        self._subset_sizes = []
        self._sorting = []
        self._components = None
        self._labels = None

        n_features = x.shape[1]
        sparse = x._sparse

        self._dimensions = self.dimensions
        if self.dimensions is None:
            self._dimensions = range(n_features)

        n_dims = len(self._dimensions)

        arranged_data, indices, sizes = _arrange_samples(x, self.n_regions,
                                                         self._dimensions)

        grid = np.empty((self.n_regions,) * n_dims, dtype=object)

        region_widths = self._compute_region_widths(x)[self._dimensions]
        sizes = compss_wait_on(sizes)

        # Create regions
        for subset_idx, region_id in enumerate(np.ndindex(grid.shape)):
            subset = arranged_data[subset_idx]
            subset_size = sizes[subset_idx]
            grid[region_id] = Region(region_id, subset, subset_size,
                                     self.eps, sparse)

        # Set region neighbours
        distances = np.ceil(self.eps / region_widths)

        for region_id in np.ndindex(grid.shape):
            self._add_neighbours(grid[region_id], grid, distances)

        # Run dbscan on each region
        for region_id in np.ndindex(grid.shape):
            region = grid[region_id]
            region.partial_dbscan(self.min_samples, self.max_samples)

        # Compute label equivalences between different regions
        equiv_list = []

        for region_id in np.ndindex(grid.shape):
            equiv_list.append(grid[region_id].get_equivalences())

        equivalences = _merge_dicts(*equiv_list)

        # Compute connected components
        self._components = _get_connected_components(equivalences)

        # Update region labels according to equivalences
        final_labels = []

        for subset_idx, region_id in enumerate(np.ndindex(grid.shape)):
            region = grid[region_id]
            region.update_labels(self._components)
            final_labels.append(region.labels)

        label_blocks = _rearrange_labels(final_labels, indices, x._n_blocks[0])

        self._labels = Array(blocks=label_blocks,
                             top_left_shape=(x._top_left_shape[0], 1),
                             reg_shape=(x._reg_shape[0], 1),
                             shape=(x._shape[0], 1), sparse=False)
        return self

    def fit_predict(self, x):
        """ Perform DBSCAN clustering on dataset and return cluster labels
        for x.

        Parameters
        ----------
        x : ds-array
            Input data.

        Returns
        -------
        y : ds-array, shape=(n_samples  , 1)
            Cluster labels.
        """
        self.fit(x)
        return self._labels

    @staticmethod
    def _add_neighbours(region, grid, distances):
        for ind in np.ndindex(grid.shape):
            if ind == region.id:
                continue

            d = np.abs(np.array(region.id) - np.array(ind))

            if (d <= distances).all():
                region.add_neighbour(grid[ind])

    @property
    def n_clusters(self):
        validation.check_is_fitted(self, '_components')
        self._components = compss_wait_on(self._components)
        return len(self._components)

    def _compute_region_widths(self, x):
        mn = x.min().collect()
        mx = x.max().collect()

        if issparse(mn):
            mn = mn.toarray()
            mx = mx.toarray()

        return ((mx - mn) / self.n_regions).reshape(-1, )


def _arrange_samples(x, n_regions, dimensions=None):
    """ Arranges samples in an n-dimensional grid. The feature space is
    divided in ``n_regions`` equally sized regions on each dimension based on
    the maximum and minimum values of each feature in x.

    Parameters
    ----------
    x : ds-array
        Input data.
    n_regions : int
        Number of regions per dimension in which to split the feature space.
    dimensions : iterable, optional (default=None)
        Integer indices of the dimensions to split. If None, all dimensions
        are split.

    Returns
    -------
    grid_data : list
        A list of nd-arrays (futures) containing the samples on each region.
        That is, grid_data[i][j] contains the samples in row block i of x
        that lie in region j.
    sorting : list of lists
        sorting[i][j] contains the sample indices of the
        samples from row block i that lie in region j. The indices
        are relative to row block i.
    sizes : list
        Sizes (futures) of the arrays in grid_data.
    """
    n_features = x.shape[1]

    if dimensions is None:
        dimensions = range(n_features)

    grid_shape = (n_regions,) * len(dimensions)

    # min() and max() calls have synchronization points
    mn = x.min()
    mx = x.max()

    bins = _generate_bins(mn._blocks, mx._blocks, dimensions, n_regions)

    total_regions = n_regions ** len(dimensions)

    return _arrange_data(x, grid_shape, bins, dimensions, total_regions)


def _arrange_data(x, g_shape, bins, dims, total_regions):
    reg_lists = list()
    ind_lists = list()

    for row in x._iterator(axis=0):
        reg_list = [object() for _ in range(total_regions)]
        ind_list = [object() for _ in range(total_regions)]

        # after calling arrange_block, reg_list contains one nd-array per
        # region with the corresponding samples, and ind_list contains
        # the indices of the samples that go to each region
        _arrange_block(row._blocks, bins, dims, g_shape, reg_list, ind_list)

        reg_lists.append(reg_list)
        ind_lists.append(ind_list)

    # the ith element of each element in lol contains the samples of
    # the ith region.
    reg_arr = np.asarray(reg_lists)
    arranged_samples = list()
    sizes = list()

    for i in range(reg_arr.shape[1]):
        # we merge together the ith element of each element in reg_arr and
        # sort_arr to obtain a single nd-array per region (convert to list
        # again because collections do not support np.arrays)
        samples, size = _merge_samples(reg_arr[:, i].tolist(), x._sparse)
        arranged_samples.append(samples)
        sizes.append(size)

    # arranged_samples is a list of nd-arrays (one per region) containing the
    # samples in each region.
    return arranged_samples, ind_lists, sizes


def _rearrange_labels(labels, indices, n_blocks):
    """
    This method rearranges computed labels back to their original position.
    """
    blocks_list = list()

    for i, arr in enumerate(labels):
        blocks = [object() for _ in range(n_blocks)]

        # blocks_list[i][j] contains the labels of region i that belong to
        # row block j in the original arrangement of the data
        _rearrange_region(arr, np.asarray(indices)[:, i].tolist(), blocks)
        blocks_list.append(blocks)

    blocks_arr = np.asarray(blocks_list)
    sorted_blocks = list()

    # merge and sort the rearranged labels to build the final array of labels
    for i in range(blocks_arr.shape[1]):
        label_block = _merge_labels(blocks_arr[:, i].tolist(), indices[i])
        sorted_blocks.append([label_block])

    return sorted_blocks


@constraint(computing_units="${ComputingUnits}")
@task(mn={Type: COLLECTION_IN, Depth: 2},
      mx={Type: COLLECTION_IN, Depth: 2},
      returns=1)
def _generate_bins(mn, mx, dimensions, n_regions):
    bins = []
    mn_arr = Array._merge_blocks(mn)[0]
    mx_arr = Array._merge_blocks(mx)[0]

    if issparse(mn_arr):
        mn_arr = mn_arr.toarray()[0]
        mx_arr = mx_arr.toarray()[0]

    # create bins for the different regions in the grid in every dimension
    for dim in dimensions:
        bin_ = np.linspace(mn_arr[dim], mx_arr[dim], n_regions + 1)
        bins.append(bin_)

    return bins


@constraint(computing_units="${ComputingUnits}")
@task(blocks={Type: COLLECTION_IN, Depth: 2},
      samples_list={Type: COLLECTION_OUT},
      indices={Type: COLLECTION_OUT})
def _arrange_block(blocks, bins, dimensions, shape, samples_list, indices):
    x = Array._merge_blocks(blocks)
    n_bins = shape[0]
    region_indices = list()

    # find the samples that belong to each region iterating over each dimension
    for dim_bins, dim in zip(bins, dimensions):
        col = x[:, dim]

        if issparse(col):
            col = col.toarray().flatten()

        # digitize a dimension of all the samples into the corresponding bins
        # region_idx represents the region index at dimension dim of each
        # sample
        region_idx = np.digitize(col, dim_bins) - 1
        region_idx[region_idx >= n_bins] = n_bins - 1
        region_indices.append(region_idx)

    # idx_arr is an nd-array of shape (n_dimensions, n_samples), where each
    # column represents the region indices of each sample (i.e., the region
    # where the sample should go)
    idx_arr = np.asarray(region_indices)

    # apply np.ravel_multi_index to each column of idx_arr to get a 1-D index
    # that represents each region in the output list
    out_idx = np.apply_along_axis(np.ravel_multi_index, 0, idx_arr, dims=shape)

    for i in range(len(samples_list)):
        # insert all the samples that belong to a region to the corresponding
        # place in the output list.
        sample_indices = np.where(out_idx == i)
        samples_list[i] = x[sample_indices]

        # sorting contains which samples go to which region
        indices[i] = sample_indices


@constraint(computing_units="${ComputingUnits}")
@task(indices=COLLECTION_IN,
      blocks=COLLECTION_OUT)
def _rearrange_region(labels, indices, blocks):
    """
    indices[i] contains the label/sample indices of row block i (in the
    original data) that lie in this region. This method
    redistributes the labels into a list representing the row blocks
    in the original data
    """
    start, end = 0, 0

    for i, ind in enumerate(indices):
        end += len(ind[0])
        blocks[i] = labels[start:end].reshape(-1, 1)
        start = end


@constraint(computing_units="${ComputingUnits}")
@task(samples_list={Type: COLLECTION_IN}, returns=2)
def _merge_samples(samples_list, sparse):
    if sparse:
        samples = vstack_sparse(samples_list)
    else:
        samples = np.vstack(samples_list)

    return samples, samples.shape[0]


@constraint(computing_units="${ComputingUnits}")
@task(labels_list=COLLECTION_IN, indices=COLLECTION_IN, returns=1)
def _merge_labels(labels_list, indices):
    labels = np.vstack(labels_list)

    # idx contains the original position of each label in labels
    idx = np.hstack(np.asarray(indices).flatten())

    return np.take(labels, idx).reshape(-1, 1)


@constraint(computing_units="${ComputingUnits}")
@task(returns=1)
def _merge_dicts(*dicts):
    merged_dict = {}

    for dct in dicts:
        merged_dict.update(dct)

    return merged_dict


@constraint(computing_units="${ComputingUnits}")
@task(returns=1)
def _get_connected_components(equiv):
    # Add inverse equivalences
    for node, neighs in equiv.items():
        for neigh in neighs:
            equiv[neigh].add(node)

    visited = set()
    connected = []

    for node, neighbours in equiv.items():
        if node in visited:
            continue

        connected.append([node])
        _visit_neighbours(equiv, neighbours, visited, connected)

    return connected


def _visit_neighbours(equiv, neighbours, visited, connected):
    to_visit = list(neighbours)

    while len(to_visit) > 0:
        neighbour = to_visit.pop()

        if neighbour in visited:
            continue

        visited.add(neighbour)
        connected[-1].append(neighbour)

        if neighbour in equiv:
            to_visit.extend(equiv[neighbour])
