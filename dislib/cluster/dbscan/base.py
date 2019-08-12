import numpy as np
from pycompss.api.api import compss_wait_on
from pycompss.api.parameter import INOUT, COLLECTION_IN, COLLECTION_INOUT, \
    Type, Depth
from pycompss.api.task import task
from scipy.sparse import issparse
from scipy.sparse import vstack as vstack_sparse

from dislib.cluster.dbscan.classes import Region
from dislib.data.array import Array


class DBSCAN():
    """ Perform DBSCAN clustering.

    This algorithm requires data to be arranged in a multidimensional grid.
    The default behavior is to re-arrange input data before running the
    clustering algorithm. See ``fit()`` for more details.

    Parameters
    ----------
    eps : float, optional (default=0.5)
        The maximum distance between two samples for them to be considered as
        in the same neighborhood.
    min_samples : int, optional (default=5)
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.
    arrange_data: boolean, optional (default=True)
        Whether to re-arrange input data before performing clustering. If
        ``arrange_data=False``, ``n_regions`` and ``dimensions`` have no
        effect.
    n_regions : int, optional (default=1)
        Number of regions per dimension in which to divide the feature space.
        The total number of regions generated is equal to ``n_regions`` ^
        ``len(dimensions)``. If ``arrange_data=False``, ``n_regions`` is
        ignored.
    dimensions : iterable, optional (default=None)
        Integer indices of the dimensions of the feature space that should be
        divided. If None, all dimensions are divided. If ``arrange_data=False``
        , ``dimensions`` is ignored.
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
        Number of clusters found.

    Examples
    --------
    >>> from dislib.cluster import DBSCAN
    >>> import numpy as np
    >>> x = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
    >>> from dislib.data import load_data
    >>> train_data = load_data(x, subset_size=2)
    >>> dbscan = DBSCAN(eps=3, min_samples=2)
    >>> dbscan.fit(train_data)
    >>> print(train_data.labels)
    """

    def __init__(self, eps=0.5, min_samples=5, arrange_data=True, n_regions=1,
                 dimensions=None, max_samples=None):
        assert n_regions >= 1, \
            "Number of regions must be greater or equal to 1."

        self._eps = eps
        self._min_samples = min_samples
        self._n_regions = n_regions
        self._dimensions_init = dimensions
        self._dimensions = dimensions
        self._arrange_data = arrange_data
        self._subset_sizes = []
        self._sorting = []
        self._max_samples = max_samples
        self._components = None

    def fit(self, x, y=None):
        """ Perform DBSCAN clustering on data and sets dataset.labels.

        If arrange_data=True, data is initially rearranged in a
        multidimensional grid with ``n_regions`` regions per dimension in
        ``dimensions``. All regions in a specific dimension have the same
        size.

        For example, suppose that data contains N partitions of 2-dimensional
        samples (``n_features=2``), where the first feature ranges from 1 to 5
        and the second feature ranges from 0 to 1. Then, n_regions=10
        re-arranges data into 10^2=100 new partitions, where each partition
        contains the samples that lie in one region of the grid.
        numpy.linspace() is employed to divide the feature space into
        uniform regions.

        If data is already arranged in a grid, then the number of partitions
        in data must be equal to ``n_regions`` ^ ``len(dimensions)``. The
        equivalence between partition and region index is computed using
        numpy.ravel_multi_index().

        Parameters
        ----------
        x : ds-array
            Input data.
        y : ignored
            Not used, present here for API consistency by convention.
        """
        n_features = x.shape[1]
        sparse = x._sparse

        if self._dimensions_init is None:
            self._dimensions = range(n_features)

        n_dims = len(self._dimensions)

        sorted_data, sorting = _arrange_samples(x, self._n_regions,
                                                self._dimensions)

        grid = np.empty((self._n_regions,) * n_dims, dtype=object)
        region_widths = self._compute_region_widths(x)[self._dimensions]

        # Create regions
        for subset_idx, region_id in enumerate(np.ndindex(grid.shape)):
            subset = sorted_data[subset_idx]
            subset_size = sorted_data.subset_size(subset_idx)
            grid[region_id] = Region(region_id, subset, subset_size,
                                     self._eps, sparse)

        # Set region neighbours
        distances = np.ceil(self._eps / region_widths)

        for region_id in np.ndindex(grid.shape):
            self._add_neighbours(grid[region_id], grid, distances)

        # Run dbscan on each region
        for region_id in np.ndindex(grid.shape):
            region = grid[region_id]
            region.partial_dbscan(self._min_samples, self._max_samples)

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

            if not self._arrange_data:
                _set_labels(dataset[subset_idx], region.labels)

        if self._arrange_data:
            self._sort_labels_back(dataset, final_labels, sorting)

    def fit_predict(self, dataset):
        """ Perform DBSCAN clustering on dataset. This method does the same
        as fit(), and is provided for API standardization purposes.

        Parameters
        ----------
        dataset : Dataset
            Input data.

        See also
        --------
        fit
        """
        self.fit(dataset)

    @staticmethod
    def _add_neighbours(region, grid, distances):
        for ind in np.ndindex(grid.shape):
            if ind == region.id:
                continue

            d = np.abs(np.array(region.id) - np.array(ind))

            if (d <= distances).all():
                region.add_neighbour(grid[ind])

    @staticmethod
    def _sort_labels_back(dataset, final_labels, sorting_ind):
        final_labels = _concatenate_labels(sorting_ind, *final_labels)
        begin = 0
        end = 0

        for subset_idx, subset in enumerate(dataset):
            end += dataset.subset_size(subset_idx)
            _set_labels(subset, final_labels, begin, end)
            begin = end

    @property
    def n_clusters(self):
        self._components = compss_wait_on(self._components)
        return len(self._components)

    def _compute_region_widths(self, x):
        mn = x.min().collect()
        mx = x.max().collect()
        return (mx - mn) / self._n_regions


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
    sorting : nd-array
        sorting[i][j] contains the sample indices of the
        samples from row block i that lie in region j. The indices
        are relative to row block i.
    """
    n_features = x.shape[1]

    if dimensions is None:
        dimensions = range(n_features)

    grid_shape = (n_regions,) * len(dimensions)

    mn = x.min()
    mx = x.max()

    bins = _generate_bins(mn._blocks, mx._blocks, dimensions, n_regions)

    total_regions = n_regions ** len(dimensions)

    return _arrange_data(x, grid_shape, bins, dimensions, total_regions)


def _arrange_data(x, g_shape, bins, dimensions, total_regions):
    out_lol = list()
    sort_lol = list()

    for row in x._iterator(axis=0):
        out_list = [object()] * total_regions
        sort_list = [object()] * total_regions

        # after calling arrange_block, out_list contains one nd-array per
        # region with the corresponding samples, and sort_list contains
        # the indices of the samples that go to each region
        _arrange_block(row._blocks, bins, dimensions, g_shape, out_list,
                       sort_list)

        out_lol.append(out_list)
        sort_lol.append(sort_list)

    # the ith element of each element in lol contains the samples of
    # the ith region.
    out_arr = np.asarray(out_lol)
    sorted_data = list()

    for i in range(out_arr.shape[1]):
        # we merge together the ith element of each element in out_arr and
        # sort_arr to obtain a single nd-array per region
        samples = _merge_samples(out_arr[:, i], x._sparse)
        sorted_data.append(samples)

    # sorted_data is a list of nd-arrays (one per region) containing the
    # samples in each region.
    return sorted_data, np.asarray(sort_lol)


@task(mn={Type: COLLECTION_IN, Depth: 2},
      mx={Type: COLLECTION_IN, Depth: 2},
      returns=1)
def _generate_bins(mn, mx, dimensions, n_regions):
    bins = []
    mn_arr = Array._merge_blocks(mn)[0]
    mx_arr = Array._merge_blocks(mx)[0]

    # create bins for the different regions in the grid in every dimension
    for dim in dimensions:
        bin_ = np.linspace(mn_arr[dim], mx_arr[dim], n_regions + 1)
        bins.append(bin_)

    return bins


@task(blocks={Type: COLLECTION_IN, Depth: 2},
      sorted_list={Type: COLLECTION_INOUT},
      sorting={Type: COLLECTION_INOUT},
      returns=1)
def _arrange_block(blocks, bins, dimensions, shape, sorted_list, sorting):
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

    for i in range(len(sorted_list)):
        # insert all the samples that belong to a region to the corresponding
        # place in the output list.
        sample_indices = np.where(out_idx == i)
        sorted_list[i] = x[sample_indices]

        # sorting contains which samples go to which region
        sorting[i] = sample_indices


@task(returns=1)
def _merge_samples(samples_arr, sparse):
    if sparse:
        return vstack_sparse(samples_arr)
    else:
        return np.vstack(samples_arr)


@task(returns=1)
def _merge_dicts(*dicts):
    merged_dict = {}

    for dict in dicts:
        merged_dict.update(dict)

    return merged_dict


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


@task(returns=1)
def _concatenate_labels(sorting, *labels):
    final_labels = np.empty(0, dtype=int)

    for label_arr in labels:
        final_labels = np.concatenate((final_labels, label_arr))

    return final_labels[sorting]


@task(subset=INOUT)
def _set_labels(subset, labels, begin=0, end=None):
    subset.labels = labels[begin:end]
