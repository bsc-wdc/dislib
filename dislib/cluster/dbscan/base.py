from collections import defaultdict

import numpy as np
from pycompss.api.api import compss_wait_on
from pycompss.api.task import task

from dislib.cluster.dbscan.classes import DisjointSet
from dislib.cluster.dbscan.classes import Square
from dislib.data import Dataset


class DBSCAN():
    """ Perform DBSCAN clustering.

    This algorithm requires data to be arranged in a multidimensional grid.
    The default behavior is to re-arrange input data before running the
    clustering algorithm. See fit() for more details.


    Parameters
    ----------
    eps : float, optional (default=0.5)
        The maximum distance between two samples for them to be considered as
        in the same neighborhood.
    min_samples : int, optional (default=5)
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.
    arrange_data: boolean, optional (default=True)
        Whether to re-arrange input data before performing clustering.
    grid_dim : int, optional (default=10)
        Number of regions per dimension in which to divide the feature space.
        The number of regions generated is equal to n_features ^ grid_dim.

    Attributes
    ----------
    labels_ : array, shape = [n_samples]
        Cluster labels for each point in the dataset given to fit(). Noisy
        samples are given the label -1.

    Methods
    -------
    fit(dataset)
        Perform DBSCAN clustering.
    """

    def __init__(self, eps=0.5, min_samples=5, arrange_data=True, grid_dim=10):
        assert grid_dim >= 1, "Grid dimensions must be greater than 1."

        self._eps = eps
        self._min_samples = min_samples
        self._grid_dim = grid_dim
        self._arrange_data = arrange_data
        self.labels_ = np.empty(0)
        self._subset_sizes = []
        self._sorting = []

    def fit(self, dataset):
        """ Perform DBSCAN clustering on data.

        If arrange_data=True, data is initially rearranged in a
        multidimensional grid with grid_dim regions per dimension. Regions
        are uniform in size.

        For example, suppose that data contains N partitions of 2-dimensional
        samples (n_features=2), where the first feature ranges from 1 to 5 and
        the second feature ranges from 0 to 1. Then, grid_dim=10 re-arranges
        data into 10^2=100 new partitions, where each partition contains the
        samples that lie in one region of the grid. numpy.linspace() is
        employed to divide the feature space into uniform regions.

        If data is already arranged in a grid, then the number of partitions
        in data must be equal to grid_dim ^ n_features. The equivalence
        between partition and region index is computed using
        numpy.ravel_multi_index().

        Parameters
        ----------
        dataset : Dataset
            Input data.
        """
        n_features = compss_wait_on(_get_n_features(dataset[0]))
        grid = np.empty([self._grid_dim] * n_features, dtype=object)

        assert (self._arrange_data or grid.size == len(dataset)), \
            "%s partitions required for grid dimension = %s and number of " \
            "features = %s. Got %s partitions instead" % (grid.size,
                                                          self._grid_dim,
                                                          n_features,
                                                          len(dataset))

        min_, max_ = self._get_min_max(dataset)
        bins, region_sizes = self._generate_bins(min_, max_, n_features)

        if self._arrange_data:
            self._set_subset_sizes(dataset)
            sorted_data = self._sort_data(dataset, grid, bins)
        else:
            sorted_data = dataset

        # This threshold determines the granularity of the tasks
        TH_1 = 1000

        links = defaultdict()

        for ind in np.ndindex(grid.shape):
            grid[ind] = Square(ind, self._eps, grid.shape, region_sizes)
            grid[ind].init_data(sorted_data, grid.shape)
            grid[ind].partial_scan(self._min_samples, TH_1)

        # In spite of not needing a synchronization we loop again since the first
        # loop initialises all the future objects that are used afterwards
        for ind in np.ndindex(grid.shape):
            # We retrieve all the neighbour square ids from the current square
            neigh_sq_id = grid[ind].neigh_sq_id
            labels_versions = []
            for neigh_comb in neigh_sq_id:
                # We obtain the labels found for our points by our neighbours.
                labels_versions.append(grid[neigh_comb].cluster_labels[ind])

            # We merge all the different labels found and return merging rules
            links[ind] = grid[ind].sync_labels(*labels_versions)

        # We synchronize all the merging loops
        for ind in np.ndindex(grid.shape):
            links[ind] = compss_wait_on(links[ind])

        # We sync all the links locally and broadcast the updated global labels
        # to all the workers
        updated_links = self._sync_relations(links)

        # We lastly output the results to text files. For performance testing
        # the following two lines could be commented.
        for ind in np.ndindex(grid.shape):
            grid[ind].update_labels(updated_links)
            labels = np.array(grid[ind].get_labels())
            self.labels_ = np.concatenate((self.labels_, labels))

        sorting_ind = self._get_sorting_indices()

        self.labels_ = self.labels_[np.argsort(sorting_ind)]

    def _sync_relations(self, cluster_rules):
        out = defaultdict(set)
        for comb in cluster_rules:
            for key in cluster_rules[comb]:
                out[key] |= cluster_rules[comb][key]

        mf_set = DisjointSet(out.keys())
        for key in out:
            tmp = list(out[key])
            for i in range(len(tmp) - 1):
                # Added this for loop to compare all vs all
                mf_set.union(tmp[i], tmp[i + 1])
                # for j in range(i, len(tmp)):
                #     mf_set.union(tmp[i], tmp[j])

        return mf_set.get()

    def _get_min_max(self, data):
        minmax = []

        for part in data:
            minmax.append(_min_max(part))

        minmax = compss_wait_on(minmax)
        return np.min(minmax, axis=0)[0], np.max(minmax, axis=0)[1]

    def _set_subset_sizes(self, dataset):
        for subset in dataset:
            self._subset_sizes.append(_get_subset_size(subset))

    def _sort_data(self, dataset, grid, bins):
        sorted_data = Dataset(dataset.n_features)

        for idx in np.ndindex(grid.shape):
            sample_list = []

            # for each partition get the vectors in a particular region
            for set_idx, subset in enumerate(dataset):
                indices, samples = _filter(subset, idx, bins)
                sample_list.append(samples)
                self._sorting.append((set_idx, indices))

            # create a new partition representing the grid region
            sorted_data.append(_merge(*sample_list))

        return sorted_data

    def _generate_bins(self, min_, max_, n_features):
        bins = []
        region_sizes = []

        # create bins for the different regions in the grid in every dimension
        for i in range(n_features):
            # Add up a small delta to the max to include it in the binarization
            delta = max_[i] / 1e8
            bin = np.linspace(min_[i], max_[i] + delta, self._grid_dim + 1)
            bins.append(bin)
            region_sizes.append(np.max(bin[1:] - bin[0:-1]))

        return bins, np.array(region_sizes)

    def _get_sorting_indices(self):
        indices = []
        self._subset_sizes = compss_wait_on(self._subset_sizes)

        for part_ind, vec_ind in self._sorting:
            vec_ind = compss_wait_on(vec_ind)
            offset = np.sum(self._subset_sizes[:part_ind])

            for ind in vec_ind:
                indices.append(ind + offset)

        return np.array(indices, dtype=int)


@task(returns=int)
def _get_subset_size(subset):
    return subset.samples.shape[0]


@task(returns=1)
def _merge(*samples):
    from dislib.data import Subset
    return Subset(np.vstack(samples))


@task(returns=2)
def _filter(subset, idx, bins):
    filtered_samples = subset.samples
    final_ind = np.array(range(filtered_samples.shape[0]))

    # filter vectors by checking if they lie in the given region (specified
    # by idx)
    for col_ind in range(filtered_samples.shape[1]):
        col = filtered_samples[:, col_ind]
        indices = np.digitize(col, bins[col_ind]) - 1
        mask = (indices == idx[col_ind])
        filtered_samples = filtered_samples[mask]
        final_ind = final_ind[mask]

        if filtered_samples.size == 0:
            break

    return final_ind, filtered_samples


@task(returns=int)
def _get_n_features(subset):
    return subset.samples.shape[1]


@task(returns=np.array)
def _min_max(subset):
    mn = np.min(subset.samples, axis=0)
    mx = np.max(subset.samples, axis=0)
    return np.array([mn, mx])


@task(returns=np.array)
def _get_sorting_indices(sorting, subset_sizes):
    indices = []

    for set_idx, sample_idx in sorting:
        offset = np.sum(subset_sizes[:set_idx])

        for idx in sample_idx:
            indices.append(idx + offset)

    return np.array(indices)