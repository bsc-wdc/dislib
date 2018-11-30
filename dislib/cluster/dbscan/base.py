from collections import defaultdict

import numpy as np
from pycompss.api.api import compss_wait_on
from pycompss.api.task import task

from dislib.cluster.dbscan.classes import Region
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
    grid_dim : int, optional (default=1)
        Number of regions per dimension in which to divide the feature space.
        The number of regions generated is equal to n_features ^ grid_dim.
    max_samples : int, optional (default=None)
        Setting max_samples to an integer results in the parallelization of
        the computation of distances inside each region of the grid. That
        is, each region is processed using various parallel tasks, where each
        task finds the neighbours of max_samples samples.

        This can be used to balance the load in scenarios where samples are not
        evenly distributed in the feature space.

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

    def __init__(self, eps=0.5, min_samples=5, arrange_data=True, grid_dim=1,
                 max_samples=None):
        assert grid_dim >= 1, "Grid dimensions must be greater or equal to 1."

        self._eps = eps
        self._min_samples = min_samples
        self._grid_dim = grid_dim
        self._arrange_data = arrange_data
        self.labels_ = np.empty(0, dtype=int)
        self._subset_sizes = []
        self._sorting = []
        self._max_samples = max_samples

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
        n_features = dataset.n_features
        grid = np.empty([self._grid_dim] * n_features, dtype=object)

        min_, max_ = self._get_min_max(dataset)
        bins, region_sizes = self._generate_bins(min_, max_, n_features)
        self._set_subset_sizes(dataset)

        if self._arrange_data:
            sorted_data = self._sort_data(dataset, grid, bins)
        else:
            sorted_data = dataset

        # Create regions
        for data_idx, region_id in enumerate(np.ndindex(grid.shape)):
            subset = sorted_data[data_idx]
            subset_size = self._subset_sizes[data_idx]
            grid[region_id] = Region(region_id, subset, subset_size, self._eps)

        # Set region neighbours
        distances = np.ceil(self._eps / region_sizes)

        for region_id in np.ndindex(grid.shape):
            self._add_neighbours(grid[region_id], grid, distances)

        # Run dbscan on each region
        for region_id in np.ndindex(grid.shape):
            region = grid[region_id]
            region.partial_scan(self._min_samples, self._max_samples)

        # Update labels computed by the different neighbouring regions
        for region_id in np.ndindex(grid.shape):
            neigh_sq_id = grid[region_id].neigh_sq_id
            labels_versions = []

            for neigh_idx in neigh_sq_id:
                labels_versions.append(grid[neigh_idx].cluster_labels[region_id])

            grid[region_id].cluster_labels[region_id] = _get_max_labels(*labels_versions)

        # Find connected clusters between regions
        # Iterate again over labels because the above loop changed them
        # FIXME: This probably can be done in a better way
        transitions = []

        for region_id in np.ndindex(grid.shape):
            neigh_sq_id = grid[region_id].neigh_sq_id

            labels_versions = []
            neigh_indices = []

            for neigh_idx in neigh_sq_id:
                labels_versions.append(grid[neigh_idx].cluster_labels[region_id])
                neigh_indices.append(neigh_idx)

            transitions.append(
                _compute_neighbour_transitions(region_id, neigh_indices,
                                               grid[region_id].cluster_labels[region_id],
                                               *labels_versions))

        transitions = _merge_transitions(*transitions)
        connected_comp = _get_connected_components(transitions)

        final_labels = []

        for region_id in np.ndindex(grid.shape):
            labels = grid[region_id].cluster_labels[region_id]
            final_labels.append(_update_labels(region_id, labels, connected_comp))

        final_labels = compss_wait_on(final_labels)

        for labels in final_labels:
            self.labels_ = np.concatenate((self.labels_, labels))

        # Modify labels to small numbers since the merging process
        # above can generate large labels (e.g., 150)
        # FIXME: This probably can be avoided by improving the merging of the
        #  labels computed by the different regions
        unique_labels = np.unique(self.labels_)
        unique_labels = unique_labels[unique_labels >= 0]

        for unique, label in enumerate(unique_labels):
            self.labels_[self.labels_ == label] = unique

        sorting_ind = self._get_sorting_indices()

        self.labels_ = self.labels_[np.argsort(sorting_ind)]

    @staticmethod
    def _get_min_max(dataset):
        minmax = []

        for subset in dataset:
            minmax.append(_min_max(subset))

        minmax = compss_wait_on(minmax)
        return np.min(minmax, axis=0)[0], np.max(minmax, axis=0)[1]

    def _set_subset_sizes(self, dataset):
        for subset in dataset:
            self._subset_sizes.append(_get_subset_size(subset))

        self._subset_sizes = compss_wait_on(self._subset_sizes)

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
            bin_ = np.linspace(min_[i], max_[i] + delta, self._grid_dim + 1)
            bins.append(bin_)
            region_sizes.append(np.max(bin_[1:] - bin_[0:-1]))

        return bins, np.array(region_sizes)

    def _get_sorting_indices(self):
        indices = []

        for part_ind, vec_ind in self._sorting:
            vec_ind = compss_wait_on(vec_ind)
            offset = np.sum(self._subset_sizes[:part_ind])

            for ind in vec_ind:
                indices.append(ind + offset)

        return np.array(indices, dtype=int)

    def _add_neighbours(self, region, grid, distances):
        for ind in np.ndindex(grid.shape):
            if ind == region.id:
                continue

            d = np.abs(np.array(region.id) - np.array(ind))

            if (d <= distances).all():
                region.add_neighbour(grid[ind])

@task(returns=1)
def _get_max_labels(*labels):
    label_arr = np.array(labels)
    return np.max(label_arr, axis=0)


@task(returns=1)
def _compute_neighbour_transitions(region_idx, neigh_indices, labels,
                                   *neigh_labels):
    transitions = defaultdict(set)

    for label_idx, label in enumerate(labels):
        if label < 0:
            continue

        label_key = region_idx + (label,)

        for neigh_idx, neigh_label in enumerate(neigh_labels):
            if neigh_indices[neigh_idx] != region_idx:
                neigh_key = neigh_indices[neigh_idx] + (
                    neigh_label[label_idx],)
                transitions[label_key].add(neigh_key)

    return transitions


@task(returns=1)
def _merge_transitions(*transitions):
    trans0 = transitions[0]

    for transition in transitions[1:]:
        trans0.update(transition)

    return trans0


@task(returns=1)
def _update_labels(region, labels, connected):
    for component in connected:
        old_label = -1
        min_ = np.inf

        for tup in component:
            min_ = min(min_, tup[-1])

            if tup[:-1] == region:
                old_label = tup[-1]

        if old_label >= 0:
            labels[labels == old_label] = min_

    return labels


@task(returns=1)
def _get_connected_components(transitions):
    visited = []
    connected = []
    for node, neighbours in transitions.items():
        if node in visited:
            continue

        connected.append([node])

        _visit_neighbours(transitions, neighbours, visited, connected)

    return connected


def _visit_neighbours(transitions, neighbours, visited, connected):
    for neighbour in neighbours:
        if neighbour in visited:
            continue

        visited.append(neighbour)
        connected[-1].append(neighbour)

        if neighbour in transitions:
            new_neighbours = transitions[neighbour]

            _visit_neighbours(transitions, new_neighbours, visited, connected)


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


@task(returns=np.array)
def _min_max(subset):
    mn = np.min(subset.samples, axis=0)
    mx = np.max(subset.samples, axis=0)
    return np.array([mn, mx])
