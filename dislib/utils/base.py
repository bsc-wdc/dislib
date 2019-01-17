import numpy as np
from pycompss.api.api import compss_wait_on
from pycompss.api.task import task

from dislib.data import Dataset, Subset


def as_grid(dataset, n_regions, dimensions=None, return_indices=False):
    """ Arranges samples in an n-dimensional grid where each Subset contains
    samples lying in one region of the feature space. The feature space is
    divided in ``n_regions`` equally sized regions on each dimension based on
    the maximum and minimum values of each feature in the dataset.

    Parameters
    ----------
    dataset : Dataset
        Input data.
    n_regions : int
        Number of regions per dimension in which to split the feature space.
    dimensions : iterable, optional (default=None)
        Integer indices of the dimensions to split. If None, all dimensions
        are split.
    return_indices : boolean, optional (default=False)
        Whether to return sorting indices.

    Returns
    -------
    grid_data : Dataset
        A new Dataset with one Subset per region in the feature space.
    index_array : array, shape = [n_samples, ]
        Array of indices that sort the samples in grid_data back to the
        order they have in the input Dataset.
    """
    n_features = dataset.n_features

    if dimensions is None:
        dimensions = range(n_features)

    grid_shape = [1] * n_features

    for dim in dimensions:
        grid_shape[dim] = n_regions

    grid_shape = tuple(grid_shape)

    min_ = dataset.min_features()
    max_ = dataset.max_features()
    bins = _generate_bins(min_, max_, dimensions, n_regions)

    sorted_data, sorting = _sort_data(dataset, grid_shape, bins, dimensions)
    sorted_data._min_features = np.copy(min_)
    sorted_data._max_features = np.copy(max_)
    ret_value = sorted_data

    if return_indices:
        subset_sizes = []

        for index, _ in enumerate(dataset):
            subset_sizes.append(dataset.subset_size(index))

        indices = _get_sorting_indices(sorting, subset_sizes)
        indices = np.argsort(indices)
        ret_value = sorted_data, indices

    return ret_value


def shuffle(dataset):
    """ Randomly shuffles a Dataset.

    Parameters
    ----------
    dataset : Dataset
        Input Dataset.

    Returns
    -------
    shuffled_data : Dataset
        A new ramdomly shuffled Dataset with the same number of Subsets as the
        input Dataset.
    """
    shuffled_data = Dataset(dataset.n_features)
    n_subsets = len(dataset)
    items = []

    for _ in dataset:
        items.append([])

    for subset_idx, subset in enumerate(dataset):
        subset_size = dataset.subset_size(subset_idx)
        sample_size = int(np.ceil(subset_size / n_subsets))
        indices = np.array(range(subset_size))
        i = 0

        while indices.size > 0:
            n_samples = min(sample_size, indices.size)
            choice = np.random.choice(indices, n_samples, replace=False)
            indices = np.setdiff1d(indices, choice)
            items[i].append(_get_items(subset, choice))
            i += 1

    for item in items:
        shuffled_data.append(_merge_subsets(*item))

    return shuffled_data


def _generate_bins(min_, max_, dimensions, n_regions):
    bins = []

    # create bins for the different regions in the grid in every dimension
    for dim in dimensions:
        # Add up a small delta to the max to include it in the binarization
        delta = max_[dim] / 1e8
        bin_ = np.linspace(min_[dim], max_[dim] + delta, n_regions + 1)
        bins.append(bin_)

    return bins


def _sort_data(dataset, grid_shape, bins, dimensions):
    sorted_data = Dataset(dataset.n_features)
    sorting_list = []

    for idx in np.ndindex(grid_shape):
        subset, subset_size, sorting = _filter(idx, bins, dimensions, *dataset)
        sorted_data.append(subset, subset_size)
        sorting_list.append(sorting)

    sorting = _merge_sorting(*sorting_list)

    return sorted_data, sorting


def _get_sorting_indices(sorting, subset_sizes):
    indices = []
    sorting = compss_wait_on(sorting)

    for part_ind, vec_ind in sorting:
        offset = np.sum(subset_sizes[:part_ind])

        for ind in vec_ind:
            indices.append(ind + offset)

    return np.array(indices, dtype=int)


@task(returns=1)
def _get_items(subset, indices):
    return subset[indices]


@task(returns=Dataset)
def _merge_subsets(*subsets):
    set0 = subsets[0]

    for setx in subsets[1:]:
        set0.concatenate(setx)

    return set0


@task(returns=3)
def _filter(idx, bins, dimensions, *dataset):
    filtered_samples = []
    filtered_labels = []
    sorting = []

    for set_idx, subset in enumerate(dataset):
        filtered_samples.append(subset.samples)
        final_ind = np.array(range(subset.samples.shape[0]))

        # filter vectors by checking if they lie in the given region (specified
        # by idx)
        for bin_idx, col_idx in enumerate(dimensions):
            col = filtered_samples[-1][:, col_idx]
            indices = np.digitize(col, bins[bin_idx]) - 1
            mask = (indices == idx[col_idx])
            filtered_samples[-1] = filtered_samples[-1][mask]
            final_ind = final_ind[mask]

            if filtered_samples[-1].size == 0:
                break

        if subset.labels is not None:
            filtered_labels.append(subset.labels[final_ind])

        sorting.append((set_idx, final_ind))

    final_samples = np.vstack(filtered_samples)
    final_labels = None

    if subset.labels is not None:
        final_labels = np.concatenate(filtered_labels)

    filtered_subset = Subset(samples=final_samples, labels=final_labels)

    return filtered_subset, final_samples.shape[0], sorting


@task(returns=1)
def _create_subset(samples, labels):
    if labels is None or None in labels:
        return Subset(samples=samples)
    else:
        return Subset(samples=samples, labels=labels)


@task(returns=1)
def _merge_sorting(*sorting_list):
    sorting_final = []

    for sorting in sorting_list:
        sorting_final.extend(sorting)

    return sorting_final
