import numpy as np
from pycompss.api.api import compss_wait_on, compss_delete_object
from pycompss.api.parameter import COLLECTION_INOUT
from pycompss.api.task import task
from scipy.sparse import vstack

from dislib.data import Dataset, Subset


def resample(dataset, n_samples, random_state=None):
    """ Resamples a dataset without replacement.

    Parameters
    ----------
    dataset : Dataset
        Input data.
    n_samples : int
        Number of samples to generate.
    random_state : int or RandomState, optional (default=None)
        Seed or numpy.random.RandomState instance to use in the generation of
        random numbers.

    Returns
    -------
    resampled_data : Dataset
        Resampled dataset. The number of subsets in the returned dataset is
        less or equal to the number of subsets in the input dataset.
    """
    r_data = Dataset(dataset.n_features, dataset.sparse)
    np.random.seed(random_state)
    sizes = dataset.subsets_sizes()
    indices = np.random.choice(range(sum(sizes)), size=n_samples)
    offset = 0

    for subset, size in zip(dataset, sizes):
        subset_indices = indices - offset
        subset_indices = subset_indices[subset_indices >= 0]
        subset_indices = subset_indices[subset_indices < size]

        if subset_indices.size > 0:
            r_data.append(_resample(subset, subset_indices))

        offset += size

    return r_data


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

    grid_shape = (n_regions,) * len(dimensions)

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


def shuffle(dataset_in, n_subsets_out=None, random_state=None):
    """ Randomly shuffles a Dataset.

    Parameters
    ----------
    dataset_in : Dataset
        Input Dataset.
    n_subsets_out : int, optional (default=None)
        Number of Subsets in the shuffled dataset. If None, it is the same as
        in the input Dataset.
    random_state : int or RandomState, optional (default=None)
        Seed or numpy.random.RandomState instance to use in the generation of
        random numbers.

    Returns
    -------
    shuffled_dataset : Dataset
        A new randomly shuffled Dataset with n_subsets_out balanced Subsets.
        If even splits are impossible, some Subsets contain 1 extra instance.
        These extra instances are evenly distributed to make k-fold splits
        (with k divisor of the number of subsets) as balanced as possible.
    """
    np.random.seed(random_state)
    if n_subsets_out is None:
        n_subsets_out = len(dataset_in)
    sizes_in = dataset_in.subsets_sizes()
    n_samples = sum(sizes_in)
    sizes_out = _balanced_distribution(n_samples, n_subsets_out)

    # Matrix of subsets of samples going from subset_in_i to subset_out_j
    all_parts = []

    # For each subset_in, get the parts going to each subset_out
    for subset, size in zip(dataset_in, sizes_in):
        parts, part_sizes = _partition_subset(subset, size, sizes_out)
        all_parts.append(parts)
        sizes_out -= part_sizes

    shuffled_dataset = Dataset(dataset_in.n_features, dataset_in.sparse)
    for j in range(n_subsets_out):
        parts_to_j = [parts[j] for parts in all_parts]
        seed = np.random.randint(np.iinfo(np.int32).max)
        shuffled_dataset.append(_merge_shuffle(seed, *parts_to_j))
        # Clean parts to save disk space
        for part in parts_to_j:
            compss_delete_object(part)

    return shuffled_dataset


def _balanced_distribution(n_elements, n_parts):
    part_lengths = np.full((n_parts,), n_elements // n_parts)
    remainder = n_elements % n_parts
    spaced_remainder = np.linspace(0, n_parts - 1, remainder, dtype=int)
    part_lengths[spaced_remainder] += 1
    return part_lengths


def _generate_bins(min_, max_, dimensions, n_regions):
    bins = []

    # create bins for the different regions in the grid in every dimension
    for dim in dimensions:
        bin_ = np.linspace(min_[dim], max_[dim], n_regions + 1)
        bins.append(bin_)

    return bins


def _sort_data(dataset, grid_shape, bins, dimensions):
    sorted_data = Dataset(dataset.n_features, dataset.sparse)
    sorting_list = []

    for idx in np.ndindex(grid_shape):
        subset, subset_size, sorting = _filter(idx, bins, dimensions,
                                               dataset.sparse, *dataset)
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
def _merge_shuffle(seed, *subsets):
    merged = subsets[0].copy()

    for setx in subsets[1:]:
        merged.concatenate(setx)

    np.random.seed(seed)
    p = np.random.permutation(merged.samples.shape[0])
    merged.samples = merged.samples[p]
    if merged.labels is not None:
        merged.labels = merged.labels[p]

    return merged


@task(returns=3)
def _filter(idx, bins, dimensions, sparse, *dataset):
    filtered_samples = []
    filtered_labels = []
    sorting = []
    n_bins = bins[0].shape[0] - 1

    for set_idx, subset in enumerate(dataset):
        filtered_samples.append(subset.samples)
        final_ind = np.array(range(subset.samples.shape[0]))

        # filter vectors by checking if they lie in the given region (specified
        # by idx)
        for bin_idx, col_idx in enumerate(dimensions):
            col = filtered_samples[-1][:, col_idx]

            if sparse:
                col = col.toarray().flatten()

            indices = np.digitize(col, bins[bin_idx]) - 1
            indices[indices >= n_bins] = n_bins - 1
            mask = (indices == idx[bin_idx])
            filtered_samples[-1] = filtered_samples[-1][mask]
            final_ind = final_ind[mask]

            if filtered_samples[-1].size == 0:
                break

        if subset.labels is not None:
            filtered_labels.append(subset.labels[final_ind])

        sorting.append((set_idx, final_ind))

    if not sparse:
        final_samples = np.vstack(filtered_samples)
    else:
        final_samples = vstack(filtered_samples)

    final_labels = None

    if subset.labels is not None:
        final_labels = np.concatenate(filtered_labels)

    filtered_subset = Subset(samples=final_samples, labels=final_labels)

    return filtered_subset, final_samples.shape[0], sorting


@task(returns=1)
def _merge_sorting(*sorting_list):
    sorting_final = []

    for sorting in sorting_list:
        sorting_final.extend(sorting)

    return sorting_final


@task(returns=1)
def _resample(subset, indices):
    return subset[indices]


def _partition_subset(subset, n_samples, sizes_out):
    n_subsets_out = len(sizes_out)
    part_sizes = np.zeros((n_subsets_out,), dtype=int)
    for j in range(n_subsets_out):
        if n_samples == 0:
            continue
        # Decide how many of the remaining elements of subset will go to
        # subset_out_j. This is given by an hypergeometric distribution.
        n_good = sizes_out[j]
        n_bad = sum(sizes_out[j:]) - sizes_out[j]
        n_selected = np.random.hypergeometric(n_good, n_bad, n_samples)
        part_sizes[j] = n_selected
        n_samples -= n_selected

    parts = [{} for _ in range(n_subsets_out)]
    seed = np.random.randint(np.iinfo(np.int32).max)
    _choose_and_assign_parts(subset, part_sizes, parts, seed)
    return parts, part_sizes


@task(parts=COLLECTION_INOUT)
def _choose_and_assign_parts(subset, part_sizes, parts, seed):
    np.random.seed(seed)
    indices = np.random.permutation(subset.samples.shape[0])
    start = 0
    for i, size in enumerate(part_sizes):
        end = start + size
        parts[i] = subset[indices[start:end]]
        start = end
