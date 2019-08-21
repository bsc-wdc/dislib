import numpy as np
from pycompss.api.api import compss_wait_on, compss_delete_object
from pycompss.api.parameter import COLLECTION_INOUT
from pycompss.api.task import task

from dislib.data import Dataset


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


def _get_sorting_indices(sorting, subset_sizes):
    indices = []
    sorting = compss_wait_on(sorting)

    for part_ind, vec_ind in sorting:
        offset = np.sum(subset_sizes[:part_ind])

        for ind in vec_ind:
            indices.append(ind + offset)

    return np.array(indices, dtype=int)


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


@task(returns=1)
def _resample(subset, indices):
    return subset[indices]


@task(parts=COLLECTION_INOUT)
def _choose_and_assign_parts(subset, part_sizes, parts, seed):
    np.random.seed(seed)
    indices = np.random.permutation(subset.samples.shape[0])
    start = 0
    for i, size in enumerate(part_sizes):
        end = start + size
        parts[i] = subset[indices[start:end]]
        start = end


def _paired_partition(x, y):
    # Generator of tuples (x_part, y_part) that partitions x and y horizontally
    # with parts with corresponding samples. It follows the x array block
    # row-wise partition, and slices y accordingly. It should work even if the
    # blocks of x and y have a different number of rows.
    top_num_rows = x._top_left_shape[0]
    regular_num_rows = x._reg_shape[0]
    start_idx = 0
    end_idx = top_num_rows
    for x_row in x._iterator(axis=0):
        y_row = y[start_idx:end_idx]
        yield x_row, y_row
        start_idx = end_idx
        end_idx = min(end_idx + regular_num_rows, x.shape[0])
