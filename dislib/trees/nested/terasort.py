import numpy as np
from dislib.trees.nested.tasks import filter_fragment, \
    combine_and_sort_bucket_elements
from dislib.data.array import Array
from pycompss.api.api import compss_delete_object, compss_wait_on


def terasorting(dataset, indexes_to_try, num_buckets,
                range_min=0, range_max=1,
                indexes_selected=None, reg_shape=None):
    # Init buckets dictionary
    list_of_buckets = []
    total_fragments = []
    if indexes_selected is not None:
        for idx, d in enumerate(dataset):
            fragment_buckets = [[object() for _ in range(num_buckets)]
                                for _ in range(len(indexes_to_try))]
            idx_selected = indexes_selected[indexes_selected <
                                            (idx + 1) * reg_shape]
            filter_fragment(d, fragment_buckets, indexes_to_try, num_buckets,
                            range_min=range_min._blocks,
                            range_max=range_max._blocks,
                            indexes_selected=idx_selected[
                                                 idx_selected >=
                                                 (idx) * reg_shape] %
                            reg_shape)
            total_fragments.append(fragment_buckets)
        total_fragments = np.array(compss_wait_on(total_fragments))
        for index in range(len(indexes_to_try)):
            buckets = {}
            for i in range(num_buckets):
                buckets[i] = []
            for i in range(num_buckets):
                buckets[i].append(total_fragments[:, index, i])
            list_of_buckets.append(buckets)
    else:
        buckets = {}
        for d in dataset:
            fragment_buckets = [[object() for _ in range(num_buckets)]
                                for _ in range(len(indexes_to_try))]
            filter_fragment(d, fragment_buckets, indexes_to_try, num_buckets,
                            range_min=range_min._blocks,
                            range_max=range_max._blocks,
                            indexes_selected=np.array([0]))
            total_fragments.append(fragment_buckets)
        total_fragments = np.array(compss_wait_on(total_fragments))
        for index in range(len(indexes_to_try)):
            buckets = {}
            for i in range(num_buckets):
                buckets[i] = []
            for i in range(num_buckets):
                buckets[i].append(total_fragments[:, index, i])
            list_of_buckets.append(buckets)
    result = dict()
    real_key = 0
    for index in range(len(indexes_to_try)):
        for key, value in list(list_of_buckets[index].items()):
            result[real_key] = combine_and_sort_bucket_elements(value[0])
            real_key += 1
    [compss_delete_object(future_objects) for
     value in buckets.items() for future_objects in value[1]]
    return_list = []
    for idx, value in enumerate(result.values()):
        if idx % num_buckets == 0:
            return_list.append([])
        return_list[-1].append(value)
    return return_list


def terasort(dataset, indexes_to_try, range_min=0, range_max=1,
             indexes_selected=None, num_buckets=4):
    """
    ----------------------
    Terasort main program
    ----------------------
    This application generates a set of fragments that contain randomly
    generated key, value tuples and sorts them all considering the key of
    each tuple.

    :param num_fragments: Number of fragments to generate
    :param num_entries: Number of entries (k,v tuples) within each fragment
    :param num_buckets: Number of buckets to consider.
    :param seed: Initial seed for the random number generator.
    """
    if isinstance(dataset, Array):
        result = terasorting(dataset._blocks, indexes_to_try, num_buckets,
                             range_min=range_min,
                             range_max=range_max,
                             indexes_selected=indexes_selected,
                             reg_shape=dataset._reg_shape[0])
        return np.array(result)
    else:
        result = terasorting(dataset, indexes_to_try, num_buckets,
                             range_min=range_min, range_max=range_max)
        return np.array(result)
