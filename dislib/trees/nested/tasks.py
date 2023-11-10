from pycompss.api.task import task
from pycompss.api.parameter import COLLECTION_IN
from pycompss.api.parameter import COLLECTION_OUT
from pycompss.api.constraint import constraint
import numpy as np


@constraint(computing_units="${ComputingUnits}")
@task(fragment=COLLECTION_IN, fragment_buckets=COLLECTION_OUT,
      range_min=COLLECTION_IN, range_max=COLLECTION_IN)
def filter_fragment(fragment, fragment_buckets, indexes_to_try,
                    num_buckets, range_min=0, range_max=1,
                    indexes_selected=np.array([0])):
    """
    Task that filters a fragment entries for the given ranges.
        * Ranges is a list of tuples where each tuple corresponds to
          a range.
        * Each tuple (range) is composed by two elements, the minimum
          and maximum of each range.
        * The filtering is performed by checking which fragment entries'
          keys belong to each range.
    The entries that belong to each range are considered a bucket.
        * The variable buckets is a list of lists, where the inner lists
          correspond to the bucket of each range.

    :param fragment: The fragment to be sorted and filtered.
    :param ranges: The ranges to apply when filtering.
    :return: Multireturn of the buckets.
    """
    if len(fragment) == 0:
        for idx in range(len(fragment_buckets)):
            for idx_2 in range(len(fragment_buckets[idx])):
                fragment_buckets[idx][idx_2] = []
        return
    fragment = np.block(fragment)
    range_min = np.block(range_min)
    range_max = np.block(range_max)
    for index, value in enumerate(indexes_to_try):
        if len(indexes_selected) > 1:
            if indexes_selected[0] != 0:
                actual_fragment = fragment[indexes_selected, value]
            else:
                actual_fragment = fragment[:, value]
        else:
            actual_fragment = fragment[:, value]
        split_indexes = np.linspace(range_min[0, value],
                                    range_max[0, value] + 1, num_buckets + 1)
        ranges = []
        for ind in range(split_indexes.size - 1):
            ranges.append((split_indexes[ind], split_indexes[ind + 1]))
        i = 0
        for _range in ranges:
            if actual_fragment is not None:
                fragment_buckets[index][i] = [k_s_v for k_s_v in
                                              actual_fragment if
                                              _range[0] <= k_s_v < _range[1]]
            else:
                fragment_buckets[index][i] = []
            i += 1


def combine_and_sort_bucket_elements(args):
    """
    Task that combines the buckets received as args parameter and final
    sorting.

    args structure = ([],[], ..., [])

    :param args: args that contains the buckets of a single range
    :return: A list of tuples with the same format as provided initially
             sorted by key.
    """
    combined = []
    for e in args:
        for kv in e:
            combined.append(kv)
    sorted_by_key = np.sort(combined)
    return np.unique(sorted_by_key)
