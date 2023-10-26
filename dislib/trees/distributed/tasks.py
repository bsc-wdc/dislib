#!/usr/bin/python
#
#  Copyright 2002-2019 Barcelona Supercomputing Center (www.bsc.es)
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

# -*- coding: utf-8 -*-

from pycompss.api.task import task
from pycompss.api.parameter import COLLECTION_IN
from pycompss.api.parameter import COLLECTION_OUT
import numpy as np


@task(fragment=COLLECTION_IN, fragment_buckets=COLLECTION_OUT,
      range_min=COLLECTION_IN, range_max=COLLECTION_IN)
def filter_fragment(fragment, fragment_buckets, indexes_to_try,
                    num_buckets, range_min=0, range_max=1,
                    indexes_selected=None):
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
    fragment = np.block(fragment)
    range_min = np.block(range_min)
    range_max = np.block(range_max)
    for index, value in enumerate(indexes_to_try):
        if indexes_selected is not None:
            actual_fragment = fragment[indexes_selected, value]
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


@task(returns=dict, args=COLLECTION_IN)
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
