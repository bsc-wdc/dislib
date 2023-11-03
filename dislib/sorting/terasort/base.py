import numpy as np
from dislib.data.array import Array
from pycompss.api.task import task
from pycompss.api.constraint import constraint
from pycompss.api.parameter import COLLECTION_IN, COLLECTION_OUT, Type, Depth
from pycompss.api.api import compss_delete_object, compss_wait_on
import math


class TeraSort:
    """TeraSort algorithm for ordering ds_arrays
    by columns, or the whole values.

        Parameters
        ----------
        range_min : int or list, optional (default=0)
            Minimum value for attribute (if int) or for columns (if list)
        range_max : int or list, optional (default=100)
            Maximum value for attribute (if int) or for columns (if list)
        column_indexes : np.ndarray, list or None, optional (default=None)
            Column indexes to sort independtly of the rest of columns.
        num_buckets : int, optional
            Number of buckets to divide the data in the ds.array and do the
            sorting. A big number of num_buckets may lead to slower execution.
        """

    def __init__(self, range_min=0, range_max=100, column_indexes=None,
                 num_buckets=None):
        if isinstance(range_min, int):
            self.range_min = [[np.array([[range_min]])]]
        elif isinstance(range_min, list):
            self.range_min = [[np.array([range_min])]]
        if isinstance(range_max, int):
            self.range_max = [[np.array([[range_max]])]]
        elif isinstance(range_max, list):
            self.range_max = [[np.array([range_max])]]
        if np.any(self.range_min[0][0] >= self.range_max[0][0]):
            raise ValueError("Minimum value range should be lower than "
                             "maximum value range.")
        if not isinstance(column_indexes, np.ndarray) and \
                column_indexes is not None:
            self.column_indexes = np.array(column_indexes)
        else:
            self.column_indexes = column_indexes
        if not isinstance(num_buckets, int) and num_buckets is not None:
            raise ValueError("The number of buckets should be an integer.")
        self.num_buckets = num_buckets

    def fit(self, x, y=None):
        """Fits the Range Max and Range Min of the TeraSort.

        Parameters
        ----------
        x : ds-array, shape=(n_rows, n_columns)
            The data to sort.
        y : None
            To maintain standard API.

        Returns
        -------
        """
        self.range_min = x.min()
        self.range_max = x.max()
        if self.column_indexes is None:
            self.range_min = [[_obtain_minimum_value(
                self.range_min._blocks)]]
            self.range_max = [[_obtain_maximum_value(
                self.range_max._blocks)]]

    def sort(self, x):
        """Sorts the data in x.

        Parameters
        ----------
        x : ds-array, shape=(n_rows, n_columns)
            The data to sort.

        Returns
        -------
        x : A copy of x sorted.
        """
        if isinstance(x, Array):
            if self.num_buckets is None:
                self.num_buckets = x._n_blocks[0] * x._n_blocks[1]
            if self.column_indexes is not None:
                if isinstance(self.range_max, list):
                    if len(self.range_max[0][0][0]) != \
                            len(self.column_indexes) and \
                            len(self.range_max[0][0][0]) > 1:
                        raise ValueError("There should be one maximum "
                                         "value specified per column"
                                         " or one unique max value "
                                         "for all the columns.")
                if isinstance(self.range_min, list):
                    if len(self.range_min[0][0][0]) != \
                            len(self.column_indexes) and \
                            len(self.range_min[0][0][0]) > 1:
                        raise ValueError("There should be one minimum "
                                         "value specified per column"
                                         " or one unique max value "
                                         "for all the columns.")
                result = []
                len_result = []
                for idx, column_index in enumerate(self.column_indexes):
                    if isinstance(self.range_min, Array):
                        range_min = self.range_min[:, column_index:
                                                   (column_index + 1)]
                    else:
                        range_min = self.range_min
                    if isinstance(self.range_min, Array):
                        range_max = self.range_max[:, column_index:
                                                   (column_index + 1)]
                    else:
                        range_max = self.range_max
                    result_attribute, len_buckets = _terasort(
                        x[:, column_index:(column_index + 1)],
                        range_min,
                        range_max, self.num_buckets)
                    result.append(result_attribute)
                    len_result.append(len_buckets)
                len_result = compss_wait_on(len_result)
                final_block = x.shape[0] % x._reg_shape[0] \
                    if x.shape[0] % x._reg_shape[0] != 0 \
                    else x._reg_shape[0]
                attribute_blocks = []
                for idx, attribute_length_buckets in enumerate(len_result):
                    out_blocks = [object() for _ in range(x._n_blocks[0])]
                    remaining_count = 0
                    positions_to_use = []
                    actual_block = 0
                    for idx_key, length_bucket in \
                            enumerate(attribute_length_buckets):
                        remaining_count += length_bucket
                        positions_to_use.append(result[idx][idx_key])
                        not_ended = True
                        while not_ended:
                            if actual_block == (x._n_blocks[0] - 1):
                                if remaining_count >= final_block:
                                    out_block = [out_blocks[actual_block]]
                                    _get_attribute_column_blocks(
                                        out_block,
                                        positions_to_use,
                                        remaining_count,
                                        final_block)
                                    out_blocks[actual_block] = \
                                        out_block[0]
                                    remaining_count -= final_block
                                    positions_to_use = \
                                        [positions_to_use[-1]]
                                    actual_block += 1
                                else:
                                    not_ended = False
                            else:
                                if remaining_count >= x._reg_shape[0]:
                                    out_block = [out_blocks[actual_block]]
                                    _get_attribute_column_blocks(
                                        out_block,
                                        positions_to_use,
                                        remaining_count,
                                        x._reg_shape[0])
                                    out_blocks[actual_block] = \
                                        out_block[0]
                                    remaining_count -= x._reg_shape[0]
                                    positions_to_use = \
                                        [positions_to_use[-1]]
                                    actual_block += 1
                                else:
                                    not_ended = False
                    attribute_blocks.append(out_blocks)
                attribute_blocks = np.array(attribute_blocks)
                used_indexes = 0
                number_columns = x._reg_shape[1] if \
                    x._reg_shape[1] < len(self.column_indexes) \
                    else len(self.column_indexes)
                out_blocks = [[object() for _ in
                               range(math.ceil(
                                   len(self.column_indexes) /
                                   x._reg_shape[1]))] for _ in
                              range(len(attribute_blocks[0]))]
                while used_indexes < len(self.column_indexes):
                    for i in range(len(out_blocks)):
                        out_block = [out_blocks[i]
                                     [math.floor(used_indexes /
                                                 number_columns)]]
                        _join_different_attribute_columns(
                            out_block,
                            attribute_blocks[used_indexes:
                                             used_indexes+number_columns,
                                             i])
                        out_blocks[i][math.floor(used_indexes /
                                                 number_columns)] = \
                            out_block[0]
                    used_indexes += number_columns
                return Array(blocks=out_blocks,
                             top_left_shape=(x._top_left_shape[0],
                                             number_columns),
                             reg_shape=(x._reg_shape[0], number_columns),
                             shape=(x._shape[0],
                                    len(self.column_indexes)),
                             sparse=x._sparse)
            else:
                result, len_result = _terasort(x, self.range_min,
                                               self.range_max,
                                               self.num_buckets)
            len_block = x._reg_shape[0] * x._reg_shape[1]
            shape_final_column_block = (x._reg_shape[0],
                                        (x._shape[1] % x._reg_shape[1])
                                        if (x._shape[1] % x._reg_shape[1])
                                        != 0 else x._reg_shape[1])
            final_column_block = shape_final_column_block[0] * \
                shape_final_column_block[1]
            shape_final_row_block = ((x._shape[0] % x._reg_shape[0]) if
                                     (x._shape[0] % x._reg_shape[0]) != 0
                                     else x._reg_shape[0], x._reg_shape[1])
            final_row_block = shape_final_row_block[0] * \
                shape_final_row_block[1]
            shape_final_block = ((x._shape[0] % x._reg_shape[0]) if
                                 (x._shape[0] % x._reg_shape[0]) != 0
                                 else x._reg_shape[0], (x._shape[1] %
                                                        x._reg_shape[1])
                                 if (x._shape[1] % x._reg_shape[1]) != 0
                                 else x._reg_shape[1])
            final_block = shape_final_block[0] * shape_final_block[1]
            final_lens = compss_wait_on(len_result)
            out_blocks = [[object() for _ in range(x._n_blocks[1])]
                          for _ in range(x._n_blocks[0])]
            remaining_count = 0
            positions_to_use = []
            actual_block = 0
            for idx, length_bucket in enumerate(final_lens):
                remaining_count += length_bucket
                positions_to_use.append(result[idx])
                not_ended = True
                while not_ended:
                    if ((actual_block + 1) % x._n_blocks[0] != 0 or
                        actual_block == 0) and actual_block < (
                            (x._n_blocks[0] - 1) * x._n_blocks[1]):
                        if remaining_count >= len_block:
                            out_block = [out_blocks[math.floor(
                                actual_block / x._n_blocks[1])]
                                         [actual_block % x._n_blocks[1]]]
                            _join_buckets_block(out_block,
                                                positions_to_use,
                                                remaining_count,
                                                len_block,
                                                (x._reg_shape[0],
                                                 x._reg_shape[1]))
                            out_blocks[math.floor(
                                actual_block / x._n_blocks[1])][
                                actual_block % x._n_blocks[1]] = \
                                out_block[0]
                            remaining_count -= len_block
                            positions_to_use = [positions_to_use[-1]]
                            actual_block += 1
                        else:
                            not_ended = False
                    elif actual_block >= ((x._n_blocks[0] - 1) *
                                          x._n_blocks[1]) and \
                            ((actual_block) % x._n_blocks[0] != 0
                             or (x._n_blocks[0] == 1 and
                                 idx < (len(final_lens) - 1))):
                        if remaining_count >= final_row_block:
                            out_block = [out_blocks[math.floor
                                                    (actual_block /
                                                     x._n_blocks[1])][
                                        actual_block % x._n_blocks[1]]]
                            _join_buckets_block(out_block,
                                                positions_to_use,
                                                remaining_count,
                                                final_row_block,
                                                shape_final_row_block)
                            out_blocks[math.floor
                                       (actual_block /
                                        x._n_blocks[1])][
                                        actual_block % x._n_blocks[1]] = \
                                out_block[0]
                            remaining_count -= final_row_block
                            positions_to_use = [positions_to_use[-1]]
                            actual_block += 1
                        else:
                            not_ended = False
                    elif actual_block < ((x._n_blocks[0] - 1) *
                                         x._n_blocks[1]) and \
                            (actual_block) % x._n_blocks[0] != 0:
                        if remaining_count >= final_column_block:
                            out_block = [out_blocks[math.floor
                                                    (actual_block /
                                                     x._n_blocks[1])][
                                             actual_block % x._n_blocks[1]]
                                         ]
                            _join_buckets_block(out_block,
                                                positions_to_use,
                                                remaining_count,
                                                final_column_block,
                                                shape_final_column_block)
                            out_blocks[math.floor
                                       (actual_block /
                                        x._n_blocks[1])][
                                actual_block % x._n_blocks[1]] = \
                                out_block[0]
                            remaining_count -= final_column_block
                            positions_to_use = [positions_to_use[-1]]
                            actual_block += 1
                        else:
                            not_ended = False
                    else:
                        if remaining_count >= final_block:
                            out_block = [out_blocks[math.floor
                                                    (actual_block /
                                                     x._n_blocks[1])][
                                             actual_block % x._n_blocks[1]]
                                         ]
                            _join_buckets_block(out_block,
                                                positions_to_use,
                                                remaining_count,
                                                final_block,
                                                shape_final_block)
                            out_blocks[math.floor
                                       (actual_block /
                                        x._n_blocks[1])][
                                actual_block % x._n_blocks[1]] = \
                                out_block[0]
                            remaining_count -= final_block
                            positions_to_use = [positions_to_use[-1]]
                            actual_block += 1
                        else:
                            not_ended = False
            if x._n_blocks[1] > 1:
                for idx, unordered_blocks in enumerate(out_blocks):
                    out_block = [object() for _ in range(x._n_blocks[1])]
                    _reorder_rows(out_block, unordered_blocks)
                    out_blocks[idx] = out_block
            return Array(blocks=out_blocks,
                         top_left_shape=x._top_left_shape,
                         reg_shape=(x._reg_shape),
                         shape=(x._shape),
                         sparse=x._sparse)
        else:
            raise ValueError("Object type not supported. "
                             "Sorting only implemented for ds-array objects.")


def _terasort(x, range_min, range_max, num_buckets):
    buckets = {}
    for i in range(num_buckets):
        buckets[i] = []
    for x_block in x._blocks:
        fragment_buckets = [object() for _ in range(num_buckets)]
        if isinstance(range_min, Array):
            _filter_fragment(x_block, fragment_buckets, num_buckets,
                             range_min=range_min._blocks,
                             range_max=range_max._blocks)
        else:
            _filter_fragment(x_block, fragment_buckets, num_buckets,
                             range_min=range_min,
                             range_max=range_max)
        for i in range(num_buckets):
            buckets[i].append(fragment_buckets[i])
    result = dict()
    len_buckets = []
    for key, value in list(buckets.items()):
        result[key], len_bucket = \
            _combine_and_sort_bucket_elements(tuple(value))
        len_buckets.append(len_bucket)
    [compss_delete_object(future_objects) for value in buckets.items()
     for future_objects in value[1]]
    return result, len_buckets


@constraint(computing_units="${ComputingUnits}")
@task(blocks={Type: COLLECTION_OUT, Depth: 1},
      single_column_data={Type: COLLECTION_IN, Depth: 1})
def _join_different_attribute_columns(blocks, single_column_data):
    blocks[0] = np.hstack(single_column_data)


@constraint(computing_units="${ComputingUnits}")
@task(blocks={Type: COLLECTION_OUT, Depth: 1},
      buckets={Type: COLLECTION_IN, Depth: 1})
def _get_attribute_column_blocks(blocks, buckets, remaining_data, len_block):
    buckets = np.block(buckets)
    if -remaining_data+len_block == 0:
        blocks[0] = buckets[-remaining_data:].reshape(-1, 1)
    else:
        blocks[0] = buckets[-remaining_data:
                            -remaining_data+len_block].reshape(-1, 1)


@constraint(computing_units="${ComputingUnits}")
@task(blocks={Type: COLLECTION_OUT, Depth: 1},
      blocks_in={Type: COLLECTION_IN, Depth: 1})
def _reorder_rows(blocks, blocks_in):
    total_block = np.sort(np.block(blocks_in).flatten()).\
        reshape(blocks_in[0].shape[0], -1)
    data_column_used = 0
    for idx in range(len(blocks)):
        data_row = blocks_in[idx].shape[0]
        data_column = blocks_in[idx].shape[1]
        blocks[idx] = total_block[:data_row, data_column_used:
                                  data_column_used+data_column
                                  ].reshape(data_row, data_column)
        data_column_used += data_column


@constraint(computing_units="${ComputingUnits}")
@task(blocks={Type: COLLECTION_OUT, Depth: 1},
      buckets={Type: COLLECTION_IN, Depth: 1})
def _join_buckets_block(blocks, buckets, remaining_data,
                        len_block, block_shape):
    buckets = np.block(buckets)
    if -remaining_data+len_block == 0:
        blocks[0] = buckets[-remaining_data:].reshape(*block_shape)
    else:
        blocks[0] = buckets[-remaining_data:
                            -remaining_data+len_block].reshape(*block_shape)


@constraint(computing_units="${ComputingUnits}")
@task(fragment={Type: COLLECTION_IN, Depth: 1},
      fragment_buckets={Type: COLLECTION_OUT, Depth: 1},
      range_min={Type: COLLECTION_IN, Depth: 2},
      range_max={Type: COLLECTION_IN, Depth: 2})
def _filter_fragment(fragment, fragment_buckets, num_buckets, range_min=0,
                     range_max=1):
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
    split_indexes = np.linspace(range_min[0][0][0][0],
                                range_max[0][0][0][0] * 1.1,
                                num_buckets + 1)
    ranges = []
    for ind in range(split_indexes.size - 1):
        ranges.append((split_indexes[ind], split_indexes[ind + 1]))
    i = 0
    for _range in ranges:
        actual_fragment_bucket = []
        for k_v in fragment:
            if k_v is not None:
                if len(k_v) > 0:
                    k_v_flat = k_v.flatten()
                    actual_fragment_bucket.extend([k_s_v for k_s_v in
                                                   k_v_flat if
                                                   _range[0] <= k_s_v
                                                   < _range[1]])
                else:
                    actual_fragment_bucket.extend([k_s_v for k_s_v in
                                                   k_v if
                                                   _range[0] <= k_s_v
                                                   < _range[1]])
            else:
                fragment_buckets[i] = []
        fragment_buckets[i] = actual_fragment_bucket
        i += 1


@constraint(computing_units="${ComputingUnits}")
@task(returns=2, args={Type: COLLECTION_IN, Depth: 1})
def _combine_and_sort_bucket_elements(args):
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
    return sorted_by_key, len(sorted_by_key)


@constraint(computing_units="${ComputingUnits}")
@task(blocks={Type: COLLECTION_IN, Depth: 2}, returns=1)
def _obtain_minimum_value(blocks):
    minimum_values = np.block(blocks)
    return np.array([[np.amin(minimum_values)]])


@constraint(computing_units="${ComputingUnits}")
@task(blocks={Type: COLLECTION_IN, Depth: 2}, returns=1)
def _obtain_maximum_value(blocks):
    maximum_values = np.block(blocks)
    return np.array([[np.amax(maximum_values)]])
