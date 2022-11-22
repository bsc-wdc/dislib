import numpy as np
from dislib.data.array import Array, array
from pycompss.api.task import task
from pycompss.api.parameter import COLLECTION_IN, COLLECTION_OUT, \
    COLLECTION_INOUT, Type, Depth
from pycompss.api.api import compss_delete_object, compss_barrier_group, \
    compss_wait_on
from pycompss.api.exceptions import COMPSsException
from pycompss.api.api import TaskGroup
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
        num_buckets_reorder : int, optional
            Number of buckets to use when reordering the data
            for the construction of the ds.array blocks.
        returns : string, optional (default="ds_array)
        synchronize : boolean, optional (default=False)
            Boolean to decide if in the case of returning a ds.array
            the blocks are synchronized before constructing the ds.array
            or if the construction of the blocks should be done in
            distributed.
        """

    def __init__(self, range_min=0, range_max=100, column_indexes=None,
                 num_buckets=None,
                 num_buckets_reorder=None,
                 returns="ds_array", synchronize=False):
        if isinstance(range_min, int):
            self.range_min = [[np.array([[range_min]])]]
        elif isinstance(range_min, list):
            self.range_min = [[np.array(range_min)]]
        if isinstance(range_max, int):
            self.range_max = [[np.array([[range_max]])]]
        elif isinstance(range_max, list):
            self.range_max = [[np.array(range_max)]]
        if np.any(self.range_min >= self.range_max):
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
        if returns != "ds_array" and returns != "list" and returns != "dict":
            raise ValueError("The algorithm can only return a list or a "
                             "dict.")
        self.returns = returns
        self.synchronize = synchronize
        if num_buckets_reorder is not None:
            self.num_buckets_reorder = num_buckets_reorder
        else:
            self.num_buckets_reorder = 2

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
            self.range_min = [[obtain_minimum_value(
                self.range_min._blocks)]]
            self.range_max = [[obtain_maximum_value(
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
                    result_attribute = terasort(x[:, column_index:
                                                  (column_index + 1)],
                                                range_min,
                                                range_max, self.num_buckets)
                    result.append(result_attribute)
                if self.synchronize and self.returns == "ds_array":
                    final_blocks = compss_wait_on(result)
                    final_blocks = redistribute_synchronized_blocks_columns(
                        final_blocks, reg_shape=x._reg_shape,
                        row_blocks=x._n_blocks[0],
                        number_columns=len(self.column_indexes),
                        outer_rows=x._shape[0] % x._reg_shape[0])

                    return array(np.block(final_blocks),
                                 (x._reg_shape[0],
                                  len(self.column_indexes)))
                elif self.returns == "ds_array":
                    final_blocks = redistribute_blocks_columns(
                        result, reg_shape=x._reg_shape,
                        row_blocks=x._n_blocks[0],
                        number_columns=len(self.column_indexes),
                        outer_rows=x._shape[0] % x._reg_shape[0])

                    return Array(blocks=final_blocks,
                                 top_left_shape=x._top_left_shape,
                                 reg_shape=(x._reg_shape[0],
                                            len(self.column_indexes)),
                                 shape=(x._shape[0],
                                        len(self.column_indexes)),
                                 sparse=x._sparse)
            else:
                result = terasort(x, self.range_min, self.range_max,
                                  self.num_buckets, returns=self.returns)
            if self.returns == "list":
                return list(result.values())
            elif self.returns == "dict":
                if self.column_indexes is not None:
                    raise NotImplementedError("The option of returning a "
                                              "dictionary when "
                                              "specifying column indexes is "
                                              "not implemented.")
                return result
            elif self.returns == "ds_array":
                if self.synchronize:
                    final_blocks = compss_wait_on(result)
                    final_blocks = redistribute_synchronized_blocks(
                        final_blocks,
                        reg_shape=x._reg_shape,
                        row_blocks=x._n_blocks[0],
                        col_blocks=x._n_blocks[1],
                        outer_columns=x._shape[1] % x._reg_shape[1],
                        outer_rows=x._shape[0] % x._reg_shape[0])
                    return array(np.block(final_blocks),
                                 x._reg_shape)
                else:
                    final_blocks = redistribute_blocks_with_exception(
                        result, self.num_buckets_reorder,
                        reg_shape=x._reg_shape,
                        row_blocks=x._n_blocks[0],
                        col_blocks=x._n_blocks[1],
                        outer_columns=x._shape[1] % x._reg_shape[1],
                        outer_rows=x._shape[0] % x._reg_shape[0])
                    return Array(blocks=final_blocks,
                                 top_left_shape=x._top_left_shape,
                                 reg_shape=x._reg_shape,
                                 shape=x._shape, sparse=x._sparse)

        else:
            raise ValueError("Object type not supported. "
                             "Sorting only implemented for ds-array objects.")


def terasort(x, range_min, range_max, num_buckets, returns="list"):
    buckets = {}
    for i in range(num_buckets):
        buckets[i] = []
    for x_block in x._blocks:
        fragment_buckets = [object() for _ in range(num_buckets)]
        if isinstance(range_min, Array):
            filter_fragment(x_block, fragment_buckets, num_buckets,
                            range_min=range_min._blocks,
                            range_max=range_max._blocks)
        else:
            filter_fragment(x_block, fragment_buckets, num_buckets,
                            range_min=range_min,
                            range_max=range_max)
        for i in range(num_buckets):
            buckets[i].append(fragment_buckets[i])
    result = dict()
    for key, value in list(buckets.items()):
        result[key] = combine_and_sort_bucket_elements(tuple(value))
    [compss_delete_object(future_objects) for value in buckets.items()
     for future_objects in value[1]]
    if returns == "dict":
        return result
    return list(result.values())


@task(checks={Type: COLLECTION_IN, Depth: 1},
      blocks={Type: COLLECTION_INOUT, Depth: 1})
def evaluate_exception(checks, blocks):
    if np.all(checks):
        raise COMPSsException("All blocks contain the correct "
                              "number of elements.")


def redistribute_synchronized_blocks_columns(blocks,
                                             reg_shape,
                                             row_blocks,
                                             number_columns, outer_rows=0):
    final_blocks = [[None for _ in range(
        math.ceil(reg_shape[1] / number_columns))]
                    for _ in range(row_blocks)]
    number_elements = reg_shape[0]
    out_blocks = []
    for attribute_blocks in blocks:
        acum_data = 0
        total_data = np.concatenate(attribute_blocks)
        out_blocks.append([])
        for idx in range(row_blocks):
            if (idx + 1) > (row_blocks - 1) and outer_rows != 0:
                out_blocks[-1].append(
                    np.expand_dims(
                        total_data[(idx * number_elements - acum_data):
                                   (idx * number_elements + outer_rows -
                                    acum_data)],
                        axis=1))
                acum_data += outer_rows
            else:
                out_blocks[-1].append(
                    np.expand_dims(
                        total_data[(idx * number_elements - acum_data):
                                   ((idx + 1) * number_elements - acum_data)],
                        axis=1))
    if outer_rows == 0:
        out_blocks.append([np.array([[0]])
                           for _ in range(len(out_blocks[0]))])
    out_blocks = np.array(out_blocks)
    final_blocks_list = []
    for idx_col in range(len(final_blocks[0])):
        for idx in range(len(final_blocks)):
            final_blocks_list.append(np.hstack(
                out_blocks[idx_col *
                           number_columns:
                           (idx_col + 1) *
                           number_columns,
                           idx: (idx + 1)].squeeze()))
    for i in range(len(final_blocks)):
        for j in range(len(final_blocks[0])):
            if i == (row_blocks - 1) and j == (len(final_blocks[0]) - 1) \
                    and outer_rows != 0 \
                    and number_columns % reg_shape[1] != 0:
                final_blocks[j][i] = final_blocks_list[
                    (i * len(final_blocks[0])) + j].reshape(
                    outer_rows, number_columns)
            elif i == (row_blocks - 1) and outer_rows != 0:
                final_blocks[i][j] = final_blocks_list[
                    (i * len(final_blocks[0])) + j].reshape(
                    -1, number_columns)
            elif j == (len(final_blocks[0]) - 1) and \
                    number_columns % reg_shape[1] != 0:
                final_blocks[i][j] = final_blocks_list[
                    (i * len(final_blocks[0])) + j].reshape(
                    -1, number_columns % reg_shape[1])
            else:
                final_blocks[i][j] = final_blocks_list[
                    (i * len(final_blocks[0])) + j].reshape(
                    -1, number_columns)
    return final_blocks


def redistribute_synchronized_blocks(blocks, reg_shape,
                                     row_blocks,
                                     col_blocks,
                                     outer_columns=0,
                                     outer_rows=0):
    final_blocks = [[object() for _ in range(col_blocks)] for _
                    in range(row_blocks)]
    total_data = np.concatenate(blocks)
    number_elements = reg_shape[0] * reg_shape[1]
    n_cols = reg_shape[0] * outer_columns
    n_rows = outer_rows * reg_shape[1]
    accum_data = 0
    out_blocks = []
    for idx in range(len(blocks) - 1):
        if (idx + 1) % col_blocks == 0 and outer_columns != 0:
            out_blocks.append(total_data[accum_data:accum_data + n_cols])
            accum_data += n_cols
        elif (idx + 1) > ((row_blocks - 1) * col_blocks) \
                and outer_rows != 0:
            out_blocks.append(total_data[
                              accum_data:accum_data + n_rows])
            accum_data += n_rows
        else:
            out_blocks.append(total_data[accum_data:
                                         number_elements + accum_data])
            accum_data += len(out_blocks[idx])
    out_blocks.append(total_data[accum_data:])
    for i in range(row_blocks):
        for j in range(col_blocks):
            if i == (row_blocks - 1) and j == (col_blocks - 1) \
                    and outer_rows != 0 and outer_columns != 0:
                final_blocks[i][j] = out_blocks[(i * col_blocks) + j]. \
                    reshape(outer_rows, outer_columns)
            elif i == (row_blocks - 1) and outer_rows:
                final_blocks[i][j] = out_blocks[(i * col_blocks) + j]. \
                    reshape(-1, outer_rows)
            elif j == (col_blocks - 1) and outer_columns:
                final_blocks[i][j] = out_blocks[(i * col_blocks) + j]. \
                    reshape(-1, outer_columns)
            else:
                final_blocks[i][j] = out_blocks[(i * col_blocks) + j]. \
                    reshape(-1, reg_shape[1])
    return final_blocks


def redistribute_blocks_columns(blocks,
                                reg_shape,
                                row_blocks,
                                number_columns,
                                outer_rows=0):
    final_blocks = [[object() for _ in range(
        math.ceil(reg_shape[1]/number_columns))]
                    for _ in range(row_blocks)]
    attribute_block = [[object() for _ in range(
        len(blocks[0]))] for _ in range(len(blocks))]
    for idx, attribute in enumerate(blocks):
        redistribute_data(attribute, attribute_block[idx], reg_shape[0])
    attribute_block = np.array(attribute_block)
    for j in range(len(final_blocks[0])):
        for i in range(row_blocks):
            final_blocks[i][j] = concatenate_rows(attribute_block[
                                                  j * number_columns:
                                                  (j+1) *
                                                  number_columns, i],
                                                  i,
                                                  j,
                                                  reg_shape,
                                                  row_blocks,
                                                  len(final_blocks[0]),
                                                  outer_rows,
                                                  number_columns)
    return final_blocks


def redistribute_blocks_with_exception(blocks, num_buckets_reorder,
                                       reg_shape, row_blocks,
                                       col_blocks, outer_columns=0,
                                       outer_rows=0):
    final_blocks = [[object() for _ in range(col_blocks)] for _ in
                    range(row_blocks)]
    number_elements_per_block = reg_shape[0] * reg_shape[1]
    number_elements_out_col = 0
    number_elements_out_row = 0
    if outer_columns != 0 and outer_rows != 0:
        number_elements_out_col = reg_shape[0] * outer_columns
        number_elements_out_row = outer_rows * reg_shape[1]
    elif outer_columns != 0:
        number_elements_out_col = reg_shape[0] * outer_columns
    elif outer_rows != 0:
        number_elements_out_row = outer_rows * reg_shape[1]
    idx_loop = 0
    while True:
        try:
            with TaskGroup('terasort_group', False):
                for iteration in range(int(len(blocks) / 2)):
                    if iteration != 0:
                        for idx in range(int((len(blocks) - 1) /
                                             num_buckets_reorder)):
                            out_blocks = [object() for _ in range(len(
                                blocks[idx * num_buckets_reorder:
                                       (idx + 1) * num_buckets_reorder]))]
                            store_column_edge = np.array(range(
                                (idx * num_buckets_reorder + 2),
                                ((idx + 1) * num_buckets_reorder + 2))) % \
                                col_blocks
                            store_row_edge = np.array(range(
                                (idx * num_buckets_reorder + 1),
                                ((idx + 1) * num_buckets_reorder + 1))) < \
                                (col_blocks * (row_blocks-1))
                            last_element = np.any(np.array(
                                range((idx * num_buckets_reorder + 2),
                                      ((idx + 1) * num_buckets_reorder + 2)))
                                      == (col_blocks * row_blocks))
                            merge_distribute_data_bucket(
                                blocks[(idx * num_buckets_reorder) + 1:
                                       (((idx + 1) *
                                         num_buckets_reorder) + 1)],
                                out_blocks,
                                n_buckets=len(blocks[
                                              (idx * num_buckets_reorder) + 1:
                                              (((idx + 1) *
                                                num_buckets_reorder) + 1)]),
                                number_elements=number_elements_per_block,
                                n_cols=number_elements_out_col,
                                n_rows=number_elements_out_row,
                                store_column_edge=store_column_edge,
                                store_row_edge=store_row_edge,
                                last_element=last_element)
                            blocks[(idx * num_buckets_reorder) + 1:
                                   (((idx + 1) * num_buckets_reorder) + 1)] =\
                                out_blocks[:num_buckets_reorder]
                    checks = []
                    for idx in range(int(len(blocks) / num_buckets_reorder)):
                        out_blocks = [object() for _ in range(
                            len(blocks[idx * num_buckets_reorder:
                                       (idx + 1) * num_buckets_reorder]))]
                        store_column_edge = np.array(range(
                            (idx * num_buckets_reorder + 1),
                            ((idx + 1) * num_buckets_reorder + 1))) % \
                            col_blocks
                        store_row_edge = np.array(range(
                            (idx * num_buckets_reorder),
                            ((idx + 1) * num_buckets_reorder))) < \
                            (col_blocks * (row_blocks-1))
                        last_element = np.any(np.array(
                            range((idx * num_buckets_reorder + 1),
                                  ((idx + 1) * num_buckets_reorder + 1)))
                                              == (col_blocks * row_blocks))
                        checks.append(merge_distribute_data_bucket(
                            blocks[idx * num_buckets_reorder:
                                   (idx + 1) * num_buckets_reorder],
                            out_blocks,
                            n_buckets=num_buckets_reorder,
                            number_elements=number_elements_per_block,
                            n_cols=number_elements_out_col,
                            n_rows=number_elements_out_row,
                            store_column_edge=store_column_edge,
                            store_row_edge=store_row_edge,
                            last_element=last_element))
                        blocks[idx * num_buckets_reorder:
                               (idx + 1) * num_buckets_reorder] = \
                            out_blocks[:num_buckets_reorder]
                    if iteration == (int(len(blocks) / 2) - 1):
                        evaluate_exception(checks, blocks)
                    idx_loop = idx_loop + 1
                compss_barrier_group('terasort_group')
        except COMPSsException:
            break
    for i in range(row_blocks):
        for j in range(col_blocks):
            final_blocks[i][j] = adjust_final_block(
                blocks[i * col_blocks + j], reg_shape[1],
                i == (row_blocks - 1), j == (col_blocks - 1),
                outer_rows, outer_columns)
    return final_blocks


@task(attribute_blocks={Type: COLLECTION_IN, Depth: 1},
      outer_blocks={Type: COLLECTION_OUT, Depth: 1})
def redistribute_data(attribute_blocks, outer_blocks, number_rows):
    total_data = np.concatenate(attribute_blocks)
    for idx in range(len(outer_blocks)):
        outer_blocks[idx] = np.expand_dims(total_data[
                                           idx*number_rows:(idx+1) *
                                           number_rows], axis=1)


@task(blocks={Type: COLLECTION_IN, Depth: 1})
def concatenate_rows(blocks, i, j,
                     reg_shape, row_blocks,
                     col_blocks, outer_rows, number_columns):
    if i == (row_blocks - 1) and j == (col_blocks - 1) \
            and outer_rows != 0 \
            and number_columns % reg_shape[1] != 0:
        return np.hstack(blocks).reshape(outer_rows, number_columns)
    elif i == (row_blocks - 1) and outer_rows != 0:
        return np.hstack(blocks).reshape(-1, number_columns)
    elif j == (col_blocks - 1) and \
            number_columns % reg_shape[1] != 0:
        return np.hstack(blocks).reshape(-1, number_columns % reg_shape[1])
    else:
        return np.hstack(blocks).reshape(-1, number_columns)


@task(returns=1)
def adjust_final_block(block, cols, row, col, outer_rows, outer_cols):
    if row and col and outer_cols != 0 and outer_rows:
        return block.reshape(outer_rows, outer_cols)
    elif row and outer_rows != 0:
        return block.reshape(-1, outer_rows)
    elif col and outer_cols != 0:
        return block.reshape(-1, outer_cols)
    return block.reshape(-1, cols)


@task(buckets={Type: COLLECTION_IN, Depth: 1},
      out_blocks={Type: COLLECTION_OUT, Depth: 1}, returns=1)
def merge_distribute_data_bucket(buckets, out_blocks, n_buckets,
                                 number_elements,
                                 n_cols, n_rows,
                                 store_column_edge,
                                 store_row_edge, last_element=False):
    total_data = np.concatenate(buckets)
    acum_data = 0
    for idx in range(n_buckets - 1):
        if store_column_edge[idx] == 0 and n_cols != 0:
            out_blocks[idx] = total_data[acum_data: acum_data + n_cols]
            acum_data += n_cols
        elif store_row_edge[idx] == 0 and n_rows != 0:
            out_blocks[idx] = total_data[acum_data: acum_data + n_rows]
            acum_data += n_rows
        else:
            out_blocks[idx] = total_data[
                              (idx*number_elements - acum_data):
                              ((idx + 1)*number_elements - acum_data)]
            acum_data += len(out_blocks[idx])
    out_blocks[n_buckets - 1] = total_data[acum_data:]
    checks = []
    for idx, bucket in enumerate(out_blocks):
        if isinstance(bucket, np.ndarray):
            if store_column_edge[idx] == 0 and store_row_edge[idx] == 0 \
                    and n_cols != 0 and n_rows != 0:
                checks.append(True)
            elif store_column_edge[idx] == 0 and n_cols != 0:
                if len(bucket) == n_cols:
                    checks.append(True)
                else:
                    checks.append(False)
            elif store_row_edge[idx] == 0 and n_rows != 0:
                if len(bucket) == n_rows:
                    checks.append(True)
                else:
                    checks.append(False)
            elif idx == (len(out_blocks) - 1) and last_element \
                    and n_rows != 0 and n_cols != 0:
                if len(bucket) == int(n_cols * n_rows):
                    checks.append(True)
                else:
                    checks.append(False)
            elif len(bucket) != number_elements or len(bucket) == 0:
                checks.append(False)
            else:
                checks.append(True)
        else:
            checks.append(False)
    return np.all(checks)


@task(fragment={Type: COLLECTION_IN, Depth: 1},
      fragment_buckets={Type: COLLECTION_OUT, Depth: 1},
      range_min={Type: COLLECTION_IN, Depth: 2},
      range_max={Type: COLLECTION_IN, Depth: 2})
def filter_fragment(fragment, fragment_buckets, num_buckets, range_min=0,
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
                                range_max[0][0][0][0] + 1,
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


@task(returns=dict, args={Type: COLLECTION_IN, Depth: 1})
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
    return sorted_by_key


@task(blocks={Type: COLLECTION_IN, Depth: 2}, returns=1)
def obtain_minimum_value(blocks):
    minimum_values = np.block(blocks)
    return np.array([[np.amin(minimum_values)]])


@task(blocks={Type: COLLECTION_IN, Depth: 2}, returns=1)
def obtain_maximum_value(blocks):
    maximum_values = np.block(blocks)
    return np.array([[np.amax(maximum_values)]])
