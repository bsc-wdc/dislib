import math

import numpy as np
from pycompss.api.api import compss_delete_object
from pycompss.api.constraint import constraint
from pycompss.api.parameter import COLLECTION_OUT, Type, COLLECTION_IN, Depth
from pycompss.api.task import task
from scipy.sparse import issparse, vstack
from sklearn.model_selection import ShuffleSplit

from dislib.data.array import Array


def train_test_split(x, y=None, test_size=None, train_size=None,
                     random_state=None):
    """ Randomly shuffles the rows of data.

    Parameters
    ----------
    x : ds-array
        Data to be splitted.
    y : ds-array, optional (default=None)
        Additional array to split using the same permutations, usually for
        labels or values. It is required that y.shape[0] == x.shape[0].
    test_size : float
        Number between 0 and 1 that defines the percentage of rows used as
        test data
    train_size : float
        Number between 0 and 1 that defines the percentage of rows used as
        train data
    random_state : int or RandomState, optional (default = None)
        Seed or numpy.random.RandomState instance to use in the generation
        of splits in the blocks.

    Returns
    -------
    train : ds-array
        A new ds-array containing the rows of x that correspond to train
        data.
    test : ds-array
        A new ds-array containing the rows of x that correspond to test
        data.
    train_y : ds-array, optional
        A new ds-array containing the rows of y that correspond to the
        rows in train.
    test_y : ds-array, optional
        A new ds-array containing the rows of y that correspond to the
        rows in test.
    """
    if test_size is None and train_size is None:
        train_size = 0.75
        test_size = 0.25
    elif test_size is None:
        test_size = 1 - train_size
    elif train_size is None:
        train_size = 1 - test_size
    if test_size > 1 or train_size > 1:
        raise ValueError("test_size and train_size arguments should be a "
                         "float between 0 and 1")
    if (test_size + train_size) > 1:
        raise ValueError("test_size and train_size should add up to one"
                         "as maximum value")
    if y is not None:
        if isinstance(x, Array) and isinstance(y, Array):
            return _make_splits(x=x, y=y, test_size=test_size,
                                train_size=train_size,
                                random_state=random_state)
        raise ValueError("The data to split should be contained "
                         "into dsarrays.")
    elif isinstance(x, Array):
        return _make_splits(x=x, test_size=test_size, train_size=train_size,
                            random_state=random_state)
    raise ValueError("The data to split should be contained "
                     "into dsarrays.")


def shuffle(x, y=None, random_state=None):
    """ Randomly shuffles the rows of data.

    Parameters
    ----------
    x : ds-array
        Data to be shuffled.
    y : ds-array, optional (default=None)
        Additional array to shuffle using the same permutation, usually for
        labels or values. It is required that y.shape[0] == x.shape[0].
    random_state : int or RandomState, optional (default=None)
        Seed or numpy.random.RandomState instance to use in the generation of
        random numbers.

    Returns
    -------
    x_shuffled : ds-array
        A new ds-array containing the rows of x shuffled.
    y_shuffled : ds-array, optional
        A new ds-array containing the rows of y shuffled using the same
        permutation. Only provided if y is not None.
    """
    if y is not None:
        assert y.shape[0] == x.shape[0]

    np.random.seed(random_state)
    block_n_rows = x._reg_shape[0]
    sizes_out = [block_n_rows for _ in range(x._shape[0] // block_n_rows)]
    remainder = x._shape[0] % block_n_rows
    if remainder != 0:
        sizes_out.append(remainder)

    # Matrix of subsets of rows (subsamples) going from part_in_i to part_out_j
    mapped_subsamples = []

    # For each part_in, get the parts going to each part_out
    if y is None:
        partition = x._iterator(axis=0)
    else:
        partition = _paired_partition(x, y)
    for part_in in partition:
        # part can be an array x_part or a tuple (x_part, y_part)
        part_sizes, part_in_subsamples = _partition_arrays(part_in, sizes_out)
        mapped_subsamples.append(part_in_subsamples)
        sizes_out -= part_sizes

    x_shuffled_blocks = []
    y_shuffled_blocks = []
    for j in range(len(sizes_out)):
        part_out_subsamples = [part_in_subsamples[j] for part_in_subsamples
                               in mapped_subsamples]
        seed = np.random.randint(np.iinfo(np.int32).max)
        # TODO: change object() for None when COMPSs version support it
        part_out_x_blocks = [object() for _ in range(x._n_blocks[1])]
        if y is None:
            _merge_shuffle_x(seed, part_out_subsamples, part_out_x_blocks,
                             x._reg_shape[1])
        else:
            part_out_y_blocks = [object() for _ in range(y._n_blocks[1])]
            _merge_shuffle_xy(seed, part_out_subsamples, part_out_x_blocks,
                              part_out_y_blocks, x._reg_shape[1],
                              y._reg_shape[1])
            y_shuffled_blocks.append(part_out_y_blocks)
        x_shuffled_blocks.append(part_out_x_blocks)

        # Clean parts to save disk space
        for part in part_out_subsamples:
            compss_delete_object(part)

    x_shuffled = Array(blocks=x_shuffled_blocks, top_left_shape=x._reg_shape,
                       reg_shape=x._reg_shape, shape=x.shape, sparse=x._sparse)
    if y is None:
        return x_shuffled
    else:
        y_shuffled = Array(blocks=y_shuffled_blocks,
                           top_left_shape=(x._reg_shape[0], y._reg_shape[1]),
                           reg_shape=(x._reg_shape[0], y._reg_shape[1]),
                           shape=y.shape, sparse=y._sparse)
        return x_shuffled, y_shuffled


def _partition_arrays(part_in, sizes_out):
    if isinstance(part_in, tuple):
        x = part_in[0]
        y = part_in[1]
    else:
        x = part_in
        y = None
    n_rows = x.shape[0]
    n_parts_out = len(sizes_out)
    subsample_sizes = np.zeros((n_parts_out,), dtype=int)
    for j in range(n_parts_out):
        if n_rows == 0:
            continue
        # Decide how many of the remaining rows of this part will go to
        # part_out_j. This is given by an hypergeometric distribution.
        n_good = sizes_out[j]
        n_bad = sum(sizes_out[j:]) - sizes_out[j]
        n_selected = np.random.hypergeometric(n_good, n_bad, n_rows)
        subsample_sizes[j] = n_selected
        n_rows -= n_selected

    subsamples = [object() for _ in range(n_parts_out)]
    seed = np.random.randint(np.iinfo(np.int32).max)
    if y is None:
        _choose_and_assign_rows_x(x._blocks, subsample_sizes, subsamples, seed)
    else:
        _choose_and_assign_rows_xy(x._blocks, y._blocks, subsample_sizes,
                                   subsamples, seed)
    return subsample_sizes, subsamples


def _make_splits(x, y=None, test_size=None, train_size=None,
                 random_state=None):
    if y:
        train_x_blocks_split = []
        test_x_blocks_split = []
        train_y_blocks_split = []
        test_y_blocks_split = []
        for index, blocks_x in enumerate(zip(x._blocks, y._blocks)):
            blocks_train_x = [object() for _ in range(x._n_blocks[1])]
            blocks_test_x = [object() for _ in range(x._n_blocks[1])]
            blocks_train_y = [object() for _ in range(y._n_blocks[1])]
            blocks_test_y = [object() for _ in range(y._n_blocks[1])]
            if index <= len(x._blocks) - 2:
                _compute_splits_x_y(blocks_x[0], blocks_x[1],
                                    blocks_train_x, blocks_test_x,
                                    blocks_train_y, blocks_test_y,
                                    test_size=test_size,
                                    train_size=train_size,
                                    random_state=random_state)
            elif index == len(x._blocks) - 1:
                if x.shape[0] % x._reg_shape[0] != 0:
                    _compute_splits_x_y(blocks_x[0], blocks_x[1],
                                        blocks_train_x, blocks_test_x,
                                        blocks_train_y, blocks_test_y,
                                        test_size=test_size,
                                        train_size=train_size,
                                        random_state=random_state)
                else:
                    _compute_splits_x_y(blocks_x[0], blocks_x[1],
                                        blocks_train_x, blocks_test_x,
                                        blocks_train_y, blocks_test_y,
                                        test_size=test_size,
                                        train_size=train_size,
                                        random_state=random_state)
            train_x_blocks_split.append(blocks_train_x)
            test_x_blocks_split.append(blocks_test_x)
            train_y_blocks_split.append(blocks_train_y)
            test_y_blocks_split.append(blocks_test_y)
        block_size_x = (math.floor(x._reg_shape[0] * train_size),
                        int(x._reg_shape[1]))
        block_size_test_x = (math.ceil(x._reg_shape[0] * test_size),
                             int(x._reg_shape[1]))
        top_train_shape_x = (math.floor(x._top_left_shape[0] * train_size),
                             int(x._top_left_shape[1]))
        top_test_shape_x = (math.ceil(x._top_left_shape[0] * test_size),
                            int(x._top_left_shape[1]))
        if x.shape[0] % x._reg_shape[0] != 0:
            shape_x = (block_size_x[0] * (len(train_x_blocks_split) - 1) +
                       math.floor((x.shape[0] % x._reg_shape[0]) * train_size),
                       int(x.shape[1]))
            shape_test_x = (math.ceil(block_size_test_x[0] *
                                      (len(test_x_blocks_split) - 1) +
                                      math.ceil((x.shape[0] % x._reg_shape[0])
                                                * test_size)),
                            int(x.shape[1]))
        else:
            shape_x = (
                block_size_x[0] * (len(train_x_blocks_split)),
                int(x.shape[1]))
            shape_test_x = (block_size_test_x[0] * (len(test_x_blocks_split)),
                            int(x.shape[1]))
        return Array(blocks=train_x_blocks_split,
                     top_left_shape=top_train_shape_x,
                     reg_shape=block_size_x, shape=shape_x,
                     sparse=False), \
            Array(blocks=test_x_blocks_split,
                  top_left_shape=top_test_shape_x,
                  reg_shape=block_size_test_x, shape=shape_test_x,
                  sparse=False), \
            Array(blocks=train_y_blocks_split,
                  top_left_shape=(top_train_shape_x[0], 1),
                  reg_shape=(block_size_x[0], 1), shape=(shape_x[0], 1),
                  sparse=False), \
            Array(blocks=test_y_blocks_split,
                  top_left_shape=(top_test_shape_x[0], 1),
                  reg_shape=(block_size_test_x[0], 1),
                  shape=(shape_test_x[0], 1),
                  sparse=False)
    train_x_blocks_split = []
    test_x_blocks_split = []
    for index, blocks_x in enumerate(x._blocks):
        blocks_train_x = [object() for _ in range(x._n_blocks[1])]
        blocks_test_x = [object() for _ in range(x._n_blocks[1])]
        if index <= len(x._blocks) - 2:
            _compute_splits_x(blocks_x, blocks_train_x,
                              blocks_test_x, test_size=test_size,
                              train_size=train_size,
                              random_state=random_state)
        elif index == len(x._blocks) - 1:
            if x.shape[0] % x._reg_shape[0] != 0:
                _compute_splits_x(blocks_x,
                                  blocks_train_x, blocks_test_x,
                                  test_size=test_size,
                                  train_size=train_size,
                                  random_state=random_state)
            else:
                _compute_splits_x(blocks_x, blocks_train_x,
                                  blocks_test_x, test_size=test_size,
                                  train_size=train_size,
                                  random_state=random_state)
        train_x_blocks_split.append(blocks_train_x)
        test_x_blocks_split.append(blocks_test_x)
    block_size_x = (int(x._reg_shape[0] * train_size),
                    int(x._reg_shape[1]))
    block_size_test_x = (int(x._reg_shape[0] * test_size),
                         int(x._reg_shape[1]))
    top_train_shape_x = (int(x._top_left_shape[0] * train_size),
                         int(x._top_left_shape[1]))
    top_test_shape_x = (int(x._top_left_shape[0] * test_size),
                        int(x._top_left_shape[1]))
    if x.shape[0] % x._reg_shape[0] != 0:
        shape_x = (block_size_x[0] * (len(train_x_blocks_split) - 1) +
                   math.floor((x.shape[0] % x._reg_shape[0]) * train_size),
                   int(x.shape[1]))
        shape_test_x = (math.ceil(block_size_test_x[0] *
                                  (len(test_x_blocks_split) - 1) +
                                  math.ceil((x.shape[0] % x._reg_shape[0]) *
                                            test_size)),
                        int(x.shape[1]))
    else:
        shape_x = (block_size_x[0] * (len(train_x_blocks_split)),
                   int(x.shape[1]))
        shape_test_x = (block_size_test_x[0] * (len(test_x_blocks_split)),
                        int(x.shape[1]))
    return Array(blocks=train_x_blocks_split,
                 top_left_shape=top_train_shape_x,
                 reg_shape=block_size_x, shape=shape_x, sparse=False), \
        Array(blocks=test_x_blocks_split, top_left_shape=top_test_shape_x,
              reg_shape=block_size_test_x, shape=shape_test_x,
              sparse=False)


# @task(returns=2)
def apply_splits_to_blocks(x, indexes_train, indexes_test):
    train_block = np.take(x, indexes_train, axis=0)
    test_block = np.take(x, indexes_test, axis=0)
    return train_block, test_block


@task(x={Type: COLLECTION_IN, Depth: 1},
      blocks_train_x={Type: COLLECTION_OUT, Depth: 1},
      blocks_test_x={Type: COLLECTION_OUT, Depth: 1})
def _compute_splits_x(x, blocks_train_x, blocks_test_x, test_size=None,
                      train_size=None, random_state=None):
    rs = ShuffleSplit(n_splits=1, train_size=train_size, test_size=test_size,
                      random_state=random_state)
    for train_index, test_index in rs.split(X=np.block(x[0])):
        for index, block in enumerate(x):
            train_block, test_block = apply_splits_to_blocks(block,
                                                             train_index,
                                                             test_index)
            blocks_train_x[index] = train_block
            blocks_test_x[index] = test_block


@task(x={Type: COLLECTION_IN, Depth: 1}, y={Type: COLLECTION_IN, Depth: 1},
      blocks_train_x={Type: COLLECTION_OUT, Depth: 1},
      blocks_test_x={Type: COLLECTION_OUT, Depth: 1},
      blocks_train_y={Type: COLLECTION_OUT, Depth: 1},
      blocks_test_y={Type: COLLECTION_OUT, Depth: 1})
def _compute_splits_x_y(x, y, blocks_train_x, blocks_test_x,
                        blocks_train_y, blocks_test_y, test_size=None,
                        train_size=None, random_state=None):
    rs = ShuffleSplit(n_splits=1, train_size=train_size, test_size=test_size,
                      random_state=random_state)
    for train_index, test_index in rs.split(X=np.block(x[0])):
        for index, block in enumerate(x):
            train_block, test_block = apply_splits_to_blocks(block,
                                                             train_index,
                                                             test_index)
            blocks_train_x[index] = train_block
            blocks_test_x[index] = test_block
        for index, block in enumerate(y):
            train_block_y, test_block_y = apply_splits_to_blocks(block,
                                                                 train_index,
                                                                 test_index)
            blocks_train_y[index] = train_block_y
            blocks_test_y[index] = test_block_y


@constraint(computing_units="${ComputingUnits}")
@task(part_out_subsamples={Type: COLLECTION_IN, Depth: 2},
      part_out_x_blocks=COLLECTION_OUT,
      returns=1)
def _merge_shuffle_x(seed, part_out_subsamples, part_out_x_blocks,
                     blocks_width):
    x_subsamples = part_out_subsamples
    if issparse(x_subsamples[0]):
        part_out_x = vstack(x_subsamples)
    else:
        part_out_x = np.concatenate(x_subsamples)

    np.random.seed(seed)
    p = np.random.permutation(part_out_x.shape[0])
    part_out_x = part_out_x[p]
    blocks = [part_out_x[:, i:i + blocks_width]
              for i in range(0, part_out_x.shape[1], blocks_width)]
    part_out_x_blocks[:] = blocks


@constraint(computing_units="${ComputingUnits}")
@task(part_out_subsamples={Type: COLLECTION_IN, Depth: 2},
      part_out_x_blocks=COLLECTION_OUT,
      part_out_y_blocks=COLLECTION_OUT,
      returns=1)
def _merge_shuffle_xy(seed, part_out_subsamples, part_out_x_blocks,
                      part_out_y_blocks, x_blocks_width, y_blocks_width):
    x_subsamples, y_subsamples = zip(*part_out_subsamples)

    if issparse(x_subsamples[0]):
        part_out_x = vstack(x_subsamples)
    else:
        part_out_x = np.concatenate(x_subsamples)

    if issparse(y_subsamples[0]):
        part_out_y = vstack(y_subsamples)
    else:
        part_out_y = np.concatenate(y_subsamples)

    np.random.seed(seed)
    p = np.random.permutation(part_out_x.shape[0])
    part_out_x = part_out_x[p]
    part_out_y = part_out_y[p]

    blocks_x = [part_out_x[:, i:i + x_blocks_width]
                for i in range(0, part_out_x.shape[1], x_blocks_width)]
    part_out_x_blocks[:] = blocks_x

    blocks_y = [part_out_y[:, i:i + y_blocks_width]
                for i in range(0, part_out_y.shape[1], y_blocks_width)]
    part_out_y_blocks[:] = blocks_y


@constraint(computing_units="${ComputingUnits}")
@task(x={Type: COLLECTION_IN, Depth: 2}, subsamples=COLLECTION_OUT)
def _choose_and_assign_rows_x(x, subsamples_sizes, subsamples, seed):
    np.random.seed(seed)
    x = Array._merge_blocks(x)
    indices = np.random.permutation(x.shape[0])
    start = 0
    for i, size in enumerate(subsamples_sizes):
        end = start + size
        subsamples[i] = x[indices[start:end]]
        start = end


@constraint(computing_units="${ComputingUnits}")
@task(x={Type: COLLECTION_IN, Depth: 2}, y={Type: COLLECTION_IN, Depth: 2},
      subsamples=COLLECTION_OUT)
def _choose_and_assign_rows_xy(x, y, subsamples_sizes, subsamples, seed):
    np.random.seed(seed)
    x = Array._merge_blocks(x)
    y = Array._merge_blocks(y)
    indices = np.random.permutation(x.shape[0])
    start = 0
    for i, size in enumerate(subsamples_sizes):
        end = start + size
        subsamples[i] = (x[indices[start:end]], y[indices[start:end]])
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
