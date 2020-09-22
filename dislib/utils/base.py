import numpy as np
from pycompss.api.api import compss_delete_object
from pycompss.api.parameter import COLLECTION_OUT, Type, COLLECTION_IN, Depth
from pycompss.api.task import task
from scipy.sparse import issparse, vstack

from dislib.data.array import Array


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
        part_out_x_blocks = [{} for _ in range(x._n_blocks[1])]
        if y is None:
            _merge_shuffle_x(seed, part_out_subsamples, part_out_x_blocks,
                             x._reg_shape[1])
        else:
            part_out_y_blocks = [{} for _ in range(y._n_blocks[1])]
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

    subsamples = [{} for _ in range(n_parts_out)]
    seed = np.random.randint(np.iinfo(np.int32).max)
    if y is None:
        _choose_and_assign_rows_x(x._blocks, subsample_sizes, subsamples, seed)
    else:
        _choose_and_assign_rows_xy(x._blocks, y._blocks, subsample_sizes,
                                   subsamples, seed)
    return subsample_sizes, subsamples


@task(part_out_subsamples=COLLECTION_IN,
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


@task(part_out_subsamples=COLLECTION_IN,
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
