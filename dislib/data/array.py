import numbers
from math import ceil

import numpy as np
from pycompss.api.api import compss_wait_on
from pycompss.api.parameter import COLLECTION_INOUT, COLLECTION_IN
from pycompss.api.parameter import Depth, Type
from pycompss.api.task import task
from scipy import sparse as sp
from scipy.sparse import issparse


def array(x, blocks_shape):
    """
    Loads data into a Distributed Array.

    Parameters
    ----------
    x : ndarray, shape=[n_samples, n_features]
        Array of samples.
    blocks_shape : (int, int)
        Block sizes in number of samples.

    Returns
    -------
    darray : Array
        A distributed representation of the data divided in blocks.
    """
    x_size, y_size = blocks_shape

    blocks = []
    for i in range(0, x.shape[0], x_size):
        row = [x[i: i + x_size, j: j + y_size] for j in
               range(0, x.shape[1], y_size)]
        blocks.append(row)

    sparse = issparse(x)
    darray = Array(blocks=blocks, blocks_shape=blocks_shape, shape=x.shape,
                   sparse=sparse)

    return darray


def random_array(shape, block_size, random_state=None):
    """
    Returns a distributed array of random floats in the open interval [0.0,
    1.0). Values are from the “continuous uniform” distribution over the
    stated interval.

    Parameters
    ----------
    shape : tuple of two ints
        Shape of the output ds-array.
    block_size : tuple of two ints
        Size of the ds-array blocks.
    random_state : int or RandomState, optional (default=None)
        Seed or numpy.random.RandomState instance to generate the random
        numbers.

    Returns
    -------
    dsarray : ds-array
        Distributed array of random floats.
    """
    if shape[0] < block_size[0] or shape[1] < block_size[1]:
        raise AttributeError("Block size is greater than the array")

    r_state = random_state

    if isinstance(r_state, (numbers.Integral, np.integer)):
        r_state = np.random.RandomState(r_state)

    seed = None
    blocks_shape = (int(np.ceil(shape[0] / block_size[0])),
                    int(np.ceil(shape[1] / block_size[1])))

    blocks = list()

    for row_idx in range(blocks_shape[0]):
        blocks.append(list())

        for col_idx in range(blocks_shape[1]):
            b_size0, b_size1 = block_size

            if row_idx == blocks_shape[0] - 1:
                b_size0 = shape[0] - (blocks_shape[0] - 1) * block_size[0]

            if col_idx == blocks_shape[1] - 1:
                b_size1 = shape[1] - (blocks_shape[1] - 1) * block_size[1]

            if r_state is not None:
                seed = r_state.randint(np.iinfo(np.int32).max)

            blocks[-1].append(_random_block((b_size0, b_size1), seed))

    return Array(blocks, block_size, shape, False)


def load_svmlight_file(path, blocks_shape, n_features, store_sparse):
    """ Loads a LibSVM file into a Distributed Array.

     Parameters
    ----------
    path : string
        File path.
    blocks_shape : (int, int)
        Block size for the output darray.
    n_features : int
        Number of features.
    store_sparse : boolean
        Whether to use scipy.sparse data structures to store data. If False,
        numpy.array is used instead.

    Returns
    -------
    x, y : (darray, darray)
        A distributed representation (darray) of the X and y.
    """
    n, m = blocks_shape
    lines = []
    x_blocks, y_blocks = [], []

    n_rows = 0
    with open(path, "r") as f:
        for line in f:
            n_rows += 1
            lines.append(line.encode())

            if len(lines) == n:
                # line 0 -> X, line 1 -> y
                out_blocks = Array._get_out_blocks(1, ceil(n_features / m))
                out_blocks.append([object()])
                # out_blocks.append([])
                _read_libsvm(lines, out_blocks, col_size=m,
                             n_features=n_features, store_sparse=store_sparse)
                # we append only the list forming the row (out_blocks depth=2)
                x_blocks.append(out_blocks[0])
                y_blocks.append(out_blocks[1])
                lines = []

    if lines:
        out_blocks = Array._get_out_blocks(1, ceil(n_features / m))
        out_blocks.append([object()])
        _read_libsvm(lines, out_blocks, col_size=m,
                     n_features=n_features, store_sparse=store_sparse)
        # we append only the list forming the row (out_blocks depth=2)
        x_blocks.append(out_blocks[0])
        y_blocks.append(out_blocks[1])

    x = Array(x_blocks, blocks_shape=blocks_shape, shape=(n_rows, n_features),
              sparse=store_sparse)

    # y has only a single line but it's treated as a 'column'
    y = Array(y_blocks, blocks_shape=(n, 1), shape=(n_rows, 1), sparse=False)

    return x, y


@task(out_blocks={Type: COLLECTION_INOUT, Depth: 2})
def _read_libsvm(lines, out_blocks, col_size, n_features, store_sparse):
    from tempfile import SpooledTemporaryFile
    from sklearn.datasets import load_svmlight_file

    # Creating a tmp file to use load_svmlight_file method should be more
    # efficient than parsing the lines manually
    tmp_file = SpooledTemporaryFile(mode="wb+", max_size=2e8)

    tmp_file.writelines(lines)

    tmp_file.seek(0)

    x, y = load_svmlight_file(tmp_file, n_features)
    if not store_sparse:
        x = x.toarray()

    # tried also converting to csc/ndarray first for faster splitting but it's
    # not worth. Position 0 contains the X
    for i in range(ceil(n_features / col_size)):
        out_blocks[0][i] = x[:, i * col_size:(i + 1) * col_size]

    # Position 1 contains the y block
    out_blocks[1][0] = y.reshape(-1, 1)


class Array(object):
    """ A dataset containing samples and, optionally, labels that can be
    # stored in a distributed manner.
    #
    # Dataset works as a list of Subset instances, which can be future objects
    # stored remotely. Accessing Dataset.labels and Dataset.samples runs
    # collect() and transfers all the data to the local machine.
    #
    # Parameters
    # ----------
    # blocks : list
    #     List of lists of numpy / scipy arrays
    # blocks_shape : tuple / list of tuples
    #     A single tuple indicates that all blocks are regular
    #     (except bottom ones, and right ones).
    #     If a list of 3 tuples is paased, the sizes correspond to top-left
    #     block, regular blocks, and bot-right block.
    # shape : int
    #     Total number of elements in the array.
    # sparse : boolean, optional (default=False)
    #     Whether this dataset uses sparse data structures.
    #
    # Attributes
    # ----------
    # _blocks : list
    #     List of lists of numpy / scipy arrays
    # _blocks_shape : tuple / list of tuples
    #     A single tuple indicates that all blocks are regular
    #     (except bottom ones, and right ones).
    #     If a list of 3 tuples is paased, the sizes correspond to top-left
    #     block, regular blocks, and bot-right block.
    # _number_of_blocks : tuple(int, int)
    #     Total number of (horizontal, vertical) blocks.
    # shape : int
    #     Total number of elements in the array.
    # _sparse: boolean
    #     True if this dataset uses sparse data structures.
    # """

    INVALID_BLOCK_SHAPE_ERROR = "Blocks shape must be: a tuple, with the " \
                                "shape of regular block; or a list with the" \
                                " shape of top-left block and regular blocks."

    def __init__(self, blocks, blocks_shape, shape, sparse):
        self._validate_blocks(blocks)

        self._blocks = blocks

        if isinstance(blocks_shape, list):
            # check that top-left is not larger than regular blocks
            if len(blocks_shape) != 2:
                raise Exception(self.INVALID_BLOCK_SHAPE_ERROR)
            (bi0, bj0), (bn, bm) = blocks_shape
            if bi0 > bn or bj0 > bm:
                raise Exception("Top-left block can not be larger than regular"
                                "blocks.")
            self._top_left_shape = (bi0, bj0)
            self._blocks_shape = (bn, bm)
        elif isinstance(blocks_shape, tuple):
            self._top_left_shape = None
            self._blocks_shape = blocks_shape
        else:
            raise Exception(self.INVALID_BLOCK_SHAPE_ERROR)
        self._number_of_blocks = (len(blocks), len(blocks[0]))
        self._shape = shape
        self._sparse = sparse

    @staticmethod
    def _validate_blocks(blocks):
        if len(blocks) == 0 or len(blocks[0]) == 0:
            raise AttributeError('Blocks must be a list of lists, with at '
                                 'least an empty numpy/scipy matrix.')
        row_length = len(blocks[0])
        for i in range(1, len(blocks)):
            if len(blocks[i]) != row_length:
                raise AttributeError(
                    'All rows must contain the same number of blocks.')

    @staticmethod
    def _merge_blocks(blocks):
        sparse = None
        b0 = blocks[0][0]
        if sparse is None:
            sparse = issparse(b0)

        if sparse:
            ret = sp.bmat(blocks, format=b0.getformat(), dtype=b0.dtype)
        else:
            ret = np.block(blocks)

        return ret

    @staticmethod
    def _get_out_blocks(x, y):
        """ Helper function that builds empty lists of lists to be filled
        as parameter of type COLLECTION_INOUT """
        return [[object() for _ in range(y)] for _ in range(x)]

    def _has_regular_blocks(self):
        return self._top_left_shape is None

    def _get_row_shape(self, row_idx):
        if row_idx == 0 and not self._has_regular_blocks():
            return self._top_left_shape[0], self.shape[1]

        if row_idx < self._number_of_blocks[0] - 1:
            return self._blocks_shape[0], self.shape[1]

        # this is the last chunk of rows, number of rows might be smaller
        if self._has_regular_blocks():
            n_r = self.shape[0] - (self._number_of_blocks[0] - 1) * \
                                  self._blocks_shape[0]
        else:  # is first block is irregular, n_r is computed differently
            n_r = self.shape[0] - self._top_left_shape[0] - \
                  (self._number_of_blocks[0] - 2) * self._blocks_shape[0]
        return n_r, self.shape[1]

    def _get_col_shape(self, col_idx):
        if col_idx == 0 and not self._has_regular_blocks():
            return self.shape[0], self._top_left_shape[1]

        if col_idx < self._number_of_blocks[1] - 1:
            return self.shape[0], self._blocks_shape[1]

        # this is the last chunk of cols, number of cols might be smaller
        if self._has_regular_blocks():
            n_c = self.shape[1] - (self._number_of_blocks[1] - 1) * \
                                  self._blocks_shape[1]
        else:  # is first block is irregular, n_r is computed differently
            n_c = self.shape[1] - self._top_left_shape[1] - \
                  (self._number_of_blocks[1] - 2) * self._blocks_shape[1]
        return self.shape[0], n_c

    def _iterator(self, axis=0):
        # iterate through rows
        if axis == 0 or axis == 'rows':
            for i, row in enumerate(self._blocks):
                row_shape = self._get_row_shape(i)
                yield Array(blocks=[row], blocks_shape=self._blocks_shape,
                            shape=row_shape, sparse=self._sparse)

        # iterate through columns
        elif axis == 1 or axis == 'columns':
            for j in range(self._number_of_blocks[1]):
                col_shape = self._get_col_shape(j)
                col_blocks = [[self._blocks[i][j]] for i in
                              range(self._number_of_blocks[0])]
                yield Array(blocks=col_blocks, blocks_shape=self._blocks_shape,
                            shape=col_shape, sparse=self._sparse)

        else:
            raise Exception(
                "Axis must be [0|'rows'] or [1|'columns']. Got: %s" % axis)

    def _get_containing_block(self, i, j):
        """ Returns the indices of the block containing coordinate (i, j) """
        if not self._has_regular_blocks():
            bi0, bj0 = self._top_left_shape
        else:
            bi0, bj0 = (0, 0)

        bn, bm = self._blocks_shape
        block_i = (i + bi0) // bn
        block_j = (j + bj0) // bm

        # if blocks are out of bounds, assume the element belongs to last block
        if block_i >= self._number_of_blocks[0]:
            block_i = self._number_of_blocks[0] - 1

        if block_j >= self._number_of_blocks[1]:
            block_j = self._number_of_blocks[1] - 1

        return block_i, block_j

    def _get_coordinates_in_block(self, block_i, block_j, i, j):
        local_i, local_j = i, j

        if block_i > 0:
            if self._has_regular_blocks():
                local_i = i - (block_i * self._blocks_shape[0])
            else:
                local_i = i - self._top_left_shape[0] - \
                          (block_i - 1) * self._blocks_shape[0]

        if block_j > 0:
            if self._has_regular_blocks():
                local_j = j - (block_j * self._blocks_shape[1])
            else:
                local_j = j - self._top_left_shape[1] - \
                          (block_j - 1) * self._blocks_shape[1]

        return local_i, local_j

    def _get_single_element(self, i, j):
        # we are returning a single element
        if i > self.shape[0] or j > self.shape[0]:
            raise IndexError("Shape is %s" % self.shape)

        bi, bj = self._get_containing_block(i, j)
        local_i, local_j = self._get_coordinates_in_block(bi, bj, i, j)
        block = self._blocks[bi][bj]

        element = _get_item(local_i, local_j, block)

        return Array(blocks=[[element]], blocks_shape=(1, 1), shape=(1, 1),
                     sparse=False)

    def _get_slice(self, rows, cols):

        if (rows.step is not None and rows.step > 1) or \
                (cols.step is not None and cols.step > 1):
            raise NotImplementedError("Variable steps not supported, contact"
                                      " the dislib team or open an issue "
                                      "in github.")

        # rows and cols are read-only
        # import ipdb
        # ipdb.set_trace()
        r_start, r_stop = rows.start, rows.stop
        c_start, c_stop = cols.start, cols.stop

        if r_start is None:
            r_start = 0
        if c_start is None:
            c_start = 0

        if r_stop is None or r_stop > self.shape[0]:
            r_stop = self.shape[0]
        if c_stop is None or c_stop > self.shape[1]:
            c_stop = self.shape[1]

        if r_start < 0 or r_stop < 0 or c_start < 0 or c_stop < 0:
            raise NotImplementedError("Negative indexes not supported, contact"
                                      " the dislib team or open an issue "
                                      "in github.")

        # get the coordinates of top-left and bot-right corners
        i_0, j_0 = self._get_containing_block(r_start, c_start)
        i_n, j_n = self._get_containing_block(r_stop, c_stop)

        # Number of blocks to be returned
        n_blocks = i_n - i_0 + 1
        m_blocks = j_n - j_0 + 1

        out_blocks = self._get_out_blocks(n_blocks, m_blocks)

        # import ipdb
        # ipdb.set_trace()
        i_indices = range(i_0, i_n + 1)
        j_indices = range(j_0, j_n + 1)
        for out_i, i in enumerate(i_indices):
            for out_j, j in enumerate(j_indices):

                top, left, bot, right = None, None, None, None
                if out_i == 0:
                    top, _ = self._get_coordinates_in_block(i_0, j_0,
                                                            r_start,
                                                            c_start)
                if out_i == len(i_indices) - 1:
                    bot, _ = self._get_coordinates_in_block(i_n, j_n,
                                                            r_stop,
                                                            c_stop)
                if out_j == 0:
                    _, left = self._get_coordinates_in_block(i_0, j_0,
                                                             r_start,
                                                             c_start)
                if out_j == len(j_indices) - 1:
                    _, right = self._get_coordinates_in_block(i_n, j_n,
                                                              r_stop,
                                                              c_stop)

                boundaries = (top, left, bot, right)
                try:
                    fb = _filter_block(block=self._blocks[i][j],
                                       boundaries=boundaries)
                except:
                    import ipdb
                    ipdb.set_trace()
                out_blocks[out_i][out_j] = fb
                print(fb)

        # import ipdb
        # ipdb.set_trace()

        # Shape of the top left block
        top, left = self._get_coordinates_in_block(0, 0, r_start,
                                                   c_start)
        bi0, bj0 = self._blocks_shape[0] - top, self._blocks_shape[1] - left

        # Regular blocks shape is the same
        bn, bm = self._blocks_shape

        # List of blocks shapes for initializer
        out_blocks_shapes = [(bi0, bj0), (bn, bm)]

        out_shape = r_stop - r_start, c_stop - c_start

        res = Array(blocks=out_blocks, blocks_shape=out_blocks_shapes,
                    shape=out_shape, sparse=self._sparse)
        return res

    def __getitem__(self, arg):
        rows, cols = arg  # unpack, assumes that we always pass in 2-arguments
        # for single indices, they will be integers, for slices, they'll be
        # slice objects here's a dummy implementation as a placeholder

        # TODO: parse/interpret the rows/cols parameters,
        if isinstance(rows, slice) or isinstance(cols, slice):
            return self._get_slice(rows, cols)

        else:
            i, j = rows, cols

            return self._get_single_element(i, j)

    @property
    def shape(self):
        return self._shape

    def transpose(self, mode='auto'):
        if mode == 'all':
            n, m = self._number_of_blocks[0], self._number_of_blocks[1]
            out_blocks = self._get_out_blocks(n, m)
            _transpose(self._blocks, out_blocks)
        elif mode == 'rows':
            out_blocks = []
            for r in self._iterator(axis=0):
                _blocks = self._get_out_blocks(*r._number_of_blocks)
                _tranpose(r._blocks, _blocks)

                out_blocks.append(_blocks[0])
        elif mode == 'columns':
            out_blocks = [[] for _ in range(self._number_of_blocks[0])]
            for i, c in enumerate(self._iterator(axis=1)):
                _blocks = self._get_out_blocks(*c._number_of_blocks)
                _tranpose(c._blocks, _blocks)

                for i2 in range(len(_blocks)):
                    out_blocks[i2].append(_blocks[i2][0])
        else:
            raise Exception(
                "Unknown transpose mode '%s'. Options are: [all|rows|columns]"
                % mode)

        blocks_t = list(map(list, zip(*out_blocks)))

        bn, bm = self._blocks_shape[0], self._blocks_shape[1]

        new_shape = self.shape[1], self.shape[0]
        # notice blocks_shape is transposed
        return Array(blocks_t, blocks_shape=(bm, bn), shape=new_shape,
                     sparse=self._sparse)

    def collect(self):
        self._blocks = compss_wait_on(self._blocks)

        res = self._merge_blocks(self._blocks)
        if not self._sparse:
            res = np.squeeze(res)
        return res


@task(returns=1)
def _get_item(i, j, block):
    return block[i][j]


# @task(out_blocks={Type: COLLECTION_INOUT, Depth: 2})
def _filter_block(block, boundaries):
    i_0, j_0, i_n, j_n = boundaries
    # if len(out_blocks) == 1 and len(out_blocks[0]) == 1:
    #     # we have a single block, filter it directly
    #     out_blocks[0][0] = out_blocks[0][0][i_0:i_n, j_0:j_n]
    res = block[i_0:i_n, j_0:j_n]

    return res


@task(returns=np.array)
def _random_block(shape, seed):
    if seed is not None:
        np.random.seed(seed)

    return np.random.random(shape)


@task(blocks={Type: COLLECTION_IN, Depth: 2},
      out_blocks={Type: COLLECTION_INOUT, Depth: 2})
def _transpose(blocks, out_blocks):
    for i in range(len(blocks)):
        for j in range(len(blocks[i])):
            out_blocks[i][j] = blocks[i][j].transpose()
