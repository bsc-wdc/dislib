import itertools
import numbers
from collections import defaultdict
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
    bn, bm = blocks_shape

    blocks = []
    for i in range(0, x.shape[0], bn):
        row = [x[i: i + bn, j: j + bm] for j in
               range(0, x.shape[1], bm)]
        blocks.append(row)

    sparse = issparse(x)
    darray = Array(blocks=blocks, blocks_shape=blocks_shape, shape=x.shape,
                   sparse=sparse)

    return darray


def random_array(shape, block_size, random_state=None):
    """
    Returns a distributed array of random floats in the open interval [0.0,
    1.0). Values are from the "continuous uniform" distribution over the
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
        raise ValueError("Block size is greater than the array")

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


def apply_along_axis(func, axis, x, *args, **kwargs):
    """ Apply a function to slices along the given axis.

    Execute func(a, *args, **kwargs) where func operates on nd-arrays and a
    is a slice of arr along axis. The size of the slices is determined
    by the blocks shape of x.

    func must meet the following conditions:

        - Take an nd-array as argument
        - Accept `axis` as a keyword argument
        - Return an array-like structure

    Parameters
    ----------
    func : function
        This function should accept nd-arrays and an axis argument. It is
        applied to slices of arr along the specified axis.
    axis : integer
        Axis along which arr is sliced. Can be 0 or 1.
    x : ds-array
        Input distributed array.
    args : any
        Additional arguments to func.
    kwargs : any
        Additional named arguments to func.

    Returns
    -------
    out : ds-array
        The output array. The shape of out is identical to the shape of arr,
        except along the axis dimension. The output ds-array is dense
        regardless of the type of the input array.

    Examples
    --------
    >>> import dislib as ds
    >>> import numpy as np
    >>> x = ds.random_array((100, 100), block_size=(25, 25))
    >>> mean = ds.apply_along_axis(np.mean, 0, x)
    >>> print(mean.collect())
    """
    if axis != 0 and axis != 1:
        raise ValueError("Axis must be 0 or 1.")

    bshape = x._blocks_shape
    shape = x.shape

    out_blocks = list()

    for block in x._iterator(axis=(not axis)):
        out = _block_apply(func, axis, block._blocks, *args, **kwargs)
        out_blocks.append(out)

    if axis == 0:
        blocks = [out_blocks]
        out_bshape = (1, bshape[1])
        out_shape = (1, shape[1])
    else:
        blocks = [[block] for block in out_blocks]
        out_bshape = (bshape[0], 1)
        out_shape = (shape[0], 1)

    return Array(blocks, blocks_shape=out_bshape, shape=out_shape,
                 sparse=False)


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
    # _n_blocks : tuple(int, int)
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
            self._top_left_shape = blocks_shape
            self._blocks_shape = blocks_shape
        else:
            raise Exception(self.INVALID_BLOCK_SHAPE_ERROR)
        self._n_blocks = (len(blocks), len(blocks[0]))
        self._shape = shape
        self._sparse = sparse

    @staticmethod
    def _validate_blocks(blocks):
        if len(blocks) == 0 or len(blocks[0]) == 0:
            raise AttributeError('Blocks must a list of lists, with at least'
                                 ' an empty numpy/scipy matrix.')
        row_length = len(blocks[0])
        for i in range(1, len(blocks)):
            if len(blocks[i]) != row_length:
                raise AttributeError(
                    'All rows must contain the same number of blocks.')

    @staticmethod
    def _merge_blocks(blocks):
        """
        Helper function that merges the _blocks attribute of a ds-array into
        a single ndarray / sparse matrix.
        """

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
        """
        Helper function that builds empty lists of lists to be filled as
        parameter of type COLLECTION_INOUT
        """
        return [[object() for _ in range(y)] for _ in range(x)]

    @staticmethod
    def _broadcast_shapes(x, y):
        if len(x) != 1 or len(y) != 1:
            raise IndexError("shape mismatch: indexing arrays could "
                             "not be broadcast together with shapes %s %s" %
                             (len(x), len(y)))

        return zip(*itertools.product(*[x, y]))

    def __str__(self):
        if self._top_left_shape is not None:
            bs = [self._top_left_shape, self._blocks_shape]
        else:
            bs = self._blocks_shape
        return "ds-array(blocks=(...), blocks_shape=%r, shape=%r, sparse=%r)" \
               % (bs, self.shape, self._sparse)

    def __repr__(self):
        if self._top_left_shape is not None:
            bs = [self._top_left_shape, self._blocks_shape]
        else:
            bs = self._blocks_shape
        return "ds-array(blocks=%r, blocks_shape=%r, shape=%r, sparse=%r)" % \
               (self._blocks, bs, self.shape, self._sparse)

    def _get_row_shape(self, row_idx):
        if row_idx == 0:
            return self._top_left_shape[0], self.shape[1]

        if row_idx < self._n_blocks[0] - 1:
            return self._blocks_shape[0], self.shape[1]

        # this is the last chunk of rows, number of rows might be smaller
        reg_blocks = self._n_blocks[0] - 2
        if reg_blocks < 0:
            reg_blocks = 0

        n_r = self.shape[0] - self._top_left_shape[0] - \
              reg_blocks * self._blocks_shape[0]
        return n_r, self.shape[1]

    def _get_col_shape(self, col_idx):
        if col_idx == 0:
            return self.shape[0], self._top_left_shape[1]

        if col_idx < self._n_blocks[1] - 1:
            return self.shape[0], self._blocks_shape[1]

        # this is the last chunk of cols, number of cols might be smaller
        reg_blocks = self._n_blocks[1] - 2
        if reg_blocks < 0:
            reg_blocks = 0
        n_c = self.shape[1] - self._top_left_shape[1] - \
              reg_blocks * self._blocks_shape[1]
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
            for j in range(self._n_blocks[1]):
                col_shape = self._get_col_shape(j)
                col_blocks = [[self._blocks[i][j]] for i in
                              range(self._n_blocks[0])]
                yield Array(blocks=col_blocks, blocks_shape=self._blocks_shape,
                            shape=col_shape, sparse=self._sparse)

        else:
            raise Exception(
                "Axis must be [0|'rows'] or [1|'columns']. Got: %s" % axis)

    def _get_containing_block(self, i, j):
        """
        Returns the indices of the block containing coordinate (i, j)
        """
        bi0, bj0 = self._top_left_shape
        bn, bm = self._blocks_shape

        # If first block is irregular, we need to add an offset to compute the
        # containing block indices
        offset_i, offset_j = bn - bi0, bm - bj0

        block_i = (i + offset_i) // bn
        block_j = (j + offset_j) // bm

        # if blocks are out of bounds, assume the element belongs to last block
        if block_i >= self._n_blocks[0]:
            block_i = self._n_blocks[0] - 1

        if block_j >= self._n_blocks[1]:
            block_j = self._n_blocks[1] - 1

        return block_i, block_j

    def _coords_in_block(self, block_i, block_j, i, j):
        """
        Return the conversion of the coords (i, j) in ds-array space to
        coordinates in the given block (block_i, block_j) space.
        """
        local_i, local_j = i, j

        if block_i > 0:
            reg_blocks = (block_i - 1) if (block_i - 1) >= 0 else 0
            local_i = i - self._top_left_shape[0] - \
                      reg_blocks * self._blocks_shape[0]

        if block_j > 0:
            reg_blocks = (block_j - 1) if (block_j - 1) >= 0 else 0
            local_j = j - self._top_left_shape[1] - \
                      reg_blocks * self._blocks_shape[1]

        return local_i, local_j

    def _get_single_element(self, i, j):
        """
        Return the element in (i, j) as a ds-array with a single element.
        """
        # we are returning a single element
        if i > self.shape[0] or j > self.shape[0]:
            raise IndexError("Shape is %s" % self.shape)

        bi, bj = self._get_containing_block(i, j)
        local_i, local_j = self._coords_in_block(bi, bj, i, j)
        block = self._blocks[bi][bj]

        # returns an list containing a single element
        element = _get_item(local_i, local_j, block)

        return Array(blocks=[[element]], blocks_shape=(1, 1), shape=(1, 1),
                     sparse=False)

    def _get_slice(self, rows, cols):
        """
         Returns a slice of the ds-array defined by the slices rows / cols.
         Only steps (as defined by slice.step) with value 1 can be used.
         """
        if (rows.step is not None and rows.step > 1) or \
                (cols.step is not None and cols.step > 1):
            raise NotImplementedError("Variable steps not supported, contact"
                                      " the dislib team or open an issue "
                                      "in github.")

        # rows and cols are read-only
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

        i_indices = range(i_0, i_n + 1)
        j_indices = range(j_0, j_n + 1)

        for out_i, i in enumerate(i_indices):
            for out_j, j in enumerate(j_indices):

                top, left, bot, right = None, None, None, None
                if out_i == 0:
                    top, _ = self._coords_in_block(i_0, j_0, r_start, c_start)
                if out_i == len(i_indices) - 1:
                    bot, _ = self._coords_in_block(i_n, j_n, r_stop, c_stop)
                if out_j == 0:
                    _, left = self._coords_in_block(i_0, j_0, r_start, c_start)
                if out_j == len(j_indices) - 1:
                    _, right = self._coords_in_block(i_n, j_n, r_stop, c_stop)

                boundaries = (top, left, bot, right)
                fb = _filter_block(block=self._blocks[i][j],
                                   boundaries=boundaries)
                out_blocks[out_i][out_j] = fb

        # Shape of the top left block
        top, left = self._coords_in_block(0, 0, r_start,
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

    def _get_by_lst_rows(self, rows):
        """
         Returns a slice of the ds-array defined by the lists of indices in
          rows.
         """

        # create dict where each key contains the adjusted row indices for that
        # block of rows
        adj_row_idxs = defaultdict(list)
        for row_idx in rows:
            containing_block = self._get_containing_block(row_idx, 0)[0]
            adj_idx = self._coords_in_block(containing_block, 0, row_idx, 0)[0]
            adj_row_idxs[containing_block].append(adj_idx)

        row_blocks = []
        for rowblock_idx, row in enumerate(self._iterator(axis='rows')):
            # create an empty list for the filtered row (single depth)
            rows_in_block = len(adj_row_idxs[rowblock_idx])
            # only launch the task if we are selecting rows from that block
            if rows_in_block > 0:
                row_block = _filter_row(blocks=row._blocks,
                                        rows=adj_row_idxs[rowblock_idx],
                                        cols=None)
                row_blocks.append((rows_in_block, [row_block]))

        # now we need to merge the rowblocks until they have as much rows as
        # self._blocks_shape[0] (i.e. number of rows per block)
        n_rows = 0
        to_merge = []
        final_blocks = []
        for rows_in_block, row in row_blocks:
            to_merge.append(row)
            n_rows += rows_in_block
            # enough rows to merge into a row_block
            if n_rows > self._blocks_shape[0]:
                out_blocks = [object() for _ in range(self._n_blocks[1])]
                new_rb = _merge_rows(to_merge, out_blocks, self._blocks_shape)
                final_blocks.append(new_rb)

        if n_rows > 0:
            out_blocks = [object() for _ in range(self._n_blocks[1])]
            _merge_rows(to_merge, out_blocks, self._blocks_shape)
            final_blocks.append(out_blocks)

        return Array(blocks=final_blocks, blocks_shape=self._blocks_shape,
                     shape=(len(rows), self._shape[1]), sparse=self._sparse)

    def _get_by_lst_cols(self, cols):
        """
         Returns a slice of the ds-array defined by the lists of indices in
          cols.
         """

        # create dict where each key contains the adjusted row indices for that
        # block of rows
        adj_col_idxs = defaultdict(list)
        for col_idx in cols:
            containing_block = self._get_containing_block(0, col_idx)[1]
            adj_idx = self._coords_in_block(0, containing_block, 0, col_idx)[1]
            adj_col_idxs[containing_block].append(adj_idx)

        col_blocks = []
        for colblock_idx, col in enumerate(self._iterator(axis='columns')):
            # create an empty list for the filtered row (single depth)
            cols_in_block = len(adj_col_idxs[colblock_idx])
            # only launch the task if we are selecting rows from that block
            if cols_in_block > 0:
                col_block = _filter_row(blocks=col._blocks,
                                        rows=None,
                                        cols=adj_col_idxs[colblock_idx])
                col_blocks.append((cols_in_block, col_block))

        # now we need to merge the rowblocks until they have as much rows as
        # self._blocks_shape[0] (i.e. number of rows per block)
        n_cols = 0
        to_merge = []
        final_blocks = []
        for cols_in_block, col in col_blocks:
            to_merge.append(col)
            n_cols += cols_in_block
            # enough cols to merge into a col_block
            if n_cols > self._blocks_shape[0]:
                out_blocks = [object() for _ in range(self._n_blocks[1])]
                new_rb = _merge_cols(to_merge, out_blocks, self._blocks_shape)
                final_blocks.append(new_rb)

        if n_cols > 0:
            out_blocks = [object() for _ in range(self._n_blocks[1])]
            _merge_cols(to_merge, out_blocks, self._blocks_shape)
            final_blocks.append(out_blocks)

        # list are in col-order transpose them for the correct ordering
        final_blocks = list(map(list, zip(*final_blocks)))

        return Array(blocks=final_blocks, blocks_shape=self._blocks_shape,
                     shape=(self._shape[0], len(cols)), sparse=self._sparse)

    def __getitem__(self, arg):

        # return a single row
        if isinstance(arg, int):
            return self._get_by_lst_rows(rows=[arg])

        # list of indices for rows
        elif isinstance(arg, list) or isinstance(arg, np.ndarray):
            return self._get_by_lst_rows(rows=arg)

        # slicing only rows
        elif isinstance(arg, slice):
            # slice only rows
            return self._get_slice(rows=arg, cols=slice(None, None))

        # we have indices for both dimensions
        if not isinstance(arg, tuple):
            raise IndexError("Invalid indexing information: %s" % arg)

        rows, cols = arg  # unpack 2-arguments

        # returning a single element
        if isinstance(rows, int) and isinstance(cols, int):
            return self._get_single_element(i=rows, j=cols)

        # all rows (slice : for rows) and list of indices for columns
        elif isinstance(rows, slice) and (
                isinstance(cols, list) or isinstance(cols, np.ndarray)):
            return self._get_by_lst_cols(cols=cols)

        # slicing both dimensions
        elif isinstance(rows, slice) and isinstance(cols, slice):
            return self._get_slice(rows, cols)

        raise IndexError("Invalid indexing information: %s" % str(arg))

        # elif isinstance(rows, list) or isinstance(rows, np.ndarray) or \
        #          isinstance(cols, list) or isinstance(cols, np.ndarray):
        #     return self._get_by_lst_idx(rows, cols)
        # for single indices, they will be integers, for slices, they'll be
        # slice objects here's a dummy implementation as a placeholder

        # if type(rows) != type(cols):
        #     raise Exception("Indexing method should be the same for both "
        #                     "dimensions. (%s != %s)" % (type(rows),
        #                     type(cols)))
        # if isinstance(rows, slice) or isinstance(cols, slice):

    @property
    def shape(self):
        """
        Total shape of the ds-array
        """
        return self._shape

    def transpose(self, mode='rows'):
        """
        Returns the transpose of the ds-array following the method indicated by
        mode. 'All' uses a single task to transpose all the blocks (slow with
        high number of blocks). 'rows' and 'columns' transpose each block of
        rows or columns independently (i.e. a task per row/col block).

        Parameters
        ----------
        mode : string, optional (default=rows)
            Array of samples.

        Returns
        -------
        darray : ds-array
            A transposed ds-array.
        """
        if mode == 'all':
            n, m = self._n_blocks[0], self._n_blocks[1]
            out_blocks = self._get_out_blocks(n, m)
            _transpose(self._blocks, out_blocks)
        elif mode == 'rows':
            out_blocks = []
            for r in self._iterator(axis=0):
                _blocks = self._get_out_blocks(*r._n_blocks)

                _transpose(r._blocks, _blocks)

                out_blocks.append(_blocks[0])
        elif mode == 'columns':
            out_blocks = [[] for _ in range(self._n_blocks[0])]
            for i, c in enumerate(self._iterator(axis=1)):
                _blocks = self._get_out_blocks(*c._n_blocks)

                _transpose(c._blocks, _blocks)

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

    def min(self, axis=0):
        """
        Returns the minimum along the given axis.

        Parameters
        ----------
        axis : int, optional (default=0)

        Returns
        -------
        min : ds-array
            Minimum along axis.
        """
        return apply_along_axis(np.min, axis, self)

    def max(self, axis=0):
        """
        Returns the maximum along the given axis.

        Parameters
        ----------
        axis : int, optional (default=0)

        Returns
        -------
        max : ds-array
            Maximum along axis.
        """
        return apply_along_axis(np.max, axis, self)

    def sum(self, axis=0):
        """
        Returns the sum along the given axis.

        Parameters
        ----------
        axis : int, optional (default=0)

        Returns
        -------
        sum : ds-array
            Sum along axis.
        """
        return apply_along_axis(np.sum, axis, self)

    def mean(self, axis=0):
        """
        Returns the mean along the given axis.

        Parameters
        ----------
        axis : int, optional (default=0)

        Returns
        -------
        mean : ds-array
            Mean along axis.
        """
        return apply_along_axis(np.mean, axis, self)

    def collect(self):
        self._blocks = compss_wait_on(self._blocks)
        res = self._merge_blocks(self._blocks)
        if not self._sparse:
            res = np.squeeze(res)
        return res


@task(returns=1)
def _get_item(i, j, block):
    """
    Returns a single item from the block. Coords must be in block space.
    """
    return block[i, j]


@task(returns=np.array)
def _random_block(shape, seed):
    if seed is not None:
        np.random.seed(seed)

    return np.random.random(shape)


@task(blocks={Type: COLLECTION_IN, Depth: 2}, returns=1)
def _filter_row(blocks, rows, cols):
    """
    Returns an array resulting of selecting rows:cols of the input   blocks
    """
    data = Array._merge_blocks(blocks)

    if issparse(blocks[0][0]):
        # sparse indexes element by element we need to do the cartesian
        # product of indices to get all coords
        rows, cols = zip(*itertools.product(*[rows, cols]))

    if rows is None:
        return data[:, cols]
    elif cols is None:
        return data[rows, :]

    return data[rows, cols]


@task(blocks={Type: COLLECTION_IN, Depth: 2},
      out_blocks={Type: COLLECTION_INOUT, Depth: 1})
def _merge_rows(blocks, out_blocks, blocks_shape):
    """
    Merges the blocks into a single list of blocks where each block has bn
    as number of rows (the number of cols remains the same per block).
    """
    bn, bm = blocks_shape
    data = Array._merge_blocks(blocks)

    for j in range(0, ceil(data.shape[1] / bm)):
        out_blocks[j] = data[:bn, j * bm: (j + 1) * bm]


@task(blocks={Type: COLLECTION_IN, Depth: 1},
      out_blocks={Type: COLLECTION_INOUT, Depth: 1})
def _merge_cols(blocks, out_blocks, blocks_shape):
    """
    Merges the blocks into a single list of blocks where each block has bn
    as number of rows (the number of cols remains the same per block).
    """
    bn, bm = blocks_shape
    data = Array._merge_blocks(blocks)

    for i in range(0, ceil(data.shape[0] / bn)):
        out_blocks[i] = data[i * bn: (i + 1) * bn, :bm]


@task(returns=1)
def _filter_block(block, boundaries):
    """
    Returns the slice of block defined by boundaries.
    Boundaries are the (x, y) coordinates of the top-left corner (i_0, j_0) and
    the bot-right one (i_n, j_n).
    """
    i_0, j_0, i_n, j_n = boundaries

    res = block[i_0:i_n, j_0:j_n]

    return res


@task(blocks={Type: COLLECTION_IN, Depth: 2},
      out_blocks={Type: COLLECTION_INOUT, Depth: 2})
def _transpose(blocks, out_blocks):
    for i in range(len(blocks)):
        for j in range(len(blocks[i])):
            out_blocks[i][j] = blocks[i][j].transpose()


@task(blocks={Type: COLLECTION_IN, Depth: 2}, returns=np.array)
def _block_apply(func, axis, blocks, *args, **kwargs):
    arr = Array._merge_blocks(blocks)
    kwargs['axis'] = axis
    out = func(arr, *args, **kwargs)

    if issparse(out):
        out = out.toarray()

    # We convert to array for consistency (otherwise the output of this
    # task is of unknown type)
    if axis == 0:
        return np.asarray(out).reshape(1, -1)
    else:
        return np.asarray(out).reshape(-1, 1)
