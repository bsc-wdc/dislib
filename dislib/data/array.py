import itertools
from collections import defaultdict
from math import ceil

import numpy as np
from pycompss.api.api import compss_wait_on
from pycompss.api.parameter import Type, COLLECTION_IN, Depth, COLLECTION_INOUT
from pycompss.api.task import task
from scipy import sparse as sp
from scipy.sparse import issparse, csr_matrix
from sklearn.utils import check_random_state


class Array(object):
    """ A distributed 2-dimensional array divided in blocks.

    Normally, this class should not be instantiated directly, but created
    using one of the array creation routines provided.

    Apart from the different methods provided, this class also supports
    the following types of indexing:

        - ``A[i]`` : returns a single row
        - ``A[i, j]`` : returns a single element
        - ``A[i:j]`` : returns a set of rows (with ``i`` and ``j`` optional)
        - ``A[:, i:j]`` : returns a set of columns (with ``i`` and ``j``
          optional)
        - ``A[[i,j,k]]`` : returns a set of non-consecutive rows
        - ``A[:, [i,j,k]]`` : returns a set of non-consecutive columns
        - ``A[i:j, k:m]`` : returns a set of elements (with ``i``, ``j``,
          ``k``, and ``m`` optional)

    Parameters
    ----------
    blocks : list
        List of lists of nd-array or spmatrix.
    top_left_shape : tuple
        A single tuple indicating the shape of the top-left block.
    reg_shape : tuple
        A single tuple indicating the shape of the regular block.
    shape : tuple (int, int)
        Total number of elements in the array.
    sparse : boolean, optional (default=False)
        Whether this array stores sparse data.

    Attributes
    ----------
    shape : tuple (int, int)
        Total number of elements in the array.
    _blocks : list
        List of lists of nd-array or spmatrix.
    _top_left_shape : tuple
        A single tuple indicating the shape of the top-left block. This
        can be different from _reg_shape when slicing arrays.
    _reg_shape : tuple
        A single tuple indicating the shape of regular blocks. Top-left and
        and bot-right blocks might have different shapes (and thus, also the
        whole first/last blocks of rows/cols).
    _n_blocks : tuple (int, int)
        Total number of (horizontal, vertical) blocks.
    _sparse: boolean
        True if this array contains sparse data.
    """

    def __init__(self, blocks, top_left_shape, reg_shape, shape, sparse):
        self._validate_blocks(blocks)

        self._blocks = blocks
        self._top_left_shape = top_left_shape
        self._reg_shape = reg_shape

        self._n_blocks = (len(blocks), len(blocks[0]))
        self._shape = shape
        self._sparse = sparse

    def __str__(self):
        return "ds-array(blocks=(...), top_left_shape=%r, reg_shape=%r, " \
               "shape=%r, sparse=%r)" % (
                   self._top_left_shape, self._reg_shape, self.shape,
                   self._sparse)

    def __repr__(self):
        return "ds-array(blocks=(...), top_left_shape=%r, reg_shape=%r, " \
               "shape=%r, sparse=%r)" % (
                   self._top_left_shape, self._reg_shape, self.shape,
                   self._sparse)

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
        elif isinstance(rows, slice) and \
                (isinstance(cols, list) or isinstance(cols, np.ndarray)):
            return self._get_by_lst_cols(cols=cols)

        # slicing both dimensions
        elif isinstance(rows, slice) and isinstance(cols, slice):
            return self._get_slice(rows, cols)

        raise IndexError("Invalid indexing information: %s" % str(arg))

    @property
    def shape(self):
        """
        Total shape of the ds-array
        """
        return self._shape

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
    def _get_out_blocks(n_blocks):
        """
        Helper function that builds empty lists of lists to be filled as
        parameter of type COLLECTION_INOUT
        """
        return [[object() for _ in range(n_blocks[1])]
                for _ in range(n_blocks[0])]

    @staticmethod
    def _broadcast_shapes(x, y):
        if len(x) != 1 or len(y) != 1:
            raise IndexError("shape mismatch: indexing arrays could "
                             "not be broadcast together with shapes %s %s" %
                             (len(x), len(y)))

        return zip(*itertools.product(*[x, y]))

    def _get_row_shape(self, row_idx):
        if row_idx == 0:
            return self._top_left_shape[0], self.shape[1]

        if row_idx < self._n_blocks[0] - 1:
            return self._reg_shape[0], self.shape[1]

        # this is the last chunk of rows, number of rows might be smaller
        reg_blocks = self._n_blocks[0] - 2
        if reg_blocks < 0:
            reg_blocks = 0

        n_r = \
            self.shape[0] - self._top_left_shape[0] - reg_blocks * \
            self._reg_shape[0]
        return n_r, self.shape[1]

    def _get_col_shape(self, col_idx):
        if col_idx == 0:
            return self.shape[0], self._top_left_shape[1]

        if col_idx < self._n_blocks[1] - 1:
            return self.shape[0], self._reg_shape[1]

        # this is the last chunk of cols, number of cols might be smaller
        reg_blocks = self._n_blocks[1] - 2
        if reg_blocks < 0:
            reg_blocks = 0
        n_c = \
            self.shape[1] - self._top_left_shape[1] - \
            reg_blocks * self._reg_shape[1]
        return self.shape[0], n_c

    def _iterator(self, axis=0):
        # iterate through rows
        if axis == 0 or axis == 'rows':
            for i, row in enumerate(self._blocks):
                row_shape = self._get_row_shape(i)
                yield Array(blocks=[row], top_left_shape=self._top_left_shape,
                            reg_shape=self._reg_shape, shape=row_shape,
                            sparse=self._sparse)

        # iterate through columns
        elif axis == 1 or axis == 'columns':
            for j in range(self._n_blocks[1]):
                col_shape = self._get_col_shape(j)
                col_blocks = [[self._blocks[i][j]] for i in
                              range(self._n_blocks[0])]
                yield Array(blocks=col_blocks,
                            top_left_shape=self._top_left_shape,
                            reg_shape=self._reg_shape,
                            shape=col_shape, sparse=self._sparse)

        else:
            raise Exception(
                "Axis must be [0|'rows'] or [1|'columns']. Got: %s" % axis)

    def _get_containing_block(self, i, j):
        """
        Returns the indices of the block containing coordinate (i, j)
        """
        bi0, bj0 = self._top_left_shape
        bn, bm = self._reg_shape

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
            local_i = \
                i - self._top_left_shape[0] - \
                reg_blocks * self._reg_shape[0]

        if block_j > 0:
            reg_blocks = (block_j - 1) if (block_j - 1) >= 0 else 0
            local_j = \
                j - self._top_left_shape[1] - \
                reg_blocks * self._reg_shape[1]

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

        return Array(blocks=[[element]], top_left_shape=(1, 1),
                     reg_shape=(1, 1), shape=(1, 1), sparse=False)

    def _get_slice(self, rows, cols):
        """
         Returns a slice of the ds-array defined by the slices rows / cols.
         Only steps (as defined by slice.step) with value 1 can be used.
         """
        if (rows.step is not None and rows.step != 1) or \
                (cols.step is not None and cols.step != 1):
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

        n_rows = r_stop - r_start
        n_cols = c_stop - c_start

        # If the slice is empty (no rows or no columns), return a ds-array with
        # a single empty block. This empty block is required by the Array
        # constructor.
        if n_rows <= 0 or n_cols <= 0:
            n_rows = max(0, n_rows)
            n_cols = max(0, n_cols)
            if self._sparse:
                empty_block = csr_matrix((0, 0))
            else:
                empty_block = np.empty((0, 0))
            res = Array(blocks=[[empty_block]], top_left_shape=self._reg_shape,
                        reg_shape=self._reg_shape, shape=(n_rows, n_cols),
                        sparse=self._sparse)
            return res

        # get the coordinates of top-left and bot-right corners
        i_0, j_0 = self._get_containing_block(r_start, c_start)
        i_n, j_n = self._get_containing_block(r_stop - 1, c_stop - 1)

        # Number of blocks to be returned
        n_blocks = i_n - i_0 + 1
        m_blocks = j_n - j_0 + 1

        out_blocks = self._get_out_blocks((n_blocks, m_blocks))

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
        top, left = self._coords_in_block(0, 0, r_start, c_start)

        bi0 = self._reg_shape[0] - (top % self._reg_shape[0])
        bj0 = self._reg_shape[1] - (left % self._reg_shape[1])

        # Regular blocks shape is the same
        bn, bm = self._reg_shape

        out_shape = n_rows, n_cols

        res = Array(blocks=out_blocks, top_left_shape=(bi0, bj0),
                    reg_shape=(bn, bm), shape=out_shape, sparse=self._sparse)
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
                row_block = _filter_rows(blocks=row._blocks,
                                         rows=adj_row_idxs[rowblock_idx])
                row_blocks.append((rows_in_block, [row_block]))

        # now we need to merge the rowblocks until they have as much rows as
        # self._reg_shape[0] (i.e. number of rows per block)
        n_rows = 0
        to_merge = []
        final_blocks = []
        skip = 0

        for rows_in_block, row in row_blocks:
            to_merge.append(row)
            n_rows += rows_in_block
            # enough rows to merge into a row_block
            if n_rows >= self._reg_shape[0]:
                out_blocks = [object() for _ in range(self._n_blocks[1])]
                _merge_rows(to_merge, out_blocks, self._reg_shape, skip)
                final_blocks.append(out_blocks)

                # if we didn't take all rows, we keep the last block and
                # remember to skip the rows that have been merged
                if n_rows > self._reg_shape[0]:
                    to_merge = [row]
                    n_rows = n_rows - self._reg_shape[0]
                    skip = rows_in_block - n_rows
                else:
                    to_merge = []
                    n_rows = 0
                    skip = 0

        if n_rows > 0:
            out_blocks = [object() for _ in range(self._n_blocks[1])]
            _merge_rows(to_merge, out_blocks, self._reg_shape, skip)
            final_blocks.append(out_blocks)

        return Array(blocks=final_blocks, top_left_shape=self._top_left_shape,
                     reg_shape=self._reg_shape,
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
                col_block = _filter_cols(blocks=col._blocks,
                                         cols=adj_col_idxs[colblock_idx])
                col_blocks.append((cols_in_block, col_block))

        # now we need to merge the rowblocks until they have as much rows as
        # self._reg_shape[0] (i.e. number of rows per block)
        n_cols = 0
        to_merge = []
        final_blocks = []
        skip = 0

        for cols_in_block, col in col_blocks:
            to_merge.append(col)
            n_cols += cols_in_block
            # enough cols to merge into a col_block
            if n_cols >= self._reg_shape[0]:
                out_blocks = [object() for _ in range(self._n_blocks[1])]
                _merge_cols([to_merge], out_blocks, self._reg_shape, skip)
                final_blocks.append(out_blocks)

                # if we didn't take all cols, we keep the last block and
                # remember to skip the cols that have been merged
                if n_cols > self._reg_shape[0]:
                    to_merge = [col]
                    n_cols = n_cols - self._reg_shape[0]
                    skip = cols_in_block - n_cols
                else:
                    to_merge = []
                    n_cols = 0
                    skip = 0

        if n_cols > 0:
            out_blocks = [object() for _ in range(self._n_blocks[1])]
            _merge_cols([to_merge], out_blocks, self._reg_shape, skip)
            final_blocks.append(out_blocks)

        # list are in col-order transpose them for the correct ordering
        final_blocks = list(map(list, zip(*final_blocks)))

        return Array(blocks=final_blocks, top_left_shape=self._top_left_shape,
                     reg_shape=self._reg_shape,
                     shape=(self._shape[0], len(cols)), sparse=self._sparse)

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
        dsarray : ds-array
            A transposed ds-array.
        """
        if mode == 'all':
            n, m = self._n_blocks[0], self._n_blocks[1]
            out_blocks = self._get_out_blocks((n, m))
            _transpose(self._blocks, out_blocks)
        elif mode == 'rows':
            out_blocks = []
            for r in self._iterator(axis=0):
                _blocks = self._get_out_blocks(r._n_blocks)

                _transpose(r._blocks, _blocks)

                out_blocks.append(_blocks[0])
        elif mode == 'columns':
            out_blocks = [[] for _ in range(self._n_blocks[0])]
            for i, c in enumerate(self._iterator(axis=1)):
                _blocks = self._get_out_blocks(c._n_blocks)

                _transpose(c._blocks, _blocks)

                for i2 in range(len(_blocks)):
                    out_blocks[i2].append(_blocks[i2][0])
        else:
            raise Exception(
                "Unknown transpose mode '%s'. Options are: [all|rows|columns]"
                % mode)

        blocks_t = list(map(list, zip(*out_blocks)))

        bi0, bj0 = self._top_left_shape[0], self._top_left_shape[1]
        bn, bm = self._reg_shape[0], self._reg_shape[1]

        new_shape = self.shape[1], self.shape[0]
        # notice blocks shapes are transposed
        return Array(blocks_t, top_left_shape=(bj0, bi0), reg_shape=(bm, bn),
                     shape=new_shape, sparse=self._sparse)

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
        """
        Collects the contents of this ds-array and returns the equivalent
        in-memory array that this ds-array represents. This method creates a
        synchronization point in the execution of the application.

        Warning: This method may fail if the ds-array does not fit in
        memory.

        Returns
        -------
        array : nd-array or spmatrix
            The actual contents of the ds-array.
        """
        self._blocks = compss_wait_on(self._blocks)
        res = self._merge_blocks(self._blocks)
        if not self._sparse:
            res = np.squeeze(res)
        return res


def array(x, block_size):
    """
    Loads data into a Distributed Array.

    Parameters
    ----------
    x : spmatrix or array-like, shape=(n_samples, n_features)
        Array of samples.
    block_size : (int, int)
        Block sizes in number of samples.

    Returns
    -------
    dsarray : ds-array
        A distributed representation of the data divided in blocks.
    """
    sparse = issparse(x)

    if sparse:
        x = csr_matrix(x, copy=True)
    else:
        x = np.array(x, copy=True)

    if len(x.shape) < 2:
        raise ValueError("Input array must have two dimensions.")

    bn, bm = block_size

    blocks = []
    for i in range(0, x.shape[0], bn):
        row = [x[i: i + bn, j: j + bm] for j in range(0, x.shape[1], bm)]
        blocks.append(row)

    sparse = issparse(x)
    arr = Array(blocks=blocks, top_left_shape=block_size,
                reg_shape=block_size, shape=x.shape, sparse=sparse)

    return arr


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

    r_state = check_random_state(random_state)

    n_blocks = (int(np.ceil(shape[0] / block_size[0])),
                int(np.ceil(shape[1] / block_size[1])))

    blocks = list()

    for row_idx in range(n_blocks[0]):
        blocks.append(list())

        for col_idx in range(n_blocks[1]):
            b_size0, b_size1 = block_size

            if row_idx == n_blocks[0] - 1:
                b_size0 = shape[0] - (n_blocks[0] - 1) * block_size[0]

            if col_idx == n_blocks[1] - 1:
                b_size1 = shape[1] - (n_blocks[1] - 1) * block_size[1]

            seed = r_state.randint(np.iinfo(np.int32).max)
            blocks[-1].append(_random_block((b_size0, b_size1), seed))

    return Array(blocks, top_left_shape=block_size, reg_shape=block_size,
                 shape=shape, sparse=False)


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

    tlshape = x._top_left_shape
    bshape = x._reg_shape
    shape = x.shape

    out_blocks = list()

    for block in x._iterator(axis=(not axis)):
        out = _block_apply(func, axis, block._blocks, *args, **kwargs)
        out_blocks.append(out)

    if axis == 0:
        blocks = [out_blocks]
        out_tlbshape = (1, tlshape[1])
        out_bshape = (1, bshape[1])
        out_shape = (1, shape[1])
    else:
        blocks = [[block] for block in out_blocks]
        out_tlbshape = (tlshape[0], 1)
        out_bshape = (bshape[0], 1)
        out_shape = (shape[0], 1)

    return Array(blocks, top_left_shape=out_tlbshape, reg_shape=out_bshape,
                 shape=out_shape, sparse=False)


def load_svmlight_file(path, block_size, n_features, store_sparse):
    """ Loads a SVMLight file into a distributed array.

    Parameters
    ----------
    path : string
        File path.
    block_size : tuple (int, int)
        Size of the blocks for the output ds-array.
    n_features : int
        Number of features.
    store_sparse : boolean
        Whether to use scipy.sparse data structures to store data. If False,
        numpy.array is used instead.

    Returns
    -------
    x, y : (ds-array, ds-array)
        A distributed representation (ds-array) of the X and y.
    """
    n, m = block_size
    lines = []
    x_blocks, y_blocks = [], []

    n_rows = 0
    with open(path, "r") as f:
        for line in f:
            n_rows += 1
            lines.append(line.encode())

            if len(lines) == n:
                # line 0 -> X, line 1 -> y
                out_blocks = Array._get_out_blocks((1, ceil(n_features / m)))
                out_blocks.append([object()])
                # out_blocks.append([])
                _read_svmlight(lines, out_blocks, col_size=m,
                               n_features=n_features,
                               store_sparse=store_sparse)
                # we append only the list forming the row (out_blocks depth=2)
                x_blocks.append(out_blocks[0])
                y_blocks.append(out_blocks[1])
                lines = []

    if lines:
        out_blocks = Array._get_out_blocks((1, ceil(n_features / m)))
        out_blocks.append([object()])
        _read_svmlight(lines, out_blocks, col_size=m,
                       n_features=n_features, store_sparse=store_sparse)
        # we append only the list forming the row (out_blocks depth=2)
        x_blocks.append(out_blocks[0])
        y_blocks.append(out_blocks[1])

    x = Array(x_blocks, top_left_shape=block_size, reg_shape=block_size,
              shape=(n_rows, n_features), sparse=store_sparse)

    # y has only a single line but it's treated as a 'column'
    y = Array(y_blocks, top_left_shape=(n, 1), reg_shape=(n, 1),
              shape=(n_rows, 1), sparse=False)

    return x, y


def load_txt_file(path, block_size, delimiter=","):
    """ Loads a text file into a distributed array.

    Parameters
    ----------
    path : string
        File path.
    block_size : tuple (int, int)
        Size of the blocks of the array.
    delimiter : string, optional (default=",")
        String that separates columns in the file.

    Returns
    -------
    x : ds-array
        A distributed representation of the data divided in blocks.
    """

    with open(path, "r") as f:
        first_line = f.readline().strip()
        n_cols = len(first_line.split(delimiter))

    n_blocks = ceil(n_cols / block_size[1])
    blocks = []
    lines = []
    n_lines = 0

    with open(path, "r") as f:
        for line in f:
            n_lines += 1
            lines.append(line.encode())

            if len(lines) == block_size[0]:
                out_blocks = [object() for _ in range(n_blocks)]
                _read_lines(lines, block_size[1], delimiter, out_blocks)
                blocks.append(out_blocks)
                lines = []

    if lines:
        out_blocks = [object() for _ in range(n_blocks)]
        _read_lines(lines, block_size[1], delimiter, out_blocks)
        blocks.append(out_blocks)

    return Array(blocks, top_left_shape=block_size, reg_shape=block_size,
                 shape=(n_lines, n_cols), sparse=False)


@task(out_blocks=COLLECTION_INOUT, returns=1)
def _read_lines(lines, block_size, delimiter, out_blocks):
    samples = np.genfromtxt(lines, delimiter=delimiter)

    for i, j in enumerate(range(0, samples.shape[1], block_size)):
        out_blocks[i] = samples[:, j:j + block_size]


@task(out_blocks={Type: COLLECTION_INOUT, Depth: 2})
def _read_svmlight(lines, out_blocks, col_size, n_features, store_sparse):
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


@task(returns=1)
def _get_item(i, j, block):
    """
    Returns a single item from the block. Coords must be in block space.
    """
    return block[i, j]


@task(blocks={Type: COLLECTION_IN, Depth: 2}, returns=1)
def _filter_rows(blocks, rows):
    """
    Returns an array resulting of selecting rows of the input blocks
    """
    data = Array._merge_blocks(blocks)
    return data[rows, :]


@task(blocks={Type: COLLECTION_IN, Depth: 2}, returns=1)
def _filter_cols(blocks, cols):
    """
    Returns an array resulting of selecting rows of the input blocks
    """
    data = Array._merge_blocks(blocks)
    return data[:, cols]


@task(blocks={Type: COLLECTION_IN, Depth: 2},
      out_blocks={Type: COLLECTION_INOUT, Depth: 1})
def _merge_rows(blocks, out_blocks, blocks_shape, skip):
    """
    Merges the blocks into a single list of blocks where each block has bn
    as number of rows (the number of cols remains the same per block).
    """
    bn, bm = blocks_shape
    data = Array._merge_blocks(blocks)

    for j in range(0, ceil(data.shape[1] / bm)):
        out_blocks[j] = data[skip:bn, j * bm: (j + 1) * bm]


@task(blocks={Type: COLLECTION_IN, Depth: 2},
      out_blocks={Type: COLLECTION_INOUT, Depth: 1})
def _merge_cols(blocks, out_blocks, blocks_shape, skip):
    """
    Merges the blocks into a single list of blocks where each block has bn
    as number of rows (the number of cols remains the same per block).
    """
    bn, bm = blocks_shape
    data = Array._merge_blocks(blocks)

    for i in range(0, ceil(data.shape[0] / bn)):
        out_blocks[i] = data[i * bn: (i + 1) * bn, skip:bm]


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


@task(returns=np.array)
def _random_block(shape, seed):
    np.random.seed(seed)
    return np.random.random(shape)


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
