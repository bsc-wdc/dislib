import operator
from collections import defaultdict
from math import ceil

import numpy as np
from pycompss.api.api import compss_wait_on, compss_delete_object
from pycompss.api.parameter import Type, COLLECTION_IN, Depth, \
    COLLECTION_OUT, INOUT
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
        - ``A[[i,j,k]]`` : returns a set of non-consecutive rows. Rows are
        returned ordered by their index in the input array.
        - ``A[:, [i,j,k]]`` : returns a set of non-consecutive columns.
        Columns are returned ordered by their index in the input array.
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
    delete : boolean, optional (default=True)
        Whether to call compss_delete_object on the blocks when the garbage
        collector deletes this ds-array.

    Attributes
    ----------
    shape : tuple (int, int)
        Total number of elements in the array.
    """

    def __init__(self, blocks, top_left_shape, reg_shape, shape, sparse,
                 delete=True):
        self._validate_blocks(blocks)

        self._blocks = blocks
        self._top_left_shape = top_left_shape
        self._reg_shape = reg_shape

        self._n_blocks = (len(blocks), len(blocks[0]))
        self._shape = shape
        self._sparse = sparse

        self._delete = delete

    def __del__(self):
        if self._delete:
            [compss_delete_object(b) for r_block in self._blocks for b in
             r_block]

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

    def __matmul__(self, x):
        if self.shape[1] != x.shape[0]:
            raise ValueError(
                "Cannot multiply ds-arrays of shapes %r and %r" % (
                    self.shape, x.shape))

        if self._n_blocks[1] != x._n_blocks[0] or \
                self._reg_shape[1] != x._reg_shape[0] or \
                self._top_left_shape[1] != x._top_left_shape[0]:
            raise ValueError("Cannot multiply ds-arrays with incompatible "
                             "number of blocks or different block shapes.")

        if self._sparse != x._sparse:
            raise ValueError("Cannot multiply sparse and dense ds-arrays.")

        n_blocks = (self._n_blocks[0], x._n_blocks[1])
        blocks = Array._get_out_blocks(n_blocks)

        for i in range(n_blocks[0]):
            for j in range(n_blocks[1]):
                hblock = self._blocks[i]
                vblock = [x._blocks[k][j] for k in range(len(x._blocks))]

                blocks[i][j] = _multiply_block_groups(hblock, vblock)

        shape = (self.shape[0], x.shape[1])
        tl_shape = (self._top_left_shape[0], x._top_left_shape[1])
        reg_shape = (self._reg_shape[0], x._reg_shape[1])

        return Array(blocks=blocks, top_left_shape=tl_shape,
                     reg_shape=reg_shape, shape=shape, sparse=self._sparse)

    def __getitem__(self, arg):

        # return a single row
        if isinstance(arg, int):
            return self._get_by_lst_rows(rows=[arg])

        # list of indices for rows
        elif isinstance(arg, list) or isinstance(arg, np.ndarray):
            return self._get_by_lst_rows(rows=arg)

        # slicing only rows
        elif isinstance(arg, slice):
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

        elif isinstance(rows, slice) and isinstance(cols, int):
            raise NotImplementedError("Single column indexing not supported.")

        raise IndexError("Invalid indexing information: %s" % str(arg))

    def __setitem__(self, key, value):
        if not np.isscalar(value):
            raise ValueError("Can only assign scalar values.")

        if not isinstance(key, tuple):
            raise IndexError("Need to provide two indexes to assign a value.")

        if key[0] >= self.shape[0] or key[1] >= self.shape[1] or \
                key[0] < 0 or key[1] < 0:
            raise IndexError("Index %r is out of bounds for ds-array with "
                             "shape %r." % (key, self.shape))

        bi, bj = self._get_containing_block(*key)
        vi, vj = self._coords_in_block(bi, bj, *key)

        _set_value(self._blocks[bi][bj], vi, vj, value)

    def __pow__(self, power, modulo=None):
        if not np.isscalar(power):
            raise NotImplementedError("Power is only supported for scalars")
        return _apply_elementwise(Array._power, self, power)

    def __sub__(self, other):
        if self.shape[1] != other.shape[1] or other.shape[0] != 1:
            raise NotImplementedError("Subtraction not implemented for the "
                                      "given objects")

        # matrix - vector
        blocks = []

        for hblock in self._iterator("rows"):
            out_blocks = [object() for _ in range(hblock._n_blocks[1])]
            _combine_blocks(hblock._blocks, other._blocks,
                            Array._subtract, out_blocks)
            blocks.append(out_blocks)

        return Array(blocks, self._top_left_shape, self._reg_shape,
                     self.shape, self._sparse)

    def __truediv__(self, other):
        if not np.isscalar(other):
            raise NotImplementedError("Non scalar division not supported")

        return _apply_elementwise(operator.truediv, self, other)

    def __mul__(self, other):
        if self.shape[1] != other.shape[1] or other.shape[0] != 1:
            raise NotImplementedError("Multiplication not implemented for the "
                                      "given arrays")

        # matrix * vector
        blocks = []

        for hblock in self._iterator("rows"):
            out_blocks = [object() for _ in range(hblock._n_blocks[1])]
            _combine_blocks(hblock._blocks, other._blocks,
                            operator.mul, out_blocks)
            blocks.append(out_blocks)

        return Array(blocks, self._top_left_shape, self._reg_shape,
                     self.shape, self._sparse)

    @property
    def shape(self):
        """
        Total shape of the ds-array
        """
        return self._shape

    @property
    def T(self):
        """ Returns the transpose of this ds-array """
        return self.transpose()

    @staticmethod
    def _subtract(a, b):
        sparse = issparse(a)

        # needed because subtract with scipy.sparse does not support
        # broadcasting
        if sparse:
            a = a.toarray()
        if issparse(b):
            b = b.toarray()

        if sparse:
            return csr_matrix(a - b)
        else:
            return a - b

    @staticmethod
    def _power(x_np, power):
        if issparse(x_np):
            return sp.csr_matrix.power(x_np, power)
        else:
            return x_np ** power

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
        parameter of type COLLECTION_OUT
        """
        return [[object() for _ in range(n_blocks[1])]
                for _ in range(n_blocks[0])]

    @staticmethod
    def _get_block_shape_static(i, j, x):
        reg_blocks = (max(0, x._n_blocks[0] - 2),
                      max(0, x._n_blocks[1] - 2))
        remain_shape = (x.shape[0] - x._top_left_shape[0] -
                        reg_blocks[0] * x._reg_shape[0],
                        x.shape[1] - x._top_left_shape[1] -
                        reg_blocks[1] * x._reg_shape[1])

        if i == 0:
            shape0 = x._top_left_shape[0]
        elif i < x._n_blocks[0] - 1:
            shape0 = x._reg_shape[0]
        else:
            shape0 = remain_shape[0]

        if j == 0:
            shape1 = x._top_left_shape[1]
        elif j < x._n_blocks[1] - 1:
            shape1 = x._reg_shape[1]
        else:
            shape1 = remain_shape[1]

        return (shape0, shape1)

    @staticmethod
    def _rechunk(blocks, shape, block_size, shape_f, *args, **kwargs):
        """ Re-partitions a set of blocks into a new ds-array of the given
        block size.

        shape_f is a function that returns the shape of the (i,j) block. It
        has to take at least two indices as arguments. This function is
        needed to rechunk an irregular set of blocks such as in the ds.kron
        operation, where the shape of a block is not trivial to compute.
        """
        if shape[0] < block_size[0] or shape[1] < block_size[1]:
            raise ValueError("Block size is greater than the array")

        cur_element = [0, 0]
        tl_shape = list(block_size)
        n_blocks = (ceil(shape[0] / block_size[0]),
                    ceil(shape[1] / block_size[1]))
        tmp_blocks = [[[] for _ in range(n_blocks[1])] for _ in
                      range(n_blocks[0])]

        # iterate over each block, split it if necessary, and place each
        # part into a new list of blocks to form the output blocks later
        for i in range(len(blocks)):
            cur_element[1] = 0
            tl_shape[1] = block_size[1]

            for j in range(len(blocks[i])):
                bshape = shape_f(i, j, *args, **kwargs)

                out_n_blocks = (ceil((bshape[0] - tl_shape[0]) /
                                     block_size[0]) + 1,
                                ceil((bshape[1] - tl_shape[1]) /
                                     block_size[1]) + 1)

                out_blocks = Array._get_out_blocks(out_n_blocks)

                _split_block(blocks[i][j], list(tl_shape), block_size,
                             out_blocks)

                cur_block = (int(cur_element[0] / block_size[0]),
                             int(cur_element[1] / block_size[1]))

                # distribute each part of the original block into the
                # corresponding new blocks. cur_block keeps track of the new
                # block that we are generating, but some parts of the
                # orignal block might go to neighbouring new blocks
                for m in range(len(out_blocks)):
                    for n in range(len(out_blocks[m])):
                        bi = cur_block[0] + m
                        bj = cur_block[1] + n
                        tmp_blocks[bi][bj].append(out_blocks[m][n])

                tl_shape[1] = block_size[1] - ((bshape[1] - tl_shape[1])
                                               % block_size[1])
                cur_element[1] += bshape[1]

            tl_shape[0] = block_size[0] - ((bshape[0] - tl_shape[0]) %
                                           block_size[0])
            cur_element[0] += bshape[0]

        final_blocks = Array._get_out_blocks(n_blocks)
        irr_shape = (shape[0] - (n_blocks[0] - 1) * block_size[0],
                     shape[1] - (n_blocks[1] - 1) * block_size[1])

        # merges the different parts of each original block into new blocks
        # of the given block size
        for i in range(n_blocks[0]):
            bs0 = block_size[0] if i < n_blocks[0] - 1 else irr_shape[0]

            for j in range(n_blocks[1]):
                bs1 = block_size[1] if j < n_blocks[1] - 1 else irr_shape[1]

                # if there is more than one part, merge them, otherwise the
                # block is already of the wanted block size
                if len(tmp_blocks[i][j]) > 1:
                    final_blocks[i][j] = _assemble_blocks(tmp_blocks[i][j],
                                                          (bs0, bs1))
                    [compss_delete_object(block) for block in tmp_blocks[i][j]]
                else:
                    final_blocks[i][j] = tmp_blocks[i][j][0]

        return Array(final_blocks, block_size, block_size, shape, False)

    def _is_regular(self):
        return self._reg_shape == self._top_left_shape

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

    def _get_block_shape(self, i, j):
        return Array._get_block_shape_static(i, j, self)

    def _get_row_block(self, i):
        row_shape = self._get_row_shape(i)
        return Array(blocks=[self._blocks[i]],
                     top_left_shape=(row_shape[0], self._top_left_shape[1]),
                     reg_shape=self._reg_shape, shape=row_shape,
                     sparse=self._sparse, delete=False)

    def _get_col_block(self, i):
        col_shape = self._get_col_shape(i)
        col_blocks = [[self._blocks[j][i]] for j in range(self._n_blocks[0])]
        return Array(blocks=col_blocks,
                     top_left_shape=(self._top_left_shape[0], col_shape[1]),
                     reg_shape=self._reg_shape, shape=col_shape,
                     sparse=self._sparse, delete=False)

    def _iterator(self, axis=0):
        # iterate through rows
        if axis == 0 or axis == 'rows':
            for i in range(self._n_blocks[0]):
                yield self._get_row_block(i)
        # iterate through columns
        elif axis == 1 or axis == 'columns':
            for j in range(self._n_blocks[1]):
                yield self._get_col_block(j)
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
        if i > self.shape[0] or j > self.shape[1]:
            raise IndexError("Shape is ", self.shape)

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

        # The shape of the top left block of the sliced array depends on the
        # slice. To compute it, we need the shape of the block of
        # the original array where the sliced array starts. This block can
        # be regular or irregular (i.e., the block is on the edges).
        b0, b1 = self._reg_shape

        if i_0 == 0:
            # block is at the top
            b0 = self._top_left_shape[0]
        elif i_0 == self._n_blocks[0] - 1:
            # block is at the bottom (can be regular or irregular)
            b0 = (self.shape[0] - self._top_left_shape[0]) % self._reg_shape[0]

            if b0 == 0:
                b0 = self._reg_shape[0]

        if j_0 == 0:
            # block is leftmost
            b1 = self._top_left_shape[1]
        elif j_0 == self._n_blocks[1] - 1:
            # block is rightmost (can be regular or irregular)
            b1 = (self.shape[1] - self._top_left_shape[1]) % self._reg_shape[1]

            if b1 == 0:
                b1 = self._reg_shape[1]

        block_shape = (b0, b1)

        top, left = self._coords_in_block(i_0, j_0, r_start, c_start)

        bi0 = min(n_rows, block_shape[0] - (top % block_shape[0]))
        bj0 = min(n_cols, block_shape[1] - (left % block_shape[1]))

        # Regular blocks shape is the same
        bn, bm = self._reg_shape

        out_shape = n_rows, n_cols

        res = Array(blocks=out_blocks, top_left_shape=(bi0, bj0),
                    reg_shape=(bn, bm), shape=out_shape,
                    sparse=self._sparse, delete=False)
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
        total_rows = 0
        for rowblock_idx, row in enumerate(self._iterator(axis='rows')):
            # create an empty list for the filtered row (single depth)
            rows_in_block = len(adj_row_idxs[rowblock_idx])
            total_rows += rows_in_block
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
                n_blocks = ceil(self.shape[1] / self._reg_shape[1])
                out_blocks = [object() for _ in range(n_blocks)]
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
            n_blocks = ceil(self.shape[1] / self._reg_shape[1])
            out_blocks = [object() for _ in range(n_blocks)]
            _merge_rows(to_merge, out_blocks, self._reg_shape, skip)
            final_blocks.append(out_blocks)

        top_left_shape = (min(total_rows, self._reg_shape[0]),
                          self._reg_shape[1])

        return Array(blocks=final_blocks, top_left_shape=top_left_shape,
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
        total_cols = 0
        for colblock_idx, col in enumerate(self._iterator(axis='columns')):
            # create an empty list for the filtered row (single depth)
            cols_in_block = len(adj_col_idxs[colblock_idx])
            total_cols += cols_in_block
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
            if n_cols >= self._reg_shape[1]:
                n_blocks = ceil(self.shape[0] / self._reg_shape[0])
                out_blocks = [object() for _ in range(n_blocks)]
                _merge_cols([to_merge], out_blocks, self._reg_shape, skip)
                final_blocks.append(out_blocks)

                # if we didn't take all cols, we keep the last block and
                # remember to skip the cols that have been merged
                if n_cols > self._reg_shape[1]:
                    to_merge = [col]
                    n_cols = n_cols - self._reg_shape[1]
                    skip = cols_in_block - n_cols
                else:
                    to_merge = []
                    n_cols = 0
                    skip = 0

        if n_cols > 0:
            n_blocks = ceil(self.shape[0] / self._reg_shape[0])
            out_blocks = [object() for _ in range(n_blocks)]
            _merge_cols([to_merge], out_blocks, self._reg_shape, skip)
            final_blocks.append(out_blocks)

        # list are in col-order transpose them for the correct ordering
        final_blocks = list(map(list, zip(*final_blocks)))

        top_left_shape = (self._reg_shape[0],
                          min(total_cols, self._reg_shape[1]))

        return Array(blocks=final_blocks, top_left_shape=top_left_shape,
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

    def norm(self, axis=0):
        """ Returns the Frobenius norm along an axis.

        Parameters
        ----------
        axis : int, optional (default=0)
            Specifies the axis of the array along which to compute the vector
            norms.

        Returns
        -------
        norm : ds-array
            Norm along axis.

        Raises
        -------
        NotImplementedError
            If the ds-array is sparse.
        """
        if self._sparse:
            raise NotImplementedError("Cannot compute the norm of sparse "
                                      "ds-arrays.")

        return apply_along_axis(np.linalg.norm, axis, self)

    def sqrt(self):
        """ Returns the element-wise square root of the elements in the
        ds-array

        Returns
        -------
        x : ds-array
        """
        return _apply_elementwise(np.sqrt, self)

    def conj(self):
        """ Returns the complex conjugate, element-wise.

        Returns
        -------
        x : ds-array
        """
        return _apply_elementwise(np.conj, self)

    def rechunk(self, block_size):
        """ Re-partitions the ds-array into blocks of the given block size.

        Parameters
        ----------
        block_size : tuple of two ints
            The desired block size.

        Returns
        -------
        x : ds-array
            Re-partitioned ds-array.
        """
        if self._sparse:
            raise NotImplementedError("Cannot rechunk a sparse ds-array.")

        return Array._rechunk(self._blocks, self.shape, block_size,
                              Array._get_block_shape_static, self)

    def copy(self):
        """ Creates a copy of this ds-array.

        Returns
        -------
        x_copy : ds-array
        """
        blocks = Array._get_out_blocks(self._n_blocks)

        for i in range(self._n_blocks[0]):
            for j in range(self._n_blocks[1]):
                blocks[i][j] = _copy_block(self._blocks[i][j])

        return Array(blocks, self._top_left_shape, self._reg_shape,
                     self.shape, self._sparse, self._delete)

    def collect(self, squeeze=True):
        """
        Collects the contents of this ds-array and returns the equivalent
        in-memory array that this ds-array represents. This method creates a
        synchronization point in the execution of the application.

        Warning: This method may fail if the ds-array does not fit in
        memory.

        Parameters
        ----------
        squeeze : boolean, optional (default=True)
            Whether to remove single-dimensional entries from the shape of
            the resulting ndarray.

        Returns
        -------
        array : nd-array or spmatrix
            The actual contents of the ds-array.
        """
        self._blocks = compss_wait_on(self._blocks)
        res = Array._merge_blocks(self._blocks)
        if not self._sparse and squeeze:
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

    if len(x.shape) > 2:
        raise ValueError("Input data has more than 2 dimensions.")

    if len(x.shape) < 2:
        if block_size[0] == 1:
            x = x.reshape(1, -1)
        elif block_size[1] == 1:
            x = x.reshape(-1, 1)
        else:
            raise ValueError("Input array is one-dimensional but "
                             "block size is greater than 1.")

    if x.shape[0] < block_size[0] or x.shape[1] < block_size[1]:
        raise ValueError("Block size is greater than the array")

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
    """ Returns a distributed array of random floats in the open interval [0.0,
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
    x : ds-array
        Distributed array of random floats.
    """
    r_state = check_random_state(random_state)
    return _full(shape, block_size, False, _random_block_wrapper, r_state)


def identity(n, block_size, dtype=None):
    """ Returns the identity matrix.

    Parameters
    ----------
    n : int
        Size of the matrix.
    block_size : tuple of two ints
        Block size.
    dtype : data type, optional (default=None)
        The desired type of the ds-array. Defaults to float.

    Returns
    -------
    x : ds-array
        Identity matrix of shape n x n.

    Raises
    ------
    ValueError
        If block_size is greater than n.
    """
    if n < block_size[0] or n < block_size[1]:
        raise ValueError("Block size is greater than the array")

    n_blocks = (int(ceil(n / block_size[0])), int(ceil(n / block_size[1])))
    blocks = list()

    for row_idx in range(n_blocks[0]):
        blocks.append(list())

        for col_idx in range(n_blocks[1]):
            b_size0, b_size1 = block_size

            if row_idx == n_blocks[0] - 1:
                b_size0 = n - (n_blocks[0] - 1) * block_size[0]

            if col_idx == n_blocks[1] - 1:
                b_size1 = n - (n_blocks[1] - 1) * block_size[1]

            block = _identity_block((b_size0, b_size1), n, block_size,
                                    row_idx, col_idx, dtype)
            blocks[-1].append(block)

    return Array(blocks, top_left_shape=block_size, reg_shape=block_size,
                 shape=(n, n), sparse=False)


def zeros(shape, block_size, dtype=None):
    """ Returns a ds-array of given shape and block size, filled with zeros.

    Parameters
    ----------
    shape : tuple of two ints
        Shape of the output ds-array.
    block_size : tuple of two ints
        Size of the ds-array blocks.
    dtype : data type, optional (default=None)
        The desired type of the array. Defaults to float.

    Returns
    -------
    x : ds-array
        Distributed array filled with zeros.
    """
    return _full(shape, block_size, False, _full_block, 0, dtype)


def full(shape, block_size, fill_value, dtype=None):
    """ Returns a ds-array of 'shape' filled with 'fill_value'.

    Parameters
    ----------
    shape : tuple of two ints
        Shape of the output ds-array.
    block_size : tuple of two ints
        Size of the ds-array blocks.
    fill_value : scalar
        Fill value.
    dtype : data type, optional (default=None)
        The desired type of the array. Defaults to float.

    Returns
    -------
    x : ds-array
        Distributed array filled with the fill value.
    """
    return _full(shape, block_size, False, _full_block, fill_value, dtype)


def apply_along_axis(func, axis, x, *args, **kwargs):
    r""" Apply a function to slices along the given axis.

    Execute func(a, \*args, \*\*kwargs) where func operates on nd-arrays and a
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
        out = _block_apply_axis(func, axis, block._blocks, *args, **kwargs)
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
                 shape=out_shape, sparse=x._sparse)


def _multiply_block_groups(hblock, vblock):
    blocks = []

    for blocki, blockj in zip(hblock, vblock):
        blocks.append(_block_apply(operator.matmul, blocki, blockj))

    while len(blocks) > 1:
        block1 = blocks.pop(0)
        block2 = blocks.pop(0)
        blocks.append(_block_apply(operator.add, block1, block2))

        compss_delete_object(block1)
        compss_delete_object(block2)

    return blocks[0]


def _full(shape, block_size, sparse, func, *args, **kwargs):
    """
    Creates a ds-array with custom contents defined by `func`. `func` must
    take `block_size` as the first argument, and must return one block of
    the resulting ds-array.

    Parameters
    ----------
    shape : tuple of two ints
        Shape of the output ds-array.
    block_size : tuple of two ints
        Size of the ds-array blocks.
    sparse : bool
        Whether `func` generates sparse blocks.
    func : function
        Function that generates the blocks of the resulting ds-array. Must
        take `block_size` as the first argument.
    args : any
        Additional arguments to pass to `func`.
    kwargs : any
        Additional keyword arguments to pass to `func`.

    Returns
    -------
    x : ds-array
    """
    if shape[0] < block_size[0] or shape[1] < block_size[1]:
        raise ValueError("Block size is greater than the array")

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

            block = func((b_size0, b_size1), *args, **kwargs)
            blocks[-1].append(block)

    return Array(blocks, top_left_shape=block_size, reg_shape=block_size,
                 shape=shape, sparse=sparse)


def _apply_elementwise(func, x, *args, **kwargs):
    """ Applies a function element-wise to each block in parallel"""
    n_blocks = x._n_blocks
    blocks = Array._get_out_blocks(n_blocks)

    for i in range(n_blocks[0]):
        for j in range(n_blocks[1]):
            blocks[i][j] = _block_apply(func, x._blocks[i][j], *args, **kwargs)

    return Array(blocks, x._top_left_shape, x._reg_shape, x.shape, x._sparse)


def _random_block_wrapper(block_size, r_state):
    seed = r_state.randint(np.iinfo(np.int32).max)
    return _random_block(block_size, seed)


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
      out_blocks={Type: COLLECTION_OUT, Depth: 1})
def _merge_rows(blocks, out_blocks, blocks_shape, skip):
    """
    Merges the blocks into a single list of blocks where each block has bn
    as number of rows (the number of cols remains the same per block).
    """
    bn, bm = blocks_shape
    data = Array._merge_blocks(blocks)

    for j in range(0, ceil(data.shape[1] / bm)):
        out_blocks[j] = data[skip:bn + skip, j * bm: (j + 1) * bm]


@task(blocks={Type: COLLECTION_IN, Depth: 2},
      out_blocks={Type: COLLECTION_OUT, Depth: 1})
def _merge_cols(blocks, out_blocks, blocks_shape, skip):
    """
    Merges the blocks into a single list of blocks where each block has bn
    as number of rows (the number of cols remains the same per block).
    """
    bn, bm = blocks_shape
    data = Array._merge_blocks(blocks)

    for i in range(0, ceil(data.shape[0] / bn)):
        out_blocks[i] = data[i * bn: (i + 1) * bn, skip:bm + skip]


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
      out_blocks={Type: COLLECTION_OUT, Depth: 2})
def _transpose(blocks, out_blocks):
    for i in range(len(blocks)):
        for j in range(len(blocks[i])):
            out_blocks[i][j] = blocks[i][j].transpose()


@task(returns=np.array)
def _random_block(shape, seed):
    np.random.seed(seed)
    return np.random.random(shape)


@task(returns=1)
def _identity_block(block_size, n, reg_shape, i, j, dtype):
    block = np.zeros(block_size, dtype)

    i_values = np.arange(i * reg_shape[0], min(n, (i + 1) * reg_shape[0]))
    j_values = np.arange(j * reg_shape[1], min(n, (j + 1) * reg_shape[1]))

    indices = np.intersect1d(i_values, j_values)

    i_ones = indices - (i * reg_shape[0])
    j_ones = indices - (j * reg_shape[1])

    block[i_ones, j_ones] = 1
    return block


@task(returns=np.array)
def _full_block(shape, value, dtype):
    return np.full(shape, value, dtype)


@task(blocks={Type: COLLECTION_IN, Depth: 2}, returns=np.array)
def _block_apply_axis(func, axis, blocks, *args, **kwargs):
    arr = Array._merge_blocks(blocks)
    kwargs['axis'] = axis
    out = func(arr, *args, **kwargs)

    # We don't know the data type that func returns (could be dense for a
    # sparse input). Therefore, we force the output to be of the same type
    # of the input. Otherwise, the result of apply_along_axis would be of
    # unknown type.
    if not issparse(arr):
        out = np.asarray(out)
    else:
        out = csr_matrix(out)

    if axis == 0:
        return out.reshape(1, -1)
    else:
        return out.reshape(-1, 1)


@task(returns=1)
def _block_apply(func, block, *args, **kwargs):
    return func(block, *args, **kwargs)


@task(block=INOUT)
def _set_value(block, i, j, value):
    block[i][j] = value


@task(blocks={Type: COLLECTION_IN, Depth: 1}, returns=1)
def _assemble_blocks(blocks, bshape):
    """ Generates a block of shape bshape from a list of blocks of arbitrary
    shapes that can be assembled together into bshape """
    merged = list()
    size = 0

    for j, block in enumerate(blocks):
        size += block.shape[1]

        if size / bshape[1] > len(merged):
            merged.append([])

        merged[-1].append(block)

    return np.block(merged)


@task(out_blocks={Type: COLLECTION_OUT, Depth: 2})
def _split_block(block, tl_shape, reg_shape, out_blocks):
    """ Splits a block into new blocks following the ds-array typical scheme
    with a top left block, regular blocks in the middle and remainder blocks
    at the edges """
    vsplit = range(tl_shape[0], block.shape[0], reg_shape[0])
    hsplit = range(tl_shape[1], block.shape[1], reg_shape[1])

    for i, rows in enumerate(np.vsplit(block, vsplit)):
        for j, cols in enumerate(np.hsplit(rows, hsplit)):
            # copy is only necessary when executing with regular Python.
            # When using PyCOMPSs the reference to the original block is broken
            # because this is executed in a task.
            out_blocks[i][j] = cols.copy()


@task(returns=1)
def _copy_block(block):
    return block.copy()


@task(blocks={Type: COLLECTION_IN, Depth: 2},
      other={Type: COLLECTION_IN, Depth: 2},
      out_blocks={Type: COLLECTION_OUT, Depth: 1})
def _combine_blocks(blocks, other, func, out_blocks):
    x = Array._merge_blocks(blocks)
    y = Array._merge_blocks(other)

    res = func(x, y)

    bsize = blocks[0][0].shape[1]

    for i in range(len(out_blocks)):
        out_blocks[i] = res[:, i * bsize: (i + 1) * bsize]
