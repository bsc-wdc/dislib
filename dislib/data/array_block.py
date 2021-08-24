import numpy as np


def _validate_args(array, block_type, shape, sparse):
    if block_type in [ArrayBlock.ZEROS, ArrayBlock.IDENTITY] and shape is None:
        raise ValueError("shape cannot be empty if the block is ZEROS or IDENTITY")
    elif block_type in [ArrayBlock.OTHER] and array is None:
        raise ValueError("array cannot be empty if the block is OTHER")
    elif sparse and array is None:
        raise ValueError("array cannot be empty if sparse")
    elif sparse and block_type not in [ArrayBlock.OTHER]:
        raise ValueError("only OTHER is accepted if sparse")


class ArrayBlock:
    ZEROS = 0
    IDENTITY = 1
    OTHER = 2

    def __init__(self, array, *, block_type=OTHER, shape=None, sparse=False):
        _validate_args(array, block_type, shape, sparse)
        self._sparse = sparse
        if self._sparse:
            self._array = array
            self._block_type = block_type
            self._shape = array.shape
        else:
            if block_type in [ArrayBlock.ZEROS, ArrayBlock.IDENTITY]:
                self._array = None
                self._block_type = block_type
                self._shape = shape
            elif block_type in [ArrayBlock.OTHER]:
                self._array = array
                self._block_type = block_type
                self._shape = array.shape

    def __repr__(self):
        return f"{self.__class__.__name__}(_array={self._array}, _block_type={self._block_type}, _shape={self._shape})"

    def __getitem__(self, arg):
        if self._block_type not in [ArrayBlock.ZEROS, ArrayBlock.IDENTITY, ArrayBlock.OTHER]:
            raise NotImplementedError("slicing over the given block type is not implemented")

        # return a single row
        if isinstance(arg, int):
            if self._block_type == ArrayBlock.ZEROS:
                return np.zeros((1, self._shape[1]))
            elif self._block_type == ArrayBlock.IDENTITY:
                return np.fromfunction(lambda i, j: 1. if j == arg else .0, (1, self._shape[1]))
            elif self._block_type == ArrayBlock.OTHER:
                return self._array[arg]

        # list of indices for rows
        elif isinstance(arg, list) or isinstance(arg, np.ndarray):
            if self._block_type == ArrayBlock.ZEROS:
                return np.zeros((len(arg), self._shape[1]))
            elif self._block_type == ArrayBlock.IDENTITY:
                # TODO
                raise NotImplementedError("IDENTITY not implemented")
                return np.fromfunction(lambda i, j: 1. if j in arg else .0, (len(arg), self._shape[1]))
            elif self._block_type == ArrayBlock.OTHER:
                return self._array[arg]

        # slicing only rows
        elif isinstance(arg, slice):
            # TODO
            raise NotImplementedError("row slicing not implemented")
            return self._get_slice(rows=arg, cols=slice(None, None))

        # we have indices for both dimensions
        if not isinstance(arg, tuple):
            raise IndexError("Invalid indexing information: %s" % arg)

        rows, cols = arg  # unpack 2-arguments

        # returning a single element
        if isinstance(rows, int) and isinstance(cols, int):
            if self._block_type == ArrayBlock.ZEROS:
                return np.zeros((1, 1))
            elif self._block_type == ArrayBlock.IDENTITY:
                return np.fromfunction(lambda i, j: 1. if j == i else .0, (1, 1))
            elif self._block_type == ArrayBlock.OTHER:
                return self._array[rows, cols]

        # all rows (slice : for rows) and list of indices for columns
        elif isinstance(rows, slice) and \
                (isinstance(cols, list) or isinstance(cols, np.ndarray)):

            if rows.step is not None and rows.step != 1:
                raise NotImplementedError("Variable steps not supported, contact"
                                          " the dislib team or open an issue "
                                          "in github.")

            if self._block_type == ArrayBlock.ZEROS:
                return np.zeros((rows.stop - rows.start, len(cols)))
            elif self._block_type == ArrayBlock.IDENTITY:
                # TODO
                raise NotImplementedError("IDENTITY not implemented")
                return np.fromfunction(lambda i, j: 1. if j == i else .0, (1, 1))
            elif self._block_type == ArrayBlock.OTHER:
                return self._array[rows.start:rows.stop, cols]

        # slicing both dimensions
        elif isinstance(rows, slice) and isinstance(cols, slice):
            if self._block_type == ArrayBlock.ZEROS:
                return np.zeros((rows.stop - rows.start, cols.stop - cols.start))
            elif self._block_type == ArrayBlock.IDENTITY:
                # TODO
                raise NotImplementedError("IDENTITY not implemented")
                return np.fromfunction(lambda i, j: 1. if j == i else .0, (1, 1))
            elif self._block_type == ArrayBlock.OTHER:
                return self._array[rows.start:rows.stop, cols.start:cols.stop]

        elif isinstance(rows, slice) and isinstance(cols, int):
            raise NotImplementedError("Single column indexing not supported.")

        raise IndexError("Invalid indexing information: %s" % str(arg))

    def __array__(self, dtype=None):
        if self._sparse or self._block_type == ArrayBlock.OTHER:
            return self._array
        elif self._block_type == ArrayBlock.ZEROS:
            return np.zeros(self._shape, dtype)
        elif self._block_type == ArrayBlock.IDENTITY:
            return np.eye(self._shape[0], self._shape[1], dtype)

    @property
    def type(self):
        return self._block_type

    @type.setter
    def type(self, type):
        if self.type == ArrayBlock.OTHER and self._block_type != type:
            raise ValueError("Cannot change type of the existing block")

        if type in [ArrayBlock.IDENTITY, ArrayBlock.ZEROS]:
            self._array = None

        self._block_type = type

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        if self._block_type == ArrayBlock.OTHER:
            raise ValueError("Cannot change shape of the existing block")
        self._shape = shape

    @property
    def sparse(self):
        return self._sparse

    @property
    def array(self):
        if self._sparse:
            return self._array
        else:
            raise AttributeError("internal array is accessible only if sparse")

    @array.setter
    def array(self, array):
        if self._sparse:
            self._array = array
            self._shape = array.shape
        else:
            raise AttributeError("internal array is accessible only if sparse. Use replace_content instead")

    def copy(self):
        return ArrayBlock(self._array, block_type=self._block_type, shape=self._shape)

    def replace_content(self, array=None, block_type=OTHER, shape=None):
        _validate_args(array, block_type, shape)
        if block_type in [ArrayBlock.ZEROS, ArrayBlock.IDENTITY]:
            self._array = None
            self._block_type = block_type
            self._shape = shape
        elif block_type in [ArrayBlock.OTHER]:
            self._array = array
            self._block_type = block_type
            self._shape = array.shape