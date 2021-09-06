import warnings
from logging import warning

import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix


def _validate_args(array, block_type, shape, sparse):
    if block_type in [ArrayBlock.ZEROS, ArrayBlock.IDENTITY] and shape is None:
        raise ValueError("shape cannot be empty if the block is ZEROS or IDENTITY")
    elif block_type in [ArrayBlock.OTHER] and array is None:
        raise ValueError("array cannot be empty if the block is OTHER")
    elif sparse and block_type not in [ArrayBlock.OTHER]:
        raise ValueError("only OTHER is accepted if sparse")


class ArrayBlock:
    ZEROS = 0
    IDENTITY = 1
    OTHER = 2

    def __init__(self, array, *, block_type=OTHER, shape=None, sparse=False):
        _validate_args(array, block_type, shape, sparse)
        self._sparse = sparse
        if self.sparse:
            self._array = csr_matrix(np.array([array]), shape=(1, 1)) if np.isscalar(array) else array
            self._block_type = block_type
            self._shape = self._array.shape
        else:
            if block_type in [ArrayBlock.ZEROS, ArrayBlock.IDENTITY]:
                self._array = None
                self._block_type = block_type
                self._shape = shape
            elif block_type in [ArrayBlock.OTHER]:
                self._array = np.full((1, 1), array) if np.isscalar(array) else array
                self._block_type = block_type
                self._shape = self._array.shape

    def __repr__(self):
        return f"{self.__class__.__name__}(_array={self._array}, _block_type={self._block_type}, _shape={self._shape}, _sparse={self._sparse})"

    def __getitem__(self, arg):
        if self._block_type not in [ArrayBlock.ZEROS, ArrayBlock.IDENTITY, ArrayBlock.OTHER]:
            raise NotImplementedError("slicing over the given block type is not implemented")

        # return a single row
        if isinstance(arg, int):
            if self._block_type == ArrayBlock.ZEROS:
                return ArrayBlock(None, block_type=ArrayBlock.ZEROS, shape=(1, self._shape[1]))
            elif self._block_type == ArrayBlock.IDENTITY:
                return ArrayBlock(np.fromfunction(lambda i, j: 1. if j == arg else .0, (1, self._shape[1])))
            elif self._block_type == ArrayBlock.OTHER:
                return ArrayBlock(self._array[arg], sparse=self.sparse)

        # list of indices for rows
        elif isinstance(arg, list) or isinstance(arg, np.ndarray):
            if self._block_type == ArrayBlock.ZEROS:
                return ArrayBlock(None, block_type=ArrayBlock.ZEROS, shape=(len(arg), self._shape[1]))
            elif self._block_type == ArrayBlock.IDENTITY:
                # TODO
                raise NotImplementedError("IDENTITY not implemented")
                return np.fromfunction(lambda i, j: 1. if j in arg else .0, (len(arg), self._shape[1]))
            elif self._block_type == ArrayBlock.OTHER:
                return ArrayBlock(self._array[arg], sparse=self.sparse)

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
                return ArrayBlock(None, block_type=ArrayBlock.ZEROS, shape=(1, 1))
            elif self._block_type == ArrayBlock.IDENTITY:
                return ArrayBlock(np.fromfunction(lambda i, j: 1. if j == i else .0, (1, 1)))
            elif self._block_type == ArrayBlock.OTHER:
                print(ArrayBlock(self._array[rows, cols], sparse=self.sparse))
                return ArrayBlock(self._array[rows, cols], sparse=self.sparse)

        # all rows (slice : for rows) and list of indices for columns
        elif isinstance(rows, slice) and \
                (isinstance(cols, list) or isinstance(cols, np.ndarray)):

            if rows.step is not None and rows.step != 1:
                raise NotImplementedError("Variable steps not supported, contact"
                                          " the dislib team or open an issue "
                                          "in github.")

            if self._block_type == ArrayBlock.ZEROS:
                return ArrayBlock(None,
                                  block_type=ArrayBlock.ZEROS,
                                  shape=(rows.stop - rows.start, len(cols)))
            elif self._block_type == ArrayBlock.IDENTITY:
                # TODO
                raise NotImplementedError("IDENTITY not implemented")
                return np.fromfunction(lambda i, j: 1. if j == i else .0, (1, 1))
            elif self._block_type == ArrayBlock.OTHER:
                return ArrayBlock(self._array[rows.start:rows.stop, cols], sparse=self.sparse)

        # slicing both dimensions
        elif isinstance(rows, slice) and isinstance(cols, slice):
            if self._block_type == ArrayBlock.ZEROS:
                return ArrayBlock(None,
                                  block_type=ArrayBlock.ZEROS,
                                  shape=(rows.stop - rows.start, cols.stop - cols.start))
            elif self._block_type == ArrayBlock.IDENTITY:
                # TODO
                raise NotImplementedError("IDENTITY not implemented")
                return np.fromfunction(lambda i, j: 1. if j == i else .0, (1, 1))
            elif self._block_type == ArrayBlock.OTHER:
                return ArrayBlock(self._array[rows.start:rows.stop, cols.start:cols.stop], sparse=self.sparse)

        elif isinstance(rows, slice) and isinstance(cols, int):
            raise NotImplementedError("Single column indexing not supported.")

        raise IndexError("Invalid indexing information: %s" % str(arg))

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            if not np.isscalar(value):
                raise IndexError("A value needs to be a scalar if two indices are provided.")

            if key[0] >= self.shape[0] or key[1] >= self.shape[1] or \
                    key[0] < 0 or key[1] < 0:
                raise IndexError("Index %r is out of bounds for ArrayBlock with "
                                 "shape %r." % (key, self.shape))

            if self._block_type == ArrayBlock.OTHER:
                self._array[key] = value
            elif self._block_type in [ArrayBlock.ZEROS, ArrayBlock.IDENTITY]:
                self._array = np.asarray(self)[key] = value
                self._block_type = ArrayBlock.OTHER
            else:
                raise NotImplementedError("Setitem for the given ArrayBlock type is not implemented")
        elif isinstance(key, slice):
            if self._block_type in [ArrayBlock.ZEROS, ArrayBlock.IDENTITY]:
                self._array = np.asarray(self)
                self._block_type = ArrayBlock.OTHER

            self._array[key.start:key.stop, :] = value
        else:
            raise NotImplementedError("Setitem is not implemented for the given type: {}".format(type(key)))

    def __matmul__(self, x):
        if self.shape[1] != x.shape[0]:
            raise ValueError(
                "Cannot multiply ArrayBlocks of shapes %r and %r" % (
                    self.shape, x.shape))

        if self._sparse != x.sparse:
            raise ValueError("Cannot multiply sparse and dense ArrayBlocks.")

        if self._sparse:
            return ArrayBlock(self.scipy() @ x.scipy(), sparse=True)
        else:
            # TODO implement optimal multiplication based on the array type
            return ArrayBlock(np.asarray(self) @ np.asarray(x))

    def __add__(self, x):
        if self.shape != x.shape:
            raise ValueError(
                "Cannot add ArrayBlocks of shapes %r and %r" % (
                    self.shape, x.shape))

        if self._sparse != x.sparse:
            raise ValueError("Cannot add sparse and dense ArrayBlocks.")

        if self._sparse:
            return ArrayBlock(self.scipy() + x.scipy(), sparse=True)
        else:
            # TODO implement optimal addition based on the array type
            return ArrayBlock(np.asarray(self) + np.asarray(x))

    def __array__(self, dtype=None):
        if self._sparse:
            warning("Conversion of scipy to numpy. Consider accessing the array via scipy() instead.")
            return np.asarray(self._array)
        elif self._block_type == ArrayBlock.OTHER:
            return self._array
        elif self._block_type == ArrayBlock.ZEROS:
            return np.zeros(self._shape, dtype=dtype)
        elif self._block_type == ArrayBlock.IDENTITY:
            return np.eye(self._shape[0], self._shape[1], dtype=dtype)

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
    def T(self):
        return self.transpose()

    '''
    @property
    def array(self):
        if self._sparse:
            return self._array
        else:
            return np.copy(self._array)

    @array.setter
    def array(self, array):
        if self._sparse:
            self._array = array
            self._shape = array.shape
        else:
            raise AttributeError("internal array is accessible only if sparse. Use replace_content instead")
    '''

    def numpy(self):
        if not self.sparse:
            return np.asarray(self)
        else:
            raise TypeError("This is not a numpy based block. Use scipy() instead")

    def scipy(self):
        if self.sparse:
            print(type(self._array))
            print(self._array)
            return csr_matrix(self._array, copy=True)
        else:
            raise TypeError("This is not a scipy based block. Use numpy() instead")

    def copy(self):
        array = self._array if self._block_type in [ArrayBlock.OTHER] else None
        return ArrayBlock(array, block_type=self._block_type, shape=self.shape)

    def replace_content(self, array=None, block_type=OTHER, shape=None, sparse=False):
        _validate_args(array, block_type, shape, sparse)
        if block_type in [ArrayBlock.ZEROS, ArrayBlock.IDENTITY]:
            self._array = None
            self._block_type = block_type
            self._shape = shape
        elif block_type in [ArrayBlock.OTHER]:
            self._array = array
            self._block_type = block_type
            self._shape = array.shape

    def apply(self, func, *args, **kwargs):
        if self._block_type in [ArrayBlock.OTHER]:
            print("old shape", self._shape)
            print("old array shape", self._array.shape)
            if len(args) > 0 and isinstance(args[0], ArrayBlock):
                print("second array shape", args[0].shape)
            # FIXME TypeError: unsupported operand type(s) for @: 'csr_matrix' and 'ArrayBlock'
            # FIXME self._array is a csr_matrix, while the other block is ArrayBlock
            # FIXME needs to be fixed
            self._array = func(self._array, *args, *kwargs)
            self._shape = self._array.shape
            print("new shape", self._shape)
            print("new array shape", self._array.shape)
        elif self._block_type in [ArrayBlock.ZEROS, ArrayBlock.IDENTITY]:
            if not self._sparse:
                self._array = func(np.asarray(self), *args, **kwargs)
                self._shape = self._array.shape
                self._block_type = ArrayBlock.OTHER
            else:
                raise ValueError("Sparse zeros/identity blocks are not handled")

    def transpose(self):
        if self._block_type in [ArrayBlock.ZEROS, ArrayBlock.IDENTITY]:
            return ArrayBlock(None,
                              block_type=self._block_type,
                              shape=(self._shape[1], self._shape[0]),
                              sparse=self.sparse)
        elif self._block_type in [ArrayBlock.OTHER]:
            return ArrayBlock(self._array.transpose(), sparse=self.sparse)
