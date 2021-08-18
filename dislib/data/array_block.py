import numpy as np


def _validate_args(array, block_type, shape):
    if block_type in [ArrayBlock.ZEROS, ArrayBlock.IDENTITY] and shape is None:
        raise ValueError("shape cannot be empty if the block is ZEROS or IDENTITY")
    elif block_type in [ArrayBlock.OTHER] and array is None:
        raise ValueError("array cannot be empty if the block is OTHER")


class ArrayBlock:
    ZEROS = 0
    IDENTITY = 1
    OTHER = 2

    def __init__(self, array=None, block_type=OTHER, shape=None):
        _validate_args(array, block_type, shape)
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

    def __array__(self, dtype=None):
        if self._block_type == ArrayBlock.ZEROS:
            return np.zeros(self._shape, dtype)
        elif self._block_type == ArrayBlock.IDENTITY:
            return np.eye(self._shape[0], self._shape[1], dtype)
        elif self._block_type == ArrayBlock.OTHER:
            return self._array

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

    def copy(self):
        return ArrayBlock(self._array, self._block_type, self._shape)

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