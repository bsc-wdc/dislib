import operator
try:
    import torch
except ImportError as e:
    raise Exception("ERROR: Pytorch is unavailable and "
                    "tensors can not be used.") from e
import os
import numpy as np
from pycompss.api.api import compss_wait_on, compss_delete_object
from pycompss.api.constraint import constraint
from pycompss.api.parameter import Type, Depth, \
    COLLECTION_OUT, INOUT, COLLECTION_IN, FILE_IN
from pycompss.api.task import task
import math
from dislib.data.array import Array


class Tensor(object):
    """ A distributed n-dimensional tensor divided in blocks.

        Normally, this class should not be instantiated directly, but created
        using one of the array creation routines provided.

        Apart from the different methods provided, this class supports the same
        indexing as a numpy array or pytorch tensor

        Parameters
        ----------
        tensors : list
            List of lists of nd-array or pytorch tensor.
        tensor_shape : tuple
            A single tuple indicating the shape of the distributed tensors.
        shape : tuple (int, int)
            Number of tensors inside the tensors attributes
            of the Tensor object in each of the dimensions.
        shape : int
            Total number of tensors in the Tensor.
        dtype : np or torch object
            Numerical type elements inside the tensor have.
        number_samples: int
            Total number of samples contained in the different distributed
            tensors.
        delete : boolean, optional (default=True)
            Whether to call compss_delete_object on the blocks when the garbage
            collector deletes this Tensor.

        Attributes
        ----------
        shape : tuple (int, int)
            Total number of elements in the array.
        """
    def __init__(self, tensors, tensor_shape, dtype, number_samples,
                 shape=None, delete=True):
        self.tensors = tensors
        if shape:
            self.shape = shape
        else:
            self.shape = (len(tensors), len(tensors[0]))
        self.n_tensors = len(tensors) * len(tensors[0])
        self.tensor_shape = tensor_shape
        self.number_samples = number_samples
        self.dtype = dtype
        self._delete = delete

    def __str__(self):
        return "ds-tensor(tensors=(...), " \
               "tensors_shape=%r," \
               "n_tensors=%r," \
               "num_samples=%r, " \
               "shape=%r)" % (self.tensor_shape,
                              self.n_tensors,
                              self.number_samples,
                              self.shape)

    def __del__(self):
        if self._delete:
            [compss_delete_object(tensor) for tensors in self.tensors for
             tensor in tensors]

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            if len(key) == 2:
                if value.shape != self.tensor_shape:
                    raise ValueError("The tensor has different shape than "
                                     "the expected on this ds-tensor")
                if isinstance(value, (np.ndarray, np.generic)) or \
                        torch.is_tensor(value) or isinstance(value, list):
                    if all(isinstance(v, int) for v in key):
                        if key[0] >= self.shape[0] or key[1] >= self.shape[1] \
                                or key[0] < 0 or key[1] < 0:
                            raise IndexError("Index %r is out of bounds for "
                                             "ds-tensor with shape %r." %
                                             (key, self.shape))
                        self.tensors[key[0]][key[1]] = value
                    elif all(isinstance(v, list) for v in key):
                        if any(idx >= self.shape[0] for idx in
                               key[0]) or any(idx >= self.shape[1] for idx in
                                              key[1]):
                            raise IndexError("Index %r is out of bounds for "
                                             "ds-tensor "
                                             "with shape %r." %
                                             (key, self.shape))
                        for row in key[0]:
                            for col in key[1]:
                                self.tensors[row][col] = value
                else:
                    raise ValueError("The type of tensor being tried "
                                     "to assign is not supported.")
            elif all(isinstance(v, int) for v in key):
                if key[0] >= self.shape[0] or key[1] >= self.shape[1] or \
                        key[0] < 0 or key[1] < 0:
                    raise IndexError("Index %r is out of bounds for "
                                     "ds-tensor with shape %r." %
                                     (key, self.shape))
                if any(value >= self.tensor_shape[idx] for idx, value in
                       enumerate(key[2:])):
                    raise IndexError("Index %r is out of bounds for tensors "
                                     "inside ds-tensor with shape %r." %
                                     (key, self.tensor_shape))
                _set_value(self.tensors[key[0]][key[1]], key[2:], value)
            else:
                if isinstance(key[0], int) and isinstance(key[1], int):
                    if key[0] >= self.shape[0] or key[0] < 0 or key[1] >= \
                            self.shape[1] or key[1] < 0:
                        raise IndexError("Index %r is out of bounds for "
                                         "ds-tensor with shape %r." %
                                         (key, self.shape))
                    _set_value(self.tensors[key[0]][key[1]], key[2:], value)
                elif isinstance(key[0], int) and isinstance(key[1], list):
                    if key[0] >= self.shape[0] or key[0] < 0:
                        raise IndexError("Index %r is out of bounds for "
                                         "ds-tensor with shape %r." %
                                         (key, self.shape))
                    if any(value >= self.shape[1] for value in key[1]):
                        raise IndexError("At least one index of tensor in "
                                         "axis=1 is out of bounds")
                    for col in key[1]:
                        _set_value(self.tensors[key[0]][col], key[2:], value)
                elif isinstance(key[1], int) and isinstance(key[0], list):
                    if key[1] >= self.shape[1] or key[1] < 0:
                        raise IndexError("Index %r is out of bounds for "
                                         "ds-tensor with shape %r." %
                                         (key, self.shape))
                    if any(value >= self.shape[0] for value in key[0]):
                        raise IndexError("At least one index of tensor "
                                         "in axis=0 is out of bounds")
                    for row in key[0]:
                        _set_value(self.tensors[row][key[1]], key[2:], value)
                elif isinstance(key[0], list) and isinstance(key[1], list):
                    if any(value >= self.shape[0] for value in key[0]):
                        raise IndexError("At least one index of tensor "
                                         "in axis=0 is out of bounds")
                    if any(value >= self.shape[1] for value in key[1]):
                        raise IndexError("At least one index of tensor "
                                         "in axis=1 is out of bounds")
                    for row in key[0]:
                        for col in key[1]:
                            _set_value(self.tensors[row][col], key[2:], value)
        else:
            raise NotImplementedError(
                f"Provided indexing by {type(key)} is not implemented."
            )

    def __getitem__(self, arg):
        if isinstance(arg, int):
            return self._get_various_tensor([arg])

        elif isinstance(arg, list) or isinstance(arg, np.ndarray):
            return self._get_various_tensor(arg)
        elif isinstance(arg, slice):
            return self._get_slice(i=arg)
        # we have indices for both dimensions
        if not isinstance(arg, tuple):
            raise IndexError("Invalid indexing information: %s" % arg)
        if len(arg) == 2:
            rows, cols = arg
            # returning a single element
            if isinstance(rows, int) and isinstance(cols, int):
                return self._get_single_tensor(i=rows, j=cols)
            elif isinstance(rows, int) and isinstance(cols, list):
                return self._get_various_tensor([rows], cols)
            elif isinstance(rows, list) and isinstance(cols, int):
                return self._get_various_tensor(rows, [cols])
            elif isinstance(rows, list) and isinstance(cols, list):
                return self._get_various_tensor(rows, cols)
            elif isinstance(rows, slice) and isinstance(cols, slice):
                return self._get_slice(rows, cols)
        else:
            rows = arg[0]
            cols = arg[1]
            positions = arg[2:]
            if isinstance(rows, int) and isinstance(cols, int) and \
                    isinstance(positions, tuple):
                return self._get_elements_from_tensors([rows], [cols],
                                                       positions)
            elif isinstance(rows, list) and isinstance(cols, list) and \
                    isinstance(positions, tuple):
                return self._get_elements_from_tensors(rows, cols,
                                                       positions)
            elif isinstance(rows, int) and isinstance(cols, list) and \
                    isinstance(positions, tuple):
                return self._get_elements_from_tensors([rows], cols,
                                                       positions)
            elif isinstance(rows, list) and isinstance(cols, int) and \
                    isinstance(positions, tuple):
                return self._get_elements_from_tensors(rows, [cols],
                                                       positions)
            elif isinstance(rows, slice) and isinstance(cols, slice) and \
                    isinstance(positions, tuple):
                return self._get_slice(rows, cols, positions)

    def __sub__(self, other):
        if self.shape != other.shape:
            raise ValueError("The number of tensors is different between the "
                             "two subtracted elements.")
        if self.tensor_shape != other.tensor_shape:
            raise ValueError("The tensors inside each object has different "
                             "dimensions.")

        tensors = []

        for htensors, othertensors in zip(self._iterator("rows"),
                                          other._iterator("rows")):
            out_tensors = [object() for _ in range(self.shape[1])]
            _combine_tensors(htensors.tensors, othertensors.tensors,
                             Tensor._subtract, out_tensors)
            tensors.append(out_tensors)

        return Tensor(tensors=tensors, tensor_shape=self.tensor_shape,
                      number_samples=self.number_samples,
                      dtype=self.dtype)

    def __add__(self, other):
        if self.shape != other.shape:
            raise ValueError("The number of tensors is different between the "
                             "two added elements.")
        if self.tensor_shape != other.tensor_shape:
            raise ValueError("The tensors inside each object has different "
                             "dimensions.")

        tensors = []

        for htensors, othertensors in zip(self._iterator("rows"),
                                          other._iterator("rows")):
            out_tensors = [object() for _ in range(self.shape[1])]
            _combine_tensors(htensors.tensors, othertensors.tensors,
                             Tensor._add, out_tensors)
            tensors.append(out_tensors)

        return Tensor(tensors=tensors, tensor_shape=self.tensor_shape,
                      number_samples=self.number_samples,
                      dtype=self.dtype)

    def __pow__(self, power, modulo=None):
        if not np.isscalar(power):
            raise NotImplementedError("Non scalar power not supported")
        return _apply_elementwise(Tensor._power, self, power)

    def __truediv__(self, other):
        if not np.isscalar(other):
            raise NotImplementedError("Non scalar division not supported")
        return _apply_elementwise(operator.truediv, self, other)

    def __mul__(self, other):
        if not np.isscalar(other):
            raise NotImplementedError("Non scalar multiplication not "
                                      "supported")
        return _apply_elementwise(operator.mul, self, other)

    def collect(self):
        self.tensors = compss_wait_on(self.tensors)
        return self.tensors

    @staticmethod
    def _subtract(a, b):
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return a - b
        elif torch.is_tensor(a) and torch.is_tensor(b):
            return torch.sub(a, b)

    @staticmethod
    def _add(a, b):
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return a + b
        elif torch.is_tensor(a) and torch.is_tensor(b):
            return torch.add(a, b)

    @staticmethod
    def _power(a, power):
        if isinstance(a, np.ndarray):
            return a ** power
        elif torch.is_tensor(a):
            return torch.pow(a, power)

    @staticmethod
    def _merge_tensors(tensors, axis_to_merge=0):
        """
        Helper function that merges the tensors attribute of a ds-tensor into
        a single ndarray / pytorch tensor
        """
        if isinstance(tensors[0][0], np.ndarray):
            return np.concatenate((tensors), axis=axis_to_merge)
        elif torch.is_tensor(tensors[0][0]):
            return torch.cat(*tensors, axis=axis_to_merge)
        raise ValueError("Type of tensors not supported.")

    @staticmethod
    def _get_out_tensors(n_tensors):
        """
        Helper function that builds empty lists of lists to be filled as
        parameter of type COLLECTION_OUT
        """
        return [[object() for _ in range(n_tensors[1])]
                for _ in range(n_tensors[0])]

    def _get_various_tensor(self, i=None, j=None):
        """
        Return the tensors in i rows or/and j cols
        """
        if i is not None and j is None:
            if any(index > self.shape[0] for index in i):
                raise IndexError("Rows shape is", self.shape[0])
            tensor = [self.tensors[idx] for idx in i]
            return Tensor(tensors=tensor,
                          tensor_shape=self.tensor_shape,
                          number_samples=len(i),
                          dtype=self.dtype)
        else:
            if len(i) == 1 and i[0] > self.shape[0]:
                raise IndexError("Shape is ", self.shape)
            if len(j) == 1 and j[0] > self.shape[1]:
                raise IndexError("Shape is ", self.shape)
            else:
                if any(index > self.shape[0] for index in i):
                    raise IndexError("Rows shape is ", self.shape[0])
                if any(index > self.shape[1] for index in j):
                    raise IndexError("Columns shape is ", self.shape[1])
                tensor = [[self.tensors[idx][idx2] for idx2 in j] for
                          idx in i]
                return Tensor(tensors=tensor,
                              tensor_shape=self.tensor_shape,
                              number_samples=len(i),
                              dtype=self.dtype)

    def _get_elements_from_tensors(self, i, j, positions):
        if len(i) == 1 and i[0] > self.shape[0]:
            raise IndexError("Shape is ", self.shape)
        if len(j) == 1 and j[0] > self.shape[1]:
            raise IndexError("Shape is ", self.shape)

        if any(index > self.shape[0] for index in i):
            raise IndexError("Rows shape is ", self.shape[0])
        if any(index > self.shape[1] for index in j):
            raise IndexError("Columns shape is ", self.shape[1])
        tensor = [[self.tensors[idx][idx2] for idx2 in j] for idx in i]
        if len(positions) > len(self.tensor_shape):
            raise IndexError("The index specified has more dimensions "
                             "than the tensor shape.")
        new_shape = []
        for dimension, dimension_index in enumerate(positions):
            if isinstance(dimension_index, slice):
                new_shape.append(dimension_index.stop - dimension_index.start)
            elif isinstance(dimension_index, list):
                new_shape.append(len(dimension_index))
            else:
                new_shape.append(1)
        definitive_tensors = []
        for tensor_row in tensor:
            tensor_out = []
            for tensor_col in tensor_row:
                tensor_out.append(_obtain_indexes_from_tensor(tensor_col,
                                                              positions))
            definitive_tensors.append(tensor_out)
        return Tensor(tensors=definitive_tensors,
                      tensor_shape=tuple(new_shape),
                      number_samples=len(i),
                      dtype=self.dtype)

    def _get_single_tensor(self, i, j):
        """
        Return the tensor in (i, j)
        """
        # we are returning a single element
        if i > self.shape[0] or j > self.shape[1]:
            raise IndexError("Shape is ", self.shape)
        tensor = self.tensors[i][j]

        # returns an list containing a single element
        # return tensor
        return Tensor(tensors=[[tensor]], tensor_shape=self.tensor_shape,
                      number_samples=1,
                      dtype=self.dtype)

    def _get_slice(self, i=None, j=None, inside_tensor=None):
        if (i.step is not None and i.step != 1):
            raise NotImplementedError("Variable steps not supported, contact"
                                      " the dislib team or open an issue "
                                      "in github.")
        # rows and cols are read-only
        r_start, r_stop = i.start, i.stop

        if r_start is None:
            r_start = 0

        if r_stop is None or r_stop > self.shape[0]:
            r_stop = self.tensor_shape[0]

        if r_start < 0 or r_stop < 0:
            raise NotImplementedError("Negative indexes not supported, contact"
                                      " the dislib team or open an issue "
                                      "in github.")
        n_rows = r_stop - r_start

        # If the slice is empty (no rows or no columns), return a ds-tensor
        # with
        # a single empty block. This empty block is required by the Tensor
        # constructor.
        if n_rows <= 0:
            empty_tensor = np.empty((0, 0))
            res = Tensor(tensors=[[empty_tensor]], tensor_shape=(0, 0),
                         number_samples=0,
                         dtype=self.dtype, delete=self._delete)
            return res
        if j is not None:
            if (j.step is not None and j.step != 1):
                raise NotImplementedError("Variable steps not supported, "
                                          "contact"
                                          " the dislib team or open an issue "
                                          "in github.")
            # rows and cols are read-only
            c_start, c_stop = j.start, j.stop

            if c_start is None:
                c_start = 0

            if c_stop is None or c_stop > self.shape[0]:
                c_stop = self.tensor_shape[0]

            if c_start < 0 or c_stop < 0:
                raise NotImplementedError("Negative indexes not supported, "
                                          "contact"
                                          " the dislib team or open an issue "
                                          "in github.")
            n_cols = c_stop - c_start

            # If the slice is empty (no rows or no columns), return a
            # ds-tensor with
            # a single empty block. This empty block is required by the Tensor
            # constructor.
            if n_cols <= 0:
                empty_tensor = np.empty((0, 0))
                res = Tensor(tensors=[[empty_tensor]], tensor_shape=(0, 0),
                             number_samples=0,
                             dtype=self.dtype, delete=self._delete)
                return res
        else:
            c_start = 0
            c_stop = self.shape[1]
        tensors = [tens[c_start:c_stop] for tens in
                   self.tensors[r_start:r_stop]]
        if inside_tensor is not None:
            new_shape = []
            for dimension, specific_slice in enumerate(inside_tensor):
                if isinstance(specific_slice, slice):
                    new_shape.append(specific_slice.stop -
                                     specific_slice.start)
                elif isinstance(specific_slice, list):
                    new_shape.append(len(specific_slice))
                    break
                else:
                    new_shape.append(1)
            definitive_tensors = []
            for tensor_row in tensors:
                tensor_out = []
                for tensor_col in tensor_row:
                    tensor_out.append(_obtain_indexes_from_tensor(
                        tensor_col, inside_tensor))
                definitive_tensors.append(tensor_out)
            return Tensor(tensors=definitive_tensors,
                          tensor_shape=tuple(new_shape),
                          number_samples=n_rows,
                          dtype=self.dtype, delete=self._delete)
        else:
            return Tensor(tensors=tensors, tensor_shape=self.tensor_shape,
                          number_samples=n_rows,
                          dtype=self.dtype, delete=self._delete)

    def _iterator(self, axis=0):
        # iterate through rows
        if axis == 0 or axis == 'rows':
            for i in range(self.shape[0]):
                yield self._get_various_tensor([i])
        # iterate through columns
        elif axis == 1 or axis == 'columns':
            for j in range(self.shape[1]):
                yield self._get_various_tensor(i=[*range(self.shape[0])],
                                               j=[j])
        else:
            raise Exception(
                "Axis must be [0|'rows'] or [1|'columns']. Got: %s" % axis)

    def apply_to_tensors(self, func):
        """
        Applies the specified function to the all the tensors
        """

        out_tensors = [[object() for _ in range(self.shape[1])]
                       for _ in range(self.shape[0])]

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                out_tensors[i][j] = apply_to_tensor(self.tensors[i][j], func)
        if func == np.transpose or func == torch.transpose:
            tensor_shape = self.tensor_shape[::-1]
        else:
            tensor_shape = self.tensor_shape
        return Tensor(tensors=out_tensors, tensor_shape=tensor_shape,
                      number_samples=self.number_samples,
                      dtype=self.dtype)


def from_array(np_array, shape=None):
    """
    Creates from a numpy array the Tensor object, the numpy array should have
    at least 3 dimensions, the first two for the lists of the tensors and the
    last one (at least)
    as data inside each of the tensors.

    Parameters
    ----------
    np_array : np.array
        Numpy array that contains the data.
    shape : tuple of two ints
        Shape of the output ds-tensor.

    Returns
    -------
    x : ds-tensor
    """
    if not isinstance(np_array, np.ndarray):
        raise ValueError("The method expects to receive a numpy array.")
    if shape:
        if len(shape) == 2 and shape[0] >= 1 and shape[1] >= 1:
            if shape[0] * shape[1] > len(np_array):
                raise ValueError("The number of tensors specified is higher "
                                 "than the number of elements")
            elements_per_tensor = (int(np_array.shape[0] /
                                       (shape[0]*shape[1])))
            new_tensors = _place_elements_in_tensors(np_array, shape,
                                                     elements_per_tensor)
            elements_per_tensor = (elements_per_tensor, *np_array.shape[1:])
            return Tensor(tensors=new_tensors,
                          tensor_shape=elements_per_tensor,
                          shape=tuple(shape),
                          number_samples=np_array.shape[0],
                          dtype=np_array.dtype)
        raise ValueError("The shape should contain two values greater "
                         "than 0")
    else:
        shape_ds_tensor = (np_array.shape[0], np_array.shape[1])
        new_tensors = Tensor._get_out_tensors(shape_ds_tensor)
        _reallocate_tensors(np_array, new_tensors)
        return Tensor(tensors=new_tensors, tensor_shape=np_array.shape[2:],
                      shape=shape_ds_tensor,
                      number_samples=np_array.shape[0],
                      dtype=np_array.dtype)


def from_pt_tensor(tensor, shape=None):
    """
    Creates from a pytorch tensor the Tensor object, the pytorch tensor should
    have at least 3 dimensions, the first two for the lists of the tensors and
    the last one (at least)
    as data inside each of the tensors.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor that contains the data.
    shape : tuple of two ints
        Shape of the output ds-tensor.

    Returns
    -------
    x : ds-tensor
    """
    if not torch.is_tensor(tensor):
        raise ValueError("The method expects to receive a pytorch tensor.")
    if shape:
        if len(shape) == 2 and shape[0] >= 1 and shape[1] >= 1:
            if shape[0] * shape[1] > len(tensor):
                raise ValueError("The number of tensors specified is higher "
                                 "than the number of elements")
            elements_per_tensor = (int(tensor.shape[0] / (shape[0]*shape[1])))
            new_tensors = _place_elements_in_tensors(tensor, shape,
                                                     elements_per_tensor)
            elements_per_tensor = (elements_per_tensor, *tensor.shape[1:])
            return Tensor(tensors=new_tensors,
                          tensor_shape=elements_per_tensor,
                          number_samples=shape[0],
                          shape=tuple(shape), dtype=tensor.dtype)
        raise ValueError("The shape should contain two values greater "
                         "than 0")
    else:
        shape_ds_tensor = (tensor.shape[0], tensor.shape[1])
        new_tensors = Tensor._get_out_tensors(shape_ds_tensor)
        _reallocate_tensors(tensor, new_tensors)
        return Tensor(tensors=new_tensors, tensor_shape=tensor.shape[2:],
                      number_samples=shape_ds_tensor[0],
                      shape=shape_ds_tensor,
                      dtype=tensor.dtype)


def from_ds_array(ds_array, shape=None):
    """
    Creates the Tensor object from a ds_array.
    This method can't generated Tensors that have more than two dimensions,
    thus the output of this method can't be used in a Convolutional Neural
    Network neither any Neural Network that requires data input with
    more than two dimensions.

    Parameters
    ----------
    ds_array : ds-array
        The ds-array to transform into ds-tensor.
    shape : tuple of two ints.
        The organization of the number of tensors,
        how many will be on axis=0 and how many will be on axis=1.
        The total number of tensors should be the same as blocks
        are in the input ds-array.

    Returns
    -------
    tensor : ds-tensor
    """
    if not isinstance(ds_array, Array):
        raise TypeError("The method expects to receive a ds-array.")
    if shape is not None and (isinstance(shape, tuple) or
                              isinstance(shape, list)):
        if len(shape) != 2:
            raise ValueError("The shape specified should have exactly "
                             "two dimensions.")
        if int(shape[0] * shape[1]) != \
           int(ds_array._n_blocks[0] * ds_array._n_blocks[1]):
            raise ValueError("The number of tensors specified in the "
                             "shape should be equal to the number of "
                             "blocks in the input ds-array")
    new_tensor = Tensor._get_out_tensors([ds_array._n_blocks[0],
                                          ds_array._n_blocks[1]])
    for block_i, idx_i in zip(ds_array._blocks, range(len(new_tensor))):
        for block_j, idx_j in zip(block_i, range(len(new_tensor[idx_i]))):
            new_tensor[idx_i][idx_j] = _assign_blocks_to_tensors(block_j)
    if shape is not None and (isinstance(shape, tuple) or
                              isinstance(shape, list)):
        return change_shape(Tensor(tensors=new_tensor,
                            tensor_shape=(ds_array._reg_shape[0],
                                          ds_array.shape[1]),
                            number_samples=ds_array.shape[0],
                            dtype=np.float64, delete=ds_array._delete), shape)
    else:
        return Tensor(tensors=new_tensor,
                      tensor_shape=(ds_array._reg_shape[0],
                                    ds_array.shape[1]),
                      number_samples=ds_array.shape[0],
                      dtype=np.float64, delete=ds_array._delete)


def cat(tensors, dimension):
    """
    Concatenates the tensors inside the tensors list using the specified
    dimension

    Parameters
    ----------
    tensors : List
        List containing the tensors to concatenate.
    dimension: int
        Dimension to use in the concatenation.

    Returns
    -------
    x : ds-tensor
    """
    if len(tensors) <= 1:
        raise ValueError("There should be at least two tensors to "
                         "concatenate")
    shape = tensors[0].shape
    if any(tensor.shape != shape for tensor in tensors):
        raise ValueError(
            "All ds-tensors should contain the same number of tensor in order"
            " to be possible to concatenate them")
    tensor_shape = (*tensors[0].tensor_shape[:dimension],
                    *tensors[0].tensor_shape[dimension + 1:])
    if any((*tensor.tensor_shape[:dimension],
            *tensor.tensor_shape[dimension + 1:]) != tensor_shape for tensor
           in tensors):
        raise ValueError("All tensors should have the same shape except in "
                         "the concatenate dimension")
    cat_tensors = []
    new_dimension_shape = 0
    for tens in tensors:
        new_dimension_shape = new_dimension_shape + \
                              tens.tensor_shape[dimension]
    for i in range(shape[0]):
        cat_rows = []
        for j in range(shape[1]):
            tensors_to_concat = [tens.tensors[i][j] for tens in tensors]
            cat_rows.append(concatenate(tensors_to_concat, dimension))
        cat_tensors.append(cat_rows)
    tensor_shape = list(tensors[0].tensor_shape)
    tensor_shape[dimension] = new_dimension_shape
    return Tensor(tensors=cat_tensors, tensor_shape=tuple(tensor_shape),
                  shape=(len(cat_tensors), len(cat_tensors[0])),
                  number_samples=(len(cat_tensors) + len(cat_tensors[0])),
                  dtype=tensors[0].dtype)


def change_shape(tensor, new_shape):
    """
    Changes the distribution of the tensors in the ds-tensor, modifying its
    shape. For example a ds-tensor with shape (2, 2)
    may be changed to (4, 1) or to (1, 4)

    Parameters
    ----------
    tensor : ds-tensor
        ds-tensor where the shape is going to be modified
    new_shape: tuple of two ints
        Shape that the tensor will have after the modification

    Returns
    -------
    x : ds-tensor
    """
    if tensor.shape[0] * tensor.shape[1] != new_shape[0] * new_shape[1]:
        raise ValueError("Incompatible number of tensors in both shapes.")
    new_tensors = Tensor._get_out_tensors(new_shape)
    _reallocate_tensors(tensor.tensors, new_tensors)
    return Tensor(tensors=new_tensors, tensor_shape=tensor.tensor_shape,
                  number_samples=tensor.number_samples,
                  dtype=tensor.dtype)


def _empty_tensor(shape, tensor_shape, dtype=np.float64):
    return Tensor(tensors=[[None for _ in range(shape[1])] for _ in
                           range(shape[0])], tensor_shape=tensor_shape,
                  number_samples=0,
                  shape=shape, dtype=dtype)


def rechunk_tensor(tensor, new_tensors_shape, dimension=0):
    """
    Changes the shape of the tensors inside the ds-tensor.
    The number of tensors, and at the same time the shape of
    the ds-tensor, will be modified
    in order to fit the total number of elements with the new
    shape of each tensor.

    Parameters
    ----------
    tensor : ds-tensor
        ds-tensor which tensors will be modified
    new_tensors_shape: int
        Shape that each of the tensors will have in the specified
        dimension after the rechunk
    dimension: int
        Dimension of the tensors where the change of the shape is
        going to be applied

    Returns
    -------
    x : ds-tensor
    """
    new_number_of_tensors = math.ceil((tensor.tensor_shape[dimension] /
                                       new_tensors_shape) * tensor.shape[1])
    tensors_shape_new = []
    for i, _ in enumerate(tensor.tensor_shape):
        if i == dimension:
            tensors_shape_new.append(new_tensors_shape)
        else:
            tensors_shape_new.append(tensor.tensor_shape[i])
    tensors = []
    for row in tensor.tensors:
        tensors_row = [object() for _ in range(new_number_of_tensors)]
        _rechunk_tensor_row(row, tensors_row, dimension, new_tensors_shape)
        tensors.append(tensors_row)
    return Tensor(tensors=tensors, tensor_shape=tuple(tensors_shape_new),
                  number_samples=tensor.number_samples,
                  dtype=tensor.dtype)


def _apply_elementwise(func, x, *args, **kwargs):
    """ Applies a function element-wise to each tensor in parallel"""
    n_tensors = x.shape
    tensors = Tensor._get_out_tensors(n_tensors)
    for i in range(n_tensors[0]):
        for j in range(n_tensors[1]):
            tensors[i][j] = _tensor_apply(func, x.tensors[i][j], *args,
                                          **kwargs)

    return Tensor(tensors, tensor_shape=x.tensor_shape,
                  number_samples=x.number_samples,
                  dtype=x.dtype)


@constraint(computing_units="${ComputingUnits}")
@task(returns=1)
def apply_to_tensor(tensor, func):
    """
    Applies the specified function to the tensor in (i, j)
    """
    return func(tensor)


@constraint(computing_units="${ComputingUnits}")
@task(old_tensor={Type: COLLECTION_IN}, new_tensor={Type: COLLECTION_OUT})
def _rechunk_tensor_row(old_tensor, new_tensor, dimension, new_tensors_shape):
    index_new_tensor = 0
    to_next_tensor = None
    for tensor in old_tensor:
        tensors_splitted = np.array_split(tensor, [new_tensors_shape *
                                                   (i + 1) for i in
                                                   range(math.floor(
                                                       tensor.shape[
                                                           dimension] /
                                                       new_tensors_shape))],
                                          axis=dimension)
        for splitted_tensor in tensors_splitted:
            if to_next_tensor is not None:
                to_next_tensor = np.concatenate((to_next_tensor,
                                                 splitted_tensor))
                if to_next_tensor.shape[dimension] > new_tensors_shape:
                    new_split = np.array_split(to_next_tensor,
                                               [new_tensors_shape],
                                               axis=dimension)
                    new_tensor[index_new_tensor] = new_split[0]
                    to_next_tensor = new_split[1]
                    index_new_tensor = index_new_tensor + 1
                    if to_next_tensor.shape[dimension] == new_tensors_shape:
                        new_tensor[index_new_tensor] = to_next_tensor
                        to_next_tensor = None
                        index_new_tensor = index_new_tensor + 1
                elif to_next_tensor.shape[dimension] == new_tensors_shape:
                    new_tensor[index_new_tensor] = to_next_tensor
                    index_new_tensor = index_new_tensor + 1
                    to_next_tensor = None
                else:
                    to_next_tensor = np.concatenate((to_next_tensor,
                                                     splitted_tensor),
                                                    axis=dimension)
            elif splitted_tensor.shape[dimension] == new_tensors_shape:
                new_tensor[index_new_tensor] = splitted_tensor
                index_new_tensor = index_new_tensor + 1
            else:
                to_next_tensor = splitted_tensor
    if to_next_tensor is not None:
        if to_next_tensor.shape[dimension] > new_tensors_shape:
            tensors_splitted = np.array_split(to_next_tensor,
                                              [new_tensors_shape],
                                              axis=dimension)
            for splitted_tensor in tensors_splitted:
                new_tensor[index_new_tensor] = splitted_tensor
                index_new_tensor = index_new_tensor + 1
        elif to_next_tensor.shape[dimension] > 0:
            new_tensor[index_new_tensor] = splitted_tensor
            index_new_tensor = index_new_tensor + 1


def create_ds_tensor(tensors, tensors_shape, shape, dtype=np.float64):
    """
    Function to create a ds-tensor from a list of lists of pytorch tensors
    or numpy arrays. If specified the shape it should match the number of
    elements in the lists
    respectively.

    Parameters
    ----------
    tensors : List of lists containing numpy arrays or pytorch tensors
        Should contain the tensors of the object.
    tensors_shape : tuple of int
        Shape of the regular tensors in the list.
    shape : tuple of int
        Shape of the object, it represents the number of tensors in both
        dimensions of the list.
    dtype : String
        Type of the data inside the tensors.

    Returns
    -------
    x : ds-tensor
    """
    if shape is not None and (len(tensors) != shape[0] or
                              len(tensors[0]) != shape[1]):
        raise ValueError("The ds-tensor shape should match the number of "
                         "tensors it contains in each direction.")
    return Tensor(tensors=tensors, tensor_shape=tensors_shape,
                  shape=shape,
                  number_samples=tensors_shape[0] * shape[0] * shape[1],
                  dtype=dtype)


def random_tensors(tensors_type, shape, dtype=None):
    """
    Function that generates a ds-tensor with random data inside it.

    Parameters
    ----------
    tensor_type : String
        Type of the tensors used, could be numpy "np" or pytorch "torch"
    shape : tuple of int
        Shape of the object, the first two numbers will be used as
        the number of column tensors and row tensors
        respectively.
    dtype : String
        Type of the data inside the tensors.

    Returns
    -------
    x : ds-tensor
    """
    return _random_tensor_wrapper(tensors_type, shape, dtype)


def _random_tensor_wrapper(tensor_type, shape, dtype=None):
    """
    Function that generates a ds-tensor with random data inside it.

    Parameters
    ----------
    tensor_type : String
        Type of the tensors used, could be numpy "np" or pytorch "torch"
    shape : tuple of int
        Shape of the object, the first two numbers will be used as the
        number of column tensors and row tensors
        respectively.
    dtype : String
        Type of the data inside the tensors.

    Returns
    -------
    x : ds-tensor
    """
    if tensor_type == "np":
        tensors = []
        for _ in range(shape[0]):
            col_tensor = []
            for _ in range(shape[1]):
                col_tensor.append(_create_random_tensor(np.random.rand,
                                                        shape[2:]))
            tensors.append(col_tensor)
        if dtype is None:
            dtype = np.float64
        return Tensor(tensors=tensors, tensor_shape=shape[2:],
                      number_samples=shape[0] * shape[1] * shape[3],
                      dtype=dtype)
    elif tensor_type == "torch":
        tensors = []
        for _ in range(shape[0]):
            col_tensor = []
            for _ in range(shape[1]):
                col_tensor.append(_create_random_tensor(torch.rand,
                                                        shape[2:]))
            tensors.append(col_tensor)
        if dtype is None:
            dtype = torch.float64
        return Tensor(tensors=tensors, tensor_shape=shape[2:],
                      number_samples=shape[0] * shape[1] * shape[3],
                      dtype=dtype)
    else:
        raise NotImplementedError("Type of tensor not supported")


def load_dataset(number_tensors_per_file, path):
    """
    Function to load data from files, these files can be numpy file ".npy" or
    pytorch files ".pt". Depending on extension of the file
    to load the data from the ds-tensor generated will contain numpy arrays as
    tensors or pytorch tensors.

    Parameters
    ----------
    number_tensors_per_file : int
        Number of tensors to load from each file
    path : String
        Path to the directory where the files with the data are located

    Returns
    -------
    x : ds-tensor
    """
    files = os.listdir(path)
    format_file = files[0][-4:]
    if format_file == '.npy':
        tensors = []
        x_train = np.load(path + "/" + files[0])
        elements_per_tensor = math.ceil(x_train.shape[0] /
                                        number_tensors_per_file)
        tensors_row = [object() for _ in range(number_tensors_per_file)]
        _read_tensor_from_npy(path + "/" + files[0], tensors_row,
                              elements_per_tensor)
        tensors.append(tensors_row)
        for file in files[1:]:
            tensors_row = [object() for _ in range(number_tensors_per_file)]
            _read_tensor_from_npy(path + "/" + file, tensors_row,
                                  elements_per_tensor)
            tensors.append(tensors_row)
        return Tensor(tensors=tensors, tensor_shape=(elements_per_tensor,
                                                     *x_train.shape[1:]),
                      number_samples=x_train.shape[0],
                      dtype="np." + str(x_train.dtype))
    else:
        format_file = files[0][-3:]
        if format_file == '.pt':
            tensors = []
            x_train = torch.load(path + "/" + files[0])
            elements_per_tensor = math.ceil(x_train.shape[0] /
                                            number_tensors_per_file)
            tensors_row = [object() for _ in range(number_tensors_per_file)]
            _read_tensor_from_pt(path + "/" + files[0],
                                 tensors_row, elements_per_tensor)
            tensors.append(tensors_row)
            for file in files[1:]:
                tensors_row = [object() for _ in
                               range(number_tensors_per_file)]
                _read_tensor_from_pt(path + "/" + file, tensors_row,
                                     elements_per_tensor)
                tensors.append(tensors_row)
            return Tensor(tensors=tensors, tensor_shape=(elements_per_tensor,
                                                         *x_train.shape[1:]),
                          number_samples=x_train.shape[0],
                          dtype="torch." + str(x_train.dtype))
        else:
            raise NotImplementedError("Only supported numpy arrays or pytorch"
                                      " tensors")


def shuffle(x, y=None, random_state=None):
    """
    Shuffles randomly the data contained inside the tensors.

    Parameters
    ----------
    x : ds-tensor
        ds-tensor to be shuffled
    y : ds-tensor
        ds-tensor to be shuffled in the same one as x
    random_state : int
        Seed that will be used in the random state functions and np.random.

    Returns
    -------
    x : ds-tensor
    y : ds-tensor or None
    """
    if y is not None:
        assert y.shape[0] == x.shape[0] and y.shape[1] == x.shape[1]
    np.random.seed(random_state)
    if random_state is None:
        random_state = 0

    permutation = np.random.permutation(x.n_tensors)
    last_blocks_irregular = (x.number_samples % x.tensor_shape[0] != 0)
    if y is None:
        tensors = [object() for _ in range(x.n_tensors)]
        if len(permutation) % 2 == 0:
            number_elements_permutation_to_use = len(permutation)
        else:
            number_elements_permutation_to_use = len(permutation) - 1
        auxiliar_indexes = []
        for i in range(0, number_elements_permutation_to_use, 2):
            last_row_ = (((permutation[i] % x.shape[0] + 1) == x.shape[0])
                         and last_blocks_irregular)
            last_row_second = (((permutation[i + 1] % x.shape[0] + 1) ==
                                x.shape[0])
                               and last_blocks_irregular)
            tensor_out_1, tensor_out_2 = _merge_shuffle_tensors(
                x.tensors[permutation[i] % x.shape[0]][
                    math.floor(permutation[i]
                               / x.shape[0])],
                x.tensors[permutation[i + 1] % x.shape[0]][
                    math.floor(permutation[i + 1]
                               / x.shape[0])],
                last_tensor=last_row_,
                second_last_tensor=last_row_second,
                random_state=random_state + i)
            tensors[i] = tensor_out_1
            tensors[i+1] = tensor_out_2
            if last_row_ and last_blocks_irregular:
                auxiliar_indexes.append(i)
            if last_row_second and last_blocks_irregular:
                auxiliar_indexes.append(i+1)
        if not len(permutation) % 2 == 0:
            tensor_out = _shuffle_last_tensor(x.tensors[
                                                  permutation[-1] % x.shape[0]]
                                              [math.floor(
                                               permutation[-1]/x.shape[0])],
                                              random_state=random_state)
            tensors[-1] = tensor_out
        tensors_auxiliar = [[object() for _ in range(x.shape[1])]
                            for _ in range(x.shape[0])]
        final_indexes = list(range(len(permutation)))
        for index in auxiliar_indexes:
            final_indexes.remove(index)
        if not len(permutation) % 2 == 0:
            final_indexes = (final_indexes[:-1] + auxiliar_indexes +
                             [final_indexes[-1]])
        else:
            final_indexes = final_indexes + auxiliar_indexes
        for idx, i in enumerate(final_indexes):
            tensors_auxiliar[idx % x.shape[0]][math.floor(idx/x.shape[0])] = (
                tensors)[i]
        return Tensor(tensors=tensors_auxiliar, shape=x.shape,
                      tensor_shape=x.tensor_shape,
                      number_samples=x.number_samples, dtype=x.dtype)
    else:
        tensors = [object() for _ in range(x.n_tensors)]
        y_tensors = [object() for _ in range(y.n_tensors)]
        if len(permutation) % 2 == 0:
            number_elements_permutation_to_use = len(permutation)
        else:
            number_elements_permutation_to_use = len(permutation) - 1
        auxiliar_indexes = []
        for i in range(0, number_elements_permutation_to_use, 2):
            last_row_ = (((math.floor(permutation[i] /
                                      x.shape[0]) + 1) == x.shape[0])
                         and last_blocks_irregular)
            last_row_second = (((math.floor(permutation[i + 1] /
                                            x.shape[0]) + 1) == x.shape[0])
                               and last_blocks_irregular)
            tensor_x_1, tensor_x_2, tensor_y_1, tensor_y_2 = (
                _merge_shuffle_tensors_xy(x.tensors[
                                              permutation[i] % x.shape[0]]
                                          [math.floor(permutation[i]
                                                      / x.shape[0])],
                                          x.tensors[
                                              permutation[i + 1] %
                                              x.shape[0]][math.floor(
                                                          permutation[i + 1]
                                                          / x.shape[0])],
                                          y.tensors[
                                              permutation[i] % y.shape[0]]
                                          [math.floor(permutation[i]
                                                      / y.shape[0])],
                                          y.tensors[
                                              permutation[i + 1] %
                                              y.shape[0]][math.floor(
                                                          permutation[i + 1]
                                                          / y.shape[0])],
                                          last_tensor=last_row_,
                                          second_last_tensor=last_row_second,
                                          random_state=random_state + i))
            tensors[i] = tensor_x_1
            y_tensors[i] = tensor_y_1
            tensors[i + 1] = tensor_x_2
            y_tensors[i + 1] = tensor_y_2
            if last_row_ and last_blocks_irregular:
                auxiliar_indexes.append(i)
            if last_row_second and last_blocks_irregular:
                auxiliar_indexes.append(i+1)
        if not len(permutation) % 2 == 0:
            tensor_out, tensor_out_y = _shuffle_last_tensor_xy(
                x.tensors[permutation[-1] % x.shape[0]]
                [math.floor(permutation[-1]/x.shape[0])],
                y.tensors[permutation[-1] % y.shape[0]]
                [math.floor(permutation[-1]/y.shape[0])],
                random_state=random_state)
            tensors[-1] = tensor_out
            y_tensors[-1] = tensor_out_y
        tensors_x = [[object() for _ in range(x.shape[1])]
                     for _ in range(x.shape[0])]
        tensors_y = [[object() for _ in range(x.shape[1])]
                     for _ in range(x.shape[0])]
        final_indexes = list(range(len(permutation)))
        for index in auxiliar_indexes:
            final_indexes.remove(index)
        if not len(permutation) % 2 == 0:
            final_indexes = (final_indexes[:-1] +
                             auxiliar_indexes + [final_indexes[-1]])
        else:
            final_indexes = final_indexes + auxiliar_indexes
        for idx, i in enumerate(final_indexes):
            tensors_x[idx % x.shape[0]][math.floor(idx/x.shape[0])] = (
                tensors)[i]
            tensors_y[idx % y.shape[0]][math.floor(idx/y.shape[0])] = (
                y_tensors)[i]

    return (Tensor(tensors=tensors_x, shape=x.shape,
                   tensor_shape=x.tensor_shape,
                   number_samples=x.number_samples, dtype=x.dtype),
            Tensor(tensors=tensors_y, shape=y.shape,
                   tensor_shape=y.tensor_shape,
                   number_samples=y.number_samples, dtype=y.dtype))


@constraint(computing_units="${ComputingUnits}")
@task(returns=2)
def _merge_shuffle_tensors(first_tensor, second_tensor,
                           last_tensor=False,
                           second_last_tensor=False,
                           random_state=None):
    number_elements = 0
    if last_tensor:
        number_elements = first_tensor.shape[0]
    if second_last_tensor:
        number_elements = second_tensor.shape[0]
    if isinstance(first_tensor, np.ndarray):
        merged_tensors = np.vstack([first_tensor, second_tensor])
        np.random.seed(random_state)
        p = np.random.permutation(len(merged_tensors))
        merged_tensors = merged_tensors[p]
    elif torch.is_tensor(first_tensor):
        merged_tensors = torch.cat([first_tensor, second_tensor])
        idx = torch.randperm(merged_tensors.shape[0])
        merged_tensors = merged_tensors[idx]
    else:
        raise TypeError("This type of tensor is not supported.")
    if last_tensor:
        return (merged_tensors[:number_elements],
                merged_tensors[number_elements:])
    else:
        return (merged_tensors[:first_tensor.shape[0]],
                merged_tensors[first_tensor.shape[0]:])


@constraint(computing_units="${ComputingUnits}")
@task(returns=1)
def _shuffle_last_tensor(tensor, random_state=None):
    if isinstance(tensor, np.ndarray):
        np.random.seed(random_state)
        p = np.random.permutation(len(tensor))
        tensor = tensor[p]
    elif torch.is_tensor(tensor):
        idx = torch.randperm(tensor.shape[0])
        tensor = tensor[idx]
    else:
        raise TypeError("This type of tensor is not supported.")
    return tensor


@constraint(computing_units="${ComputingUnits}")
@task(returns=4)
def _merge_shuffle_tensors_xy(x_tensor, x_tensor2, y_tensor, y_tensor2,
                              last_tensor=False,
                              second_last_tensor=False,
                              random_state=None):
    number_elements = 0
    if last_tensor:
        number_elements = x_tensor.shape[0]
    if second_last_tensor:
        number_elements = x_tensor2.shape[0]
    if isinstance(x_tensor, np.ndarray):
        merged_tensors = np.vstack([x_tensor, x_tensor2])
        merged_tensors_y = np.vstack([y_tensor, y_tensor2])
        np.random.seed(random_state)
        p = np.random.permutation(len(merged_tensors))
        merged_tensors = merged_tensors[p]
        merged_tensors_y = merged_tensors_y[p]
    elif torch.is_tensor(x_tensor):
        merged_tensors = torch.cat([x_tensor, x_tensor2])
        merged_tensors_y = torch.cat([y_tensor, y_tensor2])
        idx = torch.randperm(merged_tensors.shape[0])
        merged_tensors = merged_tensors[idx]
        merged_tensors_y = merged_tensors_y[idx]
    else:
        raise TypeError("This type of tensor is not supported.")
    if last_tensor:
        return (merged_tensors[:number_elements],
                merged_tensors[number_elements:],
                merged_tensors_y[:number_elements],
                merged_tensors_y[number_elements:])
    else:
        return (merged_tensors[:x_tensor.shape[0]],
                merged_tensors[x_tensor.shape[0]:],
                merged_tensors_y[:x_tensor.shape[0]],
                merged_tensors_y[x_tensor.shape[0]:])


@constraint(computing_units="${ComputingUnits}")
@task(returns=2)
def _shuffle_last_tensor_xy(tensor, tensor_y, random_state=None):
    if isinstance(tensor, np.ndarray):
        np.random.seed(random_state)
        p = np.random.permutation(len(tensor))
        tensor = tensor[p]
        tensor_y = tensor_y[p]
    elif torch.is_tensor(tensor):
        idx = torch.randperm(tensor.shape[0])
        tensor = tensor[idx]
        tensor_y = tensor_y[idx]
    else:
        raise TypeError("This type of tensor is not supported.")
    return tensor, tensor_y


@constraint(computing_units="${ComputingUnits}")
@task(returns=1)
def _assign_blocks_to_tensors(block):
    return torch.from_numpy(block)


@constraint(computing_units="${ComputingUnits}")
@task(tensor=INOUT)
def _set_value(tensor, key, value):
    tensor[key] = value


@constraint(computing_units="${ComputingUnits}")
@task(returns=1)
def _tensor_apply(func, tensor, *args, **kwargs):
    return func(tensor, *args, **kwargs)


@task(file_path=FILE_IN, tensors={Type: COLLECTION_OUT, Depth: 1})
def _read_tensor_from_npy(file_path, tensors, elements_per_tensor):
    x_train = np.load(file_path)
    for i in range(len(tensors)):
        tensors[i] = x_train[elements_per_tensor * i:
                             elements_per_tensor * (i + 1)]


@task(file_path=FILE_IN, tensors={Type: COLLECTION_OUT, Depth: 1})
def _read_tensor_from_pt(file_path, tensors, elements_per_tensor):
    x_train = torch.load(file_path)
    for i in range(len(tensors)):
        tensors[i] = x_train[elements_per_tensor * i:
                             elements_per_tensor * (i + 1)]


@constraint(computing_units="${ComputingUnits}")
@task(returns=1)
def _create_random_tensor(func, shape):
    shape = tuple(shape)
    return func(*shape)


def _place_elements_in_tensors(tensors, shape, elements_per_tensor):
    out_tensors = []
    for i in range(shape[0]):
        out_tensor = []
        for j in range(shape[1]):
            out_tensor.append(_get_specific_elements_tensor(
                tensors, elements_per_tensor, i, j))
        out_tensors.append(out_tensor)
    return out_tensors


@constraint(computing_units="${ComputingUnits}")
@task(returns=1)
def _get_specific_elements_tensor(tensors, elements_per_tensor, i, j):
    return tensors[int(i * elements_per_tensor):
                   int(i * elements_per_tensor + elements_per_tensor)]


@constraint(computing_units="${ComputingUnits}")
@task(tensors={Type: COLLECTION_IN, Depth: 3},
      out_tensors={Type: COLLECTION_OUT, Depth: 2})
def _reallocate_tensors(tensors, out_tensors):
    initial_shape = (len(tensors), len(tensors[0]))
    new_shape = (len(out_tensors), len(out_tensors[0]))
    i2 = 0
    j2 = 0
    for i in range(initial_shape[0]):
        for j in range(initial_shape[1]):
            out_tensors[i2][j2] = tensors[i][j]
            j2 = j2 + 1
            if j2 == new_shape[1]:
                j2 = 0
                i2 = i2 + 1


@constraint(computing_units="${ComputingUnits}")
@task(tensors={Type: COLLECTION_IN},
      othertensors={Type: COLLECTION_IN},
      out_tensors={Type: COLLECTION_OUT})
def _combine_tensors(tensors, othertensors, func, out_tensors):
    x = Tensor._merge_tensors(tensors)
    y = Tensor._merge_tensors(othertensors)

    res = func(x, y)
    for i in range(len(out_tensors)):
        out_tensors[i] = res[i: (i + 1)]


@constraint(computing_units="${ComputingUnits}")
@task(new_shape={Type: COLLECTION_IN}, returns=1)
def _obtain_indexes_from_tensor(tensor, new_shape):
    if isinstance(new_shape, list):
        return tensor[tuple(new_shape)]
    return tensor[new_shape]


@constraint(computing_units="${ComputingUnits}")
@task(tensors_to_concat={Type: COLLECTION_IN}, returns=1)
def concatenate(tensors_to_concat, dimension):
    if isinstance(tensors_to_concat[0], np.ndarray):
        return np.concatenate(tensors_to_concat, dimension)
    return torch.cat(tensors_to_concat, dimension)
