from pycompss.api.api import compss_delete_object
from pycompss.api.constraint import constraint
from pycompss.api.parameter import IN
from pycompss.api.task import task

from dislib.data.array import Array

import numpy as np


def pad(a: Array, pad_width, **kwargs):
    """
        Pad array blocks with the desired value.
        Parameters
        ----------
        a : array_like of rank N
            The array to pad.
        pad_width : ((top, bottom), (left, right))
            Number of values padded to the edges of each axis.
        constant_value : scalar, optional
            The value to set in the padded rows and columns.
            Default is 0.
    """
    pad_top = pad_width[0][0]
    pad_bottom = pad_width[0][1]
    pad_left = pad_width[1][0]
    pad_right = pad_width[1][1]

    if all(pad_elem == 0 for pad_elem in [
        pad_top, pad_bottom, pad_left, pad_right
    ]):
        return

    if pad_top != 0:
        raise NotImplementedError("Padding top blocks is currently "
                                  "not available")

    if pad_left != 0:
        raise NotImplementedError("Padding left blocks is currently "
                                  "not available")

    bottom_right_shape = compute_bottom_right_shape(a)
    if pad_bottom + bottom_right_shape[0] + pad_top > a._reg_shape[0]:
        raise NotImplementedError("Adding new row blocks is currently not "
                                  "available. Make sure that the new content "
                                  "does not exceed the regular block size.")

    if pad_left + bottom_right_shape[1] + pad_right > a._reg_shape[1]:
        raise NotImplementedError("Adding new column blocks is currently not "
                                  "available. Make sure that the new content "
                                  "does not exceed the regular block size.")

    fill_value = kwargs.get('constant_value', 0)

    for row_block_idx in range(a._n_blocks[0]):
        padded_block = _pad_right_block(
            a._blocks[row_block_idx][-1],
            pad_right,
            fill_value
        )
        a.replace_block(row_block_idx, a._n_blocks[1] - 1, padded_block)

    for col_block_idx in range(a._n_blocks[1]):
        padded_block = _pad_bottom_block(
            a._blocks[-1][col_block_idx],
            pad_bottom,
            fill_value
        )
        a.replace_block(a._n_blocks[0] - 1, col_block_idx, padded_block)

    a._shape = (pad_bottom + a.shape[0] + pad_top,
                pad_left + a.shape[1] + pad_right)

    if a._top_left_shape[0] < a._reg_shape[0] and a._n_blocks[0] == 1:
        a._top_left_shape[0] += pad_bottom + pad_top

    if a._top_left_shape[1] < a._reg_shape[1] and a._n_blocks[1] == 1:
        a._top_left_shape[1] += pad_left + pad_right


@constraint(computing_units="${ComputingUnits}")
@task(block=IN)
def _pad_right_block(block, pad_cols, value):
    return np.pad(
        block,
        ((0, 0), (0, pad_cols)),
        constant_values=((0, 0), (0, value))
    )


@constraint(computing_units="${ComputingUnits}")
@task(block=IN)
def _pad_bottom_block(block, pad_rows, value):
    return np.pad(
        block,
        ((0, pad_rows), (0, 0)),
        constant_values=((0, value), (0, 0))
    )


def pad_last_blocks_with_zeros(a: Array):
    """
        Pad array blocks with zeros.
        Parameters
        ----------
        a : ds-array
            The array to pad.
    """
    bottom_right_shape = compute_bottom_right_shape(a)
    if bottom_right_shape != a._reg_shape:
        rows_to_pad = a._reg_shape[0] - bottom_right_shape[0]
        cols_to_pad = a._reg_shape[1] - bottom_right_shape[1]
        pad(a, ((0, rows_to_pad), (0, cols_to_pad)), fill_value=0)


def compute_bottom_right_shape(a: Array):
    """
        Computes a shape of the bottom right block.
        Parameters
        ----------
        a : ds-array
            The array to pad.
        Returns
        -------
        size0 : int
            size of the first dimension
        size1 : int
            size of the second dimension
    """
    size0_mod = (a.shape[0] - a._top_left_shape[0]) % a._reg_shape[0]
    size0 = a._top_left_shape[0] if a._n_blocks[0] == 1 else size0_mod
    if size0_mod == 0:
        size0 = a._reg_shape[0]

    size1_mod = (a.shape[1] - a._top_left_shape[1]) % a._reg_shape[1]
    size1 = a._top_left_shape[1] if a._n_blocks[1] == 1 else size1_mod
    if size1_mod == 0:
        size1 = a._reg_shape[1]

    return size0, size1


def remove_last_rows(a: Array, n_rows):
    """
        Removes last rows from the bottom blocks of the ds-array.
        Parameters
        ----------
        a : ds-array
            The array to pad.
        n_rows : int
            The array to pad.
    """
    if n_rows <= 0:
        return

    right_bottom_shape = compute_bottom_right_shape(a)

    if n_rows >= right_bottom_shape[0]:
        # removing whole blocks
        removed_blocks = int(n_rows / right_bottom_shape[0])
        removed_rows = removed_blocks * right_bottom_shape[0]
        for i in reversed(
                range(a._n_blocks[0] - removed_blocks,
                      a._n_blocks[0])
        ):
            compss_delete_object(a._blocks[i])
            del a._blocks[i]

        a._n_blocks = (a._n_blocks[0] - removed_blocks, a._n_blocks[1])
        a._shape = (a._shape[0] - removed_rows, a._shape[1])
        n_rows = n_rows - removed_rows

    if n_rows <= 0:
        return

    for col_block_idx in range(a._n_blocks[1]):
        # removing remaining rows
        padded_block = _remove_bottom_rows(
            a._blocks[-1][col_block_idx],
            n_rows
        )
        a._blocks[-1][col_block_idx] = padded_block

    a._shape = (a._shape[0] - n_rows, a._shape[1])


def remove_last_columns(a: Array, n_columns):
    """
        Removes last columns from the right-most blocks of the ds-array.
        Parameters
        ----------
        a : ds-array
            The array to pad.
        n_columns : int
            The number of columns to remove
        Raises
        ------
        ValueError
            if n_columns >= the width of the right-most blocks
    """
    if n_columns >= compute_bottom_right_shape(a)[1]:
        raise ValueError("Number of columns to remove needs to be less than "
                         "the whole block")

    for row_block_idx in range(a._n_blocks[0]):
        padded_block = _remove_right_columns(
            a._blocks[row_block_idx][-1],
            n_columns
        )
        a._blocks[row_block_idx][-1] = padded_block

    a._shape = (a._shape[0], a._shape[1] - n_columns)


@constraint(computing_units="${ComputingUnits}")
@task(block=IN)
def _remove_right_columns(block, n_cols):
    return block[:, :-n_cols]


@constraint(computing_units="${ComputingUnits}")
@task(block=IN)
def _remove_bottom_rows(block, n_rows):
    return block[:-n_rows, :]
