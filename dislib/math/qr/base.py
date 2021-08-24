import numpy as np
import warnings

from pycompss.api.api import compss_delete_object
from pycompss.api.constraint import constraint
from pycompss.api.parameter import INOUT, IN, IN_DELETE
from pycompss.api.task import task

from dislib.data.array import Array, identity, eye
from dislib.data.array_block import ArrayBlock
from dislib.data.util import compute_bottom_right_shape, pad_last_blocks_with_zeros
from dislib.data.util.base import remove_last_rows, remove_last_columns


def qr_blocked(a: Array, mode='full', overwrite_a=False):
    """ QR Decomposition (blocked / save memory).

    Parameters
    ----------
    a : ds-arrays
        Input ds-array.
    mode : string
        Mode of the algorithm
        'full' - computes full Q matrix of size m x m and R of size m x n
        'economic' - computes Q of size m x n and R of size n x n
        'r' - computes only R of size m x n
    overwrite_a : bool
        Overwriting the input matrix as R.
    save_memory : bool
        Using the "save memory" algorithm.

    Returns
    -------
    q : ds-array
        only for modes 'full' and 'economic'
     r : ds-array
        for all modes

    Raises
    ------
    ValueError
        If m < n for the provided matrix m x n
        or
        If blocks are not square
        or
        If top left shape is different than regular
        or
        If bottom right block is different than regular
    """

    _validate_ds_array(a)

    if mode not in ['full', 'economic', 'r']:
        raise ValueError("Unsupported mode: " + mode)

    if mode == 'economic' and overwrite_a:
        warnings.warn("The economic mode does not overwrite the original matrix. "
                      "Argument overwrite_a is changed to False.", UserWarning)
        overwrite_a = False

    a_obj = a if overwrite_a else a.copy()

    padded_rows = 0
    padded_cols = 0
    bottom_right_shape = compute_bottom_right_shape(a_obj)
    if bottom_right_shape != a_obj._reg_shape:
        padded_rows = a_obj._reg_shape[0] - bottom_right_shape[0]
        padded_cols = a_obj._reg_shape[1] - bottom_right_shape[1]
        pad_last_blocks_with_zeros(a_obj)

    if mode == "economic":
        q, r = _qr_economic_save_mem(a_obj)
        _undo_padding_economic(q, r, padded_rows, padded_cols)
        return q, r
    elif mode == "full":
        q, r = _qr_full_save_mem(a_obj)
        _undo_padding_full(q, r, padded_rows, padded_cols)
        return q, r
    elif mode == "r":
        r = _qr_r_save_mem(a_obj)
        if padded_cols > 0:
            remove_last_columns(r, padded_cols)
        return r


def _qr_full_save_mem(r):
    b_size = r._reg_shape
    q = _gen_identity_save_mem(r.shape[0], r.shape[0], r._reg_shape)

    for i in range(r._n_blocks[1]):
        act_q = _qr_save_mem(r._blocks[i][i], t=True)

        for j in range(r._n_blocks[0]):
            q_block = _dot_save_mem(q._blocks[j][i], act_q, b_size, transpose_b=True)
            q.replace_block(j, i, q_block)

        for j in range(i + 1, r._n_blocks[1]):
            r_block = _dot_save_mem(act_q, r._blocks[i][j], b_size)
            r.replace_block(i, j, r_block)

        compss_delete_object(act_q)

        sub_q = [[None, None],
                [None, None]]

        # Update values of the respective column
        for j in range(i + 1, r._n_blocks[0]):
            sub_q[0][0], sub_q[0][1], sub_q[1][0], sub_q[1][1] = _little_qr_save_mem(
                r._blocks[i][i], r._blocks[j][i],
                r._reg_shape, transpose=True)

            # Update values of the row for the value updated in the column
            for k in range(i + 1, r._n_blocks[1]):
                [[r_block1], [r_block2]] = _multiply_blocked_save_mem(
                    sub_q,
                    [[r._blocks[i][k]], [r._blocks[j][k]]],
                    r._reg_shape
                )
                r.replace_block(i, k, r_block1)
                r.replace_block(j, k, r_block2)

            for k in range(r._n_blocks[0]):
                [[q_block1, q_block2]] = _multiply_blocked_save_mem(
                    [[q._blocks[k][i], q._blocks[k][j]]],
                    sub_q,
                    r._reg_shape,
                    transpose_b=True
                )
                q.replace_block(k, i, q_block1)
                q.replace_block(k, j, q_block2)

            _compss_delete_array(sub_q)

    return q, r


def _qr_r_save_mem(r):
    b_size = r._reg_shape

    for i in range(r._n_blocks[1]):
        act_q = _qr_save_mem(
            r._blocks[i][i], t=True
        )

        for j in range(i + 1, r._n_blocks[1]):
            r_block = _dot_save_mem(
                act_q, r._blocks[i][j], b_size
            )
            r.replace_block(i, j, r_block)

        compss_delete_object(act_q)

        sub_q = [[None, None],
                [None, None]]

        # Update values of the respective column
        for j in range(i + 1, r._n_blocks[0]):
            sub_q[0][0], sub_q[0][1], sub_q[1][0], sub_q[1][1] = _little_qr_save_mem(
                r._blocks[i][i], r._blocks[j][i],
                r._reg_shape, transpose=True)

            # Update values of the row for the value updated in the column
            for k in range(i + 1, r._n_blocks[1]):
                [[r_block1], [r_block2]] = _multiply_blocked_save_mem(
                    sub_q,
                    [[r._blocks[i][k]], [r._blocks[j][k]]],
                    r._reg_shape
                )
                r.replace_block(i, k, r_block1)
                r.replace_block(j, k, r_block2)

            _compss_delete_array(sub_q)

    return r


def _qr_economic_save_mem(r):
    a_shape = (r.shape[0], r.shape[1])
    a_n_blocks = (r._n_blocks[0], r._n_blocks[1])
    b_size = r._reg_shape

    q = _gen_identity_save_mem(r.shape[0], a_shape[1], b_size)

    act_q_list = []
    sub_q_list = {}

    for i in range(a_n_blocks[1]):
        act_q = _qr_save_mem(
            r._blocks[i][i], t=True
        )
        act_q_list.append(act_q)

        for j in range(i + 1, a_n_blocks[1]):
            r_block = _dot_save_mem(
                act_q, r._blocks[i][j], b_size
            )
            r.replace_block(i, j, r_block)

        # Update values of the respective column
        for j in range(i + 1, r._n_blocks[0]):
            sub_q = [[None, None],
                     [None, None]]

            sub_q[0][0], sub_q[0][1], sub_q[1][0], sub_q[1][1] = _little_qr_save_mem(
                r._blocks[i][i],
                r._blocks[j][i],
                b_size, transpose=True
            )

            sub_q_list[(j, i)] = sub_q

            # Update values of the row for the value updated in the column
            for k in range(i + 1, a_n_blocks[1]):
                [[r_block1], [r_block2]] = _multiply_blocked_save_mem(
                    sub_q,
                    [[r._blocks[i][k]], [r._blocks[j][k]]],
                    b_size
                )
                r.replace_block(i, k, r_block1)
                r.replace_block(j, k, r_block2)

    for i in reversed(range(len(act_q_list))):
        for j in reversed(range(i + 1, r._n_blocks[0])):
            for k in range(q._n_blocks[1]):
                [[q_block1], [q_block2]] = _multiply_blocked_save_mem(
                    sub_q_list[(j, i)],
                    [[q._blocks[i][k]], [q._blocks[j][k]]],
                    b_size,
                    transpose_a=True
                )
                q.replace_block(i, k, q_block1)
                q.replace_block(j, k, q_block2)

            _compss_delete_array(sub_q_list[(j, i)])
            del sub_q_list[(j, i)]

        for k in range(q._n_blocks[1]):
            q_block = _dot_save_mem(
                act_q_list[i], q._blocks[i][k], b_size, transpose_a=True
            )
            q.replace_block(i, k, q_block)

        compss_delete_object(act_q_list[i])

    # removing last rows of r to make it n x n instead of m x n
    remove_last_rows(r, r.shape[0] - r.shape[1])

    return q, r


def _undo_padding_full(q, r, n_rows, n_cols):
    if n_rows > 0:
        remove_last_rows(q, n_rows)
        remove_last_columns(q, n_rows)
    if n_cols > 0:
        remove_last_columns(r, n_cols)

    remove_last_rows(r, max(r.shape[0] - q.shape[1], 0))


def _undo_padding_economic(q, r, n_rows, n_cols):
    if n_rows > 0:
        remove_last_rows(q, n_rows)
    if n_cols > 0:
        remove_last_columns(r, n_cols)
        remove_last_rows(r, n_cols)
        remove_last_columns(q, n_cols)


def _validate_ds_array(a: Array):
    if a._n_blocks[0] < a._n_blocks[1]:
        raise ValueError("m > n is required for matrices m x n")

    if a._reg_shape[0] != a._reg_shape[1]:
        raise ValueError("Square blocks are required")

    if a._reg_shape != a._top_left_shape:
        raise ValueError("Top left block needs to be of the same shape as regular ones")


def _split_matrix(a, m_size):
    b_size = int(len(a) / m_size)
    split_matrix = [[None for m in range(m_size)] for m in range(m_size)]
    for i in range(m_size):
        for j in range(m_size):
            split_matrix[i][j] = a[i * b_size:(i + 1) * b_size, j * b_size:(j + 1) * b_size]
    return split_matrix


def _gen_identity_save_mem(n, m, b_size):
    a = eye(n, m, b_size, dtype=None)
    return a


@constraint(computing_units="${computingUnits}")
@task(a=INOUT, returns=object)
def _qr_task_save_mem(a, mode='reduced', t=False):
    from numpy.linalg import qr
    q, r = qr(np.asarray(a), mode=mode)
    if t:
        q = np.transpose(q)
    a.replace_content(r)
    return ArrayBlock(q)


def _qr_save_mem(a, mode='reduced', t=False):
    q_aux = _qr_task_save_mem(a, mode=mode, t=t)
    return q_aux


def _empty_block_save_mem(shape):
    return ArrayBlock(None, block_type=ArrayBlock.ZEROS, shape=shape)


@constraint(computing_units="${computingUnits}")
@task(returns=object)
def _dot_save_mem(a, b, b_size, transpose_result=False, transpose_a=False, transpose_b=False):
    if a.type == ArrayBlock.ZEROS:
        return _empty_block_save_mem(b_size)
    if a.type == ArrayBlock.IDENTITY:
        if transpose_b != transpose_result:
            return _transpose_block_save_mem(b)
        else:
            return b
    if b.type == ArrayBlock.ZEROS:
        return _empty_block_save_mem(b_size)
    if b.type == ArrayBlock.IDENTITY:
        if transpose_a != transpose_result:
            return _transpose_block_save_mem(a)
        return a

    if transpose_a:
        a = np.transpose(a)
    if transpose_b:
        b = np.transpose(b)
    if transpose_result:
        return ArrayBlock(np.transpose(np.dot(a, b)))
    return ArrayBlock(np.dot(a, b))


@constraint(computing_units="${computingUnits}")
@task(a=INOUT, b=INOUT, returns=(object, object, object, object))
def _little_qr_task_save_mem(a, b, b_size, transpose=False):
    regular_b_size = b_size[0]
    curr_a = np.bmat([[np.asarray(a)], [np.asarray(b)]])
    (sub_q, sub_r) = np.linalg.qr(curr_a, mode='complete')
    aa = sub_r[0:regular_b_size]
    bb = sub_r[regular_b_size:2 * regular_b_size]
    sub_q = _split_matrix(sub_q, 2)
    b00, b10, b01, b11 = (sub_q[0][0], sub_q[1][0], sub_q[0][1], sub_q[1][1])
    if transpose:
        b00, b10, b01, b11 = (np.transpose(b00), np.transpose(b10), np.transpose(b01), np.transpose(b11))

    a.replace_content(aa, ArrayBlock.OTHER)
    b.replace_content(bb, ArrayBlock.OTHER)
    return ArrayBlock(b00), ArrayBlock(b10), ArrayBlock(b01), ArrayBlock(b11)


def _little_qr_save_mem(a, b, b_size, transpose=False):
    return _little_qr_task_save_mem(a, b, b_size, transpose)


@constraint(computing_units="${computingUnits}")
@task(returns=object)
def _multiply_single_block_task_save_mem(a, b, c, b_size, transpose_a=False, transpose_b=False):
    if a.type == ArrayBlock.ZEROS or b.type == ArrayBlock.ZEROS:
        return c

    fun_a = [a.type, a]
    fun_b = [b.type, b]

    if c.type == ArrayBlock.ZEROS:
        c = np.zeros((b_size[0], b_size[1]))
    elif c.type == ArrayBlock.IDENTITY:
        c = np.identity(b_size[0])

    if fun_a[0] == ArrayBlock.IDENTITY:
        if fun_b[0] == ArrayBlock.IDENTITY:
            fun_b[1] = np.identity(b_size[0])
        if transpose_b:
            aux = np.transpose(fun_b[1])
        else:
            aux = fun_b[1]
        c += aux
        return ArrayBlock(c, block_type=ArrayBlock.OTHER, shape=c.shape)

    if fun_b[0] == ArrayBlock.IDENTITY:
        if transpose_a:
            aux = np.transpose(fun_a[1])
        else:
            aux = fun_a[1]
        c += aux
        return ArrayBlock(c, block_type=ArrayBlock.OTHER, shape=c.shape)

    if transpose_a:
        fun_a[1] = np.transpose(fun_a[1])

    if transpose_b:
        fun_b[1] = np.transpose(fun_b[1])

    return ArrayBlock(np.asarray(c) + np.dot(fun_a[1], fun_b[1]))


def _multiply_single_block_save_mem(a, b, c, b_size, transpose_a=False, transpose_b=False):
    return _multiply_single_block_task_save_mem(a, b, c, b_size, transpose_a=transpose_a, transpose_b=transpose_b)


def _multiply_blocked_save_mem(a, b, b_size, transpose_a=False, transpose_b=False):
    if transpose_a:
        new_a = []
        for i in range(len(a[0])):
            new_a.append([])
            for j in range(len(a)):
                new_a[i].append(a[j][i])
        a = new_a

    if transpose_b:
        new_b = []
        for i in range(len(b[0])):
            new_b.append([])
            for j in range(len(b)):
                new_b[i].append(b[j][i])
        b = new_b

    c = []
    for i in range(len(a)):
        c.append([])
        for j in range(len(b[0])):
            c[i].append(_empty_block_save_mem(b_size))
            for k in range(len(a[0])):
                c[i][j] = _multiply_single_block_save_mem(
                    a[i][k],
                    b[k][j],
                    c[i][j],
                    b_size, transpose_a=transpose_a, transpose_b=transpose_b)

    return c


def _transpose_block_save_mem(a):
    if a.type == ArrayBlock.ZEROS or a.type == ArrayBlock.IDENTITY:
        return a
    elif a.type == ArrayBlock.OTHER:
        a = np.transpose(a)
        return ArrayBlock(a)


def _compss_delete_array(array):
    for i in range(len(array)):
        for j in range(len(array[i])):
            compss_delete_object(array[i][j])
