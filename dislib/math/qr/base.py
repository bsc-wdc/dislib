import numpy as np
import warnings
from pycompss.api.api import compss_barrier

from pycompss.api.constraint import constraint
from pycompss.api.parameter import INOUT, IN
from pycompss.api.task import task

from dislib.data.array import Array, identity, zeros
from dislib.data.util import compute_bottom_right_shape, pad_last_blocks_with_zeros
from dislib.data.util.base import remove_last_rows, remove_last_columns
from dislib.math.qr import save_memory as save_mem


def qr_blocked(a: Array, mode='full', overwrite_a=False, save_memory=False):
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

    if save_memory:
        q, r = save_mem.qr_blocked(a_obj, mode=mode, overwrite_a=True)
        _undo_padding(q, r, padded_rows, padded_cols)
        return q, r
    elif mode == "economic":
        q, r = _qr_economic(a_obj)
        return q, r
    elif mode == "full":
        q, r = _qr_full(a_obj)
        _undo_padding(q, r, padded_rows, padded_cols)
        return q, r
    elif mode == "r":
        r = _qr_r(a_obj)
        if padded_cols > 0:
            remove_last_columns(r, padded_cols)
        return r


def _qr_full(r):
    q = identity(r.shape[0], r._reg_shape, dtype=None)

    for i in range(r._n_blocks[1]):
        act_q, r._blocks[i][i] = _qr_task(r._blocks[i][i], t=True)

        for j in range(r._n_blocks[0]):
            q._blocks[j][i] = _dot_task(q._blocks[j][i], act_q, transpose_b=True)

        for j in range(i + 1, r._n_blocks[1]):
            r._blocks[i][j] = _dot_task(act_q, r._blocks[i][j])

        # Update values of the respective column
        for j in range(i + 1, r._n_blocks[0]):
            sub_q = [[np.array([0]), np.array([0])],
                    [np.array([0]), np.array([0])]]

            sub_q[0][0], sub_q[0][1], sub_q[1][0], sub_q[1][1], r._blocks[i][i], r._blocks[j][i] = _little_qr(
                r._blocks[i][i], r._blocks[j][i],
                r._reg_shape, transpose=True)

            # Update values of the row for the value updated in the column
            for k in range(i + 1, r._n_blocks[1]):
                [[r._blocks[i][k]], [r._blocks[j][k]]] = _multiply_blocked(
                    sub_q,
                    [[r._blocks[i][k]], [r._blocks[j][k]]],
                    r._reg_shape
                )

            for k in range(r._n_blocks[0]):
                [[q._blocks[k][i], q._blocks[k][j]]] = _multiply_blocked(
                    [[q._blocks[k][i], q._blocks[k][j]]],
                    sub_q,
                    r._reg_shape,
                    transpose_b=True
                )

    return q, r


def _qr_r(r):
    for i in range(r._n_blocks[1]):
        act_q, r._blocks[i][i] = _qr_task(r._blocks[i][i], t=True)

        for j in range(i + 1, r._n_blocks[1]):
            r._blocks[i][j] = _dot_task(act_q, r._blocks[i][j])

        # Update values of the respective column
        for j in range(i + 1, r._n_blocks[0]):
            sub_q = [[np.array([0]), np.array([0])],
                    [np.array([0]), np.array([0])]]

            sub_q[0][0], sub_q[0][1], sub_q[1][0], sub_q[1][1], r._blocks[i][i], r._blocks[j][i] = _little_qr(
                r._blocks[i][i], r._blocks[j][i],
                r._reg_shape, transpose=True)

            # Update values of the row for the value updated in the column
            for k in range(i + 1, r._n_blocks[1]):
                [[r._blocks[i][k]], [r._blocks[j][k]]] = _multiply_blocked(
                    sub_q,
                    [[r._blocks[i][k]], [r._blocks[j][k]]],
                    r._reg_shape
                )

    return r


def _qr_economic(r):
    a_shape = (r.shape[0], r.shape[1])
    a_n_blocks = (r._n_blocks[0], r._n_blocks[1])
    b_size = r._reg_shape

    # FIXME generate identity above and zeros below instead of slicing
    q = identity(a_shape[0], b_size, dtype=None)[:, 0:a_shape[1]]

    act_q_list = []
    sub_q_list = {}

    for i in range(a_n_blocks[1]):
        act_q, r._blocks[i][i] = _qr_task(r._blocks[i][i], t=True)
        act_q_list.append(act_q)

        for j in range(i + 1, a_n_blocks[1]):
            r._blocks[i][j] = _dot_task(act_q, r._blocks[i][j])

        # Update values of the respective column
        for j in range(i + 1, r._n_blocks[0]):
            sub_q = [[np.array([0]), np.array([0])],
                    [np.array([0]), np.array([0])]]

            sub_q[0][0], sub_q[0][1], sub_q[1][0], sub_q[1][1], r._blocks[i][i], r._blocks[j][i] = _little_qr(
                r._blocks[i][i], r._blocks[j][i],
                b_size, transpose=True)

            sub_q_list[(j, i)] = sub_q

            # Update values of the row for the value updated in the column
            for k in range(i + 1, a_n_blocks[1]):
                [[r._blocks[i][k]], [r._blocks[j][k]]] = _multiply_blocked(
                    sub_q,
                    [[r._blocks[i][k]], [r._blocks[j][k]]],
                    b_size
                )

    for i in reversed(range(len(act_q_list))):
        for j in reversed(range(i + 1, r._n_blocks[0])):
            for k in range(q._n_blocks[1]):
                [[q._blocks[i][k]], [q._blocks[j][k]]] = _multiply_blocked(
                    sub_q_list[(j, i)],
                    [[q._blocks[i][k]], [q._blocks[j][k]]],
                    b_size,
                    transpose_a=True
                )

        for k in range(q._n_blocks[1]):
            q._blocks[i][k] = _dot_task(act_q_list[i], q._blocks[i][k], transpose_a=True)

    # removing last rows of r to make it n x n instead of m x n
    remove_last_rows(r, r.shape[0] - r.shape[1])

    return q, r


def _undo_padding(q, r, n_rows, n_cols):
    if n_rows > 0 and q is not None:
        remove_last_rows(q, n_rows)
        remove_last_columns(q, n_rows)
    if n_cols > 0:
        remove_last_columns(r, n_cols)

    remove_last_rows(r, max(r.shape[0] - q.shape[1], 0))


def _validate_ds_array(a: Array):
    if a._n_blocks[0] < a._n_blocks[1]:
        raise ValueError("m > n is required for matrices m x n")

    if a._reg_shape[0] != a._reg_shape[1]:
        raise ValueError("Square blocks are required")

    if a._reg_shape != a._top_left_shape:
        raise ValueError("Top left block needs to be of the same shape as regular ones")


@constraint(computing_units="${computingUnits}")
@task(returns=(np.array, np.array))
def _qr_task(a, mode='reduced', t=False):
    from numpy.linalg import qr
    q, r = qr(a, mode=mode)
    if t:
        q = np.transpose(q)
    return q, r


@constraint(computing_units="${computingUnits}")
@task(returns=np.array)
def _dot_task(a, b, transpose_result=False, transpose_a=False, transpose_b=False):
    if transpose_a:
        a = np.transpose(a)
    if transpose_b:
        b = np.transpose(b)
    if transpose_result:
        return np.transpose(np.dot(a, b))
    return np.dot(a, b)


@constraint(computing_units="${computingUnits}")
@task(returns=(np.array, np.array, np.array, np.array, np.array, np.array))
def _little_qr_task(a, b, b_size, transpose=False):
    regular_b_size = b_size[0]
    curr_a = np.bmat([[a], [b]])
    (sub_q, sub_r) = np.linalg.qr(curr_a, mode='complete')
    aa = sub_r[0:regular_b_size]
    bb = sub_r[regular_b_size:2 * regular_b_size]
    sub_q = _split_matrix(sub_q, 2)
    if transpose:
        return np.transpose(sub_q[0][0]), np.transpose(sub_q[1][0]), np.transpose(sub_q[0][1]), np.transpose(
            sub_q[1][1]), aa, bb
    else:
        return sub_q[0][0], sub_q[0][1], sub_q[1][0], sub_q[1][1], aa, bb


def _little_qr(a, b, b_size, transpose=False):
    sub_q00, sub_q01, sub_q10, sub_q11, aa, bb = _little_qr_task(a, b, b_size, transpose)
    return sub_q00, sub_q01, sub_q10, sub_q11, aa, bb


@constraint(computing_units="${computingUnits}")
@task(a=IN, b=IN, c=INOUT)
def _multiply_single_block_task(a, b, c, transpose_a=False, transpose_b=False):
    if transpose_a:
        a = np.transpose(a)
    if transpose_b:
        b = np.transpose(b)
    c += (a.dot(b))


def _multiply_blocked(a, b, b_size, transpose_a=False, transpose_b=False):
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
            c[i].append(np.zeros(b_size))
            for k in range(len(a[0])):
                _multiply_single_block_task(a[i][k], b[k][j], c[i][j], transpose_a=transpose_a, transpose_b=transpose_b)

    return c


def _split_matrix(a, m_size):
    b_size = int(len(a) / m_size)
    split_matrix = [[None for m in range(m_size)] for m in range(m_size)]
    for i in range(m_size):
        for j in range(m_size):
            split_matrix[i][j] = a[i * b_size:(i + 1) * b_size, j * b_size:(j + 1) * b_size]
    return split_matrix
