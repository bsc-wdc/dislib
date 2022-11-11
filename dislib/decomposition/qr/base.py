import numpy as np
from collections import deque
import warnings

from pycompss.api.api import compss_delete_object
from pycompss.api.constraint import constraint
from pycompss.api.task import task
import dislib

from dislib.data.array import Array, full, eye
from dislib.data.util import compute_bottom_right_shape, \
    pad_last_blocks_with_zeros
from dislib.data.util.base import remove_last_rows, remove_last_columns


def qr(a: Array, mode='full', overwrite_a=False):
    """ QR Decomposition (blocked).

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
        warnings.warn(
            "The economic mode does not overwrite the original matrix. "
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
        q, r = _qr_economic(a_obj)
        _undo_padding_economic(q, r, padded_rows, padded_cols)
        return q, r
    elif mode == "full":
        q, r = _qr_full(a_obj)
        _undo_padding_full(q, r, padded_rows, padded_cols)
        return q, r
    elif mode == "r":
        r = _qr_r(a_obj)
        if padded_cols > 0:
            remove_last_columns(r, padded_cols)
        return r


ZEROS = 0
IDENTITY = 1
OTHER = 2


def _qr_full(r):
    b_size = r._reg_shape
    q, q_type = _gen_identity(
        r.shape[0],
        r.shape[0],
        r._reg_shape,
        r._n_blocks[0],
        r._n_blocks[0]
    )

    r_type = full((r._n_blocks[0], r._n_blocks[1]), (1, 1), OTHER)

    for i in range(r._n_blocks[1]):

        act_q_type, act_q, r_type_block, r_block = _qr(
            r._blocks[i][i], r_type._blocks[i][i], r._reg_shape, t=True
        )

        r_type.replace_block(i, i, r_type_block)
        r.replace_block(i, i, r_block)

        for j in range(r._n_blocks[0]):
            q_type_block, q_block = _dot(
                q._blocks[j][i],
                q_type._blocks[j][i],
                act_q,
                act_q_type,
                b_size,
                transpose_b=True
            )
            q_type.replace_block(j, i, q_type_block)
            q.replace_block(j, i, q_block)

        for j in range(i + 1, r._n_blocks[1]):
            r_type_block, r_block = _dot(
                act_q,
                act_q_type,
                r._blocks[i][j],
                r_type._blocks[i][j],
                b_size
            )
            r_type.replace_block(i, j, r_type_block)
            r.replace_block(i, j, r_block)

        compss_delete_object(act_q_type)
        compss_delete_object(act_q)

        sub_q = [[np.array([0]), np.array([0])],
                 [np.array([0]), np.array([0])]]
        sub_q_type = [[_type_block(OTHER), _type_block(OTHER)],
                      [_type_block(OTHER), _type_block(OTHER)]]

        # Update values of the respective column
        for j in range(i + 1, r._n_blocks[0]):

            sub_q[0][0], sub_q[0][1], sub_q[1][0], sub_q[1][1], \
              r_type_block1, r_block1, r_type_block2, r_block2 = _little_qr(
                r._blocks[i][i],
                r_type._blocks[i][i],
                r._blocks[j][i],
                r_type._blocks[j][i],
                r._reg_shape
            )
            r_type.replace_block(i, i, r_type_block1)
            r.replace_block(i, i, r_block1)
            r_type.replace_block(j, i, r_type_block2)
            r.replace_block(j, i, r_block2)

            # Update values of the row for the value updated in the column
            for k in range(i + 1, r._n_blocks[1]):
                [[r_type_block1], [r_type_block2]], \
                  [[r_block1], [r_block2]] = _multiply_blocked(
                    sub_q,
                    sub_q_type,
                    [[r._blocks[i][k]], [r._blocks[j][k]]],
                    [[r_type._blocks[i][k]], [r_type._blocks[j][k]]],
                    r._reg_shape
                )
                r_type.replace_block(i, k, r_type_block1)
                r.replace_block(i, k, r_block1)
                r_type.replace_block(j, k, r_type_block2)
                r.replace_block(j, k, r_block2)

            for k in range(r._n_blocks[0]):
                [[q_type_block1, q_type_block2]], \
                  [[q_block1, q_block2]] = _multiply_blocked(
                    [[q._blocks[k][i], q._blocks[k][j]]],
                    [[q_type._blocks[k][i], q_type._blocks[k][j]]],
                    sub_q,
                    sub_q_type,
                    r._reg_shape,
                    transpose_b=True
                )
                q_type.replace_block(k, i, q_type_block1)
                q.replace_block(k, i, q_block1)
                q_type.replace_block(k, j, q_type_block2)
                q.replace_block(k, j, q_block2)

            compss_delete_object(sub_q[0][0])
            compss_delete_object(sub_q[0][1])
            compss_delete_object(sub_q[1][0])
            compss_delete_object(sub_q[1][1])

    return q, r


def _qr_r(r):
    b_size = r._reg_shape
    r_type = full((r._n_blocks[0], r._n_blocks[1]), (1, 1), OTHER)

    for i in range(r._n_blocks[1]):
        act_q_type, act_q, r_type_block, r_block = _qr(
            r._blocks[i][i], r_type._blocks[i][i], r._reg_shape, t=True
        )
        r_type.replace_block(i, i, r_type_block)
        r.replace_block(i, i, r_block)

        for j in range(i + 1, r._n_blocks[1]):
            r_type_block, r_block = _dot(
                act_q,
                act_q_type,
                r._blocks[i][j],
                r_type._blocks[i][j],
                b_size
            )
            r_type.replace_block(i, j, r_type_block)
            r.replace_block(i, j, r_block)

        compss_delete_object(act_q_type)
        compss_delete_object(act_q)

        sub_q = [[np.array([0]), np.array([0])],
                 [np.array([0]), np.array([0])]]
        sub_q_type = [[_type_block(OTHER), _type_block(OTHER)],
                      [_type_block(OTHER), _type_block(OTHER)]]

        # Update values of the respective column
        for j in range(i + 1, r._n_blocks[0]):
            sub_q[0][0], sub_q[0][1], sub_q[1][0], sub_q[1][1], \
              r_type_block1, r_block1, r_type_block2, r_block2 = _little_qr(
                r._blocks[i][i],
                r_type._blocks[i][i],
                r._blocks[j][i],
                r_type._blocks[j][i],
                r._reg_shape
            )
            r_type.replace_block(i, i, r_type_block1)
            r.replace_block(i, i, r_block1)
            r_type.replace_block(j, i, r_type_block2)
            r.replace_block(j, i, r_block2)

            # Update values of the row for the value updated in the column
            for k in range(i + 1, r._n_blocks[1]):
                [[r_type_block1], [r_type_block2]], \
                  [[r_block1], [r_block2]] = _multiply_blocked(
                    sub_q,
                    sub_q_type,
                    [[r._blocks[i][k]], [r._blocks[j][k]]],
                    [[r_type._blocks[i][k]], [r_type._blocks[j][k]]],
                    r._reg_shape
                )
                r_type.replace_block(i, k, r_type_block1)
                r.replace_block(i, k, r_block1)
                r_type.replace_block(j, k, r_type_block2)
                r.replace_block(j, k, r_block2)

            compss_delete_object(sub_q[0][0])
            compss_delete_object(sub_q[0][1])
            compss_delete_object(sub_q[1][0])
            compss_delete_object(sub_q[1][1])

    return r


def _qr_economic(r):
    a_shape = (r.shape[0], r.shape[1])
    a_n_blocks = (r._n_blocks[0], r._n_blocks[1])
    b_size = r._reg_shape

    q, q_type = _gen_identity(
        r.shape[0],
        a_shape[1],
        b_size,
        r._n_blocks[0],
        r._n_blocks[1]
    )

    r_type = full((r._n_blocks[0], r._n_blocks[1]), (1, 1), OTHER)

    act_q_list = []
    sub_q_list = {}

    for i in range(a_n_blocks[1]):
        act_q_type, act_q, r_type_block, r_block = _qr(
            r._blocks[i][i], r_type._blocks[i][i], b_size, t=True
        )
        r_type.replace_block(i, i, r_type_block)
        r.replace_block(i, i, r_block)
        act_q_list.append((act_q_type, act_q))

        for j in range(i + 1, a_n_blocks[1]):
            r_type_block, r_block = _dot(
                act_q,
                act_q_type,
                r._blocks[i][j],
                r_type._blocks[i][j],
                b_size
            )
            r_type.replace_block(i, j, r_type_block)
            r.replace_block(i, j, r_block)

        # Update values of the respective column
        for j in range(i + 1, r._n_blocks[0]):
            sub_q = [[np.array([0]), np.array([0])],
                     [np.array([0]), np.array([0])]]
            sub_q_type = [[_type_block(OTHER), _type_block(OTHER)],
                          [_type_block(OTHER), _type_block(OTHER)]]

            sub_q[0][0], sub_q[0][1], sub_q[1][0], sub_q[1][1], \
                r_type_block1, r_block1, \
                r_type_block2, r_block2 = _little_qr(
                    r._blocks[i][i], r_type._blocks[i][i],
                    r._blocks[j][i], r_type._blocks[j][i],
                    b_size
            )
            r_type.replace_block(i, i, r_type_block1)
            r.replace_block(i, i, r_block1)
            r_type.replace_block(j, i, r_type_block2)
            r.replace_block(j, i, r_block2)

            sub_q_list[(j, i)] = (sub_q_type, sub_q)

            # Update values of the row for the value updated in the column
            for k in range(i + 1, a_n_blocks[1]):
                [[r_type_block1], [r_type_block2]], \
                  [[r_block1], [r_block2]] = _multiply_blocked(
                    sub_q,
                    sub_q_type,
                    [[r._blocks[i][k]], [r._blocks[j][k]]],
                    [[r_type._blocks[i][k]], [r_type._blocks[j][k]]],
                    b_size
                )
                r_type.replace_block(i, k, r_type_block1)
                r.replace_block(i, k, r_block1)
                r_type.replace_block(j, k, r_type_block2)
                r.replace_block(j, k, r_block2)

    for i in reversed(range(len(act_q_list))):
        for j in reversed(range(i + 1, r._n_blocks[0])):
            for k in range(q._n_blocks[1]):
                [[q_type_block1], [q_type_block2]], \
                  [[q_block1], [q_block2]] = _multiply_blocked(
                    sub_q_list[(j, i)][1],
                    sub_q_list[(j, i)][0],
                    [[q._blocks[i][k]], [q._blocks[j][k]]],
                    [[q_type._blocks[i][k]], [q_type._blocks[j][k]]],
                    b_size,
                    transpose_a=True
                )
                q_type.replace_block(i, k, q_type_block1)
                q.replace_block(i, k, q_block1)
                q_type.replace_block(j, k, q_type_block2)
                q.replace_block(j, k, q_block2)

            compss_delete_object(sub_q_list[(j, i)][0][0])
            compss_delete_object(sub_q_list[(j, i)][0][1])
            compss_delete_object(sub_q_list[(j, i)][1][0])
            compss_delete_object(sub_q_list[(j, i)][1][1])
            del sub_q_list[(j, i)]

        for k in range(q._n_blocks[1]):
            q_type_block, q_block = _dot(
                act_q_list[i][1],
                act_q_list[i][0],
                q._blocks[i][k],
                q_type._blocks[i][k],
                b_size,
                transpose_a=True
            )
            q_type.replace_block(i, k, q_type_block)
            q.replace_block(i, k, q_block)

        compss_delete_object(act_q_list[i][0])
        compss_delete_object(act_q_list[i][1])

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
        raise ValueError(
            "Top left block needs to be of the same shape as regular ones"
        )


def _split_matrix(a, m_size):
    b_size = int(len(a) / m_size)
    blocks = len(a)//b_size
    split_matrix = np.array(a).reshape(blocks, b_size, -1, b_size) \
                              .swapaxes(1, 2) \
                              .reshape(blocks, blocks, b_size, b_size)

    return split_matrix


def _gen_identity(n, m, b_size, n_size, m_size):
    a = eye(n, m, b_size, dtype=None)
    aux_a = eye(n_size, m_size, (1, 1), dtype=np.uint8)
    return a, aux_a


@constraint(computing_units="${ComputingUnits}")
@task(returns=np.array)
def _dot_task(a, b, transpose_result=False, transpose_a=False,
              transpose_b=False):
    if transpose_a:
        a = np.transpose(a)
    if transpose_b:
        b = np.transpose(b)
    if transpose_result:
        return np.transpose(np.dot(a, b))
    return np.dot(a, b)


@constraint(processors=[
                {"processorType": "CPU", "computingUnits": "1"},
                {"processorType": "GPU", "computingUnits": "1"},
            ])
@task(returns=np.array)
def _dot_task_gpu(a, b, transpose_result=False, transpose_a=False,
                  transpose_b=False):
    import cupy as cp

    a_gpu, b_gpu = cp.asarray(a), cp.asarray(b)
    if transpose_a:
        a_gpu = np.transpose(a_gpu)
    if transpose_b:
        b_gpu = np.transpose(b_gpu)

    dot_gpu = cp.dot(a_gpu, b_gpu)

    if transpose_result:
        dot_gpu = cp.transpose(dot_gpu)

    return cp.asnumpy(dot_gpu)


@constraint(computing_units="${ComputingUnits}")
@task(returns=(np.array, np.array))
def _qr_task(a, a_type, b_size, mode='reduced', t=False):
    if type(a_type) is object or a_type[0, 0] == OTHER:
        q, r = np.linalg.qr(a, mode=mode)
    elif a_type[0, 0] == ZEROS:
        q, r = np.linalg.qr(np.zeros(b_size), mode=mode)
    else:
        q, r = np.linalg.qr(np.identity(max(b_size)), mode=mode)

    if t:
        q = np.transpose(q)

    return q, r


def _type_block(value):
    return np.full((1, 1), value, np.uint8)


def _dot(a, a_type, b, b_type, b_size, transpose_result=False,
         transpose_a=False, transpose_b=False):

    if dislib.__gpu_available__:
        dot_func = _dot_task_gpu
    else:
        dot_func = _dot_task

    result = dot_func(
        a,
        b,
        transpose_result=transpose_result,
        transpose_a=transpose_a,
        transpose_b=transpose_b
    )

    return _type_block(OTHER), result


@constraint(computing_units="${ComputingUnits}")
@task(returns=(np.array, np.array, np.array, np.array, np.array, np.array))
def _little_qr_task(a, type_a, b, type_b, b_size):
    regular_b_size = b_size[0]

    curr_a = np.bmat([[a], [b]])
    (sub_q, sub_r) = np.linalg.qr(curr_a, mode='complete')
    aa = sub_r[0:regular_b_size]
    bb = sub_r[regular_b_size:2 * regular_b_size]
    sub_q = _split_matrix(sub_q, 2)

    return (np.transpose(sub_q[0][0]), np.transpose(sub_q[1][0]),
            np.transpose(sub_q[0][1]), np.transpose(sub_q[1][1]),
            aa, bb)


def _multiply_blocked(a, type_a, b, type_b, b_size, transpose_a=False,
                      transpose_b=False):

    n_blocks = (len(a), len(b[0]))
    c = Array._get_out_blocks(n_blocks)
    type_c = Array._get_out_blocks(n_blocks)

    if transpose_a:
        a = list(map(list, zip(*a)))

    if transpose_b:
        b = list(map(list, zip(*b)))

    for i in range(n_blocks[0]):
        for j in range(n_blocks[1]):
            hblock = a[i]
            vblock = [b[k][j] for k in range(len(b))]

            c[i][j] = _multiply_block_groups(hblock, vblock,
                                             transpose_a, transpose_b)

    return type_c, c


@constraint(computing_units="${ComputingUnits}")
@task(returns=np.array)
def _matmul_with_transpose(a, b, transpose_a, transpose_b):
    return (a.T if transpose_a else a) @ (b.T if transpose_b else b)


@constraint(processors=[
                {"processorType": "CPU", "computingUnits": "1"},
                {"processorType": "GPU", "computingUnits": "1"},
            ])
@task(returns=np.array)
def _add_gpu(block1, block2):
    import cupy as cp

    res = cp.add(cp.asarray(block1), cp.asarray(block2))
    return cp.asnumpy(res)


@constraint(computing_units="${ComputingUnits}")
@task(returns=np.array)
def _add_cpu(block1, block2):
    return block1 + block2


@constraint(processors=[
                {"processorType": "CPU", "computingUnits": "1"},
                {"processorType": "GPU", "computingUnits": "1"},
            ])
@task(returns=np.array)
def _matmul_gpu(a, b, transpose_a, transpose_b):
    import cupy as cp

    a_gpu, b_gpu = cp.asarray(a), cp.asarray(b)
    if transpose_a:
        a_gpu = a_gpu.T
    if transpose_b:
        b_gpu = b_gpu.T
    res_gpu = cp.matmul(a_gpu, b_gpu)
    return cp.asnumpy(res_gpu)


def _multiply_block_groups(hblock, vblock, transpose_a=False,
                           transpose_b=False):
    blocks = deque()

    if dislib.__gpu_available__:
        matmul_func = _matmul_gpu
        add_func = _add_gpu
    else:
        matmul_func = _matmul_with_transpose
        add_func = _add_cpu

    for blocki, blockj in zip(hblock, vblock):
        blocks.append(
            matmul_func(blocki, blockj,
                        transpose_a, transpose_b)
        )

    while len(blocks) > 1:
        block1 = blocks.popleft()
        block2 = blocks.popleft()
        blocks.append(add_func(block1, block2))

        compss_delete_object(block1)
        compss_delete_object(block2)

    return blocks[0]


@constraint(processors=[
                {"processorType": "CPU", "computingUnits": "1"},
                {"processorType": "GPU", "computingUnits": "1"},
            ])
@task(returns=(np.array, np.array))
def _qr_task_gpu(a, a_type, b_size, mode='reduced', t=False):
    import cupy as cp
    from cupy.linalg import qr

    a_gpu = cp.asarray(a)
    if type(a_type) is object or a_type[0, 0] == OTHER:
        q, r = qr(a_gpu, mode=mode)
    elif a_type[0, 0] == ZEROS:
        q, r = qr(cp.zeros(b_size), mode=mode)
    else:
        q, r = qr(cp.identity(max(b_size)), mode=mode)
    if t:
        q = cp.transpose(q)
    q, r = cp.asnumpy(q), cp.asnumpy(r)
    return q, r


def _qr(a, a_type, b_size, mode='reduced', t=False):
    if dislib.__gpu_available__:
        qr_func = _qr_task_gpu
    else:
        qr_func = _qr_task
    q_aux, r_aux = qr_func(a, a_type, b_size, mode=mode, t=t)
    return _type_block(OTHER), q_aux, _type_block(OTHER), r_aux


@constraint(processors=[
                {"processorType": "CPU", "computingUnits": "1"},
                {"processorType": "GPU", "computingUnits": "1"},
            ])
@task(returns=(np.array, np.array, np.array, np.array, np.array, np.array))
def _little_qr_task_gpu(a, type_a, b, type_b, b_size, transpose=False):
    import cupy as cp

    regular_b_size = b_size[0]

    curr_a = np.bmat([[a], [b]])
    sub_q_gpu, sub_r_gpu = cp.linalg.qr(cp.asarray(curr_a), mode='complete')
    sub_q, sub_r = cp.asnumpy(sub_q_gpu), cp.asnumpy(sub_r_gpu)
    aa = sub_r[0:regular_b_size]
    bb = sub_r[regular_b_size:2 * regular_b_size]
    sub_q = _split_matrix(sub_q, 2)

    return (np.transpose(sub_q[0][0]), np.transpose(sub_q[1][0]),
            np.transpose(sub_q[0][1]), np.transpose(sub_q[1][1]),
            aa, bb)


def _little_qr(a, type_a, b, type_b, b_size):
    if dislib.__gpu_available__:
        little_qr_func = _little_qr_task_gpu
    else:
        little_qr_func = _little_qr_task

    sub_q00, sub_q01, sub_q10, sub_q11, aa, bb = little_qr_func(
        a,
        type_a,
        b,
        type_b,
        b_size
    )

    return sub_q00, sub_q01, sub_q10, sub_q11, \
        _type_block(OTHER), aa, _type_block(OTHER), bb
