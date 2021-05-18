import numpy as np
from pycompss.api.constraint import constraint
from pycompss.api.task import task

from dislib.data.array import Array, identity, full

ZEROS = 0
IDENTITY = 1
OTHER = 2


def qr_blocked(a: Array, mode='full', overwrite_a=False):
    """ QR Decomposition (blocked / save memory).

    Parameters
    ----------
    a : ds-arrays
        Input ds-array.
    overwrite_a : bool
        overwriting the input matrix as R.

    Returns
    -------
    q : ds-array
    r : ds-array
    """

    b_size = a._reg_shape
    m_size = (a._n_blocks[0], a._n_blocks[1])

    q, q_type = _gen_identity(a.shape[0], b_size, m_size[0])
    q._sparse = False

    if not overwrite_a:
        r = a.copy()
        r._sparse = False
    else:
        r = a

    r_type = full((m_size[0], m_size[1]), (1, 1), OTHER)

    for i in range(m_size[1]):
        act_q_type, act_q, r_type._blocks[i][i], r._blocks[i][i] = _qr(r._blocks[i][i], r_type._blocks[i][i], b_size, t=True)

        for j in range(m_size[0]):
            q_type._blocks[j][i], q._blocks[j][i] = _dot(q._blocks[j][i], q_type._blocks[j][i], act_q, act_q_type, transpose_b=True)

        for j in range(i + 1, m_size[1]):
            r_type._blocks[i][j], r._blocks[i][j] = _dot(act_q, act_q_type, r._blocks[i][j], r_type._blocks[i][j])

        # Update values of the respective column
        for j in range(i + 1, m_size[0]):
            subQ = [[np.array([0]), np.array([0])],
                    [np.array([0]), np.array([0])]]
            subQ_type = [[_type_block(OTHER), _type_block(OTHER)],
                        [_type_block(OTHER), _type_block(OTHER)]]

            subQ[0][0], subQ[0][1], subQ[1][0], subQ[1][1], r_type._blocks[i][i], r._blocks[i][i], r_type._blocks[j][i], r._blocks[j][i] = _little_qr(
                r._blocks[i][i], r_type._blocks[i][i], r._blocks[j][i], r_type._blocks[j][i],
                b_size, transpose=True)

            # Update values of the row for the value updated in the column
            for k in range(i + 1, m_size[1]):
                [[r_type._blocks[i][k]], [r_type._blocks[j][k]]], [[r._blocks[i][k]], [r._blocks[j][k]]] = _multiply_blocked(
                    subQ,
                    subQ_type,
                    [[r._blocks[i][k]], [r._blocks[j][k]]],
                    [[r_type._blocks[i][k]], [r_type._blocks[j][k]]],
                    b_size
                )

            for k in range(m_size[0]):
                [[q_type._blocks[k][i], q_type._blocks[k][j]]], [[q._blocks[k][i], q._blocks[k][j]]] = _multiply_blocked(
                    [[q._blocks[k][i], q._blocks[k][j]]],
                    [[q_type._blocks[k][i], q_type._blocks[k][j]]],
                    subQ,
                    subQ_type,
                    b_size,
                    transpose_b=True
                )

    return q, r


def _gen_identity(n, b_size, m_size):
    a = identity(n, b_size, dtype=None)
    aux_a = identity(m_size, (1, 1), dtype=np.uint8)
    return a, aux_a


@constraint(computing_units="${computingUnits}")
@task(returns=(np.array, np.array))
def _qr_task(a, a_type, b_size, mode='reduced', t=False):
    from numpy.linalg import qr
    if a_type[0, 0] == OTHER:
        q, r = qr(a, mode=mode)
    elif a_type[0, 0] == ZEROS:
        q, r = qr(np.zeros(b_size), mode=mode)
    else:
        q, r = qr(np.identity(max(b_size)), mode=mode)
    if t:
        q = np.transpose(q)
    return q, r


def _qr(a, a_type, b_size, mode='reduced', t=False):
    q_aux, r_aux = _qr_task(a, a_type, b_size, mode=mode, t=t)
    return _type_block(OTHER), q_aux, _type_block(OTHER), r_aux


def _type_block(value):
    return np.full((1, 1), value, np.uint8)


def _empty_block(shape):
    return np.full(shape, 0, dtype=np.uint8)


@constraint(computing_units="${computingUnits}")
@task(returns=(np.array, np.array))
def _dot(a, a_type, b, b_type, transpose_result=False, transpose_b=False):
    if a_type[0][0] == ZEROS:
        return _type_block(ZEROS), _empty_block(a.shape)
    if a_type[0][0] == IDENTITY:
        if transpose_b and transpose_result:
            return b_type, b
        if transpose_b or transpose_result:
            return _transpose_block(b, b_type)

        return b_type, b
    if b_type[0][0] == ZEROS:
        return _type_block(ZEROS), _empty_block(a.shape)
    if b_type[0][0] == IDENTITY:
        if transpose_result:
            return _transpose_block(a, a_type)
        return a_type, a
    result = _dot_task(a, b, transpose_result=transpose_result, transpose_b=transpose_b)

    return _type_block(OTHER), result


def _dot_task(a, b, transpose_result=False, transpose_b=False):
    if transpose_b:
        b = np.transpose(b)
    if transpose_result:
        return np.transpose(np.dot(a, b))
    return np.dot(a, b)


@constraint(computing_units="${computingUnits}")
@task(returns=(np.array, np.array, np.array, np.array, np.array, np.array))
def _little_qr_task(a, type_a, b, type_b, b_size, transpose=False):
    regular_b_size = b_size[0]
    ent_a = [type_a, a]
    ent_b = [type_b, b]
    for mat in [ent_a, ent_b]:
        if mat[0] == ZEROS:
            mat[1] = np.zeros((regular_b_size, regular_b_size))
        elif mat[0] == IDENTITY:
            mat[1] = np.identity(regular_b_size)
    curr_a = np.bmat([[ent_a[1]], [ent_b[1]]])
    (sub_q, sub_r) = np.linalg.qr(curr_a, mode='complete')
    aa = sub_r[0:regular_b_size]
    bb = sub_r[regular_b_size:2 * regular_b_size]
    sub_q = _split_matrix(sub_q, 2)
    if transpose:
        return np.transpose(sub_q[0][0]), np.transpose(sub_q[1][0]), np.transpose(sub_q[0][1]), np.transpose(
            sub_q[1][1]), aa, bb
    else:
        return sub_q[0][0], sub_q[0][1], sub_q[1][0], sub_q[1][1], aa, bb


def _little_qr(a, type_a, b, type_b, b_size, transpose=False):
    sub_q00, sub_q01, sub_q10, sub_q11, aa, bb = _little_qr_task(a, type_a, b, type_b, b_size, transpose)
    return sub_q00, sub_q01, sub_q10, sub_q11, _type_block(OTHER), aa, _type_block(OTHER), bb


@constraint(computing_units="${computingUnits}")
@task(returns=(np.array, np.array))
def _multiply_single_block_task(a, type_a, b, type_b, c, type_c, b_size, transpose_b=False):
    if type_a[0][0] == ZEROS or type_b[0][0] == ZEROS:
        return type_c, c

    fun_a = [type_a, a]
    fun_b = [type_b, b]
    if type_c[0][0] == ZEROS:
        c = np.zeros((b_size[0], b_size[1]))
    elif type_c[0][0] == IDENTITY:
        c = np.identity(b_size[0])
    if fun_a[0][0][0] == IDENTITY:
        if fun_b[0][0][0] == IDENTITY:
            fun_b[1] = np.identity(b_size[0])
        if transpose_b:
            aux = np.transpose(fun_b[1])
        else:
            aux = fun_b[1]
        c += aux
        return _type_block(OTHER), c
    if fun_b[0][0][0] == IDENTITY:
        c += fun_a[1]
        return _type_block(OTHER), c
    if transpose_b:
        fun_b[1] = np.transpose(fun_b[1])

    c += (fun_a[1].dot(fun_b[1]))
    return _type_block(OTHER), c


def _multiply_single_block(a, type_a, b, type_b, c, type_c, b_size, transpose_b=False):
    return _multiply_single_block_task(a, type_a, b, type_b, c, type_c, b_size, transpose_b=transpose_b)


def _multiply_blocked(a, type_a, b, type_b, b_size, transpose_b=False):
    if transpose_b:
        new_b = []
        for i in range(len(b[0])):
            new_b.append([])
            for j in range(len(b)):
                new_b[i].append(b[j][i])
        b = new_b

    c = []
    type_c = []
    for i in range(len(a)):
        c.append([])
        type_c.append([])
        for j in range(len(b[0])):
            c[i].append(_empty_block(b_size))
            type_c[i].append(_type_block(ZEROS))
            for k in range(len(a[0])):
                type_c[i][j], c[i][j] = _multiply_single_block(
                    a[i][k], type_a[i][k],
                    b[k][j], type_b[k][j],
                    c[i][j], type_c[i][j],
                    b_size, transpose_b=transpose_b)

    return type_c, c


def _transpose_block(a, a_type):
    if a_type[0][0] == ZEROS or a_type[0][0] == IDENTITY:
        return a_type, a
    return _type_block(OTHER), np.transpose(a)


def _split_matrix(a, m_size):
    b_size = int(len(a) / m_size)
    split_matrix = [[None for m in range(m_size)] for m in range(m_size)]
    for i in range(m_size):
        for j in range(m_size):
            split_matrix[i][j] = a[i * b_size:(i + 1) * b_size, j * b_size:(j + 1) * b_size]
    return split_matrix
