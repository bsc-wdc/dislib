import numpy as np
from pycompss.api.api import compss_wait_on
from pycompss.api.constraint import constraint
from pycompss.api.task import task

from dislib.data.array import Array, identity, full

ZEROS = 0
IDENTITY = 1
OTHER = 2


def qr_blocked(a: Array, mkl_proc, overwrite_a=False):
    """ QR Decomposition (blocked / save memory).

    Parameters
    ----------
    A : ds-arrays
        Input ds-arrays.

    Returns
    -------
    out : ds-array

    Raises
    ------
    NotImplementedError
        If a or b are sparse.
    """
    assert(a._n_blocks[0] == a._n_blocks[1])

    b_size = a._reg_shape  # size of each block
    m_size = a._n_blocks[0]

    # create an identity matrix together with an auxiliary matrix
    # (one element of the auxiliary matrix per one block of the original matrix) that:
    # - has zeros when the block is filled with zeros
    # - has ones when the block is an identity matrix
    q, q_type = _gen_identity(max(a.shape), b_size, m_size)
    q._sparse = True

    if not overwrite_a:
        r = a.copy()
        r._sparse = True
    else:
        r = a

    # create an identity matrix together with an auxiliary matrix
    # (one block of the auxiliary matrix per one block of the original matrix) that:
    # is full of values of 2 in order to indicate that it is a normal matrix
    r_type = full((m_size, m_size), (1, 1), OTHER)

    for i in range(m_size):
        actQ_type, actQ, r_type._blocks[i][i], r._blocks[i][i] = _qr(r._blocks[i][i], r_type._blocks[i][i], mkl_proc, b_size, t=True)

        for j in range(m_size):
            q_type._blocks[j][i], q._blocks[j][i] = _dot(q._blocks[j][i], q_type._blocks[j][i], actQ, actQ_type, mkl_proc, transposeB=True)

        for j in range(i + 1, m_size):
            r_type._blocks[i][j], r._blocks[i][j] = _dot(actQ, actQ_type, r._blocks[i][j], r_type._blocks[i][j], mkl_proc)

        # Update values of the respective column
        for j in range(i + 1, m_size):
            subQ = [[np.array([0]), np.array([0])],
                    [np.array([0]), np.array([0])]]
            # setting the following to OTHER because of _little_qr returns this type of matrices
            subQ_type = [[_type_block(OTHER), _type_block(OTHER)],
                        [_type_block(OTHER), _type_block(OTHER)]]

            subQ[0][0], subQ[0][1], subQ[1][0], subQ[1][1], r_type._blocks[i][i], r._blocks[i][i], r_type._blocks[j][i], r._blocks[j][i] = _little_qr(
                r._blocks[i][i], r_type._blocks[i][i], r._blocks[j][i], r_type._blocks[j][i], mkl_proc,
                b_size, transpose=True)

            # Update values of the row for the value updated in the column
            for k in range(i + 1, m_size):
                [[r_type._blocks[i][k]], [r_type._blocks[j][k]]], [[r._blocks[i][k]], [r._blocks[j][k]]] = _multiply_blocked(
                    subQ,
                    subQ_type,
                    [[r._blocks[i][k]], [r._blocks[j][k]]],
                    [[r_type._blocks[i][k]], [r_type._blocks[j][k]]],
                    b_size,
                    mkl_proc
                )

            for k in range(m_size):
                [[q_type._blocks[k][i], q_type._blocks[k][j]]], [[q._blocks[k][i], q._blocks[k][j]]] = _multiply_blocked(
                    [[q._blocks[k][i], q._blocks[k][j]]],
                    [[q_type._blocks[k][i], q_type._blocks[k][j]]],
                    subQ,
                    subQ_type,
                    b_size,
                    mkl_proc,
                    transposeB=True
                )

    return q, r


def _set_mkl_num_threads(mkl_proc):
    import os
    os.environ["MKL_NUM_THREADS"] = str(mkl_proc)


def _gen_identity(n, b_size, m_size):
    a = identity(n, b_size, dtype=None)
    aux_a = identity(m_size, (1, 1), dtype=np.uint8)
    return a, aux_a


@constraint(ComputingUnits="${ComputingUnits}")
@task(returns=(np.array, np.array), on_failure='FAIL')
def _qr_task(a, a_type, mkl_proc, b_size, mode='reduced', t=False, **kwargs):
    from numpy.linalg import qr
    _set_mkl_num_threads(mkl_proc)
    if a_type[0, 0] == OTHER:
        q, r = qr(a, mode=mode)
    elif a_type[0, 0] == ZEROS:
        q, r = qr(np.zeros(b_size), mode=mode)
    else:
        q, r = qr(np.identity(max(b_size)), mode=mode)
    if t:
        q = np.transpose(q)
    return q, r


def _qr(a, a_type, mkl_proc, b_size, mode='reduced', t=False):
    Qaux, Raux = _qr_task(a, a_type, mkl_proc, b_size, mode=mode, t=t)
    return _type_block(OTHER), Qaux, _type_block(OTHER), Raux


@constraint(ComputingUnits="${ComputingUnits}")
@task(returns=list)
def _transpose_block_task(A, **kwargs):
    return np.transpose(A)


def _type_block(value):
    return np.full((1, 1), value, np.uint8)


def _empty_block(shape):
    return np.empty(shape, dtype=np.uint8)

#a=COLLECTION_IN, b=COLLECTION_IN,
@constraint(ComputingUnits="${ComputingUnits}")
@task(returns=(list,list))
def _dot(a, a_type, b, b_type, mkl_proc, transposeResult=False, transposeB=False, **kwargs):
    if a_type[0][0] == ZEROS:
        return _type_block(ZEROS), _empty_block(a.shape)
    if a_type[0][0] == IDENTITY:
        if transposeB and transposeResult:
            return b_type, b
        if transposeB or transposeResult:
            return _transpose_block(b, b_type)

        return b_type, b
    if b_type[0][0] == ZEROS:
        return _type_block(ZEROS), _empty_block(a.shape)
    if b_type[0][0] == IDENTITY:
        if transposeResult:
            return _transpose_block(a, a_type)
        return a_type, a
    result = _dot_task(a, b, mkl_proc, transposeResult=transposeResult, transposeB=transposeB)

    return _type_block(OTHER), result


def _dot_task(a, b, mkl_proc, transposeResult=False, transposeB=False, **kwargs):
    _set_mkl_num_threads(mkl_proc)
    if transposeB:
        b = np.transpose(b)
    if transposeResult:
        return np.transpose(np.dot(a, b))
    return np.dot(a, b)


@constraint(ComputingUnits="${ComputingUnits}")
@task(returns=(list, list, list, list, list, list))
def _little_qr_task(A, typeA, B, typeB, mkl_proc, b_size, transpose=False, **kwargs):
    # TODO what if blocks are not square
    regular_b_size = b_size[0]
    _set_mkl_num_threads(mkl_proc)
    entA = [typeA, A]
    entB = [typeB, B]
    for mat in [entA, entB]:
        if mat[0] == ZEROS:
            mat[1] = np.zeros((regular_b_size, regular_b_size))
        elif mat[0] == IDENTITY:
            mat[1] = np.identity(regular_b_size)
    currA = np.bmat([[entA[1]], [entB[1]]])
    (subQ, subR) = np.linalg.qr(currA, mode='complete')
    AA = subR[0:regular_b_size]
    BB = subR[regular_b_size:2 * regular_b_size]
    subQ = _split_matrix(subQ, 2)
    if transpose:
        return np.transpose(subQ[0][0]), np.transpose(subQ[1][0]), np.transpose(subQ[0][1]), np.transpose(
            subQ[1][1]), AA, BB
    else:
        return subQ[0][0], subQ[0][1], subQ[1][0], subQ[1][1], AA, BB


def _little_qr(a, type_a, b, type_b, mkl_proc, BSIZE, transpose=False):
    subQ00, subQ01, subQ10, subQ11, AA, BB = _little_qr_task(a, type_a, b, type_b, mkl_proc, BSIZE, transpose)
    return subQ00, subQ01, subQ10, subQ11, _type_block(OTHER), AA, _type_block(OTHER), BB


@constraint(ComputingUnits="${ComputingUnits}")
@task(returns=(list, list))
def _multiply_single_block_task(A, typeA, B, typeB, C, typeC, mkl_proc, b_size, transposeB=False, **kwargs):
    _set_mkl_num_threads(mkl_proc)

    if typeA[0][0] == ZEROS or typeB[0][0] == ZEROS:
        # TODO to check it in the case of not square blocks
        # bool type to save memory as the matrix is filled with Nones
        return typeC, C

    funA = [typeA, A]
    funB = [typeB, B]
    if typeC[0][0] == ZEROS:
        C = np.zeros((b_size[0], b_size[1]))
    elif typeC[0][0] == IDENTITY:
        C = np.identity(b_size[0])
    if funA[0][0][0] == IDENTITY:
        if funB[0][0][0] == IDENTITY:
            funB[1] = np.identity(b_size[0])
        if transposeB:
            aux = np.transpose(funB[1])
        else:
            aux = funB[1]
        C += aux
        return _type_block(OTHER), C
    if funB[0][0][0] == IDENTITY:
        C += funA[1]
        return _type_block(OTHER), C
    if transposeB:
        funB[1] = np.transpose(funB[1])

    C += (funA[1].dot(funB[1]))
    return _type_block(OTHER), C


def _multiply_single_block(a, type_a, b, type_b, c, type_c, mkl_proc, b_size, transposeB=False):
    return _multiply_single_block_task(a, type_a, b, type_b, c, type_c, mkl_proc, b_size, transposeB=transposeB)


def _multiply_blocked(A, type_a, B, type_b, b_size, mkl_proc, transposeB=False):
    if transposeB:
        newB = []
        for i in range(len(B[0])):
            newB.append([])
            for j in range(len(B)):
                newB[i].append(B[j][i])
        B = newB

    C = []
    type_c = []
    for i in range(len(A)):
        C.append([])
        type_c.append([])
        for j in range(len(B[0])):
            C[i].append(_empty_block(b_size))
            type_c[i].append(_type_block(ZEROS))
            for k in range(len(A[0])):
                type_c[i][j], C[i][j] = _multiply_single_block(
                    A[i][k], type_a[i][k],
                    B[k][j], type_b[k][j],
                    C[i][j], type_c[i][j],
                    mkl_proc, b_size, transposeB=transposeB)

    return type_c, C


def _transpose_block(a, a_type):
    if a_type[0][0] == ZEROS or a_type[0][0] == IDENTITY:
        return a_type, a
    return _type_block(OTHER), _transpose_block_task(a)


def _split_matrix(A, m_size):
    bSize = int(len(A) / m_size)
    splittedMatrix = [[None for m in range(m_size)] for m in range(m_size)]
    for i in range(m_size):
        for j in range(m_size):
            splittedMatrix[i][j] = A[i * bSize:(i + 1) * bSize, j * bSize:(j + 1) * bSize]
    return splittedMatrix
