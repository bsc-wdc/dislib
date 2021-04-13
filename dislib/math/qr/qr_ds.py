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
    n = a.shape[0] * a.shape[1]  # number of elements in the matrix
    print("b_size = ", b_size)
    print("m_size = ", m_size)
    print("n = ", n)

    # create an identity matrix together with an auxiliary matrix
    # (one block of the auxiliary matrix per one block of the original matrix) that:
    # - has zeros when the block is filled with zeros
    # - has ones when the block is an identity matrix
    q, q_type = _gen_identity(n, b_size, m_size)

    print("identity generated")

    if not overwrite_a:
        r = a.copy()
    else:
        r = a

    # create an identity matrix together with an auxiliary matrix
    # (one block of the auxiliary matrix per one block of the original matrix) that:
    # is full of values of 2 in order to indicate that it is a normal matrix
    r_type = full((m_size, m_size), (1, 1), OTHER)

    for i in range(m_size):
        print("step1")
        print("r.shape", r.shape)
        print("r._n_blocks", r._n_blocks)
        print("all blocks", compss_wait_on(r._blocks))
        print("i blocks", compss_wait_on(r._blocks[i]))
        print("ii blocks", compss_wait_on(r._blocks[i][i]))
        [actQ_type, actQ], [r_type[i][i], r[i][i]] = _qr([r_type._blocks[i][i], r._blocks[i][i]], mkl_proc, b_size, transpose=True)
        print("step2")
        for j in range(m_size):
            q_type._blocks[j][i], q._blocks[j][i] = _dot([q_type._blocks[j][i], q._blocks[j][i]], [actQ_type, actQ], mkl_proc, transposeB=True)
            # Q[i][j] = dot(actQ, Q[i][j])
        print("step3")
        for j in range(i + 1, m_size):
            r_type._blocks[i][j], r._blocks[i][j] = _dot([actQ_type, actQ], [r_type._blocks[i][j], r._blocks[i][j]], mkl_proc)

        '''
        # Update values of the respective column
        for j in range(i + 1, m_size):
            subQ = [[np.matrix(np.array([0])), np.matrix(np.array([0]))],
                    [np.matrix(np.array([0])), np.matrix(np.array([0]))]]
            subQ[0][0], subQ[0][1], subQ[1][0], subQ[1][1], (r_type._blocks[i][i], r._blocks[i][i]), (r_type._blocks[j][i], r._blocks[j][i]) = _little_qr(
                (r_type._blocks[i][i], r._blocks[i][i]), (r_type._blocks[j][i], r._blocks[j][i]), mkl_proc,
                b_size, transpose=True)
            # subQ = blockedTranspose(subQ, mkl_proc)
            # Update values of the row for the value updated in the column
            for k in range(i + 1, m_size):
                [[r._blocks[i][k]], [r._blocks[j][k]]] = _multiply_blocked((OTHER, subQ), [[r._blocks[i][k]], [r._blocks[j][k]]], b_size, mkl_proc)

            for k in range(m_size):
                [[q._blocks[k][i], q._blocks[k][j]]] = _multiply_blocked([[q._blocks[k][i], q._blocks[k][j]]], subQ, b_size, mkl_proc, transposeB=True)
                # [[Q[k][i], Q[k][j]]] = multiplyBlocked([[Q[k][i], Q[k][j]]], blockedTranspose(subQ, mkl_proc), BSIZE, mkl_proc)
        '''
    # Q = blockedTranspose(Q,mkl_proc)
    return q, r


def _set_mkl_num_threads(mkl_proc):
    import os
    os.environ["MKL_NUM_THREADS"] = str(mkl_proc)


@constraint(ComputingUnits="${ComputingUnits}")
@task(returns=list)
def _create_block_task(b_size, mkl_proc):
    _set_mkl_num_threads(mkl_proc)
    return np.matrix(np.random.random((b_size, b_size)), dtype=np.double, copy=False)


def _create_block(b_size, mkl_proc, type='random'):
    if type == 'zeros':
        block = []
    elif type == 'identity':
        block = []
    else:
        block = _create_block_task(b_size, mkl_proc)
    return [type, block]


def _gen_identity(n, b_size, m_size):
    a = identity(n, b_size, dtype=None)
    print("identity1 ready")
    print("generating identity2 for", m_size)
    aux_a = identity(m_size, (1, 1), dtype=np.uint8)
    print("identity2 ready")
    return a, aux_a
    '''
    A = []
    for i in range(m_size):
        A.append([])
        for j in range(0, i):
            A[i].append(_create_block(b_size, mkl_proc, type='zeros'))
        A[i].append(_create_block(b_size, mkl_proc, type='identity'))
        for j in range(i + 1, m_size):
            A[i].append(_create_block(b_size, mkl_proc, type='zeros'))
    return A
    '''


@constraint(ComputingUnits="${ComputingUnits}")
@task(returns=(list, list))
def _qr_task(A, mkl_proc, b_size, type, mode='reduced', transpose=False):
    #A = compss_wait_on(A)
    #type = compss_wait_on(type)
    print("QR A = ", A, " type = ", type)
    from numpy.linalg import qr
    _set_mkl_num_threads(mkl_proc)
    if type == OTHER:
        q, r = qr(A, mode=mode)
    elif type == ZEROS:
        q, r = qr(np.matrix(np.zeros((b_size, b_size))), mode=mode)
    else:
        q, r = qr(np.matrix(np.identity(b_size)), mode=mode)
    if transpose:
        q = np.transpose(q)
    return q, r


def _qr(A, mkl_proc, b_size, mode='reduced', transpose=False):
    Qaux, Raux = _qr_task(A[1], mkl_proc, b_size, A[0], mode=mode, transpose=transpose)
    return [OTHER, Qaux], [OTHER, Raux]


@constraint(ComputingUnits="${ComputingUnits}")
@task(returns=list)
def _transpose_block_task(A):
    return np.transpose(A)


@constraint(ComputingUnits="${ComputingUnits}")
@task(returns=list)
def _dot(A, B, mkl_proc, transposeResult=False, transposeB=False):
    if A[0] == ZEROS:
        return ZEROS, []
    if A[0] == IDENTITY:
        if transposeB and transposeResult:
            return B
        if transposeB or transposeResult:
            return _transpose_block(A[1], A[0])
        return B
    if B[0] == ZEROS:
        return (ZEROS, [])
    if B[0] == IDENTITY:
        if transposeResult:
            return _transpose_block(A[1], A[0])
        return A
    return [OTHER, _dot_task(A[1], B[1], mkl_proc, transposeResult=transposeResult, transposeB=transposeB)]


@constraint(ComputingUnits="${ComputingUnits}")
@task(returns=list)
def _dot_task(A, B, mkl_proc, transposeResult=False, transposeB=False):
    _set_mkl_num_threads(mkl_proc)
    if transposeB:
        B = np.transpose(B)
    if transposeResult:
        return np.transpose(np.dot(A, B))
    return np.dot(A, B)


@constraint(ComputingUnits="${ComputingUnits}")
@task(returns=(list, list, list, list, list, list))
def _little_qr_task(A, typeA, B, typeB, mkl_proc, b_size, transpose=False):
    _set_mkl_num_threads(mkl_proc)
    entA = [typeA, A]
    entB = [typeB, B]
    for mat in [entA, entB]:
        if mat[0] == ZEROS:
            mat[1] = np.matrix(np.zeros((b_size, b_size)))
        elif mat[1] == IDENTITY:
            mat[1] = np.matrix(np.identity(b_size))
    currA = np.bmat([[entA[1]], [entB[1]]])
    (subQ, subR) = np.linalg.qr(currA, mode='complete')
    AA = subR[0:b_size]
    BB = subR[b_size:2 * b_size]
    subQ = _split_matrix(subQ, 2)
    if transpose:
        return np.transpose(subQ[0][0]), np.transpose(subQ[1][0]), np.transpose(subQ[0][1]), np.transpose(
            subQ[1][1]), AA, BB
    else:
        return subQ[0][0], subQ[0][1], subQ[1][0], subQ[1][1], AA, BB


def _little_qr(A, B, mkl_proc, BSIZE, transpose=False):
    subQ00, subQ01, subQ10, subQ11, AA, BB = _little_qr_task(A[1], A[0], B[1], B[0], mkl_proc, BSIZE, transpose)
    return subQ00, subQ01, subQ10, subQ11, [OTHER, AA], [OTHER, BB]


@constraint(ComputingUnits="${ComputingUnits}")
@task(returns=(list))
def _multiply_single_block_task(A, typeA, B, typeB, C, typeC, mkl_proc, b_size, transposeB=False):
    _set_mkl_num_threads(mkl_proc)
    funA = [typeA, A]
    funB = [typeB, B]
    if typeC == 'zeros':
        C = np.matrix(np.zeros((b_size, b_size)))
    elif typeC == 'identity':
        C = np.matrix(np.identity(b_size))
    if funA[0] == 'identity':
        if funB[0] == 'identity':
            funB[1] = np.matrix(np.identity(b_size))
        if transposeB:
            aux = np.transpose(funB[1])
        else:
            aux = funB[1]
        C += aux
        return C
    if funB[0] == 'identity':
        C += funA[1]
        return C
    if transposeB:
        funB[1] = np.transpose(funB[1])
    C += (funA[1] * funB[1])
    return C


def _multiply_single_block(A, B, C, mkl_proc, b_size, transposeB=False):
    if A[0] == ZEROS or B[0] == ZEROS:
        return C
    C[1] = _multiply_single_block_task(A[1], A[0], B[1], B[0], C[1], C[0], mkl_proc, b_size, transposeB=transposeB)
    C = [OTHER, C[1]]
    return C


def _multiply_blocked(A, B, b_size, mkl_proc, transposeB=False):
    if transposeB:
        newB = []
        for i in range(len(B[0])):
            newB.append([])
            for j in range(len(B)):
                newB[i].append(B[j][i])
        B = newB
    C = []
    for i in range(len(A)):
        C.append([])
        for j in range(len(B[0])):
            C[i].append([ZEROS, []])
            for k in range(len(A[0])):
                C[i][j] = _multiply_single_block(A[i][k], B[k][j], C[i][j], mkl_proc, b_size, transposeB=transposeB)
    return C


def _transpose_block(a, a_type):
    if a_type == ZEROS or a_type == IDENTITY:
        return [a, a_type]
    return [OTHER, _transpose_block_task(a)]


def _split_matrix(A, m_size):
    bSize = len(A) / m_size
    splittedMatrix = [[None for m in range(m_size)] for m in range(m_size)]
    for i in range(m_size):
        for j in range(m_size):
            splittedMatrix[i][j] = np.matrix(A[i * bSize:(i + 1) * bSize, j * bSize:(j + 1) * bSize])
    return splittedMatrix
