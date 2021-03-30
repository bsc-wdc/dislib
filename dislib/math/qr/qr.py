import numpy as np
from pycompss.api.constraint import constraint
from pycompss.api.task import task


def qr_blocked(A, mkl_proc, m_size, b_size, overwrite_a=False):
    """ QR Decomposition (Sparse).

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

    Q = _gen_identity(m_size, b_size, mkl_proc)

    if not overwrite_a:
        R = _copy_blocked(A)
    else:
        R = A

    for i in range(m_size):

        actQ, R[i][i] = _qr(R[i][i], mkl_proc, b_size, transpose=True)

        for j in range(m_size):
            Q[j][i] = _dot(Q[j][i], actQ, mkl_proc, transposeB=True)
            # Q[i][j] = dot(actQ, Q[i][j])

        for j in range(i + 1, m_size):
            R[i][j] = _dot(actQ, R[i][j], mkl_proc)

        # Update values of the respective column
        for j in range(i + 1, m_size):
            subQ = [[np.matrix(np.array([0])), np.matrix(np.array([0]))],
                    [np.matrix(np.array([0])), np.matrix(np.array([0]))]]
            subQ[0][0], subQ[0][1], subQ[1][0], subQ[1][1], R[i][i], R[j][i] = _little_qr(R[i][i], R[j][i], mkl_proc,
                                                                                        b_size, transpose=True)
            # subQ = blockedTranspose(subQ, mkl_proc)
            # Update values of the row for the value updated in the column
            for k in range(i + 1, m_size):
                [[R[i][k]], [R[j][k]]] = _multiply_blocked(subQ, [[R[i][k]], [R[j][k]]], b_size, mkl_proc)

            for k in range(m_size):
                [[Q[k][i], Q[k][j]]] = _multiply_blocked([[Q[k][i], Q[k][j]]], subQ, b_size, mkl_proc, transposeB=True)
                # [[Q[k][i], Q[k][j]]] = multiplyBlocked([[Q[k][i], Q[k][j]]], blockedTranspose(subQ, mkl_proc), BSIZE, mkl_proc)
    # Q = blockedTranspose(Q,mkl_proc)
    return Q, R


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


def _gen_identity(m_size, b_size, mkl_proc):
    A = []
    for i in range(m_size):
        A.append([])
        for j in range(0, i):
            A[i].append(_create_block(b_size, mkl_proc, type='zeros'))
        A[i].append(_create_block(b_size, mkl_proc, type='identity'))
        for j in range(i + 1, m_size):
            A[i].append(_create_block(b_size, mkl_proc, type='zeros'))
    return A


@constraint(ComputingUnits="${ComputingUnits}")
@task(returns=(list, list))
def _qr_task(A, mkl_proc, b_size, type, mode='reduced', transpose=False):
    from numpy.linalg import qr
    _set_mkl_num_threads(mkl_proc)
    if type == 'random':
        Q, R = qr(A, mode=mode)
    elif type == 'zeros':
        Q, R = qr(np.matrix(np.zeros((b_size, b_size))), mode=mode)
    else:
        Q, R = qr(np.matrix(np.identity(b_size)), mode=mode)
    if transpose:
        Q = np.transpose(Q)
    return Q, R


def _qr(A, mkl_proc, b_size, mode='reduced', transpose=False):
    Qaux, Raux = _qr_task(A[1], mkl_proc, b_size, A[0], mode=mode, transpose=transpose)
    return ['random', Qaux], ['random', Raux]


@constraint(ComputingUnits="${ComputingUnits}")
@task(returns=list)
def _transpose_block_task(A):
    return np.transpose(A)


def _dot(A, B, mkl_proc, transposeResult=False, transposeB=False):
    if A[0] == 'zeros':
        return ['zeros', []]
    if A[0] == 'identity':
        if transposeB and transposeResult:
            return B
        if transposeB or transposeResult:
            return _transpose_block(B)
        return B
    if B[0] == 'zeros':
        return ['zeros', []]
    if B[0] == 'identity':
        if transposeResult:
            return _transpose_block(A)
        return A
    return ['random', _dot_task(A[1], B[1], mkl_proc, transposeResult=transposeResult, transposeB=transposeB)]


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
        if mat[0] == 'zeros':
            mat[1] = np.matrix(np.zeros((b_size, b_size)))
        elif mat[1] == 'identity':
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
    return ['random', subQ00], ['random', subQ01], ['random', subQ10], ['random', subQ11], ['random', AA], ['random',
                                                                                                            BB]


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
    if A[0] == 'zeros' or B[0] == 'zeros':
        return C
    C[1] = _multiply_single_block_task(A[1], A[0], B[1], B[0], C[1], C[0], mkl_proc, b_size, transposeB=transposeB)
    C = ['random', C[1]]
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
            C[i].append(['zeros', []])
            for k in range(len(A[0])):
                C[i][j] = _multiply_single_block(A[i][k], B[k][j], C[i][j], mkl_proc, b_size, transposeB=transposeB)
    return C


def _transpose_block(A):
    if A[0] == 'zeros' or A[0] == 'identity':
        return A
    return ['random', _transpose_block_task(A[1])]


def _copy_blocked(A, transpose=False):
    B = []
    for i in range(len(A)):
        B.append([])
        for j in range(len(A[0])):
            B[i].append(np.matrix([0]))
    for i in range(len(A)):
        for j in range(len(A[0])):
            if transpose:
                B[j][i] = A[i][j]
            else:
                B[i][j] = [A[i][j][0], A[i][j][1]]
    return B


def _split_matrix(A, m_size):
    bSize = len(A) / m_size
    splittedMatrix = [[None for m in range(m_size)] for m in range(m_size)]
    for i in range(m_size):
        for j in range(m_size):
            splittedMatrix[i][j] = np.matrix(A[i * bSize:(i + 1) * bSize, j * bSize:(j + 1) * bSize])
    return splittedMatrix
