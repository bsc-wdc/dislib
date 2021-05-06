import numpy as np

from pycompss.api.constraint import constraint
from pycompss.api.parameter import INOUT, IN
from pycompss.api.task import task

from dislib.data.array import Array, identity
from dislib.math.qr import save_memory as save_mem


def qr_blocked(a: Array, overwrite_a=False, save_memory=False):
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
    if save_memory:
        return save_mem.qr_blocked(a, overwrite_a)

    assert(a._n_blocks[0] == a._n_blocks[1])

    b_size = a._reg_shape  # size of each block
    m_size = a._n_blocks[0]

    # create an identity matrix together with an auxiliary matrix
    # (one element of the auxiliary matrix per one block of the original matrix) that:
    # - has zeros when the block is filled with zeros
    # - has ones when the block is an identity matrix
    q = identity(max(a.shape), b_size, dtype=None)

    if not overwrite_a:
        r = a.copy()
    else:
        r = a

    for i in range(m_size):
        actQ, r._blocks[i][i] = _qr(r._blocks[i][i], t=True)

        for j in range(m_size):
            q._blocks[j][i] = _dot_task(q._blocks[j][i], actQ, transposeB=True)

        for j in range(i + 1, m_size):
            r._blocks[i][j] = _dot_task(actQ, r._blocks[i][j])

        # Update values of the respective column
        for j in range(i + 1, m_size):
            subQ = [[np.array([0]), np.array([0])],
                    [np.array([0]), np.array([0])]]

            subQ[0][0], subQ[0][1], subQ[1][0], subQ[1][1], r._blocks[i][i], r._blocks[j][i] = _little_qr(
                r._blocks[i][i], r._blocks[j][i],
                b_size, transpose=True)

            # Update values of the row for the value updated in the column
            for k in range(i + 1, m_size):
                [[r._blocks[i][k]], [r._blocks[j][k]]] = _multiply_blocked(
                    subQ,
                    [[r._blocks[i][k]], [r._blocks[j][k]]],
                    b_size
                )

            for k in range(m_size):
                [[q._blocks[k][i], q._blocks[k][j]]] = _multiply_blocked(
                    [[q._blocks[k][i], q._blocks[k][j]]],
                    subQ,
                    b_size,
                    transposeB=True
                )

    return q, r


@constraint(ComputingUnits="${ComputingUnits}")
@task(returns=(np.array, np.array), on_failure='FAIL')
def _qr_task(a, mode='reduced', t=False, **kwargs):
    from numpy.linalg import qr
    q, r = qr(a, mode=mode)
    if t:
        q = np.transpose(q)
    return q, r


def _qr(a, mode='reduced', t=False):
    return _qr_task(a, mode=mode, t=t)


@constraint(ComputingUnits="${ComputingUnits}")
@task(returns=list)
def _transpose_block_task(A, **kwargs):
    return np.transpose(A)


@constraint(ComputingUnits="${ComputingUnits}")
@task(returns=list)
def _dot_task(a, b, transposeResult=False, transposeB=False, **kwargs):
    if transposeB:
        b = np.transpose(b)
    if transposeResult:
        return np.transpose(np.dot(a, b))
    return np.dot(a, b)


@constraint(ComputingUnits="${ComputingUnits}")
@task(returns=(list, list, list, list, list, list))
def _little_qr_task(A, B, b_size, transpose=False, **kwargs):
    # TODO what if blocks are not square
    regular_b_size = b_size[0]
    currA = np.bmat([[A], [B]])
    (subQ, subR) = np.linalg.qr(currA, mode='complete')
    AA = subR[0:regular_b_size]
    BB = subR[regular_b_size:2 * regular_b_size]
    subQ = _split_matrix(subQ, 2)
    if transpose:
        return np.transpose(subQ[0][0]), np.transpose(subQ[1][0]), np.transpose(subQ[0][1]), np.transpose(
            subQ[1][1]), AA, BB
    else:
        return subQ[0][0], subQ[0][1], subQ[1][0], subQ[1][1], AA, BB


def _little_qr(a, b, BSIZE, transpose=False):
    subQ00, subQ01, subQ10, subQ11, AA, BB = _little_qr_task(a, b, BSIZE, transpose)
    return subQ00, subQ01, subQ10, subQ11, AA, BB


@constraint(ComputingUnits="${ComputingUnits}")
@task(A=IN, B=IN, C=INOUT)
def _multiply_single_block_task(A, B, C, transposeB=False, **kwargs):
    if transposeB:
        B = np.transpose(B)
    C += (A.dot(B))


def _multiply_blocked(A, B, b_size, transposeB=False):
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
            C[i].append(np.zeros(b_size))
            for k in range(len(A[0])):
                _multiply_single_block_task(A[i][k], B[k][j], C[i][j], transposeB=transposeB)

    return C


def _transpose_block(a):
    return _transpose_block_task(a)


def _split_matrix(A, m_size):
    bSize = int(len(A) / m_size)
    splittedMatrix = [[None for m in range(m_size)] for m in range(m_size)]
    for i in range(m_size):
        for j in range(m_size):
            splittedMatrix[i][j] = A[i * bSize:(i + 1) * bSize, j * bSize:(j + 1) * bSize]
    return splittedMatrix
