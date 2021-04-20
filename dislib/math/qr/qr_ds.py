import numpy as np
from pycompss.api.api import compss_wait_on
from pycompss.api.constraint import constraint
from pycompss.api.parameter import COLLECTION_IN
from pycompss.api.task import task
from pycompss.api.dummy.task import task  # for debugging purposes

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
    # (one element of the auxiliary matrix per one block of the original matrix) that:
    # - has zeros when the block is filled with zeros
    # - has ones when the block is an identity matrix
    q, q_type = _gen_identity(max(a.shape), b_size, m_size)

    print("identity generated")

    if not overwrite_a:
        r = a.copy()
    else:
        r = a

    # create an identity matrix together with an auxiliary matrix
    # (one block of the auxiliary matrix per one block of the original matrix) that:
    # is full of values of 2 in order to indicate that it is a normal matrix
    r_type = full((m_size, m_size), (1, 1), OTHER)

    print("q shape 0:", q.shape)
    print("r shape 0:", r.shape)

    for i in range(m_size):
        print("step1")
        print("r.shape", r.shape)
        print("r._n_blocks", r._n_blocks)
        #r._blocks = compss_wait_on(r._blocks)
        #print("all blocks", r._blocks)
        #print("i blocks", r._blocks[i])
        #print("ii blocks", r._blocks[i][i])
        print("before the main loop")
        #r._blocks[i][i] = compss_wait_on(r._blocks[i][i])
        #r_type._blocks[i][i] = compss_wait_on(r_type._blocks[i][i])
        #print("r_type._blocks[i][i]", r_type._blocks[i][i])
        #print("r._blocks[i][i]", r._blocks[i][i])
        r._blocks = compss_wait_on(r._blocks)
        r_type._blocks = compss_wait_on(r_type._blocks)
        print("1.2 r._blocks.len", r._n_blocks[0], r._n_blocks[1])
        print("1.1 r_type._blocks.len", r_type._n_blocks[0], r_type._n_blocks[1])
        print("1.2 r content before:", r.collect())
        print("1.2 r_type content before:", r_type.collect())
        actQ_type, actQ, [[r_type[i, i]]], r._blocks[i][i] = _qr(r._blocks[i][i], r_type._blocks[i][i], mkl_proc, b_size, t=True)
        r._blocks = compss_wait_on(r._blocks)
        r_type._blocks = compss_wait_on(r_type._blocks)
        print("1.2 r._blocks.len", r._n_blocks[0], r._n_blocks[1])
        print("1.2 r_type._blocks.len", r_type._n_blocks[0], r_type._n_blocks[1])
        print("1.2 r content after:", r.collect())
        print("1.2 r_type content after:", r_type.collect())

        print("main loop", i)
        actQ = compss_wait_on(actQ)
        actQ_type = compss_wait_on(actQ_type)
        print("actQ_type", actQ_type)
        print("actQ", actQ)
        #r._blocks[i][i] = compss_wait_on(r._blocks[i][i])
        #r_type._blocks[i][i] = compss_wait_on(r_type._blocks[i][i])
        print("R.type[i][i]", r_type._blocks[i][i])
        print("R[i][i]", r._blocks[i][i])
        #actQ_type, actQ, r_type[i][i], r[i][i] = _qr(r._blocks[i][i], r_type._blocks[i][i], mkl_proc, b_size, t=True)
        print("q shape 1:", q.shape)
        print("r shape 1:", r.shape)
        print("step2")
        for j in range(m_size):
            print("loop2", j)
            #print(q._blocks[j][i], q_type._blocks[j][i])
            q._blocks = compss_wait_on(q._blocks)
            q_type._blocks = compss_wait_on(q_type._blocks)
            print("2.1 q._blocks.len", len(q._blocks), len(q._blocks[0]))
            print("2.2 q_type._blocks.len", len(q_type._blocks), len(q_type._blocks[0]))
            print("2. q content before:", q.collect())
            print("2. q_type content before:", q_type.collect())
            #print(q._blocks[j][i], q_type._blocks[j][i])
            print("2. q._blocks[j][i] content before:", q._blocks[j][i])
            print("2. q_type._blocks[j][i] content before:", q_type._blocks[j][i])
            print("2. actQ:", actQ)
            print("2. actQ_type:", actQ_type)
            q_type._blocks[j][i], q._blocks[j][i] = _dot(q._blocks[j][i], q_type._blocks[j][i], actQ, actQ_type, mkl_proc, transposeB=True)
            q._blocks = compss_wait_on(q._blocks)
            q_type._blocks = compss_wait_on(q_type._blocks)
            print("2. q shape 2:", q.shape)
            print("2. q content after:", q.collect())
            print("2. q_type content after:", q_type.collect())
            # Q[i][j] = dot(actQ, Q[i][j])
        print("step3")
        for j in range(i + 1, m_size):
            print("loop3", j)
            r._blocks = compss_wait_on(r._blocks)
            r_type._blocks = compss_wait_on(r_type._blocks)
            r_type._blocks[i][j], r._blocks[i][j] = _dot(actQ, actQ_type, r._blocks[i][j], r_type._blocks[i][j], mkl_proc)
            r._blocks  = compss_wait_on(r._blocks )
            r_type._blocks  = compss_wait_on(r_type._blocks )
            print("3. q shape 3:", q.shape)
            print("3. r shape 3:", r.shape)

        # Update values of the respective column
        for j in range(i + 1, m_size):
            print("loop4", j)
            subQ = [[np.matrix(np.array([0])), np.matrix(np.array([0]))],
                    [np.matrix(np.array([0])), np.matrix(np.array([0]))]]
            # setting the following to OTHER because of _little_qr returns this type of matrices
            subQ_type = [[OTHER, OTHER],
                        [OTHER, OTHER]]
            r._blocks = compss_wait_on(r._blocks)
            r_type._blocks = compss_wait_on(r_type._blocks)
            subQ[0][0], subQ[0][1], subQ[1][0], subQ[1][1], r_type._blocks[i][i], r._blocks[i][i], r_type._blocks[j][i], r._blocks[j][i] = _little_qr(
                r._blocks[i][i], r_type._blocks[i][i], r._blocks[j][i], r_type._blocks[j][i], mkl_proc,
                b_size, transpose=True)
            r._blocks = compss_wait_on(r._blocks)
            r_type._blocks = compss_wait_on(r_type._blocks)
            print("4. q shape 4:", q.shape)
            print("4. r shape 4:", r.shape)
            # subQ = blockedTranspose(subQ, mkl_proc)
            # Update values of the row for the value updated in the column
            for k in range(i + 1, m_size):
                [[r_type._blocks[i][k]], [r_type._blocks[j][k]]], [[r._blocks[i][k]], [r._blocks[j][k]]] = _multiply_blocked(
                    subQ, subQ_type, [[r._blocks[i][k]], [r._blocks[j][k]]], [[r_type._blocks[i][k]], [r_type._blocks[j][k]]], b_size, mkl_proc
                )
                r._blocks = compss_wait_on(r._blocks)
                r_type._blocks = compss_wait_on(r_type._blocks)
                print("4. q shape 5:", q.shape)
                print("4. r shape 5:", r.shape)

            q._blocks = compss_wait_on(q._blocks)
            q_type._blocks = compss_wait_on(q_type._blocks)

            for k in range(m_size):
                [[[q_type._blocks[k][i], q_type._blocks[k][j]]]], [[[q._blocks[k][i], q._blocks[k][j]]]] = _multiply_blocked(
                    [[q._blocks[k][i], q._blocks[k][j]]], [[q_type._blocks[k][i], q_type._blocks[k][j]]], subQ, subQ_type, b_size, mkl_proc, transposeB=True
                )
                q._blocks = compss_wait_on(q._blocks)
                q_type._blocks = compss_wait_on(q_type._blocks)
                print("4. q shape 6:", q.shape)
                print("4. r shape 6:", r.shape)
                # [[Q[k][i], Q[k][j]]] = multiplyBlocked([[Q[k][i], Q[k][j]]], blockedTranspose(subQ, mkl_proc), BSIZE, mkl_proc)

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
@task(a=COLLECTION_IN, returns=(list, list))
def _qr_task(a, a_type, mkl_proc, b_size, mode='reduced', t=False, **kwargs):
    #a = compss_wait_on(a)
    #a_type = compss_wait_on(a_type)
    print("QR A = ", a, " type = ", a_type)
    from numpy.linalg import qr
    _set_mkl_num_threads(mkl_proc)
    if a_type[0, 0] == OTHER:
        q, r = qr(a, mode=mode)
    elif a_type[0, 0] == ZEROS:
        q, r = qr(np.matrix(np.zeros(b_size)), mode=mode)
    else:
        q, r = qr(np.matrix(np.identity(max(b_size))), mode=mode)
    if t:
        q = np.transpose(q)
    return q, r


def _qr(a, a_type, mkl_proc, b_size, mode='reduced', t=False):
    print("executing the qr task")
    Qaux, Raux = _qr_task(a, a_type, mkl_proc, b_size, mode=mode, t=t)
    return [[OTHER]], Qaux, [[OTHER]], Raux


@constraint(ComputingUnits="${ComputingUnits}")
@task(returns=list)
def _transpose_block_task(A, **kwargs):
    return np.transpose(A)


@constraint(ComputingUnits="${ComputingUnits}")
@task(a=COLLECTION_IN, b=COLLECTION_IN, returns=(list,list))
def _dot(a, a_type, b, b_type, mkl_proc, transposeResult=False, transposeB=False, **kwargs):
    print("DOT1", a, "of type", a_type)
    print("DOT2", b, "of type", b_type)
    if a_type[0, 0] == ZEROS:
        print("DOT RESULT: ZERO 1")
        return [[ZEROS]], []
    if a_type[0, 0] == IDENTITY:
        if transposeB and transposeResult:
            print("DOT RESULT b1:", b_type, b)
            return b_type, b
        if transposeB or transposeResult:
            print("DOT RESULT b2:", _transpose_block(b, b_type))
            return _transpose_block(b, b_type)

        print("DOT RESULT b3:", b_type, b)
        return b_type, b
    if b_type[0, 0] == ZEROS:
        print("DOT RESULT: ZERO 2")
        return [[ZEROS]], []
    if b_type[0, 0] == IDENTITY:
        if transposeResult:
            print("DOT RESULT: ID 1", _transpose_block(a, a_type))
            return _transpose_block(a, a_type)
        print("DOT RESULT: ID 2", a, a_type)
        return a_type, a
    result = _dot_task(a, b, mkl_proc, transposeResult=transposeResult, transposeB=transposeB)

    print("DOT RESULT:", result)

    return [[OTHER]], result


def _dot_task(a, b, mkl_proc, transposeResult=False, transposeB=False, **kwargs):
    _set_mkl_num_threads(mkl_proc)
    if transposeB:
        b = np.transpose(b)
    if transposeResult:
        return np.transpose(np.dot(a, b))
    print("_dot_task")
    print(a, b)
    return np.dot(a, b)


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


def _little_qr(a, type_a, b, type_b, mkl_proc, BSIZE, transpose=False):
    subQ00, subQ01, subQ10, subQ11, AA, BB = _little_qr_task(a, type_a, b, type_b, mkl_proc, BSIZE, transpose)
    return subQ00, subQ01, subQ10, subQ11, OTHER, AA, OTHER, BB


@constraint(ComputingUnits="${ComputingUnits}")
@task(returns=(list))
def _multiply_single_block_task(A, typeA, B, typeB, C, typeC, mkl_proc, b_size, transposeB=False):
    _set_mkl_num_threads(mkl_proc)

    if typeA[0, 0] == ZEROS or typeB[0, 0] == ZEROS:
        return typeC, C

    funA = [typeA, A]
    funB = [typeB, B]
    if typeC == ZEROS:
        C = np.matrix(np.zeros((b_size, b_size)))
    elif typeC == IDENTITY:
        C = np.matrix(np.identity(b_size))
    if funA[0] == IDENTITY:
        if funB[0] == IDENTITY:
            funB[1] = np.matrix(np.identity(b_size))
        if transposeB:
            aux = np.transpose(funB[1])
        else:
            aux = funB[1]
        C += aux
        return [[OTHER]], C
    if funB[0] == IDENTITY:
        C += funA[1]
        return [[OTHER]], C
    if transposeB:
        funB[1] = np.transpose(funB[1])
    C += (funA[1] * funB[1])
    return [[OTHER]], C


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
            C[i].append([])
            type_c[i].append(ZEROS)
            for k in range(len(A[0])):
                type_c[i][j], C[i][j] = _multiply_single_block(
                    A[i][k], type_a[i][k],
                    B[k][j], type_b[k][j],
                    C[i][j], type_c[i][j],
                    mkl_proc, b_size, transposeB=transposeB)
    return type_c, C


def _transpose_block(a, a_type):
    print("type:", a_type)
    if a_type[0][0] == ZEROS or a_type[0][0] == IDENTITY:
        return a_type, a
    return [[OTHER]], _transpose_block_task(a)


def _split_matrix(A, m_size):
    bSize = len(A) / m_size
    splittedMatrix = [[None for m in range(m_size)] for m in range(m_size)]
    for i in range(m_size):
        for j in range(m_size):
            splittedMatrix[i][j] = np.matrix(A[i * bSize:(i + 1) * bSize, j * bSize:(j + 1) * bSize])
    return splittedMatrix
