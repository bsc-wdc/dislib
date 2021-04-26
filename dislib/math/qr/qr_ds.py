import numpy as np
from pycompss.api.api import compss_wait_on
from pycompss.api.constraint import constraint
from pycompss.api.parameter import COLLECTION_IN
from pycompss.api.task import task

from dislib.data.array import Array, identity, full

ZEROS = 0
IDENTITY = 1
OTHER = 2

_DEBUG = True


if _DEBUG:
    from pycompss.api.dummy.task import task  # for debugging purposes


import threading

global_lock = threading.Lock()


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

    print("q shape 0:", q.shape)
    print("r shape 0:", r.shape)

    for i in range(m_size):
        if _DEBUG:
            print("step1")
            print("r.shape", r.shape)
            print("r._n_blocks", r._n_blocks)
            print("before the main loop")
            r._blocks = compss_wait_on(r._blocks)
            r_type._blocks = compss_wait_on(r_type._blocks)
            print("1.2 r._blocks.len", r._n_blocks[0], r._n_blocks[1])
            print("1.1 r_type._blocks.len", r_type._n_blocks[0], r_type._n_blocks[1])
            print("1.2 r content before:", r.collect())
            print("1.2 r_type _blocks before:", r_type._blocks)
            print("1.2 r_type content before:", r_type.collect())

        actQ_type, actQ, r_type._blocks[i][i], r._blocks[i][i] = _qr(r._blocks[i][i], r_type._blocks[i][i], mkl_proc, b_size, t=True)

        if _DEBUG:
            r._blocks = compss_wait_on(r._blocks)
            r_type._blocks = compss_wait_on(r_type._blocks)
            print("1.2 r._blocks.len", r._n_blocks[0], r._n_blocks[1])
            print("1.2 r_type._blocks.len", r_type._n_blocks[0], r_type._n_blocks[1])
            print("1.2 r content after:", r.collect())
            print("1.2 r_type _blocks after:", r_type._blocks)
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
            if _DEBUG:
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
                print("2. q_type._blocks[j][i] before:", q_type._blocks[j][i])
                print("2. actQ:", actQ)
                print("2. actQ_type:", actQ_type)

            #q._blocks[j][i] = compss_wait_on(q._blocks[j][i])
            #print("q._blocks[{}][{}] before".format(j, j), q._blocks[j][i])

            q_type._blocks[j][i], q._blocks[j][i] = _dot(q._blocks[j][i], q_type._blocks[j][i], actQ, actQ_type, mkl_proc, transposeB=True)

            #q._blocks[j][i] = compss_wait_on(q._blocks[j][i])
            #print("q._blocks[{}][{}] after".format(j, j), q._blocks[j][i])

            if _DEBUG:
                q._blocks = compss_wait_on(q._blocks)
                q_type._blocks = compss_wait_on(q_type._blocks)
                print("2. q blocks:", q._blocks)
                print("2. q shape 2:", q.shape)
                print("2. q content after:", q.collect())
                print("2. q_type content after:", q_type.collect())
                print("2. q_type blocks after:", q_type._blocks)
                print("2. r_type _blocks after:", r_type._blocks)
            # Q[i][j] = dot(actQ, Q[i][j])

        for j in range(i + 1, m_size):
            if _DEBUG:
                print("loop3", j)
                r._blocks = compss_wait_on(r._blocks)
                r_type._blocks = compss_wait_on(r_type._blocks)

            #actQ = compss_wait_on(actQ)
            #print("actQ", actQ)

            r_type._blocks[i][j], r._blocks[i][j] = _dot(actQ, actQ_type, r._blocks[i][j], r_type._blocks[i][j], mkl_proc)

            if _DEBUG:
                r._blocks  = compss_wait_on(r._blocks )
                r_type._blocks  = compss_wait_on(r_type._blocks )
                print("3. q shape 3:", q.shape)
                print("3. r shape 3:", r.shape)
                print("3. r_type _blocks before:", r_type._blocks)
                print("3. r content after:", r.collect())
                print("3. r _blocks after:", r._blocks)

        # Update values of the respective column
        for j in range(i + 1, m_size):
            subQ = [[np.matrix(np.array([0])), np.matrix(np.array([0]))],
                    [np.matrix(np.array([0])), np.matrix(np.array([0]))]]
            # setting the following to OTHER because of _little_qr returns this type of matrices
            subQ_type = [[_type_block(OTHER), _type_block(OTHER)],
                        [_type_block(OTHER), _type_block(OTHER)]]

            if _DEBUG:
                r._blocks = compss_wait_on(r._blocks)
                r_type._blocks = compss_wait_on(r_type._blocks)

            subQ[0][0], subQ[0][1], subQ[1][0], subQ[1][1], r_type._blocks[i][i], r._blocks[i][i], r_type._blocks[j][i], r._blocks[j][i] = _little_qr(
                r._blocks[i][i], r_type._blocks[i][i], r._blocks[j][i], r_type._blocks[j][i], mkl_proc,
                b_size, transpose=True)

            if _DEBUG:
                r._blocks = compss_wait_on(r._blocks)
                r_type._blocks = compss_wait_on(r_type._blocks)
                subQ = compss_wait_on(subQ)
                print("4. iteration", j)
                print("4. q shape 4:", q.shape)
                print("4. r shape 4:", r.shape)
                print("4. subQ:", subQ)
                print("4. r_type _blocks before:", r_type._blocks)
                print("4. r content after:", r.collect())
                print("4. r _blocks after:", r._blocks)
            # subQ = blockedTranspose(subQ, mkl_proc)
            # Update values of the row for the value updated in the column
            for k in range(i + 1, m_size):
                #print("r_type._blocks", r_type._blocks)
                #print("r._blocks", r._blocks)
                #print("[[r._blocks[i][k]], [r._blocks[j][k]]]", [[r._blocks[i][k]], [r._blocks[j][k]]])
                #print("[[r_type._blocks[i][k]], [r_type._blocks[j][k]]]", [[r_type._blocks[i][k]], [r_type._blocks[j][k]]])
                [[r_type._blocks[i][k]], [r_type._blocks[j][k]]], [[r._blocks[i][k]], [r._blocks[j][k]]] = _multiply_blocked(
                #r_type._blocks[i][k], r_type._blocks[j][k], r._blocks[i][k], r._blocks[j][k] = _multiply_blocked(
                    subQ,
                    subQ_type,
                    [[r._blocks[i][k]], [r._blocks[j][k]]],
                    [[r_type._blocks[i][k]], [r_type._blocks[j][k]]],
                    b_size,
                    mkl_proc
                )

                #print("[[r_type._blocks[{}][{}]], [r_type._blocks[{}][}]]]".format(i, k, j, k), [[r_type._blocks[i][k]], [r_type._blocks[j][k]]])
                #print("[[r._blocks[{}][{}]], [r._blocks[{}][{}]]]".format(i, k, j, k), [[r._blocks[i][k]], [r._blocks[j][k]]])

                if _DEBUG:
                    r._blocks = compss_wait_on(r._blocks)
                    r_type._blocks = compss_wait_on(r_type._blocks)
                    print("5. q shape 5:", q.shape)
                    print("5. r shape 5:", r.shape)
                    print("5. r content after:", r.collect())
                    print("5. r _blocks after:", r._blocks)

            if _DEBUG:
                q._blocks = compss_wait_on(q._blocks)
                q_type._blocks = compss_wait_on(q_type._blocks)

            for k in range(m_size):
                if _DEBUG:
                    print("6. q content before:", q.collect())
                    print("6. q _blocks before:", q._blocks)
                    print("6. [[q._blocks[k][i], q._blocks[k][j]]]:", [[q._blocks[k][i], q._blocks[k][j]]])
                    print("6. [[q_type._blocks[k][i], q_type._blocks[k][j]]]:", [[q_type._blocks[k][i], q_type._blocks[k][j]]])
                    print("6. subQ:", subQ)
                    print("6. subQ_type:", subQ_type)

                q._blocks[k][i] = compss_wait_on(q._blocks[k][i])
                q._blocks[k][j] = compss_wait_on(q._blocks[k][j])
                print("q._blocks[k][i] for ({}, {}) before".format(k, i), q._blocks[k][i])
                print("q._blocks[k][j] for ({}, {}) before".format(k, j), q._blocks[k][j])

                [[q_type._blocks[k][i], q_type._blocks[k][j]]], [[q._blocks[k][i], q._blocks[k][j]]] = _multiply_blocked(
                #q_type._blocks[k][i], q_type._blocks[k][j], q._blocks[k][i], q._blocks[k][j] = _multiply_blocked(
                    [[q._blocks[k][i], q._blocks[k][j]]],
                    [[q_type._blocks[k][i], q_type._blocks[k][j]]],
                    subQ,
                    subQ_type,
                    b_size,
                    mkl_proc,
                    transposeB=True
                )

                q._blocks[k][i] = compss_wait_on(q._blocks[k][i])
                q._blocks[k][j] = compss_wait_on(q._blocks[k][j])
                print("q._blocks[k][i] for ({}, {}) after".format(k, i), q._blocks[k][i])
                print("q._blocks[k][j] for ({}, {}) after".format(k, j), q._blocks[k][j])

                if _DEBUG:
                    q._blocks = compss_wait_on(q._blocks)
                    q_type._blocks = compss_wait_on(q_type._blocks)
                    print("6. q shape 6:", q.shape)
                    print("6. r shape 6:", r.shape)
                    print("6. q content after:", q.collect())
                    print("6. q _blocks after:", q._blocks)
                    print("6. q_type content after:", q_type.collect())
                    print("6. q_type _blocks after:", q_type._blocks)
                # [[Q[k][i], Q[k][j]]] = multiplyBlocked([[Q[k][i], Q[k][j]]], blockedTranspose(subQ, mkl_proc), BSIZE, mkl_proc)

    # Q = blockedTranspose(Q,mkl_proc)
    return q, r


def _set_mkl_num_threads(mkl_proc):
    import os
    os.environ["MKL_NUM_THREADS"] = str(mkl_proc)


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
@task(returns=(list, list), on_failure='FAIL')
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
    #print("DOT1", a, "of type", a_type)
    #print("DOT2", b, "of type", b_type)
    if a_type[0][0] == ZEROS:
        print("DOT RESULT: ZERO 1")
        f = open("/home/bscuser/git/dislib/dump.txt", "w")
        f.write("a")
        f.write(repr(a))
        f.write("\n")
        f.write("a_type")
        f.write(repr(a_type))
        f.write("\n")
        f.write("b")
        f.write(repr(b))
        f.write("\n")
        f.write("b_type")
        f.write(repr(b_type))
        f.close()
        return _type_block(ZEROS), _empty_block(a.shape)
    if a_type[0][0] == IDENTITY:
        if transposeB and transposeResult:
            #print("DOT RESULT b1:", b_type, b)
            return b_type, b
        if transposeB or transposeResult:
            #print("DOT RESULT b2:", _transpose_block(b, b_type))
            return _transpose_block(b, b_type)

        #print("DOT RESULT b3:", b_type, b)
        return b_type, b
    if b_type[0][0] == ZEROS:
        #print("DOT RESULT: ZERO 2")
        return _type_block(ZEROS), _empty_block(a.shape)
    if b_type[0][0] == IDENTITY:
        if transposeResult:
            #print("DOT RESULT: ID 1", _transpose_block(a, a_type))
            return _transpose_block(a, a_type)
        #print("DOT RESULT: ID 2", a, a_type)
        return a_type, a
    result = _dot_task(a, b, mkl_proc, transposeResult=transposeResult, transposeB=transposeB)

    #print("DOT RESULT:", result)

    return _type_block(OTHER), result


def _dot_task(a, b, mkl_proc, transposeResult=False, transposeB=False, **kwargs):
    _set_mkl_num_threads(mkl_proc)
    if transposeB:
        b = np.transpose(b)
    if transposeResult:
        return np.transpose(np.dot(a, b))
    #print("_dot_task")
    #print(a, b)
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
            mat[1] = np.matrix(np.zeros((regular_b_size, regular_b_size)))
        elif mat[0] == IDENTITY:
            mat[1] = np.matrix(np.identity(regular_b_size))
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

    if _DEBUG:
        print("typeA", typeA)
        print("A", A)
        print("typeB", typeB)
        print("B", B)
        print("typeC", typeC)
        print("C", C)

    if typeA[0][0] == ZEROS or typeB[0][0] == ZEROS:
        # TODO to check it in the case of not square blocks
        # bool type to save memory as the matrix is filled with Nones
        return typeC, C

    #print("b_size", b_size)

    funA = [typeA, A]
    funB = [typeB, B]
    if typeC[0][0] == ZEROS:
        C = np.matrix(np.zeros((b_size[0], b_size[1])))
    elif typeC[0][0] == IDENTITY:
        C = np.matrix(np.identity(b_size[0]))
    if funA[0][0][0] == IDENTITY:
        if funB[0][0][0] == IDENTITY:
            funB[1] = np.matrix(np.identity(b_size[0]))
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

    if _DEBUG:
        print("funA", funA[1])
        print("funA.len", len(funA))
        print("funB", funB[1])
        print("funB.len", len(funB))
        print("C", C)
        print("C.shape", len(C))

    C += (funA[1] * funB[1])
    return _type_block(OTHER), C


def _multiply_single_block(a, type_a, b, type_b, c, type_c, mkl_proc, b_size, transposeB=False):
    return _multiply_single_block_task(a, type_a, b, type_b, c, type_c, mkl_proc, b_size, transposeB=transposeB)


#@constraint(ComputingUnits="${ComputingUnits}")
#@task(returns=(list, list, list, list))
def _multiply_blocked(A, type_a, B, type_b, b_size, mkl_proc, transposeB=False):
    if transposeB:
        newB = []
        for i in range(len(B[0])):
            newB.append([])
            for j in range(len(B)):
                newB[i].append(B[j][i])
        B = newB

    if _DEBUG:
        print("_multiply_blocked")
        print("A:", A)
        print("type_a:", type_a)
        print("B:", B)
        print("type_b:", type_b)

    C = []
    type_c = []
    for i in range(len(A)):
        C.append([])
        type_c.append([])
        for j in range(len(B[0])):
            C[i].append(_empty_block(b_size))
            type_c[i].append(_type_block(ZEROS))
            for k in range(len(A[0])):
                print("indices", (i, j, k), "of", (len(A),len(B[0]), len(A[0])))
                print("A", A)
                print("type_a", type_a)
                print("B", B)
                print("type_b", type_b)
                print("C", C)
                print("type_c", type_c)

                A[i][k] = compss_wait_on(A[i][k])
                B[k][j] = compss_wait_on(B[k][j])
                C[i][j] = compss_wait_on(C[i][j])
                print("A[{}][{}]".format(i, k), A[i][k])
                print("B[{}][{}]".format(k, j), B[k][j])
                print("C[{}][{}]".format(i, j), C[i][j])

                type_c[i][j], C[i][j] = _multiply_single_block(
                    A[i][k], type_a[i][k],  # ??????????????????????????????????????????????? type???
                    B[k][j], type_b[k][j],  # ??????????????????????????????????????????????? type???
                    C[i][j], type_c[i][j],  # ??????????????????????????????????????????????? type???
                    mkl_proc, b_size, transposeB=transposeB)

                type_c[i][j] = compss_wait_on(type_c[i][j])
                C[i][j] = compss_wait_on(C[i][j])
                print("AFTER type_c[{}][{}]".format(i,j), type_c[i][j])
                print("AFTER C[{}][{}]".format(i,j), C[i][j])

    # [[q_type._blocks[k][i], q_type._blocks[k][j]]], [[q._blocks[k][i], q._blocks[k][j]]] = _multiply_blocked(

    #print("type_c")
    #print(type_c)
    #print("C")
    #print(C)
    return type_c, C
    #return type_c[0][0], type_c[1][0], C[0][0], C[1][0]


def _transpose_block(a, a_type):
    #print("type:", a_type)
    if a_type[0][0] == ZEROS or a_type[0][0] == IDENTITY:
        return a_type, a
    return _type_block(OTHER), _transpose_block_task(a)


def _split_matrix(A, m_size):
    bSize = int(len(A) / m_size)
    splittedMatrix = [[None for m in range(m_size)] for m in range(m_size)]
    for i in range(m_size):
        for j in range(m_size):
            #print("bSize", bSize, "index", (i * bSize, (i + 1) * bSize), (j * bSize, (j + 1) * bSize))
            splittedMatrix[i][j] = np.matrix(A[i * bSize:(i + 1) * bSize, j * bSize:(j + 1) * bSize])
    return splittedMatrix
