import numpy as np
import dislib as ds
from pycompss.api.task import task
from pycompss.api.parameter import (Type, COLLECTION_IN, Depth,
                                    COLLECTION_INOUT, COLLECTION_OUT)
from pycompss.api.api import compss_barrier
from pycompss.api.constraint import constraint
from pycompss.api.on_failure import on_failure
from dislib.decomposition import tsqr
from dislib.data.array import Array, concat_columns
import math


@constraint(computing_units="${ComputingUnits}")
@task(returns=1)
def _assign_elements_last_block(in_block, top_left_shape_data):
    return in_block[-top_left_shape_data:, :]


@constraint(computing_units="${ComputingUnits}")
@task(returns=1)
def _assign_corresponding_elements_block(in_block,
                                         second_in_block,
                                         block_size,
                                         top_left_shape_data):
    block = np.vstack([in_block, second_in_block])
    return block[top_left_shape_data:top_left_shape_data+block_size, :]


@constraint(computing_units="${ComputingUnits}")
@task(blocks={Type: COLLECTION_IN, Depth: 2},
      Ub={Type: COLLECTION_INOUT, Depth: 2},
      Sb={Type: COLLECTION_INOUT, Depth: 2},
      Vb={Type: COLLECTION_INOUT, Depth: 2})
def my_svd(blocks, Ub, Sb, Vb, block_size):
    arr = np.block(blocks)
    U, S, V = np.linalg.svd(arr, full_matrices=False)
    V2 = V.T
    S2 = np.diag(S)
    for j in range(len(Ub)):
        for i in range(len(Ub[j])):
            Ub[j][i] = U[j * block_size:(j+1) * block_size,
                         i * block_size:(i+1) * block_size]
            Sb[j][i] = S2[j * block_size:(j + 1) * block_size,
                          i * block_size:(i + 1) * block_size]
            Vb[j][i] = V2[j * block_size:(j + 1) * block_size,
                          i * block_size:(i + 1) * block_size]


@constraint(computing_units="${ComputingUnits}")
@task(blocks={Type: COLLECTION_INOUT, Depth: 1},
      blocks_to_assign={Type: COLLECTION_IN, Depth: 1})
def assign_blocks(blocks, blocks_to_assign, index, index_to_assign):
    for i in range(len(blocks)):
        if index == i:
            blocks[index] = blocks_to_assign[index_to_assign]
        else:
            blocks[i] = blocks[i]


@constraint(computing_units="${ComputingUnits}")
@task(blocks={Type: COLLECTION_OUT, Depth: 1},
      blocks_to_assign={Type: COLLECTION_IN, Depth: 1})
def assign_all_blocks(blocks, blocks_to_assign):
    for i in range(len(blocks)):
        blocks[i] = blocks_to_assign[i]


@constraint(computing_units="${ComputingUnits}")
@task(blocks={Type: COLLECTION_INOUT, Depth: 1})
def assign_zero_all_blocks(blocks):
    for i in range(len(blocks)):
        blocks[i] = np.zeros(blocks[i].shape)


@constraint(computing_units="${ComputingUnits}")
@task(blocks=COLLECTION_IN, q_blocks=COLLECTION_OUT, r_blocks=COLLECTION_OUT)
def my_qr_blocks(blocks, reg_shape, q_blocks, r_blocks, mode="complete"):
    block, r = np.linalg.qr(np.block(blocks), mode=mode)
    for i in range(len(q_blocks)):
        for j in range(len(q_blocks[i])):
            q_blocks[i][j] = block[i * reg_shape[0]:(i+1) * reg_shape[0],
                                   j * reg_shape[1]:(j+1) * reg_shape[1]]
    for i in range(len(r_blocks)):
        for j in range(len(r_blocks[i])):
            r_blocks[i][j] = r[i * reg_shape[1]:(i+1) * reg_shape[1],
                               j * reg_shape[1]:(j+1) * reg_shape[1]]


@constraint(computing_units="${ComputingUnits}")
@task(Sb={Type: COLLECTION_INOUT, Depth: 2},
      converged_it={Type: COLLECTION_INOUT, Depth: 1})
def actual_iteration(Sb, converged_it, actual_iteration):
    converged_it[0] = np.array(actual_iteration)


@constraint(computing_units="${ComputingUnits}")
@on_failure(management="CANCEL_SUCCESSORS", returns=0)
@task(max_blocks={Type: COLLECTION_IN, Depth: 2},
      min_blocks={Type: COLLECTION_IN, Depth: 2},
      Ub={Type: COLLECTION_INOUT, Depth: 2},
      Sb={Type: COLLECTION_INOUT, Depth: 2},
      Vb={Type: COLLECTION_INOUT, Depth: 2},
      p_blocks={Type: COLLECTION_INOUT, Depth: 2},
      b_blocks={Type: COLLECTION_INOUT, Depth: 2},
      converged_it={Type: COLLECTION_INOUT, Depth: 1})
def check_convergence(max_blocks, min_blocks, Ub, Sb, Vb,
                      p_blocks, b_blocks, converged_it,
                      singular_values, tol):
    maximum_values = abs(np.block(max_blocks).flatten())
    minimum_values = abs(np.block(min_blocks).flatten())
    max_values = np.maximum(maximum_values, minimum_values)
    converged = max_values[:singular_values] < tol
    if sum(converged) >= singular_values:
        raise Exception("Converged")


@constraint(computing_units="${ComputingUnits}")
@task(out_blocks={Type: COLLECTION_OUT, Depth: 1},
      blocks={Type: COLLECTION_IN, Depth: 1})
def _assign_column_blocks(out_blocks, blocks, start_block):
    for i in range(len(out_blocks)):
        out_blocks[i] = blocks[i + start_block]


def _obtain_blocks_of_columns(A, initial_block, final_block):
    out_blocks = [[object() for _ in range(final_block - initial_block)]
                  for _ in range(A._n_blocks[0])]
    for idx, blocks in enumerate(A._blocks):
        _assign_column_blocks(out_blocks[idx], blocks, initial_block)
    return Array(blocks=out_blocks, top_left_shape=A._top_left_shape,
                 shape=(A.shape[0],
                        A._reg_shape[1] * (final_block-initial_block)),
                 reg_shape=A._reg_shape, sparse=False)


def svd_lanczos_t_conv_criteria(A, P, Q, k=None, b=1,
                                rank=2, max_it=None,
                                singular_values=1, tol=1e-8):
    m, n = A.shape
    if k is None:
        k = min(m, n)
    converged = [np.array(0)]

    if max_it is None:
        max_it = k

    S = None

    Ub = [[object() for _ in range(math.ceil(k / b))]
          for _ in range(math.ceil(k / b))]
    Sb = [[object() for _ in range(math.ceil(k / b))]
          for _ in range(math.ceil(k / b))]
    Vb = [[object() for _ in range(math.ceil(k / b))]
          for _ in range(math.ceil(k / b))]

    B = ds.zeros(shape=(k, k), block_size=(b, b))
    for restart in range(max_it):
        for i in range(B._n_blocks[0]):
            assign_zero_all_blocks(B._blocks[i])
        if restart == 0:
            j = 0
        else:
            j = rank
            for i in range(math.ceil(rank/b)):
                assign_blocks(B._blocks[i], S._blocks[i], i, i)

        while j < k:
            p = ds.matmul(A, _obtain_blocks_of_columns(
                Q, int(j/b), int((j+b)/b)), transpose_a=True)
            if j > 0:
                h1 = ds.matmul(_obtain_blocks_of_columns(
                    P, 0, int(j/b)), p, transpose_a=True)
                p = ds.data.matsubtract(p, ds.matmul(
                    _obtain_blocks_of_columns(P, 0, int(j/b)), h1))
            p_blocks = [[object() for _ in range(p._n_blocks[1])]
                        for _ in range(p._n_blocks[0])]
            w1_blocks = [[object() for _ in range(p._n_blocks[1])]
                         for _ in range(p._n_blocks[1])]
            my_qr_blocks(p._blocks, p._reg_shape,
                         p_blocks, w1_blocks, mode='reduced')
            p = Array(p_blocks, top_left_shape=p._top_left_shape,
                      reg_shape=p._reg_shape, shape=p.shape, sparse=False)
            w1 = Array(w1_blocks, top_left_shape=(p._top_left_shape[1],
                                                  p._top_left_shape[1]),
                       reg_shape=(p._reg_shape[1], p._reg_shape[1]),
                       shape=(p.shape[1], p.shape[1]), sparse=False)
            if j > 0:
                h2 = ds.matmul(_obtain_blocks_of_columns(
                    P, 0, int((j)/b)), p, transpose_a=True)
                p = ds.data.matsubtract(p, ds.matmul(
                    _obtain_blocks_of_columns(P, 0, int(j/b)), h2))
                if restart > 0 and j == rank:
                    sum_ds_arrays = ds.data.matadd(h1, h2).T
                    assign_blocks(B._blocks[j // b],
                                  sum_ds_arrays._blocks[0],
                                  j // b, 0)
                    maximum_values = sum_ds_arrays.max()
                    minimum_values = sum_ds_arrays.min()
                    actual_iteration(Sb, converged, restart)
                    check_convergence(maximum_values._blocks,
                                      minimum_values._blocks, Ub, Sb, Vb,
                                      p._blocks, B._blocks, converged,
                                      singular_values=singular_values,
                                      tol=tol)

            p_blocks = [[object() for _ in range(p._n_blocks[1])]
                        for _ in range(p._n_blocks[0])]
            w2_blocks = [[object() for _ in range(p._n_blocks[1])]
                         for _ in range(p._n_blocks[1])]
            my_qr_blocks(p._blocks, p._reg_shape,
                         p_blocks, w2_blocks, mode='reduced')
            p = Array(p_blocks, top_left_shape=p._top_left_shape,
                      reg_shape=p._reg_shape, shape=p.shape, sparse=False)
            w2 = Array(w2_blocks, top_left_shape=(p._top_left_shape[1],
                                                  p._top_left_shape[1]),
                       reg_shape=(p._reg_shape[1], p._reg_shape[1]),
                       shape=(p.shape[1], p.shape[1]), sparse=False)

            multiplied_w_trans = ds.matmul(w2, w1).T
            assign_blocks(B._blocks[j // b],
                          multiplied_w_trans._blocks[0], j // b, 0)
            for i in range(P._n_blocks[0]):
                assign_blocks(P._blocks[i], p._blocks[i], j // b, 0)

            q = ds.matmul(A, _obtain_blocks_of_columns(
                P, int(j/b), int((j+b)/b)))
            # orthogonalization of Q
            h = ds.matmul(_obtain_blocks_of_columns(
                Q, 0, int((j+b)/b)), q, transpose_a=True)

            q = ds.data.matsubtract(q, ds.matmul(
                _obtain_blocks_of_columns(Q, 0, int((j+b)/b)), h))

            q, w1 = tsqr(q, mode='reduced')

            h = ds.matmul(_obtain_blocks_of_columns(
                Q, 0, int((j + b)/b)), q, transpose_a=True)
            q = ds.data.matsubtract(q, ds.matmul(
                _obtain_blocks_of_columns(Q, 0, int((j + b)/b)), h))

            q, w2 = tsqr(q, mode='reduced')

            if j < (k - b):
                multiplied_w = ds.matmul(w2, w1)
                assign_blocks(B._blocks[(j + b) // b],
                              multiplied_w._blocks[0], j // b, 0)
                del multiplied_w
                for i in range(Q._n_blocks[0]):
                    assign_blocks(Q._blocks[i],
                                  q._blocks[i], (j+b)//b, 0)
            j += b
        my_svd(B._blocks, Ub, Sb, Vb, b)
        U_blocks = [[object() for _ in range(math.ceil(rank / b))]
                    for _ in range(math.ceil(k / b))]
        S_blocks = [[object() for _ in range(math.ceil(rank / b))]
                    for _ in range(math.ceil(rank / b))]
        V_blocks = [[object() for _ in range(math.ceil(rank / b))]
                    for _ in range(math.ceil(k / b))]
        for j in range(len(U_blocks)):
            assign_all_blocks(U_blocks[j], Ub[j])
            assign_all_blocks(V_blocks[j], Vb[j])
        for j in range(len(S_blocks)):
            assign_all_blocks(S_blocks[j], Sb[j])
        S = Array(blocks=S_blocks, top_left_shape=(b, b), shape=(rank, rank),
                  reg_shape=(b, b), sparse=False)
        V = P @ Array(blocks=V_blocks, top_left_shape=(b, b), shape=(k, rank),
                      reg_shape=(b, b), sparse=False)
        U = Q @ Array(blocks=U_blocks, top_left_shape=(b, b), shape=(k, rank),
                      reg_shape=(b, b), sparse=False)
        for i in range(P._n_blocks[0]):
            for j in range(math.ceil(rank/b)):
                assign_blocks(P._blocks[i], V._blocks[i], j, j)
        for i in range(Q._n_blocks[0]):
            for j in range(math.ceil(rank/b)):
                assign_blocks(Q._blocks[i], U._blocks[i], j, j)
        for i in range(Q._n_blocks[0]):
            assign_blocks(Q._blocks[i], q._blocks[i], j + 1, 0)
    compss_barrier()
    my_svd(B._blocks, Ub, Sb, Vb, b)
    U_blocks = [[object() for _ in range(math.ceil(rank / b))]
                for _ in range(math.ceil(k / b))]
    S_blocks = [[object() for _ in range(math.ceil(rank / b))]
                for _ in range(math.ceil(rank / b))]
    V_blocks = [[object() for _ in range(math.ceil(rank / b))]
                for _ in range(math.ceil(k / b))]
    for j in range(len(U_blocks)):
        assign_all_blocks(U_blocks[j], Ub[j])
        assign_all_blocks(V_blocks[j], Vb[j])
    for j in range(len(S_blocks)):
        assign_all_blocks(S_blocks[j], Sb[j])
    U = Q @ Array(blocks=U_blocks, top_left_shape=(b, b), shape=(k, rank),
                  reg_shape=(b, b), sparse=False)
    S = Array(blocks=S_blocks, top_left_shape=(b, b), shape=(rank, rank),
              reg_shape=(b, b), sparse=False)
    V = P @ Array(blocks=V_blocks, top_left_shape=(b, b), shape=(k, rank),
                  reg_shape=(b, b), sparse=False)
    return U, S, V, P, Q


def check_tolerance(m, n, nsv, S, epsilon):
    min_sval = S[nsv-1]
    total_sval = min(m, n)
    sum_nsv = 0
    for isv in range(nsv):
        sum_nsv += pow(S[isv], 2)
    sum_remaining = (total_sval-nsv) * pow(min_sval, 2)
    tol = np.sqrt(sum_remaining/(sum_remaining+sum_nsv))
    if tol <= epsilon:
        return True
    return False


def lanczos_svd(a: Array, k, bs, rank, num_sv, tolerance,
                epsilon, max_num_iterations):
    """ Lanczos SVD

    Parameters
    ----------
    a : ds-arrays
        Input ds-array.
    k : int
        Number of iterations of the Lanczos algorithm,
        in order to compute the inner iterations this parameter is divided
        by the bs parameter (must be a multiple of b).
    bs : int
        Block size (in the column axis)
    rank : int
        Number of restarting vectors (must be a multiple of b)
    num_sv : int
        Number of desired singular values
    tolerance : float64
        If the residual value of a singular value is less than the
        tolerante, that singular value is considered to be converged.
    epsilon : float64
        Value that defines the number of singular values required,
        as it is reduced, the number of singular values required
        is increased.
    max_num_iterations : int
        Maximum number of iterations executed in the lanczos.
        It is supposed that the desired singular values will
        converge before reaching this value. If it is not the case
        this defines a limit on the iterations executed.

    Returns
        -------
        U : ds-array
            The U of the matrix, Unitary array returned as ds-array, the shape
            is A.shape[0] x rank, and the block size is the block size of
            A in the row axis x bs.
        S : ds-array
            The S of the matrix. It is represented as a 2-dimensional matrix,
            the diagonal of this matrix is the vector with the singular
            values. Its shape is rank x rank and the block size is bs x bs
        V : ds-array
            The V of the matrix, Unitary array returned as ds-array,
            the shape is A.shape[1] x rank, and the block size is bs x bs

        Raises
        ------
        ValueError
            If num_sv is bigger than the number of columns
            or
            If rank < num_nsv
            or
            If k <= rank
    """
    m, n = a.shape
    if num_sv > n:
        raise ValueError("Number of singular values to compute can't"
                         " be bigger than the total number of singular"
                         "values of the matrix.")
    if rank < num_sv:
        raise ValueError("Rank should be at least the number of singular"
                         "values to compute.")
    if k <= rank:
        raise ValueError("K defines the number of iterations of the "
                         "algorithm, the iterations are k/bs, it should"
                         " be higher than the Rank.")
    if k % bs != 0 or rank % bs != 0:
        raise ValueError("K and Rank must be multiple of BS.")

    Q = ds.zeros((m, k), block_size=(a._reg_shape[0], bs))
    P = ds.zeros(shape=(n, k), block_size=(bs, bs))
    np.random.seed(0)
    q = ds.data.random_array(shape=(m, bs), block_size=(a._reg_shape[0], bs))
    q, w = tsqr(q, mode='reduced')
    for i in range(Q._n_blocks[0]):
        assign_blocks(Q._blocks[i], q._blocks[i], 0, 0)
    u, s, v, P, Q = svd_lanczos_t_conv_criteria(a, P, Q, k=k, b=bs,
                                                rank=rank,
                                                max_it=max_num_iterations,
                                                singular_values=num_sv,
                                                tol=tolerance)
    s_blocks = s.collect()
    s = s_blocks.diagonal()
    while not check_tolerance(a.shape[0], a.shape[1], nsv=num_sv, S=s,
                              epsilon=epsilon):
        P2 = ds.zeros(shape=(n, bs), block_size=(bs, bs))
        Q2 = ds.zeros(shape=(m, bs), block_size=(a._reg_shape[0], bs))
        q = ds.data.array(np.random.rand(m, bs),
                          block_size=(a._reg_shape[0], bs))
        q, w = tsqr(q, mode='reduced')
        for i in range(Q2._n_blocks[0]):
            assign_blocks(Q2._blocks[i], q._blocks[i], 0, 0)
        Q = concat_columns(Q, Q2)
        P = concat_columns(P, P2)
        k = k + bs
        rank = rank + bs
        num_sv = num_sv + bs
        u, s, v, P, Q = svd_lanczos_t_conv_criteria(a, P, Q, k=k, b=bs,
                                                    rank=rank,
                                                    max_it=max_num_iterations,
                                                    singular_values=num_sv,
                                                    tol=tolerance)
        s_blocks = s.collect()
        s = s_blocks.diagonal()

    s = ds.data.array(s_blocks, block_size=(bs, bs))

    return u, s, v
