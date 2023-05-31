import numpy as np
import dislib as ds
from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.api.api import compss_barrier, compss_wait_on
from pycompss.api.constraint import constraint
from pycompss.api.on_failure import on_failure
from time import time
from dislib.decomposition import qr
from base import tsqr
from dislib.data.array import Array, _empty_array
from pycompss.api.reduction import reduction
import math
from sklearn.datasets import make_low_rank_matrix


@task(blocks={Type: COLLECTION_IN, Depth: 2},
      Ub={Type: COLLECTION_INOUT, Depth: 2},
      Sb={Type: COLLECTION_INOUT, Depth: 2},
      Vb={Type: COLLECTION_INOUT, Depth: 2})
def my_svd(blocks, Ub, Sb, Vb, block_size):
    arr = np.block(blocks)
    U, S, V = np.linalg.svd(arr, full_matrices=False)
    S2 = np.diag(S)
    for j in range(len(Ub)):
        for i in range(len(Ub[j])):
            Ub[j][i] = U[j*block_size:(j+1)*block_size, i*block_size:(i+1)*block_size]
            Sb[j][i] = S2[j * block_size:(j + 1) * block_size, i * block_size:(i + 1) * block_size]
            Vb[j][i] = V[j * block_size:(j + 1) * block_size, i * block_size:(i + 1) * block_size]

@task(blocks={Type: COLLECTION_INOUT, Depth: 1})
def assign_unique_block(blocks, blocks_to_assign, index):
    for i in range(len(blocks)):
        if index == i:
            blocks[index] = blocks_to_assign
        else:
            blocks[i] = blocks[i]

@task(blocks={Type: COLLECTION_INOUT, Depth: 1},
      blocks_to_assign={Type: COLLECTION_IN, Depth: 1})
def assign_blocks(blocks, blocks_to_assign, index, index_to_assign):
    for i in range(len(blocks)):
        if index == i:
            blocks[index] = blocks_to_assign[index_to_assign]
        else:
            blocks[i] = blocks[i]


@task(blocks={Type: COLLECTION_OUT, Depth: 1},
      blocks_to_assign={Type: COLLECTION_IN, Depth: 1})
def assign_all_blocks(blocks, blocks_to_assign):
    for i in range(len(blocks)):
        blocks[i] = blocks_to_assign[i]


@task(blocks={Type: COLLECTION_INOUT, Depth: 1})
def assign_zero_all_blocks(blocks):
    for i in range(len(blocks)):
        blocks[i] = np.zeros(blocks[i].shape)


@task(returns=1)
def  matmul_blocks(block1, block2):
    return block1 @ block2


@reduction(chunk_size="2")
@task(blocks={Type: COLLECTION_IN, Depth: 2}, returns=1)
def norm_task(blocks):
    norm = 0
    for block in blocks:
        norm += np.linalg.norm(block)
    return norm


@task(block=IN, returns=2)
def my_qr(block, mode="complete"):
    block, r = np.linalg.qr(block, mode='reduced')
    return block, r



@task(Sb={Type: COLLECTION_INOUT, Depth: 2}, converged_it={Type: COLLECTION_INOUT, Depth: 1})
def actual_iteration(Sb, converged_it, actual_iteration):
    converged_it[0] = np.array(actual_iteration)

@on_failure(management="CANCEL_SUCCESSORS", returns=0)
@task(max_blocks={Type: COLLECTION_IN, Depth: 2},
      min_blocks={Type: COLLECTION_IN, Depth: 2},
      Ub={Type: COLLECTION_INOUT, Depth: 2},
      Sb={Type: COLLECTION_INOUT, Depth: 2},
      Vb={Type: COLLECTION_INOUT, Depth: 2},
      p_blocks={Type: COLLECTION_INOUT, Depth: 2},
      b_blocks={Type: COLLECTION_INOUT, Depth: 2},
      converged_it={Type: COLLECTION_INOUT, Depth: 1})
def check_convergence(max_blocks, min_blocks, Ub, Sb, Vb, p_blocks, b_blocks, converged_it, singular_values, tol):
    maximum_values = abs(np.block(max_blocks).flatten())
    minimum_values = abs(np.block(min_blocks).flatten())
    max_values = np.maximum(maximum_values, minimum_values)
    converged = max_values < tol
    if sum(converged) >= singular_values:
        raise Exception("Converged")


def create_dsarray(block, shape, block_size=None):
    if block_size is None:
        block_size = shape
    return Array([[block]], top_left_shape=block_size, reg_shape=block_size,
                 shape=shape, sparse=False)


def svd_lanczos_t_conv_criteria(A, k=None, b=1, rank=2, max_it=None, singular_values=1, tol=1e-8):
    m, n = A.shape
    if k is None:
        k = min(m, n)
    converged = [np.array(0)]

    if max_it is None:
        max_it = k

    Q = ds.zeros((m, k), block_size=(A._reg_shape[0], b))
    P = ds.zeros(shape=(n, k), block_size=(b, b))
    np.random.seed(0)
    q = ds.data.array(np.random.rand(m, b), block_size=(A._reg_shape[0], b))
    q._blocks[0][0], w = my_qr(q._blocks[0][0])
    assign_blocks(Q._blocks[0], q._blocks[0], 0, 0)

    eres = []
    Ub = [[object() for _ in range(math.ceil(k / b))] for _ in range(math.ceil(k / b))]
    Sb = [[object() for _ in range(math.ceil(k / b))] for _ in range(math.ceil(k / b))]
    Vb = [[object() for _ in range(math.ceil(k / b))] for _ in range(math.ceil(k / b))]
    B = ds.zeros(shape=(k, k), block_size=(b, b))
    convergeeeeed = False
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
            p = ds.matmul(A, Q[:, j:j + b], transpose_a=True)

            if j > 0:
                p = p.rechunk((b, b))
                h1 = ds.matmul(P[:, 0:j], p, transpose_a=True)
                p = ds.data.matsubtract(p, ds.matmul(P[:, 0:j], h1))

            p, w1 = tsqr(p.rechunk((b, b)), mode='reduced')

            if j > 0:
                h2 = ds.matmul(P[:, 0:j], p, transpose_a=True)  # P[:, :j].T @ p
                p = ds.data.matsubtract(p, ds.matmul(P[:, 0:j], h2))  # p - P[:, :j] @ h
                if restart > 0 and j == rank:
                    sum_ds_arrays = ds.data.matadd(h1, h2)
                    assign_blocks(B._blocks[j // b], sum_ds_arrays._blocks[-1], j//b, -1)
                    maximum_values = sum_ds_arrays.T.max()
                    minimum_values = sum_ds_arrays.T.min()
                    try:
                        actual_iteration(Sb, converged, restart)
                        check_convergence(maximum_values._blocks, minimum_values._blocks, Ub, Sb, Vb, p._blocks, B._blocks, converged, singular_values=singular_values, tol=tol)  # , Ub, Sb, Vb, p._blocks, singular_values=singular_values, tol=tol)
                    except:
                        convergeeeeed = True
                        break#return U, S, V

            p, w2 = tsqr(p, mode='reduced')

            eres.append(p)
            multiplied_w_trans = ds.matmul(w2, w1).T
            assign_blocks(B._blocks[j // b], multiplied_w_trans._blocks[0], j // b, 0)
            del multiplied_w_trans
            for i in range(P._n_blocks[0]):
                assign_blocks(P._blocks[i], p._blocks[i], j // b, 0)

            q = ds.matmul(A, P[:, j:j + b].rechunk((A._reg_shape[1], A._reg_shape[1])))
            h = ds.matmul(Q[:, 0:j + b], q, transpose_a=True)
            q = ds.data.matsubtract(q, ds.matmul(Q[:, 0:j + b], h))

            q, w1 = tsqr(q, mode='reduced')

            h = ds.matmul(Q[:, 0:j + b], q, transpose_a=True)
            q = ds.data.matsubtract(q, ds.matmul(Q[:, 0:j + b], h))

            q, w2 = tsqr(q, mode='reduced')

            if j < (k - b):
                if np.sum(q._n_blocks) == 2:
                    multiplied_w = ds.matmul(w2, w1)
                    assign_blocks(B._blocks[(j + b) // b], multiplied_w._blocks[0], j // b, 0)
                    del multiplied_w
                    #assign_unique_block(B._blocks[(j + b) // b], matmul_blocks(w2, w1), j//b)#TODO ESTO ESTABA ANTES Y FUNCIONABA CORRECTAMENTE
                else:
                    multiplied_w = ds.matmul(w2, w1)
                    assign_blocks(B._blocks[(j + b) // b], multiplied_w._blocks[0], j // b, 0)
                    del multiplied_w
                for i in range(Q._n_blocks[0]):
                    assign_blocks(Q._blocks[i], q._blocks[i], (j+b)//b, 0)
            j += b
        if convergeeeeed:
            break
        my_svd(B._blocks, Ub, Sb, Vb, b)
        U_blocks = [[object() for _ in range(math.ceil(k / b))] for _ in range(math.ceil(k / b))]
        S_blocks = [[object() for _ in range(math.ceil(k / b))] for _ in range(math.ceil(k / b))]
        V_blocks = [[object() for _ in range(math.ceil(k / b))] for _ in range(math.ceil(k / b))]
        for j in range(len(U_blocks)):
            assign_all_blocks(U_blocks[j], Ub[j])
            assign_all_blocks(S_blocks[j], Sb[j])
            assign_all_blocks(V_blocks[j], Vb[j])
        U = Q @ Array(blocks=U_blocks, top_left_shape=(b, b), shape=(k, k),
                         reg_shape=(b, b), sparse=False)
        S = Array(blocks=S_blocks, top_left_shape=(b, b), shape=(k, k),
                         reg_shape=(b, b), sparse=False)
        V = P @ Array(blocks=V_blocks, top_left_shape=(b, b), shape=(k, k),
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
    print("Number iteration converged: " + str(compss_wait_on(converged)))
    my_svd(B._blocks, Ub, Sb, Vb, b)
    U_blocks = [[object() for _ in range(math.ceil(k / b))] for _ in range(math.ceil(k / b))]
    S_blocks = [[object() for _ in range(math.ceil(k / b))] for _ in range(math.ceil(k / b))]
    V_blocks = [[object() for _ in range(math.ceil(k / b))] for _ in range(math.ceil(k / b))]
    for j in range(len(U_blocks)):
        assign_all_blocks(U_blocks[j], Ub[j])
        assign_all_blocks(S_blocks[j], Sb[j])
        assign_all_blocks(V_blocks[j], Vb[j])
    U = Q @ Array(blocks=Ub, top_left_shape=(b, b), shape=(k, k),
                  reg_shape=(b, b), sparse=False)
    S = Array(blocks=Sb, top_left_shape=(b, b), shape=(k, k),
              reg_shape=(b, b), sparse=False)
    V = P @ Array(blocks=Vb, top_left_shape=(b, b), shape=(k, k),
                  reg_shape=(b, b), sparse=False)
    return U, S, V
