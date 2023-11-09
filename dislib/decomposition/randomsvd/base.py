import numpy as np
import dislib as ds
from pycompss.api.task import task
from pycompss.api.constraint import constraint
from pycompss.api.reduction import reduction
from dislib.data.array import Array
from pycompss.api.parameter import COLLECTION_IN, COLLECTION_OUT, \
    COLLECTION_INOUT, Depth, Type
from pycompss.api.api import compss_wait_on
from dislib.decomposition import tsqr
import math


@constraint(computing_units="${ComputingUnits}")
@task(
    Ur_blocks={Type: COLLECTION_OUT, Depth: 2},
    S_blocks={Type: COLLECTION_OUT, Depth: 2},
    Vr_blocks={Type: COLLECTION_OUT, Depth: 2},
)
def svd(r_block, Ur_blocks, S_blocks, Vr_blocks):
    u, s, v = np.linalg.svd(r_block, full_matrices=False)
    s = np.diag(s)
    v = v.T

    Ur_shape = len(Ur_blocks), len(Ur_blocks[0])
    for i, row in enumerate(np.array_split(u, Ur_shape[0])):
        for j, val in enumerate(np.array_split(row, Ur_shape[1], axis=1)):
            Ur_blocks[i][j] = val

    S_shape = len(S_blocks), len(S_blocks[0])
    for i, row in enumerate(np.array_split(s, S_shape[0])):
        for j, val in enumerate(np.array_split(row, S_shape[1], axis=1)):
            S_blocks[i][j] = val

    Vr_shape = len(Vr_blocks), len(Vr_blocks[0])
    for i, row in enumerate(np.array_split(v, Vr_shape[0])):
        for j, val in enumerate(np.array_split(row, Vr_shape[1], axis=1)):
            Vr_blocks[i][j] = val


def check_convergeces(res, tol, nsv):
    res = np.block(res).flatten()
    converged = res < tol
    return np.all(converged[:nsv]), max(res[:nsv])


@constraint(computing_units="${ComputingUnits}")
@task(B_blocks={Type: COLLECTION_INOUT, Depth: 2}, returns=1)
def R_ortho(B_blocks, R, k, b, B_reg_shape):
    B = np.block(B_blocks)
    R = R.astype(np.float64)

    for j in range(0, k, b):
        w = B[:, j:j+b]
        if j > 0:
            h = B[:, 0:j].T @ w
            w = w - B[:, 0:j] @ h
        w, r1 = np.linalg.qr(w)

        if j > 0:
            h2 = B[:, 0:j].T @ w
            w = w - B[:, 0:j] @ h2
            R[0:j, j:j+b][:] = h + h2
        w, r2 = np.linalg.qr(w)

        B[:, j:j+b] = w
        R[j:j+b, j:j+b][:] = r1 @ r2

    n, m = B_reg_shape
    for i in range(len(B_blocks)):
        for j in range(len(B_blocks[i])):
            B_blocks[i][j] = B[i*n:(i+1)*n, j*m:(j+1)*m]

    return R


def _get_all_cols(A, i):
    blocks = [[] for j in range(A._n_blocks[0])]
    shape = [A.shape[0], 0]
    for col_idx in range(i+1):
        col_shape = A._get_col_shape(col_idx)
        shape[1] += col_shape[1]
        for j in range(A._n_blocks[0]):
            blocks[j].append(A._blocks[j][i])
    return Array(blocks=blocks,
                 top_left_shape=A._reg_shape,
                 reg_shape=A._reg_shape, shape=col_shape,
                 sparse=A._sparse, delete=False)


@task(returns=1)
def _norm(b):
    return np.linalg.norm(b, axis=0)


@reduction(chunk_size="8")
@task(blocks={Type: COLLECTION_IN, Depth: 1}, returns=1)
def _norm_red(blocks):
    return np.linalg.norm(blocks, axis=0)


def my_norm(A):
    norm_blocks = Array._get_out_blocks(A._n_blocks)
    for i in range(A._n_blocks[0]):
        for j in range(A._n_blocks[1]):
            norm_blocks[i][j] = _norm(A._blocks[i][j])

    norms = []
    for j in range(A._n_blocks[1]):
        col = [norm_blocks[i][j] for i in range(A._n_blocks[0])]
        if len(col) > 1:
            norms.append(_norm_red(col))
        else:
            norms.append(col[0])
    return norms


def nsv_tolerance(m, n, nsv, S):
    min_sval = S[nsv-1]
    total_sval = min(m, n)
    sum_nsv = sum([s**2 for s in S])
    sum_remaining = (total_sval-nsv) * min_sval**2
    tol = np.sqrt(sum_remaining / (sum_remaining + sum_nsv))
    return tol


def _svd_random(A, b, k, max_restarts, nsv, tol, verbose=False):
    # A: matrix
    # b: block size
    # k: number of vectors (must be multiple of b)
    # max_restarts: maximum number of restarts
    # nsv: number of desired singular values
    # tol: consider a singular converged when its residual is less than tol
    m, n = A.shape

    B = ds.random_array((n, k), (b, b), random_state=0)
    R = ds.zeros((k, k), (k, k))
    Q = ds.matmul(A, B)

    for iters in range(max_restarts):
        for j in range(0, k, b):
            col_idx = j // b
            q = Q._get_col_block(col_idx)
            if j > 0:
                h = ds.matmul(_get_all_cols(Q, col_idx), q, transpose_a=True)
                q = ds.data.matsubtract(
                    q, ds.matmul(_get_all_cols(Q, col_idx), h))
                h = ds.matmul(_get_all_cols(Q, col_idx), q, transpose_a=True)
                q = ds.data.matsubtract(
                    q, ds.matmul(_get_all_cols(Q, col_idx), h))

            q, _ = tsqr(q, mode='reduced')
            q._delete = False

            for i in range(Q._n_blocks[0]):
                Q._blocks[i][col_idx] = q._blocks[i][0]

        B = ds.matmul(A, Q, transpose_a=True)

        R._blocks[0][0] = R_ortho(
            B._blocks, R._blocks[0][0], k, b, B._reg_shape)

        Ur_blocks = Array._get_out_blocks((k//b, k//b))
        Vr_blocks = Array._get_out_blocks((k//b, k//b))
        S_blocks = Array._get_out_blocks((k//b, k//b))
        svd(R._blocks[0][0], Ur_blocks, S_blocks, Vr_blocks)

        Ur = Array(Ur_blocks, (b, b), (b, b), (R.shape[0], R.shape[0]), False)
        S = Array(S_blocks, (b, b), (b, b), (R.shape[1], R.shape[1]), False)
        Vr = Array(Vr_blocks, (b, b), (b, b), (R.shape[1], R.shape[1]), False)

        U = ds.matmul(Q, Vr)
        V = ds.matmul(B, Ur)

        Q = ds.matmul(A, V)

        mmm = ds.data.matsubtract(Q, ds.matmul(U, S))

        res = np.block(compss_wait_on(my_norm(mmm)))
        converged, conv_tol = check_convergeces(res, tol, nsv)
        if converged:
            if verbose:
                print("Converged in {} iterations to {} precision.".format(
                    iters+1, conv_tol))
            break
        else:
            if verbose:
                print("""Not converged in {} iterations.
                      Current precision: {}.""".format(iters+1, conv_tol))

    return U, S, V, conv_tol


def random_svd(a, iters, epsilon, tol, nsv=None, k=None, verbose=False):
    """ Random SVD

    Parameters
    ----------
    a : ds-arrays
        Input ds-array. Its blocksize will condition this
    iters : int
        Number of inner iterations for the Random algorithm to converge.
    epsilon : float64
        Value that defines a tolerance for how many singular values are
        required to satisfy that value, as it is reduced, the number of
        singular values required is increased. The algorithm will automatically
        try to reach that level of tolerance.
    tol : float64
        If the residual value of a singular value is smaller than this
        tolerance, that singular value is considered to be converged.
    nsv : int
        Number of desired singular values
    k : int
        Number of restarting vectors. Must be a multiple of `a` blocksize
        and greater than `nsv`.
    verbose : bool
        Controls the verbosity for the algorithm convergence.
        Shows convergence accuracy and singular value criteria
        for each iteration.

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
            If k < num_nsv
            or
            If k % b != 0
    """
    m, n = a.shape
    b = a._reg_shape[1]

    if nsv is None:
        nsv = np.argmin((np.logspace(0, -20, n) - epsilon)**2)

    if k is None:
        if b == nsv:
            k = b
        elif b > nsv:
            k = b
        else:
            k = math.ceil(nsv / b) * b

    if nsv > n:
        raise ValueError("Number of singular values to compute can't"
                         " be bigger than the total number of singular"
                         "values of the matrix.")
    if k < nsv:
        raise ValueError("Rank should be at least the number of singular"
                         "values to compute.")

    if k % b != 0:
        raise ValueError("K should be multiple of `A` block size.")

    if verbose:
        print(f'''Random-SVD: Starting computing {nsv}
                  singular values using {k} vectors.''')

    epsilon_conv = False
    while not epsilon_conv:
        U, S, V, conv_tol = _svd_random(
            a, b, k, iters, nsv, tol, verbose=verbose)

        if conv_tol > tol:
            print(f'''WARNING: Random-SVD did not converge in
                      {iters} iterations to {tol} precision.''')
            print(f'         Maximum accuracy was {conv_tol}')

        S_flat = np.diag(S.collect())
        epsilon_hat = nsv_tolerance(m, n, nsv, S_flat)
        if epsilon_hat <= epsilon:
            epsilon_conv = True
        else:
            k += b
            nsv += b
            if verbose:
                print(f'''NOTE: Random-SVD singular value
                        criteria {epsilon_hat} > {epsilon}.''')
                print(f'''      Increasing number of vectors
                        to {k} and {nsv} singular values.''')

    if verbose:
        print(
            f'Random-SVD: Converged to {epsilon_hat} singular value tolerance')

    return U, S, V
