import numpy as np
from pycompss.api.api import compss_delete_object
from pycompss.api.parameter import COLLECTION_INOUT, Type, Depth, COLLECTION_IN
from pycompss.api.task import task

from dislib.data.array import Array


def kron(a, b, block_size=None):
    """ Kronecker product of two ds-arrays.

    Parameters
    ----------
    a, b : ds-arrays
        Input ds-arrays.
    block_size : tuple of two ints, optional
        Block size of the resulting array. Defaults to the block size of `b`.

    Returns
    -------
    out : ds-array

    Raises
    ------
    NotImplementedError
        If a or b are sparse.
    """
    if a._sparse or b._sparse:
        raise NotImplementedError("Sparse ds-arrays not supported.")

    k_n_blocks = ((a.shape[0] * b._n_blocks[0]),
                  a.shape[1] * b._n_blocks[1])
    k_blocks = Array._get_out_blocks(k_n_blocks)

    # compute the kronecker product by multipliying b by each element in a.
    # The resulting array keeps the block structure of b repeated many
    # times. This is why we need to rechunk it at the end.
    offseti = 0

    for i in range(a._n_blocks[0]):
        offsetj = 0

        for j in range(a._n_blocks[1]):
            bshape_a = a._get_block_shape(i, j)

            for k in range(b._n_blocks[0]):
                for l in range(b._n_blocks[1]):
                    out_blocks = Array._get_out_blocks(bshape_a)
                    _kron(a._blocks[i][j], b._blocks[k][l], out_blocks)

                    for m in range(bshape_a[0]):
                        for n in range(bshape_a[1]):
                            bi = (offseti + m) * b._n_blocks[0] + k
                            bj = (offsetj + n) * b._n_blocks[1] + l
                            k_blocks[bi][bj] = out_blocks[m][n]

            offsetj += bshape_a[1]
        offseti += bshape_a[0]

    shape = (a.shape[0] * b.shape[0], a.shape[1] * b.shape[1])

    if not block_size:
        bsize = b._reg_shape
    else:
        bsize = block_size

    # rechunk the array unless all blocks of b are of the same size and
    # block_size is None
    if (not block_size or block_size == b._reg_shape) and (
            b.shape[0] % b._reg_shape[0] == 0 and
            b.shape[1] % b._reg_shape[1] == 0 and
            b._is_regular()):
        return Array(k_blocks, bsize, bsize, shape, False)
    else:
        out = Array._rechunk(k_blocks, shape, bsize, _kron_shape_f, b)

        for blocks in k_blocks:
            for block in blocks:
                compss_delete_object(block)

        return out


def hessenberg(a):
    """ Compute Hessenberg form of a matrix.

    Parameters
    ----------
    a : ds-array, shape=(n, n)
        Square matrix.

    Returns
    -------
    H : ds-array, shape=(n, n)
        Hessenberg form of a.

    Raises
    ------
    ValueError
        If a is not square.
    NotImplementedError
        If a is sparse.
    """
    if a.shape[0] != a.shape[1]:
        raise ValueError("Input ds-array is not square.")

    if a._sparse:
        raise NotImplementedError("Sparse ds-arrays not supported.")

    k = 0
    col_block = a._get_col_block(0)
    a1 = a

    for block_i in range(a._n_blocks[1]):
        for i in range(col_block.shape[1]):
            p = _compute_p(col_block, i, k)
            a1 = p @ a1 @ p
            k += 1

            if k == a.shape[1] - 2:
                return a1

            col_block = a1._get_col_block(block_i)


def _compute_p(col_block, col_i, k):
    """
    P = I - 2 * v @ v.T

    We generate each block of 2v and v.T in a single task and then perform
    matrix multiplication
    """
    alpha_r = _compute_alpha(col_block._blocks, col_i, k)

    n_blocks = col_block._n_blocks[0]

    v_blocks = Array._get_out_blocks((n_blocks, 1))
    vt_blocks = Array._get_out_blocks((1, n_blocks))
    j = 0

    for i in range(n_blocks):
        block = col_block._blocks[i][0]
        n_rows = col_block._get_block_shape(i, 0)[0]

        if k >= j + n_rows - 1:
            # v[i] is 0 for all i <= k
            v_block, vt_block = _gen_zero_blocks(n_rows)
        else:
            v_block, vt_block = _gen_v_blocks(block, col_i, alpha_r, j, k)

        v_blocks[i][0] = v_block
        vt_blocks[0][i] = vt_block
        j += n_rows

    v = Array(v_blocks, (col_block._top_left_shape[0], 1),
              (col_block._reg_shape[0], 1), (col_block.shape[0], 1), False)
    vt = Array(vt_blocks, (1, col_block._top_left_shape[0]),
               (1, col_block._reg_shape[0]), (1, col_block.shape[0]), False)

    prod = v @ vt

    # now we substract the 2v @ v.T from I to get P
    p_blocks = Array._get_out_blocks((n_blocks, n_blocks))

    for i in range(n_blocks):
        for j in range(n_blocks):
            p_blocks[i][j] = _substract_from_i(prod._blocks[i][j], i == j)

    return Array(p_blocks, prod._top_left_shape, prod._reg_shape,
                 prod.shape, False)


@task(returns=1)
def _substract_from_i(block, is_diag):
    if is_diag:
        return np.identity(block.shape[0]) - block
    else:
        return -1 * block


@task(returns=2)
def _gen_zero_blocks(n):
    return np.zeros((n, 1)), np.zeros((1, n))


@task(returns=2)
def _gen_v_blocks(col_block, col_i, alpha_r, i, k):
    alpha, r = alpha_r
    col = col_block[:, col_i]
    n = col.shape[0]

    v = np.zeros((n, 1))

    if alpha == 0:
        return v, v.T

    k1 = k - i + 1

    if 0 <= k1 < n:
        v[k1] = (col[k1] - alpha) / (2 * r)

    for j in range(max(0, k - i + 2), n):
        v[j] = col[j] / (2 * r)

    return 2 * v, v.T


@task(col_block={Type: COLLECTION_IN, Depth: 2}, returns=1)
def _compute_alpha(col_block, col_i, k):
    col = Array._merge_blocks(col_block)[:, col_i]
    alpha = - np.sign(col[k + 1]) * np.linalg.norm(col[k + 1:])
    r = np.sqrt(0.5 * (alpha ** 2 - col[k + 1] * alpha))
    return alpha, r


def _kron_shape_f(i, j, b):
    return b._get_block_shape(i % b._n_blocks[0], j % b._n_blocks[1])


@task(out_blocks={Type: COLLECTION_INOUT, Depth: 2})
def _kron(block1, block2, out_blocks):
    """ Computes the kronecker product of two blocks and returns one ndarray
    per (element-in-block1, block2) pair."""
    for i in range(block1.shape[0]):
        for j in range(block1.shape[1]):
            out_blocks[i][j] = block1[i, j] * block2
