from pycompss.api.parameter import Type, Depth, COLLECTION_INOUT
from pycompss.api.task import task

from dislib.data.array import Array


def kron(a, b):
    """ Kronecker product of two ds-arrays.

    If `a` has `n x m` blocks and `b` has `p x q` blocks, the resulting
    ds-array has `np x mq` blocks.

    Parameters
    ----------
    a, b : ds-arrays

    Returns
    -------
    out : ds-array
    """
    if a._sparse or b._sparse:
        raise NotImplementedError("Kronecker product of sparse ds-arrays not "
                                  "supported yet.")

    n_blocks = (a._shape[0] * b._n_blocks[0],
                a._shape[1] * b._n_blocks[1])

    blocks = Array._get_out_blocks(n_blocks)
    offseti = 0

    for i in range(a._n_blocks[0]):
        offsetj = 0

        for j in range(a._n_blocks[1]):
            block_shape = a._get_block_shape(i, j)

            for k in range(b._n_blocks[0]):
                for l in range(b._n_blocks[1]):
                    out_blocks = Array._get_out_blocks(block_shape)
                    _kron(a._blocks[i][j], b._blocks[k][l], out_blocks)

                    for m in range(block_shape[0]):
                        for n in range(block_shape[1]):
                            bi = (offseti + m) * b._n_blocks[0] + k
                            bj = (offsetj + n) * b._n_blocks[1] + l
                            blocks[bi][bj] = out_blocks[m][n]

            offsetj += block_shape[1]
        offseti += block_shape[0]

    tl_shape = (a._top_left_shape[0] * b._top_left_shape[0],
                a._top_left_shape[1] * b._top_left_shape[1])
    shape = (a.shape[0] * b.shape[0], a.shape[1] * b.shape[1])
    r_shape = (a._reg_shape[0] * b._reg_shape[0],
               a._reg_shape[1] * b._reg_shape[1])

    return Array(blocks, tl_shape, r_shape, shape, False)


@task(out_blocks={Type: COLLECTION_INOUT, Depth: 2})
def _kron(block1, block2, out_blocks):
    """ Computes the product of two blocks and returns one ndarray per
    (element-in-block1, block2) pair."""
    for i in range(block1.shape[0]):
        for j in range(block1.shape[1]):
            out_blocks[i][j] = block1[i][j] * block2
