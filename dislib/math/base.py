from pycompss.api.api import compss_delete_object
from pycompss.api.parameter import COLLECTION_INOUT, Type, Depth
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
    """
    if a._sparse or b._sparse:
        raise NotImplementedError("Sparse ds-arrays not supported.")

    if not block_size:
        block_size = b._reg_shape

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

    out = Array._rechunk(k_blocks, shape, block_size, _kron_shape_f, b)

    for blocks in k_blocks:
        for block in blocks:
            compss_delete_object(block)

    return out


def _kron_shape_f(i, j, b):
    return b._get_block_shape(i % b._n_blocks[0], j % b._n_blocks[1])


@task(out_blocks={Type: COLLECTION_INOUT, Depth: 2})
def _kron(block1, block2, out_blocks):
    """ Computes the kronecker product of two blocks and returns one ndarray
    per (element-in-block1, block2) pair."""
    for i in range(block1.shape[0]):
        for j in range(block1.shape[1]):
            out_blocks[i][j] = block1[i, j] * block2
