from dislib.data.array import Array, _empty_array
import math
import warnings
import numpy as np
from numpy.linalg import qr

from pycompss.api.task import task
from pycompss.api.parameter import Type, COLLECTION_IN, COLLECTION_INOUT, Depth


def tsqr(a: Array, n_reduction=2):
    """ QR Decomposition for vertically long arrays.

        Parameters
        ----------
        a : ds-arrays
            Input ds-array.
        n_reduction : int
            Number of row of blocks used in each reduction operation.

        Returns
        -------
        q : ds-array
            The q of the matrix, it is an orthonormal matrix,
            multiplying it by r will return the initial
            matrix
        r : ds-array
            The r of the matrix, it is the upper triangular matrix,
            being multiplied by q it will return the
            initial matrix

        Raises
        ------
        ValueError
            If top left shape is different than regular
            or
            If m < n

        UserWarning
            If the decomposed ds-array contains more than one block columns.
        """
    if a._n_blocks[1] > 1:
        warnings.warn("The method you are trying to use works with "
                      "one column ds-arrays. The returned q and r will "
                      "not preserve the number of columns.", UserWarning)

    if a._reg_shape != a._top_left_shape:
        raise ValueError(
            "Top left block needs to be of the same shape as regular ones"
        )

    if a.shape[0] < a.shape[1]:
        raise ValueError(
            "It is necessary that the matrix has equal or higher number of "
            "rows than columns."
        )

    qs = []
    rs = []

    for block in a._blocks:
        q, r = _compute_qr([block])
        qs.append(q)
        rs.append(r)

    while len(rs) > 1:
        reduction_number = math.ceil(len(rs)/n_reduction)
        qsaux = qs
        rsaux = rs
        rs = []
        qs = []
        for i in range(reduction_number):
            q, r = _compute_reduction_qr(rsaux[int(i*n_reduction):
                                               int((i+1)*n_reduction)])
            q = _compute_q_from_qs(qsaux[int(i*n_reduction):
                                         int((i+1)*n_reduction)], q, qsaux[0])
            qs.append(q)
            rs.append(r)
    q_blocks = [[object()] for _ in range(int(a.shape[0] / a._reg_shape[0]))]
    r_blocks = [[object()]]
    _construct_blocks(q_blocks, q, a._reg_shape)
    _construct_blocks(r_blocks, r, (a.shape[1], a.shape[1]))
    q = _empty_array(shape=a.shape, block_size=(a._reg_shape[0], a.shape[1]))
    r = _empty_array(shape=(a.shape[1], a.shape[1]),
                     block_size=(a.shape[1], a.shape[1]))
    q._blocks = q_blocks
    r._blocks = r_blocks
    return q, r


@task(blocks={Type: COLLECTION_INOUT, Depth: 3})
def _construct_blocks(blocks, array_to_place, block_shape):
    for idx, block in enumerate(blocks):
        blocks[idx][0] = array_to_place[idx*block_shape[0]:
                                        (idx+1)*block_shape[0]]


@task(qs={Type: COLLECTION_IN, Depth: 3}, returns=np.array)
def _compute_q_from_qs(qs, new_q, qsaux):
    if len(qs) > 1:
        final_q = []
        for idx, q in enumerate(qs):
            final_q.append(np.dot(q, new_q[idx*qsaux.shape[1]:
                                           int(idx+1)*qsaux.shape[1]]))
        return np.vstack(final_q)
    else:
        return np.dot(qs[0], new_q[:qsaux.shape[1]])


@task(rs={Type: COLLECTION_IN, Depth: 3}, returns=(np.array, np.array))
def _compute_reduction_qr(rs):
    if len(rs) > 1:
        q, r = qr(np.vstack(rs[:]))
    else:
        q, r = qr(rs[0])
    return q, r


@task(block={Type: COLLECTION_IN, Depth: 3}, returns=(np.array, np.array))
def _compute_qr(block):
    if len(block[0]) > 1:
        block = np.block(block[0])
        q, r = qr(block)
    else:
        q, r = qr(block[0][0])
    return q, r
