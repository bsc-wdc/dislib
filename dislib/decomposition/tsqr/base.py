from dislib.data.array import Array, _empty_array
import math
import warnings
import numpy as np
from numpy.linalg import qr

from pycompss.api.task import task
from pycompss.api.parameter import Type, COLLECTION_IN, COLLECTION_OUT, Depth


def tsqr(a: Array, n_reduction=2, mode="complete", indexes=None):
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
    if mode == "complete":

        for block in a._blocks:
            q, r = _compute_qr([block], "complete")
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
                                                   int((i+1)*n_reduction)],
                                             "complete")
                q = _compute_q_from_qs(qsaux[int(i*n_reduction):
                                             int((i+1)*n_reduction)], q,
                                       qsaux[0])
                qs.append(q)
                rs.append(r)
        if indexes is not None:
            matrix_indices = _construct_identity(indexes, a.shape[0])
            q = _multiply(q, matrix_indices)
        q_blocks = [[object()] for _ in range(int(a.shape[0] /
                                                  a._reg_shape[0]))]
        r_blocks = [[object()]]
        _construct_blocks(r_blocks, r, (a.shape[0], a.shape[1]))
        if indexes is not None:
            _construct_blocks(q_blocks, q, (a._reg_shape[0], len(indexes)))
            q = _empty_array(shape=(a.shape[0], len(indexes)),
                             block_size=(a._reg_shape[0], len(indexes)))
        else:
            _construct_blocks(q_blocks, q, (a._reg_shape[0], a.shape[0]))
            q = _empty_array(shape=(a.shape[0], a.shape[0]),
                             block_size=(a._reg_shape[0], a.shape[0]))
        r = _empty_array(shape=(a.shape[0], a.shape[1]),
                         block_size=(a.shape[0], a.shape[1]))
        q._blocks = q_blocks
        r._blocks = r_blocks
        return q, r

    elif mode == "complete_inverse":
        if n_reduction == 2:
            if _is_not_power_of_two(a._n_blocks[0]):
                raise ValueError("This mode only works if the number of "
                                 "blocks is a direct power of the "
                                 "reduction number")
        else:
            if _is_not_power(n_reduction, a._n_blocks[0]):
                raise ValueError(
                    "This mode only works if the number of blocks is a"
                    " direct power of the reduction number")

        for block in a._blocks:
            q, r = _compute_qr([block], "complete")
            qs.append(q)
            rs.append(r)

        while len(rs) > 1:
            reduction_number = math.ceil(len(rs)/n_reduction)
            rsaux = rs
            rs = []
            for i in range(reduction_number):
                q, r = _compute_reduction_qr(rsaux[int(i*n_reduction):
                                                   int((i+1)*n_reduction)],
                                             "complete")
                qs.append(q)
                rs.append(r)
        if indexes is not None:
            matrix_indices = _construct_identity(indexes, a.shape[0])
            q = _construct_q_from_the_end(qs, n_reduction,
                                          indexes=matrix_indices)
        else:
            q = _construct_q_from_the_end(qs, n_reduction)
        q_blocks = [[object()] for _ in range(int(a.shape[0] /
                                                  a._reg_shape[0]))]
        r_blocks = [[object()]]
        _construct_blocks(r_blocks, r, (a.shape[0], a.shape[1]))
        if indexes is not None:
            _construct_blocks(q_blocks, q, (a._reg_shape[0], len(indexes)))
            q = _empty_array(shape=(a.shape[0], len(indexes)),
                             block_size=(a._reg_shape[0], len(indexes)))
        else:
            _construct_blocks(q_blocks, q, (a._reg_shape[0], a.shape[0]))
            q = _empty_array(shape=(a.shape[0], a.shape[0]),
                             block_size=(a._reg_shape[0], a.shape[0]))
        r = _empty_array(shape=(a.shape[0], a.shape[1]),
                         block_size=(a.shape[0], a.shape[1]))
        q._blocks = q_blocks
        r._blocks = r_blocks
        return q, r
    elif mode == "reduced":
        for block in a._blocks:
            q, r = _compute_qr([block], "reduced")
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
                                                   int((i+1)*n_reduction)],
                                             "reduced")
                q = _compute_q_from_qs(qsaux[int(i * n_reduction):
                                             int((i + 1) * n_reduction)], q,
                                       qsaux[0])
                qs.append(q)
                rs.append(r)
        q_blocks = [[object()] for _ in range(int(a.shape[0] /
                                                  a._reg_shape[0]))]
        r_blocks = [[object()]]
        _construct_blocks(q_blocks, q, (a._reg_shape[0], a.shape[1]))
        _construct_blocks(r_blocks, r, (a.shape[1], a.shape[1]))
        q = _empty_array(shape=(a.shape[0], a.shape[1]),
                         block_size=(a._reg_shape[0], a.shape[1]))
        r = _empty_array(shape=(a.shape[1], a.shape[1]),
                         block_size=(a.shape[1], a.shape[1]))
        q._blocks = q_blocks
        r._blocks = r_blocks
        return q, r
    elif mode == "reduced_inverse":
        if n_reduction == 2:
            if _is_not_power_of_two(a._n_blocks[0]):
                raise ValueError("This mode only works if the number of "
                                 "blocks is a direct power of the "
                                 "reduction number")
        else:
            if _is_not_power(n_reduction, a._n_blocks[0]):
                raise ValueError(
                    "This mode only works if the number of blocks is a"
                    " direct power of the reduction number")
        for block in a._blocks:
            q, r = _compute_qr([block], "reduced")
            qs.append(q)
            rs.append(r)

        while len(rs) > 1:
            reduction_number = math.ceil(len(rs)/n_reduction)
            rsaux = rs
            rs = []
            for i in range(reduction_number):
                q, r = _compute_reduction_qr(rsaux[int(i*n_reduction):
                                                   int((i+1)*n_reduction)],
                                             "reduced")
                qs.append(q)
                rs.append(r)
        if indexes is not None:
            matrix_indices = _construct_identity(indexes, a.shape[1])
            q = _construct_q_from_the_end(qs, n_reduction,
                                          indexes=matrix_indices)
        else:
            q = _construct_q_from_the_end(qs, n_reduction)
        q_blocks = [[object()] for _ in range(int(a.shape[0] /
                                                  a._reg_shape[0]))]
        r_blocks = [[object()]]
        _construct_blocks(r_blocks, r, (a.shape[1], a.shape[1]))
        if indexes is not None:
            _construct_blocks(q_blocks, q, (a._reg_shape[0], len(indexes)))
            q = _empty_array(shape=(a.shape[0], len(indexes)),
                             block_size=(a._reg_shape[0], len(indexes)))
        else:
            _construct_blocks(q_blocks, q, (a._reg_shape[0], a.shape[1]))
            q = _empty_array(shape=(a.shape[0], a.shape[1]),
                             block_size=(a._reg_shape[0], a.shape[1]))
        r = _empty_array(shape=(a.shape[1], a.shape[1]),
                         block_size=(a.shape[1], a.shape[1]))
        q._blocks = q_blocks
        r._blocks = r_blocks
        return q, r
    elif mode == "r_complete":
        for block in a._blocks:
            q, r = _compute_qr([block], "complete")
            qs.append(q)
            rs.append(r)

        while len(rs) > 1:
            reduction_number = math.ceil(len(rs)/n_reduction)
            rsaux = rs
            rs = []
            for i in range(reduction_number):
                q, r = _compute_reduction_qr(rsaux[int(i*n_reduction):
                                                   int((i+1)*n_reduction)],
                                             "complete")
                rs.append(r)
        r_blocks = [[object()]]
        _construct_blocks(r_blocks, r, (a.shape[0], a.shape[1]))
        r = _empty_array(shape=(a.shape[0], a.shape[1]),
                         block_size=(a.shape[0], a.shape[1]))
        r._blocks = r_blocks
        return r
    elif mode == "r_reduced":
        for block in a._blocks:
            q, r = _compute_qr([block], "reduced")
            qs.append(q)
            rs.append(r)

        while len(rs) > 1:
            reduction_number = math.ceil(len(rs)/n_reduction)
            rsaux = rs
            rs = []
            for i in range(reduction_number):
                q, r = _compute_reduction_qr(rsaux[int(i*n_reduction):
                                                   int((i+1)*n_reduction)],
                                             "reduced")
                rs.append(r)
        r_blocks = [[object()]]
        _construct_blocks(r_blocks, r, (a.shape[1], a.shape[1]))
        r = _empty_array(shape=(a.shape[1], a.shape[1]),
                         block_size=(a.shape[1], a.shape[1]))
        r._blocks = r_blocks
        return r


def _construct_q_from_the_end(qs, n_reduction, indexes=None):
    if indexes is not None:
        q = _multiply(qs[-1], indexes)
    else:
        q = qs[-1]
    qs.pop()
    depth = 1
    idx = n_reduction**depth
    qs_aux = []
    for q_aux in reversed(qs):
        qs_aux.append(_multiply(q_aux, q, index_start=(idx-1),
                                index_end=(idx)))
        idx = idx - 1
        if idx == 0:
            depth = depth + 1
            idx = n_reduction**depth
            q = _stack_matrices(qs_aux)
            qs_aux = []
        qs.pop()
    return q


def _is_not_power_of_two(number):
    if number < 1:
        return False
    return 0 != (number & (number - 1))


def _is_not_power(n_reduction, number_blocks):
    res1 = math.log(number_blocks) // math.log(n_reduction)
    res2 = math.log(number_blocks) / math.log(n_reduction)
    return 1 if (res1 != res2) else 0


@task(q={Type: COLLECTION_IN, Depth: 3},
      to_multiply={Type: COLLECTION_IN, Depth: 3}, returns=np.array)
def _multiply(q, to_multiply, index_start=None, index_end=None):
    if index_start is not None and index_end is not None:
        return np.dot(q, to_multiply[index_start*q.shape[1]:
                                     index_end*q.shape[1]])
    else:
        return np.dot(q, to_multiply)


@task(matrices={Type: COLLECTION_IN, Depth: 4}, returns=np.array)
def _stack_matrices(matrices):
    return np.vstack(reversed(matrices))


@task(indexes={Type: COLLECTION_IN, Depth: 1}, returns=np.array)
def _construct_identity(indexes, shape):
    identity = np.eye(shape)
    return identity[:, indexes]


@task(blocks={Type: COLLECTION_OUT, Depth: 3})
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
def _compute_reduction_qr(rs, mode):
    if mode == "complete":
        if len(rs) > 1:
            q, r = qr(np.vstack(rs[:]), mode="complete")
        else:
            q, r = qr(rs[0], mode="complete")
        return q, r
    elif mode == "reduced":
        if len(rs) > 1:
            q, r = qr(np.vstack(rs[:]))
        else:
            q, r = qr(rs[0])
        return q, r


@task(block={Type: COLLECTION_IN, Depth: 3}, returns=(np.array, np.array))
def _compute_qr(block, mode):
    if mode == "complete":
        if len(block[0]) > 1:
            block = np.block(block[0])
            q, r = qr(block, mode="complete")
        else:
            q, r = qr(block[0][0], mode="complete")
        return q, r
    elif mode == "reduced":
        if len(block[0]) > 1:
            block = np.block(block[0])
            q, r = qr(block)
        else:
            q, r = qr(block[0][0])
        return q, r
