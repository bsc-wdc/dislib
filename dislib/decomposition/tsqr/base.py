from dislib.data.array import Array, _empty_array
import math
import warnings
import numpy as np
from numpy.linalg import qr

from pycompss.api.task import task
from pycompss.api.parameter import Type, COLLECTION_IN, COLLECTION_OUT, Depth
from dislib.data.array import matmul, concat_rows
from pycompss.api.constraint import constraint


def tsqr(a: Array, n_reduction=2, mode="complete", indexes=None):
    """ QR Decomposition for vertically long arrays.

        Parameters
        ----------
        a : ds-arrays
            Input ds-array.
        n_reduction : int
            Number of row of blocks used in each reduction operation.
        mode: basestring
            Mode of execution of the tsqr. The options are:
            - complete: q=mxm, r=mxn computed from beginning to end
            - complete_inverse: q=mxm, r=mxn computed from end to beginning
            - reduced: q=mxn, r=nxn computed from beginning to end
            - reduced_inverse: q=mxn, r=nxn computed from end to beginning
            - r_complete: returns only r. This r is mxn
            - r_reduced: returns only r. This r is nxn

        Returns
        -------
        q : ds-array
            The q of the matrix, it is an orthonormal matrix,
            multiplying it by r will return the initial
            matrix
            In r_complete and r_reduced modes this will not be returned
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
            or
            If the mode is complete_inverse and the number of blocks
            is not a power of the reduction number
            or
            If the mode is reduced_inverse and the number of blocks is
            not a power of the reduction number

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
        for i, block in enumerate(a._blocks):
            q, r = _compute_qr([block], "complete")
            q_ds_array = Array([[q]], top_left_shape=(a._top_left_shape[0],
                                                      a._top_left_shape[0]),
                               reg_shape=(a._reg_shape[0], a._reg_shape[0]),
                               shape=(a._reg_shape[0], a._reg_shape[0]),
                               sparse=False)
            qs.append(q_ds_array)
            rs.append(r)
        while len(rs) > 1:
            reduction_number = math.ceil(len(rs) / n_reduction)
            qsaux = qs
            rsaux = rs
            shape_to_use = qsaux[0].shape[1]
            rs = []
            qs = []
            for i in range(reduction_number):
                q, r = _compute_reduction_qr(rsaux[int(i * n_reduction):
                                                   int((i + 1) * n_reduction)],
                                             "complete")
                small_q = Array([[q]], top_left_shape=(a._reg_shape[0],
                                                       a._reg_shape[0]),
                                reg_shape=(a._reg_shape[0], a._reg_shape[0]),
                                shape=(shape_to_use * 2, shape_to_use * 2),
                                sparse=False)
                q_1 = matmul(qsaux.pop(0), small_q[:shape_to_use])
                if len(qsaux) > 0:
                    q_2 = matmul(qsaux.pop(0), small_q[shape_to_use:])
                    qs.append(concat_rows(q_1, q_2))
                else:
                    q_1._shape = (q_1.shape[0], q_1.shape[1] - a._reg_shape[0])
                    qs.append(q_1)
                rs.append(r)
        q = qs[0]
        if indexes is not None:
            q = q[:, indexes]
        elif a._n_blocks[0] % n_reduction != 0:
            q._shape = (q._shape[0], q._shape[1] - a._reg_shape[0])

        r_blocks = [[object()]]
        _construct_blocks(r_blocks, r, (a.shape[0], a.shape[1]))
        r = _empty_array(shape=(a.shape[0], a.shape[1]),
                         block_size=(a.shape[0], a.shape[1]))
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
            reduction_number = math.ceil(len(rs) / n_reduction)
            rsaux = rs
            rs = []
            for i in range(reduction_number):
                q, r = _compute_reduction_qr(rsaux[int(i * n_reduction):
                                                   int((i + 1) *
                                                       n_reduction)],
                                             "complete")
                qs.append(q)
                rs.append(r)

        if indexes is not None:
            matrix_indices = _construct_identity(indexes, a.shape[0])
            q = _construct_q_from_the_end(qs, n_reduction, a._reg_shape[0],
                                          a._reg_shape[1], a.shape,
                                          indexes=matrix_indices,
                                          len_indexes=len(indexes),
                                          complete=True)
        else:
            q = _construct_q_from_the_end(qs, n_reduction, a._reg_shape[0],
                                          a._reg_shape[1], a.shape,
                                          complete=True)
        r_blocks = [[object()]]
        _construct_blocks(r_blocks, r, (a.shape[0], a.shape[1]))
        r = _empty_array(shape=(a.shape[0], a.shape[1]),
                         block_size=(a.shape[0], a.shape[1]))
        r._blocks = r_blocks
        return q, r
    elif mode == "reduced":
        for i, block in enumerate(a._blocks):
            q, r = _compute_qr([block], "reduced")
            q_ds_array = Array([[q]], top_left_shape=a._top_left_shape,
                               reg_shape=a._reg_shape, shape=a._reg_shape,
                               sparse=False)
            qs.append(q_ds_array)
            rs.append(r)

        while len(rs) > 1:
            reduction_number = math.ceil(len(rs) / n_reduction)
            qsaux = qs
            rsaux = rs
            rs = []
            qs = []
            for i in range(reduction_number):
                q, r = _compute_reduction_qr(rsaux[int(i * n_reduction):
                                                   int((i + 1) *
                                                       n_reduction)],
                                             "reduced")
                small_q = Array([[q]], top_left_shape=(a._reg_shape[1] *
                                                       n_reduction,
                                                       a._reg_shape[1]),
                                reg_shape=(a._reg_shape[1], a._reg_shape[1]),
                                shape=(a._reg_shape[1] * n_reduction,
                                       a._reg_shape[1]), sparse=False)

                q_1 = matmul(qsaux.pop(0), small_q[:a._reg_shape[1]])
                if len(qsaux) > 0:
                    q_2 = matmul(qsaux.pop(0), small_q[a._reg_shape[1]:])
                    qs.append(concat_rows(q_1, q_2))
                else:
                    qs.append(q_1)
                rs.append(r)
        r_blocks = [[object()]]
        _construct_blocks(r_blocks, r, (a.shape[1], a.shape[1]))
        r = _empty_array(shape=(a.shape[1], a.shape[1]),
                         block_size=(a.shape[1], a.shape[1]))
        r._blocks = r_blocks
        return qs[-1], r
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
        total_depth = 1
        for block in a._blocks:
            q, r = _compute_qr([block], "reduced")
            qs.append(q)
            rs.append(r)

        while len(rs) > 1:
            reduction_number = math.ceil(len(rs) / n_reduction)
            total_depth += 1
            rsaux = rs
            rs = []
            for i in range(reduction_number):
                q, r = _compute_reduction_qr(rsaux[int(i * n_reduction):
                                                   int((i + 1) *
                                                       n_reduction)],
                                             "reduced")
                qs.append(q)
                rs.append(r)
        if indexes is not None:
            matrix_indices = _construct_identity(indexes, a.shape[1])
            q = _construct_q_from_the_end(qs, n_reduction, a._reg_shape[0],
                                          a._reg_shape[1], a.shape,
                                          indexes=matrix_indices,
                                          len_indexes=len(indexes),
                                          total_depth=total_depth,
                                          complete=False)
        else:
            q = _construct_q_from_the_end(qs, n_reduction, a._reg_shape[0],
                                          a._reg_shape[1], a.shape,
                                          total_depth=total_depth,
                                          complete=False)
        r_blocks = [[object()]]
        _construct_blocks(r_blocks, r, (a.shape[1], a.shape[1]))
        r = _empty_array(shape=(a.shape[1], a.shape[1]),
                         block_size=(a.shape[1], a.shape[1]))
        r._blocks = r_blocks
        return q, r
    elif mode == "r_complete":
        for block in a._blocks:
            q, r = _compute_qr([block], "complete")
            qs.append(q)
            rs.append(r)

        while len(rs) > 1:
            reduction_number = math.ceil(len(rs) / n_reduction)
            rsaux = rs
            rs = []
            for i in range(reduction_number):
                q, r = _compute_reduction_qr(rsaux[int(i * n_reduction):
                                                   int((i + 1) * n_reduction)],
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
            reduction_number = math.ceil(len(rs) / n_reduction)
            rsaux = rs
            rs = []
            for i in range(reduction_number):
                q, r = _compute_reduction_qr(rsaux[int(i * n_reduction):
                                                   int((i + 1) * n_reduction)],
                                             "reduced")
                rs.append(r)
        r_blocks = [[object()]]
        _construct_blocks(r_blocks, r, (a.shape[1], a.shape[1]))
        r = _empty_array(shape=(a.shape[1], a.shape[1]),
                         block_size=(a.shape[1], a.shape[1]))
        r._blocks = r_blocks
        return r


def _construct_q_from_the_end(qs, n_reduction, reg_shape_0, reg_shape_1,
                              shape, indexes=None, len_indexes=0,
                              total_depth=1,
                              complete=False):
    if indexes is not None:
        q = _multiply(qs[-1], indexes)
    else:
        q = qs[-1]
    qs.pop()
    depth = 1
    idx = n_reduction ** depth
    qs_aux = []
    if complete:
        if indexes is None:
            indexes_shape = shape[0]
        else:
            indexes_shape = len_indexes
        q = Array([[q]], top_left_shape=(shape[0], shape[0]),
                  reg_shape=(shape[0], shape[0]),
                  shape=(shape[0], indexes_shape),
                  sparse=False)
        last_block_shape = shape[0] - (math.ceil(shape[0] / reg_shape_0)) \
            * reg_shape_0 / 2
        reg_shape = shape[0] - last_block_shape
    else:
        if indexes is None:
            q = Array([[q]], top_left_shape=(reg_shape_1 * n_reduction,
                                             reg_shape_1),
                      reg_shape=(reg_shape_1 * n_reduction, reg_shape_1),
                      shape=(reg_shape_1 * n_reduction, reg_shape_1),
                      sparse=False)
        else:
            q = Array([[q]], top_left_shape=(reg_shape_1 * n_reduction,
                                             len_indexes),
                      reg_shape=(reg_shape_1 * n_reduction, len_indexes),
                      shape=(reg_shape_1 * n_reduction, len_indexes),
                      sparse=False)
    for q_aux in reversed(qs):
        if complete:
            iteration_shape = (int(last_block_shape *
                                   (n_reduction ** depth == idx) +
                                   reg_shape * (n_reduction ** depth != idx)),
                               int(last_block_shape *
                                   (n_reduction ** depth == idx) +
                                   reg_shape * (n_reduction ** depth != idx)))
        else:
            if depth == (total_depth - 1):
                iteration_shape = (reg_shape_0, reg_shape_1)
            else:
                iteration_shape = (reg_shape_1 * n_reduction, reg_shape_1)
        q_aux = Array([[q_aux]], top_left_shape=iteration_shape,
                      reg_shape=iteration_shape,
                      shape=iteration_shape,
                      sparse=False)
        if complete:
            q_1 = q[int(reg_shape * (idx - 1)):int(reg_shape * idx)]
        else:
            q_1 = q[q_aux.shape[1] * (idx - 1):q_aux.shape[1] * (idx)]
        q_1._reg_shape = q_1._top_left_shape
        qs_aux.append(matmul(q_aux, q_1))
        idx = idx - 1
        if idx == 0:
            depth = depth + 1
            if complete:
                last_block_shape = int(shape[0] -
                                       (((math.ceil(shape[0] / reg_shape_0))
                                         * reg_shape_0) / 2 ** depth) *
                                       (2 ** depth - 1))
                reg_shape = int((shape[0] - last_block_shape) /
                                (2 ** depth - 1))
            idx = n_reduction ** depth
            q = concat_rows(qs_aux[-1], qs_aux[-2])
            qs_aux.pop(-1)
            qs_aux.pop(-1)
            for q_auxiliar in reversed(qs_aux):
                q = concat_rows(q, q_auxiliar)
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


@constraint(computing_units="${ComputingUnits}")
@task(returns=np.array)
def _multiply(q, to_multiply):
    return np.dot(q, to_multiply)


@constraint(computing_units="${ComputingUnits}")
@task(indexes={Type: COLLECTION_IN, Depth: 1}, returns=np.array)
def _construct_identity(indexes, shape):
    identity = np.eye(shape)
    return identity[:, indexes]


@constraint(computing_units="${ComputingUnits}")
@task(blocks={Type: COLLECTION_OUT, Depth: 2})
def _construct_blocks(blocks, array_to_place, block_shape):
    for idx, block in enumerate(blocks):
        blocks[idx][0] = array_to_place[idx * block_shape[0]:
                                        (idx + 1) * block_shape[0]]


@constraint(computing_units="${ComputingUnits}")
@task(rs={Type: COLLECTION_IN, Depth: 2}, returns=(np.array, np.array))
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


@constraint(computing_units="${ComputingUnits}")
@task(block={Type: COLLECTION_IN, Depth: 2}, returns=(np.array, np.array))
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
