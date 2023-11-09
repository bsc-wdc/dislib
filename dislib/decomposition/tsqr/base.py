from dislib.data.array import Array, matmul, concat_rows
import math
import numpy as np
from numpy.linalg import qr

from pycompss.api.task import task
from pycompss.api.parameter import Type, COLLECTION_IN, COLLECTION_OUT, Depth
from pycompss.api.constraint import constraint


def tsqr(a: Array, mode="complete", indexes=None):
    """ QR Decomposition for vertically long arrays.

        Parameters
        ----------
        a : ds-arrays
            Input ds-array.
        mode: basestring
            Mode of execution of the tsqr. The options are:
            - complete: q=mxm, r=mxn computed from beginning to end
            - complete_inverse: q=mxm, r=mxn computed from end to beginning
            - reduced: q=mxn, r=nxn computed from beginning to end
            - reduced_inverse: q=mxn, r=nxn computed from end to beginning
            - r_complete: returns only r. This r is mxn
            - r_reduced: returns only r. This r is nxn
        indexes: list
            Columns to return, it only works when it is set with an inverse
            mode. In other cases it will be ignored.

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
            If the mode is reduced or reduced_inverse and the number
            of rows per block is smaller than the total number of
            columns of the matrix
            or
            If the mode is complete_inverse and the number of blocks
            is not a power of 2
            or
            If the mode is reduced_inverse and the number of blocks is
            not a power of 2
        """

    if a._reg_shape != a._top_left_shape:
        raise ValueError(
            "Top left block needs to be of the same shape as regular ones"
        )

    if a.shape[0] < a.shape[1]:
        raise ValueError(
            "It is necessary that the matrix has equal or higher number of "
            "rows than columns."
        )
    n_reduction = 2
    qs = []
    rs = []
    if mode == "complete":
        for i, block in enumerate(a._blocks):
            q_blocks = [[object()]]
            r_blocks = [[object() for _ in range(len(block))]]
            _compute_qr([block], q_blocks, r_blocks, "complete")
            if i == len(a._blocks) - 1 and a._shape[0] % a._reg_shape[0] != 0:
                q_ds_array = Array(q_blocks, top_left_shape=(
                    a.shape[0] % a._top_left_shape[0],
                    a.shape[0] % a._top_left_shape[0]),
                    reg_shape=(a.shape[0] % a._reg_shape[0],
                               a.shape[0] % a._reg_shape[0]),
                    shape=(a.shape[0] % a._reg_shape[0],
                           a.shape[0] % a._reg_shape[0]),
                    sparse=False)
            else:
                q_ds_array = Array(q_blocks,
                                   top_left_shape=(a._top_left_shape[0],
                                                   a._top_left_shape[0]),
                                   reg_shape=(a._reg_shape[0],
                                              a._reg_shape[0]),
                                   shape=(a._reg_shape[0], a._reg_shape[0]),
                                   sparse=False)
            qs.append(q_ds_array)
            rs.append(r_blocks)
        depth = 1
        last_block_irregular = False
        while len(rs) > 1:
            reduction_number = math.ceil(len(rs) / n_reduction)
            qsaux = qs
            rsaux = rs
            shape_to_use = qsaux[0].shape[1]
            irregular_shape_to_use = qsaux[-1].shape[1] % a._reg_shape[0]
            rs = []
            qs = []
            for i in range(reduction_number):
                if len(rsaux[int(i * n_reduction):
                             int((i + 1) * n_reduction)]) > 1 and (
                             int((i + 1) * n_reduction) != len(rsaux) or
                             a.shape[0] % a._reg_shape[0] == 0):
                    auxiliar_rs = len(rsaux[int(i * n_reduction)])
                    auxiliar_rs_2 = len(rsaux[int(i * n_reduction) + 1])
                    number_blocks = auxiliar_rs + auxiliar_rs_2
                    q_blocks = [[object() for _ in range(number_blocks)]
                                for _ in range(auxiliar_rs)]
                    q_blocks_2 = [[object() for _ in range(number_blocks)]
                                  for _ in range(auxiliar_rs_2)]
                    r_blocks = [[object() for _ in range(len(block))]
                                for _ in range(number_blocks)]
                    if (irregular_shape_to_use != 0
                            and i == (reduction_number - 1)):
                        last_block_irregular = True
                    _compute_reduction_qr(rsaux[int(i * n_reduction):
                                                int((i + 1) * n_reduction)],
                                          q_blocks,
                                          q_blocks_2,
                                          r_blocks,
                                          last_block_irregular,
                                          irregular_shape_to_use,
                                          "complete")
                    small_q = Array(q_blocks, top_left_shape=(a._reg_shape[0],
                                                              a._reg_shape[0]),
                                    reg_shape=(a._reg_shape[0],
                                               a._reg_shape[0]),
                                    shape=(a._reg_shape[0] * auxiliar_rs,
                                           a._reg_shape[0] * number_blocks),
                                    sparse=False)
                    if (irregular_shape_to_use != 0
                            and i == (reduction_number - 1)):
                        small_q_2 = Array(q_blocks,
                                          top_left_shape=(a._reg_shape[0],
                                                          a._reg_shape[0]),
                                          reg_shape=(a._reg_shape[0],
                                                     a._reg_shape[0]),
                                          shape=(irregular_shape_to_use,
                                                 irregular_shape_to_use),
                                          sparse=False)
                    else:
                        small_q_2 = Array(q_blocks_2,
                                          top_left_shape=(a._reg_shape[0],
                                                          a._reg_shape[0]),
                                          reg_shape=(a._reg_shape[0],
                                                     a._reg_shape[0]),
                                          shape=(a._reg_shape[0] *
                                                 auxiliar_rs_2,
                                                 a._reg_shape[0] *
                                                 number_blocks),
                                          sparse=False)
                elif (len(rsaux[int(i * n_reduction):
                                int((i + 1) * n_reduction)]) > 1
                      and int((i + 1) * n_reduction) == len(rsaux)):
                    auxiliar_rs = len(rsaux[int(i * n_reduction)])
                    auxiliar_rs_2 = len(rsaux[-1])
                    number_blocks = auxiliar_rs + auxiliar_rs_2
                    q_blocks = [[object() for _ in range(number_blocks)]
                                for _ in range(auxiliar_rs)]
                    q_blocks_2 = [[object() for _ in range(number_blocks)]
                                  for _ in range(auxiliar_rs_2)]
                    r_blocks = [[object() for _ in range(len(block))]
                                for _ in range(number_blocks)]
                    true_columns = a._reg_shape[0] * number_blocks
                    if (irregular_shape_to_use != 0
                            and i == (reduction_number - 1)):
                        last_block_irregular = True
                        true_columns -= a._reg_shape[0]
                        true_columns += irregular_shape_to_use
                    _compute_reduction_qr(rsaux[int(i * n_reduction):
                                                int((i + 1) * n_reduction)],
                                          q_blocks,
                                          q_blocks_2,
                                          r_blocks,
                                          last_block_irregular,
                                          irregular_shape_to_use,
                                          "complete")
                    small_q = Array(q_blocks, top_left_shape=(a._reg_shape[0],
                                                              a._reg_shape[0]),
                                    reg_shape=(a._reg_shape[0],
                                               a._reg_shape[0]),
                                    shape=(shape_to_use, true_columns),
                                    sparse=False)
                    if (irregular_shape_to_use != 0
                            and i == (reduction_number - 1)):
                        small_q_2 = Array(q_blocks_2,
                                          top_left_shape=(
                                              a._reg_shape[0] if
                                              auxiliar_rs_2 > 1
                                              else irregular_shape_to_use,
                                              a._reg_shape[0] if
                                              number_blocks > 1
                                              else irregular_shape_to_use),
                                          reg_shape=(a._reg_shape[0] if
                                                     auxiliar_rs_2 > 1
                                                     else
                                                     irregular_shape_to_use,
                                                     a._reg_shape[0] if
                                                     number_blocks > 1
                                                     else
                                                     irregular_shape_to_use),
                                          shape=(
                                              a._reg_shape[0] *
                                              (auxiliar_rs_2 - 1) +
                                              irregular_shape_to_use,
                                              true_columns),
                                          sparse=False)
                    else:
                        small_q_2 = Array(q_blocks_2,
                                          top_left_shape=(a._reg_shape[0],
                                                          a._reg_shape[0]),
                                          reg_shape=(a._reg_shape[0],
                                                     a._reg_shape[0]),
                                          shape=(a._reg_shape[0] *
                                                 auxiliar_rs_2, true_columns),
                                          sparse=False)
                else:
                    auxiliar_rs = len(rsaux[int(i * n_reduction)])
                    number_blocks = auxiliar_rs
                    if auxiliar_rs == 1 and a.shape[0] % a._reg_shape[0] != 0:
                        shape_to_use = a.shape[0] % a._reg_shape[0]
                    else:
                        shape_to_use = a._reg_shape[0]
                    q_blocks = [[object() for _ in range(number_blocks)]
                                for _ in range(auxiliar_rs)]
                    q_blocks_2 = [[object()]]
                    r_blocks = [[object() for _ in range(len(block))]
                                for _ in range(number_blocks)]
                    if (irregular_shape_to_use != 0
                            and i == (reduction_number - 1)):
                        last_block_irregular = True
                    _compute_reduction_qr(rsaux[int(i * n_reduction):
                                                int((i + 1) * n_reduction)],
                                          q_blocks,
                                          q_blocks_2,
                                          r_blocks,
                                          last_block_irregular,
                                          irregular_shape_to_use,
                                          "complete")
                    small_q = Array(q_blocks, top_left_shape=(shape_to_use,
                                                              shape_to_use),
                                    reg_shape=(shape_to_use, shape_to_use),
                                    shape=(shape_to_use * auxiliar_rs,
                                           shape_to_use * number_blocks),
                                    sparse=False)
                q_1 = matmul(qsaux.pop(0), small_q)
                if len(qsaux) > 0:
                    q_2 = matmul(qsaux.pop(0), small_q_2)
                    qs.append(concat_rows(q_1, q_2))
                else:
                    q_1._shape = (q_1.shape[0], q_1.shape[1])
                    qs.append(q_1)
                rs.append(r_blocks)
                last_block_irregular = False
            depth = depth + 1
        q = qs[0]
        r = Array(r_blocks, top_left_shape=a._reg_shape,
                  reg_shape=a._reg_shape,
                  shape=a.shape,
                  sparse=False)
        return q, r

    elif mode == "complete_inverse":
        if _is_not_power_of_two(a._n_blocks[0]):
            raise ValueError("This mode only works if the number of "
                             "blocks is a power 2")

        for i, block in enumerate(a._blocks):
            q_blocks = [[object()]]
            r_blocks = [[object() for _ in range(len(block))]]
            _compute_qr([block], q_blocks, r_blocks, "complete")
            if i == len(a._blocks) - 1 and a._shape[0] % a._reg_shape[0] != 0:
                q_ds_array = Array(q_blocks, top_left_shape=(
                    a.shape[0] % a._top_left_shape[0],
                    a.shape[0] % a._top_left_shape[0]),
                                   reg_shape=(
                                       a.shape[0] % a._reg_shape[0],
                                       a.shape[0] % a._reg_shape[0]),
                                   shape=(a.shape[0] % a._reg_shape[0],
                                          a.shape[0] % a._reg_shape[0]),
                                   sparse=False)
            else:
                q_ds_array = Array(q_blocks, top_left_shape=(
                    a._top_left_shape[0],
                    a._top_left_shape[0]),
                                   reg_shape=(a._reg_shape[0],
                                              a._reg_shape[0]),
                                   shape=(a._reg_shape[0],
                                          a._reg_shape[0]),
                                   sparse=False)
            qs.append(q_ds_array)
            rs.append(r_blocks)
        shape_to_use = qs[0].shape[1]
        irregular_shape_to_use = qs[-1].shape[1]
        while len(rs) > 1:
            reduction_number = math.ceil(len(rs) / n_reduction)
            rsaux = rs
            rs = []
            for i in range(reduction_number):
                if len(rsaux[int(i * n_reduction):
                             int((i + 1) * n_reduction)]) > 1 and \
                        (int((i + 1) * n_reduction) != len(rsaux) or
                         a.shape[0] % a._reg_shape[0] == 0):
                    auxiliar_rs = len(rsaux[int(i * n_reduction)])
                    auxiliar_rs_2 = len(rsaux[int(i * n_reduction) + 1])
                    number_blocks = auxiliar_rs + auxiliar_rs_2
                    q_blocks = [[object() for _ in range(number_blocks)]
                                for _ in range(number_blocks)]
                    r_blocks = [[object() for _ in range(len(block))]
                                for _ in range(number_blocks)]
                    _compute_reduction_qr_one_q(rsaux[int(i * n_reduction):
                                                int((i + 1) * n_reduction)],
                                                q_blocks,
                                                r_blocks,
                                                "complete")
                    small_q = Array(q_blocks, top_left_shape=(
                        a._reg_shape[0],
                        a._reg_shape[0]),
                                    reg_shape=(a._reg_shape[0],
                                               a._reg_shape[0]),
                                    shape=(a._reg_shape[0] * number_blocks,
                                           a._reg_shape[0] * number_blocks),
                                    sparse=False)
                elif len(rsaux[int(i * n_reduction):
                               int((i + 1) * n_reduction)]) > 1 and \
                        int((i + 1) * n_reduction) == len(rsaux):
                    auxiliar_rs = len(rsaux[int(i * n_reduction)])
                    auxiliar_rs_2 = len(rsaux[-1])
                    number_blocks = auxiliar_rs + auxiliar_rs_2
                    q_blocks = [[object() for _ in range(number_blocks)]
                                for _ in range(number_blocks)]
                    r_blocks = [[object() for _ in range(len(block))]
                                for _ in range(number_blocks)]
                    _compute_reduction_qr_one_q(rsaux[int(i * n_reduction):
                                                int((i + 1) * n_reduction)],
                                                q_blocks,
                                                r_blocks,
                                                "complete")
                    small_q = Array(q_blocks, top_left_shape=(
                        a._reg_shape[0],
                        a._reg_shape[0]),
                                    reg_shape=(a._reg_shape[0],
                                               a._reg_shape[0]),
                                    shape=(
                                        shape_to_use * (number_blocks - 1) +
                                        irregular_shape_to_use, shape_to_use *
                                        (number_blocks - 1) +
                                        irregular_shape_to_use),
                                    sparse=False)
                qs.append(small_q)
                rs.append(r_blocks)

        if indexes is not None:
            blocks_matrix_indices = [[object() for _ in range(
                                      math.ceil(len(indexes) /
                                                a._reg_shape[0]))]
                                     for _ in range(a._n_blocks[0])]
            _construct_identity(indexes,
                                blocks_matrix_indices,
                                a.shape[0], a._reg_shape)
            matrix_indices = Array(blocks_matrix_indices,
                                   top_left_shape=(a._reg_shape[0],
                                                   len(indexes) if
                                                   len(indexes) <
                                                   a._reg_shape[0] else
                                                   a._reg_shape[0]),
                                   reg_shape=(a._reg_shape[0],
                                              len(indexes) if
                                              len(indexes) <
                                              a._reg_shape[0]
                                              else a._reg_shape[0]),
                                   shape=(a.shape[0], len(indexes)),
                                   sparse=False)
            q = _construct_q_from_the_end(qs, n_reduction, a._reg_shape[0],
                                          a.shape,
                                          indexes=matrix_indices,
                                          complete=True)
        else:
            q = _construct_q_from_the_end(qs, n_reduction, a._reg_shape[0],
                                          a.shape,
                                          complete=True)
        r = Array(rs[-1], top_left_shape=a._reg_shape,
                  reg_shape=a._reg_shape,
                  shape=a.shape,
                  sparse=False)
        return q, r
    elif mode == "reduced":
        if a._reg_shape[0] < a.shape[1]:
            raise ValueError("The number of rows in each block needs to be "
                             "greater than the total number of columns")
        true_shape = (a._reg_shape[0], a._shape[1])
        for i, block in enumerate(a._blocks):
            q_blocks = [[object() for _ in range(len(block))]]
            r_blocks = [[object() for _ in range(len(block))] for _ in
                        range(math.ceil(a.shape[1] / a._reg_shape[1]))]
            _compute_qr([block], q_blocks, r_blocks, "reduced")
            if i == len(a._blocks) - 1 and a._shape[0] % a._reg_shape[0] != 0:
                q_ds_array = Array(q_blocks,
                                   top_left_shape=(a._shape[0] %
                                                   a._top_left_shape[0],
                                                   a._reg_shape[1]),
                                   reg_shape=(a._shape[0] %
                                              a._reg_shape[0],
                                              a._reg_shape[1]),
                                   shape=(a._shape[0] %
                                          a._reg_shape[0],
                                          a._reg_shape[1]),
                                   sparse=False)
            else:
                reg_shape_to_assign = a._reg_shape
                q_ds_array = Array(q_blocks,
                                   top_left_shape=reg_shape_to_assign,
                                   reg_shape=reg_shape_to_assign,
                                   shape=true_shape,
                                   sparse=False)
            qs.append(q_ds_array)
            rs.append(r_blocks)
        last_block_irregular = False
        shape_last_block = 0
        while len(rs) > 1:
            reduction_number = math.ceil(len(rs) / n_reduction)
            qsaux = qs
            rsaux = rs
            rs = []
            qs = []
            for i in range(reduction_number):
                if (i < (reduction_number - 1)
                        or len(rsaux) % n_reduction == 0):
                    if a.shape[1] < a._reg_shape[0]:
                        q_blocks = [[object() for _ in range(len(block))]
                                    for _ in range(math.ceil(a.shape[1] /
                                                             a._reg_shape[1]))]
                        q_blocks_2 = [[object() for _ in range(len(block))]
                                      for _ in range(
                                math.ceil(a.shape[1] / a._reg_shape[1]))]
                        if a.shape[1] % a._reg_shape[1] != 0:
                            last_block_irregular = True
                            shape_last_block = a.shape[1] % a._reg_shape[1]
                    else:
                        q_blocks = [[object() for _ in range(len(block))]
                                    for _ in range(math.ceil
                                                   (a._reg_shape[0] /
                                                    a._reg_shape[1]))]
                        q_blocks_2 = [[object() for _ in range(len(block))]
                                      for _ in range(
                                math.ceil(a._reg_shape[0] / a._reg_shape[1]))]
                        if a.shape[1] % a._reg_shape[1] != 0:
                            last_block_irregular = True
                            shape_last_block = (a._reg_shape[0] %
                                                a._reg_shape[1])
                else:
                    q_blocks = [[object() for _ in range(len(block))]
                                for _ in range(math.ceil(
                                                math.ceil(
                                                   a.shape[1] /
                                                   a._reg_shape[1]) / 2))]
                    q_blocks_2 = [[object() for _ in range(len(block))]
                                  for _ in range(math.floor(
                                                 math.ceil(
                                                     a.shape[1] /
                                                     a._reg_shape[1]) / 2))
                                  ]
                r_blocks = [[object() for _ in range(len(block))]
                            for _ in range(math.ceil(
                                           a.shape[1] / a._reg_shape[1]))]
                _compute_reduction_qr(rsaux[int(i * n_reduction):
                                            int((i + 1) *
                                                n_reduction)],
                                      q_blocks,
                                      q_blocks_2,
                                      r_blocks,
                                      last_block_irregular,
                                      shape_last_block,
                                      "reduced")
                if i < (reduction_number - 1) or len(rsaux) % n_reduction == 0:
                    small_q = Array(q_blocks,
                                    top_left_shape=(a._reg_shape[1],
                                                    a._reg_shape[1]),
                                    reg_shape=(a._reg_shape[1],
                                               a._reg_shape[1]),
                                    shape=(a.shape[1] if
                                           a.shape[1] < a._reg_shape[0]
                                           else a._reg_shape[0], a.shape[1]),
                                    sparse=False)
                    small_q_2 = Array(q_blocks_2,
                                      top_left_shape=(a._reg_shape[1],
                                                      a._reg_shape[1]),
                                      reg_shape=(a._reg_shape[1],
                                                 a._reg_shape[1]),
                                      shape=(
                                          a.shape[1] if
                                          a.shape[1] < a._reg_shape[0]
                                          else a._reg_shape[0], a.shape[1]),
                                      sparse=False)
                else:
                    small_q = Array(q_blocks,
                                    top_left_shape=(a._reg_shape[1],
                                                    a._reg_shape[1]),
                                    reg_shape=(a._reg_shape[1],
                                               a._reg_shape[1]),
                                    shape=(a.shape[1] if
                                           a.shape[1] < a._reg_shape[0]
                                           else a._reg_shape[0],
                                           a.shape[1]), sparse=False)

                q_1 = matmul(qsaux.pop(0), small_q)
                if len(qsaux) > 0:
                    q_2 = matmul(qsaux.pop(0), small_q_2)
                    qs.append(concat_rows(q_1, q_2))
                else:
                    qs.append(q_1)
                rs.append(r_blocks)
                last_block_irregular = False
        r = Array(rs[0],
                  top_left_shape=(a._reg_shape[1], a._reg_shape[1]),
                  reg_shape=(a._reg_shape[1], a._reg_shape[1]),
                  shape=(a.shape[1], a.shape[1]), sparse=False)
        return qs[-1], r

    elif mode == "reduced_inverse":
        if _is_not_power_of_two(a._n_blocks[0]):
            raise ValueError("This mode only works if the number of "
                             "blocks is a power of 2")
        if a._reg_shape[0] < a.shape[1]:
            raise ValueError("The number of rows in each block needs to be "
                             "greater than the total number of columns")
        if (a._shape[0] % a._reg_shape[0]) != 0 and \
                (a._shape[0] % a._reg_shape[0]) < a.shape[1]:
            raise ValueError("The number of rows in the last block is "
                             "smaller than the total number of columns")
        for i, block in enumerate(a._blocks):
            q_blocks = [[object() for _ in range(len(block))]]
            r_blocks = [[object() for _ in range(len(block))] for _ in
                        range(math.ceil(a.shape[1] / a._reg_shape[1]))]
            if i == len(a._blocks) - 1 and a._shape[0] % a._reg_shape[0] != 0:
                _compute_qr([block], q_blocks, r_blocks, "reduced")
                if a._shape[0] % a._reg_shape[0] < a._shape[1]:
                    col_shape = a._shape[0] % a._reg_shape[0]
                else:
                    col_shape = a._reg_shape[1]
                q_ds_array = Array(q_blocks,
                                   top_left_shape=(
                                       a._shape[0] % a._top_left_shape[0],
                                       col_shape),
                                   reg_shape=(a._shape[0] % a._reg_shape[0],
                                              col_shape),
                                   shape=(a._shape[0] % a._reg_shape[0],
                                          a.shape[1]),
                                   sparse=False)
            else:
                reg_shape_to_assign = a._reg_shape
                _compute_qr([block], q_blocks, r_blocks, "reduced")
                q_ds_array = Array(q_blocks,
                                   top_left_shape=reg_shape_to_assign,
                                   reg_shape=reg_shape_to_assign,
                                   shape=(a._reg_shape[0], a._shape[1]),
                                   sparse=False)
            qs.append(q_ds_array)
            rs.append(r_blocks)

        while len(rs) > 1:
            reduction_number = math.ceil(len(rs) / n_reduction)
            rsaux = rs
            rs = []
            for i in range(reduction_number):
                if i < (reduction_number - 1) or len(rsaux) % n_reduction == 0:
                    if a.shape[1] < a._reg_shape[0]:
                        q_blocks = [[object() for _ in range(len(block))]
                                    for _ in range(
                                     math.ceil(a.shape[1] * 2 /
                                               a._reg_shape[1]))]
                    else:
                        q_blocks = [[object() for _ in range(len(block))]
                                    for _ in range(
                                     math.ceil(a._reg_shape[0] * 2 /
                                               a._reg_shape[1]))]
                else:
                    if a.shape[1] < a._reg_shape[0]:
                        q_blocks = [[object() for _ in range(len(block))]
                                    for _ in range(
                                     math.ceil(a.shape[1] /
                                               a._reg_shape[1]))]
                    else:
                        q_blocks = [[object() for _ in range(len(block))]
                                    for _ in range(
                                     math.ceil(a._reg_shape[0] /
                                               a._reg_shape[1]))]
                r_blocks = [[object() for _ in range(len(block))]
                            for _ in range(
                             math.ceil(a.shape[1] /
                                       a._reg_shape[1]))]
                _compute_reduction_qr_one_q(rsaux[int(i * n_reduction):
                                            int((i + 1) *
                                                n_reduction)],
                                            q_blocks,
                                            r_blocks,
                                            "reduced")
                if (i < (reduction_number - 1)
                        or len(rsaux) % n_reduction == 0):
                    small_q = Array(q_blocks,
                                    top_left_shape=(a._reg_shape[1],
                                                    a._reg_shape[1]),
                                    reg_shape=(a._reg_shape[1],
                                               a._reg_shape[1]),
                                    shape=(a.shape[1] * 2, a.shape[1]),
                                    sparse=False)
                else:
                    small_q = Array(q_blocks,
                                    top_left_shape=(a._reg_shape[1],
                                                    a._reg_shape[1]),
                                    reg_shape=(a._reg_shape[1],
                                               a._reg_shape[1]),
                                    shape=(a.shape[1],
                                           a.shape[1]), sparse=False)
                qs.append(small_q)
                rs.append(r_blocks)
        if indexes is not None:
            blocks_matrix_indices = [[object() for _ in range(
                math.ceil(len(indexes) /
                          a._reg_shape[1]))]
                                     for _ in range(a._n_blocks[0])]
            _construct_identity(indexes,
                                blocks_matrix_indices,
                                a.shape[0], a._reg_shape, complete=False)
            matrix_indices = Array(blocks_matrix_indices,
                                   top_left_shape=(a._reg_shape[1],
                                                   len(indexes) if
                                                   len(indexes) <
                                                   a._reg_shape[1] else
                                                   a._reg_shape[1]),
                                   reg_shape=(a._reg_shape[1],
                                              len(indexes) if
                                              len(indexes) <
                                              a._reg_shape[1] else
                                              a._reg_shape[1]),
                                   shape=(a.shape[0], len(indexes)),
                                   sparse=False)
            q = _construct_q_from_the_end(qs, n_reduction,
                                          a._reg_shape[0], a.shape,
                                          indexes=matrix_indices,
                                          complete=False)
        else:
            q = _construct_q_from_the_end(qs, n_reduction,
                                          a._reg_shape[0], a.shape,
                                          complete=False)

        r = Array(rs[-1], top_left_shape=(a._reg_shape[1], a._reg_shape[1]),
                  reg_shape=(a._reg_shape[1], a._reg_shape[1]),
                  shape=(a.shape[1], a.shape[1]), sparse=False)
        return q, r
    elif mode == "r_complete":
        for block in a._blocks:
            q_blocks = [[object()]]
            r_blocks = [[object() for _ in range(len(block))]]
            _compute_qr([block], q_blocks, r_blocks, "complete")
            rs.append(r_blocks)

        while len(rs) > 1:
            reduction_number = math.ceil(len(rs) / n_reduction)
            rsaux = rs
            rs = []
            for i in range(reduction_number):
                if len(rsaux[int(i * n_reduction):
                             int((i + 1) * n_reduction)]) > 1 and \
                        (int((i + 1) * n_reduction) != len(rsaux) or
                         a.shape[0] % a._reg_shape[0] == 0):
                    auxiliar_rs = len(rsaux[int(i * n_reduction)])
                    auxiliar_rs_2 = len(rsaux[int(i * n_reduction) + 1])
                    number_blocks = auxiliar_rs + auxiliar_rs_2
                elif len(rsaux[int(i * n_reduction):
                               int((i + 1) * n_reduction)]) > 1 and \
                        int((i + 1) * n_reduction) == len(rsaux):
                    auxiliar_rs = len(rsaux[int(i * n_reduction)])
                    auxiliar_rs_2 = len(rsaux[int(i * n_reduction) + 1])
                    number_blocks = auxiliar_rs + auxiliar_rs_2
                else:
                    number_blocks = len(rsaux[-1])

                r_blocks = [[object() for _ in range(len(block))]
                            for _ in range(number_blocks)]
                _compute_reduction_qr_only_r(rsaux[int(i * n_reduction):
                                                   int((i + 1) * n_reduction)],
                                             r_blocks,
                                             "complete")
                rs.append(r_blocks)
        r = Array(rs[-1], top_left_shape=a._reg_shape,
                  reg_shape=a._reg_shape,
                  shape=a.shape,
                  sparse=False)
        return r
    elif mode == "r_reduced":
        for block in a._blocks:
            q_blocks = [[object() for _ in range(len(block))]]
            r_blocks = [[object() for _ in range(len(block))] for _ in
                        range(math.ceil(a.shape[1] / a._reg_shape[1]))]
            _compute_qr([block], q_blocks, r_blocks, "reduced")
            rs.append(r_blocks)

        while len(rs) > 1:
            reduction_number = math.ceil(len(rs) / n_reduction)
            rsaux = rs
            rs = []
            for i in range(reduction_number):
                r_blocks = [[object() for _ in range(len(block))] for _ in
                            range(math.ceil(a.shape[1] / a._reg_shape[1]))]
                _compute_reduction_qr_only_r(rsaux[int(i * n_reduction):
                                                   int((i + 1) * n_reduction)],
                                             r_blocks,
                                             "reduced")
                rs.append(r_blocks)
        r = Array(rs[-1],
                  top_left_shape=(a._reg_shape[1], a._reg_shape[1]),
                  reg_shape=(a._reg_shape[1], a._reg_shape[1]),
                  shape=(a.shape[1], a.shape[1]), sparse=False)
        return r


def _construct_q_from_the_end(qs, n_reduction, reg_shape_0,
                              shape, indexes=None,
                              complete=False):
    if indexes is not None:
        q = matmul(qs[-1], indexes)
        '''Array(indexes._blocks,
                                 top_left_shape=indexes._top_left_shape,
                                 reg_shape=(qs[-1]._reg_shape[0],
                                            indexes.shape[1]),
                                 shape=indexes.shape,
                                 sparse=False))'''
    else:
        q = qs[-1]
    qs.pop()
    depth = 1
    idx = n_reduction ** depth
    qs_aux = []
    if complete:
        last_block_shape = shape[0] - (math.ceil(shape[0] / reg_shape_0)) \
            * reg_shape_0 / 2
        reg_shape = shape[0] - last_block_shape
    for q_aux in reversed(qs):
        if complete:
            q_1 = q[int(reg_shape * (idx - 1)):int(reg_shape * idx)]
        else:
            q_1 = q[q_aux.shape[1] * (idx - 1):q_aux.shape[1] * (idx)]
        if q_1._top_left_shape != q_1._reg_shape:
            q_1 = small_rechunk(q_1)
        q_aux = small_rechunk(q_aux)
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
            q = q[:q.shape[0]]
            qs_aux.pop(-1)
            qs_aux.pop(-1)
            for q_auxiliar in reversed(qs_aux):
                q = concat_rows(q, q_auxiliar)
            qs_aux = []
        qs.pop(-1)
    return q


def _is_not_power_of_two(number):
    if number < 1:
        return False
    return 0 != (number & (number - 1))


def small_rechunk(rechunk_array):
    number_blocks_inside = math.ceil(
        rechunk_array.shape[0] / rechunk_array._reg_shape[0])
    number_blocks_outside = math.ceil(
        rechunk_array.shape[1] / rechunk_array._reg_shape[1])
    out_blocks = [[object() for _ in range(number_blocks_outside)]
                  for _ in range(number_blocks_inside)]
    if rechunk_array._reg_shape[0] > rechunk_array.shape[0] and \
            rechunk_array._reg_shape[1] > rechunk_array.shape[1]:
        reg_shape = rechunk_array.shape
    elif rechunk_array._reg_shape[0] > rechunk_array.shape[0]:
        reg_shape = (rechunk_array.shape[0], rechunk_array._reg_shape[1])
    else:
        reg_shape = (rechunk_array._reg_shape[0], rechunk_array._reg_shape[1])
    j = 0
    top_left_shape_data_to_advance = \
        rechunk_array._reg_shape[0] - \
        rechunk_array._top_left_shape[0]
    for j in range(len(out_blocks)-1):
        for i in range(len(out_blocks[j])):
            out_blocks[j][i] = _assign_corresponding_elements_block(
                rechunk_array._blocks[j][i],
                rechunk_array._blocks[j+1][i],
                rechunk_array._reg_shape[0],
                top_left_shape_data_to_advance if j != 0 else 0)
    if len(out_blocks) == 1:
        j = 0
    else:
        j = j + 1
    if rechunk_array._n_blocks[0] > number_blocks_outside:
        for i in range(len(out_blocks[j])):
            if j < (rechunk_array._n_blocks[0] - 1):
                out_blocks[j][i] = _assign_corresponding_elements_block(
                    rechunk_array._blocks[j][i], rechunk_array._blocks[j+1][i],
                    rechunk_array.shape[0] % rechunk_array._reg_shape[0],
                    top_left_shape_data_to_advance)
            else:
                out_blocks[j][i] = _assign_elements_last_block(
                    rechunk_array._blocks[j][i],
                    rechunk_array.shape[0] % rechunk_array._reg_shape[0])
    else:
        for i in range(len(out_blocks[j])):
            out_blocks[j][i] = _assign_elements_last_block(
                rechunk_array._blocks[j][i],
                rechunk_array.shape[0] % rechunk_array._reg_shape[0])
    return Array(out_blocks, top_left_shape=reg_shape,
                 reg_shape=reg_shape,
                 shape=rechunk_array.shape,
                 sparse=False)


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
@task(returns=np.array)
def _multiply(q, to_multiply):
    return np.dot(q, to_multiply)


@constraint(computing_units="${ComputingUnits}")
@task(indexes={Type: COLLECTION_IN, Depth: 1},
      blocks_out={Type: COLLECTION_OUT, Depth: 2})
def _construct_identity(indexes, blocks_out, shape, reg_shape, complete=True):
    identity = np.eye(shape)
    identity = identity[:, indexes]
    if complete:
        for i in range(len(blocks_out)):
            for j in range(len(blocks_out[i])):
                blocks_out[i][j] = identity[i * reg_shape[0]:
                                            (i + 1) * reg_shape[0],
                                            j * reg_shape[0]:
                                            (j + 1) * reg_shape[0]]
    else:
        for i in range(len(blocks_out)):
            for j in range(len(blocks_out[i])):
                blocks_out[i][j] = identity[i*reg_shape[1]:
                                            (i+1)*reg_shape[1],
                                            j*reg_shape[1]:
                                            (j+1)*reg_shape[1]]


@constraint(computing_units="${ComputingUnits}")
@task(rs={Type: COLLECTION_IN, Depth: 3},
      q_blocks={Type: COLLECTION_OUT, Depth: 2},
      q_blocks_2={Type: COLLECTION_OUT, Depth: 2},
      r_blocks={Type: COLLECTION_OUT, Depth: 2})
def _compute_reduction_qr(rs, q_blocks, q_blocks_2,
                          r_blocks, last_block_irregular,
                          size_last_block, mode):
    block_shape = rs[0][0][0].shape
    if mode == "complete":
        r_first = np.hstack(np.hstack(rs[0]))
        if len(rs) > 1:
            r_second = np.hstack(np.hstack(rs[1]))
            q, r = qr(np.vstack((r_first, r_second)), mode="complete")
        else:
            q, r = qr(r_first, mode="complete")
        for i in range(len(q_blocks)):
            for j in range(len(q_blocks[i])):
                q_blocks[i][j] = q[block_shape[0] * i:
                                   block_shape[0] * (i + 1),
                                   block_shape[0] * j:
                                   block_shape[0] * (j + 1)]
        i += 1
        for k in range(len(q_blocks_2)):
            for j in range(len(q_blocks_2[0])):
                if last_block_irregular and k == len(q_blocks_2) - 1:
                    q_blocks_2[k][j] = q[block_shape[0] * (i + k):
                                         block_shape[0] * (i + k) +
                                         size_last_block,
                                         block_shape[0] * j:
                                         block_shape[0] * (j + 1)]
                else:
                    q_blocks_2[k][j] = q[block_shape[0] * (i + k):
                                         block_shape[0] * (i + k + 1),
                                         block_shape[0] * j:
                                         block_shape[0] * (j + 1)]
        for i in range(len(r_blocks)):
            for j in range(len(r_blocks[i])):
                r_blocks[i][j] = r[block_shape[0] * i:
                                   block_shape[0] * (i + 1),
                                   block_shape[1] * j:
                                   block_shape[1] * (j + 1)]
    elif mode == "reduced":
        if rs[0][0][-1].shape[1] != block_shape[1]:
            intermediate_rs = []
            for aux_rs in rs[0]:
                intermediate_rs.append(np.hstack(aux_rs))
            r_first = np.vstack(intermediate_rs)
        else:
            r_first = np.hstack(np.hstack(rs[0]))
        if len(rs) > 1:
            if rs[0][0][-1].shape[1] != block_shape[1]:
                intermediate_rs = []
                for aux_rs in rs[1]:
                    intermediate_rs.append(np.hstack(aux_rs))
                r_second = np.vstack(intermediate_rs)
            else:
                r_second = np.hstack(np.hstack(rs[1]))
        if len(rs) > 1:
            q, r = qr(np.vstack((r_first, r_second)))
        else:
            q, r = qr(r_first)
        block_shape_to_use = block_shape[0] if (
                block_shape[1] > block_shape[0]) else (
            block_shape)[1]
        for i in range(len(q_blocks)):
            for j in range(len(q_blocks[0])):
                if i == len(q_blocks) - 1 and last_block_irregular:
                    q_blocks[i][j] = q[block_shape_to_use * i:
                                       size_last_block +
                                       block_shape_to_use * i,
                                       block_shape_to_use * j:
                                       block_shape_to_use * (j + 1)]
                else:
                    q_blocks[i][j] = q[block_shape_to_use * i:
                                       block_shape_to_use * (i + 1),
                                       block_shape_to_use * j:
                                       block_shape_to_use * (j + 1)]
        i += 1
        for k in range(len(q_blocks_2)):
            for j in range(len(q_blocks_2[0])):
                if k == len(q_blocks_2) - 1 and last_block_irregular:
                    q_blocks_2[k][j] = q[block_shape_to_use * (i + k - 1) +
                                         size_last_block:
                                         block_shape_to_use * (i + k) +
                                         size_last_block * 2,
                                         block_shape_to_use * j:
                                         block_shape_to_use * (j + 1)]
                elif last_block_irregular:
                    q_blocks_2[k][j] = q[block_shape_to_use * (i + k - 1) +
                                         size_last_block:
                                         block_shape_to_use * (i + k) +
                                         size_last_block,
                                         block_shape_to_use * j:
                                         block_shape_to_use * (j + 1)]
                else:
                    q_blocks_2[k][j] = q[block_shape[0] * (i + k):
                                         block_shape[0] * (i + k + 1),
                                         block_shape[0] * j:
                                         block_shape[0] * (j + 1)]
        for i in range(len(r_blocks)):
            for j in range(len(r_blocks[0])):
                r_blocks[i][j] = r[block_shape_to_use * i:
                                   block_shape_to_use * (i + 1),
                                   block_shape_to_use * j:
                                   block_shape_to_use * (j + 1)]


@constraint(computing_units="${ComputingUnits}")
@task(rs={Type: COLLECTION_IN, Depth: 3},
      q_blocks={Type: COLLECTION_OUT, Depth: 2},
      r_blocks={Type: COLLECTION_OUT, Depth: 2})
def _compute_reduction_qr_one_q(rs, q_blocks, r_blocks, mode):
    block_shape = rs[0][0][0].shape
    if mode == "complete":
        r_first = np.hstack(np.hstack(rs[0]))
        if len(rs) > 1:
            r_second = np.hstack(np.hstack(rs[1]))
            q, r = qr(np.vstack((r_first, r_second)), mode="complete")
        else:
            q, r = qr(r_first, mode="complete")
        for i in range(len(q_blocks)):
            for j in range(len(q_blocks[i])):
                q_blocks[i][j] = q[block_shape[0] * i:block_shape[0] * (i + 1),
                                   block_shape[0] * j:block_shape[0] * (j + 1)]
        for i in range(len(r_blocks)):
            for j in range(len(r_blocks[i])):
                r_blocks[i][j] = r[block_shape[0] * i:block_shape[0] * (i + 1),
                                   block_shape[1] * j:block_shape[1] * (j + 1)]
    elif mode == "reduced":
        r_first = np.hstack(np.hstack(rs[0]))
        if len(rs) > 1:
            r_second = np.hstack(np.hstack(rs[1]))
        if len(rs) > 1:
            q, r = qr(np.vstack((r_first, r_second)))
        else:
            q, r = qr(r_first)
        block_shape_to_use = block_shape[0] if \
            block_shape[1] > block_shape[0] else block_shape[1]
        for j in range(len(q_blocks)):
            for i in range(len(q_blocks[0])):
                q_blocks[j][i] = q[block_shape[0] * j:
                                   block_shape[0] * (j + 1),
                                   block_shape_to_use * i:
                                   block_shape_to_use * (i + 1)]
        for j in range(len(r_blocks)):
            for i in range(len(r_blocks[0])):
                r_blocks[j][i] = r[block_shape_to_use * j:
                                   block_shape_to_use * (j + 1),
                                   block_shape[1] * i:
                                   block_shape[1] * (i + 1)]


@constraint(computing_units="${ComputingUnits}")
@task(rs={Type: COLLECTION_IN, Depth: 3},
      r_blocks={Type: COLLECTION_OUT, Depth: 2})
def _compute_reduction_qr_only_r(rs, r_blocks, mode):
    block_shape = rs[0][0][0].shape
    if mode == "complete":
        r_first = np.hstack(np.hstack(rs[0]))
        if len(rs) > 1:
            r_second = np.hstack(np.hstack(rs[1]))
            q, r = qr(np.vstack((r_first, r_second)), mode="complete")
        else:
            q, r = qr(r_first, mode="complete")
        for i in range(len(r_blocks)):
            for j in range(len(r_blocks[i])):
                r_blocks[i][j] = r[block_shape[0] * i:block_shape[0] * (i + 1),
                                   block_shape[1] * j:block_shape[1] * (j + 1)]
    elif mode == "reduced":
        r_first = np.hstack(np.hstack(rs[0]))
        if len(rs) > 1:
            r_second = np.hstack(np.hstack(rs[1]))
        if len(rs) > 1:
            q, r = qr(np.vstack((r_first, r_second)))
        else:
            q, r = qr(r_first)
        block_shape_to_use = block_shape[0] if \
            block_shape[1] > block_shape[0] else block_shape[1]
        for j in range(len(r_blocks)):
            for i in range(len(r_blocks[0])):
                r_blocks[j][i] = r[block_shape_to_use * j:
                                   block_shape_to_use * (j + 1),
                                   block_shape_to_use * i:
                                   block_shape_to_use * (i + 1)]


@constraint(computing_units="${ComputingUnits}")
@task(block={Type: COLLECTION_IN, Depth: 2},
      q_blocks={Type: COLLECTION_OUT, Depth: 2},
      r_blocks={Type: COLLECTION_OUT, Depth: 2})
def _compute_qr(block, q_blocks, r_blocks, mode):
    block_shape = block[0][0].shape
    if mode == "complete":
        if len(block[0]) > 1:
            block = np.block(block[0])
            q, r = qr(block, mode="complete")
        else:
            q, r = qr(block[0][0], mode="complete")
        for i in range(len(q_blocks[0])):
            q_blocks[0][i] = q[:, block_shape[0]*i:block_shape[0]*(i+1)]
        for i in range(len(r_blocks[0])):
            r_blocks[0][i] = r[:, block_shape[1]*i:block_shape[1]*(i+1)]
    elif mode == "reduced":
        if len(block[0]) > 1:
            block = np.block(block[0])
            q, r = qr(block)
        else:
            q, r = qr(block[0][0])
        for i in range(len(q_blocks[0])):
            q_blocks[0][i] = q[:, block_shape[1]*i:block_shape[1]*(i+1)]
        for i in range(len(r_blocks)):
            for j in range(len(r_blocks[i])):
                r_blocks[i][j] = r[block_shape[1] * i:block_shape[1] * (i + 1),
                                   block_shape[1] * j:block_shape[1] * (j + 1)]
