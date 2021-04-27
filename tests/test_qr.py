import unittest

import numpy as np
from parameterized import parameterized
from pycompss.api.api import compss_barrier, compss_wait_on
from pycompss.api.constraint import constraint
from pycompss.api.task import task

from dislib.data.array import Array, random_array, array
from dislib.math import qr_blocked


#class QRTest(unittest.TestCase):
class QRTest(object):

    @parameterized.expand([(1, 2), (1, 4), (2, 2), (2, 4), (3, 3), (3, 4), (4, 2), (4, 3), (4, 4)])
    #@parameterized.expand([(2, 2)])
    def test_qr(self, m_size, b_size):
        """Tests qr_blocked"""
        np.set_printoptions(precision=2)
        np.random.seed(8)

        #m_size = 2
        #b_size = 2
        mkl_threads = 512
        shape = (m_size * b_size, m_size * b_size)

        #m2b = self._gen_matrix(mkl_threads, m_size, b_size)
        #m2b = np.random.random(shape)
        #m2b_ds = Array(m2b, (b_size, b_size), (b_size, b_size), shape, sparse=False)
        m2b_ds = random_array(shape, (b_size, b_size))

        #m2b_ds = array([[0.79, 0.68, 0.07, 0.45], [0.14, 0.51, 0.07, 0.74], [0.42, 0.08, 0.26, 0.54], [0.01, 0.34, 0.01, 0.21]], (2, 2))

        compss_barrier()

        (Q, R) = qr_blocked(m2b_ds, mkl_threads)

        Q = compss_wait_on(Q).collect()
        R = compss_wait_on(R).collect()
        m2b_ds = compss_wait_on(m2b_ds)
        m2b = m2b_ds.collect()

        Q_blocked_np = self._ds_to_np(Q)
        R_blocked_np = self._ds_to_np(R)

        #print("Entered matrix")
        #print(m2b)
        #print("Q_blocked * R_blocked")
        #print(Q_blocked_np * R_blocked_np)

        q_np, r_np = np.linalg.qr(m2b)

        #print("Q_blocked")
        #print(Q_blocked_np)
        #print("Q_np")
        #print(q_np)
        #print("R_blocked")
        #print(R_blocked_np)
        #print("R_np")
        #print(r_np)

        # check if Q matrix is orthogonal
        self.assertTrue(np.allclose(Q_blocked_np.dot(Q_blocked_np.T), np.identity(m_size * b_size)))
        # check if R matrix is upper triangular
        self.assertTrue(np.allclose(np.triu(R_blocked_np), R_blocked_np))
        # check if the product Q * R is the original matrix
        self.assertTrue(np.allclose(Q_blocked_np.dot(R_blocked_np), m2b))

    def _ds_to_np(self, ds):
        ds_np = np.zeros(ds.shape)
        for i in range(ds.shape[0]):
            for j in range(ds.shape[1]):
                ds_np[i, j] = ds[i, j]
        return ds_np

    def _join_matrix(self, A, BSIZE):
        joinMat = np.matrix([[]])
        for i in range(0, len(A)):
            if A[i][0][0] == 'zeros':
                A[i][0][1] = np.matrix(np.zeros((BSIZE, BSIZE)))
            elif A[i][0][0] == 'identity':
                A[i][0][1] = np.matrix(np.identity(BSIZE))
            currRow = A[i][0][1]
            for j in range(1, len(A[i])):
                if A[i][j][0] == 'zeros':
                    A[i][j][1] = np.matrix(np.zeros((BSIZE, BSIZE)))
                elif A[i][j][0] == 'identity':
                    A[i][j][1] = np.matrix(np.identity(BSIZE))
                currRow = np.bmat([[currRow, A[i][j][1]]])
            if i == 0:
                joinMat = currRow
            else:
                joinMat = np.bmat([[joinMat], [currRow]])
        return np.matrix(joinMat)

    def _set_mkl_num_threads(self, mkl_proc):
        import os
        os.environ["MKL_NUM_THREADS"] = str(mkl_proc)

    #@constraint(ComputingUnits="${ComputingUnits}")
    #@task(returns=list)
    def _create_block_task(self, BSIZE, MKLProc):
        self._set_mkl_num_threads(MKLProc)
        return np.matrix(np.random.random((BSIZE, BSIZE)), dtype=np.double, copy=False)

    def _create_block(self, BSIZE, MKLProc, type='random'):
        if type == 'zeros':
            block = []
        elif type == 'identity':
            block = []
        else:
            block = self._create_block_task(BSIZE, MKLProc)
        return [type, block]

    def _gen_matrix(self, MKLProc, MSIZE, BSIZE):
        A = []
        for i in range(MSIZE):
            A.append([])
            for j in range(MSIZE):
                A[i].append(self._create_block(BSIZE, MKLProc, type='random'))
        return A


#def main():
#    unittest.main()


#if __name__ == '__main__':
#    main()
