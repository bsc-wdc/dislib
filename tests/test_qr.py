import unittest

import numpy as np
from pycompss.api.api import compss_barrier, compss_wait_on
from pycompss.api.constraint import constraint
from pycompss.api.task import task
from dislib.math import qr_blocked


class QRTest(unittest.TestCase):

    def test_qr(self):
        """Tests qr_blocked"""
        np.random.seed(8)

        m_size = 1
        b_size = 4
        mkl_threads = 512

        m2b = self._gen_matrix(mkl_threads, m_size, b_size)

        compss_barrier()

        #m2b = [[['random', np.matrix(np.array([[0.04, 0.22, 0.15], [0.82, 0.23, 0.9], [0.03, 0.53, 0.12]]))],
        #        ['random', np.matrix(np.array([[0.16, 0.38, 0.07], [0.02, 0.53, 0.78], [0.22, 0.99, 0.24]]))]],
        #       [['random', np.matrix(np.array([[0.18, 0.46, 0.33], [0.92, 0.25, 0.75], [0.29, 0.23, 0.95]]))],
        #        ['random', np.matrix(np.array([[0.99, 0.68, 0.87], [0.16, 0.48, 0.45], [0.41, 0.4, 0.07]]))]]]

        (Q, R) = qr_blocked(m2b, mkl_threads, m_size, b_size)

        Q = compss_wait_on(Q)
        R = compss_wait_on(R)
        m2b = compss_wait_on(m2b)

        print("Matriu entrada")
        print(self._join_matrix(m2b, b_size))
        print("Q*R")
        print(self._join_matrix(Q, b_size))
        print(R)
        print(self._join_matrix(R, b_size))
        print(self._join_matrix(Q, b_size) * self._join_matrix(R, b_size))
        print("R generada")
        print(self._join_matrix(R, b_size))
        q, r = np.linalg.qr(self._join_matrix(m2b, b_size))
        print("R numpy")
        print(r)
        print("Q generada")
        print(self._join_matrix(Q, b_size))
        print("Q numpy")
        print(q)

        self.assertTrue(np.array_equal(self._join_matrix(Q, b_size), q))
        self.assertTrue(np.array_equal(self._join_matrix(R, b_size), r))

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


def main():
    unittest.main()


if __name__ == '__main__':
    main()
