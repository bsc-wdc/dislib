import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class ImportTests(unittest.TestCase):

    def test_import_fft(self):
        from dislib.math import fft
        from dislib.math.fft import fft

    def test_import_cascadecsvm(self):
        from dislib.ml.classification import CascadeSVM


    def test_import_kmeans(self):
        from dislib.ml.clustering import kmeans

    def test_import_dbscan(self):
        from dislib.ml.clustering import DBSCAN



class ResultsTest(unittest.TestCase):
    #import pycompss.interactive as ipycompss
    #ipycompss.start(graph=False, trace=False, debug=False)
    #ipycompss.start(graph=True, trace=True, debug=True, project_xml='../project.xml', resources_xml='../resources.xml')

    def test_cascadecsvm(self):
        from dislib.ml.classification import CascadeSVM

    def test_fft(self):
        import numpy as np
        from dislib.math.fft import fft

        arr = np.random.rand(32)

        nfft = np.fft.fft(arr)
        pfft = fft(arr)

        self.assertTrue(np.allclose(nfft, pfft))

    def test_matmul(self):
        from dislib.math.linalg.matmul import dot
        from dislib.math.linalg.matmul import initialize_variables
        import numpy as np

        A, B, C = initialize_variables(4, 16, 1)
        dot(A, B, C, 4, 1)

        Ajoin = np.bmat(A)
        Bjoin = np.bmat(B)
        Cjoin = np.bmat(C)

        self.assertTrue(np.allclose(Cjoin, np.dot(Ajoin, Bjoin)))

def main():
    unittest.main()

if __name__ == '__main__':
    main()
