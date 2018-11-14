import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class ImportTests(unittest.TestCase):

    def test_import_fft(self):
        from dislib.fft import fft

    def test_import_cascadecsvm(self):
        from dislib.classification import CascadeSVM

    def test_import_kmeans(self):
        from dislib.cluster import KMeans

    def test_import_dbscan(self):
        from dislib.cluster import DBSCAN



class ResultsTest(unittest.TestCase):

    def test_cascadecsvm(self):
        from dislib.classification import CascadeSVM

def main():
    unittest.main()

if __name__ == '__main__':
    main()
