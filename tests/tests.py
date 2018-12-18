import unittest


# the import tests should be removed; import should be tests in class specific
#  tests
class ImportTests(unittest.TestCase):
    def test_import_fft(self):
        from dislib.fft import fft
        self.assertIsNotNone(fft)

    def test_import_dbscan(self):
        from dislib.cluster import DBSCAN
        self.assertIsNotNone(DBSCAN)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
