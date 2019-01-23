import unittest

import numpy as np


class FFTTest(unittest.TestCase):
    def test_fft(self):
        """ Tests the fft against np.fft """
        from dislib.fft import fft
        our = fft(np.exp(2j * np.pi * np.arange(8) / 8))
        their = np.fft.fft(np.exp(2j * np.pi * np.arange(8) / 8))

        self.assertTrue(np.allclose(our, their))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
