from time import time
import unittest
import numpy as np


class BaseTimedTestCase(unittest.TestCase):
    def setUp(self):
        np.random.seed()
        self.start_time = time()

    def tearDown(self):
        self.end_time = time()
        print("Test %s took: %.3f seconds" %
              (self.id(), self.end_time - self.start_time))
