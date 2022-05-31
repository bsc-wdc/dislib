from time import time
import unittest


class BaseTimedTestCase(unittest.TestCase):
    def setUp(self):
        self.start_time = time()

    def tearDown(self):
        self.end_time = time()
        print("Test %s took: %.3f seconds" %
              (self.id(), self.end_time - self.start_time))
