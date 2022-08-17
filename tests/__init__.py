import datetime
from time import time
import unittest


class BaseTimedTestCase(unittest.TestCase):
    def setUp(self):
        # self.start_time = time()
        print(f'Test {self.id()} START AT', datetime.datetime.now(), flush=True)

    def tearDown(self):
        print(f'Test {self.id()} END AT', datetime.datetime.now(), flush=True)
        # self.end_time = time()
        # print("Test %s took: %.3f seconds" %
        #       (self.id(), self.end_time - self.start_time))
