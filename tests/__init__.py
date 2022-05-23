import unittest
from time import time

class DislibTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.start_time = time()
        return super().setUp()

    def tearDown(self) -> None:
        end_time = time()
        print(f"{self.id()} took {end_time - self.start_time} seconds")
        return super().tearDown()