import unittest
import numpy as np
import dislib as ds
from dislib.sorting import TeraSort
from tests import BaseTimedTestCase
from pycompss.api.api import compss_wait_on
from parameterized import parameterized


class TerasortTest(BaseTimedTestCase):

    def test_exceptions(self):
        with self.assertRaises(ValueError):
            TeraSort(range_min=3, range_max=2)
        with self.assertRaises(ValueError):
            TeraSort(num_buckets=1.5)
        with self.assertRaises(ValueError):
            x = ds.array(np.array([258, 129, 528, 248, 0]), block_size=(2, 1))
            ts = TeraSort(range_max=[10, 30, 20],
                          column_indexes=np.array([1, 2, 3, 4]))
            ts.sort(x)
        with self.assertRaises(ValueError):
            x = ds.array(np.array([258, 129, 528, 248, 0]), block_size=(2, 1))
            ts = TeraSort(range_min=[10, 30, 20],
                          column_indexes=np.array([3, 4]))
            ts.sort(x)
        with self.assertRaises(ValueError):
            x = ds.array(np.array([258, 129, 528, 248, 0]), block_size=(2, 1))
            ts = TeraSort(range_min=100, range_max=0, column_indexes=[3, 4])
            ts.sort(x)
        with self.assertRaises(ValueError):
            x = np.array([258, 129, 528, 248, 0])
            ts = TeraSort(column_indexes=[3, 4])
            ts.sort(x)

    def test_fit(self):
        ts = TeraSort()
        x = ds.array(np.array([[0],
                               [1],
                               [2],
                               [3],
                               [4]]), block_size=(2, 1))
        ts.fit(x)
        self.assertTrue(compss_wait_on(ts.range_max) == [[[[4]]]])
        self.assertTrue(compss_wait_on(ts.range_min) == [[[[0]]]])
        x = ds.array(np.array([[0, 1, 2, 3, 4], [3, 2, 1, 0, 8]]),
                     block_size=(2, 1))
        ts = TeraSort(column_indexes=np.array([0, 1, 2, 3, 4]))
        ts.fit(x)
        self.assertTrue(np.all(ts.range_max.collect() == [3, 2, 2, 3, 8]))
        self.assertTrue(np.all(ts.range_min.collect() == [0, 1, 1, 0, 4]))

    def test_sort_no_fit(self):
        ts = TeraSort()
        x = ds.array(np.array([[3],
                               [4],
                               [1],
                               [2],
                               [0]]), block_size=(2, 1))
        sorted_x = ts.sort(x)
        self.assertTrue(np.all(sorted_x.collect() ==
                               np.array([0, 1, 2, 3, 4])))
        x = ds.array(np.array([[0, 1, 2, 3, 4], [3, 2, 1, 0, 8]]),
                     block_size=(2, 1))
        ts = TeraSort(column_indexes=np.array([0, 1, 2, 3, 4]))
        sorted_x = ts.sort(x)
        self.assertTrue(np.all(sorted_x.collect() ==
                               np.array([[0, 1, 1, 0, 4], [3, 2, 2, 3, 8]])))

    @parameterized.expand([(ds.array(np.array([258, 129, 528, 248, 0]),
                                     block_size=(1, 2)),
                            np.array([0, 129, 248, 258, 528]),
                            (1, 5), (1, 2)),
                           (ds.array(np.array([[95, 90, 20, 80, 69],
                                               [93, 93, 79, 47, 64],
                                               [97, 99, 82, 99, 88],
                                               [93, 95, 49, 29, 19],
                                               [92, 94, 19, 14, 39],
                                               [97, 96, 32, 65,  9],
                                               [98, 98, 57, 32, 31],
                                               [91, 96, 74, 23, 35],
                                               [97, 97, 75, 55, 28],
                                               [98, 91, 34,  0,  0],
                                               [95, 99, 36, 53,  5],
                                               [98, 99, 38, 17, 79]]), (5, 2)),
                            np.array([[0, 0, 5, 9, 14],
                                      [17, 19, 19, 20, 23],
                                      [28, 29, 31, 32, 32],
                                      [34, 35, 36, 38, 39],
                                      [47, 49, 53, 55, 57],
                                      [64, 65, 69, 74, 75],
                                      [79, 79, 80, 82, 88],
                                      [90, 91, 91, 92, 93],
                                      [93, 93, 94, 95, 95],
                                      [95, 96, 96, 97, 97],
                                      [97, 97, 98, 98, 98],
                                      [98, 99, 99, 99, 99]]),
                            (12, 5), (5, 2)
                            ),
                           (
                                   ds.array(np.array([[95, 90, 20, 80, 69],
                                                      [93, 93, 79, 47, 64],
                                                      [97, 99, 82, 99, 88],
                                                      [93, 95, 49, 29, 19],
                                                      [92, 94, 19, 14, 39],
                                                      [97, 96, 32, 65, 9],
                                                      [98, 98, 57, 32, 31],
                                                      [91, 96, 74, 23, 35],
                                                      [97, 97, 75, 55, 28],
                                                      [98, 91, 34, 0, 0],
                                                      [95, 99, 36, 53, 5],
                                                      [98, 99, 38, 17, 79]]),
                                            (4, 2)),
                                   np.array([[0, 0, 5, 9, 14],
                                             [17, 19, 19, 20, 23],
                                             [28, 29, 31, 32, 32],
                                             [34, 35, 36, 38, 39],
                                             [47, 49, 53, 55, 57],
                                             [64, 65, 69, 74, 75],
                                             [79, 79, 80, 82, 88],
                                             [90, 91, 91, 92, 93],
                                             [93, 93, 94, 95, 95],
                                             [95, 96, 96, 97, 97],
                                             [97, 97, 98, 98, 98],
                                             [98, 99, 99, 99, 99]]),
                                   (12, 5),
                                   (4, 2)
                           ),
                           (
                                   ds.array(np.array([[95, 90, 20, 80],
                                                      [93, 93, 69, 79],
                                                      [97, 99, 47, 64],
                                                      [93, 95, 82, 99],
                                                      [92, 94, 88, 49],
                                                      [97, 96, 29, 19],
                                                      [98, 98, 19, 14],
                                                      [91, 96, 39, 32],
                                                      [97, 97, 65, 9],
                                                      [98, 91, 57, 32],
                                                      [95, 99, 31, 74],
                                                      [98, 99, 23, 35]]),
                                            (5, 2)),
                                   np.array([[9, 14, 19, 19],
                                             [20, 23, 29, 31],
                                             [32, 32, 35, 39],
                                             [47, 49, 57, 64],
                                             [65, 69, 74, 79],
                                             [80, 82, 88, 90],
                                             [91, 91, 92, 93],
                                             [93, 93, 94, 95],
                                             [95, 95, 96, 96],
                                             [97, 97, 97, 97],
                                             [98, 98, 98, 98],
                                             [99, 99, 99, 99]]),
                                   (12, 4),
                                   (5, 2)
                           ),
                           (
                                   ds.array(np.array([[95, 90, 20, 80],
                                                      [93, 93, 69, 79],
                                                      [97, 99, 47, 64],
                                                      [93, 95, 82, 99],
                                                      [92, 94, 88, 49],
                                                      [97, 96, 29, 19],
                                                      [98, 98, 19, 14],
                                                      [91, 96, 39, 32],
                                                      [97, 97, 65, 9],
                                                      [98, 91, 57, 32],
                                                      [95, 99, 31, 74],
                                                      [98, 99, 23, 35]]),
                                            (4, 2)),
                                   np.array([[9, 14, 19, 19],
                                             [20, 23, 29, 31],
                                             [32, 32, 35, 39],
                                             [47, 49, 57, 64],
                                             [65, 69, 74, 79],
                                             [80, 82, 88, 90],
                                             [91, 91, 92, 93],
                                             [93, 93, 94, 95],
                                             [95, 95, 96, 96],
                                             [97, 97, 97, 97],
                                             [98, 98, 98, 98],
                                             [99, 99, 99, 99]]),
                                   (12, 4),
                                   (4, 2)
                           )])
    def test_sort(self, x, sorted_np, shape, reg_shape):
        ts = TeraSort()
        ts.fit(x)
        sorted_x = ts.sort(x)
        self.assertTrue(sorted_x.shape == shape)
        self.assertTrue(sorted_x._reg_shape == reg_shape)
        sorted_x = sorted_x.collect()
        self.assertTrue(np.all(sorted_x == sorted_np))

    @parameterized.expand([(ds.array(np.array([[95, 90, 20, 80, 69],
                                               [93, 93, 79, 47, 64],
                                               [97, 99, 82, 99, 88],
                                               [93, 95, 49, 29, 19],
                                               [92, 94, 19, 14, 39],
                                               [97, 96, 32, 65, 9],
                                               [98, 98, 57, 32, 31],
                                               [91, 96, 74, 23, 35],
                                               [97, 97, 75, 55, 28],
                                               [98, 91, 34, 0, 0],
                                               [95, 99, 36, 53, 5],
                                               [98, 99, 38, 17, 79]]),
                                     (5, 2)),
                            np.array([[90, 19],
                                      [91, 20],
                                      [93, 32],
                                      [94, 34],
                                      [95, 36],
                                      [96, 38],
                                      [96, 49],
                                      [97, 57],
                                      [98, 74],
                                      [99, 75],
                                      [99, 79],
                                      [99, 82]]),
                            [1, 2], (12, 2), (5, 2)
                            ),
                           (
                                   ds.array(np.array([[95, 90, 20, 80, 69],
                                                      [93, 93, 79, 47, 64],
                                                      [97, 99, 82, 99, 88],
                                                      [93, 95, 49, 29, 19],
                                                      [92, 94, 19, 14, 39],
                                                      [97, 96, 32, 65, 9],
                                                      [98, 98, 57, 32, 31],
                                                      [91, 96, 74, 23, 35],
                                                      [97, 97, 75, 55, 28],
                                                      [98, 91, 34, 0, 0],
                                                      [95, 99, 36, 53, 5],
                                                      [98, 99, 38, 17, 79]]),
                                            (4, 2)),
                                   np.array([[90, 19],
                                             [91, 20],
                                             [93, 32],
                                             [94, 34],
                                             [95, 36],
                                             [96, 38],
                                             [96, 49],
                                             [97, 57],
                                             [98, 74],
                                             [99, 75],
                                             [99, 79],
                                             [99, 82]]),
                                   [1, 2],
                                   (12, 2),
                                   (4, 2)

                           ),
                           (
                                   ds.array(np.array([[95, 90, 20, 80],
                                                      [93, 93, 69, 79],
                                                      [97, 99, 47, 64],
                                                      [93, 95, 82, 99],
                                                      [92, 94, 88, 49],
                                                      [97, 96, 29, 19],
                                                      [98, 98, 19, 14],
                                                      [91, 96, 39, 32],
                                                      [97, 97, 65, 9],
                                                      [98, 91, 57, 32],
                                                      [95, 99, 31, 74],
                                                      [98, 99, 23, 35]]),
                                            (5, 2)),
                                   np.array([[90, 19],
                                             [91, 20],
                                             [93, 23],
                                             [94, 29],
                                             [95, 31],
                                             [96, 39],
                                             [96, 47],
                                             [97, 57],
                                             [98, 65],
                                             [99, 69],
                                             [99, 82],
                                             [99, 88]]),
                                   [1, 2], (12, 2), (5, 2)
                           ),
                           (
                                   ds.array(np.array([[95, 90, 20, 80],
                                                      [93, 93, 69, 79],
                                                      [97, 99, 47, 64],
                                                      [93, 95, 82, 99],
                                                      [92, 94, 88, 49],
                                                      [97, 96, 29, 19],
                                                      [98, 98, 19, 14],
                                                      [91, 96, 39, 32],
                                                      [97, 97, 65, 9],
                                                      [98, 91, 57, 32],
                                                      [95, 99, 31, 74],
                                                      [98, 99, 23, 35]]),
                                            (4, 2)),
                                   np.array([[90, 19],
                                             [91, 20],
                                             [93, 23],
                                             [94, 29],
                                             [95, 31],
                                             [96, 39],
                                             [96, 47],
                                             [97, 57],
                                             [98, 65],
                                             [99, 69],
                                             [99, 82],
                                             [99, 88]]),
                                   [1, 2], (12, 2), (4, 2)
                           )])
    def test_sorted_columns(self, x, sorted_np, columns_to_sort,
                            shape, block_shape):
        ts = TeraSort(column_indexes=columns_to_sort)
        ts.fit(x)
        sorted_x = ts.sort(x)
        self.assertTrue(sorted_x.shape == shape)
        self.assertTrue(sorted_x._reg_shape == block_shape)
        sorted_x = sorted_x.collect()
        self.assertTrue(np.all(sorted_x == sorted_np))


def main():
    unittest.main()


if __name__ == "__main__":
    main()
