import unittest
import numpy as np
import dislib as ds
from dislib.sorting import TeraSort
from tests import BaseTimedTestCase
from pycompss.api.api import compss_wait_on


class TerasortTest(BaseTimedTestCase):

    def test_exceptions(self):
        with self.assertRaises(ValueError):
            TeraSort(range_min=3, range_max=2)
        with self.assertRaises(ValueError):
            TeraSort(num_buckets=1.5)
        with self.assertRaises(ValueError):
            TeraSort(returns="Array")
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
        with self.assertRaises(NotImplementedError):
            x = ds.array(np.array([258, 129, 528, 248, 0]), block_size=(2, 1))
            ts = TeraSort(column_indexes=[3, 4],
                          returns="dict")
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

    def test_sort(self):
        ts = TeraSort()
        x = ds.array(np.array([258, 129, 528, 248, 0]), block_size=(1, 2))
        ts.fit(x)
        sorted_x = ts.sort(x)
        self.assertTrue(sorted_x.shape == (1, 5))
        self.assertTrue(sorted_x._reg_shape == (1, 2))
        sorted_x = sorted_x.collect()
        self.assertTrue(np.all(sorted_x == np.array([0, 129, 248, 258, 528])))
        self.assertTrue(sorted_x.shape == (5,))
        x = ds.array(np.array([[95, 90, 20, 80, 69],
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
                               [98, 99, 38, 17, 79]]), (5, 2))
        ts = TeraSort(synchronize=False)
        ts.fit(x)
        sorted_x = ts.sort(x)
        self.assertTrue(sorted_x.shape == (12, 5))
        self.assertTrue(sorted_x._reg_shape == (5, 2))
        sorted_x = sorted_x.collect()
        self.assertTrue(np.all(sorted_x == np.array([[0, 0, 28, 29, 47],
                                                    [5, 9, 31, 32, 49],
                                                    [14, 17, 32, 34, 53],
                                                    [19, 19, 35, 36, 55],
                                                    [20, 23, 38, 39, 57],
                                                    [64, 65, 90, 91, 95],
                                                    [69, 74, 91, 92, 96],
                                                    [75, 79, 93, 93, 96],
                                                    [79, 80, 93, 94, 97],
                                                    [82, 88, 95, 95, 97],
                                                    [97, 97, 98, 98, 99],
                                                    [98, 98, 99, 99, 99]])))
        self.assertTrue(sorted_x.shape == (12, 5))

        x = ds.array(np.array([[95, 90, 20, 80, 69],
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
                               [98, 99, 38, 17, 79]]), (4, 2))
        ts = TeraSort(synchronize=False)
        ts.fit(x)
        sorted_x = ts.sort(x)
        self.assertTrue(sorted_x.shape == (12, 5))
        self.assertTrue(sorted_x._reg_shape == (4, 2))
        sorted_x = sorted_x.collect()
        self.assertTrue(np.all(sorted_x == np.array([[0, 0, 20, 23, 35],
                                                    [5, 9, 28, 29, 36],
                                                    [14, 17, 31, 32, 38],
                                                    [19, 19, 32, 34, 39],
                                                    [47, 49, 74, 75, 91],
                                                    [53, 55, 79, 79, 91],
                                                    [57, 64, 80, 82, 92],
                                                    [65, 69, 88, 90, 93],
                                                    [93, 93, 97, 97, 99],
                                                    [94, 95, 97, 97, 99],
                                                    [95, 95, 98, 98, 99],
                                                    [96, 96, 98, 98, 99]])))
        self.assertTrue(sorted_x.shape == (12, 5))

        x = ds.array(np.array([[95, 90, 20, 80],
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
                               [98, 99, 23, 35]]), (5, 2))
        ts = TeraSort(synchronize=False)
        ts.fit(x)
        sorted_x = ts.sort(x)
        self.assertTrue(sorted_x.shape == (12, 4))
        self.assertTrue(sorted_x._reg_shape == (5, 2))
        sorted_x = sorted_x.collect()
        self.assertTrue(np.all(sorted_x == np.array([[9, 14, 35, 39],
                                                    [19, 19, 47, 49],
                                                    [20, 23, 57, 64],
                                                    [29, 31, 65, 69],
                                                    [32, 32, 74, 79],
                                                    [80, 82, 94, 95],
                                                    [88, 90, 95, 95],
                                                    [91, 91, 96, 96],
                                                    [92, 93, 97, 97],
                                                    [93, 93, 97, 97],
                                                    [98, 98, 99, 99],
                                                    [98, 98, 99, 99]])))
        self.assertTrue(sorted_x.shape == (12, 4))
        x = ds.array(np.array([[95, 90, 20, 80],
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
                               [98, 99, 23, 35]]), (4, 2))
        ts = TeraSort(synchronize=False)
        ts.fit(x)
        sorted_x = ts.sort(x)
        self.assertTrue(sorted_x.shape == (12, 4))
        self.assertTrue(sorted_x._reg_shape == (4, 2))
        sorted_x = sorted_x.collect()
        self.assertTrue(np.all(sorted_x == np.array([[9, 14, 32, 32],
                                                    [19, 19, 35, 39],
                                                    [20, 23, 47, 49],
                                                    [29, 31, 57, 64],
                                                    [65, 69, 91, 91],
                                                    [74, 79, 92, 93],
                                                    [80, 82, 93, 93],
                                                    [88, 90, 94, 95],
                                                    [95, 95, 98, 98],
                                                    [96, 96, 98, 98],
                                                    [97, 97, 99, 99],
                                                    [97, 97, 99, 99]])))
        self.assertTrue(sorted_x.shape == (12, 4))

    def test_sort_dict(self):
        x = ds.array(np.array([[95, 90, 20, 80, 69],
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
                               [98, 99, 38, 17, 79]]), (5, 2))
        ts = TeraSort(synchronize=False, returns="dict")
        ts.fit(x)
        sorted_x = ts.sort(x)
        sorted_x = compss_wait_on(sorted_x)
        final_values = []
        for key, values in sorted_x.items():
            self.assertTrue(np.all(values[:-1] <= values[1:]))
            final_values.append(values)
        for idx in range(len(final_values)-1):

            self.assertTrue(final_values[idx][-1] < final_values[idx + 1][0])

    def test_sort_synchronized(self):
        x = ds.array(np.array([[95, 90, 20, 80, 69],
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
                               [98, 99, 38, 17, 79]]), (5, 2))
        ts = TeraSort(synchronize=True)
        ts.fit(x)
        sorted_x = ts.sort(x)
        self.assertTrue(sorted_x.shape == (12, 5))
        self.assertTrue(sorted_x._reg_shape == (5, 2))
        sorted_x = sorted_x.collect()
        self.assertTrue(np.all(sorted_x == np.array([[0, 0, 28, 29, 47],
                                                    [5, 9, 31, 32, 49],
                                                    [14, 17, 32, 34, 53],
                                                    [19, 19, 35, 36, 55],
                                                    [20, 23, 38, 39, 57],
                                                    [64, 65, 90, 91, 95],
                                                    [69, 74, 91, 92, 96],
                                                    [75, 79, 93, 93, 96],
                                                    [79, 80, 93, 94, 97],
                                                    [82, 88, 95, 95, 97],
                                                    [97, 97, 98, 98, 99],
                                                    [98, 98, 99, 99, 99]])))
        self.assertTrue(sorted_x.shape == (12, 5))

        x = ds.array(np.array([[95, 90, 20, 80, 69],
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
                               [98, 99, 38, 17, 79]]), (4, 2))
        ts = TeraSort(synchronize=True)
        ts.fit(x)
        sorted_x = ts.sort(x)
        self.assertTrue(sorted_x.shape == (12, 5))
        self.assertTrue(sorted_x._reg_shape == (4, 2))
        sorted_x = sorted_x.collect()
        self.assertTrue(np.all(sorted_x == np.array([[0, 0, 20, 23, 35],
                                                    [5, 9, 28, 29, 36],
                                                    [14, 17, 31, 32, 38],
                                                    [19, 19, 32, 34, 39],
                                                    [47, 49, 74, 75, 91],
                                                    [53, 55, 79, 79, 91],
                                                    [57, 64, 80, 82, 92],
                                                    [65, 69, 88, 90, 93],
                                                    [93, 93, 97, 97, 99],
                                                    [94, 95, 97, 97, 99],
                                                    [95, 95, 98, 98, 99],
                                                    [96, 96, 98, 98, 99]])))
        self.assertTrue(sorted_x.shape == (12, 5))

        x = ds.array(np.array([[95, 90, 20, 80],
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
                               [98, 99, 23, 35]]), (5, 2))
        ts = TeraSort(synchronize=True)
        ts.fit(x)
        sorted_x = ts.sort(x)
        self.assertTrue(sorted_x.shape == (12, 4))
        self.assertTrue(sorted_x._reg_shape == (5, 2))
        sorted_x = sorted_x.collect()
        self.assertTrue(np.all(sorted_x == np.array([[9, 14, 35, 39],
                                                    [19, 19, 47, 49],
                                                    [20, 23, 57, 64],
                                                    [29, 31, 65, 69],
                                                    [32, 32, 74, 79],
                                                    [80, 82, 94, 95],
                                                    [88, 90, 95, 95],
                                                    [91, 91, 96, 96],
                                                    [92, 93, 97, 97],
                                                    [93, 93, 97, 97],
                                                    [98, 98, 99, 99],
                                                    [98, 98, 99, 99]])))
        self.assertTrue(sorted_x.shape == (12, 4))
        x = ds.array(np.array([[95, 90, 20, 80],
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
                               [98, 99, 23, 35]]), (4, 2))
        ts = TeraSort(synchronize=True)
        ts.fit(x)
        sorted_x = ts.sort(x)
        self.assertTrue(sorted_x.shape == (12, 4))
        self.assertTrue(sorted_x._reg_shape == (4, 2))
        sorted_x = sorted_x.collect()
        self.assertTrue(np.all(sorted_x == np.array([[9, 14, 32, 32],
                                                    [19, 19, 35, 39],
                                                    [20, 23, 47, 49],
                                                    [29, 31, 57, 64],
                                                    [65, 69, 91, 91],
                                                    [74, 79, 92, 93],
                                                    [80, 82, 93, 93],
                                                    [88, 90, 94, 95],
                                                    [95, 95, 98, 98],
                                                    [96, 96, 98, 98],
                                                    [97, 97, 99, 99],
                                                    [97, 97, 99, 99]])))
        self.assertTrue(sorted_x.shape == (12, 4))

    def test_sorted_columns(self):
        x = ds.array(np.array([[95, 90, 20, 80, 69],
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
                               [98, 99, 38, 17, 79]]), (5, 2))
        ts = TeraSort(synchronize=False, column_indexes=[1, 2])
        ts.fit(x)
        sorted_x = ts.sort(x)
        self.assertTrue(sorted_x.shape == (12, 2))
        self.assertTrue(sorted_x._reg_shape == (5, 2))
        sorted_x = sorted_x.collect()
        self.assertTrue(np.all(sorted_x == np.array([[90, 19],
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
                                                    [99, 82]])))
        self.assertTrue(sorted_x.shape == (12, 2))

        x = ds.array(np.array([[95, 90, 20, 80, 69],
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
                               [98, 99, 38, 17, 79]]), (4, 2))
        ts = TeraSort(synchronize=False, column_indexes=np.array([1, 2]))
        ts.fit(x)
        sorted_x = ts.sort(x)
        self.assertTrue(sorted_x.shape == (12, 2))
        self.assertTrue(sorted_x._reg_shape == (4, 2))
        sorted_x = sorted_x.collect()
        self.assertTrue(np.all(sorted_x == np.array([[90, 19],
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
                                                    [99, 82]])))
        self.assertTrue(sorted_x.shape == (12, 2))

        x = ds.array(np.array([[95, 90, 20, 80],
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
                               [98, 99, 23, 35]]), (5, 2))
        ts = TeraSort(synchronize=False, column_indexes=[1, 2])
        ts.fit(x)
        sorted_x = ts.sort(x)
        self.assertTrue(sorted_x.shape == (12, 2))
        self.assertTrue(sorted_x._reg_shape == (5, 2))
        sorted_x = sorted_x.collect()
        self.assertTrue(np.all(sorted_x == np.array([[90, 19],
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
                                                    [99, 88]])))
        self.assertTrue(sorted_x.shape == (12, 2))
        x = ds.array(np.array([[95, 90, 20, 80],
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
                               [98, 99, 23, 35]]), (4, 2))
        ts = TeraSort(synchronize=False, column_indexes=[1, 2])
        ts.fit(x)
        sorted_x = ts.sort(x)
        self.assertTrue(sorted_x.shape == (12, 2))
        self.assertTrue(sorted_x._reg_shape == (4, 2))
        sorted_x = sorted_x.collect()
        self.assertTrue(np.all(sorted_x == np.array([[90, 19],
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
                                                    [99, 88]])))
        self.assertTrue(sorted_x.shape == (12, 2))

    def test_sorted_columns_synchronized(self):
        x = ds.array(np.array([[95, 90, 20, 80, 69],
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
                               [98, 99, 38, 17, 79]]), (5, 2))
        ts = TeraSort(synchronize=True, column_indexes=[1, 2])
        ts.fit(x)
        sorted_x = ts.sort(x)
        self.assertTrue(sorted_x.shape == (12, 2))
        self.assertTrue(sorted_x._reg_shape == (5, 2))
        sorted_x = sorted_x.collect()
        self.assertTrue(np.all(sorted_x == np.array([[90, 19],
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
                                                    [99, 82]])))
        self.assertTrue(sorted_x.shape == (12, 2))

        x = ds.array(np.array([[95, 90, 20, 80, 69],
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
                               [98, 99, 38, 17, 79]]), (4, 2))
        ts = TeraSort(synchronize=True, column_indexes=np.array([1, 2]))
        ts.fit(x)
        sorted_x = ts.sort(x)
        self.assertTrue(sorted_x.shape == (12, 2))
        self.assertTrue(sorted_x._reg_shape == (4, 2))
        sorted_x = sorted_x.collect()
        self.assertTrue(np.all(sorted_x == np.array([[90, 19],
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
                                                    [99, 82]])))
        self.assertTrue(sorted_x.shape == (12, 2))

        x = ds.array(np.array([[95, 90, 20, 80],
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
                               [98, 99, 23, 35]]), (5, 2))
        ts = TeraSort(synchronize=True, column_indexes=[1, 2])
        ts.fit(x)
        sorted_x = ts.sort(x)
        self.assertTrue(sorted_x.shape == (12, 2))
        self.assertTrue(sorted_x._reg_shape == (5, 2))
        sorted_x = sorted_x.collect()
        self.assertTrue(np.all(sorted_x == np.array([[90, 19],
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
                                                    [99, 88]])))
        self.assertTrue(sorted_x.shape == (12, 2))
        x = ds.array(np.array([[95, 90, 20, 80],
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
                               [98, 99, 23, 35]]), (4, 2))
        ts = TeraSort(synchronize=True, column_indexes=[1, 2])
        ts.fit(x)
        sorted_x = ts.sort(x)
        self.assertTrue(sorted_x.shape == (12, 2))
        self.assertTrue(sorted_x._reg_shape == (4, 2))
        sorted_x = sorted_x.collect()
        self.assertTrue(np.all(sorted_x == np.array([[90, 19],
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
                                                    [99, 88]])))
        self.assertTrue(sorted_x.shape == (12, 2))


def main():
    unittest.main()


if __name__ == "__main__":
    main()
