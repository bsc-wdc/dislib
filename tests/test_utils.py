import unittest

import numpy as np

from dislib.data import Subset, Dataset
from dislib.utils import as_grid, shuffle


class UtilsTest(unittest.TestCase):
    def test_shuffle(self):
        s1 = Subset(samples=np.array([[1, 2], [4, 5], [2, 2], [6, 6]]),
                    labels=np.array([0, 1, 1, 1]))
        s2 = Subset(samples=np.array([[7, 8], [9, 8], [0, 4]]),
                    labels=np.array([0, 1, 1]))
        s3 = Subset(samples=np.array([[3, 9], [0, 7], [6, 1], [0, 8]]),
                    labels=np.array([0, 1, 1, 1]))

        dataset = Dataset(2)
        dataset.extend(s1, s2, s3)
        dataset._sizes = [2, 3, 4]

        shuffled = shuffle(dataset)
        shuffled.collect()

        total_size = 0

        for i, subset in enumerate(shuffled):
            equal = np.array_equal(shuffled[i].samples, dataset[i].samples)
            total_size += subset.samples.shape[0]
            self.assertFalse(equal)

        self.assertEqual(total_size, 9)

    def test_as_grid(self):
        s1 = Subset(samples=np.array([[1, 1], [8, 8], [2, 5]]))
        s2 = Subset(samples=np.array([[1, 7], [4, 4], [5, 9]]))
        s3 = Subset(samples=np.array([[4, 0], [8, 1], [7, 4]]))

        dataset = Dataset(n_features=2)
        dataset.extend(s1, s2, s3)
        grid = as_grid(dataset, n_regions=3)
        grid.collect()

        self.assertEqual(len(grid), 9)
        all_samples = np.empty((0, 2), dtype=int)

        for subset in grid:
            all_samples = np.concatenate((all_samples, subset.samples))

        true_samples = np.array(
            [[1, 1],
             [2, 5],
             [1, 7],
             [4, 0],
             [4, 4],
             [5, 9],
             [8, 1],
             [7, 4],
             [8, 8]])

        self.assertTrue(np.array_equal(all_samples, true_samples))

    def test_as_grid_indices(self):
        s1 = Subset(samples=np.array([[1, 1], [8, 8], [2, 5]]))
        s2 = Subset(samples=np.array([[1, 7], [4, 4], [5, 9]]))
        s3 = Subset(samples=np.array([[4, 0], [8, 1], [7, 4]]))

        dataset = Dataset(n_features=2)
        dataset.extend(s1, s2, s3)
        _, indices = as_grid(dataset, n_regions=3, return_indices=True)
        true_indices = np.array([0, 8, 1, 2, 4, 5, 3, 6, 7])

        self.assertTrue(np.array_equal(indices, true_indices))

    def test_as_grid_labels(self):
        s1 = Subset(samples=np.array([[1, 1], [8, 8], [2, 5]]),
                    labels=np.array([1, 2, 3]))
        s2 = Subset(samples=np.array([[1, 7], [4, 4], [5, 9]]),
                    labels=np.array([4, 5, 6]))
        s3 = Subset(samples=np.array([[4, 0], [8, 1], [7, 4]]),
                    labels=np.array([7, 8, 9]))

        dataset = Dataset(n_features=2)
        dataset.extend(s1, s2, s3)
        grid = as_grid(dataset, n_regions=3)
        grid.collect()

        self.assertEqual(len(grid), 9)
        all_samples = np.empty((0, 2), dtype=int)
        all_labels = np.empty(0, dtype=int)

        for subset in grid:
            all_samples = np.concatenate((all_samples, subset.samples))
            all_labels = np.concatenate((all_labels, subset.labels))

        true_samples = np.array(
            [[1, 1],
             [2, 5],
             [1, 7],
             [4, 0],
             [4, 4],
             [5, 9],
             [8, 1],
             [7, 4],
             [8, 8]])
        true_labels = np.array([1, 3, 4, 7, 5, 6, 8, 9, 2])

        self.assertTrue(np.array_equal(all_samples, true_samples))
        self.assertTrue(np.array_equal(all_labels, true_labels))

    def test_as_grid_sizes(self):
        """
        Tests whether as_grid correctly sets subset sizes, and that sizes
        can be properly retrieved later by calling Dataset.subset_size,
        which involves a synchronization.
        """
        s1 = Subset(samples=np.array([[1, 1], [8, 8], [2, 5]]))
        s2 = Subset(samples=np.array([[1, 7], [4, 4], [5, 9]]))
        s3 = Subset(samples=np.array([[4, 0], [8, 1], [7, 4]]))

        dataset = Dataset(n_features=2)
        dataset.extend(s1, s2, s3)
        sort = as_grid(dataset, n_regions=3)

        self.assertEqual(sort.subset_size(0), 1)
