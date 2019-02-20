import unittest

import numpy as np
from scipy.sparse import csr_matrix

from dislib.data import Subset, Dataset, load_libsvm_file, load_data
from dislib.utils import as_grid, shuffle, resample


class UtilsTest(unittest.TestCase):
    def test_shuffle(self):
        """ Tests that the content of the subsets of a dataset changes after
        running shuffle, and that the sizes of the original and shuffled
        datasets are equal.
        """
        s1 = Subset(samples=np.array([[1, 2], [4, 5], [2, 2], [6, 6]]),
                    labels=np.array([0, 1, 1, 1]))
        s2 = Subset(samples=np.array([[7, 8], [9, 8], [0, 4]]),
                    labels=np.array([0, 1, 1]))
        s3 = Subset(samples=np.array([[3, 9], [0, 7], [6, 1], [0, 8]]),
                    labels=np.array([0, 1, 1, 1]))

        dataset = Dataset(2)
        dataset.extend([s1, s2, s3])
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
        """ Tests the as_grid method with toy data."""
        s1 = Subset(samples=np.array([[1, 1], [8, 8], [2, 5]]))
        s2 = Subset(samples=np.array([[1, 7], [4, 4], [5, 9]]))
        s3 = Subset(samples=np.array([[4, 0], [8, 1], [7, 4]]))

        dataset = Dataset(n_features=2)
        dataset.extend([s1, s2, s3])
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
        """ Tests that as_grid returns correct indices with toy data.
        """
        s1 = Subset(samples=np.array([[1, 1], [8, 8], [2, 5]]))
        s2 = Subset(samples=np.array([[1, 7], [4, 4], [5, 9]]))
        s3 = Subset(samples=np.array([[4, 0], [8, 1], [7, 4]]))

        dataset = Dataset(n_features=2)
        dataset.extend([s1, s2, s3])
        _, indices = as_grid(dataset, n_regions=3, return_indices=True)
        true_indices = np.array([0, 8, 1, 2, 4, 5, 3, 6, 7])

        self.assertTrue(np.array_equal(indices, true_indices))

    def test_as_grid_labels(self):
        """ Tests as_grid method with a labeled toy dataset.
        """
        s1 = Subset(samples=np.array([[1, 1], [8, 8], [2, 5]]),
                    labels=np.array([1, 2, 3]))
        s2 = Subset(samples=np.array([[1, 7], [4, 4], [5, 9]]),
                    labels=np.array([4, 5, 6]))
        s3 = Subset(samples=np.array([[4, 0], [8, 1], [7, 4]]),
                    labels=np.array([7, 8, 9]))

        dataset = Dataset(n_features=2)
        dataset.extend([s1, s2, s3])
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
        dataset.extend([s1, s2, s3])
        sort = as_grid(dataset, n_regions=3)

        self.assertEqual(sort.subset_size(0), 1)
        self.assertEqual(len(sort), 9)

    def test_as_grid_dimensions(self):
        """
        Tests as_grid method using a subset of the dimensions.
        """
        s1 = Subset(samples=np.array([[0, 1, 9], [8, 8, 2], [2, 5, 4]]))
        s2 = Subset(samples=np.array([[1, 7, 6], [4, 4, 2], [5, 9, 0]]))
        s3 = Subset(samples=np.array([[4, 0, 1], [9, 1, 7], [7, 4, 3]]))

        dataset = Dataset(n_features=3)
        dataset.extend([s1, s2, s3])
        sort = as_grid(dataset, n_regions=3, dimensions=[0])

        self.assertEqual(sort.subset_size(0), 3)
        self.assertEqual(sort.subset_size(1), 3)
        self.assertEqual(sort.subset_size(2), 3)
        self.assertEqual(len(sort), 3)

        sort = as_grid(dataset, n_regions=3, dimensions=[0, 1])

        self.assertEqual(sort.subset_size(0), 1)
        self.assertEqual(sort.subset_size(1), 1)
        self.assertEqual(sort.subset_size(2), 1)
        self.assertEqual(sort.subset_size(4), 1)
        self.assertEqual(sort.subset_size(5), 1)
        self.assertEqual(len(sort), 9)

        sort = as_grid(dataset, n_regions=3, dimensions=[1, 2])

        self.assertEqual(sort.subset_size(0), 1)
        self.assertEqual(sort.subset_size(1), 0)
        self.assertEqual(sort.subset_size(2), 2)
        self.assertEqual(sort.subset_size(3), 1)
        self.assertEqual(sort.subset_size(4), 2)
        self.assertEqual(sort.subset_size(5), 0)
        self.assertEqual(sort.subset_size(6), 2)
        self.assertEqual(sort.subset_size(7), 0)
        self.assertEqual(sort.subset_size(8), 1)
        self.assertEqual(len(sort), 9)

    def test_as_grid_same_min_max(self):
        """ Tests that as_grid works when one of the features only takes
        one value """
        s1 = Subset(samples=np.array([[1, 0], [8, 0], [2, 0]]))
        s2 = Subset(samples=np.array([[2, 0], [3, 0], [5, 0]]))
        dataset = Dataset(n_features=2)
        dataset.extend([s1, s2])

        sort = as_grid(dataset, n_regions=3)
        sort.collect()

        self.assertEqual(len(sort), 9)
        self.assertTrue(sort.subset_size(2), 4)
        self.assertTrue(sort.subset_size(5), 1)
        self.assertTrue(sort.subset_size(8), 1)

    def test_as_grid_sparse(self):
        """ Tests that as_grid produces the same results with sparse and
        dense data structures."""
        file_ = "tests/files/libsvm/2"

        sparse = load_libsvm_file(file_, 10, 780)
        dense = load_libsvm_file(file_, 10, 780, store_sparse=False)
        grid_d, ind_d = as_grid(dense, 3, [128, 184], True)
        grid_sp, ind_sp = as_grid(sparse, 3, [128, 184], True)
        grid_sp.collect()
        grid_d.collect()

        self.assertEqual(len(grid_sp), len(grid_d))
        self.assertTrue(np.array_equal(ind_sp, ind_d))
        self.assertFalse(grid_d.sparse)
        self.assertTrue(grid_sp.sparse)

        for index in range(len(grid_sp)):
            samples_sp = grid_sp[index].samples.toarray()
            samples_d = grid_d[index].samples
            self.assertTrue(np.array_equal(samples_sp, samples_d))

    def test_shuffle_sparse(self):
        """ Tests that shuffle produces the same results with sparse and
        dense data structures. """
        file_ = "tests/files/libsvm/1"
        random_state = 170

        sparse = load_libsvm_file(file_, 10, 780)
        dense = load_libsvm_file(file_, 10, 780, store_sparse=False)
        shuf_d = shuffle(dense, random_state)
        shuf_sp = shuffle(sparse, random_state)
        shuf_sp.collect()
        shuf_d.collect()

        self.assertEqual(len(shuf_sp), len(shuf_d))
        self.assertFalse(shuf_d.sparse)
        self.assertTrue(shuf_sp.sparse)

        for index in range(len(shuf_sp)):
            samples_sp = shuf_sp[index].samples.toarray()
            samples_d = shuf_d[index].samples
            self.assertTrue(np.array_equal(samples_sp, samples_d))

    def test_resample(self):
        """ Tests resample with random data """
        dataset = load_data(np.random.random((1000, 500)), subset_size=100)

        r1 = resample(dataset, n_samples=500)
        r2 = resample(dataset, n_samples=500)
        r3 = resample(dataset, n_samples=500)
        r4 = resample(dataset, n_samples=500)

        self.assertEqual(r1.samples.shape[0], 500)
        self.assertEqual(r2.samples.shape[0], 500)
        self.assertEqual(r3.samples.shape[0], 500)
        self.assertEqual(r4.samples.shape[0], 500)
        self.assertEqual(len(r1), 10)
        self.assertEqual(len(r2), 10)
        self.assertEqual(len(r3), 10)
        self.assertEqual(len(r4), 10)
        self.assertFalse(np.array_equal(r1.samples, r2.samples))
        self.assertFalse(np.array_equal(r1.samples, r3.samples))
        self.assertFalse(np.array_equal(r1.samples, r4.samples))
        self.assertFalse(np.array_equal(r2.samples, r3.samples))
        self.assertFalse(np.array_equal(r2.samples, r4.samples))
        self.assertFalse(np.array_equal(r3.samples, r4.samples))

        r5 = resample(dataset, n_samples=500, random_state=5)
        r6 = resample(dataset, n_samples=500, random_state=5)

        self.assertTrue(np.array_equal(r5.samples, r6.samples))

    def test_resample_empty(self):
        """ Tests resample with empty subsets """
        dataset = load_data(np.random.random((1000, 500)), subset_size=100)
        r1 = resample(dataset, n_samples=1)

        self.assertEqual(r1.samples.shape[0], 1)
        self.assertEqual(len(r1), 1)

    def test_resample_sparse(self):
        """ Tests resample with sparse data """
        csr = csr_matrix(np.random.random((1000, 500)))
        dataset = load_data(csr, subset_size=100)

        r1 = resample(dataset, n_samples=500)
        r2 = resample(dataset, n_samples=500)

        self.assertEqual(r1.samples.shape[0], 500)
        self.assertEqual(r2.samples.shape[0], 500)
        self.assertEqual(len(r1), 10)
        self.assertEqual(len(r2), 10)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
