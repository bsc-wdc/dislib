import unittest

import numpy as np
from scipy.sparse import csr_matrix

from dislib.data import Subset, Dataset, load_libsvm_file, load_data
from dislib.utils import shuffle, resample


class UtilsTest(unittest.TestCase):
    def test_shuffle(self):
        """ Tests shuffle for a given dataset and random_state. Tests that the
        shuffled dataset contains the same instances as the original dataset,
        that the position has changed for some instance, that the shuffled
        dataset is balanced, and that a K-fold partition of the shuffled
        dataset is balanced.
        """
        s1 = Subset(samples=np.array([[1, 1], [0, 0]]),
                    labels=np.array([0, 1]))
        s2 = Subset(samples=np.array([[4, 4], [3, 3], [2, 2]]),
                    labels=np.array([0, 1, 1]))
        s3 = Subset(samples=np.array([[8, 8], [7, 7], [6, 6], [5, 5]]),
                    labels=np.array([0, 1, 1, 1]))
        s4 = Subset(samples=np.array([[9, 9], [9, 8], [9, 7], [9, 6], [9, 5]]),
                    labels=np.array([0, 0, 1, 1, 0]))

        dataset = Dataset(2)
        dataset.extend([s1, s2, s3, s4])

        shuffled = shuffle(dataset, random_state=0)
        shuffled.collect()

        sizes_shuffled = shuffled.subsets_sizes()

        # Assert that at least one of the first 2 samples has changed
        self.assertFalse(np.array_equal([[1, 1], [0, 0]],
                                        shuffled[0].samples[0:2]))
        # Assert that the shuffled dataset has the same n_samples
        self.assertEqual(sum(sizes_shuffled), 14)
        # Assert that all the original instances are in the shuffled dataset
        shuffled_instances = set()
        for subset in shuffled:
            instances = [(tuple(sample), label) for sample, label in
                         zip(subset.samples, subset.labels)]
            shuffled_instances.update(instances)
        for subset in dataset:
            instances = [(tuple(sample), label) for sample, label in
                         zip(subset.samples, subset.labels)]
            self.assertTrue(shuffled_instances.issuperset(instances))
        # Assert that the shuffled dataset is balanced
        for size in sizes_shuffled:
            self.assertTrue(size == 3 or size == 4)
        # Assert that a 2-Fold dataset partition is balanced
        self.assertEqual(sum(sizes_shuffled[0:2]), sum(sizes_shuffled[2:4]))

    def test_shuffle_sparse(self):
        """ Tests that shuffle produces the same results with sparse and
        dense data structures. """
        file_ = "tests/files/libsvm/1"
        random_state = 170

        sparse = load_libsvm_file(file_, 10, 780)
        dense = load_libsvm_file(file_, 10, 780, store_sparse=False)
        shuf_d = shuffle(dense, random_state=random_state)
        shuf_sp = shuffle(sparse, random_state=random_state)
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
