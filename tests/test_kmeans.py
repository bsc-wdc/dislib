import unittest

import numpy as np
from sklearn.datasets import make_blobs

from dislib.cluster import KMeans
from dislib.data import Dataset
from dislib.data import Subset
from dislib.data import load_data, load_libsvm_file


class KMeansTest(unittest.TestCase):
    def test_init_params(self):
        """ Tests that KMeans object correctly sets the initialization
        parameters """
        n_clusters = 2
        max_iter = 1
        tol = 1e-4
        seed = 666
        arity = 2
        km = KMeans(n_clusters=n_clusters, max_iter=max_iter, tol=tol,
                    arity=arity, random_state=seed)

        expected = (n_clusters, max_iter, tol, seed, arity)
        real = (km._n_clusters, km._max_iter, km._tol, km._random_state,
                km._arity)
        self.assertEqual(expected, real)

    def test_fit(self):
        """ Tests that the fit method returns the expected centers using a
        toy dataset. """
        dataset = Dataset(n_features=2)

        dataset.append(Subset(np.array([[1, 2], [2, 1]])))
        dataset.append(Subset(np.array([[-1, -2], [-2, -1]])))

        km = KMeans(n_clusters=2, random_state=666, verbose=False)
        km.fit(dataset)

        expected_centers = np.array([[1.5, 1.5], [-1.5, -1.5]])

        self.assertTrue((km.centers == expected_centers).all())

    def test_predict(self):
        """ Tests that labels are correctly predicted using a toy dataset. """
        dataset = Dataset(n_features=2)
        p1, p2, p3, p4 = [1, 2], [2, 1], [-1, -2], [-2, -1]

        dataset.append(Subset(np.array([p1, p2])))
        dataset.append(Subset(np.array([p3, p4])))

        km = KMeans(n_clusters=2, random_state=666)
        km.fit(dataset)

        p5, p6 = [10, 10], [-10, -10]

        test_set = load_data(np.array([p1, p2, p3, p4, p5, p6]), subset_size=2)
        km.predict(test_set)

        l1, l2, l3, l4, l5, l6 = test_set.labels

        self.assertTrue(l1 == l2 == l5 == 0)
        self.assertTrue(l3 == l4 == l6 == 1)

    def test_fit_predict(self):
        """ Tests that fit_predict computes the right centers, and that the
        number of labels predicted is the expected one using a toy dataset."""
        x, y = make_blobs(n_samples=1500, random_state=170)
        x_filtered = np.vstack(
            (x[y == 0][:500], x[y == 1][:100], x[y == 2][:10]))

        dataset = load_data(x_filtered, subset_size=300)

        kmeans = KMeans(n_clusters=3, random_state=170)
        kmeans.fit_predict(dataset)
        labels = dataset.labels

        centers = np.array([[-8.941375656533449, -5.481371322614891],
                            [-4.524023204953875, 0.06235042593214654],
                            [2.332994701667008, 0.37681003933082696]])

        self.assertTrue((centers == kmeans.centers).all())
        self.assertEqual(labels.size, 610)

    def test_sparse(self):
        """ Tests K-means produces the same results using dense and sparse
        data structures. """
        file_ = "tests/files/libsvm/2"

        sparse = load_libsvm_file(file_, 10, 780)
        dense = load_libsvm_file(file_, 10, 780, store_sparse=False)

        kmeans = KMeans(random_state=170)
        kmeans.fit_predict(sparse)
        sparse_c = kmeans.centers.toarray()

        kmeans.fit_predict(dense)
        dense_c = kmeans.centers

        self.assertTrue(np.allclose(sparse_c, dense_c))
        self.assertTrue(np.array_equal(sparse.labels, dense.labels))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
