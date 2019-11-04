import unittest

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans as SKMeans
from sklearn.datasets import make_blobs

import dislib as ds
from dislib.cluster import KMeans


class KMeansTest(unittest.TestCase):
    def test_init_params(self):
        """ Tests that KMeans object correctly sets the initialization
        parameters """
        n_clusters = 2
        max_iter = 1
        tol = 1e-4
        seed = 666
        arity = 2
        init = "random"

        km = KMeans(n_clusters=n_clusters, max_iter=max_iter, tol=tol,
                    arity=arity, random_state=seed)

        expected = (n_clusters, init, max_iter, tol, arity)
        real = (km.n_clusters, km.init, km.max_iter, km.tol, km.arity)
        self.assertEqual(expected, real)

    def test_fit(self):
        """ Tests that the fit method returns the expected centers using toy
        data.
        """
        arr = np.array([[1, 2], [2, 1], [-1, -2], [-2, -1]])
        x = ds.array(arr, block_size=(2, 2))

        km = KMeans(n_clusters=2, random_state=666, verbose=False)
        km.fit(x)

        expected_centers = np.array([[1.5, 1.5], [-1.5, -1.5]])

        self.assertTrue((km.centers == expected_centers).all())

    def test_predict(self):
        """ Tests that labels are correctly predicted using toy data. """
        p1, p2, p3, p4 = [1, 2], [2, 1], [-1, -2], [-2, -1]

        arr1 = np.array([p1, p2, p3, p4])
        x = ds.array(arr1, block_size=(2, 2))

        km = KMeans(n_clusters=2, random_state=666)
        km.fit(x)

        p5, p6 = [10, 10], [-10, -10]

        arr2 = np.array([p1, p2, p3, p4, p5, p6])
        x_test = ds.array(arr2, block_size=(2, 2))

        labels = km.predict(x_test).collect()
        expected_labels = np.array([0, 0, 1, 1, 0, 1])

        self.assertTrue(np.array_equal(labels, expected_labels))

    def test_fit_predict(self):
        """ Tests fit_predict."""
        x, y = make_blobs(n_samples=1500, random_state=170)
        x_filtered = np.vstack(
            (x[y == 0][:500], x[y == 1][:100], x[y == 2][:10]))

        x_train = ds.array(x_filtered, block_size=(300, 2))

        kmeans = KMeans(n_clusters=3, random_state=170)
        labels = kmeans.fit_predict(x_train).collect()

        skmeans = SKMeans(n_clusters=3, random_state=170)
        sklabels = skmeans.fit_predict(x_filtered)

        centers = np.array([[-8.941375656533449, -5.481371322614891],
                            [-4.524023204953875, 0.06235042593214654],
                            [2.332994701667008, 0.37681003933082696]])

        self.assertTrue(np.allclose(centers, kmeans.centers))
        self.assertTrue(np.allclose(labels, sklabels))

    def test_sparse(self):
        """ Tests K-means produces the same results using dense and sparse
        data structures. """
        file_ = "tests/files/libsvm/2"

        x_sp, _ = ds.load_svmlight_file(file_, (10, 300), 780, True)
        x_ds, _ = ds.load_svmlight_file(file_, (10, 300), 780, False)

        kmeans = KMeans(random_state=170)

        y_sparse = kmeans.fit_predict(x_sp).collect()
        sparse_c = kmeans.centers.toarray()

        kmeans = KMeans(random_state=170)

        y_dense = kmeans.fit_predict(x_ds).collect()
        dense_c = kmeans.centers

        self.assertTrue(np.allclose(sparse_c, dense_c))
        self.assertTrue(np.array_equal(y_sparse, y_dense))

    def test_init(self):
        # With dense data
        x, y = make_blobs(n_samples=1500, random_state=170)
        x_filtered = np.vstack(
            (x[y == 0][:500], x[y == 1][:100], x[y == 2][:10]))
        x_train = ds.array(x_filtered, block_size=(300, 2))

        init = np.random.random((5, 2))
        km = KMeans(n_clusters=5, init=init)
        km.fit(x_train)

        self.assertTrue(np.array_equal(km.init, init))
        self.assertFalse(np.array_equal(km.centers, init))

        # With sparse data
        x_sp = ds.array(csr_matrix(x_filtered), block_size=(300, 2))
        init = csr_matrix(np.random.random((5, 2)))

        km = KMeans(n_clusters=5, init=init)
        km.fit(x_sp)

        self.assertTrue(np.array_equal(km.init.toarray(), init.toarray()))
        self.assertFalse(np.array_equal(km.centers.toarray(), init.toarray()))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
