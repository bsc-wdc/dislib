import unittest

import numpy as np
from pycompss.api.api import compss_wait_on
from scipy.sparse import csr_matrix, issparse
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

import dislib as ds
from dislib.cluster import DBSCAN
from dislib.cluster.dbscan.base import _arrange_samples, _rearrange_labels


class ArrangeTest(unittest.TestCase):

    def test_arrange(self):
        """ Tests the arrange method with toy data."""
        x = ds.array(np.array([[1, 1], [8, 8], [2, 5], [1, 7], [4, 4], [5, 9],
                               [4, 0], [8, 1], [7, 4]]), block_size=(3, 2))

        arranged, _, _ = _arrange_samples(x, n_regions=3)
        arranged = compss_wait_on(arranged)

        self.assertEqual(len(arranged), 9)

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

        self.assertTrue(np.array_equal(np.vstack(arranged), true_samples))

    def test_rearrange(self):
        """ Tests rearrange function """
        original = np.array([[7], [1], [5], [2], [0], [6]])
        x = ds.array(original, block_size=(3, 1))

        arranged, sorting, _ = _arrange_samples(x, n_regions=3)
        blocks = compss_wait_on(_rearrange_labels(arranged, sorting, 2))

        self.assertEqual(len(blocks), 2)
        self.assertTrue(np.array_equal(np.concatenate(blocks).flatten(),
                                       x.collect()))

    def test_arrange_indices(self):
        """ Tests that arrange returns correct indices with toy data.
        """
        x = ds.array(np.array([[1, 1], [8, 8], [2, 5], [1, 7], [4, 4], [5, 9],
                               [4, 0], [8, 1], [7, 4]]), block_size=(3, 2))

        arranged, sorting, _ = _arrange_samples(x, n_regions=3)

        arranged = compss_wait_on(arranged)
        arranged = np.vstack(arranged)
        sorting = np.asarray(compss_wait_on(sorting))

        indices = np.empty(x.shape[0], dtype=int)
        oldidx = 0

        # generate new indices based on sorting
        for j in range(sorting.shape[1]):
            for i in range(sorting.shape[0]):
                if sorting[i][j][0].size > 0:
                    newidx = sorting[i][j][0] + 3 * i
                    indices[newidx] = oldidx
                    oldidx += 1

        indices = np.squeeze(indices)

        self.assertTrue(np.array_equal(arranged[indices], x.collect()))

    def test_arrange_dimensions(self):
        """ Tests arrange method using a subset of the dimensions.
        """
        x = ds.array(np.array([[0, 1, 9], [8, 8, 2], [2, 5, 4], [1, 7, 6],
                               [4, 4, 2], [5, 9, 0], [4, 0, 1], [9, 1, 7],
                               [7, 4, 3]]), block_size=(3, 2))

        arranged, _, _ = _arrange_samples(x, n_regions=3, dimensions=[0])
        arranged = compss_wait_on(arranged)

        self.assertEqual(arranged[0].shape[0], 3)
        self.assertEqual(arranged[1].shape[0], 3)
        self.assertEqual(arranged[2].shape[0], 3)
        self.assertEqual(len(arranged), 3)

        arranged, _, _ = _arrange_samples(x, n_regions=3, dimensions=[0, 1])
        arranged = compss_wait_on(arranged)

        self.assertEqual(arranged[0].shape[0], 1)
        self.assertEqual(arranged[1].shape[0], 1)
        self.assertEqual(arranged[2].shape[0], 1)
        self.assertEqual(arranged[4].shape[0], 1)
        self.assertEqual(arranged[5].shape[0], 1)
        self.assertEqual(len(arranged), 9)

        arranged, _, _ = _arrange_samples(x, n_regions=3, dimensions=[1, 2])
        arranged = compss_wait_on(arranged)

        self.assertEqual(arranged[0].shape[0], 1)
        self.assertEqual(arranged[1].shape[0], 0)
        self.assertEqual(arranged[2].shape[0], 2)
        self.assertEqual(arranged[3].shape[0], 1)
        self.assertEqual(arranged[4].shape[0], 2)
        self.assertEqual(arranged[5].shape[0], 0)
        self.assertEqual(arranged[6].shape[0], 2)
        self.assertEqual(arranged[7].shape[0], 0)
        self.assertEqual(arranged[8].shape[0], 1)
        self.assertEqual(len(arranged), 9)

    def test_arrange_same_min_max(self):
        """ Tests arrange when one of the features only takes one value
        """
        x = ds.array(np.array([[1, 0], [8, 0], [2, 0],
                               [2, 0], [3, 0], [5, 0]]), block_size=(3, 2))

        arranged, _, _ = _arrange_samples(x, n_regions=3)
        arranged = compss_wait_on(arranged)

        self.assertEqual(len(arranged), 9)
        self.assertTrue(arranged[2].shape[0], 4)
        self.assertTrue(arranged[5].shape[0], 1)
        self.assertTrue(arranged[8].shape[0], 1)

    def test_arrange_sparse(self):
        """ Tests that arrange produces the same results with sparse and
        dense data structures."""
        file_ = "tests/files/libsvm/2"

        sparse, _ = ds.load_svmlight_file(file_, (10, 300), 780, True)
        dense, _ = ds.load_svmlight_file(file_, (10, 200), 780, False)

        arranged_d, sort_d, _ = _arrange_samples(dense, 3, [128, 184])
        arranged_sp, sort_sp, _ = _arrange_samples(sparse, 3, [128, 184])

        arranged_sp = compss_wait_on(arranged_sp)
        arranged_d = compss_wait_on(arranged_d)
        sort_d = compss_wait_on(sort_d)
        sort_sp = compss_wait_on(sort_sp)

        self.assertEqual(len(arranged_sp), len(arranged_d))
        self.assertFalse(issparse(arranged_d[0]))
        self.assertTrue(issparse(arranged_sp[0]))

        self.assertTrue(
            np.array_equal(np.concatenate(np.concatenate(sort_sp).flatten()),
                           np.concatenate(np.concatenate(sort_d).flatten())))

        for index in range(len(arranged_sp)):
            samples_sp = arranged_sp[index].toarray()
            samples_d = arranged_d[index]
            self.assertTrue(np.array_equal(samples_sp, samples_d))


class DBSCANTest(unittest.TestCase):
    def test_n_clusters_blobs(self):
        """ Tests that DBSCAN finds the correct number of clusters with blob
        data.
        """
        n_samples = 1500
        x, y = make_blobs(n_samples=n_samples, n_features=2, random_state=8)
        dbscan = DBSCAN(n_regions=1, eps=.3)
        x = StandardScaler().fit_transform(x)
        ds_x = ds.array(x, block_size=(300, 2))
        dbscan.fit(ds_x)
        self.assertEqual(dbscan.n_clusters, 3)

    def test_n_clusters_circles(self):
        """ Tests that DBSCAN finds the correct number of clusters with
        circle data.
        """
        n_samples = 1500
        x, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)
        dbscan = DBSCAN(n_regions=1, eps=.15)
        x = StandardScaler().fit_transform(x)
        ds_x = ds.array(x, block_size=(300, 2))
        dbscan.fit(ds_x)
        self.assertEqual(dbscan.n_clusters, 2)

    def test_n_clusters_moons(self):
        """ Tests that DBSCAN finds the correct number of clusters with
        moon data.
        """
        n_samples = 1500
        x, y = make_moons(n_samples=n_samples, noise=.05)
        dbscan = DBSCAN(n_regions=1, eps=.3)
        x = StandardScaler().fit_transform(x)
        ds_x = ds.array(x, block_size=(300, 2))
        dbscan.fit(ds_x)
        self.assertEqual(dbscan.n_clusters, 2)

    def test_n_clusters_aniso(self):
        """ Tests that DBSCAN finds the correct number of clusters with
        anisotropicly distributed data.
        """
        n_samples = 1500
        x, y = make_blobs(n_samples=n_samples, random_state=170)
        dbscan = DBSCAN(n_regions=1, eps=.15)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        x = np.dot(x, transformation)
        x = StandardScaler().fit_transform(x)
        ds_x = ds.array(x, block_size=(300, 2))
        y_pred = dbscan.fit_predict(ds_x).collect()
        true_sizes = {19, 496, 491, 488, 6}
        cluster_sizes = {y_pred[y_pred == -1].size,
                         y_pred[y_pred == 0].size,
                         y_pred[y_pred == 1].size,
                         y_pred[y_pred == 2].size,
                         y_pred[y_pred == 3].size}

        self.assertEqual(dbscan.n_clusters, 4)
        self.assertEqual(true_sizes, cluster_sizes)

    def test_n_clusters_blobs_max_samples(self):
        """ Tests that DBSCAN finds the correct number of clusters when
        defining max_samples with blob data.
        """
        n_samples = 1500
        x, y = make_blobs(n_samples=n_samples, n_features=2, random_state=8)
        dbscan = DBSCAN(n_regions=1, eps=.3, max_samples=500)
        x = StandardScaler().fit_transform(x)
        ds_x = ds.array(x, block_size=(300, 2))
        dbscan.fit(ds_x)
        self.assertEqual(dbscan.n_clusters, 3)

    def test_n_clusters_circles_max_samples(self):
        """ Tests that DBSCAN finds the correct number of clusters when
        defining max_samples with circle data.
        """
        n_samples = 1500
        x, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)
        dbscan = DBSCAN(n_regions=1, eps=.15, max_samples=500)
        x = StandardScaler().fit_transform(x)
        ds_x = ds.array(x, block_size=(300, 2))
        dbscan.fit(ds_x)
        self.assertEqual(dbscan.n_clusters, 2)

    def test_n_clusters_moons_max_samples(self):
        """ Tests that DBSCAN finds the correct number of clusters when
        defining max_samples with moon data.
        """
        n_samples = 1500
        x, y = make_moons(n_samples=n_samples, noise=.05)
        dbscan = DBSCAN(n_regions=1, eps=.3, max_samples=500)
        x = StandardScaler().fit_transform(x)
        ds_x = ds.array(x, block_size=(300, 2))
        dbscan.fit(ds_x)
        self.assertEqual(dbscan.n_clusters, 2)

    def test_n_clusters_aniso_max_samples(self):
        """ Tests that DBSCAN finds the correct number of clusters when
        defining max_samples with anisotropicly distributed data.
        """
        n_samples = 1500
        x, y = make_blobs(n_samples=n_samples, random_state=170)
        dbscan = DBSCAN(n_regions=1, eps=.15, max_samples=500)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        x = np.dot(x, transformation)
        x = StandardScaler().fit_transform(x)
        ds_x = ds.array(x, block_size=(300, 2))
        y_pred = dbscan.fit_predict(ds_x).collect()

        true_sizes = {19, 496, 491, 488, 6}
        cluster_sizes = {y_pred[y_pred == -1].size,
                         y_pred[y_pred == 0].size,
                         y_pred[y_pred == 1].size,
                         y_pred[y_pred == 2].size,
                         y_pred[y_pred == 3].size}

        self.assertEqual(dbscan.n_clusters, 4)
        self.assertEqual(true_sizes, cluster_sizes)

    def test_n_clusters_blobs_grid(self):
        """ Tests that DBSCAN finds the correct number of clusters when
        setting n_regions > 1 with blob data.
        """
        n_samples = 1500
        x, y = make_blobs(n_samples=n_samples, n_features=2, random_state=8)
        dbscan = DBSCAN(n_regions=4, eps=.3, max_samples=300)
        x = StandardScaler().fit_transform(x)
        ds_x = ds.array(x, block_size=(300, 2))
        dbscan.fit(ds_x)
        self.assertEqual(dbscan.n_clusters, 3)

    def test_n_clusters_circles_grid(self):
        """ Tests that DBSCAN finds the correct number of clusters when
        setting n_regions > 1 with circle data.
        """
        n_samples = 1500
        x, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)
        dbscan = DBSCAN(n_regions=4, eps=.15, max_samples=700)
        x = StandardScaler().fit_transform(x)
        ds_x = ds.array(x, block_size=(300, 2))
        dbscan.fit(ds_x)
        self.assertEqual(dbscan.n_clusters, 2)

    def test_n_clusters_moons_grid(self):
        """ Tests that DBSCAN finds the correct number of clusters when
        setting n_regions > 1 with moon data.
        """
        n_samples = 1500
        x, y = make_moons(n_samples=n_samples, noise=.05)
        dbscan = DBSCAN(n_regions=4, eps=.3, max_samples=600)
        x = StandardScaler().fit_transform(x)
        ds_x = ds.array(x, block_size=(300, 2))
        dbscan.fit(ds_x)
        self.assertEqual(dbscan.n_clusters, 2)

    def test_n_clusters_aniso_grid(self):
        """ Tests that DBSCAN finds the correct number of clusters when
        setting n_regions > 1 with anisotropicly distributed data.
        """
        n_samples = 1500
        x, y = make_blobs(n_samples=n_samples, random_state=170)
        dbscan = DBSCAN(n_regions=4, eps=.15, max_samples=500)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        x = np.dot(x, transformation)
        x = StandardScaler().fit_transform(x)
        ds_x = ds.array(x, block_size=(300, 2))
        y_pred = dbscan.fit_predict(ds_x).collect()
        true_sizes = {19, 496, 491, 488, 6}
        cluster_sizes = {y_pred[y_pred == -1].size,
                         y_pred[y_pred == 0].size,
                         y_pred[y_pred == 1].size,
                         y_pred[y_pred == 2].size,
                         y_pred[y_pred == 3].size}

        self.assertEqual(dbscan.n_clusters, 4)
        self.assertEqual(true_sizes, cluster_sizes)

    def test_zero_samples(self):
        """ Tests DBSCAN fit when some regions contain zero samples.
        """
        n_samples = 2
        x, y = make_blobs(n_samples=n_samples, n_features=2, random_state=8)
        dbscan = DBSCAN(n_regions=3, eps=.2, max_samples=100)
        x = StandardScaler().fit_transform(x)
        ds_x = ds.array(x, block_size=(2, 2))
        dbscan.fit(ds_x)
        self.assertEqual(dbscan.n_clusters, 0)

    def test_n_clusters_aniso_dimensions(self):
        """ Tests that DBSCAN finds the correct number of clusters when
        dimensions is not None.
        """
        n_samples = 1500
        x, y = make_blobs(n_samples=n_samples, random_state=170)
        dbscan = DBSCAN(n_regions=5, dimensions=[1], eps=.15)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        x = np.dot(x, transformation)
        x = StandardScaler().fit_transform(x)
        ds_x = ds.array(x, block_size=(300, 2))
        y_pred = dbscan.fit_predict(ds_x).collect()
        true_sizes = {19, 496, 491, 488, 6}
        cluster_sizes = {y_pred[y_pred == -1].size,
                         y_pred[y_pred == 0].size,
                         y_pred[y_pred == 1].size,
                         y_pred[y_pred == 2].size,
                         y_pred[y_pred == 3].size}

        self.assertEqual(dbscan.n_clusters, 4)
        self.assertEqual(true_sizes, cluster_sizes)

    def test_sparse(self):
        """ Tests that DBSCAN produces the same results with sparse and
        dense data.
        """
        n_samples = 1500
        x, y = make_blobs(n_samples=n_samples, random_state=170)
        dbscan = DBSCAN(n_regions=1, eps=.15)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        x = np.dot(x, transformation)
        x = StandardScaler().fit_transform(x)

        dense = ds.array(x, block_size=(300, 2))
        sparse = ds.array(csr_matrix(x), block_size=(300, 2))

        y_dense = dbscan.fit_predict(dense).collect()
        y_sparse = dbscan.fit_predict(sparse).collect()

        self.assertTrue(np.array_equal(y_dense, y_sparse))

    def test_small_cluster_1(self):
        """ Tests that DBSCAN can find clusters with less than min_samples. """
        x = np.array([[0, 0], [0, 1], [1, 0], [3, 0], [5.1, 0], [6, 0], [6, 1],
                      [10, 10]])
        ds_x = ds.array(x, block_size=(5, 2))

        # n_regions=1
        dbscan1 = DBSCAN(n_regions=1, eps=2.5, min_samples=4)
        dbscan1.fit(ds_x)
        self.assertEqual(dbscan1.n_clusters, 2)

    def test_small_cluster_2(self):
        """ Tests that DBSCAN can find clusters with less than min_samples. """
        x = np.array([[0, 0], [0, 1], [1, 0], [3, 0], [5.1, 0], [6, 0], [6, 1],
                      [10, 10]])
        ds_x = ds.array(x, block_size=(5, 2))

        # n_regions=10
        dbscan2 = DBSCAN(n_regions=10, eps=2.5, min_samples=4)
        dbscan2.fit(ds_x)
        self.assertEqual(dbscan2.n_clusters, 2)

    def test_cluster_between_regions_1(self):
        """ Tests that DBSCAN can find clusters between regions. """
        x = np.array([[0, 0], [3.9, 0], [4.1, 0], [4.1, 0.89], [4.1, 0.88],
                      [5.9, 0], [5.9, 0.89], [5.9, 0.88], [6.1, 0], [10, 10],
                      [4.6, 0], [5.4, 0]])
        ds_x = ds.array(x, block_size=(5, 2))

        dbscan = DBSCAN(n_regions=10, eps=0.9, min_samples=4)
        dbscan.fit(ds_x)
        self.assertEqual(dbscan.n_clusters, 1)

    def test_cluster_between_regions_2(self):
        """ Tests that DBSCAN can find clusters between regions. """
        x = np.array([[0, 0], [0.6, 0], [0.9, 0], [1.1, 0.2], [0.9, 0.6],
                      [1.1, 0.8], [1.4, 0.8], [2, 2]])
        ds_x = ds.array(x, block_size=(5, 2))

        dbscan = DBSCAN(n_regions=2, eps=0.5, min_samples=3)
        dbscan.fit(ds_x)
        self.assertEqual(dbscan.n_clusters, 1)

    def test_cluster_between_regions_3(self):
        """ Tests that DBSCAN can find clusters between regions. """
        x = np.array([[0, 0], [0.6, 0], [0.6, 0.01], [0.9, 0], [1.1, 0.2],
                      [1.4, 0.2], [1.4, 0.21], [0.9, 0.6], [0.6, 0.6],
                      [0.6, 0.61], [1.1, 0.8], [1.4, 0.8], [1.4, 0.81],
                      [2, 2]])
        ds_x = ds.array(x, block_size=(5, 2))

        dbscan = DBSCAN(n_regions=2, eps=0.5, min_samples=3)
        dbscan.fit(ds_x)
        self.assertEqual(dbscan.n_clusters, 1)

    def test_random_clusters_1(self):
        """ Tests DBSCAN on random data with multiple clusters. """
        # 1 dimension
        np.random.seed(1)
        x = np.random.uniform(0, 10, size=(1000, 1))
        ds_x = ds.array(x, block_size=(300, 1))
        dbscan = DBSCAN(n_regions=100, eps=0.1, min_samples=20)
        y = dbscan.fit_predict(ds_x).collect()

        self.assertEqual(dbscan.n_clusters, 18)
        self.assertEqual(np.count_nonzero(y == -1), 72)

    def test_random_clusters_2(self):
        """ Tests DBSCAN on random data with multiple clusters. """
        # 2 dimensions
        np.random.seed(2)
        x = np.random.uniform(0, 10, size=(1000, 2))
        ds_x = ds.array(x, block_size=(300, 2))
        dbscan = DBSCAN(n_regions=10, max_samples=10, eps=0.5, min_samples=10)
        y = dbscan.fit_predict(ds_x).collect()

        self.assertEqual(dbscan.n_clusters, 27)
        self.assertEqual(np.count_nonzero(y == -1), 206)

    def test_random_clusters_3(self):
        """ Tests DBSCAN on random data with multiple clusters. """
        # 3 dimensions
        np.random.seed(3)
        x = np.random.uniform(0, 10, size=(1000, 3))
        ds_x = ds.array(x, block_size=(300, 3))
        dbscan = DBSCAN(n_regions=10, dimensions=[0, 1],
                        eps=0.9, min_samples=4)
        y = dbscan.fit_predict(ds_x).collect()

        self.assertEqual(dbscan.n_clusters, 50)
        self.assertEqual(np.count_nonzero(y == -1), 266)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
