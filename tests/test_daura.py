import unittest

import numpy as np

import dislib as ds
from dislib.cluster import Daura


class GaussianMixtureTest(unittest.TestCase):

    def test_init_params(self):
        """Tests that Daura fit_predict"""
        dist = np.array([[0, 0.12, 0.19, 0.21, 0.29],
                         [0.12, 0, 0.10, 0.23, 0.22],
                         [0.19, 0.10, 0, 0.25, 0.21],
                         [0.21, 0.23, 0.25, 0, 0.15],
                         [0.29, 0.22, 0.21, 0.15, 0]])
        ds_dist = ds.array(dist, block_size=(2, 2))
        est = Daura(cutoff=0.17)
        clusters = est.fit_predict(ds_dist)
        expected_clusters = [[1, 0, 2], [3, 4]]
        self.assertEqual(clusters, expected_clusters)
