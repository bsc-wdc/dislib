import gc
import os
import unittest

import numpy as np

os.environ["CONTACT_NAMES"] = "cassandra_container"
from hecuba import config
from pycompss.api.api import compss_wait_on
from sklearn.datasets import make_blobs

from pycompss.api.task import task    # Import @task decorator
from pycompss.api.parameter import *  # Import parameter metadata for the @task decorator

from pycompss.util.serialization.serializer import serialize_to_file
from pycompss.util.serialization.serializer import deserialize_from_file

import dislib as ds
from dislib.cluster import KMeans
from dislib.decomposition import PCA
from dislib.neighbors import NearestNeighbors
from dislib.regression import LinearRegression
import time


def equal(arr1, arr2):
    equal = not (arr1 != arr2).any()

    if not equal:
        print("\nArr1: \n%s" % arr1)
        print("Arr2: \n%s" % arr2)

    return equal


class HecubaTest(unittest.TestCase):

    def test_already_persistent(self):
        """ Tests K-means fit_predict and compares the result with regular
            ds-arrays, using an already persistent Hecuba array """
        config.session.execute("TRUNCATE TABLE hecuba.istorage")
        config.session.execute("DROP KEYSPACE IF EXISTS hecuba_dislib")
        x, y = make_blobs(n_samples=1500, random_state=170)
        x_filtered = np.vstack(
            (x[y == 0][:500], x[y == 1][:100], x[y == 2][:10]))

        block_size = (x_filtered.shape[0] // 10, x_filtered.shape[1])
        print("shape del objeo")
        print(x_filtered.shape)
        x_train = ds.array(x_filtered, block_size=block_size)
        x_train_hecuba = ds.array(x=x_filtered,
                                  block_size=block_size)
        x_train_hecuba.make_persistent(name="hecuba_dislib.test_array")

        # ensure that all data is released from memory
        blocks = x_train_hecuba._blocks
        for block in blocks:
            del block
        del x_train_hecuba
        gc.collect()

        x_train_hecuba = ds.load_from_hecuba(name="hecuba_dislib.test_array",
                                             block_size=block_size)

        # kmeans = KMeans(n_clusters=3, random_state=170)
        # labels = kmeans.fit_predict(x_train).collect()
        print("tipo de dato")
        print(x_train_hecuba)
        #kmeans2 = KMeans(n_clusters=3, random_state=170)

        serialize_to_file(x_train_hecuba, "test_ob")
        x_train_hecuba2=deserialize_from_file("test_ob")
        print(x_train_hecuba2)

        #h_labels = kmeans2.fit_predict(x_train_hecuba).collect()

        # self.assertTrue(np.allclose(kmeans.centers, kmeans2.centers))
        # self.assertTrue(np.allclose(labels, h_labels))