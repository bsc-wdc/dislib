import unittest
import uuid

import numpy as np
from hecuba import StorageNumpy, config
from sklearn.datasets import make_blobs

import dislib as ds
from dislib.cluster import KMeans


class HecubaDislibTest(unittest.TestCase):

    def test_iterate_rows_hecuba(self):
        """
        Tests iterating through the rows of the Hecuba array
        """
        config.session.execute("TRUNCATE TABLE hecuba.istorage")
        config.session.execute("DROP TABLE IF EXISTS hecuba_dislib.test_array")
        block_size = (20, 10)
        x = np.array([[i] * 10 for i in range(100)])
        storage_id = uuid.uuid4()
        persistent_data = StorageNumpy(input_array=x, name="hecuba_dislib.test_array", storage_id=storage_id)

        data = ds.hecuba_array(x=persistent_data, block_size=block_size)
        for i, chunk in enumerate(data._iterator(axis="rows")):
            r_data = chunk.collect()
            r_x = np.array([[j] * 10 for j in range(i * block_size[0], i * block_size[0] + block_size[0])])
            self.assertTrue(np.array_equal(r_data, r_x))

        self.assertEqual(i + 1, len(persistent_data) // block_size[0])

    def test_fit_predict(self):
        """ Tests fit_predict."""
        config.session.execute("TRUNCATE TABLE hecuba.istorage")
        config.session.execute("DROP TABLE IF EXISTS hecuba_dislib.test_array")

        x, y = make_blobs(n_samples=1500, random_state=170)
        x_filtered = np.vstack(
            (x[y == 0][:500], x[y == 1][:100], x[y == 2][:10]))
        storage_id = uuid.uuid4()

        x_train = ds.array(x_filtered, block_size=(300, 2))
        persistent_data = StorageNumpy(input_array=x_filtered, name="hecuba_dislib.test_array", storage_id=storage_id)
        x_train_hecuba = ds.hecuba_array(persistent_data, block_size=(300, 2))

        kmeans = KMeans(n_clusters=3, random_state=170)
        labels = kmeans.fit_predict(x_train).collect()

        kmeans = KMeans(n_clusters=3, random_state=170)
        h_labels = kmeans.fit_predict(x_train_hecuba).collect()

        centers = np.array([[-8.941375656533449, -5.481371322614891],
                            [-4.524023204953875, 0.06235042593214654],
                            [2.332994701667008, 0.37681003933082696]])

        self.assertTrue(np.allclose(centers, kmeans.centers))
        self.assertTrue(np.allclose(labels, h_labels))

        print("Nothing in fit_predict failed")
