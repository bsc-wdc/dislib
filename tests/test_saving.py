import unittest
import sys
import json

import numpy as np

# Workaround to mask cbor2
if True:
    sys.modules["cbor2"] = None
from dislib.cluster import KMeans
from dislib.cluster import DBSCAN
from dislib.classification import RandomForestClassifier
from dislib.data import array
from dislib.utils.saving import save_model, load_model, _sync_obj

from sklearn.datasets import make_classification
from pycompss.api.api import compss_wait_on


class SavingTest(unittest.TestCase):
    filepath = "tests/files/saving/model.json"

    def test_errors(self):
        """Test that errors are raised"""

        # Models
        km = KMeans(n_clusters=2)
        km2 = KMeans(n_clusters=10)
        dbscan = DBSCAN()
        rf = RandomForestClassifier()
        x, y = make_classification(
            n_samples=3000,
            n_features=10,
            n_classes=3,
            n_informative=4,
            n_clusters_per_class=2,
        )
        x_train = array(x[: len(x) // 2], (300, 10))
        y_train = array(y[: len(y) // 2][:, np.newaxis], (300, 1))
        rf.fit(x_train, y_train)

        # Import error
        with self.assertRaises(ModuleNotFoundError):
            save_model(km, self.filepath, save_format="cbor")
        with self.assertRaises(ModuleNotFoundError):
            load_model(self.filepath, load_format="cbor")

        # Saving model not implemented
        with self.assertRaises(NotImplementedError):
            save_model(dbscan, self.filepath)

        # Wrong save format
        with self.assertRaises(ValueError):
            save_model(km, self.filepath, save_format="xxxx")

        # Overwrite
        save_model(km, self.filepath, save_format="json")
        with open(self.filepath, "r") as f:
            json_str = f.read()
        save_model(km2, self.filepath, overwrite=False, save_format="json")
        with open(self.filepath, "r") as f:
            json_str2 = f.read()
        self.assertEqual(json_str, json_str2)

        # Wrong load format
        with self.assertRaises(ValueError):
            load_model(self.filepath, load_format="xxxx")

        # Load model not implemented
        model_data = {"model_name": "dbscan"}
        with open(self.filepath, "w") as f:
            json.dump(model_data, f)
        with self.assertRaises(NotImplementedError):
            load_model(self.filepath, load_format="json")

        # Not JSON serializable
        setattr(km, "n_clusters", dbscan)
        with self.assertRaises(TypeError):
            save_model(km, self.filepath, save_format="json")

        # Not dict or list
        with self.assertRaises(TypeError):
            _sync_obj(km)

        # Future not synchronized
        compss_wait_on(rf.trees[0].try_features)
        with self.assertRaises(TypeError):
            save_model(rf, self.filepath, save_format="json")


def main():
    unittest.main()


if __name__ == "__main__":
    main()
