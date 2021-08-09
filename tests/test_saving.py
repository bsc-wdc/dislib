import unittest
import json
import os
import shutil
from dislib.cluster import KMeans
from dislib.cluster import DBSCAN
import dislib.utils.saving as saving

DIRPATH = "tests/files/saving"


class SavingTest(unittest.TestCase):
    def setUp(self) -> None:
        os.makedirs(DIRPATH, exist_ok=True)
        return super().setUp()

    def tearDown(self) -> None:
        shutil.rmtree(DIRPATH)
        return super().tearDown()

    def test_errors(self):
        filepath = os.path.join(DIRPATH, "model.json")

        # Models
        km = KMeans(n_clusters=2)
        km2 = KMeans(n_clusters=10)
        dbscan = DBSCAN()

        # Import error
        cbor2_module = saving.cbor2
        saving.cbor2 = None
        with self.assertRaises(ModuleNotFoundError):
            saving.save_model(km, filepath, save_format="cbor")
        with self.assertRaises(ModuleNotFoundError):
            saving.load_model(filepath, load_format="cbor")
        saving.cbor2 = cbor2_module

        # Saving model not implemented
        with self.assertRaises(NotImplementedError):
            saving.save_model(dbscan, filepath)

        # Wrong save format
        with self.assertRaises(ValueError):
            saving.save_model(km, filepath, save_format="xxxx")

        # Overwrite
        saving.save_model(km, filepath, save_format="json")
        with open(filepath, "r") as f:
            json_str = f.read()
        saving.save_model(
            km2, filepath, overwrite=False, save_format="json"
        )
        with open(filepath, "r") as f:
            json_str2 = f.read()
        self.assertEqual(json_str, json_str2)

        # Wrong load format
        with self.assertRaises(ValueError):
            saving.load_model(filepath, load_format="xxxx")

        # Load model not implemented
        model_data = {"model_name": "dbscan"}
        with open(filepath, "w") as f:
            json.dump(model_data, f)
        with self.assertRaises(NotImplementedError):
            saving.load_model(filepath, load_format="json")

        # Not JSON serializable
        setattr(km, "n_clusters", dbscan)
        with self.assertRaises(TypeError):
            saving.save_model(km, filepath, save_format="json")

        # Not dict or list
        with self.assertRaises(TypeError):
            saving._sync_obj(km)


def main():
    unittest.main()


if __name__ == "__main__":
    main()
