import unittest
from unittest.mock import patch

import numpy as np
import sys

from dislib.cluster import KMeans
from dislib.utils import save_model, load_model


class SavingTest(unittest.TestCase):
    filepath = "tests/files/saving/kmeans.json"

    def test_errors(self):
        """Test that errors are raised"""
        km = KMeans(n_clusters=2, verbose=False)

        with patch(sys.modules["cbor"]) as mock_cbor:
            mock_cbor.return_value = None
            self.assertRaises(
                ModuleNotFoundError,
                save_model(km, self.filepath, save_format="json"),
            )


def main():
    unittest.main()


if __name__ == "__main__":
    main()
