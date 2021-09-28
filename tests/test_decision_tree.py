import unittest

import numpy as np
from pycompss.api.api import compss_wait_on

import dislib as ds
import dislib.trees.decision_tree as dt
from dislib.trees import RfClassifierDataset, transform_to_rf_dataset


class DecisionTreeTest(unittest.TestCase):
    def test_decision_tree(self):
        x1 = np.array(
            [
                [0.3, -0.3],
                [0.4, -0.5],
                [0.5, -0.4],
                [0.3, 0.3],
                [0.4, 0.5],
                [0.5, 0.4],
                [-0.3, -0.3],
                [-0.4, -0.5],
                [-0.5, -0.4],
            ]
        )
        x2 = np.array([[0.4, -0.3], [0.4, 0.3], [-0.4, -0.3]])
        y1 = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        y2 = np.array([0, 1, 2])

        x1_ds = ds.array(x1, (3, 2))
        x2_ds = ds.array(x2, (3, 2))
        y1_ds = ds.array(y1[:, np.newaxis], (3, 1))

        data1 = transform_to_rf_dataset(
            x1_ds, y1_ds, RfClassifierDataset, features_file=True
        )

        # Model
        try_features = 2
        max_depth = np.inf
        distr_depth = 2
        sklearn_max = 1e8
        bootstrap = True
        seed = 0
        random_state = np.random.RandomState(seed)
        n_samples, n_features = x1.shape
        n_classes = np.bincount(y1).shape[0]
        features_mmap = x1.T

        # Test bootstrap
        sample1, y_s1 = compss_wait_on(
            dt._sample_selection(n_samples, y1, True, seed)
        )
        sample2, y_s2 = compss_wait_on(
            dt._sample_selection(n_samples, y1, False, seed)
        )
        self.assertTrue(
            np.array_equal(sample1, np.array([0, 2, 3, 3, 3, 4, 5, 5, 7]))
        )
        self.assertTrue(
            np.array_equal(sample2, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]))
        )
        self.assertTrue(
            np.array_equal(y_s1, np.array([0, 0, 1, 1, 1, 1, 1, 1, 2]))
        )
        self.assertTrue(
            np.array_equal(y_s2, np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]))
        )

        # Assert split wrapper
        sample, y_s = sample2, y_s2
        with self.assertRaises(ValueError):
            dt._split_node_wrapper(
                sample,
                n_features,
                y_s,
                n_classes,
                try_features,
                random_state,
                samples_file=None,
                features_file=None,
            )

        split = dt._split_node_wrapper(
            sample,
            n_features,
            y_s,
            n_classes,
            try_features,
            random_state,
            samples_file=data1.samples_path,
            features_file=data1.features_path,
        )
        split = compss_wait_on(split)
        node_info, left_group, y_l, right_group, y_r = split
        self.assertTrue(node_info.index in (0, 1))
        if node_info.index == 0:
            self.assertTrue(np.array_equal(left_group, np.array([6, 7, 8])))
            self.assertTrue(np.array_equal(y_l, np.array([2, 2, 2])))
            self.assertTrue(
                np.array_equal(right_group, np.array([0, 1, 2, 3, 4, 5]))
            )
            self.assertTrue(np.array_equal(y_r, np.array([0, 0, 0, 1, 1, 1])))
            self.assertAlmostEqual(node_info.value, 0.0)
            split_l = dt._compute_split(
                left_group,
                n_features,
                y_l,
                n_classes,
                try_features,
                features_mmap,
                random_state,
            )
            node_info, left_group, y_l, right_group, y_r = split_l
            self.assertTrue(np.array_equal(left_group, np.array([6, 7, 8])))
            self.assertTrue(np.array_equal(y_l, np.array([2, 2, 2])))
            self.assertTrue(np.array_equal(right_group, np.array([])))
            self.assertTrue(np.array_equal(y_r, np.array([])))
            self.assertTrue(
                np.array_equal(node_info.frequencies, np.array([0, 0, 3]))
            )
            self.assertEqual(node_info.size, 3)
            self.assertEqual(node_info.target, 2)
        elif node_info.index == 1:
            self.assertTrue(
                np.array_equal(left_group, np.array([0, 1, 2, 6, 7, 8]))
            )
            self.assertTrue(np.array_equal(y_l, np.array([0, 0, 0, 2, 2, 2])))
            self.assertTrue(np.array_equal(right_group, np.array([3, 4, 5])))
            self.assertTrue(np.array_equal(y_r, np.array([1, 1, 1])))
            self.assertAlmostEqual(node_info.value, 0.0)
            split_r = dt._compute_split(
                right_group,
                n_features,
                y_r,
                n_classes,
                try_features,
                features_mmap,
                random_state,
            )
            node_info, left_group, y_l, right_group, y_r = split_r
            self.assertTrue(np.array_equal(left_group, np.array([3, 4, 5])))
            self.assertTrue(np.array_equal(y_l, np.array([1, 1, 1])))
            self.assertTrue(np.array_equal(right_group, np.array([])))
            self.assertTrue(np.array_equal(y_r, np.array([])))
            self.assertTrue(
                np.array_equal(node_info.frequencies, np.array([0, 3, 0]))
            )
            self.assertEqual(node_info.size, 3)
            self.assertEqual(node_info.target, 1)

        # Test tree
        tree = dt.DecisionTreeClassifier(
            try_features,
            max_depth,
            distr_depth,
            sklearn_max,
            bootstrap,
            random_state,
        )
        tree.fit(data1)
        y_pred = compss_wait_on(tree.predict(x2_ds))
        self.assertTrue(np.array_equal(y_pred, y2))


def main():
    unittest.main()


if __name__ == "__main__":
    main()
