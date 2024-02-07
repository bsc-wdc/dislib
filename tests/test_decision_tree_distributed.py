import unittest

import numpy as np
from pycompss.api.api import compss_wait_on

import dislib as ds
import dislib.trees.distributed.decision_tree as dt_distributed
from dislib.trees.mmap.decision_tree import _InnerNodeInfo, _LeafInfo
from sklearn.datasets import make_regression, make_classification
from sklearn.metrics import r2_score, accuracy_score
from tests import BaseTimedTestCase
from dislib.trees.decision_tree import DecisionTreeClassifier, \
    DecisionTreeRegressor


class DecisionTreeTest(BaseTimedTestCase):
    def test_decision_tree_classifier(self):
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

        # Model
        try_features = 2
        max_depth = np.inf
        distr_depth = 1
        sklearn_max = 1e8
        bootstrap = True
        seed = 0
        random_state = np.random.RandomState(seed)
        n_classes = np.bincount(y1).shape[0]
        # Test bootstrap
        sample1 = dt_distributed._sample_selection(x1, random_state,
                                                   bootstrap=True)
        sample2 = dt_distributed._sample_selection(x1, random_state,
                                                   bootstrap=False)
        sample1 = compss_wait_on(sample1)
        sample2 = compss_wait_on(sample2)
        self.assertTrue(
            np.array_equal(sample1, np.array([0, 2, 3, 3, 3, 4, 5, 5, 7]))
        )
        self.assertTrue(
            np.array_equal(sample2, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]))
        )

        # Assert split wrapper
        sample = sample2
        rang_min = x1_ds.min()
        rang_max = x1_ds.max()

        split = dt_distributed._compute_split(
            x1_ds,
            y1_ds,
            n_classes,
            indexes_selected=sample,
            num_buckets=1,
            range_min=rang_min,
            range_max=rang_max,
            number_split_points=2,
            random_state=0,
        )
        node_info, data = split
        node_info = compss_wait_on(node_info)
        data = compss_wait_on(data)
        left_group = data[0][0]
        y_l = data[0][1]
        right_group = data[1][0]
        y_r = data[1][1]
        left_group_compare = np.block(left_group)
        y_l_compare = np.block(y_l)
        right_group_compare = np.block(right_group)
        y_r_compare = np.block(y_r)

        self.assertTrue(node_info.node_info.index in (0, 1))

        self.assertTrue(np.array_equal(left_group_compare,
                                       np.array([[0.3, -0.3],
                                                 [0.3, 0.3],
                                                 [-0.3, -0.3],
                                                 [-0.4, -0.5],
                                                 [-0.5, -0.4]]
                                                )))
        self.assertTrue(np.array_equal(y_l_compare,
                                       np.array([[0], [1], [2],
                                                 [2], [2]])))
        self.assertTrue(
            np.array_equal(right_group_compare, np.array([[0.4, -0.5],
                                                          [0.5, -0.4],
                                                          [0.4, 0.5],
                                                          [0.5, 0.4]]))
        )
        self.assertTrue(np.array_equal(y_r_compare, np.array([[0], [0],
                                                              [1], [1]])))
        self.assertAlmostEqual(node_info.node_info.value, 0.35)

        # Test tree
        tree = DecisionTreeClassifier(
            3,
            try_features,
            max_depth,
            distr_depth,
            sklearn_max,
            bootstrap,
            random_state,
            range_max=rang_max,
            range_min=rang_min,
            n_split_points=2,
            split_computation="raw",
            sync_after_fit=True,
            mmap=False,
        )
        tree.fit(x1_ds, y1_ds)
        y_pred = compss_wait_on(tree.predict(x2_ds))
        self.assertTrue(np.array_equal(np.argmax(y_pred, axis=1)[0], y2))
        y_pred_proba = tree.predict_proba(x2_ds, collect=True)
        self.assertTrue(np.array_equal(np.argmax(y_pred_proba, axis=1)[0], y2))

        random_state = np.random.RandomState(seed)

        tree = DecisionTreeClassifier(
            3,
            try_features,
            max_depth,
            distr_depth,
            sklearn_max,
            bootstrap,
            random_state,
            n_split_points="auto",
            split_computation="raw",
            sync_after_fit=True,
            mmap=False
        )

        tree.fit(x1_ds, y1_ds)
        y_pred = compss_wait_on(tree.predict(x2_ds))
        self.assertTrue(np.array_equal(np.argmax(y_pred, axis=1)[0], y2))
        y_pred_proba = tree.predict_proba(x2_ds, collect=True)
        self.assertTrue(np.array_equal(np.argmax(y_pred_proba, axis=1)[0], y2))

        random_state = np.random.RandomState(seed)

        x, y = make_classification(
            n_samples=3000,
            n_features=10,
            n_classes=3,
            n_informative=4,
            n_redundant=2,
            n_repeated=1,
            n_clusters_per_class=2,
            shuffle=True,
            random_state=0,
        )
        x_train = ds.array(x[::2], (500, 10))
        y_train = ds.array(y[::2][:, np.newaxis], (500, 1))

        tree = DecisionTreeClassifier(
            3,
            try_features,
            max_depth,
            distr_depth,
            sklearn_max,
            bootstrap,
            random_state,
            n_split_points="sqrt",
            split_computation="uniform_approximation",
            sync_after_fit=True,
            mmap=False
        )

        tree.fit(x1_ds, y1_ds)
        y_pred = compss_wait_on(tree.predict(x2_ds))
        self.assertTrue(np.array_equal(np.argmax(y_pred, axis=1)[0], y2))
        y_pred_proba = tree.predict_proba(x2_ds, collect=True)
        self.assertTrue(np.array_equal(np.argmax(y_pred_proba, axis=1)[0], y2))

        random_state = np.random.RandomState(seed)

        tree = DecisionTreeClassifier(
            3,
            try_features,
            max_depth,
            2,
            sklearn_max,
            bootstrap,
            random_state,
            n_split_points=0.444,
            split_computation="gaussian_approximation",
            sync_after_fit=True,
            mmap=False,
        )
        tree.fit(x_train[:100], y_train[:100])
        y_pred = tree.predict(x_train, collect=True)
        y_pred = np.argmax(np.vstack(y_pred), axis=1)
        y_train = y_train.collect()
        self.assertGreater(accuracy_score(y_train,
                                          y_pred), 0.6)
        y_pred_proba = tree.predict_proba(x_train, collect=True)
        y_pred_proba = np.argmax(np.vstack(y_pred_proba), axis=1)
        self.assertTrue(accuracy_score(y_train,
                                       y_pred_proba), 0.6)

    def test_decision_tree_regressor(self):
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

        # Model
        try_features = 2
        max_depth = np.inf
        distr_depth = 1
        sklearn_max = 1e8
        bootstrap = True
        seed = 0
        random_state = np.random.RandomState(seed)
        # Test bootstrap
        sample1 = dt_distributed._sample_selection(x1, random_state,
                                                   bootstrap=True)
        sample2 = dt_distributed._sample_selection(x1, random_state,
                                                   bootstrap=False)

        sample1 = compss_wait_on(sample1)
        sample2 = compss_wait_on(sample2)

        self.assertTrue(
            np.array_equal(sample1, np.array([0, 2, 3, 3, 3, 4, 5, 5, 7]))
        )
        self.assertTrue(
            np.array_equal(sample2, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]))
        )

        x1, y1 = make_regression(
            n_samples=1000,
            n_features=10,
            n_informative=4,
            shuffle=True,
            random_state=0,
        )

        x2 = x1[800:]
        x1 = x1[:800]
        y2 = y1[800:]
        y1 = y1[:800]

        x1_ds = ds.array(x1, (400, 10))
        x2_ds = ds.array(x2, (100, 10))

        y1_ds = ds.array(y1, (400, 1))
        rang_min = x1_ds.min()
        rang_max = x1_ds.max()

        # Test tree
        tree = DecisionTreeRegressor(
            try_features,
            max_depth,
            distr_depth,
            sklearn_max,
            bootstrap,
            random_state,
            range_max=rang_max,
            range_min=rang_min,
            n_split_points=2,
            split_computation="raw",
            sync_after_fit=True,
            mmap=False,
        )
        tree.fit(x1_ds, y1_ds)
        y_pred = compss_wait_on(tree.predict(x2_ds))
        y_pred = np.block(y_pred)
        self.assertGreater(r2_score(y_pred.flatten(), y2), 0.2)

        tree = DecisionTreeRegressor(
            try_features,
            max_depth,
            distr_depth,
            sklearn_max,
            bootstrap,
            random_state,
            n_split_points="auto",
            split_computation="raw",
            sync_after_fit=True,
            mmap=False,
        )
        tree.fit(x1_ds, y1_ds)
        y_pred = compss_wait_on(tree.predict(x2_ds))
        y_pred = np.block(y_pred)
        self.assertGreater(r2_score(y_pred.flatten(), y2), 0.3)

        tree = DecisionTreeRegressor(
            try_features,
            max_depth,
            distr_depth,
            sklearn_max,
            bootstrap,
            random_state,
            n_split_points="sqrt",
            split_computation="raw",
            sync_after_fit=True,
            mmap=False,
        )
        tree.fit(x1_ds, y1_ds)
        y_pred = compss_wait_on(tree.predict(x2_ds))
        y_pred = np.block(y_pred)
        self.assertTrue(isinstance(y_pred, np.ndarray))

    def test_objects(self):
        leaf_info = _LeafInfo(3, 0.7, 1)
        self.assertTrue(leaf_info.size == 3)
        self.assertTrue(leaf_info.frequencies == 0.7)
        self.assertTrue(leaf_info.target == 1)

        json_leaf_info = leaf_info.toJson()
        self.assertTrue(isinstance(json_leaf_info, dict))
        self.assertTrue(json_leaf_info['items']['size'] == 3)

    def test_functions_failure(self):
        solution = np.array([0])
        node_info = dt_distributed._NodeInfo()
        dt_distributed.generate_nodes_with_data_compressed_regression(
            node_info, solution,
            [np.array([0])], [np.array([0])],
            [np.array([0])], [np.array([0])],
            None, [3], 3)
        solution = compss_wait_on(solution)
        self.assertTrue(solution[0] == 1)
        value = dt_distributed.apply_split_points_to_blocks_regression(
            [np.array([0])],
            [np.array([0])],
            0, None,
            np.array([0, 1]))
        value = compss_wait_on(value)
        self.assertTrue(value[0] is None)
        value = dt_distributed.apply_split_points_to_blocks(
            [np.array([0])],
            [np.array([0])],
            0, None,
            np.array([0, 1]), 3)
        value = compss_wait_on(value)
        self.assertTrue(value[0] is None)

        value = dt_distributed.merge_partial_results_compute_mse_both_sides(
            [[None, None], [None, None]],
            [[None, None], [None, None]]
        )
        value = compss_wait_on(value)
        self.assertTrue(value[0] == np.array([np.inf]))
        value = dt_distributed.merge_partial_results_compute_mse_both_sides(
            [[np.array([0, 1])], [np.array([0, 1])]],
            [[None, None], [None, None]]
        )
        value = compss_wait_on(value)
        self.assertTrue(value[0] == np.array([np.inf]))

        value = dt_distributed.merge_partial_results_compute_mse_both_sides(
            [[np.array([0, 1])], [np.array([0, 1])]],
            [np.array([None, None])]
        )
        value = compss_wait_on(value)
        self.assertTrue(value[0] == np.array([np.inf]))

        optimal_split_point = dt_distributed.select_optimal_split_point(
            None, 0, [0, 1, 2], 0)
        optimal_split_point = compss_wait_on(optimal_split_point)
        self.assertTrue(compss_wait_on(optimal_split_point) is None)

        minimum_gini = dt_distributed.get_minimum_measure([[3, 5, 6]], 0)
        minimum_gini = compss_wait_on(minimum_gini)
        self.assertTrue(minimum_gini[-1] == 1)
        minimum_gini = dt_distributed.get_minimum_measure([[np.nan]], 0, False)
        minimum_gini = compss_wait_on(minimum_gini)
        self.assertTrue(minimum_gini[-1] == np.inf)

    def test_to_json(self):
        """Tests toJson method of _InnerNodeInfo and _LeafInfo"""
        node_info = _InnerNodeInfo(0, 0)
        self.assertTrue(isinstance(node_info.toJson(), dict))

        leaf_info = _LeafInfo(3, None, np.mean([0, 2, 4]))
        self.assertTrue(isinstance(leaf_info.toJson(), dict))


def main():
    unittest.main()


if __name__ == "__main__":
    main()
