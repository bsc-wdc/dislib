from pycompss.api.task import task
from tests import BaseTimedTestCase
import numpy as np
import dislib as ds
import dislib.trees.nested.decision_tree as dt_nested
from dislib.trees.nested.tasks import filter_fragment
from pycompss.api.api import compss_wait_on
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import make_classification, make_regression


def test_decision_tree_classifier():
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
    sample1 = dt_nested._sample_selection(x1, random_state,
                                          bootstrap=True)
    sample2 = dt_nested._sample_selection(x1, random_state,
                                          bootstrap=False)
    condition = np.array_equal(sample1, np.array([0, 2, 3, 3, 3, 4, 5, 5, 7]))
    condition = condition and np.array_equal(sample2, np.array([0, 1, 2, 3,
                                                                4, 5, 6, 7,
                                                                8]))

    # Assert split wrapper
    sample = sample2
    rang_min = x1_ds.min()
    rang_max = x1_ds.max()
    rang_max._blocks = compss_wait_on(rang_max._blocks)
    rang_min._blocks = compss_wait_on(rang_min._blocks)

    split = dt_nested._compute_split(
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
    node_info, results_l, results_l_2, results_r, results_r_2 = split
    node_info = compss_wait_on(node_info)
    left_group = compss_wait_on(results_l)
    y_l = compss_wait_on(results_l_2)
    right_group = compss_wait_on(results_r)
    y_r = compss_wait_on(results_r_2)
    left_group_compare = np.block(left_group)
    y_l_compare = np.block(y_l)
    right_group_compare = np.block(right_group)
    y_r_compare = np.block(y_r)

    condition = condition and node_info.node_info.index in (0, 1)

    condition = condition and np.array_equal(left_group_compare,
                                             np.array([[0.3, -0.3],
                                                       [0.3, 0.3],
                                                       [-0.3, -0.3],
                                                       [-0.4, -0.5],
                                                       [-0.5, -0.4]]
                                                      ))

    condition = condition and np.array_equal(y_l_compare,
                                             np.array([[0], [1], [2],
                                                       [2], [2]]))

    condition = condition and np.array_equal(right_group_compare,
                                             np.array([[0.4, -0.5],
                                                       [0.5, -0.4],
                                                       [0.4, 0.5],
                                                       [0.5, 0.4]]))

    condition = condition and np.array_equal(y_r_compare,
                                             np.array([[0], [0],
                                                       [1], [1]]))

    condition = condition and np.isclose(node_info.node_info.value, 0.35)

    rang_min = x1_ds.min()
    rang_max = x1_ds.max()
    rang_max._blocks = compss_wait_on(rang_max._blocks)
    rang_min._blocks = compss_wait_on(rang_min._blocks)
    # Test tree
    tree = dt_nested.DecisionTreeClassifier(
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
    )
    tree.fit(x1_ds, y1_ds)

    y_pred = compss_wait_on(tree.predict(x2_ds))
    condition = condition and np.array_equal(np.argmax(y_pred, axis=1)[0], y2)
    y_pred_proba = compss_wait_on(tree.predict_proba(x2_ds))
    condition = condition and np.array_equal(
        np.argmax(y_pred_proba, axis=1)[0], y2)

    random_state = np.random.RandomState(seed)

    tree = dt_nested.DecisionTreeClassifier(
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
    )

    tree.fit(x1_ds, y1_ds)

    y_pred = compss_wait_on(tree.predict(x2_ds))
    condition = condition and np.array_equal(np.argmax(y_pred, axis=1)[0], y2)
    y_pred_proba = compss_wait_on(tree.predict_proba(x2_ds))
    condition = condition and np.array_equal(
        np.argmax(y_pred_proba, axis=1)[0], y2)

    random_state = np.random.RandomState(seed)

    x, y = make_classification(
        n_samples=300,
        n_features=10,
        n_classes=3,
        n_informative=4,
        n_redundant=2,
        n_repeated=1,
        n_clusters_per_class=2,
        shuffle=True,
        random_state=0,
    )
    x_train = ds.array(x[::2], (50, 10))
    y_train = ds.array(y[::2][:, np.newaxis], (50, 1))

    tree = dt_nested.DecisionTreeClassifier(
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
    )

    tree.fit(x1_ds, y1_ds)

    y_pred = compss_wait_on(tree.predict(x2_ds))
    condition = condition and np.array_equal(np.argmax(y_pred, axis=1)[0], y2)
    y_pred_proba = compss_wait_on(tree.predict_proba(x2_ds))
    condition = condition and np.array_equal(
        np.argmax(y_pred_proba, axis=1)[0], y2)

    random_state = np.random.RandomState(seed)

    tree = dt_nested.DecisionTreeClassifier(
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
    )
    tree.fit(x_train, y_train)
    y_pred = compss_wait_on(tree.predict(x_train))
    y_pred = np.argmax(np.vstack(y_pred), axis=1)
    y_train = y_train.collect()
    condition = condition and accuracy_score(y_train, y_pred) > 0.6
    y_pred_proba = compss_wait_on(tree.predict_proba(x_train))
    y_pred_proba = np.argmax(np.vstack(y_pred_proba), axis=1)
    condition = condition and accuracy_score(y_train,
                                             y_pred_proba) > 0.6
    return condition


def test_decision_tree_regressor():
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
    sample1 = dt_nested._sample_selection(x1, random_state,
                                          bootstrap=True)
    sample2 = dt_nested._sample_selection(x1, random_state,
                                          bootstrap=False)
    condition = np.array_equal(sample1, np.array([0, 2, 3, 3, 3, 4, 5, 5, 7]))
    condition = condition and np.array_equal(sample2,
                                             np.array([0, 1, 2, 3, 4,
                                                       5, 6, 7, 8]))

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
    rang_max._blocks = compss_wait_on(rang_max._blocks)
    rang_min._blocks = compss_wait_on(rang_min._blocks)

    # Test tree
    tree = dt_nested.DecisionTreeRegressor(
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
    )
    tree.fit(x1_ds, y1_ds)
    y_pred = compss_wait_on(tree.predict(x2_ds))
    y_pred = np.block(y_pred)
    condition = condition and r2_score(y_pred.flatten(), y2) > 0.1

    tree = dt_nested.DecisionTreeRegressor(
        try_features,
        max_depth,
        distr_depth,
        sklearn_max,
        bootstrap,
        random_state,
        n_split_points="auto",
        split_computation="uniform_approximation",
        sync_after_fit=True,
    )
    tree.fit(x1_ds, y1_ds)
    y_pred = compss_wait_on(tree.predict(x2_ds))
    y_pred = np.block(y_pred)
    condition = condition and r2_score(y_pred.flatten(), y2) > 0.15

    tree = dt_nested.DecisionTreeRegressor(
        try_features,
        max_depth,
        distr_depth,
        sklearn_max,
        bootstrap,
        random_state,
        n_split_points="sqrt",
        split_computation="gaussian_approximation",
        sync_after_fit=True,
    )
    tree.fit(x1_ds, y1_ds)
    y_pred = compss_wait_on(tree.predict(x2_ds))
    y_pred = np.block(y_pred)
    condition = condition and r2_score(y_pred.flatten(), y2) > 0.15

    tree = dt_nested.DecisionTreeRegressor(
        try_features,
        max_depth,
        distr_depth,
        sklearn_max,
        bootstrap,
        random_state,
        n_split_points=0.1,
        split_computation="gaussian_approximation",
        sync_after_fit=True,
    )
    tree.fit(x1_ds, y1_ds)
    y_pred = compss_wait_on(tree.predict(x2_ds))
    y_pred = np.block(y_pred)
    condition = condition and r2_score(y_pred.flatten(), y2) > 0.15
    return condition


def test_auxiliar_functions():
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
    y1 = np.array([0, 0, 0, 1, 1, 0, 1, 0, 1])
    right_x, right_y, x, y, aggregate_r, aggregate = \
        dt_nested.apply_split_points_to_blocks(x1, y1, 1,
                                               None, [2],
                                               2, np.array([]),
                                               np.array([0, 0]))

    condition = right_x is None
    condition = condition and right_y is None
    condition = condition and np.all(x == x1)
    condition = condition and np.all(y == y1)
    condition = condition and np.all(aggregate_r == np.array([]))
    condition = condition and np.all(aggregate == np.array([5, 4]))

    x1 = np.array(
        [
            [0.3, -0.3],
            [0.4, -0.5],
            [0.5, -0.4],
        ]
    )
    y1 = np.array([0, 0, 0])
    right_x, right_y, x, y, aggregate_r, aggregate = \
        dt_nested.apply_split_points_to_blocks(x1, y1, 1,
                                               None, [2],
                                               2, np.array([]),
                                               np.array([0, 0]))
    condition = condition and right_x is None
    condition = condition and right_y is None
    condition = condition and np.all(x == x1)
    condition = condition and np.all(y == y1)
    condition = condition and np.all(aggregate_r == np.array([]))
    condition = condition and np.all(aggregate == np.array([3, 0]))

    right_x, right_y, x, y, aggregate_r, aggregate = \
        dt_nested.apply_split_points_to_blocks(None, None, 1,
                                               1, [2],
                                               2, np.array([]),
                                               np.array([0, 0]))
    condition = condition and right_x is None
    condition = condition and right_y is None
    condition = condition and x is None
    condition = condition and y is None
    condition = condition and np.all(aggregate_r == np.array([]))
    condition = condition and np.all(aggregate == np.array([0, 0]))

    right_x, right_y, x, y, aggregate_r, \
        len_aggregate_r, aggregate_l, len_aggregate_l = \
        dt_nested.apply_split_points_to_blocks_regression(x1, y1, 1,
                                                          None, [2])
    condition = condition and right_x is None
    condition = condition and right_y is None
    condition = condition and np.all(x == x1)
    condition = condition and np.all(y == y1)
    condition = condition and np.all(aggregate_r == np.array([0]))
    condition = condition and np.all(len_aggregate_r == np.array([0]))
    condition = condition and np.all(aggregate_l == np.array([0]))
    condition = condition and np.all(len_aggregate_l == np.array([3]))

    optimal_split_point = dt_nested.select_optimal_split_point(None, 3,
                                                               4, 5)
    condition = condition and optimal_split_point is None

    gini_value_when_empty_list = dt_nested.get_minimum_measure([], 3)
    condition = condition and gini_value_when_empty_list[-1] == 1

    mse_value_when_empty_list = dt_nested.get_minimum_measure([],
                                                              3,
                                                              gini=False)
    condition = condition and mse_value_when_empty_list[-1] == np.inf

    mse_value, produces_split = dt_nested. \
        merge_partial_results_compute_mse_both_sides([[None], [None]],
                                                     np.array([]))
    mse_value = compss_wait_on(mse_value)
    produces_split = compss_wait_on(produces_split)
    condition = condition and np.all(mse_value == np.array([np.inf]))
    condition = condition and produces_split is False
    l_par_results = \
        [[[-4.93362945e+01, -2.91577501e+04, 5.91000000e+02],
          [-4.64000975e+01, -3.03920638e+04, 6.55000000e+02],
          [-3.81689727e+01, -2.71381396e+04, 7.11000000e+02]],
         [[-4.90482439e+01, -1.46654249e+04, 2.99000000e+02],
          [-4.67085998e+01, -1.50868777e+04, 3.23000000e+02],
          [-3.98015317e+01, -1.38111315e+04, 3.47000000e+02]]]
    mse_value, produces_split = dt_nested. \
        merge_partial_results_compute_mse_both_sides(l_par_results,
                                                     [[None], [None]])
    mse_value = compss_wait_on(mse_value)
    produces_split = compss_wait_on(produces_split)
    condition = condition and np.all(mse_value == np.array([np.inf]))
    condition = condition and produces_split is False

    mse_value, produces_split = dt_nested. \
        merge_partial_results_compute_mse_both_sides(l_par_results,
                                                     [None])
    mse_value = compss_wait_on(mse_value)
    produces_split = compss_wait_on(produces_split)
    condition = condition and np.all(mse_value == np.array([np.inf]))
    condition = condition and produces_split is False

    return condition


class RandomForestRegressorTest(BaseTimedTestCase):
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
        sample1 = dt_nested._sample_selection(x1, random_state,
                                              bootstrap=True)
        sample2 = dt_nested._sample_selection(x1, random_state,
                                              bootstrap=False)
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

        split = dt_nested._compute_split(
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
        node_info, results_l, results_l_2, results_r, results_r_2 = split
        node_info = compss_wait_on(node_info)
        left_group = results_l
        y_l = results_l_2
        right_group = results_r
        y_r = results_r_2
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
        tree = dt_nested.DecisionTreeClassifier(
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
        )
        tree.fit(x1_ds, y1_ds)
        y_pred = compss_wait_on(tree.predict(x2_ds))
        self.assertTrue(np.array_equal(np.argmax(y_pred, axis=1)[0], y2))
        y_pred_proba = tree.predict_proba(x2_ds)
        self.assertTrue(np.array_equal(np.argmax(y_pred_proba, axis=1)[0], y2))

        random_state = np.random.RandomState(seed)

        tree = dt_nested.DecisionTreeClassifier(
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
        )

        tree.fit(x1_ds, y1_ds)
        y_pred = compss_wait_on(tree.predict(x2_ds))
        self.assertTrue(np.array_equal(np.argmax(y_pred, axis=1)[0], y2))
        y_pred_proba = tree.predict_proba(x2_ds)
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

        tree = dt_nested.DecisionTreeClassifier(
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
        )

        tree.fit(x1_ds, y1_ds)
        y_pred = compss_wait_on(tree.predict(x2_ds))
        self.assertTrue(np.array_equal(np.argmax(y_pred, axis=1)[0], y2))
        y_pred_proba = tree.predict_proba(x2_ds)
        self.assertTrue(np.array_equal(np.argmax(y_pred_proba, axis=1)[0], y2))

        random_state = np.random.RandomState(seed)

        tree = dt_nested.DecisionTreeClassifier(
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
        )
        tree.fit(x_train[:100], y_train[:100])
        y_pred = tree.predict(x_train)
        y_pred = np.argmax(np.vstack(y_pred), axis=1)
        y_train = y_train.collect()
        self.assertGreater(accuracy_score(y_train,
                                          y_pred), 0.6)
        y_pred_proba = tree.predict_proba(x_train)
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
        sample1 = dt_nested._sample_selection(x1, random_state,
                                              bootstrap=True)
        sample2 = dt_nested._sample_selection(x1, random_state,
                                              bootstrap=False)
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
        tree = dt_nested.DecisionTreeRegressor(
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
        )
        tree.fit(x1_ds, y1_ds)
        y_pred = compss_wait_on(tree.predict(x2_ds))
        y_pred = np.block(y_pred)
        self.assertGreater(r2_score(y_pred.flatten(), y2), 0.15)

        tree = dt_nested.DecisionTreeRegressor(
            try_features,
            max_depth,
            distr_depth,
            sklearn_max,
            bootstrap,
            random_state,
            n_split_points="auto",
            split_computation="uniform_approximation",
            sync_after_fit=True,
        )
        tree.fit(x1_ds, y1_ds)
        y_pred = compss_wait_on(tree.predict(x2_ds))
        y_pred = np.block(y_pred)
        self.assertGreater(r2_score(y_pred.flatten(), y2), 0.15)

        tree = dt_nested.DecisionTreeRegressor(
            try_features,
            max_depth,
            distr_depth,
            sklearn_max,
            bootstrap,
            random_state,
            n_split_points="sqrt",
            split_computation="gaussian_approximation",
            sync_after_fit=True,
        )
        tree.fit(x1_ds, y1_ds)
        y_pred = compss_wait_on(tree.predict(x2_ds))
        y_pred = np.block(y_pred)
        self.assertGreater(r2_score(y_pred.flatten(), y2), 0.15)

        tree = dt_nested.DecisionTreeRegressor(
            try_features,
            max_depth,
            distr_depth,
            sklearn_max,
            bootstrap,
            random_state,
            n_split_points=0.1,
            split_computation="gaussian_approximation",
            sync_after_fit=True,
        )
        tree.fit(x1_ds, y1_ds)
        y_pred = compss_wait_on(tree.predict(x2_ds))
        y_pred = np.block(y_pred)
        self.assertGreater(r2_score(y_pred.flatten(), y2), 0.15)

    def test_auxiliar_functions(self):
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
        y1 = np.array([0, 0, 0, 1, 1, 0, 1, 0, 1])
        right_x, right_y, x, y, aggregate_r, aggregate = \
            dt_nested.apply_split_points_to_blocks(x1, y1, 1,
                                                   None, [2],
                                                   2, np.array([]),
                                                   np.array([0, 0]))
        self.assertTrue(right_x is None)
        self.assertTrue(right_y is None)
        self.assertTrue(np.all(x == x1))
        self.assertTrue(np.all(y == y1))
        self.assertTrue(np.all(aggregate_r == np.array([])))
        self.assertTrue(np.all(aggregate == np.array([5, 4])))

        x1 = np.array(
            [
                [0.3, -0.3],
                [0.4, -0.5],
                [0.5, -0.4],
            ]
        )
        y1 = np.array([0, 0, 0])
        right_x, right_y, x, y, aggregate_r, aggregate = \
            dt_nested.apply_split_points_to_blocks(x1, y1, 1,
                                                   None, [2],
                                                   2, np.array([]),
                                                   np.array([0, 0]))
        self.assertTrue(right_x is None)
        self.assertTrue(right_y is None)
        self.assertTrue(np.all(x == x1))
        self.assertTrue(np.all(y == y1))
        self.assertTrue(np.all(aggregate_r == np.array([])))
        self.assertTrue(np.all(aggregate == np.array([3, 0])))

        right_x, right_y, x, y, aggregate_r, aggregate = \
            dt_nested.apply_split_points_to_blocks(None, None, 1,
                                                   1, [2],
                                                   2, np.array([]),
                                                   np.array([0, 0]))
        self.assertTrue(right_x is None)
        self.assertTrue(right_y is None)
        self.assertTrue(x is None)
        self.assertTrue(y is None)
        self.assertTrue(np.all(aggregate_r == np.array([])))
        self.assertTrue(np.all(aggregate == np.array([0, 0])))

        right_x, right_y, x, y, aggregate_r, \
            len_aggregate_r, aggregate_l, len_aggregate_l = \
            dt_nested.apply_split_points_to_blocks_regression(x1, y1, 1,
                                                              None, [2])
        self.assertTrue(right_x is None)
        self.assertTrue(right_y is None)
        self.assertTrue(np.all(x == x1))
        self.assertTrue(np.all(y == y1))
        self.assertTrue(np.all(aggregate_r == np.array([0])))
        self.assertTrue(np.all(len_aggregate_r == np.array([0])))
        self.assertTrue(np.all(aggregate_l == np.array([0])))
        self.assertTrue(np.all(len_aggregate_l == np.array([3])))

        optimal_split_point = dt_nested.select_optimal_split_point(None, 3,
                                                                   4, 5)
        self.assertTrue(optimal_split_point is None)

        gini_value_when_empty_list = dt_nested.get_minimum_measure([], 3)
        self.assertTrue(gini_value_when_empty_list[-1] == 1)

        mse_value_when_empty_list = dt_nested.get_minimum_measure([],
                                                                  3,
                                                                  gini=False)
        self.assertTrue(mse_value_when_empty_list[-1] == np.inf)

        mse_value, produces_split = dt_nested.\
            merge_partial_results_compute_mse_both_sides([[None], [None]],
                                                         np.array([]))
        self.assertTrue(np.all(mse_value == np.array([np.inf])))
        self.assertTrue(produces_split is False)
        l_par_results = \
            [[[-4.93362945e+01, -2.91577501e+04,  5.91000000e+02],
              [-4.64000975e+01, -3.03920638e+04,  6.55000000e+02],
              [-3.81689727e+01, -2.71381396e+04,  7.11000000e+02]],
             [[-4.90482439e+01, -1.46654249e+04,  2.99000000e+02],
              [-4.67085998e+01, -1.50868777e+04,  3.23000000e+02],
              [-3.98015317e+01, -1.38111315e+04,  3.47000000e+02]]]
        mse_value, produces_split = dt_nested. \
            merge_partial_results_compute_mse_both_sides(l_par_results,
                                                         [[None], [None]])
        self.assertTrue(np.all(mse_value == np.array([np.inf])))
        self.assertTrue(produces_split is False)

        mse_value, produces_split = dt_nested.\
            merge_partial_results_compute_mse_both_sides(l_par_results,
                                                         [None])
        self.assertTrue(np.all(mse_value == np.array([np.inf])))
        self.assertTrue(produces_split is False)

        fragment_buckets = [[object()]]
        filter_fragment([], fragment_buckets, np.array([2, 3]),
                        3, range_min=[0], range_max=[1],
                        indexes_selected=np.array([0]))
        self.assertTrue(fragment_buckets == [[[]]])


@task()
def main():
    test = test_decision_tree_classifier()
    test2 = test_decision_tree_regressor()
    test3 = test_auxiliar_functions()
    test = test and test2 and test3
    if test:
        print("Result tests: Passed", flush=True)
    else:
        print("Result tests: Failed", flush=True)


if __name__ == "__main__":
    main()
