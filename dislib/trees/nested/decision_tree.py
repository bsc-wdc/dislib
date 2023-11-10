import math
import numpy as np
from sklearn.tree import DecisionTreeClassifier as SklearnDTClassifier
from sklearn.tree import DecisionTreeRegressor as SklearnDTRegressor
from pycompss.api.parameter import COLLECTION_IN, IN
from sklearn.utils import check_random_state
from pycompss.api.api import compss_delete_object, compss_wait_on
from dislib.data.array import Array
from pycompss.api.task import task
from pycompss.api.constraint import constraint
import scipy
from dislib.trees.nested.terasort import terasort


class BaseDecisionTree:
    """Base class for distributed decision trees.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    def __init__(
            self,
            try_features,
            max_depth,
            distr_depth,
            sklearn_max,
            bootstrap,
            random_state,
            base_node,
            base_tree,
            n_classes=None,
            range_min=None,
            range_max=None,
            n_split_points="auto",
            split_computation="raw",
            sync_after_fit=True,
    ):
        self.n_classes = n_classes
        self.try_features = try_features
        self.max_depth = max_depth
        self.sklearn_max = sklearn_max
        self.distr_depth = distr_depth
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.base_node = base_node
        self.base_tree = base_tree

        self.n_features = None

        self.tree = None
        self.nodes_info = None
        self.range_min = range_min
        self.range_max = range_max
        self.n_split_points = n_split_points
        self.split_computation = split_computation
        self.sync_after_fit = sync_after_fit

    @constraint(computing_units="${ComputingUnits}")
    @task()
    def fit(self, x, y):
        """Fits the DecisionTree.

        Parameters
        ----------
        x : ds-array
            Samples of the dataset.
        y: ds-array
            Labels of the dataset.
        """
        if self.range_max is None:
            self.range_max = x.max()
        if self.range_min is None:
            self.range_min = x.min()
        self.range_max._blocks = compss_wait_on(self.range_max._blocks)
        self.range_min._blocks = compss_wait_on(self.range_min._blocks)
        if self.n_split_points == "auto":
            self.n_split_points = int(math.log(x.shape[0]))
        elif self.n_split_points == "sqrt":
            self.n_split_points = int(math.sqrt(x.shape[0]))
        elif self.n_split_points < 1 and self.n_split_points > 0:
            self.n_split_points = int(self.n_split_points * x.shape[0])
        elif isinstance(self.n_split_points, int):
            pass
        self.total_length = x.shape[0]
        self.number_attributes = x.shape[1]
        self.tree = self.base_node()
        branches = [[x, y, self.tree]]
        nodes_info = []
        selection = _sample_selection(x,
                                      random_state=self.random_state,
                                      bootstrap=self.bootstrap)
        num_buckets = x._n_blocks[0] * x._n_blocks[1]
        for i in range(self.distr_depth):
            branches_pair = []
            for idx, branch_data in enumerate(branches):
                x, y, actual_node = branch_data
                node_info, results_l, results_l_2, results_r, results_r_2 = (
                    _compute_split(
                        x, y, n_classes=self.n_classes,
                        range_min=self.range_min,
                        range_max=self.range_max,
                        num_buckets=int(num_buckets/(i+1)),
                        m_try=self.try_features,
                        number_attributes=self.number_attributes,
                        indexes_selected=selection,
                        number_split_points=int(self.n_split_points*(i+1)),
                        split_computation=self.split_computation,
                        random_state=self.random_state))
                actual_node.content = int(math.pow(2, int(i)) - 1 + idx)
                actual_node.left = self.base_node()
                actual_node.right = self.base_node()
                splits_computed = []
                splits_computed.append(results_l)
                splits_computed.append(results_l_2)
                splits_computed.append(actual_node.left)
                branches_pair.append(splits_computed)
                splits_computed = []
                splits_computed.append(results_r)
                splits_computed.append(results_r_2)
                splits_computed.append(actual_node.right)
                branches_pair.append(splits_computed)
                nodes_info.append(node_info)
            branches = branches_pair
        for branch in branches:
            x, y, actual_node = branch
            actual_node = construct_subtree(x, y, actual_node,
                                            self.try_features,
                                            self.distr_depth,
                                            max_depth=self.max_depth,
                                            random_state=self.random_state)
            nodes_info.append(actual_node)
        nodes_info = compss_wait_on(nodes_info)
        self.nodes_info = nodes_info

    @constraint(computing_units="${ComputingUnits}")
    @task(returns=list)
    def predict(self, x):
        """Predicts target values or classes for the given samples using
        a fitted tree.

        Parameters
        ----------
        x_row : ds-array
            A row block of samples.

        Returns
        -------
        predicted : ndarray
            An array with the predicted classes or values for the given
            samples. For classification, the values are codes of the fitted
            dislib.classification.rf.data.RfDataset. The returned object can
            be a pycompss.runtime.Future object.
        """
        assert self.tree is not None, "The decision tree is not fitted."

        block_predictions = []
        for x_block in x._blocks:
            block_predictions.append(_predict_tree_class(x_block,
                                                         self.nodes_info,
                                                         0, self.n_classes))
        return block_predictions


class DecisionTreeClassifier(BaseDecisionTree):
    """A distributed decision tree classifier.

    Parameters
    ----------
    try_features : int
        The number of features to consider when looking for the best split.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires
        to effectively inspect more than ``try_features`` features.
    max_depth : int
        The maximum depth of the tree. If np.inf, then nodes are expanded
        until all leaves are pure.
    distr_depth : int
        Number of levels of the tree in which the nodes are split in a
        distributed way.
    bootstrap : bool
        Randomly select n_instances samples with repetition (used in random
        forests).
    random_state : RandomState instance
        The random number generator.

    Attributes
    ----------
    n_features : int
        The number of features of the dataset. It can be a
        pycompss.runtime.Future object.
    n_classes : int
        The number of classes of this RfDataset. It can be a
        pycompss.runtime.Future object.
    tree : None or _Node
        The root node of the tree after the tree is fitted.
    nodes_info : None or list of _InnerNodeInfo and _LeafInfo
        List of the node information for the nodes of the tree in the same
        order as obtained in the fit() method, up to ``distr_depth`` depth.
        After fit(), it is a pycompss.runtime.Future object.
    subtrees : None or list of _Node
        List of subtrees of the tree at ``distr_depth`` depth  obtained in the
        fit() method. After fit(), it is a list of pycompss.runtime.Future
        objects.

    Methods
    -------
    fit(dataset)
        Fits the DecisionTreeClassifier.
    predict(x_row)
        Predicts classes for the given samples using a fitted tree.
    predict_proba(x_row)
        Predicts class probabilities for the given smaples using a fitted tree.

    """

    def __init__(
            self,
            n_classes,
            try_features,
            max_depth,
            distr_depth,
            sklearn_max,
            bootstrap,
            random_state,
            range_min=None,
            range_max=None,
            n_split_points="auto",
            split_computation="raw",
            sync_after_fit=True,
    ):
        super().__init__(
            try_features,
            max_depth,
            distr_depth,
            sklearn_max,
            bootstrap,
            random_state,
            _ClassificationNode,
            SklearnDTClassifier,
            n_classes=n_classes,
            range_min=range_min,
            range_max=range_max,
            n_split_points=n_split_points,
            split_computation=split_computation,
            sync_after_fit=sync_after_fit,
        )

    @constraint(computing_units="${ComputingUnits}")
    @task(returns=1)
    def predict_proba(self, x):
        """Predicts class probabilities for a row block using a fitted tree.

                Parameters
                ----------
                x_row : ds-array
                    A row block of samples.

                Returns
                -------
                predicted_proba : list
                    A list with the predicted probabilities
                    for the given samples.
                    It contains a numpy array (if collect=True)
                    or Future object (if collect=False) for each of the blocks
                    in the ds-array to predict.
                    Thus the length of the list is the same
                    as the number of blocks the ds-array contains.
                    The shape inside each prediction is (len(x.reg_shape[0]),
                     self.n_classes).
                    The returned object can be a
                    pycompss.runtime.Future object.
                """

        assert self.tree is not None, "The decision tree is not fitted."

        block_predictions = []
        for x_block in x._blocks:
            block_predictions.append(_predict_proba_tree(x_block,
                                                         self.nodes_info,
                                                         0, self.n_classes))
        block_predictions = compss_wait_on(block_predictions)
        return block_predictions


class DecisionTreeRegressor(BaseDecisionTree):
    """A distributed decision tree regressor.

    Parameters
    ----------
    try_features : int
        The number of features to consider when looking for the best split.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires
        to effectively inspect more than ``try_features`` features.
    max_depth : int
        The maximum depth of the tree. If np.inf, then nodes are expanded
        until all leaves are pure.
    distr_depth : int
        Number of levels of the tree in which the nodes are split in a
        distributed way.
    bootstrap : bool
        Randomly select n_instances samples with repetition (used in random
        forests).
    random_state : RandomState instance
        The random number generator.

    Attributes
    ----------
    n_features : int
        The number of features of the dataset. It can be a
        pycompss.runtime.Future object.
    tree : None or _Node
        The root node of the tree after the tree is fitted.
    nodes_info : None or list of _InnerNodeInfo and _LeafInfo
        List of the node information for the nodes of the tree in the same
        order as obtained in the fit() method, up to ``distr_depth`` depth.
        After fit(), it is a pycompss.runtime.Future object.
    subtrees : None or list of _Node
        List of subtrees of the tree at ``distr_depth`` depth  obtained in the
        fit() method. After fit(), it is a list of pycompss.runtime.Future
        objects.

    Methods
    -------
    fit(dataset)
        Fits the DecisionTreeRegressor.
    predict(x_row)
        Predicts target values for the given samples using a fitted tree.
    """

    def __init__(
            self,
            try_features,
            max_depth,
            distr_depth,
            sklearn_max,
            bootstrap,
            random_state,
            range_min=None,
            range_max=None,
            n_split_points="auto",
            split_computation="raw",
            sync_after_fit=True
    ):
        super().__init__(
            try_features,
            max_depth,
            distr_depth,
            sklearn_max,
            bootstrap,
            random_state,
            _RegressionNode,
            SklearnDTRegressor,
            n_classes=None,
            range_min=range_min,
            range_max=range_max,
            n_split_points=n_split_points,
            split_computation=split_computation,
            sync_after_fit=sync_after_fit,
        )

    @constraint(computing_units="${ComputingUnits}")
    @task()
    def fit(self, x, y):
        """Fits the DecisionTreeRegressor.

        Parameters
        ----------
        x : ds-array
            Samples of the dataset.
        y: ds-array
            Labels of the dataset.
        """
        if self.range_max is None:
            self.range_max = x.max()
        if self.range_min is None:
            self.range_min = x.min()
        self.range_max._blocks = compss_wait_on(self.range_max._blocks)
        self.range_min._blocks = compss_wait_on(self.range_min._blocks)
        if self.n_split_points == "auto":
            self.n_split_points = int(math.log(x.shape[0]))
        elif self.n_split_points == "sqrt":
            self.n_split_points = int(math.sqrt(x.shape[0]))
        elif self.n_split_points < 1 and self.n_split_points > 0:
            self.n_split_points = int(self.n_split_points*x.shape[0])
        elif isinstance(self.n_split_points, int):
            pass
        self.total_length = x.shape[0]
        self.number_attributes = x.shape[1]
        self.tree = self.base_node()
        branches = [[x, y, self.tree]]
        nodes_info = []
        selection = _sample_selection(x, random_state=self.random_state,
                                      bootstrap=self.bootstrap)
        num_buckets = x._n_blocks[0] * x._n_blocks[1]
        for i in range(self.distr_depth):
            branches_pair = []
            for idx, branch_data in enumerate(branches):
                x, y, actual_node = branch_data
                node_info, results_l, results_l_2, results_r, results_r_2 = (
                    _compute_split_regressor(
                        x, y, range_min=self.range_min,
                        range_max=self.range_max,
                        num_buckets=int(
                            num_buckets/(i+1)),
                        m_try=self.try_features,
                        number_attributes=self.number_attributes,
                        indexes_selected=selection,
                        number_split_points=int(self.n_split_points*(i+1)),
                        split_computation=self.split_computation,
                        random_state=self.random_state))
                actual_node.content = int(math.pow(2, int(i)) - 1 + idx)
                actual_node.left = self.base_node()
                actual_node.right = self.base_node()
                splits_computed = [results_l, results_l_2, actual_node.left]
                branches_pair.append(splits_computed)
                splits_computed = [results_r, results_r_2, actual_node.right]
                branches_pair.append(splits_computed)
                nodes_info.append(node_info)
            branches = branches_pair
        for branch in branches:
            x, y, actual_node = branch
            actual_node = construct_subtree(x, y, actual_node,
                                            self.try_features,
                                            self.distr_depth,
                                            max_depth=self.max_depth,
                                            random_state=self.random_state)
            nodes_info.append(actual_node)
        nodes_info = compss_wait_on(nodes_info)
        self.nodes_info = nodes_info


@constraint(computing_units="${ComputingUnits}")
@task(returns=5, priority=True)
def _compute_split_regressor(x, y, num_buckets=4,
                             range_min=0, range_max=1, indexes_selected=None,
                             number_attributes=2, m_try=2,
                             number_split_points=100, split_computation="raw",
                             random_state=1):
    if x[0] is None:
        return None, [None], [None], [None], [None]
    indexes_to_try = []
    random_state = check_random_state(random_state)
    untried_indices = np.setdiff1d(np.arange(number_attributes),
                                   indexes_to_try)
    index_selection = _feature_selection(
        untried_indices, m_try, random_state
    )
    indexes_to_try.append(index_selection)
    node_info = _NodeInfo()
    final_rights_x = [object()]
    final_rights_y = [object()]
    final_lefts_x = [object()]
    final_lefts_y = [object()]
    if num_buckets < 1:
        num_buckets = 1
    tried_indices = []
    for _ in range(number_attributes):
        untried_indices = np.setdiff1d(np.arange(number_attributes),
                                       tried_indices)
        index_selection = _feature_selection(
            untried_indices, m_try, random_state
        )
        results = terasort(x, index_selection, range_min=range_min,
                           range_max=range_max,
                           indexes_selected=indexes_selected,
                           num_buckets=num_buckets)
        split_points_per_attribute = []
        for i in range(len(results[0])):
            split_points_per_attribute.append(
                get_split_point_various_attributes_bucket(
                    results[:, i], number_split_points=number_split_points,
                    split_computation=split_computation))
        [compss_delete_object(b) for results_2 in results for b in results_2]
        del results
        split_points_per_attribute = compss_wait_on(split_points_per_attribute)
        partial_results_left = []
        partial_results_right = []
        for idx, split_values in enumerate(split_points_per_attribute):
            partial_results_left.append([])
            partial_results_right.append([])
            if isinstance(x, Array):
                for index_blocks, block_s in enumerate(zip(
                        x._blocks, y._blocks)):
                    idx_selected = indexes_selected[
                        indexes_selected < (index_blocks + 1) *
                        x._reg_shape[0]]
                    block_x, block_y = block_s
                    left_class, right_class = classes_per_split(
                        block_x, block_y, split_values, index_selection,
                        idx_selected[idx_selected >= (index_blocks) *
                                     x._reg_shape[0]] % x._reg_shape[0],
                        regression=True)
                    partial_results_left[idx].append(left_class)
                    partial_results_right[idx].append(right_class)
                    del idx_selected
            else:
                for block_x, block_y in zip(x, y):
                    left_class, right_class = classes_per_split(
                        block_x, block_y, split_values, index_selection,
                        np.array([0]), regression=True)
                    partial_results_left[idx].append(left_class)
                    partial_results_right[idx].append(right_class)
        partial_results_right_array = np.array(compss_wait_on(
            partial_results_right))
        partial_results_left_array = np.array(compss_wait_on(
            partial_results_left))
        store_mse_values = []
        evaluation_of_splits = []
        for idx in range(partial_results_right_array.shape[0]):
            for j in range(partial_results_right_array.shape[2]):
                global_gini_values, produces_split = (
                    merge_partial_results_compute_mse_both_sides(
                        partial_results_left_array[idx, :, j],
                        partial_results_right_array[idx, :, j]))
                store_mse_values.append(global_gini_values)
                evaluation_of_splits.append(produces_split)

        store_mse_values = compss_wait_on(store_mse_values)
        evaluation_of_splits = compss_wait_on(evaluation_of_splits)
        del partial_results_right_array
        del partial_results_left_array
        [compss_delete_object(result) for results in
         partial_results_right for result in results]
        [compss_delete_object(result) for results in
         partial_results_left for result in results]
        best_attribute, position_m_g, bucket_minimum_gini, minimum_mse = (
            get_minimum_measure(store_mse_values, m_try, gini=False))
        optimal_split_point = select_optimal_split_point(
            best_attribute, position_m_g, split_points_per_attribute,
            bucket_minimum_gini)
        compss_delete_object(position_m_g)
        compss_delete_object(bucket_minimum_gini)
        compss_delete_object(*evaluation_of_splits)
        compss_delete_object(*store_mse_values)
        compss_delete_object(*split_points_per_attribute)
        rights_x = []
        rights_y = []
        lefts_x = []
        lefts_y = []
        right_sums = []
        right_lengths = []
        left_sums = []
        left_lengths = []
        if isinstance(x, Array):
            for block_x, block_y in zip(x._blocks, y._blocks):
                (right_x, right_y, left_x, left_y, compress_r,
                 len_compress_r, compress_l, len_compress_l) = (
                    apply_split_points_to_blocks_regression(
                        block_x, block_y, best_attribute,
                        optimal_split_point, index_selection))
                rights_x.append([right_x])
                rights_y.append([right_y])
                lefts_x.append([left_x])
                lefts_y.append([left_y])
                right_sums.append(compress_r)
                right_lengths.append(len_compress_r)
                left_sums.append(compress_l)
                left_lengths.append(len_compress_l)
        else:
            for block_x, block_y in zip(x, y):
                (right_x, right_y, left_x, left_y, compress_r,
                 len_compress_r, compress_l, len_compress_l) = (
                    apply_split_points_to_blocks_regression(
                        block_x, block_y, best_attribute,
                        optimal_split_point, index_selection))
                rights_x.append([right_x])
                rights_y.append([right_y])
                lefts_x.append([left_x])
                lefts_y.append([left_y])
                right_sums.append(compress_r)
                right_lengths.append(len_compress_r)
                left_sums.append(compress_l)
                left_lengths.append(len_compress_l)
            [compss_delete_object(x_data[0]) for x_data in x]
            [compss_delete_object(y_data[0]) for y_data in y]
        final_rights_x[0] = rights_x
        final_rights_y[0] = rights_y
        final_lefts_x[0] = lefts_x
        final_lefts_y[0] = lefts_y
        if (np.sum(left_lengths) + np.sum(right_lengths)) <= 4:
            node_info.set(_compute_leaf_info((np.sum(left_sums) +
                                              np.sum(right_sums)) /
                                             (np.sum(left_lengths) +
                                              np.sum(right_lengths)), None,
                          occurrences=np.sum(left_lengths)
                                             +
                                             np.sum(right_lengths)
                                             ))
        elif np.sum(right_lengths) == 0:
            node_info.set(_compute_leaf_info(
                (np.sum(left_sums) + np.sum(right_sums)) /
                (np.sum(left_lengths) + np.sum(right_lengths)), None,
                occurrences=np.sum(left_lengths) + np.sum(right_lengths)))
        elif np.sum(left_lengths) == 0:
            node_info.set(_compute_leaf_info(
                (np.sum(left_sums) + np.sum(right_sums)) /
                (np.sum(left_lengths) + np.sum(right_lengths)), None,
                occurrences=np.sum(left_lengths) + np.sum(right_lengths)))
        elif best_attribute is None:
            node_info.set(_compute_leaf_info(
                (np.sum(left_sums) + np.sum(right_sums)) /
                (np.sum(left_lengths) + np.sum(right_lengths)), None,
                occurrences=np.sum(left_lengths) + np.sum(right_lengths)))
        else:
            node_info.set(_InnerNodeInfo(index_selection[
                                             best_attribute],
                                         optimal_split_point))
            del right_sums
            del right_lengths
            del left_lengths
            del left_sums
            del minimum_mse
            del optimal_split_point
            del best_attribute
            return (node_info, final_lefts_x[0], final_lefts_y[0],
                    final_rights_x[0], final_rights_y[0])
        del right_sums
        del right_lengths
        del left_lengths
        del left_sums
        del minimum_mse
        del optimal_split_point
        del best_attribute
        tried_indices.extend(index_selection)
        if len(tried_indices) == number_attributes:
            break
    return node_info, [None], [None], [None], [None]


@constraint(computing_units="${ComputingUnits}")
@task(returns=5, priority=True)
def _compute_split(x, y, n_classes=None, num_buckets=4,
                   range_min=0, range_max=1,
                   indexes_selected=None, number_attributes=2, m_try=2,
                   number_split_points=100,
                   split_computation="raw", random_state=None):
    if x[0] is None:
        return None, [None], [None], [None], [None]
    indexes_to_try = []
    random_state = check_random_state(random_state)
    untried_indices = np.setdiff1d(np.arange(number_attributes),
                                   indexes_to_try)
    index_selection = _feature_selection(
        untried_indices, m_try, random_state
    )
    indexes_to_try.append(index_selection)
    node_info = _NodeInfo()
    final_rights_x = [object()]
    final_rights_y = [object()]
    final_lefts_x = [object()]
    final_lefts_y = [object()]
    tried_indices = []
    if num_buckets < 1:
        num_buckets = 2
    for _ in range(number_attributes):
        untried_indices = np.setdiff1d(np.arange(
            number_attributes), tried_indices)
        index_selection = _feature_selection(
            untried_indices, m_try, random_state
        )
        results = terasort(x, index_selection, range_min=range_min,
                           range_max=range_max,
                           indexes_selected=indexes_selected,
                           num_buckets=num_buckets)
        split_points_per_attribute = []
        for i in range(len(
                results[0])):
            split_points_per_attribute.append(
                get_split_point_various_attributes_bucket(
                    results[:, i], number_split_points=number_split_points,
                    split_computation=split_computation))
        [compss_delete_object(b) for results_2 in results for b in results_2]
        del results
        split_points_per_attribute = compss_wait_on(
            split_points_per_attribute)
        partial_results_left = []
        partial_results_right = []
        for idx, split_values in enumerate(split_points_per_attribute):
            partial_results_left.append([])
            partial_results_right.append([])
            if isinstance(x, Array):
                for index_blocks, block_s in enumerate(
                        zip(x._blocks, y._blocks)):
                    idx_selected = indexes_selected[
                        indexes_selected < (index_blocks + 1) *
                        x._reg_shape[0]]
                    block_x, block_y = block_s
                    left_class, right_class = classes_per_split(
                        block_x, block_y, split_values, index_selection,
                        idx_selected[idx_selected >= (index_blocks) *
                                     x._reg_shape[0]] % x._reg_shape[0])
                    partial_results_left[idx].append(left_class)
                    partial_results_right[idx].append(right_class)
            else:
                for block_x, block_y in zip(x, y):
                    left_class, right_class = classes_per_split(
                        block_x, block_y, split_values,
                        index_selection, np.array([0]))
                    partial_results_left[idx].append(left_class)
                    partial_results_right[idx].append(right_class)
        partial_results_right_array = np.array(compss_wait_on(
            partial_results_right))
        partial_results_left_array = np.array(compss_wait_on(
            partial_results_left))
        store_gini_values = []
        evaluation_of_splits = []
        for idx in range(partial_results_right_array.shape[0]):
            for j in range(partial_results_right_array.shape[2]):
                global_gini_values, produces_split = (
                    merge_partial_results_compute_gini_both_sides(
                        partial_results_left_array[idx, :, j],
                        partial_results_right_array[idx, :, j],
                        n_classes))
                store_gini_values.append(global_gini_values)
                evaluation_of_splits.append(produces_split)
        store_gini_values = compss_wait_on(store_gini_values)
        evaluation_of_splits = compss_wait_on(evaluation_of_splits)
        del partial_results_right_array
        del partial_results_left_array
        [compss_delete_object(result) for results in
         partial_results_right for result in results]
        [compss_delete_object(result) for results in
         partial_results_left for result in results]
        best_attribute, position_m_g, bucket_minimum_gini, minimum_ginis = (
            get_minimum_measure(store_gini_values,
                                len(index_selection),
                                gini=True))
        optimal_split_point = select_optimal_split_point(
            best_attribute, position_m_g, split_points_per_attribute,
            bucket_minimum_gini)
        compss_delete_object(position_m_g)
        compss_delete_object(bucket_minimum_gini)
        compss_delete_object(minimum_ginis)
        compss_delete_object(*evaluation_of_splits)
        compss_delete_object(*store_gini_values)
        compss_delete_object(*split_points_per_attribute)
        rights_x = []
        rights_y = []
        lefts_x = []
        lefts_y = []
        aggregate = np.zeros(n_classes, dtype=np.int64)
        aggregate_r = np.zeros(n_classes, dtype=np.int64)
        if isinstance(x, Array):
            for block_x, block_y in zip(x._blocks, y._blocks):
                right_x, right_y, left_x, left_y, aggregate_r, aggregate = (
                    apply_split_points_to_blocks(
                        block_x, block_y, best_attribute,
                        optimal_split_point, index_selection, n_classes,
                        aggregate, aggregate_r))
                rights_x.append([right_x])
                rights_y.append([right_y])
                lefts_x.append([left_x])
                lefts_y.append([left_y])
        else:
            for block_x, block_y in zip(x, y):
                right_x, right_y, left_x, left_y, aggregate_r, aggregate = (
                    apply_split_points_to_blocks(
                        block_x, block_y, best_attribute, optimal_split_point,
                        index_selection, n_classes, aggregate, aggregate_r))
                rights_x.append([right_x])
                rights_y.append([right_y])
                lefts_x.append([left_x])
                lefts_y.append([left_y])
            [compss_delete_object(x_data[0]) for x_data in x]
            [compss_delete_object(y_data[0]) for y_data in y]
        final_rights_x[0] = rights_x
        final_rights_y[0] = rights_y
        final_lefts_x[0] = lefts_x
        final_lefts_y[0] = lefts_y

        if (np.sum(aggregate) + np.sum(aggregate)) <= 4:
            node_info.set(_compute_leaf_info(aggregate +
                                             aggregate_r, n_classes))
        elif np.sum(aggregate_r) == 0:
            node_info.set(_compute_leaf_info(aggregate + aggregate_r,
                                             n_classes))
        elif np.sum(aggregate) == 0:
            node_info.set(_compute_leaf_info(aggregate + aggregate_r,
                                             n_classes))
        elif best_attribute is None:
            node_info.set(_compute_leaf_info(aggregate + aggregate_r,
                                             n_classes))
        else:
            node_info.set(_InnerNodeInfo(index_selection[best_attribute],
                                         optimal_split_point))
            del best_attribute
            del evaluation_of_splits
            del optimal_split_point
            del aggregate
            del aggregate_r
            del minimum_ginis
            return (node_info, final_lefts_x[0], final_lefts_y[0],
                    final_rights_x[0], final_rights_y[0])
        del best_attribute
        del evaluation_of_splits
        del optimal_split_point
        del aggregate
        del aggregate_r
        del minimum_ginis
        tried_indices.extend(index_selection)
        if len(tried_indices) == number_attributes:
            break
    return node_info, [None], [None], [None], [None]


def _feature_selection(untried_indices, m_try, random_state):
    selection_len = min(m_try, len(untried_indices))
    return random_state.choice(
        untried_indices, size=selection_len, replace=False
    )


def _compute_leaf_info(y_s, n_classes, occurrences=None):
    if n_classes is not None:
        y_s = y_s.squeeze()
        mode = np.argmax(y_s)
        return _LeafInfo(np.sum(y_s), y_s, mode)
    else:
        return _LeafInfo(occurrences, None, y_s)


def _predict_tree_class(x, node, node_content_num, n_classes=None,
                        rights=0, depth=0):
    if node_content_num == 0:
        node_content_num = node_content_num + 1
    else:
        node_content_num = node_content_num * 2 + rights
    x = np.block(x)
    node_content = node[node_content_num - 1]
    if len(x) == 0:
        if n_classes is not None:
            return np.empty((0, n_classes), dtype=np.float64)
        else:
            return np.empty((0,), dtype=np.float64)
    if isinstance(node_content, _NodeInfo):
        if isinstance(node_content.get(), _LeafInfo):
            if n_classes is not None:
                return np.full((len(x), n_classes), node_content.get().target)
            return np.full((len(x),), node_content.get().target)
        elif isinstance(node_content.get(), _InnerNodeInfo):
            if n_classes is not None:
                pred = np.empty((x.shape[0], n_classes), dtype=np.float64)
                l_msk = (x[:, node_content.get().index:
                           (node_content.get().index + 1)] <=
                         node_content.get().value)
                pred[l_msk.flatten(), :] = _predict_tree_class(
                    x[l_msk.flatten(), :], node, node_content_num,
                    n_classes=n_classes,
                    rights=0, depth=depth + 1)
                pred[~l_msk.flatten(), :] = _predict_tree_class(
                    x[~l_msk.flatten(), :], node, node_content_num,
                    n_classes=n_classes,
                    rights=1, depth=depth + 1)
                return pred
            else:
                pred = np.empty((x.shape[0],), dtype=np.float64)
                l_msk = (x[:, node_content.get().index:
                           (node_content.get().index + 1)] <=
                         node_content.get().value)
                pred[l_msk.flatten()] = _predict_tree_class(
                    x[l_msk.flatten()], node, node_content_num,
                    n_classes=n_classes,
                    rights=0, depth=depth + 1)
                pred[~l_msk.flatten()] = _predict_tree_class(
                    x[~l_msk.flatten()], node, node_content_num,
                    n_classes=n_classes,
                    rights=1, depth=depth + 1)
                return pred
    elif isinstance(node_content, _ClassificationNode):
        if len(x) > 0:
            sk_tree_pred = node_content.content.sk_tree.predict(x)
            b = np.zeros((sk_tree_pred.size, n_classes))
            b[np.arange(sk_tree_pred.size), sk_tree_pred] = 1
            sk_tree_pred = b
            pred = np.zeros((len(x), n_classes), dtype=np.float64)
            pred[:, np.arange(n_classes)] = sk_tree_pred
            return pred
    elif isinstance(node_content, _RegressionNode):
        if len(x) > 0:
            sk_tree_pred = node_content.content.sk_tree.predict(x)
            return sk_tree_pred


def _predict_proba_tree(x, node, node_content_num,
                        n_classes=None, rights=0, depth=0):
    if node_content_num == 0:
        node_content_num = node_content_num + 1
    else:
        node_content_num = node_content_num * 2 + rights
    x = np.block(x)
    node_content = node[node_content_num - 1]
    if len(x) == 0:
        return np.empty((0, n_classes), dtype=np.float64)
    if isinstance(node_content, _NodeInfo):
        if isinstance(node_content.get(), _LeafInfo):
            single_pred = (node_content.get().frequencies /
                           node_content.get().size)
            return np.tile(single_pred, (len(x), 1))
        elif isinstance(node_content.get(), _InnerNodeInfo):
            pred = np.empty((x.shape[0], n_classes), dtype=np.float64)
            l_msk = (x[:, node_content.get().index:
                       (node_content.get().index + 1)] <=
                     node_content.get().value)
            pred[l_msk.flatten(), :] = compss_wait_on(
                _predict_proba_tree(x[l_msk.flatten(), :],
                                    node, node_content_num,
                                    n_classes=n_classes,
                                    rights=0, depth=depth + 1))
            pred[~l_msk.flatten(), :] = compss_wait_on(
                _predict_proba_tree(x[~l_msk.flatten(), :],
                                    node, node_content_num,
                                    n_classes=n_classes,
                                    rights=1, depth=depth + 1))
            return pred
    elif isinstance(node_content, _ClassificationNode):
        if len(x) > 0:
            sk_tree_pred = node_content.content.sk_tree.predict_proba(x)
            pred = np.zeros((len(x), n_classes), dtype=np.float64)
            pred[:, node_content.content.sk_tree.classes_] = sk_tree_pred
            return pred


def apply_split_points_to_blocks_regression(x_block, y_block,
                                            best_attribute,
                                            optimal_value, indexes_to_try):
    if optimal_value is None:
        data_to_compress = np.block(y_block)
        len_compress_l = np.array([0])
        compress_l = np.array([0])
        if len(data_to_compress) > 0:
            compress_l = np.sum(data_to_compress)
            len_compress_l = len(data_to_compress)
        return (None, None, np.block(x_block), np.block(y_block),
                np.array([0]), np.array([0]), compress_l, len_compress_l)
    if x_block is None:
        return (None, None, None, None, np.array([np.nan]),
                np.array([np.nan]), np.array([np.nan]), np.array([np.nan]))
    else:
        x_block = np.block(x_block)
        y_block = np.block(y_block)
    left_x = x_block[x_block[:, indexes_to_try[best_attribute]] <
                     optimal_value]
    right_x = x_block[x_block[:, indexes_to_try[best_attribute]] >=
                      optimal_value]
    right_y = y_block[x_block[:, indexes_to_try[best_attribute]] >=
                      optimal_value]
    left_y = y_block[x_block[:, indexes_to_try[best_attribute]] <
                     optimal_value]
    data_to_compress = np.block(right_y)
    data_to_compress_2 = np.block(left_y)
    if len(data_to_compress) > 0:
        compress_r = np.sum(data_to_compress)
        len_compress_r = len(data_to_compress)
    else:
        compress_r = np.array([0])
        len_compress_r = np.array([0])
    if len(data_to_compress_2) > 0:
        compress_l = np.sum(data_to_compress_2)
        len_compress_l = len(data_to_compress_2)
    else:
        compress_l = np.array([0])
        len_compress_l = np.array([0])
    del x_block
    del y_block
    return (right_x, right_y, left_x, left_y, compress_r,
            len_compress_r, compress_l, len_compress_l)


def apply_split_points_to_blocks(x_block, y_block, best_attribute,
                                 optimal_value, indexes_to_try,
                                 n_classes, aggregate_r, aggregate):
    if optimal_value is None:
        y_block = np.block(y_block)
        if y_block is not None:
            if len(y_block) > 0:
                data_bincount = np.bincount(y_block.astype(int).flatten())
                if len(data_bincount) < n_classes:
                    aggregate[:len(data_bincount)] += data_bincount
                else:
                    aggregate += data_bincount
        return (None, None, np.block(x_block), np.block(y_block),
                aggregate_r, aggregate)
    if x_block is None:
        return None, None, None, None, aggregate_r, aggregate
    else:
        x_block = np.block(x_block)
        y_block = np.block(y_block)
    left_x = x_block[x_block[:, indexes_to_try[best_attribute]] <
                     optimal_value]
    right_x = x_block[x_block[:, indexes_to_try[best_attribute]] >=
                      optimal_value]
    right_y = y_block[x_block[:, indexes_to_try[best_attribute]] >=
                      optimal_value]
    left_y = y_block[x_block[:, indexes_to_try[best_attribute]] <
                     optimal_value]
    del x_block
    del y_block
    if right_y is not None:
        if len(right_y) > 0:
            data_bincount = np.bincount(right_y.astype(int).flatten())
            if len(data_bincount) < n_classes:
                aggregate_r[:len(data_bincount)] += data_bincount
            else:
                aggregate_r += data_bincount
    if left_y is not None:
        if len(left_y) > 0:
            data_bincount = np.bincount(left_y.astype(int).flatten())
            if len(data_bincount) < n_classes:
                aggregate[:len(data_bincount)] += data_bincount
            else:
                aggregate += data_bincount
    return right_x, right_y, left_x, left_y, aggregate_r, aggregate


def select_optimal_split_point(best_attribute, position_m_g,
                               split_points, bucket_minimum_gini):
    if best_attribute is None:
        return None
    return split_points[bucket_minimum_gini][best_attribute][position_m_g]


def get_minimum_measure(ginis_list, number_attributes, gini=True):
    if gini:
        minimum_measure = 1
    else:
        minimum_measure = np.inf
    for idx, ginis in enumerate(ginis_list):
        if ginis[np.argmin(ginis)] < minimum_measure:
            position_m_g = np.argmin(ginis)
            minimum_measure = ginis[position_m_g]
            best_attribute = idx % number_attributes
            actual_bucket = int(math.floor(idx / number_attributes))
    if minimum_measure == 1:
        return None, None, None, 1
    if minimum_measure == np.inf:
        return None, None, None, np.inf
    return best_attribute, position_m_g, actual_bucket, minimum_measure


@constraint(computing_units="${ComputingUnits}")
@task(returns=2)
def merge_partial_results_compute_mse_both_sides(partial_results_l,
                                                 partial_results_r):
    if partial_results_l[0] is None or len(partial_results_l[0]) < 1:
        return np.array([np.inf]), False
    if partial_results_l[0][0] is None:
        return np.array([np.inf]), False
    concatted_values_l = []
    value_to_compute_mse = []
    for k in range(len(partial_results_l[0])):
        value_to_concat = []
        value_to_mse = []
        for j in range(len(partial_results_l)):
            value_to_concat.append(partial_results_l[j][k][1:])
            if not np.isnan(partial_results_l[j][k][0]):
                value_to_mse.extend([partial_results_l[j][k][0]])
            else:
                value_to_mse.extend([0])
        concatted_values_l.append(np.sum(value_to_concat, axis=0))
        value_to_compute_mse.append(value_to_mse)
    number_occurrences = [occurrences[1] for
                          occurrences in concatted_values_l]
    mse_values = []
    for individual_values, value in zip(value_to_compute_mse,
                                        concatted_values_l):
        mse_values.append(np.sum(np.square(np.subtract(individual_values,
                                                       value[0] / value[1]))))
        del value
    del concatted_values_l
    if partial_results_r[0] is None or len(partial_results_r[0]) < 1:
        return np.array([np.inf]), False
    if partial_results_r[0][0] is None:
        return np.array([np.inf]), False
    concatted_values_r = []
    value_to_compute_mse = []
    for k in range(len(partial_results_r[0])):
        value_to_concat = []
        value_to_mse = []
        for j in range(len(partial_results_r)):
            value_to_concat.append(partial_results_r[j][k][1:])
            if not np.isnan(partial_results_r[j][k][0]):
                value_to_mse.extend([partial_results_r[j][k][0]])
            else:
                value_to_mse.extend([0])
        concatted_values_r.append(np.sum(value_to_concat, axis=0))
        value_to_compute_mse.append(value_to_mse)
    number_occurrences_r = [occurrences[1] for
                            occurrences in concatted_values_r]
    mse_values_r = []
    for individual_values, value in zip(value_to_compute_mse,
                                        concatted_values_r):
        mse_values_r.append(np.sum(np.square(np.subtract(
            individual_values, value[0] / value[1]))))
        del value
    del concatted_values_r
    if mse_values is None:
        return np.array([np.inf]), False
    return np.add(mse_values, mse_values_r), np.array(
        [number_occurrences_r[i] != 0 and number_occurrences[i] != 0
         for i in range(len(mse_values))])


def gini_function_compressed(y, classes):
    if not len(y) != 0:
        return 0
    probs = []
    total_y = np.sum(y)
    for idx in range(len(classes)):
        if len(y) > idx:
            probs.append(y[idx]/total_y)
    p = np.array(probs)
    return 1 - ((p * p).sum())


@constraint(computing_units="${ComputingUnits}")
@task(returns=2)
def merge_partial_results_compute_gini_both_sides(partial_results_l,
                                                  partial_results_r,
                                                  n_classes):
    if partial_results_l[0] is None or len(partial_results_l[0]) < 1:
        return np.array([5]), False
    if partial_results_l[0][0] is None:
        return np.array([5]), False
    concatted_values_l = []
    for k in range(len(partial_results_l[0])):
        value_to_concat = np.zeros(n_classes)
        for j in range(len(partial_results_l)):
            if len(partial_results_l[j][k]) > 0:
                value_to_concat[:len(partial_results_l[j][k])] = (
                        value_to_concat[:len(partial_results_l[j][k])] +
                        partial_results_l[j][k])
        concatted_values_l.append(value_to_concat)
    number_occurrences = [np.sum(occurrences).astype(int) for
                          occurrences in concatted_values_l]
    gini_values = []
    for value in concatted_values_l:
        gini_values.append(gini_function_compressed(value,
                                                    np.arange(n_classes)))
    if partial_results_r[0] is None or len(partial_results_r[0]) < 1:
        return np.array([5]), False
    if partial_results_r[0][0] is None:
        return np.array([5]), False
    concatted_values_r = []
    for k in range(len(partial_results_r[0])):
        value_to_concat = np.zeros(n_classes)
        for j in range(len(partial_results_r)):
            value_to_concat[:len(partial_results_r[j][k])] = (
                    value_to_concat[:len(partial_results_r[j][k])] +
                    partial_results_r[j][k])
        concatted_values_r.append(value_to_concat)
    gini_values_r = []
    for value in concatted_values_r:
        gini_values_r.append(gini_function_compressed(
            value, np.arange(n_classes)))
    number_occurrences_r = [np.sum(occurrences).astype(int) for
                            occurrences in concatted_values_r]
    del concatted_values_r
    return np.array(
        [(number_occurrences_r[i] / (number_occurrences_r[i] +
                                     number_occurrences[i]) *
          gini_values_r[i]) + (number_occurrences[i] / (
                number_occurrences_r[i] + number_occurrences[i]) *
                               gini_values[i])
         if number_occurrences[i] >= 4 and number_occurrences_r[i] >= 4 else
         5 for i in range(len(gini_values))]), \
        np.array([number_occurrences_r[i] != 0 and number_occurrences[i] != 0
                  for i in range(len(gini_values))])


@constraint(computing_units="${ComputingUnits}")
@task(x_block=COLLECTION_IN, y_block=COLLECTION_IN, returns=2)
def classes_per_split(x_block, y_block, split_points, indexes_to_compare,
                      indexes_to_select=np.array([0]), regression=False):
    number_classes_l = [np.array([]) for _ in range(len(indexes_to_compare))]
    number_classes_r = [np.array([]) for _ in range(len(indexes_to_compare))]
    number_none_split_points = 0
    for inner_split in split_points:
        if np.any(inner_split) is None:
            number_none_split_points = number_none_split_points + 1
    if x_block is None or len(x_block) == 0 or \
            number_none_split_points == len(split_points):
        for idx in range(len(indexes_to_compare)):
            number_classes_l[idx] = np.array([])
            number_classes_r[idx] = np.array([])
        return number_classes_l, number_classes_r
    x_block = np.block(x_block)
    y_block = np.block(y_block)
    if indexes_to_select is not None:
        if len(indexes_to_select) == 1:
            if indexes_to_select[0] == 0:
                x_block = x_block[:, indexes_to_compare]
            else:
                y_block = y_block[indexes_to_select]
                x_block = x_block[indexes_to_select]
                x_block = x_block[:, indexes_to_compare]
        else:
            y_block = y_block[indexes_to_select]
            x_block = x_block[indexes_to_select]
            x_block = x_block[:, indexes_to_compare]
    else:
        x_block = x_block[:, indexes_to_compare]
    if regression:
        for idx, attribute_split_points in enumerate(split_points):
            attribute_splittings_l = []
            attribute_splittings_r = []
            for value in attribute_split_points:
                attribute_splittings_l.append(np.array(
                    [np.mean(y_block[x_block[:, idx] < value, 0]),
                     np.sum(y_block[x_block[:, idx] < value, 0]),
                     len(y_block[x_block[:, idx] < value, 0])]))
                attribute_splittings_r.append(np.array(
                    [np.mean(y_block[x_block[:, idx] >= value, 0]),
                     np.sum(y_block[x_block[:, idx] >= value, 0]),
                     len(y_block[x_block[:, idx] >= value, 0])]))
            if len(attribute_splittings_r) == 0:
                attribute_splittings_r = np.array([])
            if len(attribute_splittings_l) == 0:
                attribute_splittings_l = np.array([])
            number_classes_l[idx] = attribute_splittings_l
            number_classes_r[idx] = attribute_splittings_r
    else:
        for idx, attribute_split_points in enumerate(split_points):
            attribute_splittings_l = []
            attribute_splittings_r = []
            for value in attribute_split_points:
                attribute_splittings_l.append(np.bincount(
                    y_block[x_block[:, idx] < value, 0].astype(int)))
                attribute_splittings_r.append(np.bincount(
                    y_block[x_block[:, idx] >= value, 0].astype(int)))
            if len(attribute_splittings_r) == 0:
                attribute_splittings_r = np.array([])
            if len(attribute_splittings_l) == 0:
                attribute_splittings_l = np.array([])
            number_classes_l[idx] = attribute_splittings_l
            number_classes_r[idx] = attribute_splittings_r
    del x_block
    del y_block
    return number_classes_l, number_classes_r


@constraint(computing_units="${ComputingUnits}")
@task(returns=1)
def get_split_point_various_attributes_bucket(unique_values,
                                              number_split_points=100,
                                              split_computation="raw"):
    sample_blocks_list = []
    for idx, bucket in enumerate(unique_values):
        if bucket is None:
            sample_blocks_list.append([])
            return sample_blocks_list
        sample_blocks = np.copy(bucket)
        if len(sample_blocks) == 0:
            sample_blocks_list.append([])
            return sample_blocks_list
        number_split_points_actual = number_split_points
        if split_computation == "raw":
            sample_blocks[:-1] += sample_blocks[1:]
            sample_blocks[-1] = sample_blocks[-1] * 2
            sample_blocks = sample_blocks / 2
            if number_split_points_actual == 0:
                number_split_points_actual = 1
            distance_between_split_points = int(len(
                sample_blocks) / number_split_points_actual)
            if distance_between_split_points == 0:
                sample_blocks_list.append(sample_blocks)
            else:
                sample_blocks_list.append(
                    sample_blocks[0::distance_between_split_points])
        elif split_computation == "gaussian_approximation":
            std = np.std(sample_blocks)
            mean = np.mean(sample_blocks)
            sample_blocks = np.array([mean + std * scipy.stats.norm.ppf(
                (i + 1) / (number_split_points_actual + 1)) for i in
                                      range(number_split_points_actual - 1)])
            sample_blocks_list.append(sample_blocks)
        elif split_computation == "uniform_approximation":
            maximum = np.max(sample_blocks)
            minimum = np.min(sample_blocks)
            sample_blocks = np.array([minimum + i * ((maximum - minimum) / (
                    number_split_points_actual + 1)) for i in
                                      range(number_split_points_actual)])
            sample_blocks_list.append(sample_blocks)
    return sample_blocks_list


@constraint(computing_units="${ComputingUnits}")
@task(x=COLLECTION_IN, y=COLLECTION_IN, actual_node=IN, returns=1)
def construct_subtree(x, y, actual_node, m_try, depth, max_depth=25,
                      random_state=0):
    if x is None or x[0] is None:
        actual_node.content = None
        return actual_node
    else:
        if max_depth == np.inf:
            sklearn_max_depth = None
        else:
            sklearn_max_depth = max_depth - depth
        if isinstance(actual_node, _ClassificationNode):
            dt = SklearnDTClassifier(
                max_features=m_try,
                max_depth=sklearn_max_depth,
                random_state=random_state,
            )
        elif isinstance(actual_node, _RegressionNode):
            dt = SklearnDTRegressor(
                max_features=m_try,
                max_depth=sklearn_max_depth,
                random_state=random_state,
            )
        x = np.block(x)
        y = np.block(y)
        if len(y) == 0 or np.any(y) is None:
            actual_node.content = None
        else:
            dt.fit(x, y.astype(int), check_input=False)
            actual_node.content = _SkTreeWrapper(dt)
        return actual_node


def _sample_selection(x, random_state, bootstrap=True):
    if bootstrap:  # bootstrap:
        selection = random_state.choice(
            x.shape[0], size=x.shape[0], replace=True
        )
        selection.sort()
    else:
        selection = np.arange(x.shape[0])
    return selection


class _SkTreeWrapper:
    def __init__(self, tree):
        self.sk_tree = tree

    def toJson(self):
        return {
            "class_name": self.__class__.__name__,
            "module_name": self.__module__,
            "items": self.__dict__,
        }


class _LeafInfo:
    def __init__(self, size=None, frequencies=None, target=None):
        self.size = size
        self.frequencies = frequencies
        self.target = target

    def toJson(self):
        return {
            "class_name": self.__class__.__name__,
            "module_name": self.__module__,
            "items": self.__dict__,
        }


class _InnerNodeInfo:
    def __init__(self, index=None, value=None):
        self.index = index
        self.value = value

    def toJson(self):
        return {
            "class_name": self.__class__.__name__,
            "module_name": self.__module__,
            "items": self.__dict__,
        }


class _Node:
    """Base class for tree nodes"""

    def __init__(self, is_classifier):
        self.content = None
        self.left = None
        self.right = None
        self.is_classifier = is_classifier
        self.predict_dtype = np.int64 if is_classifier else np.float64

    '''def predict(self, sample):
        node_content = self.content
        if isinstance(node_content, _LeafInfo):
            return np.full((len(sample),), node_content.target)
        if isinstance(node_content, _SkTreeWrapper):
            if len(sample) > 0:
                return node_content.sk_tree.predict(sample)
        if isinstance(node_content, _InnerNodeInfo):
            pred = np.empty((len(sample),), dtype=self.predict_dtype)
            left_mask = sample[:, node_content.index] <= node_content.value
            pred[left_mask] = self.left.predict(sample[left_mask])
            pred[~left_mask] = self.right.predict(sample[~left_mask])
            return pred
        assert len(sample) == 0, "Type not supported"
        return np.empty((0,), dtype=self.predict_dtype)'''


class _ClassificationNode(_Node):
    def __init__(self):
        super().__init__(is_classifier=True)

    '''def predict_proba(self, sample, n_classes):
        node_content = self.content
        if isinstance(node_content, _LeafInfo):
            single_pred = node_content.frequencies / node_content.size
            return np.tile(single_pred, (len(sample), 1))
        if isinstance(node_content, _SkTreeWrapper):
            if len(sample) > 0:
                sk_tree_pred = node_content.sk_tree.predict_proba(sample)
                pred = np.zeros((len(sample), n_classes), dtype=np.float64)
                pred[:, node_content.sk_tree.classes_] = sk_tree_pred
                return pred
        if isinstance(node_content, _InnerNodeInfo):
            pred = np.empty((len(sample), n_classes), dtype=np.float64)
            l_msk = sample[:, node_content.index] <= node_content.value
            pred[l_msk] = self.left.predict_proba(sample[l_msk], n_classes)
            pred[~l_msk] = self.right.predict_proba(sample[~l_msk], n_classes)
            return pred
        assert len(sample) == 0, "Type not supported"
        return np.empty((0, n_classes), dtype=np.float64)'''

    def toJson(self):
        return {
            "class_name": self.__class__.__name__,
            "module_name": self.__module__,
            "items": self.__dict__,
        }


class _RegressionNode(_Node):
    def __init__(self):
        super().__init__(is_classifier=False)

    def toJson(self):
        return {
            "class_name": self.__class__.__name__,
            "module_name": self.__module__,
            "items": self.__dict__,
        }


class _NodeInfo:
    def __init__(self):
        self.node_info = None

    def set(self, node_info):
        self.node_info = node_info

    def get(self):
        return self.node_info

    def toJson(self):
        return {
            "class_name": self.__class__.__name__,
            "module_name": self.__module__,
            "items": self.__dict__,
        }


def encode_forest_helper(obj):
    if isinstance(obj, (DecisionTreeClassifier, DecisionTreeRegressor, _Node,
                        _NodeInfo,
                        _ClassificationNode, _RegressionNode, _InnerNodeInfo,
                        _LeafInfo, _SkTreeWrapper)):
        return obj.toJson()


def decode_forest_helper(class_name, obj):
    if class_name == 'DecisionTreeClassifier':
        model = eval(class_name)(
            n_classes=obj.pop("n_classes"),
            try_features=obj.pop("try_features"),
            max_depth=obj.pop("max_depth"),
            distr_depth=obj.pop("distr_depth"),
            sklearn_max=obj.pop("sklearn_max"),
            bootstrap=obj.pop("bootstrap"),
            random_state=obj.pop("random_state"),
            range_min=obj.pop("range_min"),
            range_max=obj.pop("range_max"),
            n_split_points=obj.pop("n_split_points"),
            sync_after_fit=obj.pop("sync_after_fit"),
        )
    elif class_name == 'DecisionTreeRegressor':
        model = eval(class_name)(
            try_features=obj.pop("try_features"),
            max_depth=obj.pop("max_depth"),
            distr_depth=obj.pop("distr_depth"),
            sklearn_max=obj.pop("sklearn_max"),
            bootstrap=obj.pop("bootstrap"),
            random_state=obj.pop("random_state"),
            range_min=obj.pop("range_min"),
            range_max=obj.pop("range_max"),
            n_split_points=obj.pop("n_split_points"),
            sync_after_fit=obj.pop("sync_after_fit"),
        )
    elif class_name == '_SkTreeWrapper':
        sk_tree = obj.pop("sk_tree")
        model = _SkTreeWrapper(sk_tree)
    else:
        model = eval(class_name)()
    model.__dict__.update(obj)
    return model
