import pickle
from sys import float_info

import numpy as np
from numpy.random.mtrand import RandomState
from pycompss.api.api import compss_delete_object
from pycompss.api.constraint import constraint
from pycompss.api.parameter import FILE_IN, Type, COLLECTION_IN, Depth
from pycompss.api.task import task
from sklearn.tree import DecisionTreeClassifier as SklearnDTClassifier
from sklearn.tree import DecisionTreeRegressor as SklearnDTRegressor
import dislib.data.util.model as utilmodel


from dislib.trees.mmap.test_split import test_split
from dislib.data.array import Array


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
    ):
        self.try_features = try_features
        self.max_depth = max_depth
        self.distr_depth = distr_depth
        self.sklearn_max = sklearn_max
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.base_node = base_node
        self.base_tree = base_tree

        self.n_features = None
        self.n_classes = None

        self.tree = None
        self.nodes_info = None
        self.subtrees = None

    def fit(self, dataset):
        """Fits the DecisionTree.

        Parameters
        ----------
        dataset : dislib.classification.rf._data.RfDataset
        """

        self.n_features = dataset.get_n_features()
        self.n_classes = dataset.get_n_classes()
        samples_path = dataset.samples_path
        features_path = dataset.features_path
        n_samples = dataset.get_n_samples()
        y_targets = dataset.get_y_targets()

        seed = self.random_state.randint(np.iinfo(np.int32).max)

        sample, y_s = _sample_selection(
            n_samples, y_targets, self.bootstrap, seed
        )

        self.tree = self.base_node()
        self.nodes_info = []
        self.subtrees = []
        tree_traversal = [(self.tree, sample, y_s, 0)]
        while tree_traversal:
            node, sample, y_s, depth = tree_traversal.pop()
            if depth < self.distr_depth:
                split = _split_node_wrapper(
                    sample,
                    self.n_features,
                    y_s,
                    self.n_classes,
                    self.try_features,
                    self.random_state,
                    samples_file=samples_path,
                    features_file=features_path,
                )
                node_info, left_group, y_l, right_group, y_r = split
                compss_delete_object(sample)
                compss_delete_object(y_s)
                node.content = len(self.nodes_info)
                self.nodes_info.append(node_info)
                node.left = self.base_node()
                node.right = self.base_node()
                depth = depth + 1
                tree_traversal.append((node.right, right_group, y_r, depth))
                tree_traversal.append((node.left, left_group, y_l, depth))
            else:
                subtree = _build_subtree_wrapper(
                    sample,
                    y_s,
                    self.n_features,
                    self.max_depth - depth,
                    self.n_classes,
                    self.try_features,
                    self.sklearn_max,
                    self.random_state,
                    self.base_node,
                    self.base_tree,
                    samples_path,
                    features_path,
                )
                node.content = len(self.subtrees)
                self.subtrees.append(subtree)
                compss_delete_object(sample)
                compss_delete_object(y_s)
        self.nodes_info = _merge(*self.nodes_info)

    def predict(self, x_row):
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

        branch_predictions = []
        for i, subtree in enumerate(self.subtrees):
            pred = _predict_branch(
                x_row._blocks,
                self.tree,
                self.nodes_info,
                i,
                subtree,
                self.distr_depth,
            )
            branch_predictions.append(pred)
        return _merge_branches(
            None, *branch_predictions,
            classification=self.n_classes is not None
        )


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
        try_features,
        max_depth,
        distr_depth,
        sklearn_max,
        bootstrap,
        random_state,
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
        )

    def predict_proba(self, x_row):
        """Predicts class probabilities for a row block using a fitted tree.

        Parameters
        ----------
        x_row : ds-array
            A row block of samples.

        Returns
        -------
        predicted_proba : ndarray
            An array with the predicted probabilities for the given samples.
            The shape is (len(subset.samples), self.n_classes), with the index
            of the column being codes of the fitted
            dislib.classification.rf.data.RfDataset. The returned object can be
            a pycompss.runtime.Future object.
        """

        assert self.tree is not None, "The decision tree is not fitted."

        branch_predictions = []
        for i, subtree in enumerate(self.subtrees):
            pred = _predict_branch_proba(
                x_row._blocks,
                self.tree,
                self.nodes_info,
                i,
                subtree,
                self.distr_depth,
                self.n_classes,
            )
            branch_predictions.append(pred)
        return _merge_branches(
            self.n_classes, *branch_predictions, classification=True
        )


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
        )


class _Node:
    """Base class for tree nodes"""

    def __init__(self, is_classifier):
        self.content = None
        self.left = None
        self.right = None
        self.is_classifier = is_classifier
        self.predict_dtype = np.int64 if is_classifier else np.float64

    def predict(self, sample):
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
        return np.empty((0,), dtype=self.predict_dtype)


class _ClassificationNode(_Node):
    def __init__(self):
        super().__init__(is_classifier=True)

    def predict_proba(self, sample, n_classes):
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
        return np.empty((0, n_classes), dtype=np.float64)

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


class _SkTreeWrapper:
    def __init__(self, tree):
        self.sk_tree = tree

    def toJson(self):
        return {
            "class_name": self.__class__.__name__,
            "module_name": self.__module__,
            "items": self.__dict__,
        }


def _get_sample_attributes(samples_file, indices):
    samples_mmap = np.load(samples_file, mmap_mode="r", allow_pickle=False)
    x = samples_mmap[indices]
    return x


@constraint(computing_units="${ComputingUnits}")
@task(priority=True, returns=2)
def _sample_selection(n_samples, y_targets, bootstrap, seed):
    if bootstrap:
        random_state = RandomState(seed)
        selection = random_state.choice(
            n_samples, size=n_samples, replace=True
        )
        selection.sort()
        return selection, y_targets[selection]
    else:
        return np.arange(n_samples), y_targets


def _feature_selection(untried_indices, m_try, random_state):
    selection_len = min(m_try, len(untried_indices))
    return random_state.choice(
        untried_indices, size=selection_len, replace=False
    )


def _get_groups(sample, y_s, features_mmap, index, value):
    if index is None:
        empty_sample = np.array([], dtype=np.int64)
        empty_target = np.array([], dtype=y_s.dtype)
        return sample, y_s, empty_sample, empty_target
    feature = features_mmap[index][sample]
    mask = feature < value
    left = sample[mask]
    right = sample[~mask]
    y_l = y_s[mask]
    y_r = y_s[~mask]
    return left, y_l, right, y_r


def _compute_leaf_info(y_s, n_classes):
    if n_classes is not None:
        frequencies = np.bincount(y_s, minlength=n_classes)
        mode = np.argmax(frequencies)
        return _LeafInfo(len(y_s), frequencies, mode)
    else:
        return _LeafInfo(len(y_s), None, np.mean(y_s))


def _split_node_wrapper(
    sample,
    n_features,
    y_s,
    n_classes,
    m_try,
    random_state,
    samples_file=None,
    features_file=None,
):
    seed = random_state.randint(np.iinfo(np.int32).max)

    if features_file is not None:
        return _split_node_using_features(
            sample, n_features, y_s, n_classes, m_try, features_file, seed
        )
    elif samples_file is not None:
        return _split_node(
            sample, n_features, y_s, n_classes, m_try, samples_file, seed
        )
    else:
        raise ValueError(
            "Invalid combination of arguments. samples_file is "
            "None and features_file is None."
        )


@constraint(computing_units="${ComputingUnits}")
@task(features_file=FILE_IN, returns=(object, list, list, list, list))
def _split_node_using_features(
    sample, n_features, y_s, n_classes, m_try, features_file, seed
):
    features_mmap = np.load(features_file, mmap_mode="r", allow_pickle=False)
    random_state = RandomState(seed)
    return _compute_split(
        sample, n_features, y_s, n_classes, m_try, features_mmap, random_state
    )


@constraint(computing_units="${ComputingUnits}")
@task(samples_file=FILE_IN, returns=(object, list, list, list, list))
def _split_node(sample, n_features, y_s, n_classes, m_try, samples_file, seed):
    features_mmap = np.load(samples_file, mmap_mode="r", allow_pickle=False).T
    random_state = RandomState(seed)
    return _compute_split(
        sample, n_features, y_s, n_classes, m_try, features_mmap, random_state
    )


def _compute_split(
    sample, n_features, y_s, n_classes, m_try, features_mmap, random_state
):
    node_info = left_group = y_l = right_group = y_r = None
    split_ended = False
    tried_indices = []
    while not split_ended:
        untried_indices = np.setdiff1d(np.arange(n_features), tried_indices)
        index_selection = _feature_selection(
            untried_indices, m_try, random_state
        )
        b_score = float_info.max
        b_index = None
        b_value = None
        for index in index_selection:
            feature = features_mmap[index]
            score, value = test_split(sample, y_s, feature, n_classes)
            if score < b_score:
                b_score, b_value, b_index = score, value, index
        groups = _get_groups(sample, y_s, features_mmap, b_index, b_value)
        left_group, y_l, right_group, y_r = groups
        if left_group.size and right_group.size:
            split_ended = True
            node_info = _InnerNodeInfo(b_index, b_value)
        else:
            tried_indices.extend(list(index_selection))
            if len(tried_indices) == n_features:
                split_ended = True
                node_info = _compute_leaf_info(y_s, n_classes)
                left_group = sample
                y_l = y_s
                right_group = np.array([], dtype=np.int64)
                y_r = np.array([], dtype=y_s.dtype)

    return node_info, left_group, y_l, right_group, y_r


def _build_subtree_wrapper(
    sample,
    y_s,
    n_features,
    max_depth,
    n_classes,
    m_try,
    sklearn_max,
    random_state,
    base_node,
    base_tree,
    samples_file,
    features_file,
):
    seed = random_state.randint(np.iinfo(np.int32).max)
    if features_file is not None:
        return _build_subtree_using_features(
            sample,
            y_s,
            n_features,
            max_depth,
            n_classes,
            m_try,
            sklearn_max,
            seed,
            base_node,
            base_tree,
            samples_file,
            features_file,
        )
    else:
        return _build_subtree(
            sample,
            y_s,
            n_features,
            max_depth,
            n_classes,
            m_try,
            sklearn_max,
            seed,
            base_node,
            base_tree,
            samples_file,
        )


@constraint(computing_units="${ComputingUnits}")
@task(samples_file=FILE_IN, features_file=FILE_IN, returns=_Node)
def _build_subtree_using_features(
    sample,
    y_s,
    n_features,
    max_depth,
    n_classes,
    m_try,
    sklearn_max,
    seed,
    base_node,
    base_tree,
    samples_file,
    features_file,
):
    random_state = RandomState(seed)
    return _compute_build_subtree(
        sample,
        y_s,
        n_features,
        max_depth,
        n_classes,
        m_try,
        sklearn_max,
        random_state,
        base_node,
        base_tree,
        samples_file,
        features_file=features_file,
    )


@constraint(computing_units="${ComputingUnits}")
@task(samples_file=FILE_IN, returns=_Node)
def _build_subtree(
    sample,
    y_s,
    n_features,
    max_depth,
    n_classes,
    m_try,
    sklearn_max,
    seed,
    base_node,
    base_tree,
    samples_file,
):
    random_state = RandomState(seed)
    return _compute_build_subtree(
        sample,
        y_s,
        n_features,
        max_depth,
        n_classes,
        m_try,
        sklearn_max,
        random_state,
        base_node,
        base_tree,
        samples_file,
    )


def _compute_build_subtree(
    sample,
    y_s,
    n_features,
    max_depth,
    n_classes,
    m_try,
    sklearn_max,
    random_state,
    base_node,
    base_tree,
    samples_file,
    features_file=None,
    use_sklearn=True,
):
    if not sample.size:
        return base_node()
    if features_file is not None:
        mmap = np.load(features_file, mmap_mode="r", allow_pickle=False)
    else:
        mmap = np.load(samples_file, mmap_mode="r", allow_pickle=False).T
    subtree = base_node()
    tree_traversal = [(subtree, sample, y_s, 0)]
    while tree_traversal:
        node, sample, y_s, depth = tree_traversal.pop()
        if depth < max_depth:
            if use_sklearn and n_features * len(sample) <= sklearn_max:
                if max_depth == np.inf:
                    sklearn_max_depth = None
                else:
                    sklearn_max_depth = max_depth - depth
                dt = base_tree(
                    max_features=m_try,
                    max_depth=sklearn_max_depth,
                    random_state=random_state,
                )
                unique = np.unique(
                    sample, return_index=True, return_counts=True
                )
                sample, new_indices, sample_weight = unique
                x = _get_sample_attributes(samples_file, sample)
                y_s = y_s[new_indices]
                dt.fit(x, y_s, sample_weight=sample_weight, check_input=False)
                node.content = _SkTreeWrapper(dt)
            else:
                split = _compute_split(
                    sample,
                    n_features,
                    y_s,
                    n_classes,
                    m_try,
                    mmap,
                    random_state,
                )
                node_info, left_group, y_l, right_group, y_r = split
                node.content = node_info
                if isinstance(node_info, _InnerNodeInfo):
                    node.left = base_node()
                    node.right = base_node()
                    tree_traversal.append(
                        (node.right, right_group, y_r, depth + 1)
                    )
                    tree_traversal.append(
                        (node.left, left_group, y_l, depth + 1)
                    )
        else:
            node.content = _compute_leaf_info(y_s, n_classes)
    return subtree


@constraint(computing_units="${ComputingUnits}")
@task(returns=list)
def _merge(*object_list):
    return object_list


def _get_subtree_path(subtree_index, distr_depth):
    if distr_depth == 0:
        return ""
    return bin(subtree_index)[2:].zfill(distr_depth)


def _get_predicted_indices(samples, tree, nodes_info, path):
    idx_mask = np.full((len(samples),), True)
    for direction in path:
        node_info = nodes_info[tree.content]
        if isinstance(node_info, _LeafInfo):
            if direction == "1":
                idx_mask[:] = 0
        else:
            col = node_info.index
            value = node_info.value
            if direction == "0":
                idx_mask[idx_mask] = samples[idx_mask, col] <= value
                tree = tree.left
            else:
                idx_mask[idx_mask] = samples[idx_mask, col] > value
                tree = tree.right
    return idx_mask


@constraint(computing_units="${ComputingUnits}")
@task(row_blocks={Type: COLLECTION_IN, Depth: 2}, returns=1)
def _predict_branch(
    row_blocks, tree, nodes_info, subtree_index, subtree, distr_depth
):
    samples = Array._merge_blocks(row_blocks)
    path = _get_subtree_path(subtree_index, distr_depth)
    indices_mask = _get_predicted_indices(samples, tree, nodes_info, path)
    prediction = subtree.predict(samples[indices_mask])
    return indices_mask, prediction


@constraint(computing_units="${ComputingUnits}")
@task(row_blocks={Type: COLLECTION_IN, Depth: 2}, returns=1)
def _predict_branch_proba(
    row_blocks,
    tree,
    nodes_info,
    subtree_index,
    subtree,
    distr_depth,
    n_classes,
):
    samples = Array._merge_blocks(row_blocks)
    path = _get_subtree_path(subtree_index, distr_depth)
    indices_mask = _get_predicted_indices(samples, tree, nodes_info, path)
    prediction = subtree.predict_proba(samples[indices_mask], n_classes)
    return indices_mask, prediction


@constraint(computing_units="${ComputingUnits}")
@task(returns=list)
def _merge_branches(n_classes, *predictions, classification):
    samples_len = len(predictions[0][0])
    if classification:
        if n_classes is not None:  # predict class
            shape = (samples_len, n_classes)
            dtype = np.float64
        else:  # predict_proba
            shape = (samples_len,)
            dtype = np.int64
    else:  # predict value
        shape = (samples_len,)
        dtype = np.float64

    merged_prediction = np.empty(shape, dtype=dtype)
    for selected, prediction in predictions:
        merged_prediction[selected] = prediction
    if len(shape) == 1 and not classification:
        return np.expand_dims(merged_prediction, axis=1)
    return merged_prediction


def encode_forest_helper(obj):
    if isinstance(obj, (DecisionTreeClassifier, DecisionTreeRegressor, _Node,
                        _ClassificationNode, _RegressionNode, _InnerNodeInfo,
                        _LeafInfo, _SkTreeWrapper)):
        return obj.toJson()


def decode_forest_helper(class_name, obj, cbor=False):
    if class_name in ('DecisionTreeClassifier', 'DecisionTreeRegressor'):
        if cbor and utilmodel.blosc2 is not None:
            obj = pickle.loads(utilmodel.blosc2.decompress2(obj))
        model = eval(class_name)(
            try_features=obj.pop("try_features"),
            max_depth=obj.pop("max_depth"),
            distr_depth=obj.pop("distr_depth"),
            sklearn_max=obj.pop("sklearn_max"),
            bootstrap=obj.pop("bootstrap"),
            random_state=obj.pop("random_state"),
        )
    elif class_name == '_SkTreeWrapper':
        sk_tree = obj.pop("sk_tree")
        model = _SkTreeWrapper(sk_tree)
    else:
        model = eval(class_name)()
    model.__dict__.update(obj)
    return model
