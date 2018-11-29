from sys import float_info

from pycompss.api.parameter import *

import numpy as np
from pycompss.api.task import task
from pycompss.api.api import compss_delete_object
from pycompss.api.parameter import FILE_IN, FILE_INOUT
from pycompss.api.task import task
from six.moves import range
from sklearn.tree import DecisionTreeClassifier as SklearnDTClassifier

from dislib.classification.rf.test_split import test_split


class Node:

    def __init__(self):
        self.content = None
        self.left = None
        self.right = None

    def predict(self, sample):
        node_content = self.content
        if isinstance(node_content, LeafInfo):
            return np.full((len(sample),), node_content.mode)
        if isinstance(node_content, SkTreeWrapper):
            return node_content.sk_tree.predict(sample)
        if isinstance(node_content, InnerNodeInfo):
            prediction = np.empty((len(sample),), dtype=np.int64)
            left_indices = sample[:, node_content.index] <= node_content.value
            prediction[left_indices] = self.left.predict(sample[left_indices])
            prediction[~left_indices] = self.right.predict(sample[~left_indices])
            return prediction
        assert False, 'Node.predict() does not support this node type'

    def predict_proba(self, sample, n_classes):
        node_content = self.content
        if isinstance(node_content, LeafInfo):
            return np.repeat(node_content.frequencies/node_content.size, len(sample), 1)
        if isinstance(node_content, SkTreeWrapper):
            prediction = np.zeros((len(sample), n_classes), dtype=np.int64)
            prediction[:, node_content.sk_tree.classes_] = node_content.sk_tree.predict_proba(sample)
            return prediction
        if isinstance(node_content, InnerNodeInfo):
            prediction = np.empty((len(sample), n_classes), dtype=np.int64)
            left_indices = sample[:, node_content.index] <= node_content.value
            prediction[left_indices] = self.left.predict_proba(sample[left_indices])
            prediction[~left_indices] = self.right.predict_proba(sample[~left_indices])
            return prediction
        assert False, 'Node.predict_proba() does not support this node_content type'


class InnerNodeInfo:

    def __init__(self, index=None, value=None):
        self.index = index
        self.value = value


class LeafInfo:

    def __init__(self, size=None, frequencies=None, mode=None):
        self.size = size
        self.frequencies = frequencies
        self.mode = mode


class SkTreeWrapper:
    def __init__(self, tree):
        self.sk_tree = tree
        self.classes = tree.classes_

    # def write_to(self, tree_file):
    #     nodes_to_write = [(0, self.tree_path)]
    #     while nodes_to_write:
    #         node_id, tree_path = nodes_to_write.pop()
    #         if self.i_tree.children_left[node_id] == _tree.TREE_LEAF:
    #             frequencies = dict((self.classes[k], int(v)) for k, v in enumerate(self.i_tree.value[node_id][0]))
    #             mode = max(frequencies, key=frequencies.get)
    #             n_node_samples = self.i_tree.n_node_samples[node_id]
    #             frequencies_str = ', '.join(['"{}": {}'.format(k, v) for k, v in frequencies.items()])
    #             frequencies_str = '{' + frequencies_str + '}'
    #             tree_file.write('{{"tree_path": "{}", "type": "LEAF", '
    #                             '"size": {}, "mode": {}, "frequencies": {}}}\n'
    #                             .format(tree_path, n_node_samples, mode, frequencies_str))
    #         else:
    #             tree_file.write('{{"tree_path": "{}", "type": "NODE", '
    #                             '"index": {}, "value": {}}}\n'
    #                             .format(tree_path, self.i_tree.feature[node_id], self.i_tree.threshold[node_id]))
    #             nodes_to_write.append((self.i_tree.children_right[node_id], tree_path + 'R'))
    #             nodes_to_write.append((self.i_tree.children_left[node_id], tree_path + 'L'))


def get_sample_attributes(samples_file, indices):
    samples_mmap = np.load(samples_file, mmap_mode='r', allow_pickle=False)
    x = samples_mmap[indices]
    return x


def get_feature_mmap(features_file, i):
    return get_features_mmap(features_file)[i]


def get_features_mmap(features_file):
    return np.load(features_file, mmap_mode='r', allow_pickle=False)


@task(priority=True, returns=2)
def sample_selection(n_instances, y_codes, bootstrap):
    if bootstrap:
        np.random.seed()
        selection = np.random.choice(n_instances, size=n_instances,
                                     replace=True)
        selection.sort()
        return selection, y_codes[selection]
    else:
        return np.arange(n_instances), y_codes


def feature_selection(feature_indices, m_try):
    return np.random.choice(feature_indices,
                            size=min(m_try, len(feature_indices)),
                            replace=False)


@task(returns=tuple)
def test_splits(sample, y_s, n_classes, feature_indices, *features):
    min_score = float_info.max
    b_value = None
    b_index = None
    for t in range(len(feature_indices)):
        feature = features[t]
        score, value = test_split(sample, y_s, feature, n_classes)
        if score < min_score:
            min_score = score
            b_index = feature_indices[t]
            b_value = value
    return min_score, b_value, b_index


def get_groups(sample, y_s, features_mmap, index, value):
    if index is None:
        return (sample, y_s, np.array([], dtype=np.int64),
                np.array([], dtype=np.int8))
    feature = features_mmap[index][sample]
    mask = feature < value
    left = sample[mask]
    right = sample[~mask]
    y_l = y_s[mask]
    y_r = y_s[~mask]
    return left, y_l, right, y_r


def compute_leaf_info(y_s, n_classes):
    frequencies = np.bincount(y_s, minlength=n_classes)
    mode = np.argmax(frequencies)
    return LeafInfo(len(y_s), frequencies, mode)


def split_node(sample, n_features, y_s, n_classes, m_try, samples_file=None, features_file=None):
    if features_file is not None:
        return split_node_a(sample, n_features, features_file, y_s, n_classes, m_try)
    elif samples_file is not None:
        return split_node_b(sample, n_features, samples_file, y_s, n_classes, m_try)
    else:
        raise ValueError('Invalid combination of arguments. samples_file is'
                         ' None and features_file is None.')


@task(features_file=FILE_IN, returns=(object, list, list, list, list))
def split_node_a(sample, n_features, features_file, y_s, n_classes, m_try):
    features_mmap = np.load(features_file, mmap_mode='r', allow_pickle=False)
    return compute_split(sample, n_features, features_mmap, y_s, n_classes, m_try)


@task(samples_file=FILE_IN, returns=(object, list, list, list, list))
def split_node_b(sample, n_features, samples_file, y_s, n_classes, m_try):
    features_mmap = np.load(samples_file, mmap_mode='r', allow_pickle=False).T
    return compute_split(sample, n_features, features_mmap, y_s, n_classes, m_try)


def compute_split(sample, n_features, features_mmap, y_s, n_classes, m_try):
    node_info = left_group = y_l = right_group = y_r = None
    split_ended = False
    tried_indices = []
    while not split_ended:
        untried_indices = np.setdiff1d(np.arange(n_features),
                                       tried_indices)  # type: np.ndarray
        index_selection = feature_selection(untried_indices, m_try)
        b_score = float_info.max
        b_index = None
        b_value = None
        for index in index_selection:
            feature = features_mmap[index]
            score, value = test_split(sample, y_s, feature, n_classes)
            if score < b_score:
                b_score, b_value, b_index = score, value, index
        left_group, y_l, right_group, y_r = get_groups(sample, y_s,
                                                       features_mmap, b_index,
                                                       b_value)
        if left_group.size and right_group.size:
            split_ended = True
            node_info = InnerNodeInfo(b_index, b_value)
        else:
            tried_indices.extend(list(index_selection))
            if len(tried_indices) == n_features:
                split_ended = True
                node_info = compute_leaf_info(y_s, n_classes)

    return node_info, left_group, y_l, right_group, y_r


def build_subtree(sample, y_s, n_features, max_depth, n_classes, m_try, samples_file, features_file):
    if features_file is not None:
        return build_subtree_a(sample, y_s, n_features, max_depth, n_classes, m_try, samples_file,
                               features_file)
    else:
        return build_subtree_b(sample, y_s, n_features, max_depth, n_classes, m_try, samples_file)


@task(samples_file=FILE_IN, features_file=FILE_IN, returns=Node)
def build_subtree_a(sample, y_s, n_features, max_depth, n_classes, m_try, samples_file, features_file):
    return build_subtree_in(sample, y_s, n_features, max_depth, n_classes, m_try, samples_file,
                            features_file=features_file)


@task(samples_file=FILE_IN, returns=Node)
def build_subtree_b(sample, y_s, n_features, max_depth, n_classes, m_try, samples_file):
    return build_subtree_in(sample, y_s, n_features, max_depth, n_classes, m_try, samples_file)


def build_subtree_in(sample, y_s, n_features, max_depth, n_classes, m_try, samples_file, features_file=None,
                     use_sklearn_internally=True, sklearn_max_elements=100000000):
    np.random.seed()
    if not sample.size:
        return []
    if features_file is not None:
        features_mmap = np.load(features_file, mmap_mode='r',
                                allow_pickle=False)
    else:
        features_mmap = np.load(samples_file, mmap_mode='r', allow_pickle=False).T
    subtree = Node()
    tree_traversal = [(subtree, sample, y_s, 0)]
    while tree_traversal:
        node, sample, y_s, depth = tree_traversal.pop()
        if depth < max_depth:
            if use_sklearn_internally and n_features * len(sample) <= sklearn_max_elements:
                sklearn_max_depth = None if max_depth == np.inf else max_depth - depth
                dt = SklearnDTClassifier(max_features=m_try, max_depth=sklearn_max_depth)
                sample, new_indices, sample_weight = np.unique(sample, return_index=True, return_counts=True)
                x = get_sample_attributes(samples_file, sample)
                y_s = y_s[new_indices]
                dt.fit(x, y_s, sample_weight=sample_weight, check_input=False)
                node.content = SkTreeWrapper(dt)
            else:
                node_info, left_group, y_l, right_group, y_r = compute_split(sample, n_features, features_mmap,
                                                                             y_s, n_classes, m_try)
                node.content = node_info
                if isinstance(node_info, InnerNodeInfo):
                    node.left = Node()
                    node.right = Node()
                    tree_traversal.append((node.right, right_group, y_r, depth + 1))
                    tree_traversal.append((node.left, left_group, y_l, depth + 1))
        else:
            node.content = compute_leaf_info(y_s, n_classes)
    return subtree


@task(returns=list)
def collect(*object_list):
    return object_list


def get_subtree_path(subtree_index, distr_depth):
    if distr_depth == 0:
        return ''
    return bin(subtree_index)[2:].zfill(distr_depth)


def get_predicted_indices(samples, tree, nodes_info, path):
    indices_mask = np.full((len(samples),), True)
    for direction in path:
        node_info = nodes_info[tree.content]
        if direction == '0':
            indices_mask[indices_mask] = samples[indices_mask, node_info.index] <= node_info.value
            tree = tree.left
        else:
            indices_mask[indices_mask] = samples[indices_mask, node_info.index] > node_info.value
            tree = tree.right
    return indices_mask


@task(returns=1)
def predict_branch(samples, tree, nodes_info, subtree_index, subtree, distr_depth):
    path = get_subtree_path(subtree_index, distr_depth)
    indices_mask = get_predicted_indices(samples, tree, nodes_info, path)
    prediction = subtree.predict(samples[indices_mask])
    return indices_mask, prediction


@task(returns=1)
def predict_branch_proba(samples, tree, nodes_info, subtree_index, subtree, distr_depth, n_classes):
    path = get_subtree_path(subtree_index, distr_depth)
    indices_mask = get_predicted_indices(samples, tree, nodes_info, path)
    prediction = subtree.predict_proba(samples[indices_mask], n_classes)
    return indices_mask, prediction


@task(returns=list)
def merge_branches(shape_0, shape_1, *predictions):
    if shape_1 is not None:
        shape = (shape_0, shape_1)
    else:
        shape = (shape_0,)
    merged_prediction = np.empty(shape, dtype=np.int64)
    for selected, prediction in predictions:
        merged_prediction[selected] = prediction
    return merged_prediction


class DecisionTreeClassifier:

    def __init__(self, try_features, max_depth, distr_depth, bootstrap):
        """
        Decision tree with distributed splits using pyCOMPSs.

        :param try_features: Number of features to try (at least) for splitting each node.
        :param max_depth: Depth of the decision tree.
        :param distr_depth: Nodes are split in a distributed way up to this depth.
        :param bootstrap: Randomly select n_instances samples with repetition (used in random forests).
        """
        self.try_features = try_features
        self.max_depth = max_depth
        self.distr_depth = distr_depth
        self.bootstrap = bootstrap

        self.n_features = None
        self.n_classes = None

        self.tree = None
        self.nodes_info = None
        self.subtrees = None

    def fit(self, dataset):
        """
        Fits the DecisionTreeClassifier.

        :param dataset: dislib.classification.rf.data.RfDataset.
        """

        self.n_features = dataset.get_n_features()
        self.n_classes = dataset.get_n_classes()
        samples_path = dataset.samples_path
        features_path = dataset.features_path
        n_samples = dataset.get_n_samples()
        y_codes = dataset.get_y_codes()

        tree_sample, y_s = sample_selection(n_samples, y_codes, self.bootstrap)
        self.tree = Node()
        self.nodes_info = []
        self.subtrees = []
        tree_traversal = [(self.tree, tree_sample, y_s, 0)]
        while tree_traversal:
            node, sample, y_s, depth = tree_traversal.pop()  # type: (Node, list, list, int)
            if depth < self.distr_depth:
                node_info, left_group, y_l, right_group, y_r = split_node(sample, self.n_features, y_s,
                                                                          self.n_classes, self.try_features,
                                                                          samples_file=samples_path,
                                                                          features_file=features_path)
                compss_delete_object(sample)
                compss_delete_object(y_s)
                node.content = len(self.nodes_info)
                self.nodes_info.append(node_info)
                node.left = Node()
                node.right = Node()
                tree_traversal.append((node.right, right_group, y_r, depth + 1))
                tree_traversal.append((node.left, left_group, y_l, depth + 1))
            else:
                subtree = build_subtree(sample, y_s, self.n_features, self.max_depth - depth, self.n_classes,
                                        self.try_features, samples_path, features_path)
                node.content = len(self.subtrees)
                self.subtrees.append(subtree)
                compss_delete_object(sample)
                compss_delete_object(y_s)
        self.nodes_info = collect(*self.nodes_info)

    def predict(self, samples):
        """ Predicts class codes for the input data using a fitted tree and returns an array."""

        assert self.tree is not None, 'The decision tree is not fitted.'
        assert samples.shape[1] == self.n_features, 'Wrong number of features.'

        branch_predictions = []
        for i, subtree in enumerate(self.subtrees):
            branch_predictions.append(predict_branch(samples, self.tree, self.nodes_info, i, subtree, self.distr_depth))
        return merge_branches(len(samples), None, *branch_predictions)

    def predict_proba(self, samples):
        """ Predicts class probabilities by class code using a fitted tree and returns a 1D or 2D array. """

        assert self.tree is not None, 'The decision tree is not fitted.'
        assert samples.shape[1] == self.n_features, 'Wrong number of features.'

        branch_predictions = []
        for i, subtree in enumerate(self.subtrees):
            branch_predictions.append(predict_branch_proba(samples, self.tree, self.nodes_info, i, subtree,
                                                           self.distr_depth, self.n_classes))
        return merge_branches(len(samples), self.n_classes, *branch_predictions)
