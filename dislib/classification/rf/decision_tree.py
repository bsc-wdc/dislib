import json
import os
from sys import float_info

from pycompss.api.parameter import *
from six.moves import range

from collections import Counter

import numpy as np
from math import sqrt, frexp
from pandas import read_csv
from pandas.api.types import CategoricalDtype
from pycompss.api.task import task
from pycompss.api.api import compss_delete_object

from sklearn.tree import DecisionTreeClassifier as SklearnDTClassifier
from sklearn.tree import _tree

import dislib.classification.rf.prediction as prediction
from dislib.classification.rf.test_split import test_split


class TreeWrapper(object):
    def __init__(self, tree_path, tree):
        self.tree_path = tree_path
        self.i_tree = tree.tree_
        self.classes = tree.classes_

    def write_to(self, tree_file):
        nodes_to_write = [(0, self.tree_path)]
        while nodes_to_write:
            node_id, tree_path = nodes_to_write.pop()
            if self.i_tree.children_left[node_id] == _tree.TREE_LEAF:
                frequencies = dict((self.classes[k], int(v)) for k, v in enumerate(self.i_tree.value[node_id][0]))
                mode = max(frequencies, key=frequencies.get)
                n_node_samples = self.i_tree.n_node_samples[node_id]
                frequencies_str = ', '.join(['"{}": {}'.format(k, v) for k, v in frequencies.items()])
                frequencies_str = '{' + frequencies_str + '}'
                tree_file.write('{{"tree_path": "{}", "type": "LEAF", '
                                '"size": {}, "mode": {}, "frequencies": {}}}\n'
                                .format(tree_path, n_node_samples, mode, frequencies_str))
            else:
                tree_file.write('{{"tree_path": "{}", "type": "NODE", '
                                '"index": {}, "value": {}}}\n'
                                .format(tree_path, self.i_tree.feature[node_id], self.i_tree.threshold[node_id]))
                nodes_to_write.append((self.i_tree.children_right[node_id], tree_path + 'R'))
                nodes_to_write.append((self.i_tree.children_left[node_id], tree_path + 'L'))


class InternalNode(object):

    def __init__(self, tree_path=None, index=None, value=None):
        self.tree_path = tree_path
        self.index = index
        self.value = value

    def to_json(self):
        if self.value == np.inf:
            self.value = json.dumps(np.inf)
        return ('{{"tree_path": "{}", "type": "NODE", '
                '"index": {}, "value": {}}}\n'
                .format(self.tree_path, self.index, self.value))


class Leaf(object):

    def __init__(self, tree_path=None, size=None, frequencies=None, mode=None):
        self.tree_path = tree_path
        self.size = size
        self.frequencies = frequencies
        self.mode = mode

    def to_json(self):
        frequencies_str = ', '.join(map('"%r": %r'.__mod__, list(self.frequencies.items())))
        frequencies_str = '{' + frequencies_str + '}'
        return ('{{"tree_path": "{}", "type": "LEAF", '
                '"size": {}, "mode": {}, "frequencies": {}}}\n'
                .format(self.tree_path, self.size, self.mode, frequencies_str))


def get_features_file(path):
    return os.path.join(path, 'x_t.npy')


def get_sample_attributes(samples_file, indices):
    samples_mmap = np.load(samples_file, mmap_mode='r', allow_pickle=False)
    x = samples_mmap[indices]
    return x


def get_samples_file(path):
    return os.path.join(path, 'x.npy')


def get_feature_mmap(features_file, i):
    return get_features_mmap(features_file)[i]


def get_features_mmap(features_file):
    return np.load(features_file, mmap_mode='r', allow_pickle=False)


@task(priority=True, returns=2)
def sample_selection(n_instances, y_codes, bootstrap):
    if bootstrap:
        np.random.seed()
        selection = np.random.choice(n_instances, size=n_instances, replace=True)
        selection.sort()
        return selection, y_codes[selection]
    else:
        return np.arange(n_instances), y_codes


def feature_selection(feature_indices, m_try):
    return np.random.choice(feature_indices, size=min(m_try, len(feature_indices)), replace=False)


@task(returns=3)
def get_y(path):
    y = read_csv(os.path.join(path, 'y.dat'), dtype=CategoricalDtype(), header=None, squeeze=True).values
    return y, y.codes, len(y.categories)


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
        return sample, y_s, np.array([], dtype=np.int64), np.array([], dtype=np.int8)
    feature = features_mmap[index][sample]
    mask = feature < value
    left = sample[mask]
    right = sample[~mask]
    y_l = y_s[mask]
    y_r = y_s[~mask]
    return left, y_l, right, y_r


def build_leaf(y_s, tree_path):
    frequencies = Counter(y_s)
    most_common = frequencies.most_common(1)
    if most_common:
        mode = most_common[0][0]
    else:
        mode = None
    return Leaf(tree_path, len(y_s), frequencies, mode)


@task(features_file=FILE_IN, returns=(InternalNode, list, list, list, list))
def compute_split_node(tree_path, sample, n_features, features_file, y_s, n_classes, m_try):
    features_mmap = np.load(features_file, mmap_mode='r', allow_pickle=False)
    return compute_split(tree_path, sample, n_features, features_mmap, y_s, n_classes, m_try)


def compute_split(tree_path, sample, n_features, features_mmap, y_s, n_classes, m_try):
    split_ended = False
    tried_indices = []
    while not split_ended:
        untried_indices = np.setdiff1d(np.arange(n_features), tried_indices)  # type: np.ndarray
        index_selection = feature_selection(untried_indices, m_try)
        b_score = float_info.max
        b_index = None
        b_value = None
        for index in index_selection:
            feature = features_mmap[index]
            score, value = test_split(sample, y_s, feature, n_classes)
            if score < b_score:
                b_score, b_value, b_index = score, value, index
        left_group, y_l, right_group, y_r = get_groups(sample, y_s, features_mmap, b_index, b_value)
        if left_group.size and right_group.size:
            split_ended = True
            node = InternalNode(tree_path, b_index, b_value)
        else:
            tried_indices.extend(list(index_selection))
            if len(tried_indices) == n_features:
                split_ended = True
                node = build_leaf(y_s, tree_path)

    return node, left_group, y_l, right_group, y_r


def flush_remove_nodes(file_out, nodes_to_persist):
    flush_nodes(file_out, *nodes_to_persist)
    for obj in nodes_to_persist:
        compss_delete_object(obj)
    del nodes_to_persist[:]


@task(file_out=FILE_INOUT)
def flush_nodes(file_out, *nodes_to_persist):
    with open(file_out, "a") as tree_file:
        for item in nodes_to_persist:
            if isinstance(item, (Leaf, InternalNode)):
                tree_file.write(item.to_json())
            else:
                for nested_item in item:
                    if isinstance(nested_item, TreeWrapper):
                        nested_item.write_to(tree_file)
                    else:
                        tree_file.write(nested_item.to_json())


@task(samples_file=FILE_IN, features_file=FILE_IN, returns=list)
def build_subtree(sample, y_s, n_features, tree_path, max_depth, n_classes, features_file, m_try, samples_file,
                  use_sklearn_internally=True, sklearn_max_elements=100000000):
    np.random.seed()
    if not sample.size:
        return []
    features_mmap = np.load(features_file, mmap_mode='r', allow_pickle=False)
    tree_traversal = [(tree_path, sample, y_s, 0)]
    node_list_to_persist = []
    while tree_traversal:
        tree_path, sample, y_s, depth = tree_traversal.pop()
        if depth < max_depth:
            if use_sklearn_internally and n_features * len(sample) <= sklearn_max_elements:
                dt = SklearnDTClassifier(max_features=m_try, max_depth=None if max_depth == np.inf else max_depth - depth)
                sample, new_indices, sample_weight = np.unique(sample, return_index=True, return_counts=True)
                x = get_sample_attributes(samples_file, sample)
                y_s = y_s[new_indices]
                dt.fit(x, y_s, sample_weight=sample_weight, check_input=False)
                node_list_to_persist.append(TreeWrapper(tree_path, dt))
            else:
                node, left_group, y_l, right_group, y_r = compute_split(tree_path, sample, n_features, features_mmap,
                                                                        y_s, n_classes, m_try)
                node_list_to_persist.append(node)
                if isinstance(node, InternalNode):
                    tree_traversal.append((tree_path + 'R', right_group, y_r, depth + 1))
                    tree_traversal.append((tree_path + 'L', left_group, y_l, depth + 1))
        else:
            node_list_to_persist.append(build_leaf(y_s, tree_path))
    return node_list_to_persist


class DecisionTreeClassifier:

    def __init__(self, path_in, n_instances, n_features, path_out, name_out, max_depth=None, distr_depth=None,
                 bootstrap=False, try_features=None):
        """
        Decision tree with distributed splits using pyCOMPSs.

        :param path_in: Path of the dataset directory.
        :param n_instances: Number of instances in the sample.
        :param n_features: Number of attributes in the sample.
        :param path_out: Path of the output directory.
        :param name_out: Name of the output file.
        :param max_depth: Depth of the decision tree.
        :param distr_depth: Nodes are split in a distributed way up to this depth.
        :param bootstrap: Randomly select n_instances samples with repetition (used in random forests).
        :param try_features: Number of features to try (at least) for splitting each node.
        """
        self.path_in = path_in
        self.n_instances = n_instances
        self.n_features = n_features
        self.path_out = path_out
        self.name_out = name_out
        self.max_depth = max_depth if max_depth is not None else np.inf
        self.distr_depth = distr_depth if distr_depth is not None else (frexp(self.n_instances)[1] - 1) // 3
        self.y = None
        self.y_codes = None
        self.n_classes = None
        self.bootstrap = bootstrap
        if try_features is None:
            self.m_try = n_features
        elif try_features == 'sqrt':
            self.m_try = max(1, int(sqrt(n_features)))
        elif try_features == 'third':
            self.m_try = max(1, int(n_features / 3))
        else:
            self.m_try = int(try_features)

    def fit(self):
        """
        Fits the DecisionTreeClassifier.
        """
        if self.y_codes is None:
            self.y, self.y_codes, self.n_classes = get_y(self.path_in)
        tree_sample, y_s = sample_selection(self.n_instances, self.y_codes, self.bootstrap)
        features_file = get_features_file(self.path_in)
        samples_file = get_samples_file(self.path_in)
        tree_traversal = [('/', tree_sample, y_s, 0)]
        file_out = os.path.join(self.path_out, self.name_out)
        open(file_out, 'w').close()  # Create new empty file deleting previous content
        nodes_to_persist = []

        while tree_traversal:
            tree_path, sample, y_s, depth = tree_traversal.pop()
            if depth < self.max_depth:
                if depth < self.distr_depth:
                    node, left_group, y_l, right_group, y_r = compute_split_node(tree_path, sample,
                                                                                 self.n_features, features_file, y_s,
                                                                                 self.n_classes, self.m_try)
                    compss_delete_object(sample)
                    compss_delete_object(y_s)
                    nodes_to_persist.append(node)
                    tree_traversal.append((tree_path + 'R', right_group, y_r, depth + 1))
                    tree_traversal.append((tree_path + 'L', left_group, y_l, depth + 1))
                else:
                    subtree_nodes = build_subtree(sample, y_s, self.n_features, tree_path, self.max_depth - depth,
                                                  self.n_classes, features_file, self.m_try, samples_file)
                    nodes_to_persist.append(subtree_nodes)
                    compss_delete_object(sample)
                    compss_delete_object(y_s)

            else:
                nodes_to_persist.append(build_leaf(y_s, tree_path))

            if len(nodes_to_persist) >= 1000:
                flush_remove_nodes(file_out, nodes_to_persist)
        
        flush_remove_nodes(file_out, nodes_to_persist)

    def predict(self, x_test):
        """ Predicts class codes for the input data using a fitted tree and returns an integer or an array. """
        file_tree = os.path.join(self.path_out, self.name_out)
        return prediction.predict(file_tree, x_test)

    def predict_probabilities(self, x_test):
        """ Predicts class probabilities by class code using a fitted tree and returns a 1D or 2D array. """
        file_tree = os.path.join(self.path_out, self.name_out)
        return prediction.predict_probabilities(file_tree, x_test, self.n_classes)
