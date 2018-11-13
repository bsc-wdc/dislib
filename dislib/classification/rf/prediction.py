import os
import json

import numpy as np
from pycompss.api.api import compss_open


def predict(file_name, test_data):
    with compss_open(file_name, 'r') as tree_file:
        if len(test_data.shape) == 1:
            leaf = get_leaf(tree_file, test_data)
            return leaf['mode']
        elif len(test_data.shape) == 2:
            n_samples = test_data.shape[0]
            res = np.zeros(n_samples, np.int64)
            for i, test_instance in enumerate(test_data):
                leaf = get_leaf(tree_file, test_instance)
                res[i] = leaf['mode']
            return res
        else:
            raise ValueError


def predict_probabilities(file_name, test_data, n_classes):
    with compss_open(file_name, 'r') as tree_file:
        if len(test_data.shape) == 1:
            leaf = get_leaf(tree_file, test_data)
            res = np.zeros((n_classes,))
            for cla, freq in leaf['frequencies'].items():
                res[int(cla)] = freq
            return res/leaf['size']
        elif len(test_data.shape) == 2:
            n_samples = test_data.shape[0]
            res = np.zeros((n_samples, n_classes))
            for i, test_instance in enumerate(test_data):
                leaf = get_leaf(tree_file, test_instance)
                for cla, freq in leaf['frequencies'].items():
                    res[i, int(cla)] = freq
                res[i] /= leaf['size']
            return res
        else:
            raise ValueError


def get_leaf(tree_file, instance):
    tree_path = '//'
    node, pos = get_node(tree_file, tree_path, 0)
    while True:
        node_type = node['type']
        if node_type == 'LEAF':
            return node
        elif node_type == 'NODE':
            if instance[node['index']] > node['value']:
                tree_path += 'R'
            else:
                tree_path += 'L'
            node, pos = get_node(tree_file, tree_path, pos)
        else:
            raise Exception('Invalid node')


def get_node(tree_file, tree_path, start):
    f_p = find(tree_file, tree_path, start=start)
    if f_p == -1:
        raise Exception('Node ' + tree_path + ' not found at ' + tree_file.name + ' starting at ' + str(start))
    tree_file.seek(f_p-15)
    line = tree_file.readline()
    node = json.loads(line)
    return node, tree_file.tell()


def find(f, s, start=0):
    file_size = os.path.getsize(f.name)
    bsize = 4096
    f.seek(start)
    overlap = len(s) - 1
    while True:
        if overlap <= f.tell() < file_size:
            f.seek(f.tell() - overlap)
        buffer_in = f.read(bsize)
        if buffer_in:
            pos = buffer_in.find(s)
            if pos >= 0:
                return f.tell() - (len(buffer_in) - pos)
        else:
            return -1
